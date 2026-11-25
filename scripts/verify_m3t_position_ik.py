#!/usr/bin/env python3
"""Replay native M3T estimates and run a simulated position-feedback loop."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np


ARMS = ("so101_left", "so101_right")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TRACKED_JOINTS = (*JOINTS, "gripper")
CAMERA_FRAME = "orbbec335/color_optical"
TOOL_LINK = "gripper_link"
MAX_PRIOR_AGE_NS = 100_000_000
MAX_FK_MATRIX_DIFFERENCE = 2e-4  # Match the gripper pose output check.
TARGET_OFFSET_M = (0.005, 0.0, 0.0)
TARGET_TOLERANCE_MM = 0.5
MAX_JOINT_STEP_DEG = 0.5
FOLLOW_FRACTION = 0.65
MAX_SIMULATION_STEPS = 40
MAX_COMMAND_BACKTRACKS = 32


@dataclass(frozen=True)
class Sample:
    frame_sequence: int
    capture_ns: int
    prior_sequence: int
    prior_age_ns: int
    camera_from_base: np.ndarray
    camera_from_gripper: np.ndarray
    visual_deg: np.ndarray
    encoder_deg: np.ndarray


def finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def native_pose(value) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 16 or not all(map(finite_number, value)):
        raise ValueError("expected a finite row-major 4x4 link pose")
    pose = np.array(value, dtype=float).reshape(4, 4)
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or not math.isclose(np.linalg.det(rotation), 1.0, abs_tol=1e-5)):
        raise ValueError("expected a rigid link pose")
    return pose


def validate_mount_refinement(event: dict) -> None:
    """Accept only a complete six-axis mount prior with unlocked encoder references."""
    if event.get("joints_locked_to_feedback", False) is not False:
        raise ValueError("encoder-reference estimates must leave joints unlocked")
    if "mount_refinement" not in event:
        return
    report = event["mount_refinement"]
    if (not isinstance(report, dict) or report.get("enabled") is not True
            or report.get("joints_locked_to_feedback") is not False
            or not isinstance(report.get("arms"), list) or len(report["arms"]) != len(ARMS)):
        raise ValueError("expected complete unlocked mount-refinement diagnostics")
    links = event.get("links")
    if not isinstance(links, list):
        raise ValueError("mount refinement requires the actual base pose for each arm")
    bases = {}
    for link in links:
        if not isinstance(link, dict):
            raise ValueError("invalid native link in mount-refinement estimate")
        if link.get("link") == "base_link":
            arm = link.get("arm")
            if arm not in ARMS or arm in bases or link.get("frame_id") != f"{arm}/base_link":
                raise ValueError("expected exactly one identified base pose for each arm")
            bases[arm] = native_pose(link.get("pose"))
    if set(bases) != set(ARMS):
        raise ValueError("mount refinement requires the actual base pose for each arm")
    seen = set()
    for entry in report["arms"]:
        if not isinstance(entry, dict) or entry.get("arm") not in ARMS or entry["arm"] in seen:
            raise ValueError("expected exactly one mount-refinement diagnostic for each arm")
        arm = entry["arm"]
        seen.add(arm)
        nominal = native_pose(entry.get("nominal_base_in_camera"))
        refined = native_pose(entry.get("refined_base_in_camera"))
        if not np.allclose(refined, bases[arm], atol=1e-6, rtol=0):
            raise ValueError("refined mount diagnostic disagrees with the actual base pose")
        relative_rotation = refined[:3, :3] @ nominal[:3, :3].T
        sine = .5 * np.linalg.norm([
            relative_rotation[2, 1] - relative_rotation[1, 2],
            relative_rotation[0, 2] - relative_rotation[2, 0],
            relative_rotation[1, 0] - relative_rotation[0, 1]])
        rotation = math.atan2(sine, np.clip((np.trace(relative_rotation) - 1.) / 2., -1., 1.))
        translation = float(np.linalg.norm(refined[:3, 3] - nominal[:3, 3]))
        for key, expected in (("translation_correction_m", translation), ("rotation_correction_rad", rotation)):
            value = entry.get(key)
            if not finite_number(value) or value < 0 or not math.isclose(value, expected, abs_tol=1e-4, rel_tol=0):
                raise ValueError("mount correction magnitude disagrees with its nominal and refined poses")


def native_sample(event: dict, arm: str) -> Sample:
    if (event.get("schema_version") != 1 or event.get("encoder_reference_links") is not True
            or event.get("frame_id") != CAMERA_FRAME):
        raise ValueError("expected a native encoder-reference estimate in the color optical frame")
    validate_mount_refinement(event)
    expected = {(name, joint) for name in ARMS for joint in TRACKED_JOINTS}
    report = event["joint_priors"]
    priors = report["priors"]
    age_ns = report["age_ns"]
    if (report.get("enabled") is not True or report.get("status") != "applied"
            or report.get("time_basis") != "received" or report.get("capture_time_associated") is not False
            or report.get("rejected_mappings") != [] or not isinstance(priors, list)
            or report.get("prior_count") != len(expected) or len(priors) != len(expected)
            or type(age_ns) is not int or not 0 <= age_ns <= MAX_PRIOR_AGE_NS):
        raise ValueError("expected one applied receive-time report of all twelve priors at age ≤100 ms")
    encoder = {(entry["arm"], entry["joint"]): entry["angle_rad"] for entry in priors}
    if set(encoder) != expected or not all(map(finite_number, encoder.values())):
        raise ValueError("applied priors must contain each of the twelve finite arm/joint angles")

    visual = event["joint_angles_rad"]
    if set(visual) != set(ARMS) or any(
            set(angles) != set(TRACKED_JOINTS) or not all(map(finite_number, angles.values()))
            for angles in visual.values()):
        raise ValueError("expected complete finite native visual joint angles for both arms")

    poses = {}
    for link in event["links"]:
        name = link["link"]
        if link["arm"] == arm and name in {"base_link", TOOL_LINK}:
            if name in poses or link.get("frame_id") != f"{arm}/{name}":
                raise ValueError(f"unexpected frame ID or duplicate pose for {arm}/{name}")
            poses[name] = native_pose(link["pose"])
    if set(poses) != {"base_link", TOOL_LINK}:
        raise ValueError(f"expected base_link and {TOOL_LINK} poses for {arm}")

    frame_sequence = event["frame_sequence"]
    capture_ns = event["color_capture_ns"]
    prior_sequence = report["sequence"]
    if (type(frame_sequence) is not int or frame_sequence < 0
            or type(capture_ns) is not int or capture_ns <= 0
            or type(prior_sequence) is not int or prior_sequence <= 0):
        raise ValueError("invalid native frame sequence, color capture timestamp, or prior sequence")
    return Sample(frame_sequence, capture_ns, prior_sequence, age_ns, poses["base_link"], poses[TOOL_LINK],
                  np.rad2deg([visual[arm][joint] for joint in JOINTS]),
                  np.rad2deg([encoder[arm, joint] for joint in JOINTS]))


def load_recording(path: Path, arm: str) -> list[Sample]:
    samples = []
    previous_clock = previous_driver = None
    with path.open(encoding="utf-8") as recording:
        for line_number, line in enumerate(recording, start=1):
            if not line.lstrip().startswith("{"):
                continue  # Native stdout may also contain startup messages.
            try:
                event = json.loads(line)
                if event.get("event") != "estimate":
                    continue
                sample = native_sample(event, arm)
                clock = (event["capture_clock_kind"], event["capture_clock_source"], event["capture_clock_generation"])
                kind, source, generation = clock
                if (kind not in {"HostMonotonic", "host_monotonic"} or not isinstance(source, str)
                        or not any(bytes.fromhex(source)) or type(generation) is not int or generation < 0):
                    raise ValueError("expected an identified host monotonic capture clock")
                prior = event["joint_priors"]
                driver = (prior["controller_id"], prior["driver_session"])
                if not all(isinstance(value, str) and value for value in driver):
                    raise ValueError("expected a named prior controller and driver session")
                if samples and (clock != previous_clock or driver != previous_driver
                                or sample.capture_ns <= samples[-1].capture_ns
                                or sample.frame_sequence <= samples[-1].frame_sequence
                                or sample.prior_sequence < samples[-1].prior_sequence):
                    raise ValueError("capture/feedback source changed or its frame sequence or time regressed")
                samples.append(sample)
                previous_clock, previous_driver = clock, driver
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"recording line {line_number}: {error}") from error
    if len(samples) < 2:
        raise ValueError("expected at least two native estimates to check capture progression")
    return samples


def distance_mm(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second) * 1000)


def position_step(kinematics, camera_from_base: np.ndarray, goal_camera: np.ndarray,
                  visual_deg: np.ndarray, encoder_deg: np.ndarray,
                  recorded_gripper: np.ndarray | None = None,
                  command_reference_deg: np.ndarray | None = None) -> tuple[dict, np.ndarray | None]:
    """Evaluate one position step with the same FK and IK in replay and simulation."""
    base_from_gripper = kinematics.forward(visual_deg)
    current_camera = camera_from_base @ base_from_gripper
    current_error = distance_mm(current_camera[:3, 3], goal_camera)
    row = {
        "fk_camera_xyz_m": current_camera[:3, 3].tolist(),
        "position_error_mm": current_error,
        "visual_deg": visual_deg.tolist(),
        "encoder_deg": encoder_deg.tolist(),
    }
    if recorded_gripper is not None:
        fk_difference = float(np.max(np.abs(current_camera - recorded_gripper)))
        row.update(recorded_camera_xyz_m=recorded_gripper[:3, 3].tolist(),
                   recorded_position_error_mm=distance_mm(recorded_gripper[:3, 3], goal_camera),
                   fk_recorded_position_difference_mm=distance_mm(current_camera[:3, 3], recorded_gripper[:3, 3]),
                   fk_recorded_max_matrix_difference=fk_difference)
    if not np.all(np.isfinite(current_camera)):
        raise ValueError("kinematics returned non-finite FK")
    command_reference_camera = None
    command_reference_error = None
    if command_reference_deg is not None:
        command_reference_deg = np.asarray(command_reference_deg, dtype=float)
        if (command_reference_deg.shape != visual_deg.shape
                or not np.all(np.isfinite(command_reference_deg))):
            raise ValueError("command reference must be a finite joint vector matching the visual estimate")
        command_reference_camera = camera_from_base @ kinematics.forward(command_reference_deg)
        if not np.all(np.isfinite(command_reference_camera)):
            raise ValueError("kinematics returned non-finite FK for the command reference")
        command_reference_error = distance_mm(command_reference_camera[:3, 3], goal_camera)
        row["command_reference_error_mm"] = command_reference_error
        if command_reference_error <= TARGET_TOLERANCE_MM:
            return row, None
    elif current_error <= TARGET_TOLERANCE_MM:
        return row, None

    base_goal = base_from_gripper.copy()
    # The rigid transform maps base coordinates to camera coordinates.
    base_goal[:3, 3] = camera_from_base[:3, :3].T @ (goal_camera - camera_from_base[:3, 3])
    ik_deg = np.asarray(kinematics.inverse(visual_deg, base_goal, orientation_weight=0.0,
                                          max_joint_step_deg=MAX_JOINT_STEP_DEG), dtype=float)
    if ik_deg.shape != visual_deg.shape or not np.all(np.isfinite(ik_deg)):
        raise ValueError("kinematics returned an invalid joint vector")
    # Both IK and the Gazebo sidecar use canonical URDF coordinates. The
    # command reference is the last settled absolute target, independent of
    # the potentially biased encoder prior. Stateless replay has no such
    # reference, so it keeps the absolute IK target without backtracking.
    visual_step = ik_deg - visual_deg
    if command_reference_deg is None:
        proposed_command_deg = ik_deg
        step_reference_deg = visual_deg
        command_step_scale = 1.0
        command_backtracking_steps = 0
        command_backtracking_enabled = False
        command_step = proposed_command_deg - step_reference_deg
        within_model_limits = bool(np.all(
            (kinematics.solver.lower_deg - 1e-6 <= proposed_command_deg)
            & (proposed_command_deg <= kinematics.solver.upper_deg + 1e-6)))
        predicted_camera = camera_from_base @ kinematics.forward(proposed_command_deg)
        if not np.all(np.isfinite(predicted_camera)):
            raise ValueError("kinematics returned non-finite FK for the IK result")
        predicted_error = distance_mm(predicted_camera[:3, 3], goal_camera)
        reference_camera = current_camera
        reference_error = current_error
        reduces_reference_error = predicted_error < reference_error
    else:
        step_reference_deg = command_reference_deg
        command_backtracking_enabled = True
        reference_camera = command_reference_camera
        reference_error = command_reference_error
        proposed_command_deg = None
        command_step = np.zeros_like(step_reference_deg)
        command_step_scale = 0.0
        command_backtracking_steps = 0
        within_model_limits = None
        predicted_camera = None
        predicted_error = None
        reduces_reference_error = False
        for reductions in range(MAX_COMMAND_BACKTRACKS + 1):
            scale = 0.5 ** reductions
            candidate = step_reference_deg + scale * visual_step
            candidate_step = candidate - step_reference_deg
            if np.array_equal(candidate, step_reference_deg):
                break
            candidate_in_limits = bool(np.all(
                (kinematics.solver.lower_deg - 1e-6 <= candidate)
                & (candidate <= kinematics.solver.upper_deg + 1e-6)))
            command_backtracking_steps = reductions
            if candidate_in_limits:
                candidate_camera = camera_from_base @ kinematics.forward(candidate)
                if not np.all(np.isfinite(candidate_camera)):
                    raise ValueError("kinematics returned non-finite FK for a command candidate")
                candidate_error = distance_mm(candidate_camera[:3, 3], goal_camera)
                if candidate_error < reference_error:
                    proposed_command_deg = candidate
                    command_step = candidate_step
                    command_step_scale = scale
                    within_model_limits = True
                    predicted_camera = candidate_camera
                    predicted_error = candidate_error
                    reduces_reference_error = True
                    break
            if reductions == MAX_COMMAND_BACKTRACKS:
                command_backtracking_steps = reductions + 1

    max_step = float(np.max(np.abs(command_step)))
    row.update(ik_status=kinematics.last_ik.get("status"), ik_deg=ik_deg.tolist(),
               proposed_command_deg=(proposed_command_deg.tolist() if proposed_command_deg is not None else None),
               max_joint_step_deg=max_step,
               command_reference_deg=step_reference_deg.tolist(), command_step_deg=command_step.tolist(),
               command_target_encoder_delta_deg=(
                   (proposed_command_deg - encoder_deg).tolist() if proposed_command_deg is not None else None),
               command_step_scale=command_step_scale,
               command_backtracking_steps=command_backtracking_steps,
               command_backtracking_enabled=command_backtracking_enabled,
               command_target_in_model_limits=within_model_limits,
               command_reference_error_mm=reference_error,
               command_target_reduces_reference_error=reduces_reference_error,
               predicted_error_mm=predicted_error)
    return row, proposed_command_deg


def step_violations(row: dict) -> list[str]:
    violations = []
    if row.get("fk_recorded_max_matrix_difference", 0) > MAX_FK_MATRIX_DIFFERENCE:
        violations.append("FK at the visual estimate disagrees with the recorded gripper_link transform")
    if row.get("max_joint_step_deg", 0) > MAX_JOINT_STEP_DEG + 1e-9:
        violations.append("absolute command target exceeds the bounded visual IK step")
    if row.get("command_target_in_model_limits") is False:
        violations.append("absolute command target lies outside the URDF joint limits")
    if row.get("command_target_reduces_reference_error") is False:
        if row.get("command_backtracking_enabled"):
            violations.append("no URDF-limited positive IK step scale reduces error from the command reference")
        else:
            violations.append("the stateless IK command target does not reduce predicted position error")
    progress_reference_error = row.get("command_reference_error_mm", row.get("position_error_mm"))
    if (row.get("command_target_reduces_reference_error") is not False
            and row.get("predicted_error_mm") is not None and progress_reference_error is not None
            and row["predicted_error_mm"] >= progress_reference_error):
        violations.append("the command target does not reduce predicted error from its command reference")
    return violations


def replay(kinematics, samples: list[Sample], goal_camera: np.ndarray) -> dict:
    trace = []
    violations = []
    for sample in samples:
        frame = {"frame_sequence": sample.frame_sequence, "color_capture_ns": sample.capture_ns,
                 "prior_sequence": sample.prior_sequence, "prior_age_ns": sample.prior_age_ns}
        try:
            row, _ = position_step(kinematics, sample.camera_from_base, goal_camera,
                                   sample.visual_deg, sample.encoder_deg, sample.camera_from_gripper)
            frame.update(row)
            failures = step_violations(row)
        except (ValueError, RuntimeError) as error:
            failures = [str(error)]
        trace.append(frame)
        violations.extend({"frame_sequence": sample.frame_sequence, "reason": reason} for reason in failures)

    return {
        "source": "historical board capture with receive-time encoder priors",
        "frames": len(samples),
        "capture_duration_s": (samples[-1].capture_ns - samples[0].capture_ns) / 1e9,
        "max_prior_age_ms": max(sample.prior_age_ns for sample in samples) / 1e6,
        "max_fk_recorded_position_difference_mm": max(
            (row["fk_recorded_position_difference_mm"] for row in trace
             if "fk_recorded_position_difference_mm" in row), default=None),
        "max_proposed_joint_step_deg": max((row.get("max_joint_step_deg", 0) for row in trace), default=0),
        "proposed_steps": sum("proposed_command_deg" in row for row in trace),
        "violations": violations,
        "trace": trace,
    }


def simulate(kinematics, first: Sample, goal_camera: np.ndarray) -> dict:
    # Start the simplified plant and command reference at the visual seed;
    # preserve the first-frame encoder-minus-visual offset only in the
    # simulated encoder observation. Each fractional plant update is treated
    # as settled before the next visual feedback step.
    joint_deg = first.visual_deg.copy()
    encoder_bias = first.encoder_deg - joint_deg
    visual_bias = np.zeros_like(joint_deg)
    command_reference_deg = joint_deg.copy()
    trace = []
    violations = []
    for iteration in range(MAX_SIMULATION_STEPS + 1):
        visual_deg = joint_deg + visual_bias
        encoder_deg = joint_deg + encoder_bias
        row, proposed = position_step(
            kinematics, first.camera_from_base, goal_camera, visual_deg, encoder_deg,
            command_reference_deg=command_reference_deg)
        trace.append({"iteration": iteration, **row})
        violations.extend(step_violations(row))
        if violations or proposed is None or iteration == MAX_SIMULATION_STEPS:
            break
        joint_deg += FOLLOW_FRACTION * (proposed - joint_deg)
        command_reference_deg = joint_deg.copy()

    final_error = trace[-1]["position_error_mm"]
    reached = final_error <= TARGET_TOLERANCE_MM
    if not reached:
        violations.append("simulated feedback did not reach the position tolerance")
    return {
        "source": "simulated fractional joint following with a visual-seeded canonical command reference, noiseless visual feedback, and the first-frame encoder offset",
        "follow_fraction": FOLLOW_FRACTION,
        "encoder_minus_visual_bias_deg": encoder_bias.tolist(),
        "visual_minus_joint_bias_deg": visual_bias.tolist(),
        "feedback_updates": len(trace) - 1,
        "initial_error_mm": trace[0]["position_error_mm"],
        "final_error_mm": final_error,
        "reached_tolerance": reached,
        "violations": violations,
        "trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True, help="native M3T estimate JSONL or stdout log")
    parser.add_argument("--urdf", type=Path, required=True, help="kinematic URDF matching the M3T model")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--arm", choices=ARMS, default=ARMS[0])
    args = parser.parse_args()
    if args.output.resolve() in {args.recording.resolve(), args.urdf.resolve()}:
        parser.error("--output must differ from --recording and --urdf")

    result = {
        "arm": args.arm,
        "camera_frame": CAMERA_FRAME,
        "target_link": TOOL_LINK,
        "joint_names": JOINTS,
        "recording": str(args.recording.resolve()),
        "urdf": str(args.urdf.resolve()),
        "target_offset_camera_m": TARGET_OFFSET_M,
        "position_tolerance_mm": TARGET_TOLERANCE_MM,
        "max_joint_step_deg": MAX_JOINT_STEP_DEG,
        "orientation_weight": 0.0,
        "hardware_commands_sent": 0,
        "passed": False,
    }
    try:
        from so101_control.kinematics import ArmKinematics

        samples = load_recording(args.recording, args.arm)
        kinematics = ArmKinematics(str(args.urdf), target_frame=TOOL_LINK, joint_names=JOINTS)
        goal_camera = samples[0].camera_from_gripper[:3, 3] + TARGET_OFFSET_M
        result["goal_camera_xyz_m"] = goal_camera.tolist()
        result["recorded_replay"] = replay(kinematics, samples, goal_camera)
        result["simulated_feedback"] = simulate(kinematics, samples[0], goal_camera)
        result["passed"] = not (result["recorded_replay"]["violations"] or result["simulated_feedback"]["violations"])
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        result["error"] = str(error)
        print(f"Position IK smoke failed: {error}", file=sys.stderr)

    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Position IK smoke {'passed' if result['passed'] else 'failed'}; report: {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
