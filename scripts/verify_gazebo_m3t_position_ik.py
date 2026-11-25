#!/usr/bin/env python3
"""Test a position-only M3T servo loop on an isolated Gazebo SO101 arm."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

import numpy as np

from prepare_gazebo_m3t_scene import INITIAL_JOINTS, prepare
from verify_m3t_position_ik import (
    JOINTS,
    MAX_JOINT_STEP_DEG,
    TARGET_OFFSET_M,
    TOOL_LINK,
    distance_mm,
    native_pose,
    position_step,
    step_violations,
)
from verify_software import runtime_environment, start, stop_children, wait_until


FRAME_COUNT = 5
COMMAND_REFERENCE_STOP_THRESHOLD_MM = 1.0  # Internal stop only; not an accuracy criterion.
SETTLE_NS = 150_000_000
MAX_PRIOR_AGE_NS = 40_000_000
MAX_TRUTH_BRACKET_NS = 80_000_000
MAX_COMMANDS = 24


class Journal:
    def __init__(self, path):
        self.channel = path.open("r", encoding="utf-8")
        self.records = defaultdict(list)

    def update(self):
        position = self.channel.tell()
        for line in iter(self.channel.readline, ""):
            if not line.endswith("\n"):
                self.channel.seek(position)
                break
            position = self.channel.tell()
            if line.startswith("{"):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.records[record["event"]].append(record)


def validate_biases(robot_model: Path, bias_deg: list[float]) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    if len(bias_deg) != len(JOINTS) or not all(math.isfinite(value) for value in bias_deg):
        raise ValueError("--bias-deg must contain five finite values in SO101 joint order")
    model = json.loads(robot_model.read_text(encoding="utf-8"))
    records = {joint["name"]: joint for joint in model["joints"]}
    initial_rad = np.asarray([INITIAL_JOINTS[name] for name in JOINTS], dtype=float)
    bias_rad = np.deg2rad(np.asarray(bias_deg, dtype=float))
    limits = {}
    for name, seed, bias in zip(JOINTS, initial_rad, bias_rad):
        joint = records.get(name)
        if joint is None or joint.get("type") != "revolute":
            raise ValueError(f"robot model is missing canonical revolute joint {name}")
        lower, upper = float(joint["lower"]), float(joint["upper"])
        biased = float(seed + bias)
        if not lower <= biased <= upper:
            raise ValueError(f"prior bias moves initial {name} angle outside URDF limits")
        limits[name] = {"lower_rad": lower, "upper_rad": upper,
                        "seed_rad": float(seed), "biased_prior_rad": biased}
    return bias_rad, limits


def configure_tracker(scene, metadata, service, bias_rad: np.ndarray | None = None):
    path = scene / "tracking.json"
    config = json.loads(path.read_text())
    jaw_min = metadata["gripper_lower_rad"]
    jaw_scale = (metadata["gripper_upper_rad"] - jaw_min) / 100
    bias_rad = np.zeros(len(JOINTS), dtype=float) if bias_rad is None else np.asarray(bias_rad, dtype=float)
    if bias_rad.shape != (len(JOINTS),) or not np.all(np.isfinite(bias_rad)):
        raise ValueError("joint-prior bias must be a finite five-joint vector")
    bias_by_joint = dict(zip(JOINTS, bias_rad))
    config["encoder_reference_links"] = True
    config["joint_priors"] = {
        "service_name": service,
        "controller_id": "gazebo-so101",
        "required": True,
        "initialize_from_measurement": True,
        "time_basis": "capture",
        "maximum_age_ms": MAX_PRIOR_AGE_NS // 1_000_000,
        "mappings": [
            {"arm": metadata["arm"], "joint": name, "source_arm": "left", "source_joint": name,
             "scale_rad_per_unit": jaw_scale if name == "gripper" else math.pi / 180,
             "offset_rad": jaw_min if name == "gripper" else float(bias_by_joint[name]),
             "standard_deviation_rad": 0.01 if name == "gripper" else 0.1}
            for name in (*JOINTS, "gripper")
        ],
    }
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def optical_xyz(pose, optical_from_base):
    base_xyz = np.asarray(pose["base_xyz_m"], dtype=float)
    gripper_xyz = np.asarray(pose["gripper_xyz_m"], dtype=float)
    q = np.asarray(pose["base_quat_xyzw"], dtype=float)
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-5):
        raise ValueError("Gazebo reported a non-unit base quaternion")
    vector, scalar = q[:3], q[3]
    relative = gripper_xyz - base_xyz
    # Apply the inverse quaternion to the model-frame translation difference.
    cross = np.cross(-vector, relative)
    base_point = relative + 2 * (scalar * cross + np.cross(-vector, cross))
    return optical_from_base[:3, :3] @ base_point + optical_from_base[:3, 3]


def interpolate_truth(poses, stamp, optical_from_base):
    offset = bisect_left(poses, stamp, key=lambda pose: pose["stamp_ns"])
    if offset == 0 or offset == len(poses):
        return None
    before, after = poses[offset - 1:offset + 1]
    span = after["stamp_ns"] - before["stamp_ns"]
    if not 0 < span <= MAX_TRUTH_BRACKET_NS:
        return None
    factor = (stamp - before["stamp_ns"]) / span
    first = optical_xyz(before, optical_from_base)
    last = optical_xyz(after, optical_from_base)
    return first + factor * (last - first), span


def settled_after(journal, target_rad, cutoff_ns, processes):
    settled_at = None
    index = 0
    stable_since = None
    previous_stamp = None

    def ready():
        nonlocal settled_at, index, stable_since, previous_stamp
        journal.update()
        states = journal.records["state"]
        for state in states[index:]:
            index += 1
            stamp = state["stamp_ns"]
            if stamp <= cutoff_ns:
                continue
            angle_error = np.max(np.abs(np.asarray(state["q_rad"]) - target_rad))
            speed = np.max(np.abs(state["dq_rad_s"]))
            if angle_error <= math.radians(0.10) and speed <= math.radians(0.5):
                if stable_since is None or stamp - previous_stamp > 50_000_000:
                    stable_since = stamp
                if stamp - stable_since >= SETTLE_NS:
                    settled_at = stamp
                    return True
            else:
                stable_since = None
            previous_stamp = stamp
        return False

    wait_until("waiting for Gazebo joint positions and velocities to settle", time.monotonic() + 12,
               ready, processes)
    return settled_at


def sample_estimate(event, metadata):
    arm = metadata["arm"]
    stamp = event["color_capture_ns"]
    if (event["schema_version"] != 1 or event["frame_id"] != metadata["camera_frame"]
            or event["capture_clock_kind"] != "Device" or event["capture_ns"] != stamp
            or event["pair_skew_ns"] != 0 or event.get("encoder_reference_links") is not True):
        raise ValueError("M3T output is not an aligned Gazebo encoder-reference estimate")
    report = event["joint_priors"]
    if (report["status"] != "applied" or report["time_basis"] != "capture"
            or not report["capture_time_associated"] or report["controller_id"] != "gazebo-so101"
            or report["prior_count"] != len(JOINTS) + 1 or report["rejected_mappings"]
            or not 0 <= report["age_ns"] <= MAX_PRIOR_AGE_NS):
        raise ValueError("M3T did not use complete, fresh, capture-aligned Gazebo encoder priors")
    encoder = {p["joint"]: p["angle_rad"] for p in report["priors"] if p["arm"] == arm}
    visual = event["joint_angles_rad"][arm]
    poses = {link["link"]: native_pose(link["pose"]) for link in event["links"]
             if link["arm"] == arm and link["link"] in ("base_link", TOOL_LINK)}
    return (np.rad2deg([visual[name] for name in JOINTS]), np.rad2deg([encoder[name] for name in JOINTS]),
            poses["base_link"], poses[TOOL_LINK])


def capture_window(tracker, gazebo, after_ns, metadata, optical_from_base, processes, timeout_s=14):
    samples = []
    index = 0
    latest_stamp = after_ns
    clock_identity = None

    def ready():
        nonlocal index, latest_stamp, clock_identity
        tracker.update()
        gazebo.update()
        estimates = tracker.records["estimate"]
        while index < len(estimates):
            event = estimates[index]
            stamp = event["color_capture_ns"]
            if stamp <= latest_stamp:
                index += 1
                continue
            truth = interpolate_truth(gazebo.records["pose"], stamp, optical_from_base)
            if truth is None:
                if not gazebo.records["pose"] or gazebo.records["pose"][-1]["stamp_ns"] <= stamp:
                    return False  # Wait for the following pose, never extrapolate.
                index += 1
                samples.clear()
                continue
            identity = (event["capture_clock_source"], event["capture_clock_generation"])
            if clock_identity is not None and identity != clock_identity:
                raise ValueError("Gazebo camera clock changed while sampling M3T")
            clock_identity = identity
            samples.append((event, sample_estimate(event, metadata), truth))
            index += 1
            latest_stamp = stamp
            if len(samples) == FRAME_COUNT:
                return True
        return False

    wait_until("collecting five post-settle M3T frames with independent Gazebo poses", time.monotonic() + timeout_s,
               ready, processes)
    return samples


def run_control(kinematics, commands, tracker, gazebo, metadata, processes, result):
    optical_from_base = native_pose(metadata["optical_from_base"])
    initial_rad = np.array([INITIAL_JOINTS[name] for name in JOINTS])
    settled_ns = settled_after(gazebo, initial_rad, 0, processes)
    initial = capture_window(tracker, gazebo, settled_ns, metadata, optical_from_base, processes, timeout_s=65)
    start_xyz = np.median([sample[2][0] for sample in initial], axis=0)
    seed_fk = optical_from_base @ kinematics.forward(np.rad2deg(initial_rad))
    if distance_mm(start_xyz, seed_fk[:3, 3]) > 0.2:
        raise ValueError("Gazebo initial pose and the installed IK model disagree on the scene seed")
    goal = start_xyz + TARGET_OFFSET_M
    result.update(initial_truth_camera_xyz_m=start_xyz.tolist(), goal_camera_xyz_m=goal.tolist(), trace=[])
    latest_frame = None
    window = initial
    command_reference_deg = np.rad2deg(initial_rad)

    for command_id in range(MAX_COMMANDS + 1):
        checks = []
        for event, (visual, encoder, base, gripper), (truth, bracket_ns) in window:
            if latest_frame is not None and event["frame_sequence"] <= latest_frame:
                raise ValueError("M3T frame sequence regressed during the loop")
            latest_frame = event["frame_sequence"]
            visual_xyz = (base @ kinematics.forward(visual))[:3, 3]
            command_reference_xyz = (base @ kinematics.forward(command_reference_deg))[:3, 3]
            checks.append({"frame_sequence": latest_frame, "capture_ns": event["color_capture_ns"],
                           "prior_age_ns": event["joint_priors"]["age_ns"], "truth_bracket_ns": bracket_ns,
                           "visual_error_mm": distance_mm(visual_xyz, goal),
                           "command_reference_error_mm": distance_mm(command_reference_xyz, goal),
                           "truth_error_mm": distance_mm(truth, goal), "visual_camera_xyz_m": visual_xyz.tolist(),
                           "command_reference_camera_xyz_m": command_reference_xyz.tolist(),
                           "truth_camera_xyz_m": truth.tolist()})
        if all(check["command_reference_error_mm"] <= COMMAND_REFERENCE_STOP_THRESHOLD_MM for check in checks):
            tick = {"index": command_id, "settled_ns": settled_ns, "frames": checks, "ik": None,
                    "stop_reason": "fresh command-reference FK window reached the internal stop threshold"}
            result["trace"].append(tick)
            result["final_truth_displacement_camera_mm"] = (
                (np.asarray(checks[-1]["truth_camera_xyz_m"]) - start_xyz) * 1000).tolist()
            result["final_truth_errors_mm"] = [check["truth_error_mm"] for check in checks]
            result["final_visual_errors_mm"] = [check["visual_error_mm"] for check in checks]
            result["final_command_reference_errors_mm"] = [
                check["command_reference_error_mm"] for check in checks]
            visual_truth_offset = (
                np.asarray(checks[-1]["visual_camera_xyz_m"])
                - np.asarray(checks[-1]["truth_camera_xyz_m"])
            ) * 1000
            result["final_visual_truth_offset_camera_mm"] = visual_truth_offset.tolist()
            result["final_visual_truth_offset_norm_mm"] = float(np.linalg.norm(visual_truth_offset))
            print(f"step {command_id}: command reference {checks[-1]['command_reference_error_mm']:.3f} mm, "
                  f"visual {checks[-1]['visual_error_mm']:.3f} mm, "
                  f"Gazebo truth {checks[-1]['truth_error_mm']:.3f} mm; internal stop reached", flush=True)
            return
        if command_id == MAX_COMMANDS:
            raise RuntimeError("command-reference feedback did not reach the internal stop threshold within the command budget")
        event, (visual, encoder, base, gripper), _ = window[-1]
        step, proposed = position_step(
            kinematics, base, goal, visual, encoder, gripper,
            command_reference_deg=command_reference_deg)
        tick = {"index": command_id, "settled_ns": settled_ns, "frames": checks, "ik": step}
        result["trace"].append(tick)
        failures = step_violations(step)
        if failures:
            raise RuntimeError("; ".join(failures))
        print(f"step {command_id}: command reference {checks[-1]['command_reference_error_mm']:.3f} mm, "
              f"visual {checks[-1]['visual_error_mm']:.3f} mm, "
              f"Gazebo truth {checks[-1]['truth_error_mm']:.3f} mm", flush=True)
        last_capture = event["color_capture_ns"]
        if proposed is not None:
            target = np.deg2rad(proposed)
            commands.write(json.dumps({"command_id": command_id + 1, "q_rad": target.tolist()}) + "\n")
            commands.flush()

            def acknowledged():
                gazebo.update()
                acknowledgements = gazebo.records["command"]
                return acknowledgements and acknowledgements[-1]["command_id"] == command_id + 1

            wait_until("publishing the grouped Gazebo command", time.monotonic() + 4, acknowledged, processes)
            sent_at = gazebo.records["command"][-1]["stamp_ns"]
            tick["command_ns"] = sent_at
            result["commands"] += 1
            settled_ns = settled_after(gazebo, target, max(sent_at, last_capture), processes)
            # Only advance the independent canonical command reference once
            # Gazebo has acknowledged that the requested pose settled.
            command_reference_deg = proposed.copy()
        window = capture_window(tracker, gazebo, max(settled_ns, last_capture), metadata, optical_from_base, processes)


def run(args):
    from so101_control.kinematics import ArmKinematics

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "INCOMPLETE", "commands": 0, "max_joint_step_deg": MAX_JOINT_STEP_DEG,
              "command_reference_stop_threshold_mm": COMMAND_REFERENCE_STOP_THRESHOLD_MM,
              "stop_condition": "all fresh-frame FK(command_reference_deg) errors <= command_reference_stop_threshold_mm; internal measurement stop, not accuracy acceptance",
              "accuracy_acceptance": None,
              "fresh_frames_required": FRAME_COUNT,
              "joint_order": list(JOINTS),
              "requested_prior_bias_deg": dict(zip(JOINTS, args.bias_deg)),
              "joint_limits_and_seed": None,
              "applied_prior_offsets_rad": None}
    children, handles, forced_kills = [], [], []
    scripts = Path(__file__).resolve().parent
    build = args.build_dir.resolve()
    try:
        bias_rad, limits = validate_biases(args.robot_model, args.bias_deg)
        result["joint_limits_and_seed"] = limits
        scene = output / "scene"
        prepare(args.robot_model, scene, position_ik=True)
        metadata_path = scene / "scene_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if tuple(metadata["joint_names"]) != JOINTS:
            raise ValueError("Gazebo joint order differs from the IK configuration")
        scope = uuid.uuid4().hex
        service = f"rbt/so101/gazebo/{scope}/state/v1"
        config = configure_tracker(scene, metadata, service, bias_rad)
        tracker_config = json.loads(config.read_text(encoding="utf-8"))
        result["applied_prior_offsets_rad"] = {
            mapping["joint"]: float(mapping["offset_rad"])
            for mapping in tracker_config["joint_priors"]["mappings"]
            if mapping.get("arm") == metadata["arm"]
        }
        (output / "run_manifest.json").write_text(json.dumps({
            "mode": "dynamic_position_ik",
            "robot_model": str(args.robot_model.resolve()),
            "urdf": str(args.urdf.resolve()),
            "joint_order": list(JOINTS),
            "scene_seed_rad": {name: INITIAL_JOINTS[name] for name in JOINTS},
            "requested_prior_bias_deg": dict(zip(JOINTS, args.bias_deg)),
            "applied_prior_offsets_rad": result["applied_prior_offsets_rad"],
            "target_offset_camera_m": TARGET_OFFSET_M,
            "max_commands": MAX_COMMANDS,
            "command_reference_initialized_from_scene_seed": True,
            "tracking_config": str(config.resolve()),
        }, indent=2) + "\n", encoding="utf-8")
        env = runtime_environment("hardware")
        env.update(GZ_PARTITION=f"gazebo-m3t-{scope}", GZ_HOMEDIR=str(output))
        with socket.socket() as socket_handle:
            socket_handle.bind(("127.0.0.1", 0))
            port = socket_handle.getsockname()[1]
        env["RBT_COMPOSER_ADDR"] = f"127.0.0.1:{port}"

        def composer_ready():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return True
            except OSError:
                return False

        start("composer", [str(args.composer.resolve())], env, output, children, handles)
        wait_until("starting isolated Composer", time.monotonic() + 10, composer_ready, children)
        start("gazebo", ["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "3", str(scene / "world.sdf")],
              env, output, children, handles)
        start("bridge", [str(args.gazebo_python), str(scripts / "bridge_gazebo_rgbd.py"),
                         "--topic", metadata["rgbd_topic"], "--fps", "15",
                         "--provider-binary", str(build / "rbt_m3t_gazebo_rgbd_provider"),
                         "--joint-state-service", service, "--scene-metadata", str(metadata_path)],
              env, output, children, handles)
        bridge = Journal(output / "bridge.log")
        handles.append(bridge.channel)

        def provider_ready():
            bridge.update()
            return bool(bridge.records["gazebo_rgbd_ready"])

        wait_until("starting Gazebo RGB-D and encoder bridge", time.monotonic() + 25, provider_ready, children)
        io_log = (output / "gazebo_io.log").open("w", encoding="utf-8")
        handles.append(io_log)
        io = subprocess.Popen([str(args.gazebo_python), str(scripts / "gazebo_position_ik_io.py"),
                               "--scene-metadata", str(metadata_path)], env=env, stdin=subprocess.PIPE,
                              stdout=io_log, stderr=subprocess.STDOUT, start_new_session=True, text=True)
        children.append(io)
        handles.append(io.stdin)
        start("tracker", [str(build / "rbt-m3t"), "run", "--config", str(config),
                          "--provider", metadata["camera_provider"], "--count", "3000", "--duration-sec", "180"],
              env, output, children, handles)
        gazebo, tracker = Journal(output / "gazebo_io.log"), Journal(output / "tracker.log")
        handles.extend((gazebo.channel, tracker.channel))
        kinematics = ArmKinematics(str(args.urdf.resolve()), target_frame=TOOL_LINK, joint_names=JOINTS)
        run_control(kinematics, io.stdin, tracker, gazebo, metadata, children, result)
        result["status"] = "MEASURED"
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        result["error"] = str(error)
    finally:
        stop_children(children, forced_kills)
        for handle in handles:
            handle.close()
        result["forced_kills"] = forced_kills
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"{result['status']} {output / 'result.json'}", flush=True)
    return 0 if result["status"] == "MEASURED" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-model", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--composer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bias-deg", nargs=5, type=float, default=[0.0] * len(JOINTS), metavar=(
        "SHOULDER_PAN", "SHOULDER_LIFT", "ELBOW_FLEX", "WRIST_FLEX", "WRIST_ROLL"),
        help="fixed encoder-prior offsets in SO101 joint order, in degrees (default: all zero)")
    parser.add_argument("--gazebo-python", default=Path("/usr/bin/python3"), type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
