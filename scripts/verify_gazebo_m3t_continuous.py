#!/usr/bin/env python3
"""Measure continuous resolved-rate servoing with M3T or encoder feedback in Gazebo.

MEASURED records completed motion and available measurements, with no accuracy
pass/fail threshold. Gazebo poses and joint values are used only before motion
for initial settling and after all children stop for independent evaluation.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from prepare_gazebo_m3t_scene import INITIAL_JOINTS, prepare
from verify_gazebo_m3t_position_ik import (
    Journal, MAX_PRIOR_AGE_NS, MAX_TRUTH_BRACKET_NS, configure_tracker, interpolate_truth,
    optical_xyz, sample_estimate, settled_after, validate_biases,
)
from verify_m3t_position_ik import JOINTS, TOOL_LINK, native_pose
from verify_software import runtime_environment, start, stop_children, wait_until


GAIN_PER_S = 1.0
DAMPING_M_PER_RAD = 0.002
MAX_SPEED_RAD_S = math.radians(6.0)
MAX_FEEDBACK_AGE_NS = 500_000_000
LIMIT_HORIZON_S = 0.5
MOVING_SPEED_RAD_S = math.radians(0.1)  # Reporting definition, not acceptance.
RECORDING_TAIL_NS = 100_000_000
RECORDING_TAIL_WALL_S = 3.0


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def emit(channel, value):
    channel.write(json.dumps(value, allow_nan=False) + "\n")
    channel.flush()


def velocity_scene(scene):
    """Change only this run's generated scene to ideal joint velocity control."""
    path = scene / "world.sdf"
    tree = ET.parse(path)
    metadata_path = scene / "scene_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    robot = tree.getroot().find(f"world/model[@name='{metadata['arm']}']")
    if robot is None:
        raise ValueError("prepared scene has no expected SO101 model")
    controllers = [plugin for plugin in robot.findall("plugin")
                   if plugin.get("name") == "gz::sim::systems::JointPositionController"]
    if len(controllers) != len(JOINTS):
        raise ValueError("expected exactly five generated joint position controllers")
    for index, (plugin, name) in enumerate(zip(controllers, JOINTS)):
        if (plugin.findtext("joint_name") != name
                or plugin.findtext("topic") != metadata["joint_command_topic"]
                or plugin.findtext("actuator_number") != str(index)):
            raise ValueError("generated grouped position controller order is unexpected")
        plugin.set("filename", "gz-sim-joint-controller-system")
        plugin.set("name", "gz::sim::systems::JointController")
        for child in list(plugin):
            if child.tag not in {"joint_name", "topic", "actuator_number"}:
                plugin.remove(child)
        ET.SubElement(plugin, "use_actuator_msg").text = "true"
        ET.SubElement(plugin, "use_force_commands").text = "false"
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    metadata.update(joint_command_mode="velocity", joint_command_field="velocity",
                    joint_command_units="radians_per_second",
                    joint_command_offsets_applied=False,
                    joint_controller="gz::sim::systems::JointController",
                    joint_controller_use_force_commands=False)
    write_json(metadata_path, metadata)
    return metadata


def trajectory(origin, elapsed_s, period_s=16.0, amplitude_x_m=0.010, amplitude_y_m=0.005):
    """Camera XY ellipse and analytic feedforward, starting at seed FK."""
    phase = 2 * math.pi * max(0.0, elapsed_s) / period_s
    omega = 2 * math.pi / period_s
    position = np.asarray(origin, dtype=float) + np.array([
        amplitude_x_m * math.sin(phase), amplitude_y_m * (1 - math.cos(phase)), 0.0])
    velocity = np.array([amplitude_x_m * omega * math.cos(phase),
                         amplitude_y_m * omega * math.sin(phase), 0.0])
    return position, velocity


def camera_jacobian(kinematics, joint_deg, camera_from_base):
    # The installed Pinocchio solver returns linear rows first, in base axes.
    jacobian = np.asarray(kinematics.solver._fk_jac(joint_deg, True)[1], dtype=float)
    if jacobian.shape != (6, len(JOINTS)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("installed kinematics returned an invalid six-row Jacobian")
    return camera_from_base[:3, :3] @ jacobian[:3, :]


def resolved_rate(jacobian, velocity_camera, joint_rad, lower_rad, upper_rad, age_s):
    """DLS in radians/second, uniformly speed capped, then limited near bounds."""
    if not 0 <= age_s <= MAX_FEEDBACK_AGE_NS / 1e9:
        raise ValueError("feedback is stale or from a future simulation time")
    jacobian = np.asarray(jacobian, dtype=float)
    velocity_camera = np.asarray(velocity_camera, dtype=float)
    joint_rad = np.asarray(joint_rad, dtype=float)
    if (jacobian.shape != (3, len(JOINTS)) or velocity_camera.shape != (3,)
            or joint_rad.shape != (len(JOINTS),)
            or not all(np.all(np.isfinite(value)) for value in (jacobian, velocity_camera, joint_rad))):
        raise ValueError("resolved-rate inputs must be finite and correctly shaped")
    raw = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + DAMPING_M_PER_RAD ** 2 * np.eye(3), velocity_camera)
    scale = min(1.0, MAX_SPEED_RAD_S / max(float(np.max(np.abs(raw))), 1e-30))
    capped = raw * scale
    # The actual joint could have moved by max_speed*age since the image. Reserve
    # that entire distance before allowing an outward velocity for the horizon.
    uncertainty = MAX_SPEED_RAD_S * age_s
    lower_speed = np.minimum(0.0, (np.asarray(lower_rad) - joint_rad + uncertainty) / LIMIT_HORIZON_S)
    upper_speed = np.maximum(0.0, (np.asarray(upper_rad) - joint_rad - uncertainty) / LIMIT_HORIZON_S)
    limited = np.clip(capped, lower_speed, upper_speed)
    if not np.all(np.isfinite(limited)):
        raise ValueError("DLS produced non-finite joint velocities")
    return limited, scale, bool(np.any(limited != capped))


def feedback_sample(event, metadata, kinematics, source, sample=None):
    """Select control state; visual fields always describe the passive M3T output."""
    if source not in ("m3t", "encoder"):
        raise ValueError("feedback source must be m3t or encoder")
    capture_ns, prior_age_ns = event["color_capture_ns"], event["joint_priors"]["age_ns"]
    if (type(capture_ns) is not int or capture_ns <= 0 or type(prior_age_ns) is not int
            or not 0 <= prior_age_ns <= MAX_PRIOR_AGE_NS or capture_ns - prior_age_ns <= 0):
        raise ValueError("feedback capture and prior sample timestamps must be valid")
    visual_deg, encoder_deg, visual_base, _gripper = (
        sample_estimate(event, metadata) if sample is None else sample)
    visual_xyz = (visual_base @ kinematics.forward(visual_deg))[:3, 3]
    joint_deg = visual_deg if source == "m3t" else encoder_deg
    control_base = visual_base if source == "m3t" else native_pose(metadata["optical_from_base"])
    control_xyz = visual_xyz if source == "m3t" else (control_base @ kinematics.forward(joint_deg))[:3, 3]
    if not all(np.all(np.isfinite(value)) for value in (visual_xyz, control_xyz, joint_deg)):
        raise ValueError("feedback FK or joint state is non-finite")
    return {"feedback_source": source, "q_feedback_rad": np.deg2rad(joint_deg).tolist(),
            "feedback_camera_xyz_m": control_xyz.tolist(),
            "feedback_sample_ns": capture_ns if source == "m3t" else capture_ns - prior_age_ns,
            "q_visual_rad": np.deg2rad(visual_deg).tolist(),
            "visual_camera_xyz_m": visual_xyz.tolist(),
            "jacobian": camera_jacobian(kinematics, joint_deg, control_base)}


def feedback_sample_age(feedback, now_ns):
    sample_ns = feedback["feedback_sample_ns"]
    if type(now_ns) is not int or type(sample_ns) is not int or sample_ns <= 0:
        raise ValueError("feedback sample and clock timestamps must be positive integers")
    age_ns = now_ns - sample_ns
    if not 0 <= age_ns <= MAX_FEEDBACK_AGE_NS:
        raise RuntimeError(f"{feedback['feedback_source']} feedback sample age {age_ns / 1e9:.6f}s exceeds the 0.5s bound")
    return age_ns


class SimulationClock:
    """The only Gazebo information read during motion is clock timestamps."""

    def __init__(self, journal):
        self.journal = journal
        self.index = 0
        self.stamp_ns = None

    def update(self):
        self.journal.update()
        if self.journal.records["clock_regression"]:
            raise ValueError("Gazebo simulation clock regressed")
        records = self.journal.records["clock"]
        for event in records[self.index:]:
            stamp = event["stamp_ns"]
            if (type(stamp) is not int or stamp < 0
                    or (self.stamp_ns is not None and stamp <= self.stamp_ns)):
                raise ValueError("Gazebo simulation clock is invalid or regressed")
            self.stamp_ns = stamp
            self.index += 1
        return self.stamp_ns


class VisionReader:
    def __init__(self, journal, metadata, kinematics, trace, feedback_source="m3t"):
        self.journal, self.metadata, self.kinematics, self.trace = journal, metadata, kinematics, trace
        self.index = 0
        self.last_sequence = self.last_capture = self.last_prior_sequence = None
        self.clock_identity = self.driver_identity = None
        self.feedback_source = feedback_source
        self.last_feedback_sample_ns = None
        self.latest = None

    def consume(self, now_ns, motion_start_ns=None, origin=None, args=None):
        self.journal.update()
        for event in self.journal.records["estimate"][self.index:]:
            # A 100 Hz clock journal can lag the camera callback by one tick.
            # Leave the event queued until a subsequent clock brackets it.
            if event["color_capture_ns"] > now_ns:
                return self.latest
            self.index += 1
            sample = sample_estimate(event, self.metadata)
            visual_deg, encoder_deg, _base, _gripper = sample
            stamp, sequence = event["color_capture_ns"], event["frame_sequence"]
            report = event["joint_priors"]
            prior_sequence = report["sequence"]
            identity = (event["capture_clock_source"], event["capture_clock_generation"])
            driver = (report["controller_id"], report["driver_session"])
            expected = {(self.metadata["arm"], name) for name in (*JOINTS, "gripper")}
            prior_keys = [(prior["arm"], prior["joint"]) for prior in report["priors"]]
            if (len(prior_keys) != len(expected) or set(prior_keys) != expected
                    or not all(type(prior["angle_rad"]) in (int, float)
                               and math.isfinite(prior["angle_rad"]) for prior in report["priors"])
                    or report.get("enabled") is not True
                    or not all(isinstance(value, str) and value for value in driver)
                    or not isinstance(identity[0], str) or not identity[0]
                    or type(identity[1]) is not int or identity[1] < 0
                    or type(prior_sequence) is not int or prior_sequence <= 0
                    or type(sequence) is not int or sequence < 0
                    or type(stamp) is not int or stamp <= 0
                    or not np.all(np.isfinite(visual_deg)) or not np.all(np.isfinite(encoder_deg))):
                raise ValueError("M3T estimate has missing or invalid priors, clock, frame, or visual joints")
            if self.last_sequence is not None and (
                    sequence <= self.last_sequence or stamp <= self.last_capture
                    or prior_sequence < self.last_prior_sequence or identity != self.clock_identity
                    or driver != self.driver_identity):
                raise ValueError("M3T frame, capture clock, or prior source regressed or changed")
            self.last_sequence, self.last_capture, self.last_prior_sequence = sequence, stamp, prior_sequence
            self.clock_identity, self.driver_identity = identity, driver
            selected = feedback_sample(event, self.metadata, self.kinematics, self.feedback_source, sample)
            sample_ns = selected["feedback_sample_ns"]
            if self.last_feedback_sample_ns is not None and sample_ns < self.last_feedback_sample_ns:
                raise ValueError("feedback sample time regressed")
            self.last_feedback_sample_ns = sample_ns
            age_ns = now_ns - sample_ns
            row = {"frame_sequence": sequence, "capture_ns": stamp,
                   "observed_clock_ns": now_ns, "feedback_age_ns": now_ns - stamp,
                   "feedback_sample_age_ns": age_ns,
                   "prior_age_ns": report["age_ns"], "prior_sequence": prior_sequence,
                   "capture_clock_identity": list(identity), "prior_driver_identity": list(driver),
                   **{key: value for key, value in selected.items() if key != "jacobian"},
                   "phase": "warmup" if motion_start_ns is None else "motion"}
            if motion_start_ns is None:
                row["used_for_feedback"] = 0 <= age_ns <= MAX_FEEDBACK_AGE_NS
            else:
                desired, _ = target_at(origin, sample_ns, motion_start_ns, args)
                row["desired_camera_xyz_m"] = desired.tolist()
                row["feedback_correction_camera_m_s"] = (
                    GAIN_PER_S * (desired - np.asarray(selected["feedback_camera_xyz_m"]))).tolist()
                row["used_for_feedback"] = 0 <= age_ns <= MAX_FEEDBACK_AGE_NS
            emit(self.trace, row)
            if row["used_for_feedback"]:
                self.latest = {**row, "jacobian": selected["jacobian"]}
        return self.latest


def target_at(origin, stamp_ns, start_ns, args):
    return trajectory(origin, (stamp_ns - start_ns) / 1e9, args.period_sec,
                      args.amplitude_x_m, args.amplitude_y_m)


class CommandStream:
    def __init__(self, pipe, journal, trace, result):
        self.pipe, self.journal, self.trace, self.result = pipe, journal, trace, result
        self.next_id = 0
        self.ack_index = 0
        self.rows = {}
        self.latest_feedback = self.latest_desired = self.latest_feedforward = None

    def send(self, now_ns, dq, feedback=None, desired=None, feedforward=None, **details):
        if feedback is not None:
            self.latest_feedback = feedback
        if desired is not None:
            self.latest_desired = desired
        if feedforward is not None:
            self.latest_feedforward = feedforward
        feedback = self.latest_feedback if feedback is None else feedback
        desired = self.latest_desired if desired is None else desired
        feedforward = self.latest_feedforward if feedforward is None else feedforward
        row = {"command_id": self.next_id, "issued_clock_ns": now_ns, "command_mode": "velocity",
               "clock_ns": now_ns, "capture_ns": None if feedback is None else feedback["capture_ns"],
               "dq_rad_s": np.asarray(dq, dtype=float).tolist(), "actual_publish_ns": None,
               "source_frame_sequence": None if feedback is None else feedback["frame_sequence"],
               "source_capture_ns": None if feedback is None else feedback["capture_ns"],
               "feedback_age_ns": None if feedback is None or now_ns is None else now_ns - feedback["capture_ns"],
               "feedback_source": self.result.get("feedback_source", "m3t") if feedback is None else feedback["feedback_source"],
               "feedback_sample_ns": None if feedback is None else feedback["feedback_sample_ns"],
               "source_sample_ns": None if feedback is None else feedback["feedback_sample_ns"],
               "feedback_sample_age_ns": None if feedback is None or now_ns is None else now_ns - feedback["feedback_sample_ns"],
               "prior_sequence": None if feedback is None else feedback["prior_sequence"],
               "prior_age_ns": None if feedback is None else feedback["prior_age_ns"],
               "q_feedback_rad": None if feedback is None else feedback["q_feedback_rad"],
               "feedback_camera_xyz_m": None if feedback is None else feedback["feedback_camera_xyz_m"],
               "desired_camera_xyz_m": None if desired is None else np.asarray(desired).tolist(),
               "visual_camera_xyz_m": None if feedback is None else feedback["visual_camera_xyz_m"],
               "q_visual_rad": None if feedback is None else feedback["q_visual_rad"],
               "feedforward_camera_m_s": None if feedforward is None else np.asarray(feedforward).tolist(),
               **details}
        self.rows[self.next_id] = row
        self.result["command_trace"].append(row)
        self.next_id += 1
        emit(self.trace, row)
        self.pipe.write(json.dumps({"command_id": row["command_id"], "dq_rad_s": row["dq_rad_s"]}) + "\n")
        self.pipe.flush()
        return row["command_id"]

    def acknowledge(self):
        self.journal.update()
        records = self.journal.records["command"]
        for receipt in records[self.ack_index:]:
            row = self.rows.get(receipt["command_id"])
            if row is None or row["actual_publish_ns"] is not None:
                raise ValueError("Gazebo acknowledged an unknown or duplicate command")
            if (receipt.get("command_mode") != "velocity" or receipt.get("dq_rad_s") != row["dq_rad_s"]
                    or receipt.get("actuator_values") != row["dq_rad_s"]):
                raise ValueError("Gazebo velocity acknowledgement differs from the requested command")
            row["actual_publish_ns"] = receipt["stamp_ns"]
            row["actual_feedback_age_ns"] = (None if row["source_capture_ns"] is None
                                             else receipt["stamp_ns"] - row["source_capture_ns"])
            row["actual_feedback_sample_age_ns"] = (None if row["feedback_sample_ns"] is None
                                                    else receipt["stamp_ns"] - row["feedback_sample_ns"])
            self.ack_index += 1


def check_children(processes):
    for process in processes:
        if process.poll() is not None:
            raise RuntimeError(f"child {process.args[0]} exited {process.returncode} during continuous control")


def timestamp_bracket(records, stamp):
    """Check recorded timestamp coverage without inspecting truth values."""
    index = bisect_left(records, stamp, key=lambda record: record["stamp_ns"])
    if index < len(records) and records[index]["stamp_ns"] == stamp:
        return True
    return (0 < index < len(records)
            and 0 < records[index]["stamp_ns"] - records[index - 1]["stamp_ns"] <= MAX_TRUTH_BRACKET_NS)


def collect_recording_tail(journal, processes, result):
    """After acknowledged zero, retain the timestamps needed for offline truth."""
    end_ns = result["motion_end_requested_ns"]
    zero_ns = result["final_zero_publish_ns"]
    target_ns = max(end_ns, zero_ns) + RECORDING_TAIL_NS
    began = time.monotonic()
    tail = {"complete": False, "minimum_sim_tail_ns": RECORDING_TAIL_NS,
            "wall_timeout_sec": RECORDING_TAIL_WALL_S, "target_clock_ns": target_ns,
            "zero_publish_ns": zero_ns, "motion_end_requested_ns": end_ns}
    result["recording_tail"] = tail
    while time.monotonic() - began < RECORDING_TAIL_WALL_S:
        check_children(processes)
        journal.update()
        if journal.records["clock_regression"]:
            raise ValueError("Gazebo clock regressed during post-zero recording tail")
        clocks = journal.records["clock"]
        poses, states = journal.records["pose"], journal.records["state"]
        now = clocks[-1]["stamp_ns"] if clocks else None
        pose_covered = timestamp_bracket(poses, end_ns)
        state_covered = timestamp_bracket(states, end_ns)
        tail.update(last_clock_ns=now, last_pose_ns=poses[-1]["stamp_ns"] if poses else None,
                    last_state_ns=states[-1]["stamp_ns"] if states else None,
                    motion_end_pose_covered=pose_covered, motion_end_state_covered=state_covered,
                    elapsed_wall_sec=time.monotonic() - began)
        if now is not None and now >= target_ns and pose_covered and state_covered:
            tail["complete"] = True
            return
        time.sleep(0.005)
    raise TimeoutError("post-zero recording tail did not reach 100ms simulation time and motion-end truth brackets")


def run_control(kinematics, commands, tracker, gazebo, metadata, limits, processes, result, args, vision_trace):
    initial_rad = np.array([INITIAL_JOINTS[name] for name in JOINTS])
    # This is the sole permitted truth-based setup check, before any trajectory.
    settled_ns = settled_after(gazebo, initial_rad, 0, processes)
    result["initial_settled_ns"] = settled_ns
    camera_from_base = native_pose(metadata["optical_from_base"])
    origin = (camera_from_base @ kinematics.forward(np.rad2deg(initial_rad)))[:3, 3]
    result["target_origin_camera_xyz_m"] = origin.tolist()
    result["anchor_camera_xyz_m"] = origin.tolist()
    result["target_origin_source"] = "known scene seed FK"
    clock = SimulationClock(gazebo)
    vision = VisionReader(tracker, metadata, kinematics, vision_trace, args.feedback_source)

    def fresh_initial():
        now = clock.update()
        if now is None:
            return False
        feedback = vision.consume(now)
        return (feedback is not None and feedback["feedback_sample_ns"] > settled_ns
                and 0 <= now - feedback["feedback_sample_ns"] <= MAX_FEEDBACK_AGE_NS)

    wait_until("waiting for one current M3T capture after tracker warmup",
               time.monotonic() + args.tracker_warmup_timeout_sec,
               fresh_initial, processes)
    start_ns = clock.stamp_ns
    duration_ns = round(args.duration_sec * 1e9)
    end_ns = start_ns + duration_ns
    result.update(motion_start_ns=start_ns, motion_end_ns=end_ns, motion_end_requested_ns=end_ns,
                  requested_duration_ns=duration_ns, trajectory_completed=False)
    feedback = vision.latest
    desired_capture, _ = target_at(origin, feedback["feedback_sample_ns"], start_ns, args)
    feedback["feedback_correction_camera_m_s"] = (
        GAIN_PER_S * (desired_capture - np.asarray(feedback["feedback_camera_xyz_m"]))).tolist()
    lower = np.array([limits[name]["lower_rad"] for name in JOINTS])
    upper = np.array([limits[name]["upper_rad"] for name in JOINTS])
    tick_ns = round(1e9 / args.command_hz)
    next_tick = start_ns
    wall_deadline = time.monotonic() + args.wall_timeout_sec
    last_feedback = feedback
    while time.monotonic() < wall_deadline:
        check_children(processes)
        now = clock.update()
        commands.acknowledge()
        if now >= end_ns:
            result.update(trajectory_completed=True, motion_end_observed_ns=now,
                          clock_stop_overshoot_ns=now - end_ns)
            return now, last_feedback
        latest = vision.consume(now, start_ns, origin, args)
        if latest is not None:
            last_feedback = latest
        age_ns = feedback_sample_age(last_feedback, now)
        if now >= next_tick:
            desired, feedforward = target_at(origin, now, start_ns, args)
            correction = np.asarray(last_feedback["feedback_correction_camera_m_s"])
            dq, scale, bounded = resolved_rate(last_feedback["jacobian"], feedforward + correction,
                                                last_feedback["q_feedback_rad"], lower, upper, age_ns / 1e9)
            commands.send(now, dq, last_feedback, desired, feedforward,
                          feedback_correction_camera_m_s=correction.tolist(),
                          uniform_speed_scale=scale, joint_bound_limited=bounded, reason="trajectory")
            # Do not burst obsolete commands when simulation time jumps.
            next_tick = start_ns + ((now - start_ns) // tick_ns + 1) * tick_ns
        time.sleep(0.001)
    raise TimeoutError("continuous trajectory exceeded its wall-clock bound")


def summary(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return {"count": int(values.size), "rms": float(np.sqrt(np.mean(values ** 2))),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def rate(stamps):
    return None if len(stamps) < 2 or stamps[-1] <= stamps[0] else (len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0])


def interpolate_state(states, stamp):
    index = bisect_left(states, stamp, key=lambda state: state["stamp_ns"])
    if index < len(states) and states[index]["stamp_ns"] == stamp:
        state = states[index]
        return np.asarray(state["q_rad"]), np.asarray(state["dq_rad_s"]), 0
    if index == 0 or index == len(states):
        return None
    before, after = states[index - 1:index + 1]
    span = after["stamp_ns"] - before["stamp_ns"]
    if not 0 < span <= MAX_TRUTH_BRACKET_NS:
        return None
    fraction = (stamp - before["stamp_ns"]) / span
    return tuple(np.asarray(before[field]) * (1 - fraction) + np.asarray(after[field]) * fraction
                 for field in ("q_rad", "dq_rad_s")) + (span,)


def truth_at_capture(poses, stamp, optical_from_base):
    index = bisect_left(poses, stamp, key=lambda pose: pose["stamp_ns"])
    if index < len(poses) and poses[index]["stamp_ns"] == stamp:
        return optical_xyz(poses[index], optical_from_base), 0
    return interpolate_truth(poses, stamp, optical_from_base)


def offline_analysis(output, result, metadata, kinematics, args):
    """Independent Gazebo q/dq/pose are first evaluated after children stop."""
    gazebo, tracker = Journal(output / "gazebo_io.log"), Journal(output / "tracker.log")
    try:
        gazebo.update()
        tracker.update()
        start_ns = result.get("motion_start_ns")
        if start_ns is None:
            return
        end_ns = result["motion_end_requested_ns"]
        optical_from_base = native_pose(metadata["optical_from_base"])
        origin = result["target_origin_camera_xyz_m"]
        poses, states = gazebo.records["pose"], gazebo.records["state"]
        rows, captures, missing_pose, missing_state = [], [], 0, 0
        missing_feedback_pose = missing_feedback_state = 0
        for event in tracker.records["estimate"]:
            stamp = event["color_capture_ns"]
            if not start_ns <= stamp <= end_ns:
                continue
            captures.append(stamp)
            sample = sample_estimate(event, metadata)
            visual_deg, _encoder, base, _gripper = sample
            selected = feedback_sample(event, metadata, kinematics, args.feedback_source, sample)
            visual_xyz = (base @ kinematics.forward(visual_deg))[:3, 3]
            truth = truth_at_capture(poses, stamp, optical_from_base)
            state = interpolate_state(states, stamp)
            if truth is None:
                missing_pose += 1
                continue
            desired, _ = target_at(origin, stamp, start_ns, args)
            truth_xyz, bracket_ns = truth
            row = {"frame_sequence": event["frame_sequence"], "capture_ns": stamp,
                   "elapsed_s": (stamp - start_ns) / 1e9, "desired_camera_xyz_m": desired.tolist(),
                   "truth_camera_xyz_m": truth_xyz.tolist(), "visual_camera_xyz_m": visual_xyz.tolist(),
                   "truth_pose_bracket_ns": bracket_ns,
                   "true_path_error_mm": float(np.linalg.norm(truth_xyz - desired) * 1000),
                   "m3t_to_truth_error_mm": float(np.linalg.norm(visual_xyz - truth_xyz) * 1000),
                   "q_visual_rad": np.deg2rad(visual_deg).tolist()}
            sample_ns = selected["feedback_sample_ns"]
            feedback_truth = truth if sample_ns == stamp else truth_at_capture(poses, sample_ns, optical_from_base)
            feedback_state = state if sample_ns == stamp else interpolate_state(states, sample_ns)
            row.update({key: selected[key] for key in (
                "feedback_source", "feedback_sample_ns", "q_feedback_rad", "feedback_camera_xyz_m")})
            row.update(feedback_sample_age_at_capture_ns=stamp - sample_ns,
                       prior_sequence=event["joint_priors"]["sequence"],
                       prior_age_ns=event["joint_priors"]["age_ns"])
            if feedback_truth is None:
                missing_feedback_pose += 1
            else:
                sample_truth_xyz, sample_bracket_ns = feedback_truth
                row.update(feedback_truth_camera_xyz_m=sample_truth_xyz.tolist(),
                           feedback_truth_pose_bracket_ns=sample_bracket_ns,
                           feedback_to_truth_error_mm=float(np.linalg.norm(
                               np.asarray(selected["feedback_camera_xyz_m"]) - sample_truth_xyz) * 1000))
            if feedback_state is None:
                missing_feedback_state += 1
            else:
                sample_truth_q, _sample_truth_dq, sample_state_bracket_ns = feedback_state
                row.update(feedback_sample_q_truth_rad=sample_truth_q.tolist(),
                           feedback_truth_state_bracket_ns=sample_state_bracket_ns,
                           max_feedback_joint_error_deg=float(np.max(np.abs(
                               np.rad2deg(np.asarray(selected["q_feedback_rad"]) - sample_truth_q)))))
            if state is None:
                missing_state += 1
            else:
                q, dq, span = state
                row.update(q_truth_rad=q.tolist(), dq_truth_rad_s=dq.tolist(), truth_state_bracket_ns=span,
                           moving=bool(np.max(np.abs(dq)) > MOVING_SPEED_RAD_S),
                           max_visual_joint_error_deg=float(np.max(np.abs(np.rad2deg(q) - visual_deg))))
            rows.append(row)
        with (output / "offline_measurements.jsonl").open("w", encoding="utf-8") as channel:
            for row in rows:
                emit(channel, row)
        receipts = [event for event in gazebo.records["command"]
                    if start_ns <= event["stamp_ns"] < end_ns and event["command_id"] in {
                        row["command_id"] for row in result["command_trace"] if row.get("reason") == "trajectory"}]
        moving = [row["moving"] for row in rows if "moving" in row]
        active_states = [state for state in states if start_ns <= state["stamp_ns"] <= end_ns]
        metrics = {"true_path_error_mm": summary([row["true_path_error_mm"] for row in rows]),
                   "m3t_to_truth_error_mm": summary([row["m3t_to_truth_error_mm"] for row in rows]),
                   "feedback_source": args.feedback_source,
                   "feedback_to_truth_error_mm": summary([
                       row["feedback_to_truth_error_mm"] for row in rows if "feedback_to_truth_error_mm" in row]),
                   "feedback_truth_alignment": "feedback_sample_ns; encoder=capture_ns-prior_age_ns; m3t=capture_ns",
                   "moving_frame_fraction": None if not moving else sum(moving) / len(moving),
                   "moving_frame_speed_threshold_deg_s": math.degrees(MOVING_SPEED_RAD_S),
                   "moving_frame_denominator": len(moving), "truth_frames": len(rows),
                   "missing_pose_brackets": missing_pose, "missing_state_brackets": missing_state,
                   "missing_feedback_pose_brackets": missing_feedback_pose,
                   "missing_feedback_state_brackets": missing_feedback_state,
                   "command_rate_hz_sim": rate([row["stamp_ns"] for row in receipts]),
                   "command_count_during_motion": len(receipts),
                   "vision_capture_rate_hz_sim": rate(captures),
                   "vision_frame_count": len(captures),
                   "truth_joint_state_rate_hz_sim": rate([row["stamp_ns"] for row in active_states]),
                   "max_truth_joint_speed_deg_s": None if not active_states else float(np.rad2deg(max(
                       np.max(np.abs(state["dq_rad_s"])) for state in active_states))),
                   "joint_order": list(JOINTS), "accuracy_acceptance": None}
        result["metrics"] = metrics
        result["measurement_complete"] = (
            len(rows) >= 2 and len(moving) == len(captures) and missing_pose == 0 and missing_state == 0
            and missing_feedback_pose == 0 and missing_feedback_state == 0)
    finally:
        gazebo.channel.close()
        tracker.channel.close()


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "INCOMPLETE", "accuracy_acceptance": None,
              "status_meaning": "MEASURED means duration and independent measurement completed; no accuracy cutoff",
              "command_mode": "velocity", "command_trace": [], "joint_order": list(JOINTS),
              "feedback_source": args.feedback_source, "control_gain_per_s": GAIN_PER_S,
              "tracker_warmup_timeout_sec": args.tracker_warmup_timeout_sec,
              "runner_source_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
              "encoder_feedback_source": "validated mapped angles from each M3T event joint_priors; configured bias retained",
              "m3t_role": "passive_observer" if args.feedback_source == "encoder" else "control_feedback",
              "feedback_update_cadence": "one update per consumed M3T estimate through the same capture/processing chain; actual latency measured independently",
              "requested_prior_bias_deg": dict(zip(JOINTS, args.bias_deg))}
    result["trajectory"] = {"duration_sec": args.duration_sec, "period_sec": args.period_sec,
                            "amplitude_x_m": args.amplitude_x_m, "amplitude_y_m": args.amplitude_y_m}
    children, handles, forced_kills = [], [], []
    command_stream = metadata = kinematics = None
    scripts = Path(__file__).resolve().parent
    try:
        from so101_control.kinematics import ArmKinematics

        bias_rad, limits = validate_biases(args.robot_model, args.bias_deg)
        model = json.loads(args.robot_model.read_text(encoding="utf-8"))
        if hashlib.sha256(args.urdf.read_bytes()).hexdigest() != model["source_sha256"]:
            raise ValueError("--urdf must match the canonical robot model source revision")
        result["joint_limits_and_seed"] = limits
        scene = output / "scene"
        prepare(args.robot_model, scene, position_ik=True)
        metadata = velocity_scene(scene)
        result["feedback_base_source"] = (
            "scene_metadata.optical_from_base" if args.feedback_source == "encoder" else "M3T base_link pose")
        result["fk_source"] = (
            "mapped prior q + fixed scene_metadata.optical_from_base" if args.feedback_source == "encoder"
            else "M3T fused visual q + current M3T base_link pose")
        result["jacobian_source"] = result["fk_source"] + "; installed Pinocchio linear Jacobian rotated into camera axes"
        result["known_optical_from_base"] = metadata["optical_from_base"]
        if tuple(metadata["joint_names"]) != JOINTS:
            raise ValueError("Gazebo joint order differs from the IK configuration")
        metadata_path = scene / "scene_metadata.json"
        scope = uuid.uuid4().hex
        service = f"rbt/so101/gazebo/{scope}/state/v1"
        config = configure_tracker(scene, metadata, service, bias_rad)
        tracker_config = json.loads(config.read_text(encoding="utf-8"))
        result["applied_prior_offsets_rad"] = {
            entry["joint"]: entry["offset_rad"] for entry in tracker_config["joint_priors"]["mappings"]}
        lock_path = scripts.parents[2] / "framework.lock.json"
        manifest = {"mode": "continuous_resolved_rate_visual_servo", "command_mode": "velocity",
                    "feedback_source": args.feedback_source, "control_gain_per_s": GAIN_PER_S,
                    "tracker_warmup_timeout_sec": result["tracker_warmup_timeout_sec"],
                    "runner_source_sha256": result["runner_source_sha256"],
                    "encoder_feedback_source": result["encoder_feedback_source"],
                    "m3t_role": result["m3t_role"],
                    "feedback_update_cadence": result["feedback_update_cadence"],
                    "feedback_base_source": result["feedback_base_source"],
                    "fk_source": result["fk_source"], "jacobian_source": result["jacobian_source"],
                    "known_optical_from_base": metadata["optical_from_base"],
                    "scene_metadata_path": str(metadata_path.resolve()),
                    "feedback_sample_timestamp": "capture_ns-prior_age_ns" if args.feedback_source == "encoder" else "capture_ns",
                    "robot_model": str(args.robot_model.resolve()), "urdf": str(args.urdf.resolve()),
                    "framework_lock": json.loads(lock_path.read_text()) if lock_path.exists() else None,
                    "joint_order": list(JOINTS), "scene_seed_rad": {name: INITIAL_JOINTS[name] for name in JOINTS},
                    "target_origin": "camera transform of known scene seed FK; independent of truth",
                    "target_formula": "x=x0+Ax*sin(2*pi*t/T); y=y0+Ay*(1-cos(2*pi*t/T)); z=z0",
                    "duration_sec": args.duration_sec, "period_sec": args.period_sec,
                    "amplitude_x_m": args.amplitude_x_m, "amplitude_y_m": args.amplitude_y_m,
                    "target_x_offset_range_m": [-args.amplitude_x_m, args.amplitude_x_m],
                    "target_y_offset_range_m": [0.0, 2 * args.amplitude_y_m],
                    "command_hz_sim_requested": args.command_hz, "visual_gain_per_s": GAIN_PER_S,
                    "dls_damping_m_per_rad": DAMPING_M_PER_RAD, "max_joint_speed_deg_s": 6.0,
                    "maximum_feedback_age_sec": MAX_FEEDBACK_AGE_NS / 1e9,
                    "joint_limit_horizon_sec": LIMIT_HORIZON_S,
                    "joint_limit_age_reserve": "6deg/s * image age before outward horizon bound",
                    "requested_prior_bias_deg": result["requested_prior_bias_deg"],
                    "applied_prior_offsets_rad": result["applied_prior_offsets_rad"],
                    "joint_limits_and_seed": limits, "tracking_config": str(config.resolve()),
                    "control_feedback": (
                        "mapped encoder joint_priors and known scene base transform; no fused joints or M3T poses"
                        if args.feedback_source == "encoder" else "M3T visual joints and base transform only"),
                    "truth_evaluation": "offline after child shutdown; path/M3T at capture times, feedback at its own sample timestamp",
                    "accuracy_acceptance": None, "wall_timeout_sec": args.wall_timeout_sec}
        write_json(output / "run_manifest.json", manifest)
        env = runtime_environment("hardware")
        env.update(GZ_PARTITION=f"gazebo-m3t-{scope}", GZ_HOMEDIR=str(output))
        with socket.socket() as handle:
            handle.bind(("127.0.0.1", 0))
            port = handle.getsockname()[1]
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
        build = args.build_dir.resolve()
        start("bridge", [str(args.gazebo_python), str(scripts / "bridge_gazebo_rgbd.py"),
                         "--topic", metadata["rgbd_topic"], "--fps", "15", "--provider-binary",
                         str(build / "rbt_m3t_gazebo_rgbd_provider"), "--joint-state-service", service,
                         "--scene-metadata", str(metadata_path)], env, output, children, handles)
        bridge = Journal(output / "bridge.log")
        handles.append(bridge.channel)

        def provider_ready():
            bridge.update()
            return bool(bridge.records["gazebo_rgbd_ready"])

        wait_until("starting Gazebo RGB-D and encoder bridge", time.monotonic() + 25, provider_ready, children)
        io_log = (output / "gazebo_io.log").open("w", encoding="utf-8")
        handles.append(io_log)
        io = subprocess.Popen([str(args.gazebo_python), str(scripts / "gazebo_position_ik_io.py"),
                               "--scene-metadata", str(metadata_path), "--command-mode", "velocity"],
                              env=env, stdin=subprocess.PIPE, stdout=io_log, stderr=subprocess.STDOUT,
                              start_new_session=True, text=True)
        children.append(io)
        handles.append(io.stdin)
        start("tracker", [str(build / "rbt-m3t"), "run", "--config", str(config),
                          "--provider", metadata["camera_provider"], "--count", "100000",
                          "--duration-sec", str(math.ceil(40 + args.tracker_warmup_timeout_sec + args.wall_timeout_sec))],
              env, output, children, handles)
        gazebo, tracker = Journal(output / "gazebo_io.log"), Journal(output / "tracker.log")
        handles.extend((gazebo.channel, tracker.channel))
        command_trace = (output / "commands.jsonl").open("w", encoding="utf-8")
        vision_trace = (output / "vision_trace.jsonl").open("w", encoding="utf-8")
        handles.extend((command_trace, vision_trace))
        command_stream = CommandStream(io.stdin, gazebo, command_trace, result)
        kinematics = ArmKinematics(str(args.urdf.resolve()), target_frame=TOOL_LINK, joint_names=JOINTS)
        now, feedback = run_control(kinematics, command_stream, tracker, gazebo, metadata,
                                    limits, children, result, args, vision_trace)
        command_stream.send(now, np.zeros(len(JOINTS)), feedback, reason="trajectory_complete_zero")
    except (ImportError, KeyError, OSError, RuntimeError, ValueError, TypeError, OverflowError,
            np.linalg.LinAlgError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    except KeyboardInterrupt:
        result["error"] = "interrupted"
    finally:
        # JointController retains velocity indefinitely. Always publish zero
        # before stopping any child, including failures during warmup/control.
        if command_stream is not None:
            try:
                command_stream.journal.update()
                clocks = command_stream.journal.records["clock"]
                now = clocks[-1]["stamp_ns"] if clocks else None
                zero_id = command_stream.send(now, np.zeros(len(JOINTS)), reason="finally_zero")
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    command_stream.acknowledge()
                    if command_stream.rows[zero_id]["actual_publish_ns"] is not None:
                        break
                    time.sleep(0.005)
                result["final_zero_acknowledged"] = command_stream.rows[zero_id]["actual_publish_ns"] is not None
                result["final_zero_publish_ns"] = command_stream.rows[zero_id]["actual_publish_ns"]
                if not result["final_zero_acknowledged"]:
                    result.setdefault("error", "final zero velocity command was not acknowledged")
            except (OSError, RuntimeError, ValueError) as error:
                result["zero_velocity_error"] = str(error)
                result["final_zero_acknowledged"] = False
                result.setdefault("error", "could not publish final zero velocity")
        if (command_stream is not None and metadata is not None
                and result.get("trajectory_completed") and result.get("final_zero_acknowledged")):
            try:
                collect_recording_tail(command_stream.journal, children, result)
            except (KeyError, OSError, RuntimeError, ValueError, KeyboardInterrupt) as error:
                result["recording_tail_error"] = f"{type(error).__name__}: {error}"
                result.setdefault("error", "post-zero recording tail was incomplete")
        stop_children(children, forced_kills)
        for handle in handles:
            handle.close()
        result["forced_kills"] = forced_kills
        if command_stream is not None:
            # Finalize the immediate request journal with actual Transport
            # publication timestamps, while retaining the unmodified IO log.
            with (output / "commands.jsonl").open("w", encoding="utf-8") as channel:
                for row in result["command_trace"]:
                    emit(channel, row)
        if metadata is not None and kinematics is not None and (output / "gazebo_io.log").exists():
            try:
                offline_analysis(output, result, metadata, kinematics, args)
            except (KeyError, OSError, RuntimeError, ValueError, TypeError) as error:
                result["offline_analysis_error"] = f"{type(error).__name__}: {error}"
        if (result.get("trajectory_completed") and result.get("measurement_complete")
                and result.get("final_zero_acknowledged") and "error" not in result
                and "offline_analysis_error" not in result):
            result["status"] = "MEASURED"
        write_json(output / "result.json", result)
        print(f"{result['status']} {output / 'result.json'}", flush=True)
    return 0 if result["status"] == "MEASURED" else 1


def main():
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("robot-model", "urdf", "build-dir", "composer", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--gazebo-python", type=Path, default=Path("/usr/bin/python3"))
    parser.add_argument("--feedback-source", choices=("m3t", "encoder"), default="m3t",
                        help="control state selected at the same M3T event cadence (default: m3t)")
    parser.add_argument("--bias-deg", type=float, nargs=5, default=[0.0] * len(JOINTS),
                        help="fixed encoder-prior offsets in SO101 joint order, in degrees")
    parser.add_argument("--duration-sec", type=float, default=16.0)
    parser.add_argument("--period-sec", type=float, default=16.0)
    parser.add_argument("--amplitude-x-m", type=float, default=0.010)
    parser.add_argument("--amplitude-y-m", type=float, default=0.005)
    parser.add_argument("--command-hz", type=float, default=50.0, help="requested simulation-time command rate")
    parser.add_argument("--wall-timeout-sec", type=float, default=180.0, help="wall-clock bound for motion only")
    parser.add_argument("--tracker-warmup-timeout-sec", type=float, default=90.0,
                        help="wall-clock bound for a fresh M3T capture during startup only (default: 90)")
    args = parser.parse_args()
    for name in ("duration_sec", "period_sec", "command_hz", "wall_timeout_sec", "tracker_warmup_timeout_sec"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("amplitude_x_m", "amplitude_y_m"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative and finite")
    if args.command_hz > 100 or round(args.duration_sec * 1e9) < 1:
        parser.error("command rate must be <=100Hz and duration must be at least one nanosecond")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
