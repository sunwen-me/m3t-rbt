#!/usr/bin/env python3
"""Capture static Gazebo/M3T joint estimates with fixed encoder-prior biases."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import subprocess
import time
import uuid

import numpy as np

from prepare_gazebo_m3t_scene import INITIAL_JOINTS, prepare
from verify_gazebo_m3t_position_ik import (
    MAX_PRIOR_AGE_NS,
    Journal,
    configure_tracker,
    settled_after,
    sample_estimate,
)
from verify_m3t_position_ik import JOINTS
from verify_software import runtime_environment, start, stop_children, wait_until


STATE_BRACKET_MAX_NS = 80_000_000
DEFAULT_FRAME_COUNT = 30
DEFAULT_WARMUP_FRAMES = 5
DEFAULT_TIMEOUT_S = 120.0


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


def add_prior_bias(config_path: Path, metadata: dict, bias_rad: np.ndarray) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mappings = config["joint_priors"]["mappings"]
    by_joint = {name: float(value) for name, value in zip(JOINTS, bias_rad)}
    applied = []
    for mapping in mappings:
        if mapping.get("arm") != metadata["arm"]:
            continue
        joint = mapping.get("joint")
        if joint in by_joint:
            mapping["offset_rad"] = float(mapping["offset_rad"] + by_joint[joint])
            applied.append({"arm": mapping["arm"], "joint": joint,
                            "additional_offset_rad": by_joint[joint]})
    if {entry["joint"] for entry in applied} != set(JOINTS):
        raise ValueError("could not apply a prior offset to each of the five SO101 joints")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return applied


def interpolate_joint_truth(states: list[dict], stamp_ns: int) -> tuple[np.ndarray, int] | None:
    if len(states) < 2:
        return None
    offset = bisect_left(states, stamp_ns, key=lambda state: state["stamp_ns"])
    if offset == 0 or offset == len(states):
        return None
    before, after = states[offset - 1:offset + 1]
    span = after["stamp_ns"] - before["stamp_ns"]
    if not 0 < span <= STATE_BRACKET_MAX_NS:
        return None
    factor = (stamp_ns - before["stamp_ns"]) / span
    first = np.asarray(before["q_rad"], dtype=float)
    last = np.asarray(after["q_rad"], dtype=float)
    return first + factor * (last - first), span


def collect_static_samples(tracker: Journal, gazebo: Journal, metadata: dict,
                           after_ns: int, frame_count: int, timeout_s: float,
                           processes: list[subprocess.Popen]) -> list[dict]:
    samples: list[dict] = []
    event_index = 0
    latest_stamp = after_ns
    latest_sequence = None
    clock_identity = None

    def ready() -> bool:
        nonlocal event_index, latest_stamp, latest_sequence, clock_identity
        tracker.update()
        gazebo.update()
        estimates = tracker.records["estimate"]
        states = gazebo.records["state"]
        while event_index < len(estimates):
            event = estimates[event_index]
            stamp = event.get("color_capture_ns")
            if type(stamp) is not int or stamp <= latest_stamp:
                event_index += 1
                continue
            truth = interpolate_joint_truth(states, stamp)
            if truth is None:
                if not states or states[-1]["stamp_ns"] <= stamp:
                    return False  # Wait for a future state; never extrapolate.
                event_index += 1
                continue
            state_rad, bracket_ns = truth
            sequence = event.get("frame_sequence")
            if type(sequence) is not int or (latest_sequence is not None and sequence <= latest_sequence):
                raise ValueError("M3T frame sequence did not advance during static capture")
            identity = (event.get("capture_clock_source"), event.get("capture_clock_generation"))
            if clock_identity is not None and identity != clock_identity:
                raise ValueError("Gazebo camera capture clock changed while sampling M3T")
            clock_identity = identity
            visual_deg, prior_deg, _, _ = sample_estimate(event, metadata)
            prior_report = event["joint_priors"]
            age_ns = prior_report.get("age_ns")
            if (type(age_ns) is not int
                    or prior_report.get("capture_time_associated") is not True
                    or not 0 <= age_ns <= MAX_PRIOR_AGE_NS):
                raise ValueError("M3T estimate is not a fresh, capture-aligned joint-prior sample")
            samples.append({
                "frame_sequence": sequence,
                "capture_ns": stamp,
                "prior_age_ns": age_ns,
                "truth_bracket_ns": bracket_ns,
                "truth_rad": state_rad.tolist(),
                "prior_rad": np.deg2rad(prior_deg).tolist(),
                "estimate_rad": np.deg2rad(visual_deg).tolist(),
            })
            event_index += 1
            latest_stamp = stamp
            latest_sequence = sequence
            if len(samples) >= frame_count:
                return True
        return False

    wait_until(f"collecting {frame_count} fresh M3T estimates with timestamp-aligned joint truth",
               time.monotonic() + timeout_s, ready, processes)
    return samples


def error_stats(values: np.ndarray) -> dict:
    return {
        "mean_deg": float(np.mean(values)),
        "median_deg": float(np.median(values)),
        "std_deg": float(np.std(values)),
        "mean_absolute_deg": float(np.mean(np.abs(values))),
        "max_absolute_deg": float(np.max(np.abs(values))),
        "min_deg": float(np.min(values)),
        "max_deg": float(np.max(values)),
    }


def summarize(samples: list[dict], bias_deg: np.ndarray, states: list[dict]) -> dict:
    truth = np.asarray([sample["truth_rad"] for sample in samples], dtype=float)
    prior = np.asarray([sample["prior_rad"] for sample in samples], dtype=float)
    estimate = np.asarray([sample["estimate_rad"] for sample in samples], dtype=float)
    truth_deg = np.rad2deg(truth)
    prior_error_deg = np.rad2deg(prior - truth)
    estimate_error_deg = np.rad2deg(estimate - truth)
    first_stamp, last_stamp = samples[0]["capture_ns"], samples[-1]["capture_ns"]
    window_states = [state for state in states
                     if first_stamp <= state["stamp_ns"] <= last_stamp]
    if len(window_states) < 2:
        raise ValueError("not enough Gazebo joint-state samples cover the measured capture window")
    state_q_deg = np.rad2deg(np.asarray([state["q_rad"] for state in window_states], dtype=float))
    state_dq_deg_s = np.rad2deg(np.asarray([state["dq_rad_s"] for state in window_states], dtype=float))
    axes = {}
    for index, name in enumerate(JOINTS):
        measured_bias = prior_error_deg[:, index]
        residual = float(np.mean(measured_bias) - bias_deg[index])
        axes[name] = {
            "truth_deg": error_stats(truth_deg[:, index]),
            "truth_peak_to_peak_deg": float(np.ptp(state_q_deg[:, index])),
            "truth_max_abs_speed_deg_s": float(np.max(np.abs(state_dq_deg_s[:, index]))),
            "prior_minus_truth_deg": error_stats(measured_bias),
            "estimate_minus_truth_deg": error_stats(estimate_error_deg[:, index]),
            "requested_prior_bias_deg": float(bias_deg[index]),
            "measured_prior_bias_mean_deg": float(np.mean(measured_bias)),
            "bias_residual_deg": residual,
        }
    return {
        "frames": len(samples),
        "capture_duration_s": (samples[-1]["capture_ns"] - samples[0]["capture_ns"]) / 1e9,
        "max_prior_age_ms": max(sample["prior_age_ns"] for sample in samples) / 1e6,
        "max_truth_bracket_ms": max(sample["truth_bracket_ns"] for sample in samples) / 1e6,
        "bias_effect_check": "compare requested_prior_bias_deg with measured_prior_bias_mean_deg; no acceptance threshold",
        "truth_motion_check": "per-axis peak-to-peak angle and maximum absolute speed are reported; no stationarity threshold",
        "axes": axes,
        "samples": samples,
    }


def run(args) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "INCOMPLETE",
        "mode": "static_capture_only",
        "commands_written_by_script": 0,
        "gazebo_command_events": 0,
        "requested_bias_deg": dict(zip(JOINTS, args.bias_deg)),
        "joint_limits_and_seed": None,
        "accuracy_acceptance": None,
    }
    children: list[subprocess.Popen] = []
    handles: list = []
    forced_kills: list[str] = []
    io_process = None
    try:
        bias_rad, limits = validate_biases(args.robot_model, args.bias_deg)
        result["joint_limits_and_seed"] = limits
        scene = output / "scene"
        prepare(args.robot_model, scene, position_ik=True)
        metadata_path = scene / "scene_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata["joint_names"]) != JOINTS:
            raise ValueError("Gazebo joint order differs from the static bias experiment")

        scope = uuid.uuid4().hex
        service = f"rbt/so101/gazebo/{scope}/state/v1"
        config_path = configure_tracker(scene, metadata, service)
        applied_offsets = add_prior_bias(config_path, metadata, bias_rad)
        env = runtime_environment("hardware")
        env.update(GZ_PARTITION=f"gazebo-m3t-joint-bias-{scope}", GZ_HOMEDIR=str(output))
        with socket.socket() as socket_handle:
            socket_handle.bind(("127.0.0.1", 0))
            port = socket_handle.getsockname()[1]
        env["RBT_COMPOSER_ADDR"] = f"127.0.0.1:{port}"

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "static_capture_only",
            "robot_model": str(args.robot_model.resolve()),
            "build_dir": str(args.build_dir.resolve()),
            "composer": str(args.composer.resolve()),
            "gazebo_python": str(args.gazebo_python.resolve()),
            "output_dir": str(output),
            "joint_order": list(JOINTS),
            "scene_seed_rad": {name: INITIAL_JOINTS[name] for name in JOINTS},
            "requested_bias_deg": dict(zip(JOINTS, args.bias_deg)),
            "injected_prior_offsets": applied_offsets,
            "frames_requested": args.frames,
            "warmup_frames_discarded": args.warmup_frames,
            "measurement_frames": args.frames - args.warmup_frames,
            "capture_timeout_s": args.timeout_s,
            "commands_written_by_script": 0,
            "gazebo_command_events_expected": 0,
            "tracking_config": str(config_path.resolve()),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        scene_metadata = metadata_path
        scripts = Path(__file__).resolve().parent
        start("composer", [str(args.composer.resolve())], env, output, children, handles)

        def composer_ready() -> bool:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return True
            except OSError:
                return False

        wait_until("starting isolated Composer", time.monotonic() + 10, composer_ready, children)
        start("gazebo", ["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "3",
                         str(scene / "world.sdf")], env, output, children, handles)
        start("bridge", [str(args.gazebo_python.resolve()), str(scripts / "bridge_gazebo_rgbd.py"),
                          "--topic", metadata["rgbd_topic"], "--fps", "15",
                          "--provider-binary", str(args.build_dir.resolve() / "rbt_m3t_gazebo_rgbd_provider"),
                          "--joint-state-service", service, "--scene-metadata", str(scene_metadata)],
              env, output, children, handles)
        bridge = Journal(output / "bridge.log")
        handles.append(bridge.channel)

        def provider_ready() -> bool:
            bridge.update()
            return bool(bridge.records["gazebo_rgbd_ready"])

        wait_until("starting Gazebo RGB-D and encoder bridge", time.monotonic() + 25,
                   provider_ready, children)

        io_log = (output / "gazebo_io.log").open("w", encoding="utf-8")
        handles.append(io_log)
        io_process = subprocess.Popen(
            [str(args.gazebo_python.resolve()), str(scripts / "gazebo_position_ik_io.py"),
             "--scene-metadata", str(scene_metadata)],
            env=env, stdin=subprocess.PIPE, stdout=io_log, stderr=subprocess.STDOUT,
            start_new_session=True, text=True)
        children.append(io_process)
        # Keep stdin open but never write to it: the sidecar's command path is unused.
        start("tracker", [str(args.build_dir.resolve() / "rbt-m3t"), "run", "--config",
                          str(config_path), "--provider", metadata["camera_provider"],
                          "--count", "3000", "--duration-sec", "180"],
              env, output, children, handles)
        gazebo = Journal(output / "gazebo_io.log")
        tracker = Journal(output / "tracker.log")
        handles.extend((gazebo.channel, tracker.channel))

        initial_rad = np.asarray([INITIAL_JOINTS[name] for name in JOINTS], dtype=float)
        settled_ns = settled_after(gazebo, initial_rad, 0, children)
        samples = collect_static_samples(tracker, gazebo, metadata, settled_ns,
                                         args.frames, args.timeout_s, children)
        warmup_samples = samples[:args.warmup_frames]
        measurement_samples = samples[args.warmup_frames:]
        gazebo.update()
        tracker.update()
        command_events = len(gazebo.records["command"])
        result["gazebo_command_events"] = command_events
        if command_events != 0:
            raise RuntimeError("Gazebo IO observed a command event during capture-only experiment")
        result["settled_ns"] = settled_ns
        result["warmup_frames_discarded"] = len(warmup_samples)
        result["warmup_frame_sequences"] = [sample["frame_sequence"] for sample in warmup_samples]
        result["measurement"] = summarize(
            measurement_samples, np.asarray(args.bias_deg, dtype=float), gazebo.records["state"])
        result["measurement"]["frames_collected_including_warmup"] = len(samples)
        result["status"] = "MEASURED"
    except (KeyError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        result["error"] = str(error)
    finally:
        stop_children(children, forced_kills)
        if io_process is not None and io_process.stdin is not None:
            try:
                io_process.stdin.close()
            except OSError:
                pass
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass
        result["forced_kills"] = forced_kills
        (output / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{result['status']} {output / 'result.json'}", flush=True)
    return 0 if result["status"] == "MEASURED" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-model", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--composer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bias-deg", required=True, nargs=5, type=float, metavar=(
        "SHOULDER_PAN", "SHOULDER_LIFT", "ELBOW_FLEX", "WRIST_FLEX", "WRIST_ROLL"),
        help="fixed encoder-prior offsets in the five-joint URDF order, in degrees")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_COUNT,
                        help=f"total fresh, timestamp-aligned estimates before warmup removal (default: {DEFAULT_FRAME_COUNT})")
    parser.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES,
                        help=f"initial estimates discarded from statistics (default: {DEFAULT_WARMUP_FRAMES})")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S,
                        help=f"capture timeout after settling (default: {DEFAULT_TIMEOUT_S:g})")
    parser.add_argument("--gazebo-python", default=Path("/usr/bin/python3"), type=Path)
    args = parser.parse_args()
    if args.warmup_frames < 0 or args.frames - args.warmup_frames < 2:
        parser.error("--frames must exceed --warmup-frames by at least two measured frames")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be a finite positive duration")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
