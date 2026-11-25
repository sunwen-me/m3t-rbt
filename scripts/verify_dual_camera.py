#!/usr/bin/env python3
"""Verify native SOFTWARE_FIXTURE dual-camera tracking with independent providers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_software as software


FRAME_PERIOD_NS = 1_000_000_000 // 30
BLIND_PHASES = range(20)
SECONDARY_BLANK_PHASES = range(50, 62)
PHASE_PERIOD = 120
SAMPLE_COUNT = 240
TIMEOUT_SECONDS = 90.0
PRIMARY_PROVIDER = "orbbec335"
SECONDARY_PROVIDER = "orbbec335-secondary"
SECONDARY_SERIAL = "SOFTWARE_FIXTURE_SECONDARY"
SECONDARY_FRAME = "fixture/secondary_color_optical"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def number(value, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise RuntimeError(f"{description} must be finite numeric data")
    return float(value)


def integer(value, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{description} must be an integer >= {minimum}")
    return value


def secondary_transform() -> list[float]:
    # T_primary<-secondary: a small translated and rolled camera pose. The
    # fixture renders in its inverse, so ignoring this transform is observable.
    angle = math.radians(4.0)
    cosine, sine = math.cos(angle), math.sin(angle)
    return [cosine, -sine, 0.0, 0.01,
            sine, cosine, 0.0, 0.015,
            0.0, 0.0, 1.0, 0.02,
            0.0, 0.0, 0.0, 1.0]


def create_config(directory: Path, profile: str) -> tuple[Path, dict, list[float]]:
    build_profile = "extended" if profile == "extended" else "baseline"
    config_path = software.fixture_config(directory, profile=build_profile)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_divides"] = 2
    config["model_points"] = 200
    config["model_image_size"] = 512
    if profile == "depth-only":
        config["modalities"] = {"region": False, "depth": True, "texture": False}
    transform = secondary_transform()
    config["secondary_camera"] = {
        "provider": SECONDARY_PROVIDER,
        "serial": SECONDARY_SERIAL,
        "coordinate_frame": SECONDARY_FRAME,
        "camera_in_primary": transform,
        "maximum_skew_ms": 20,
        "time_basis": "capture",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path, config, transform


def verify_fixture_frames(primary_events: list[dict], secondary_events: list[dict]) -> dict:
    def collect(events: list[dict], role: str) -> dict[int, dict]:
        frames = [event for event in events
                  if event.get("event") == "fixture_frame"
                  and event.get("camera_role") == role]
        result = {}
        for frame in frames:
            index = integer(frame.get("frame_index"), f"{role} frame_index", 1)
            capture = integer(frame.get("capture_ns"), f"{role} capture_ns", 1)
            require(capture == index * FRAME_PERIOD_NS,
                    f"{role} frame {index} does not carry its shared synthetic capture stamp")
            require(index not in result, f"{role} fixture duplicated frame index {index}")
            blank = frame.get("image_blank")
            require(isinstance(blank, bool), f"{role} image_blank must be boolean")
            pixels = integer(frame.get("target_depth_pixels"), f"{role} target depth pixels")
            require(blank == (pixels == 0), f"{role} blank diagnostic disagrees with depth image")
            phase = index % PHASE_PERIOD
            if role == "primary":
                blind = frame.get("primary_blind")
                require(blind == (phase in BLIND_PHASES),
                        f"primary blind schedule is wrong at frame {index}")
                require(blank == (phase in BLIND_PHASES),
                        f"primary blank image schedule is wrong at frame {index}")
            else:
                secondary_blank = frame.get("secondary_blank")
                require(secondary_blank == (phase in SECONDARY_BLANK_PHASES),
                        f"secondary blank image schedule is wrong at frame {index}")
                require(blank == (phase in SECONDARY_BLANK_PHASES),
                        f"secondary blank image diagnostic is wrong at frame {index}")
            result[index] = frame
        require(len(result) >= 120, f"{role} fixture produced too few frame diagnostics")
        return result

    primary = collect(primary_events, "primary")
    secondary = collect(secondary_events, "secondary")
    shared = set(primary) & set(secondary)
    require(len(shared) >= 100, "independent fixtures did not produce enough matching capture frames")
    blind = [index for index in shared if index % PHASE_PERIOD in BLIND_PHASES]
    secondary_blank = [index for index in shared
                       if index % PHASE_PERIOD in SECONDARY_BLANK_PHASES]
    require(len(blind) >= 12, "matched fixture feeds did not include the primary blind interval")
    require(len(secondary_blank) >= 8, "matched fixture feeds did not include the secondary blank interval")
    require(all(not secondary[index]["image_blank"] for index in blind),
            "secondary image was blank during the primary blind interval")
    require(all(not primary[index]["image_blank"] for index in secondary_blank),
            "primary image was blank during the secondary blank interval")
    return {
        "primary_frames": len(primary),
        "primary_nonblank_frames": sum(not frame["image_blank"] for frame in primary.values()),
        "primary_blind_frames": sum(frame["primary_blind"] for frame in primary.values()),
        "secondary_frames": len(secondary),
        "secondary_nonblank_frames": sum(not frame["image_blank"] for frame in secondary.values()),
        "secondary_blank_frames": sum(frame["secondary_blank"] for frame in secondary.values()),
        "matched_capture_frames": len(shared),
        "matched_primary_blind_frames": len(blind),
        "matched_secondary_blank_frames": len(secondary_blank),
    }


def reference_angles(frame_index: int) -> dict[str, float]:
    phase = ((frame_index - 1) % 100) / 100.0
    triangle = -1.0 + 4.0 * phase if phase < 0.5 else 3.0 - 4.0 * phase
    left = 0.15 + 0.22 * triangle
    return {"so101_left": left, "so101_right": -left}


def verify_tracking(events: list[dict], summary: dict, profile: str,
                    transform: list[float], frame_counts: dict) -> dict:
    expected_modalities = {"baseline": ["region", "depth"],
                           "depth-only": ["depth"],
                           "extended": ["region", "depth", "texture"]}[profile]
    expected_algorithm = ("m3t_region_depth_articulated"
                          if profile == "baseline" else "m3t_articulated")
    require(summary.get("event") == "tracking_samples_consumed"
            and summary.get("mode") == "SOFTWARE_FIXTURE"
            and summary.get("hardware") == "NOT RUN"
            and summary.get("provider") == "m3t-so101"
            and summary.get("count") == SAMPLE_COUNT,
            "native consumer did not report the requested SOFTWARE_FIXTURE sample count")
    samples = [event for event in events if event.get("event") == "tracking_sample"]
    require(len(samples) == SAMPLE_COUNT,
            f"expected {SAMPLE_COUNT} native tracking samples, got {len(samples)}")
    expected_clock = "534f4654574152455f46495854555245" + "0" * 64
    previous_capture = 0
    samples_by_frame = {}
    total_errors = {"so101_left": [], "so101_right": []}
    total_seed_errors = {"so101_left": [], "so101_right": []}
    blind_errors = {"so101_left": [], "so101_right": []}
    blind_seed_errors = {"so101_left": [], "so101_right": []}
    blind_estimates = {"so101_left": [], "so101_right": []}
    blind_truth = {"so101_left": [], "so101_right": []}
    recovered_errors = {"so101_left": [], "so101_right": []}
    recovered_truth = {"so101_left": [], "so101_right": []}
    estimates_by_frame = {"so101_left": {}, "so101_right": {}}
    provenance_samples = 0

    for sample in samples:
        require(sample.get("mode") == "SOFTWARE_FIXTURE"
                and sample.get("hardware") == "NOT RUN"
                and sample.get("provider") == "m3t-so101",
                "tracking sample is not labeled SOFTWARE_FIXTURE")
        capture_ns = integer(sample.get("capture_ns"), "primary capture_ns", 1)
        require(capture_ns > previous_capture and capture_ns % FRAME_PERIOD_NS == 0,
                "tracking captures are not strictly increasing shared fixture stamps")
        previous_capture = capture_ns
        frame_index = capture_ns // FRAME_PERIOD_NS
        estimate = sample.get("estimate")
        require(isinstance(estimate, dict)
                and estimate.get("event") == "estimate"
                and estimate.get("schema_version") == 1
                and estimate.get("algorithm") == expected_algorithm
                and estimate.get("modalities") == expected_modalities,
                "tracking estimate does not report the configured M3T modality set")
        require(estimate.get("frame_id") == "fixture/color_optical"
                and estimate.get("capture_ns") == capture_ns,
                "estimate primary frame identity does not match the consumer sample")
        provenance = estimate.get("secondary_camera")
        require(isinstance(provenance, dict),
                "tracking estimate omitted secondary camera timing and calibration provenance")
        require(provenance.get("provider") == SECONDARY_PROVIDER
                and provenance.get("serial") == SECONDARY_SERIAL
                and provenance.get("frame_id") == SECONDARY_FRAME
                and provenance.get("time_basis") == "capture"
                and provenance.get("maximum_skew_ns") == 20_000_000,
                "tracking estimate reports the wrong secondary camera identity or timing policy")
        emitted_transform = provenance.get("camera_in_primary")
        require(isinstance(emitted_transform, list) and len(emitted_transform) == 16,
                "secondary camera calibration must be a 16-value row-major transform")
        require(all(abs(number(actual, "camera_in_primary") - expected) < 1.0e-5
                    for actual, expected in zip(emitted_transform, transform)),
                "secondary provenance does not preserve the nonidentity configured extrinsic")

        primary_timing = provenance.get("primary_timing")
        secondary_timing = provenance.get("secondary_timing")
        require(isinstance(primary_timing, dict) and isinstance(secondary_timing, dict),
                "secondary provenance omitted per-view timing")
        view_capture = {}
        for view, timing in (("primary", primary_timing), ("secondary", secondary_timing)):
            capture = timing.get("capture")
            received = timing.get("received")
            require(isinstance(capture, dict) and isinstance(received, dict),
                    f"{view} provenance timing must contain capture and received clocks")
            stamp = integer(capture.get("timestamp_ns"), f"{view} capture timestamp", 1)
            require(capture.get("clock_kind") == "Device"
                    and capture.get("clock_source") == expected_clock
                    and capture.get("clock_generation") == 0,
                    f"{view} capture clock is not the shared synthetic device clock")
            integer(received.get("timestamp_ns"), f"{view} received timestamp", 1)
            require(received.get("clock_kind") == "HostMonotonic"
                    and isinstance(received.get("clock_source"), str)
                    and len(received["clock_source"]) > 0,
                    f"{view} receive timestamp is not a complete host-monotonic timestamp")
            integer(received.get("clock_generation"), f"{view} received clock generation")
            require(stamp % FRAME_PERIOD_NS == 0,
                    f"{view} capture does not use the fixture's shared frame clock")
            view_capture[view] = stamp
        skew = integer(provenance.get("cross_view_skew_ns"), "cross-view skew")
        require(skew == abs(view_capture["primary"] - view_capture["secondary"])
                and skew == 0 and skew <= 20_000_000,
                "capture-basis cross-view skew does not satisfy the configured bound")
        require(view_capture["primary"] == capture_ns,
                "primary provenance timestamp differs from the tracking output capture")
        require(view_capture["secondary"] == capture_ns,
                "secondary capture stamp does not match the shared truth frame")
        provenance_samples += 1
        samples_by_frame[frame_index] = sample

        truth = reference_angles(frame_index)
        angles = estimate.get("joint_angles_rad")
        require(isinstance(angles, dict) and set(angles) == set(truth),
                "tracking estimate must report both fixture shoulder joints")
        for arm in truth:
            joints = angles[arm]
            require(isinstance(joints, dict) and "shoulder" in joints,
                    f"{arm} shoulder angle is missing")
            observed = number(joints["shoulder"], f"{arm} shoulder")
            estimates_by_frame[arm][frame_index] = observed
            error = (observed - truth[arm]) ** 2
            total_errors[arm].append(error)
            total_seed_errors[arm].append(truth[arm] ** 2)
            phase = frame_index % PHASE_PERIOD
            if phase in BLIND_PHASES:
                blind_errors[arm].append(error)
                blind_seed_errors[arm].append(truth[arm] ** 2)
                blind_estimates[arm].append(observed)
                blind_truth[arm].append(truth[arm])
            elif phase in range(62, 74):
                recovered_errors[arm].append(error)
                recovered_truth[arm].append(truth[arm])

    require(provenance_samples == SAMPLE_COUNT,
            "not every native estimate contains secondary camera provenance")
    blind_indices = [index for index in samples_by_frame
                     if index % PHASE_PERIOD in BLIND_PHASES]
    secondary_blank_indices = [index for index in samples_by_frame
                               if index % PHASE_PERIOD in SECONDARY_BLANK_PHASES]
    require(len(blind_indices) >= 24,
            "tracking output did not cover enough primary blind frames to evaluate secondary handoff")
    require(len(secondary_blank_indices) >= 12,
            "tracking output did not cover enough secondary blank-image frames")
    require(all(frame_counts["primary_by_index"].get(index, {}).get("image_blank") is True
                and frame_counts["secondary_by_index"].get(index, {}).get("image_blank") is False
                for index in blind_indices),
            "primary blind output samples do not align with a nonblank secondary capture")
    require(all(frame_counts["primary_by_index"].get(index, {}).get("image_blank") is False
                and frame_counts["secondary_by_index"].get(index, {}).get("image_blank") is True
                for index in secondary_blank_indices),
            "secondary blank output samples do not align with a nonblank primary capture")

    rmse = {arm: math.sqrt(math.fsum(total_errors[arm]) / len(samples)) for arm in total_errors}
    zero_seed = {arm: math.sqrt(math.fsum(total_seed_errors[arm]) / len(samples))
                 for arm in total_seed_errors}
    blind_rmse = {arm: math.sqrt(math.fsum(blind_errors[arm]) / len(blind_errors[arm]))
                  for arm in blind_errors}
    blind_zero_seed = {
        arm: math.sqrt(math.fsum(blind_seed_errors[arm]) / len(blind_seed_errors[arm]))
        for arm in blind_seed_errors}
    blind_motion = {arm: max(blind_estimates[arm]) - min(blind_estimates[arm])
                    for arm in blind_estimates}
    blind_reference_motion = {arm: max(blind_truth[arm]) - min(blind_truth[arm])
                              for arm in blind_truth}
    recovery_rmse = {
        arm: math.sqrt(math.fsum(recovered_errors[arm]) / len(recovered_errors[arm]))
        for arm in recovered_errors}
    for arm in rmse:
        require(zero_seed[arm] >= 0.02 and rmse[arm] <= 0.85 * zero_seed[arm],
                f"{arm} dual-camera RMSE {rmse[arm]:.5f} did not beat frozen-seed RMSE {zero_seed[arm]:.5f}")
        require(blind_zero_seed[arm] >= 0.02
                and blind_rmse[arm] <= 0.85 * blind_zero_seed[arm],
                f"{arm} primary-blind RMSE {blind_rmse[arm]:.5f} did not beat frozen-seed RMSE "
                f"{blind_zero_seed[arm]:.5f}")
        require(blind_reference_motion[arm] >= 0.03
                and blind_motion[arm] >= max(0.02, 0.35 * blind_reference_motion[arm]),
                f"{arm} did not follow fixture motion while primary RGB-D was blank")
        require(bool(recovered_errors[arm]) and recovery_rmse[arm] <= 0.9 * zero_seed[arm],
                f"{arm} estimate did not recover after secondary visibility returned")

    blind_frame_indices = sorted(index for index in samples_by_frame
                                 if index % PHASE_PERIOD in BLIND_PHASES)
    blind_runs: list[list[int]] = []
    for index in blind_frame_indices:
        if not blind_runs or index != blind_runs[-1][-1] + 1:
            blind_runs.append([index])
        else:
            blind_runs[-1].append(index)
    blind_runs = [run for run in blind_runs
                  if len(run) >= 8 and run[0] % PHASE_PERIOD == 0
                  and run[0] - 1 in samples_by_frame]
    require(bool(blind_runs),
            "tracking output did not contain a complete contiguous primary-blind run "
            "with its preceding visible estimate")
    blind_run_metrics = []
    for run in blind_runs:
        hold_index = run[0] - 1
        require(frame_counts["primary_by_index"].get(hold_index, {}).get("image_blank") is False,
                "primary-blind run lacks a visible primary frame immediately before blanking")
        run_result = {"first_frame_index": run[0], "last_frame_index": run[-1],
                      "sample_count": len(run), "hold_last_frame_index": hold_index,
                      "arms": {}}
        for arm in estimates_by_frame:
            observed = [estimates_by_frame[arm][index] for index in run]
            truth = [reference_angles(index)[arm] for index in run]
            hold = estimates_by_frame[arm][hold_index]
            run_rmse = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(observed, truth))
                                 / len(run))
            hold_rmse = math.sqrt(math.fsum((hold - value) ** 2 for value in truth) / len(run))
            run_motion = max(observed) - min(observed)
            truth_motion = max(truth) - min(truth)
            require(truth_motion >= 0.03
                    and run_motion >= max(0.015, 0.35 * truth_motion),
                    f"{arm} did not follow within-run motion during primary sensor blanking")
            require(hold_rmse >= 0.015 and run_rmse <= 0.85 * hold_rmse,
                    f"{arm} primary-blind estimate did not beat the hold-last estimate "
                    f"(RMSE {run_rmse:.5f} vs {hold_rmse:.5f})")
            run_result["arms"][arm] = {
                "joint_rmse_rad": run_rmse,
                "hold_last_rmse_rad": hold_rmse,
                "joint_motion_rad": run_motion,
                "reference_motion_rad": truth_motion,
            }
        blind_run_metrics.append(run_result)

    return {
        "mode": "SOFTWARE_FIXTURE",
        "hardware": "NOT RUN",
        "profile": profile,
        "algorithm": expected_algorithm,
        "modalities": expected_modalities,
        "provider": "m3t-so101",
        "count": len(samples),
        "secondary_provenance_samples": provenance_samples,
        "primary_blind_samples": len(blind_indices),
        "secondary_blank_samples": len(secondary_blank_indices),
        "joint_rmse_rad": rmse,
        "frozen_seed_rmse_rad": zero_seed,
        "primary_blind_joint_rmse_rad": blind_rmse,
        "primary_blind_frozen_seed_rmse_rad": blind_zero_seed,
        "primary_blind_joint_motion_rad": blind_motion,
        "primary_blind_reference_motion_rad": blind_reference_motion,
        "primary_blind_contiguous_runs": blind_run_metrics,
        "post_visibility_recovery_rmse_rad": recovery_rmse,
    }


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--composer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for fixture files, logs, and result.json")
    parser.add_argument("--profile", choices=("baseline", "depth-only", "extended"),
                        default="baseline")
    args = parser.parse_args()
    build, composer, output = args.build_dir.resolve(), args.composer.resolve(), args.output_dir.resolve()
    if not build.is_dir() or not composer.is_file():
        parser.error("--build-dir and --composer must identify existing paths")
    fixture_binary = build / "rbt_m3t_rgbd_fixture"
    tracker_binary = build / "rbt-m3t"
    consumer_binary = build / "rbt_m3t_tracking_output_consumer_test"
    missing = [str(path) for path in (fixture_binary, tracker_binary, consumer_binary)
               if not path.is_file()]
    if missing:
        parser.error("missing build outputs: " + ", ".join(missing))
    output.mkdir(parents=True, exist_ok=False)

    result = {"status": "FAIL", "mode": "SOFTWARE_FIXTURE", "hardware": "NOT RUN",
              "profile": args.profile, "checks": [], "children_reaped": False,
              "forced_kill": []}
    processes: list[subprocess.Popen] = []
    handles = []
    deadline = started + TIMEOUT_SECONDS
    env = software.runtime_environment("software")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env["RBT_COMPOSER_ADDR"] = f"127.0.0.1:{port}"
    try:
        config_path, config, transform = create_config(output / "fixture", args.profile)
        result["checks"].append({"name": "nonidentity_dual_camera_fixture_config", "status": "PASS"})
        software.start("composer", [str(composer)], env, output, processes, handles)

        def composer_ready() -> bool:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return True
            except OSError:
                return False

        software.wait_until("waiting for Composer", min(deadline, time.monotonic() + 10.0),
                            composer_ready, processes)
        software.start("primary_fixture", [str(fixture_binary), "--config", str(config_path)],
                       env, output, processes, handles)
        software.start("secondary_fixture",
                       [str(fixture_binary), "--config", str(config_path), "--secondary-camera"],
                       env, output, processes, handles)
        primary_log, secondary_log = output / "primary_fixture.log", output / "secondary_fixture.log"

        def fixtures_ready() -> bool:
            primary_ready = any(event.get("event") == "fixture_ready"
                                and event.get("camera_role") == "primary"
                                and event.get("provider") == PRIMARY_PROVIDER
                                and event.get("serial") == "SOFTWARE_FIXTURE"
                                and event.get("frame_id") == "fixture/color_optical"
                                for event in software.json_events(primary_log))
            secondary_ready = any(event.get("event") == "fixture_ready"
                                  and event.get("camera_role") == "secondary"
                                  and event.get("provider") == SECONDARY_PROVIDER
                                  and event.get("serial") == SECONDARY_SERIAL
                                  and event.get("frame_id") == SECONDARY_FRAME
                                  for event in software.json_events(secondary_log))
            return primary_ready and secondary_ready

        software.wait_until("waiting for both native fixture providers",
                            min(deadline, time.monotonic() + 25.0), fixtures_ready, processes)
        result["checks"].append({"name": "independent_native_camera_providers_ready", "status": "PASS"})
        consumer = software.start("tracking_output_consumer",
            [str(consumer_binary), "--count", str(SAMPLE_COUNT), "--timeout-sec", "80",
             "--provider", "m3t-so101"], env, output, processes, handles)
        time.sleep(0.2)
        if consumer.poll() is not None:
            raise RuntimeError("tracking output consumer exited before tracker startup")
        result["checks"].append({"name": "native_output_consumer_started", "status": "PASS"})
        tracker = software.start("tracker",
            [str(tracker_binary), "run", "--config", str(config_path),
             "--provider", PRIMARY_PROVIDER, "--count", str(SAMPLE_COUNT),
             "--duration-sec", "85"], env, output, processes, handles)
        consumer_log = output / "tracking_output_consumer.log"
        software.wait_until("waiting for dual-camera tracking output",
            deadline,
            lambda: any(event.get("event") == "tracking_samples_consumed"
                        for event in software.json_events(consumer_log)),
            [tracker, consumer])

        primary_events = software.json_events(primary_log)
        secondary_events = software.json_events(secondary_log)
        counts = verify_fixture_frames(primary_events, secondary_events)
        primary_activations = [event for event in primary_events
                               if event.get("event") == "activation"]
        secondary_activations = [event for event in secondary_events
                                 if event.get("event") == "activation"]
        require(len(primary_activations) == 1 and len(secondary_activations) == 1,
                "fixture activation evidence must contain one independent process per camera")
        require(primary_activations[0].get("component_id") == PRIMARY_PROVIDER
                and secondary_activations[0].get("component_id") == SECONDARY_PROVIDER,
                "camera streams were not published by their configured independent providers")
        counts["primary_by_index"] = {
            frame["frame_index"]: frame for frame in primary_events
            if frame.get("event") == "fixture_frame"}
        counts["secondary_by_index"] = {
            frame["frame_index"]: frame for frame in secondary_events
            if frame.get("event") == "fixture_frame"}
        result["camera_frame_counts"] = {key: value for key, value in counts.items()
                                         if not key.endswith("_by_index")}
        result["checks"].append({"name": "paired_capture_identity_and_visibility_windows", "status": "PASS"})

        consumer_events = software.json_events(consumer_log)
        summaries = [event for event in consumer_events
                     if event.get("event") == "tracking_samples_consumed"]
        tracking = verify_tracking(consumer_events, summaries[-1], args.profile,
                                   transform, counts)
        result["tracking"] = tracking
        result["checks"].append({"name": "secondary_provenance_and_capture_skew", "status": "PASS"})
        result["checks"].append({"name": "joint_solver_tracks_motion_during_primary_blind_window", "status": "PASS"})
        result["checks"].append({"name": "secondary_blank_interval_recovery", "status": "PASS"})
        result["status"] = "PASS"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        software.stop_children(processes, result["forced_kill"])
        for handle in handles:
            handle.close()
        result["children_reaped"] = all(process.poll() is not None for process in processes)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
