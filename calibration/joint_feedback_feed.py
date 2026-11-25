#!/usr/bin/env python3
"""Publish real dual SO-101 telemetry without opening a command subscriber.

This helper is intended for the M3T joint-prior input. It owns both serial
ports while running, reads complete snapshots through ``DriverService``, and
publishes the same state object it records in a freshness cache. It never calls
the driver's command handling or motor goal/configuration write methods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


CACHE_SCHEMA = "rbt-so101-readonly-feedback-v1"
STATE_PROTOCOL = "rbt.so101.dual.v1"
STATE_UNITS = "joint_deg_gripper_percent"
DEFAULT_CHANNEL_PREFIX = "rbt/so101/dual"
REGISTER_EXPECTATIONS = (
    ("Homing_Offset", "homing_offset"),
    ("Min_Position_Limit", "range_min"),
    ("Max_Position_Limit", "range_max"),
)


@dataclass(frozen=True)
class DriverApi:
    backend_type: type
    load_calibration: Callable[[str | Path], Any]
    motor_names: tuple[str, ...]
    service_type: type
    channels_type: type


def load_driver_api(driver_path: str | Path) -> DriverApi:
    """Load the installed driver from its import root, not a bundled copy."""
    import_root = Path(driver_path).expanduser().resolve()
    if not import_root.is_dir():
        raise ValueError(f"driver path is not a directory: {import_root}")
    path_text = str(import_root)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)

    from so101_rbt.backend import FeetechSTS3215Backend
    from so101_rbt.calibration import MOTOR_NAMES, load_arm_calibration
    from so101_rbt.service import DriverService
    from so101_rbt.transport import RbtChannels

    return DriverApi(
        backend_type=FeetechSTS3215Backend,
        load_calibration=load_arm_calibration,
        motor_names=tuple(MOTOR_NAMES),
        service_type=DriverService,
        channels_type=RbtChannels,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_existing_cache(path: Path) -> None:
    """Allow an absent cache or this tool's cache schema; preserve other files."""
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlink cache path: {path}")
    if not path.exists():
        return
    if not path.is_file():
        raise ValueError(f"cache output is not a regular file: {path}")
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"refusing to replace an unreadable or unknown cache file: {path}") from exc
    if not isinstance(current, dict) or current.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"refusing to replace a cache file with an unknown schema: {path}")


def prepare_cache_path(path: str | Path) -> Path:
    """Create the parent directory and confirm an existing file is ours."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    _validate_existing_cache(output)
    return output


def atomic_write_cache(
    path: str | Path,
    *,
    state: dict[str, Any],
    calibration_sha256: dict[str, str],
    raw: dict[str, dict[str, int]],
    written_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Atomically replace a known feedback cache with an actual published state."""
    output = prepare_cache_path(path)
    stamp = time.monotonic_ns() if written_monotonic_ns is None else written_monotonic_ns
    if type(stamp) is not int or stamp < 0:
        raise ValueError("written_monotonic_ns must be a nonnegative integer")
    envelope = {
        "schema": CACHE_SCHEMA,
        "state": state,
        "calibration_sha256": calibration_sha256,
        "raw": raw,
        "written_monotonic_ns": stamp,
    }
    payload = (json.dumps(envelope, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")

    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp",
                                           dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        # Recheck immediately before replacement in case another process
        # changed the destination while the sample was being encoded.
        _validate_existing_cache(output)
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return envelope


def _validate_ports(left_port: str, right_port: str) -> tuple[str, str]:
    if not left_port or not right_port:
        raise ValueError("both arm ports are required")
    left_real = os.path.realpath(left_port)
    right_real = os.path.realpath(right_port)
    if left_real == right_real:
        raise ValueError("left and right ports must identify distinct devices")
    return left_port, right_port


def verify_ports_free(left_port: str, right_port: str,
                      run: Callable[..., Any] = subprocess.run) -> None:
    """Require fuser's exact unused result for both devices before opening either."""
    left_port, right_port = _validate_ports(left_port, right_port)
    results: list[tuple[str, int | None, BaseException | None]] = []
    for port in (left_port, right_port):
        try:
            result = run(["fuser", port], capture_output=True, check=False)
            results.append((port, result.returncode, None))
        except BaseException as exc:
            results.append((port, None, exc))
    failures = [
        f"{port}: {error}" if error is not None else f"{port}: fuser returned {code}"
        for port, code, error in results if error is not None or code != 1
    ]
    if failures:
        raise RuntimeError("could not verify both SO-101 ports are unused (fuser must return 1): "
                           + "; ".join(failures))


def verify_hardware_calibration(backend: Any, calibration: Any,
                                motor_names: tuple[str, ...], arm: str) -> None:
    """Compare all six motors' persisted offset and position-limit registers."""
    for name in motor_names:
        motor = calibration.motors[name]
        for register, calibration_field in REGISTER_EXPECTATIONS:
            expected = getattr(motor, calibration_field)
            actual = backend._read_register(motor.motor_id, register)
            if actual != expected:
                raise RuntimeError(
                    f"{arm} hardware/calibration mismatch for {name}/{register}: "
                    f"register {actual}, calibration {expected}"
                )


def _wrap_read_state(backend: Any, calibration: Any,
                     raw_by_arm: dict[str, dict[str, int] | None], arm: str) -> None:
    """Make every service state read retain the exact raw values it normalized."""
    def read_state() -> tuple[float, ...]:
        raw = backend.read_raw()
        raw_by_arm[arm] = dict(raw)
        return calibration.normalize_raw(raw)

    backend.read_state = read_state


def _validate_snapshot(state: Any, controller_id: str) -> tuple[int, str]:
    if not isinstance(state, dict):
        raise ValueError("SO-101 driver returned a non-object snapshot")
    if state.get("protocol") != STATE_PROTOCOL or state.get("kind") != "state":
        raise ValueError("SO-101 snapshot protocol or kind is invalid")
    if state.get("units") != STATE_UNITS:
        raise ValueError("SO-101 snapshot units are invalid")
    if state.get("mode") != "readonly":
        raise ValueError("SO-101 driver did not return readonly mode")
    if state.get("controller_id") != controller_id:
        raise ValueError("SO-101 snapshot controller_id differs from the requested identity")
    sequence = state.get("sequence")
    if type(sequence) is not int or sequence < 1:
        raise ValueError("SO-101 snapshot sequence is invalid")
    session = state.get("driver_session")
    if not isinstance(session, str) or not session:
        raise ValueError("SO-101 snapshot driver_session is invalid")
    if state.get("capture_time_known") is not False:
        raise ValueError("SO-101 snapshot must not claim a known capture time")

    arms = state.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"left", "right"}:
        raise ValueError("SO-101 snapshot must contain both left and right arms")
    for arm in ("left", "right"):
        values = arms[arm]
        if (not isinstance(values, (list, tuple)) or len(values) != 6
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in values)):
            raise ValueError(f"SO-101 {arm} snapshot must contain six finite joint values")

    intervals = state.get("read_intervals")
    if not isinstance(intervals, dict) or set(intervals) != {"left", "right"}:
        raise ValueError("SO-101 snapshot must contain actual read intervals for both arms")
    for arm in ("left", "right"):
        interval = intervals[arm]
        if not isinstance(interval, dict):
            raise ValueError(f"SO-101 {arm} read interval is invalid")
        start_ns, end_ns = interval.get("start_ns"), interval.get("end_ns")
        if (type(start_ns) is not int or type(end_ns) is not int
                or start_ns < 0 or end_ns < start_ns):
            raise ValueError(f"SO-101 {arm} read interval nanoseconds are invalid")
    return sequence, session


def _validate_raw_snapshot(raw: dict[str, dict[str, int] | None],
                           motor_names: tuple[str, ...]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    expected_names = set(motor_names)
    for arm in ("left", "right"):
        arm_raw = raw.get(arm)
        if not isinstance(arm_raw, dict) or set(arm_raw) != expected_names:
            raise ValueError(f"SO-101 {arm} raw telemetry is missing or incomplete")
        if any(type(value) is not int for value in arm_raw.values()):
            raise ValueError(f"SO-101 {arm} raw telemetry must contain integer encoder readings")
        result[arm] = dict(arm_raw)
    return result


def _check_calibration_hashes(paths: dict[str, Path], expected: dict[str, str]) -> None:
    actual = {arm: sha256_file(path) for arm, path in paths.items()}
    if actual != expected:
        raise RuntimeError("an SO-101 calibration file changed while the readonly feed was running")


def run_feed(args: argparse.Namespace, *, stop_event: threading.Event | None = None,
             api: DriverApi | None = None,
             fuser_run: Callable[..., Any] = subprocess.run) -> int:
    """Own both real arms and publish/cache fresh read-only snapshots."""
    if not math.isfinite(args.poll_hz) or not 0 < args.poll_hz <= 200:
        raise ValueError("poll_hz must be finite and in (0, 200]")
    output = prepare_cache_path(args.output)
    port_left, port_right = _validate_ports(args.left_port, args.right_port)
    calibration_paths = {
        "left": Path(args.left_calibration).expanduser(),
        "right": Path(args.right_calibration).expanduser(),
    }
    calibration_hashes = {arm: sha256_file(path) for arm, path in calibration_paths.items()}
    driver = api or load_driver_api(args.driver_path)
    calibrations = {
        arm: driver.load_calibration(path)
        for arm, path in calibration_paths.items()
    }
    _check_calibration_hashes(calibration_paths, calibration_hashes)

    # No serial backend is constructed until both fuser checks have succeeded.
    verify_ports_free(port_left, port_right, run=fuser_run)

    channels = None
    left = None
    right = None
    service = None
    raw_by_arm: dict[str, dict[str, int] | None] = {"left": None, "right": None}
    stopped = stop_event or threading.Event()
    try:
        channels = driver.channels_type(
            publish_states=True,
            subscribe_commands=False,
            channel_prefix=args.channel_prefix,
        )
        left = driver.backend_type(
            port_left, calibrations["left"], allow_goal_writes=False,
            write_timeout_s=0.2, num_read_retries=2,
        )
        right = driver.backend_type(
            port_right, calibrations["right"], allow_goal_writes=False,
            write_timeout_s=0.2, num_read_retries=2,
        )
        verify_hardware_calibration(left, calibrations["left"], driver.motor_names, "left")
        verify_hardware_calibration(right, calibrations["right"], driver.motor_names, "right")
        _wrap_read_state(left, calibrations["left"], raw_by_arm, "left")
        _wrap_read_state(right, calibrations["right"], raw_by_arm, "right")
        service = driver.service_type(
            left,
            right,
            controller_id=args.controller_id,
            mode="readonly",
            poll_hz=args.poll_hz,
            channel_prefix=args.channel_prefix,
        )

        period = 1.0 / args.poll_hz
        while not stopped.is_set():
            cycle_started = time.monotonic()
            _check_calibration_hashes(calibration_paths, calibration_hashes)
            raw_by_arm["left"] = None
            raw_by_arm["right"] = None
            state = service.read_snapshot()
            sequence, _session = _validate_snapshot(state, args.controller_id)
            raw = _validate_raw_snapshot(raw_by_arm, driver.motor_names)
            _check_calibration_hashes(calibration_paths, calibration_hashes)
            if stopped.is_set():
                break
            channels.publish_state(state, sequence)
            _check_calibration_hashes(calibration_paths, calibration_hashes)
            if stopped.is_set():
                break
            atomic_write_cache(
                output,
                state=state,
                calibration_sha256=calibration_hashes,
                raw=raw,
            )
            remaining = period - (time.monotonic() - cycle_started)
            if remaining > 0:
                stopped.wait(remaining)
    finally:
        close_errors: list[BaseException] = []
        for resource in (channels, left, right):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as exc:  # keep closing the remaining owned resources
                close_errors.append(exc)
        if close_errors and not sys.exc_info()[0]:
            raise RuntimeError("failed to close readonly feed resources: "
                               + "; ".join(str(error) for error in close_errors))
    return 0


def _positive_poll_hz(value: str) -> float:
    try:
        hz = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(hz) or not 0 < hz <= 200:
        raise argparse.ArgumentTypeError("must be finite and in (0, 200]")
    return hz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-path", required=True,
                        help="import root containing the installed so101_rbt package")
    parser.add_argument("--left-port", required=True)
    parser.add_argument("--right-port", required=True)
    parser.add_argument("--left-calibration", required=True)
    parser.add_argument("--right-calibration", required=True)
    parser.add_argument("--output", required=True, help="atomic feedback cache JSON file")
    parser.add_argument("--channel-prefix", default=DEFAULT_CHANNEL_PREFIX)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--poll-hz", type=_positive_poll_hz, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stop_event = threading.Event()
    old_sigterm = signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_event.set())
    old_sigint = signal.signal(signal.SIGINT, lambda _signum, _frame: stop_event.set())
    try:
        return run_feed(args, stop_event=stop_event)
    except Exception as exc:
        print(f"joint feedback feed failed: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)


if __name__ == "__main__":
    raise SystemExit(main())
