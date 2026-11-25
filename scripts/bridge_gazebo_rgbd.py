#!/usr/bin/env python3
"""Forward an aligned Gazebo RGB-D stream to the native RBT provider."""

from __future__ import annotations

import argparse
from array import array
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gz.msgs.camera_info_pb2 import CameraInfo
    from gz.msgs.image_pb2 import Image
    from gz.msgs.model_pb2 import Model


MAX_PENDING = 8
MAX_PENDING_JOINT_STATES = 256
MAX_JOINT_STATE_AGE_NS = 40_000_000
POLL_SECONDS = 0.1
NS_PER_SECOND = 1_000_000_000
CAPTURE_TIME = struct.Struct("<q")
JOINT_STATE_RECORD = struct.Struct("<q6d")
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Compare with the nearest representable sensor value at the U16 limit.
MAX_DEPTH_M = struct.unpack("<f", struct.pack("<f", 65_535 / 1000))[0]


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_message(cls, info: CameraInfo) -> Calibration:
        k = info.intrinsics.k
        if info.width <= 0 or info.height <= 0 or len(k) != 9:
            raise ValueError("CameraInfo requires positive dimensions and a 3x3 intrinsics.k")
        if (any(k[index] != 0 for index in (1, 3, 6, 7)) or k[8] != 1
                or any(coefficient != 0 for coefficient in info.distortion.k)):
            raise ValueError("Gazebo bridge requires an undistorted camera with zero skew")
        intrinsics = (k[0], k[4], k[2], k[5])
        fx, fy, cx, cy = intrinsics
        if (not 0 < fx < math.inf or not 0 < fy < math.inf
                or not 0 <= cx < info.width or not 0 <= cy < info.height):
            raise ValueError("CameraInfo requires positive finite focal lengths and a principal point in the image")
        return cls(info.width, info.height, *intrinsics)


def capture_time_ns(message: Image | Model, label: str = "image") -> int:
    if not message.HasField("header") or not message.header.HasField("stamp"):
        raise ValueError(f"Gazebo {label} is missing its capture timestamp")
    sec, nsec = message.header.stamp.sec, message.header.stamp.nsec
    stamp = sec * NS_PER_SECOND + nsec
    if sec < 0 or not 0 <= nsec < NS_PER_SECOND or stamp >= 1 << 63:
        raise ValueError(f"Gazebo {label} has an invalid capture timestamp")
    return stamp


class PairedFrames:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.calibration: Calibration | None = None
        self.color: dict[int, Image] = {}
        self.depth: dict[int, Image] = {}
        self.last_stamp = 0
        self.dropped = 0
        self.error: ValueError | None = None

    def add_calibration(self, info: CameraInfo) -> None:
        try:
            calibration = Calibration.from_message(info)
        except ValueError as error:
            self.record_error(error)
            return
        with self.condition:
            if self.calibration is None:
                self.calibration = calibration
                self.condition.notify()
            elif self.calibration != calibration:
                self.record_error(ValueError("Gazebo camera calibration changed"))

    def add_image(self, image: Image, *, depth: bool = False) -> None:
        try:
            stamp = capture_time_ns(image)
        except ValueError as error:
            self.record_error(error)
            return
        with self.condition:
            pending = self.depth if depth else self.color
            if stamp <= self.last_stamp or stamp in pending:
                self.dropped += 1
                return
            pending[stamp] = image
            if len(pending) > MAX_PENDING:
                del pending[min(pending)]
                self.dropped += 1
            self.condition.notify()

    def record_error(self, error: ValueError) -> None:
        with self.condition:
            if self.error is None:
                self.error = error
            self.condition.notify()

    def take(self, timeout: float) -> tuple[Calibration, int, Image, Image] | None:
        with self.condition:
            self.condition.wait_for(
                lambda: self.error is not None
                or (self.calibration is not None and not self.color.keys().isdisjoint(self.depth)),
                timeout,
            )
            if self.error is not None:
                raise self.error
            if self.calibration is None:
                return None
            common = self.color.keys() & self.depth.keys()
            if not common:
                return None
            stamp = min(common)
            color, depth = self.color.pop(stamp), self.depth.pop(stamp)
            self.last_stamp = stamp
            for pending in (self.color, self.depth):
                stale = [key for key in pending if key < stamp]
                for key in stale:
                    del pending[key]
                self.dropped += len(stale)
            return self.calibration, stamp, color, depth


def metadata_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not -sys.float_info.max <= value <= sys.float_info.max:
        raise ValueError(f"scene metadata {label} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class JointCalibration:
    topic: str
    offsets_rad: dict[str, float]
    gripper_percent: float

    @classmethod
    def load(cls, path: Path) -> JointCalibration:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("scene metadata must be a JSON object")
        names = metadata.get("joint_names")
        if (not isinstance(names, list) or len(names) != len(ARM_JOINTS)
                or any(joint not in names for joint in ARM_JOINTS)):
            raise ValueError("scene metadata joint_names must contain all five SO101 arm joints")
        source_offsets = metadata.get("joint_offsets_rad")
        if not isinstance(source_offsets, dict):
            raise ValueError("scene metadata joint_offsets_rad must map names to radians")
        offsets = {name: metadata_number(source_offsets.get(name), f"offset for {name}")
                   for name in ARM_JOINTS}
        topic = metadata.get("joint_state_topic")
        if not isinstance(topic, str) or not topic.startswith("/") or not topic.rstrip("/"):
            raise ValueError("scene metadata joint_state_topic must be a non-root absolute Gazebo topic")
        jaw = metadata_number(metadata.get("gripper_rad"), "gripper_rad")
        lower = metadata_number(metadata.get("gripper_lower_rad"), "gripper_lower_rad")
        upper = metadata_number(metadata.get("gripper_upper_rad"), "gripper_upper_rad")
        span = upper - lower
        if not lower <= jaw <= upper or not 0 < span < math.inf:
            raise ValueError("scene metadata gripper angle must lie within positive-span limits")
        return cls(topic, offsets, (jaw - lower) / span * 100)


class CausalJointStates:
    def __init__(self, calibration: JointCalibration) -> None:
        self.calibration = calibration
        self.lock = threading.Lock()
        self.samples: deque[tuple[int, tuple[float, ...]]] = deque(maxlen=MAX_PENDING_JOINT_STATES)
        self.last_forwarded_stamp = 0
        self.rejected = 0

    def sample(self, model: Model) -> tuple[int, tuple[float, ...]]:
        stamp = capture_time_ns(model, "joint state")
        angles = {}
        for joint in model.joint:
            offset = self.calibration.offsets_rad.get(joint.name)
            if offset is None:
                continue
            if joint.name in angles or not joint.HasField("axis1"):
                raise ValueError("expected distinct joint states with axis1")
            angle = joint.axis1.position + offset
            if not math.isfinite(angle):
                raise ValueError("expected finite Gazebo joint positions")
            angles[joint.name] = angle
        if stamp <= 0 or len(angles) != len(ARM_JOINTS):
            raise ValueError("expected a positive timestamp and complete Gazebo arm state")
        return stamp, (*(angles[name] for name in ARM_JOINTS), self.calibration.gripper_percent)

    def add(self, model: Model) -> None:
        try:
            sample = self.sample(model)
        except ValueError:
            with self.lock:
                self.rejected += 1
            return
        with self.lock:
            if self.samples and sample[0] <= self.samples[-1][0]:
                self.rejected += 1
                return
            self.samples.append(sample)

    def packet_for(self, capture_stamp: int) -> bytes | None:
        with self.lock:
            for stamp, values in reversed(self.samples):
                if stamp > capture_stamp:
                    continue
                if stamp <= self.last_forwarded_stamp or capture_stamp - stamp > MAX_JOINT_STATE_AGE_NS:
                    break
                self.last_forwarded_stamp = stamp
                return JOINT_STATE_RECORD.pack(stamp, *values)
        return None


def packed_image(
    image: Image, calibration: Calibration, pixel_format: int, bytes_per_pixel: int, label: str
) -> bytes:
    if (image.width, image.height) != (calibration.width, calibration.height):
        raise ValueError(f"Gazebo {label} dimensions differ from CameraInfo")
    if image.pixel_format_type != pixel_format:
        raise ValueError(f"Gazebo {label} has unsupported pixel format {image.pixel_format_type}")
    row_bytes = image.width * bytes_per_pixel
    data = image.data
    if image.step < row_bytes or len(data) != image.step * image.height:
        raise ValueError(f"Gazebo {label} has invalid stride or byte count")
    if image.step == row_bytes:
        return data
    return b"".join(data[offset:offset + row_bytes] for offset in range(0, len(data), image.step))


def meters_to_u16_mm(data: bytes) -> bytes:
    if sys.byteorder == "little":
        meters = memoryview(data).cast("f")
    else:
        meters = array("f")
        meters.frombytes(data)
        meters.byteswap()
    millimeters = array("H", (
        int(value * 1000 + 0.5) if 0 < value <= MAX_DEPTH_M else 0 for value in meters
    ))
    if sys.byteorder != "little":
        millimeters.byteswap()
    return millimeters.tobytes()


def write_frame(
    fd: int, stamp: int, color: bytes, depth: bytes, stopping: Callable[[], bool],
    joint_packet: bytes = b"",
) -> bool:
    for part in (CAPTURE_TIME.pack(stamp), color, depth, joint_packet):
        remaining = memoryview(part)
        while remaining:
            if stopping():
                return False
            try:
                remaining = remaining[os.write(fd, remaining):]
            except BlockingIOError:
                select.select([], [fd], [], POLL_SECONDS)
    return True


def stop_provider(provider: subprocess.Popen) -> None:
    if provider.stdin is not None:
        provider.stdin.close()
    try:
        provider.wait(timeout=2)
    except subprocess.TimeoutExpired:
        provider.terminate()
        try:
            provider.wait(timeout=2)
        except subprocess.TimeoutExpired:
            provider.kill()
            provider.wait()


def bridge(args: argparse.Namespace, stopping: Callable[[], bool]) -> None:
    from gz.msgs.camera_info_pb2 import CameraInfo
    from gz.msgs.image_pb2 import Image, RGB_INT8, R_FLOAT32
    from gz.transport import Node

    base = args.topic.rstrip("/")
    frames = PairedFrames()
    node = Node()
    subscriptions = [
        (CameraInfo, f"{base}/camera_info", frames.add_calibration),
        (Image, f"{base}/image", frames.add_image),
        (Image, f"{base}/depth_image", lambda msg: frames.add_image(msg, depth=True)),
    ]
    joint_states = None
    skipped_for_joint_state = 0
    if args.scene_metadata is not None:
        from gz.msgs.model_pb2 import Model

        joint_states = CausalJointStates(JointCalibration.load(args.scene_metadata))
        subscriptions.append((Model, joint_states.calibration.topic, joint_states.add))
    subscribed = []
    provider = None
    forwarded = 0
    try:
        for message, topic, callback in subscriptions:
            if not node.subscribe(message, topic, callback):
                raise RuntimeError(f"could not subscribe to Gazebo topic {topic}")
            subscribed.append(topic)
        deadline = time.monotonic() + args.startup_timeout
        while not stopping():
            if provider is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    needs = f"CameraInfo and an exact RGB-D pair on {base}"
                    if joint_states is not None:
                        needs += f" with a fresh causal joint state on {joint_states.calibration.topic}"
                    raise RuntimeError(f"timed out waiting for {needs}")
                timeout = min(remaining, POLL_SECONDS)
            else:
                status = provider.poll()
                if status is not None:
                    raise RuntimeError(f"native RGB-D provider exited with status {status}")
                timeout = POLL_SECONDS
            sample = frames.take(timeout)
            if stopping():
                break
            if sample is None:
                continue
            calibration, stamp, color, depth = sample
            joint_packet = b""
            if joint_states is not None:
                joint_packet = joint_states.packet_for(stamp)
                if joint_packet is None:
                    skipped_for_joint_state += 1
                    continue
            rgb = packed_image(color, calibration, RGB_INT8, 3, "RGB image")
            meters = packed_image(depth, calibration, R_FLOAT32, 4, "depth image")
            if provider is None:
                command = [
                    str(args.provider_binary), str(calibration.width), str(calibration.height),
                    str(args.fps), str(calibration.fx), str(calibration.fy),
                    str(calibration.cx), str(calibration.cy),
                ]
                if joint_states is not None:
                    command.extend(("--joint-state-service", args.joint_state_service))
                provider = subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=0)
                fd = provider.stdin.fileno()
                os.set_blocking(fd, False)
                print(f"Gazebo RGB-D bridge {base} {calibration.width}x{calibration.height} "
                      f"at {args.fps} FPS", file=sys.stderr)
            if write_frame(fd, stamp, rgb, meters_to_u16_mm(meters), stopping, joint_packet):
                forwarded += 1
    finally:
        if provider is not None:
            stop_provider(provider)
        for topic in subscribed:
            node.unsubscribe(topic)
        summary = f"Gazebo RGB-D bridge forwarded={forwarded} dropped_images={frames.dropped}"
        if joint_states is not None:
            summary += (f" skipped_for_joint_state={skipped_for_joint_state}"
                        f" rejected_joint_states={joint_states.rejected}")
        print(summary, file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True, help="Gazebo RGB-D sensor base topic")
    parser.add_argument("--fps", required=True, type=int, help="sensor frame rate")
    parser.add_argument("--provider-binary", required=True, type=Path, help="native provider executable path")
    parser.add_argument("--joint-state-service", help="native service to publish simulated arm feedback")
    parser.add_argument("--scene-metadata", type=Path, help="scene JSON with joint-state topic and calibration")
    parser.add_argument("--startup-timeout", type=float, default=15.0,
                        help="seconds to wait for calibration and the first paired frame (default 15)")
    args = parser.parse_args()
    if not args.topic.startswith("/") or not args.topic.rstrip("/"):
        parser.error("--topic must be a non-root absolute Gazebo topic")
    if args.fps <= 0:
        parser.error("--fps must be a positive integer")
    if (args.scene_metadata is None) != (args.joint_state_service is None):
        parser.error("--scene-metadata and --joint-state-service must be supplied together")
    if args.joint_state_service is not None and not args.joint_state_service.strip():
        parser.error("--joint-state-service must be nonempty")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("--startup-timeout must be finite and positive")
    args.provider_binary = args.provider_binary.resolve()
    if not args.provider_binary.is_file() or not os.access(args.provider_binary, os.X_OK):
        parser.error("--provider-binary must refer to an executable file")

    stopped = False

    def stop(_signum, _frame) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        bridge(args, lambda: stopped)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Gazebo RGB-D bridge: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
