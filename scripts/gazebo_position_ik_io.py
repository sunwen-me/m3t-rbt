#!/usr/bin/env python3
"""Exchange SO101 position or velocity commands and samples with Gazebo Transport."""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import sys
import threading


CLOCK_TOPIC = "/world/gazebo_m3t/clock"
JOINT_NAMES = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"}
STATE_INTERVAL_NS = 10_000_000
MAX_COMMAND_BYTES = 4096


def finite_number(value):
    if type(value) in (int, float):
        try:
            number = float(value)
            if math.isfinite(number):
                return number
        except OverflowError:
            pass
    raise ValueError("expected a finite number")


def topic_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"(/[A-Za-z0-9_]+)+", value):
        raise ValueError("expected an absolute Gazebo topic name")
    return value


@dataclass(frozen=True)
class Scene:
    arm: str
    joint_names: tuple[str, ...]
    offsets: tuple[float, ...]
    command_topic: str
    state_topic: str
    pose_topic: str
    command_mode: str = "position"

    @classmethod
    def load(cls, path, command_mode="position"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("scene metadata must be a JSON object")
        if command_mode not in ("position", "velocity"):
            raise ValueError("command_mode must be position or velocity")
        if metadata.get("joint_command_mode", command_mode) != command_mode:
            raise ValueError("requested command mode differs from scene metadata")
        arm = metadata.get("arm", "so101_left")
        if not isinstance(arm, str) or not re.fullmatch(r"[A-Za-z0-9_]+", arm):
            raise ValueError("scene arm must be a Gazebo model name")
        names = metadata["joint_names"]
        if (not isinstance(names, list) or len(names) != len(JOINT_NAMES)
                or not all(isinstance(name, str) for name in names) or set(names) != JOINT_NAMES):
            raise ValueError("joint_names must contain the five SO101 position IK joints")
        offsets = metadata["joint_offsets_rad"]
        if not isinstance(offsets, dict) or offsets.keys() != JOINT_NAMES:
            raise ValueError("joint_offsets_rad must contain an offset for each position IK joint")
        topics = tuple(topic_name(metadata[key]) for key in (
            "joint_command_topic", "joint_state_topic", "link_truth_topic",
        ))
        if len(set(topics + (CLOCK_TOPIC,))) != 4:
            raise ValueError("scene topics must be distinct")
        return cls(arm, tuple(names), tuple(finite_number(offsets[name]) for name in names),
                   *topics, command_mode)


def stamp_ns(stamp):
    if stamp.sec >= 0 and 0 <= stamp.nsec < 1_000_000_000:
        return stamp.sec * 1_000_000_000 + stamp.nsec
    return None


def header_stamp_ns(message):
    if message.HasField("header") and message.header.HasField("stamp"):
        return stamp_ns(message.header.stamp)
    return None


def matching_entries(entries, arm, wanted):
    matches = {}
    for entry in entries:
        scope, _, name = entry.name.rpartition("::")
        if name not in wanted or (scope and scope.split("::")[-1] != arm):
            continue
        if name in matches:
            return None
        matches[name] = entry
    return matches if len(matches) == len(wanted) else None


class Sidecar:
    def __init__(self, scene, publisher, actuators):
        self.scene = scene
        self.publisher = publisher
        self.actuators = actuators
        self.lock = threading.Lock()
        self.clock = None
        self.last_clock_event = None
        self.last_state = None
        self.last_pose = None
        self.closed = False
        self.output_error = None

    def emit(self, event):
        # The caller holds the lock, keeping callback and command lines ordered.
        try:
            print(json.dumps(event, allow_nan=False), flush=True)
        except OSError as error:
            self.output_error = error
            self.closed = True

    def on_clock(self, message):
        stamp = stamp_ns(message.sim) if message.HasField("sim") else None
        if stamp is not None:
            with self.lock:
                if self.closed:
                    return
                if self.clock is not None and stamp < self.clock and self.scene.command_mode == "velocity":
                    self.emit({"event": "clock_regression", "stamp_ns": stamp,
                               "previous_stamp_ns": self.clock})
                if self.clock is None or stamp > self.clock:
                    self.clock = stamp
                    if (self.scene.command_mode == "velocity" and
                            (self.last_clock_event is None or stamp - self.last_clock_event >= STATE_INTERVAL_NS)):
                        self.last_clock_event = stamp
                        self.emit({"event": "clock", "stamp_ns": stamp})

    def on_state(self, message):
        stamp = header_stamp_ns(message)
        if stamp is None or (message.name and message.name != self.scene.arm):
            return
        joints = matching_entries(message.joint, self.scene.arm, self.scene.joint_names)
        if joints is None or not all(joint.HasField("axis1") for joint in joints.values()):
            return
        axes = [joints[name].axis1 for name in self.scene.joint_names]
        q = [axis.position + offset for axis, offset in zip(axes, self.scene.offsets)]
        dq = [axis.velocity for axis in axes]
        if not all(math.isfinite(value) for value in q + dq):
            return
        with self.lock:
            if self.closed or (self.last_state is not None and stamp - self.last_state < STATE_INTERVAL_NS):
                return
            self.last_state = stamp
            self.emit({"event": "state", "stamp_ns": stamp, "q_rad": q, "dq_rad_s": dq})

    def on_pose(self, message):
        poses = matching_entries(message.pose, self.scene.arm, ("base_link", "gripper_link"))
        if poses is None:
            return
        base = poses["base_link"]
        gripper = poses["gripper_link"]
        stamp = header_stamp_ns(base)
        if stamp is None or stamp != header_stamp_ns(gripper):
            return
        if not (base.HasField("position") and base.HasField("orientation") and gripper.HasField("position")):
            return
        base_xyz = [base.position.x, base.position.y, base.position.z]
        orientation = base.orientation
        base_quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        gripper_xyz = [gripper.position.x, gripper.position.y, gripper.position.z]
        if not all(math.isfinite(value) for value in base_xyz + base_quat + gripper_xyz):
            return
        if abs(math.hypot(*base_quat) - 1.0) > 1e-3:
            return
        with self.lock:
            if self.closed or (self.last_pose is not None and stamp <= self.last_pose):
                return
            self.last_pose = stamp
            self.emit({"event": "pose", "stamp_ns": stamp, "base_xyz_m": base_xyz,
                       "base_quat_xyzw": base_quat, "gripper_xyz_m": gripper_xyz})

    def command(self, line):
        command = json.loads(line)
        field = "dq_rad_s" if self.scene.command_mode == "velocity" else "q_rad"
        if not isinstance(command, dict) or command.keys() != {"command_id", field}:
            raise ValueError(f"command must provide command_id and {field}")
        command_id = command["command_id"]
        q = command[field]
        if type(command_id) is not int or command_id < 0:
            raise ValueError("command_id must be a nonnegative integer")
        if not isinstance(q, list) or len(q) != len(self.scene.offsets):
            raise ValueError(f"{field} must contain five joint values in scene metadata order")
        canonical = [finite_number(value) for value in q]
        values = canonical if self.scene.command_mode == "velocity" else [
            value - offset for value, offset in zip(canonical, self.scene.offsets)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Gazebo joint commands must be finite")
        message = self.actuators(**{
            "velocity" if self.scene.command_mode == "velocity" else "position": values})
        with self.lock:
            if self.output_error is not None:
                raise self.output_error
            if self.clock is None or not self.publisher.has_connections():
                raise RuntimeError(f"cannot publish command {command_id} without a clock and command subscriber")
            if not self.publisher.publish(message):
                raise RuntimeError(f"Gazebo Transport rejected command {command_id}")
            self.emit({"event": "command", "command_id": command_id, "stamp_ns": self.clock,
                       "command_mode": self.scene.command_mode, field: canonical,
                       "actuator_values": values})


def read_commands(sidecar, stopping):
    # Read the descriptor directly so multiple lines already in a Python IO buffer cannot stall.
    fd = sys.stdin.fileno()
    pending = b""
    while not stopping():
        with sidecar.lock:
            if sidecar.output_error is not None:
                raise sidecar.output_error
        if not select.select([fd], [], [], 0.1)[0] or stopping():
            continue
        chunk = os.read(fd, MAX_COMMAND_BYTES)
        if not chunk:
            if pending:
                sidecar.command(pending)
            return
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        if len(pending) > MAX_COMMAND_BYTES:
            raise ValueError("command line is too long")
        for line in lines:
            if stopping():
                return
            if len(line) > MAX_COMMAND_BYTES:
                raise ValueError("command line is too long")
            sidecar.command(line)


def run(scene, stopping):
    from gz.msgs.actuators_pb2 import Actuators
    from gz.msgs.clock_pb2 import Clock
    from gz.msgs.model_pb2 import Model
    from gz.msgs.pose_v_pb2 import Pose_V
    from gz.transport import Node

    node = Node()
    publisher = node.advertise(scene.command_topic, Actuators)
    if not publisher.valid():
        raise RuntimeError(f"could not advertise Gazebo topic {scene.command_topic}")
    sidecar = Sidecar(scene, publisher, Actuators)
    subscriptions = (
        (Clock, CLOCK_TOPIC, sidecar.on_clock),
        (Model, scene.state_topic, sidecar.on_state),
        (Pose_V, scene.pose_topic, sidecar.on_pose),
    )
    subscribed = []
    try:
        for message, topic, callback in subscriptions:
            if not node.subscribe(message, topic, callback):
                raise RuntimeError(f"could not subscribe to Gazebo topic {topic}")
            subscribed.append(topic)
        read_commands(sidecar, stopping)
        with sidecar.lock:
            if sidecar.output_error is not None:
                raise sidecar.output_error
    finally:
        with sidecar.lock:
            sidecar.closed = True
        for topic in reversed(subscribed):
            node.unsubscribe(topic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--command-mode", choices=("position", "velocity"), default="position")
    args = parser.parse_args()
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        run(Scene.load(args.scene_metadata, args.command_mode), lambda: stopped)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"Gazebo position IK IO: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
