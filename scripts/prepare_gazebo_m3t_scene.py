#!/usr/bin/env python3
"""Generate a one-arm Gazebo RGB-D scene and native M3T initialization."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


WORLD = "gazebo_m3t"
ARM = "so101_left"
SERIAL = "GAZEBO_RGBD"
CAMERA_FRAME = "gazebo/color_optical"
CAMERA_PROVIDER = "gazebo-rgbd"
RGBD_TOPIC = "/gazebo_m3t/rgbd"
JOINT_STATE_TOPIC = "/gazebo_m3t/joint_states"
JOINT_COMMAND_TOPIC = "/gazebo_m3t/joint_position_cmd"
LINK_TRUTH_TOPIC = "/gazebo_m3t/link_truth"
PAN = "shoulder_pan"
INITIAL_JOINTS = {
    PAN: 0.0,
    "shoulder_lift": 0.02,
    "elbow_flex": 0.77,
    "wrist_flex": 0.89,
    "wrist_roll": -1.80,
    "gripper": 0.04,
}
POSITION_JOINTS = tuple(name for name in INITIAL_JOINTS if name != "gripper")
CANONICAL_URDF = (
    Path(__file__).resolve().parents[1]
    / "models/so101_lerobot_soft_fin/so101_lerobot_soft_fin.urdf"
)
LINK_COLORS = {
    "base_link": "0.20 0.44 0.67 1",
    "shoulder_link": "0.80 0.28 0.13 1",
    "upper_arm_link": "0.83 0.57 0.13 1",
    "lower_arm_link": "0.17 0.52 0.46 1",
    "wrist_link": "0.65 0.33 0.58 1",
    "gripper_link": "0.87 0.73 0.25 1",
    "moving_jaw_so101_v1_link": "0.76 0.39 0.19 1",
    "fixed_soft_fin": "0.86 0.60 0.28 1",
    "moving_soft_fin": "0.78 0.46 0.24 1",
    "soft_fin_camera_mount_link": "0.35 0.60 0.75 1",
}


def element(parent: ET.Element, tag: str, value=None, **attributes: str) -> ET.Element:
    node = ET.SubElement(parent, tag, attributes)
    if value is not None:
        node.text = str(value)
    return node


def values(numbers) -> str:
    return " ".join(f"{number:.12g}" for number in numbers)


def pose_text(position, rotation: np.ndarray) -> str:
    # SDF and URDF both encode fixed-axis roll/pitch/yaw as Rz @ Ry @ Rx.
    xy = math.hypot(rotation[0, 0], rotation[1, 0])
    pitch = math.atan2(-rotation[2, 0], xy)
    if xy < 1e-12:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    else:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return values((*position, roll, pitch, yaw))


def add_plugin(parent: ET.Element, filename: str, name: str, **parameters) -> ET.Element:
    plugin = element(parent, "plugin", filename=f"gz-sim-{filename}-system", name=f"gz::sim::systems::{name}")
    for key, value in parameters.items():
        element(plugin, key, value)
    return plugin


def add_material(visual: ET.Element, color: str) -> None:
    material = element(visual, "material")
    element(material, "ambient", color)
    element(material, "diffuse", color)


def add_inertia(link: ET.Element, urdf_link: ET.Element) -> None:
    urdf_inertial = urdf_link.find("inertial")
    inertial = element(link, "inertial")
    if urdf_inertial is None:
        # The visual accessory and tool-reference links have no source inertia.
        mass = "0.001"
        tensor = {"ixx": "1e-6", "iyy": "1e-6", "izz": "1e-6"}
    else:
        origin = urdf_inertial.find("origin")
        if origin is not None:
            element(inertial, "pose", f"{origin.get('xyz', '0 0 0')} {origin.get('rpy', '0 0 0')}")
        mass = urdf_inertial.find("mass").attrib["value"]
        tensor = urdf_inertial.find("inertia").attrib
    element(inertial, "mass", mass)
    inertia = element(inertial, "inertia")
    for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        element(inertia, key, tensor.get(key, "0"))


def add_robot(world: ET.Element, model: dict, model_path: Path, position_ik: bool) -> None:
    if (model["root_link"] != "base_link" or model["units"] != "metres_radians"
            or model["source_sha256"] != hashlib.sha256(CANONICAL_URDF.read_bytes()).hexdigest()):
        raise ValueError("--robot-model must be generated from the canonical elbow100 SO101 URDF")
    joints = model["joints"]
    driven_joints = POSITION_JOINTS if position_ik else (PAN,)
    if {joint["name"] for joint in joints if joint["type"] == "revolute"} != INITIAL_JOINTS.keys():
        raise ValueError("--robot-model must contain the six named SO101 joints")
    urdf_links = {link.attrib["name"]: link for link in ET.parse(CANONICAL_URDF).getroot().findall("link")}
    robot = element(world, "model", name=ARM, canonical_link="base_link")
    element(robot, "self_collide", "false")
    by_child = {joint["child"]: joint for joint in joints}
    for record in model["links"]:
        name = record["name"]
        link = element(robot, "link", name=name)
        joint = by_child.get(name)
        if joint is not None:
            origin = np.array(joint["origin"]).reshape(4, 4)
            angle = INITIAL_JOINTS.get(joint["name"], 0.0)
            if not joint["lower"] <= angle <= joint["upper"]:
                raise ValueError(f"initial angle is outside {joint['name']} limits")
            if joint["type"] == "revolute" and joint["axis"] != [0, 0, 1]:
                raise ValueError(f"unexpected canonical axis for {joint['name']}")
            sine, cosine = math.sin(angle), math.cos(angle)
            rotation = origin[:3, :3] @ np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
            element(link, "pose", pose_text(origin[:3, 3], rotation), relative_to=joint["parent"])
        add_inertia(link, urdf_links[name])
        if record["mesh"]:
            mesh_path = (model_path.parent / record["mesh"]).resolve(strict=True)
            visual = element(link, "visual", name=f"{name}_mesh")
            mesh = element(element(visual, "geometry"), "mesh")
            element(mesh, "uri", mesh_path.as_uri())
            add_material(visual, LINK_COLORS[name])

    mount = element(robot, "joint", name="world_mount", type="fixed")
    element(mount, "parent", "world")
    element(mount, "child", "base_link")
    for record in joints:
        name = record["name"]
        driven = name in driven_joints
        joint = element(robot, "joint", name=name, type="revolute" if driven else "fixed")
        element(joint, "parent", record["parent"])
        element(joint, "child", record["child"])
        if driven:
            # The child link is already at the seed. Its Gazebo joint zero is
            # relative to that seed and the same offset translates both limits.
            offset = INITIAL_JOINTS[name]
            axis = element(joint, "axis")
            element(axis, "xyz", values(record["axis"]))
            limit = element(axis, "limit")
            element(limit, "lower", record["lower"] - offset)
            element(limit, "upper", record["upper"] - offset)
            element(limit, "effort", 10)
            element(limit, "velocity", 2)
    if position_ik:
        for index, name in enumerate(driven_joints):
            add_plugin(robot, "joint-position-controller", "JointPositionController", joint_name=name,
                       topic=JOINT_COMMAND_TOPIC, use_actuator_msg="true", actuator_number=index,
                       initial_position=0, use_velocity_commands="true", cmd_min=-2, cmd_max=2)
    else:
        add_plugin(robot, "joint-position-controller", "JointPositionController", joint_name=PAN,
                   topic=f"/{WORLD}/{PAN}/cmd", initial_position=0, p_gain=4, i_gain=0, d_gain=0.25,
                   cmd_min=-2.5, cmd_max=2.5)
    state = add_plugin(robot, "joint-state-publisher", "JointStatePublisher", topic=JOINT_STATE_TOPIC)
    for name in driven_joints:
        element(state, "joint_name", name)
    if position_ik:
        element(state, "update_rate", 60)
    poses = add_plugin(robot, "pose-publisher", "PosePublisher", publish_link_pose="true",
                       use_pose_vector_msg="true", update_frequency=30)
    if position_ik:
        element(poses, "topic", LINK_TRUTH_TOPIC)


def add_camera(world: ET.Element) -> np.ndarray:
    position = np.array([0.45, -0.42, 0.30])
    forward = np.array([0.10, 0.0, 0.15]) - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0.0, 0.0, 1.0], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    world_from_gazebo_camera = np.column_stack((forward, left, up))

    model = element(world, "model", name="rgbd_station")
    element(model, "static", "true")
    element(model, "pose", pose_text(position, world_from_gazebo_camera))
    sensor = element(element(model, "link", name="camera_link"), "sensor", name="rgbd", type="rgbd_camera")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", 15)
    element(sensor, "topic", RGBD_TOPIC)
    camera = element(sensor, "camera")
    element(camera, "horizontal_fov", 0.90)
    image = element(camera, "image")
    element(image, "width", 320)
    element(image, "height", 240)
    element(image, "format", "R8G8B8")
    clip = element(camera, "clip")
    element(clip, "near", 0.10)
    element(clip, "far", 3.0)
    element(camera, "optical_frame_id", CAMERA_FRAME)

    # Gazebo is x-forward/y-left/z-up and M3T optical is x-right/y-down/z-forward.
    gazebo_from_optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    optical_from_world_rotation = (world_from_gazebo_camera @ gazebo_from_optical).T
    optical_from_base = np.eye(4)
    optical_from_base[:3, :3] = optical_from_world_rotation
    optical_from_base[:3, 3] = -optical_from_world_rotation @ position
    return optical_from_base


def add_background(world: ET.Element) -> None:
    scene = element(world, "scene")
    element(scene, "ambient", "0.65 0.65 0.65 1")
    element(scene, "background", "0.15 0.19 0.24 1")
    for name, direction, color in (
            ("key", "-0.4 0.6 -1", "0.9 0.86 0.8 1"),
            ("fill", "0.5 -0.7 -0.5", "0.35 0.45 0.55 1")):
        light = element(world, "light", name=name, type="directional")
        element(light, "cast_shadows", "false")
        element(light, "diffuse", color)
        element(light, "specular", "0.16 0.16 0.16 1")
        element(light, "direction", direction)
    floor = element(world, "model", name="backdrop")
    element(floor, "static", "true")
    element(floor, "pose", "0 0 -0.035 0 0 0")
    visual = element(element(floor, "link", name="floor"), "visual", name="background")
    box = element(element(visual, "geometry"), "box")
    element(box, "size", "1.8 1.8 0.02")
    add_material(visual, "0.24 0.29 0.35 1")


def prepare(robot_path: Path, output: Path, position_ik: bool = False) -> None:
    robot_path = robot_path.resolve(strict=True)
    model = json.loads(robot_path.read_text(encoding="utf-8"))
    sdf = ET.Element("sdf", version="1.12")
    world = element(sdf, "world", name=WORLD)
    element(world, "gravity", "0 0 0")
    physics = element(world, "physics", name="1ms", type="ignored")
    element(physics, "max_step_size", 0.001)
    element(physics, "real_time_factor", 1.0)
    add_plugin(world, "physics", "Physics")
    add_plugin(world, "user-commands", "UserCommands")
    add_plugin(world, "scene-broadcaster", "SceneBroadcaster")
    add_plugin(world, "sensors", "Sensors", render_engine="ogre2")
    add_background(world)
    add_robot(world, model, robot_path, position_ik)
    optical_from_base = add_camera(world)

    initialization = {
        "schema_version": 1,
        "coordinate_frame": CAMERA_FRAME,
        "serial": SERIAL,
        "arms": [{"name": ARM, "base_in_camera": optical_from_base.ravel().tolist(),
                  "joint_angles": INITIAL_JOINTS, "fixed_base": True}],
        "provenance": {"mode": "Gazebo", "source": "world.sdf", "driven_joint": PAN},
    }
    if position_ik:
        initialization["provenance"] = {
            "mode": "Gazebo", "source": "world.sdf", "driven_joints": POSITION_JOINTS,
            "joint_coordinates": "canonical radians at Gazebo joint zero",
        }
    tracking = {
        "schema_version": 1,
        "robot_model": os.path.relpath(robot_path, output.resolve()),
        "initialization": "initialization.json",
        "cache_directory": "cache",
        "model_divides": 2,
        "model_points": 200,
        "model_image_size": 512,
        "correspondence_iterations": 4,
        "update_iterations": 2,
    }
    output.mkdir(parents=True, exist_ok=False)
    ET.indent(sdf, space="  ")
    ET.ElementTree(sdf).write(output / "world.sdf", encoding="utf-8", xml_declaration=True)
    for name, value in (("initialization.json", initialization), ("tracking.json", tracking)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    if position_ik:
        gripper = next(joint for joint in model["joints"] if joint["name"] == "gripper")
        metadata = {
            "schema_version": 1,
            "world": WORLD,
            "arm": ARM,
            "camera_provider": CAMERA_PROVIDER,
            "camera_frame": CAMERA_FRAME,
            "camera_serial": SERIAL,
            "rgbd_topic": RGBD_TOPIC,
            "optical_from_base": optical_from_base.ravel().tolist(),
            "joint_names": POSITION_JOINTS,
            "joint_offsets_rad": {name: INITIAL_JOINTS[name] for name in POSITION_JOINTS},
            "joint_coordinates": "q_canonical_rad = q_gazebo_rad + joint_offsets_rad[name]",
            "gripper_rad": INITIAL_JOINTS["gripper"],
            "gripper_lower_rad": gripper["lower"],
            "gripper_upper_rad": gripper["upper"],
            "joint_command_topic": JOINT_COMMAND_TOPIC,
            "joint_command_message": "gz.msgs.Actuators",
            "joint_state_topic": JOINT_STATE_TOPIC,
            "link_truth_topic": LINK_TRUTH_TOPIC,
            "link_truth_message": "gz.msgs.Pose_V",
            "link_truth_reference": ARM,
            "base_link": "base_link",
            "gripper_link": "gripper_link",
        }
        (output / "scene_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-model", type=Path, required=True, help="canonical M3T robot.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for the world and M3T config")
    parser.add_argument("--position-ik", action="store_true",
                        help="drive all five IK joints through one Gazebo command, with joint zero at the tracking seed")
    args = parser.parse_args()
    try:
        prepare(args.robot_model, args.output_dir, args.position_ik)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Scene generation failed: {error}\n")
    print(f"Generated {args.output_dir / 'world.sdf'} and native M3T config for {CAMERA_PROVIDER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
