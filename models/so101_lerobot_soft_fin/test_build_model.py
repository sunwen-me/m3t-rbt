"""Contract tests for the offline SO-101 LeRobot soft-fin URDF builder."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock
from urllib.parse import urlparse

import numpy as np

import build_model as bm


MODEL_DIR = Path(__file__).resolve().parent
OFFICIAL_URDF = MODEL_DIR / "sources" / "so101_new_calib.urdf"
URDF_NAME = "so101_lerobot_soft_fin.urdf"
LEGACY_LINKS = {
    "base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
    "jaw", "gripper_camera_mount", "camera_module", "handeye_cam",
}
LEGACY_CAMERA_LINKS = {"gripper_camera_mount", "camera_module", "handeye_cam"}
ADDED_LINKS = {"fixed_soft_fin", "moving_soft_fin", "soft_fin_camera_mount_link"}


def _signature(element: ET.Element | None):
    """Return a structural XML value, independent of serializer formatting."""
    if element is None:
        return None
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_signature(child) for child in element),
    )


def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    xyz = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


def _joint_motion(joint: ET.Element, value: float) -> np.ndarray:
    transform = np.eye(4)
    axis_node = joint.find("axis")
    axis = np.array([0.0, 0.0, 1.0]) if axis_node is None else np.fromstring(axis_node.get("xyz", "0 0 1"), sep=" ")
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(value), np.sin(value)
    # Rodrigues' formula, implemented independently of the model builder.
    transform[:3, :3] = np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])
    return transform


def _forward_kinematics(root: ET.Element, joint_values: dict[str, float]) -> dict[str, np.ndarray]:
    """Evaluate a URDF tree using parent_T_child = origin * joint_motion."""
    links = [link.get("name") for link in root.findall("link")]
    joints = root.findall("joint")
    child_to_joint = {joint.find("child").get("link"): joint for joint in joints}
    roots = [name for name in links if name not in child_to_joint]
    if len(roots) != 1:
        raise AssertionError(f"expected one URDF root, found {roots}")
    by_parent: dict[str, list[ET.Element]] = {name: [] for name in links}
    for joint in joints:
        by_parent[joint.find("parent").get("link")].append(joint)

    transforms = {roots[0]: np.eye(4)}
    pending = [roots[0]]
    while pending:
        parent = pending.pop()
        for joint in by_parent[parent]:
            name = joint.get("name")
            value = joint_values.get(name, 0.0)
            child = joint.find("child").get("link")
            transforms[child] = (
                transforms[parent]
                @ _origin_transform(joint.find("origin"))
                @ (_joint_motion(joint, value) if joint.get("type") in {"revolute", "continuous"} else np.eye(4))
            )
            pending.append(child)
    return transforms


def _mesh_filename(element: ET.Element) -> str:
    mesh = element.find("geometry/mesh")
    if mesh is None:
        raise AssertionError("expected a mesh geometry")
    return mesh.get("filename", "")


def _pin_records(value) -> dict[str, str]:
    """Collect relative-path SHA-256 records from the input lock documents."""
    records: dict[str, str] = {}
    if isinstance(value, dict):
        digest = value.get("sha256")
        for key in ("local_path", "relative_path", "output_path", "runtime_path", "path"):
            candidate = value.get(key)
            if isinstance(candidate, str) and isinstance(digest, str):
                records[Path(candidate).as_posix().removeprefix("./")] = digest
        for child in value.values():
            records.update(_pin_records(child))
    elif isinstance(value, list):
        for child in value:
            records.update(_pin_records(child))
    return records


class BuildModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.official = ET.parse(OFFICIAL_URDF).getroot()
        cls.upstream_lock = json.loads((MODEL_DIR / "upstream.lock.json").read_text(encoding="utf-8"))

    def _build(self, output: Path) -> Path:
        result = bm.build(output)
        self.assertEqual(Path(result), output / URDF_NAME)
        self.assertTrue(Path(result).is_file())
        return Path(result)

    def test_upstream_source_and_assets_match_immutable_lock(self) -> None:
        source_record = self.upstream_lock["sources"]["urdf"]
        self.assertEqual(source_record["local_path"], "sources/so101_new_calib.urdf")
        self.assertEqual(hashlib.sha256(OFFICIAL_URDF.read_bytes()).hexdigest(), source_record["sha256"])
        self.assertEqual(OFFICIAL_URDF.stat().st_size, source_record["bytes"])
        self.assertEqual(len(source_record["sha256"]), 64)
        for asset in self.upstream_lock["sources"]["assets"]:
            path = MODEL_DIR / asset["local_path"]
            with self.subTest(asset=asset["local_path"]):
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, asset["bytes"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), asset["sha256"])

        soft_fin_path = MODEL_DIR / "soft_fin.lock.json"
        soft_fin_lock = json.loads(soft_fin_path.read_text(encoding="utf-8"))
        self.assertEqual(soft_fin_lock["official_lock"]["path"], "upstream.lock.json")
        self.assertEqual(
            hashlib.sha256((MODEL_DIR / soft_fin_lock["official_lock"]["path"]).read_bytes()).hexdigest(),
            soft_fin_lock["official_lock"]["sha256"],
        )
        for record in soft_fin_lock["inputs"]:
            path = MODEL_DIR / record["path"]
            with self.subTest(input=record["path"]):
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])

    def test_builder_preserves_official_joint_graph_and_original_link_geometry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="so101-lr-build-") as temporary:
            output = Path(temporary) / "model"
            urdf_path = self._build(output)
            result = ET.parse(urdf_path).getroot()
            provenance = json.loads((output / "model_provenance.json").read_text(encoding="utf-8"))

        original_joints = {joint.get("name"): joint for joint in self.official.findall("joint")}
        result_joints = {joint.get("name"): joint for joint in result.findall("joint")}
        self.assertEqual(len(original_joints), len(self.official.findall("joint")))
        self.assertEqual(
            [joint.get("name") for joint in result.findall("joint") if joint.get("name") in original_joints],
            [joint.get("name") for joint in self.official.findall("joint")],
        )
        expected_limits = {
            "shoulder_lift": (-np.deg2rad(110), np.deg2rad(110)),
            "elbow_flex": (-1.69, np.deg2rad(100)),
            "wrist_flex": (-np.deg2rad(110), np.deg2rad(110)),
            "wrist_roll": (-np.pi, np.pi),
        }
        self.assertEqual(set(bm.JOINT_LIMIT_OVERRIDES), set(expected_limits))
        for name, original in original_joints.items():
            with self.subTest(joint=name):
                self.assertIn(name, result_joints)
                expected = copy.deepcopy(original)
                if name in bm.JOINT_LIMIT_OVERRIDES:
                    expected_limit = expected.find("limit")
                    actual_limit = result_joints[name].find("limit")
                    self.assertIsNotNone(expected_limit)
                    self.assertIsNotNone(actual_limit)
                    lower, upper = bm.JOINT_LIMIT_OVERRIDES[name]
                    expected_limit.set("lower", f"{lower:.16g}")
                    expected_limit.set("upper", f"{upper:.16g}")
                    self.assertAlmostEqual(float(actual_limit.get("lower")), expected_limits[name][0], places=14)
                    self.assertAlmostEqual(float(actual_limit.get("upper")), expected_limits[name][1], places=14)
                self.assertEqual(_signature(expected), _signature(result_joints[name]))

        preservation = provenance["preservation_checks"]
        self.assertFalse(preservation["official_joints_semantically_unchanged"])
        self.assertTrue(preservation["official_joint_kinematics_unchanged"])
        changes = preservation["authorized_joint_limit_overrides"]
        self.assertEqual(set(changes), set(expected_limits))
        for name, record in changes.items():
            with self.subTest(provenance_joint=name):
                self.assertAlmostEqual(record["new_lower"], expected_limits[name][0], places=14)
                self.assertAlmostEqual(record["new_upper"], expected_limits[name][1], places=14)
                self.assertTrue(record["reason"])
        elbow_limit = result_joints["elbow_flex"].find("limit")
        self.assertGreaterEqual(np.deg2rad(97.758), float(elbow_limit.get("lower")))
        self.assertLessEqual(np.deg2rad(97.758), float(elbow_limit.get("upper")))
        self.assertEqual(changes["elbow_flex"]["old_lower"], -1.69)
        self.assertEqual(changes["elbow_flex"]["old_upper"], 1.69)
        wrist_limit = result_joints["wrist_roll"].find("limit")
        for angle_degrees in (-180.0, -168.66, 180.0):
            with self.subTest(wrist_roll_degrees=angle_degrees):
                angle = np.deg2rad(angle_degrees)
                self.assertGreaterEqual(angle, float(wrist_limit.get("lower")))
                self.assertLessEqual(angle, float(wrist_limit.get("upper")))
        source_wrist_limit = original_joints["wrist_roll"].find("limit")
        self.assertEqual(changes["wrist_roll"]["old_lower"], float(source_wrist_limit.get("lower")))
        self.assertEqual(changes["wrist_roll"]["old_upper"], float(source_wrist_limit.get("upper")))
        self.assertIn("v2 wrist full-turn compatibility", changes["wrist_roll"]["reason"])
        self.assertNotEqual(changes["wrist_roll"]["reason"], changes["wrist_flex"]["reason"])
        self.assertEqual(sum(joint.get("type") == "revolute" for joint in result.findall("joint")), 6)
        self.assertEqual(
            {joint.get("name") for joint in result.findall("joint") if joint.get("type") in {"revolute", "continuous"}},
            {joint.get("name") for joint in self.official.findall("joint") if joint.get("type") in {"revolute", "continuous"}},
        )

        original_links = {link.get("name"): link for link in self.official.findall("link")}
        result_links = {link.get("name"): link for link in result.findall("link")}
        self.assertTrue(set(original_links).issubset(result_links))
        self.assertEqual(set(result_links) - set(original_links), ADDED_LINKS)
        self.assertTrue(LEGACY_LINKS.isdisjoint(result_links))
        self.assertTrue(LEGACY_CAMERA_LINKS.isdisjoint(result_links))
        self.assertFalse({str(index) for index in range(1, 7)} & set(result_joints))

        substitutions = {
            ("gripper_link", "wrist_roll_follower_so101_v1.stl"): "assets/soft_fin/fixed_palm.stl",
            ("moving_jaw_so101_v1_link", "moving_jaw_so101_v1.stl"): "assets/soft_fin/moving_jaw_body.stl",
        }
        for name, original in original_links.items():
            actual = result_links[name]
            with self.subTest(link=name):
                self.assertEqual(_signature(original.find("inertial")), _signature(actual.find("inertial")))
                for role in ("visual", "collision"):
                    expected_elements = original.findall(role)
                    actual_elements = actual.findall(role)
                    self.assertEqual(len(expected_elements), len(actual_elements), role)
                    for expected, observed in zip(expected_elements, actual_elements):
                        expected = ET.fromstring(ET.tostring(expected, encoding="utf-8"))
                        old_filename = _mesh_filename(expected)
                        old_name = Path(old_filename).name
                        new_filename = substitutions.get((name, old_name), f"assets/official/{old_name}")
                        expected.find("geometry/mesh").set("filename", new_filename)
                        self.assertEqual(_signature(expected), _signature(observed), role)

    def test_independent_fk_matches_every_original_link_for_nonzero_joint_vectors(self) -> None:
        with tempfile.TemporaryDirectory(prefix="so101-lr-fk-") as temporary:
            result = ET.parse(self._build(Path(temporary) / "model")).getroot()
        active = [joint for joint in self.official.findall("joint") if joint.get("type") == "revolute"]
        vectors = [
            {joint.get("name"): value for joint, value in zip(active, values)}
            for values in (
                (0.23, -0.41, 0.37, -0.28, 0.19, 0.44),
                (-0.51, 0.32, -0.17, 0.63, -0.39, 0.21),
                (0.08, 0.72, -0.55, 0.14, 0.31, -0.62),
            )
        ]
        source_links = {link.get("name") for link in self.official.findall("link")}
        for values in vectors:
            with self.subTest(values=values):
                expected = _forward_kinematics(self.official, values)
                actual = _forward_kinematics(result, values)
                for name in source_links:
                    np.testing.assert_allclose(actual[name], expected[name], atol=1e-10, rtol=0, err_msg=name)

    def test_preservation_rejects_unauthorized_joint_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="so101-lr-preservation-") as temporary:
            result_path = self._build(Path(temporary) / "model")
            built = ET.parse(result_path).getroot()

        for joint_name, element, attribute, changed_value in (
            ("elbow_flex", "axis", "xyz", "0 1 0"),
            ("elbow_flex", "limit", "lower", "-2.0"),
        ):
            with self.subTest(joint=joint_name, element=element, attribute=attribute):
                corrupted = copy.deepcopy(built)
                joint = corrupted.find(f"joint[@name='{joint_name}']")
                self.assertIsNotNone(joint)
                target = joint.find(element)
                self.assertIsNotNone(target)
                target.set(attribute, changed_value)
                with self.assertRaisesRegex(ValueError, "Official joint XML changed"):
                    bm.verify_preserved(self.official, corrupted)

    def test_fin_and_camera_attachments_keep_expected_anchor_geometry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="so101-lr-attachments-") as temporary:
            output = Path(temporary) / "model"
            result = ET.parse(self._build(output)).getroot()
            parent_by_child = {
                joint.find("child").get("link"): joint.find("parent").get("link")
                for joint in result.findall("joint")
            }
            joint_by_child = {
                joint.find("child").get("link"): joint for joint in result.findall("joint")
            }
            expected_parents = {
                "fixed_soft_fin": "gripper_link",
                "moving_soft_fin": "moving_jaw_so101_v1_link",
                "soft_fin_camera_mount_link": "gripper_link",
            }
            for child, parent in expected_parents.items():
                with self.subTest(child=child):
                    self.assertEqual(parent_by_child.get(child), parent)
                    self.assertEqual(joint_by_child[child].get("type"), "fixed")

            runtime_meshes = {
                Path(mesh.get("filename", "")).name: mesh.get("filename", "")
                for mesh in result.findall(".//mesh")
            }
            self.assertIn("fixed_fin.stl", runtime_meshes)
            self.assertIn("moving_fin.stl", runtime_meshes)
            self.assertIn("bracket_34.stl", runtime_meshes)
            self.assertIn("assets/soft_fin/fixed_palm.stl", runtime_meshes.values())
            self.assertIn("assets/soft_fin/moving_jaw_body.stl", runtime_meshes.values())
            self.assertFalse({"wrist_roll_follower_so101_v1.stl", "moving_jaw_so101_v1.stl"} & set(runtime_meshes))

            # The frozen legacy reference records the reviewed local geometry and
            # the camera-mount anchor used to derive the new children.
            legacy_root = ET.parse(MODEL_DIR / "sources/soft_fin_reference.urdf").getroot()
            for link_name, mesh_name in (
                ("fixed_soft_fin", "fixed_fin.stl"),
                ("moving_soft_fin", "moving_fin.stl"),
            ):
                with self.subTest(fin=link_name):
                    self.assertEqual(_mesh_filename(result.find(f"link[@name='{link_name}']/visual")), f"assets/soft_fin/{mesh_name}")
                    legacy_link = legacy_root.find(f"link[@name='{link_name}']")
                    self.assertIsNotNone(legacy_link)
                    self.assertEqual(_mesh_filename(legacy_link.find("visual")), f"assets/soft_fin/{mesh_name}")
                    self.assertEqual((output / runtime_meshes[mesh_name]).read_bytes(), (MODEL_DIR / f"assets/soft_fin/{mesh_name}").read_bytes())
                    output_link = result.find(f"link[@name='{link_name}']")
                    for role in ("visual", "collision"):
                        self.assertEqual(
                            [_signature(element) for element in output_link.findall(role)],
                            [_signature(element) for element in legacy_link.findall(role)],
                        )
                    legacy_joint_name = f"{link_name}_joint"
                    legacy_joint = legacy_root.find(f"joint[@name='{legacy_joint_name}']")
                    self.assertIsNotNone(legacy_joint)
                    np.testing.assert_allclose(
                        _origin_transform(joint_by_child[link_name].find("origin")),
                        _origin_transform(legacy_joint.find("origin")),
                        atol=1e-6,
                        rtol=0,
                    )

            legacy_mount = legacy_root.find("joint[@name='gripper_mount']")
            self.assertIsNotNone(legacy_mount)
            np.testing.assert_allclose(
                _origin_transform(joint_by_child["soft_fin_camera_mount_link"].find("origin")),
                _origin_transform(legacy_mount.find("origin")),
                atol=1e-6,
                rtol=0,
            )
            legacy_mount_link = legacy_root.find("link[@name='gripper_camera_mount']")
            actual_mount_link = result.find("link[@name='soft_fin_camera_mount_link']")
            for role in ("visual", "collision"):
                self.assertEqual(
                    [_signature(element) for element in actual_mount_link.findall(role)],
                    [_signature(element) for element in legacy_mount_link.findall(role)],
                )

    def test_runtime_meshes_are_local_and_match_lock_hashes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="so101-lr-local-") as temporary:
            output = Path(temporary) / "model"
            result = ET.parse(self._build(output)).getroot()
            pins = _pin_records(self.upstream_lock)
            soft_fin_lock_path = MODEL_DIR / "soft_fin.lock.json"
            soft_fin_lock = json.loads(soft_fin_lock_path.read_text(encoding="utf-8"))
            pins.update(_pin_records(soft_fin_lock))
            references = [mesh.get("filename", "") for mesh in result.findall(".//mesh")]
            self.assertTrue(references)
            for filename in references:
                path = Path(filename)
                with self.subTest(mesh=filename):
                    self.assertFalse(path.is_absolute())
                    self.assertNotIn("..", path.parts)
                    self.assertEqual(urlparse(filename).scheme, "")
                    local_file = output / path
                    self.assertTrue(local_file.is_file())
                    self.assertTrue(local_file.resolve().is_relative_to(output.resolve()))
                    self.assertIn(path.as_posix(), pins, "runtime mesh must have a lock-file hash")
                    self.assertEqual(hashlib.sha256(local_file.read_bytes()).hexdigest(), pins[path.as_posix()])

    def test_fresh_offline_builds_are_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="so101-lr-repro-") as temporary:
            parent = Path(temporary)
            first = parent / "first"
            second = parent / "second"
            self._build(first)
            self._build(second)
            first_files = {path.relative_to(first).as_posix(): path.read_bytes() for path in first.rglob("*") if path.is_file()}
            second_files = {path.relative_to(second).as_posix(): path.read_bytes() for path in second.rglob("*") if path.is_file()}
            self.assertTrue(first_files)
            self.assertEqual(first_files.keys(), second_files.keys())
            for relative, data in first_files.items():
                with self.subTest(file=relative):
                    self.assertEqual(data, second_files[relative])

    def test_generated_model_loads_through_m3t_converter(self) -> None:
        script_dir = Path(__file__).resolve().parents[2] / "scripts"
        sys.path.insert(0, str(script_dir))
        try:
            from prepare_robot import convert_urdf
        finally:
            sys.path.remove(str(script_dir))

        with tempfile.TemporaryDirectory(prefix="so101-lr-convert-") as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            urdf = self._build(model_dir)
            converted_json = convert_urdf(urdf, root / "converted")
            converted = json.loads(converted_json.read_text(encoding="utf-8"))
            self.assertTrue(all(Path(converted_json.parent / link["mesh"]).is_file()
                                for link in converted["links"] if link["mesh"]))

        self.assertEqual(converted["root_link"], "base_link")
        self.assertEqual(len(converted["links"]), len(self.official.findall("link")) + len(ADDED_LINKS))
        self.assertEqual(sum(joint["type"] == "revolute" for joint in converted["joints"]), 6)
        self.assertEqual(len(converted["joints"]), len(converted["links"]) - 1)
        self.assertEqual({joint["name"] for joint in converted["joints"] if joint["type"] == "revolute"},
                         {joint.get("name") for joint in self.official.findall("joint") if joint.get("type") == "revolute"})
        self.assertTrue(all(link["mesh"] for link in converted["links"] if link["visuals"]))
        known_links = {link["name"] for link in converted["links"]}
        self.assertTrue(all(joint["parent"] in known_links and joint["child"] in known_links for joint in converted["joints"]))

    def test_corrupt_pinned_source_is_rejected_without_touching_shared_assets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="so101-lr-corrupt-") as temporary:
            root = Path(temporary)
            isolated_model = root / "isolated-model"
            shutil.copytree(MODEL_DIR, isolated_model)
            source = isolated_model / "sources" / "so101_new_calib.urdf"
            source.write_bytes(source.read_bytes() + b"\n<!-- corruption fixture -->\n")
            output = root / "rejected-output"
            with mock.patch.object(bm, "MODEL_DIR", isolated_model):
                with self.assertRaises(ValueError):
                    bm.build(output)
            self.assertFalse(output.exists())
            self.assertNotEqual(source.read_bytes(), OFFICIAL_URDF.read_bytes())


if __name__ == "__main__":
    unittest.main()
