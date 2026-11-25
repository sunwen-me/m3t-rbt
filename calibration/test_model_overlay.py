"""Focused offline checks for SO101 contour projection and link transforms."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import shared_board_web as web
from model_overlay import ModelOverlay


class ModelOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        names = ("base_link", "shoulder_link", "upper_arm_link", "lower_arm_link",
                 "wrist_link", "gripper_link", "fixed_soft_fin", "soft_fin_camera_mount_link",
                 "moving_jaw_so101_v1_link", "moving_soft_fin")
        rendered = set(names)
        model = {
            "units": "metres_radians",
            "root_link": "base_link",
            "links": [{"name": name, "mesh": "triangle.obj" if name in rendered else ""}
                      for name in names],
            "joints": [],
        }
        def origin(x=0.0):
            transform = np.eye(4)
            transform[0, 3] = x
            return transform.reshape(-1).tolist()
        chain = (("shoulder_pan", "base_link", "shoulder_link", 0.1),
                 ("shoulder_lift", "shoulder_link", "upper_arm_link", 0.2),
                 ("elbow_flex", "upper_arm_link", "lower_arm_link", 0.3),
                 ("wrist_flex", "lower_arm_link", "wrist_link", 0.4),
                 ("wrist_roll", "wrist_link", "gripper_link", 0.5))
        for name, parent, child, x in chain:
            model["joints"].append({"name": name, "type": "revolute", "parent": parent,
                                    "child": child, "origin": origin(x), "axis": [0, 0, 1],
                                    "lower": -3.14, "upper": 3.14})
        model["joints"].extend((
            {"name": "fixed_soft_fin_joint", "type": "fixed", "parent": "gripper_link",
             "child": "fixed_soft_fin", "origin": origin(0.06), "axis": [0, 0, 0],
             "lower": 0, "upper": 0},
            {"name": "soft_fin_camera_mount_joint", "type": "fixed", "parent": "gripper_link",
             "child": "soft_fin_camera_mount_link", "origin": origin(0.08), "axis": [0, 0, 0],
             "lower": 0, "upper": 0},
            {"name": "gripper", "type": "revolute", "parent": "gripper_link",
             "child": "moving_jaw_so101_v1_link", "origin": origin(), "axis": [0, 0, 1],
             "lower": -0.2, "upper": 1.8},
            {"name": "moving_soft_fin_joint", "type": "fixed",
             "parent": "moving_jaw_so101_v1_link", "child": "moving_soft_fin",
             "origin": origin(), "axis": [0, 0, 0], "lower": 0, "upper": 0},
        ))
        (cls.directory / "triangle.obj").write_text(
            "# small link-local triangle in metres\n"
            "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 3 2\n"
        )
        cls.model_path = cls.directory / "robot.json"
        cls.model_path.write_text(json.dumps(model))
        open_rad = 1435 * 2 * np.pi / 4095
        cls.gripper_calibration = {
            "closed_rad": 0.0,
            "open_rad": open_rad,
            "scale_rad_per_percent": open_rad / 100,
            "calibration_sha256": "a" * 64,
            "mapping": "v2 calibrated encoder travel; no URDF limit scaling",
            "zero_reference": "physical closed endpoint assigned nominal model zero; alignment not independently measured",
        }
        cls.overlay = ModelOverlay(cls.model_path, cls.gripper_calibration)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_arm_link_transforms_match_shared_fk_and_fixed_child(self):
        degrees = [12.0, -30.0, 45.0, 11.0, -13.0]
        expected = web.fk(self.overlay.model, degrees)
        transforms = self.overlay._link_transforms(degrees, expected)
        np.testing.assert_allclose(transforms["gripper_link"], expected, atol=1e-10)
        fixed = next(j for j in self.overlay.model["joints"]
                     if j["name"] == "fixed_soft_fin_joint")
        np.testing.assert_allclose(
            transforms["fixed_soft_fin"],
            transforms["gripper_link"] @ np.asarray(fixed["origin"]).reshape(4, 4),
            atol=1e-10,
        )
        self.assertIn("moving_jaw_so101_v1_link", transforms)
        self.assertIn("moving_soft_fin", transforms)
        self.assertEqual(self.overlay.omitted_links, ())

    def test_positive_depth_projection_and_gripper_percent_metadata(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        camera_from_base = np.eye(4)
        camera_from_base[0, 3] = 0.2
        camera_from_base[2, 3] = 1.0
        intrinsics = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])

        with patch.object(self.overlay, "_draw_camera_poses",
                          wraps=self.overlay._draw_camera_poses) as draw:
            first, meta = self.overlay.render(
                image, [0, 0, 0, 0, 0], camera_from_base, intrinsics)
            closed_explicit, closed_meta = self.overlay.render(
                image, [0, 0, 0, 0, 0, 0.0], camera_from_base, intrinsics)
            _, midpoint = self.overlay.render(
                image, [0, 0, 0, 0, 0, 50.0], camera_from_base, intrinsics)
            _, with_percent = self.overlay.render(
                image, [0, 0, 0, 0, 0, 100.0], camera_from_base, intrinsics)

        self.assertFalse(np.array_equal(first, image))
        self.assertGreater(int(first[50, 70, 0]), 0)
        self.assertGreater(int(first[50, 70, 1]), 0)
        self.assertEqual(int(first[50, 70, 2]), 0)
        np.testing.assert_array_equal(first, closed_explicit)
        self.assertEqual(meta["T_base_from_gripper"], with_percent["T_base_from_gripper"])
        self.assertNotEqual(meta["gripper_angle_rad"], with_percent["gripper_angle_rad"])
        self.assertEqual(meta["gripper_percent"], 0.0)
        self.assertEqual(meta["gripper_angle_rad"], 0.0)
        self.assertEqual(meta["gripper_closed_rad"], 0.0)
        self.assertAlmostEqual(meta["gripper_open_rad"], 1435 * 2 * np.pi / 4095)
        self.assertGreater(meta["gripper_open_rad"], 1.8)
        self.assertEqual(meta["gripper_mapping"], self.gripper_calibration["mapping"])
        self.assertEqual(meta["gripper_calibration_sha256"], "a" * 64)
        self.assertEqual(meta["gripper_scale_rad_per_percent"],
                         self.gripper_calibration["scale_rad_per_percent"])
        self.assertEqual(meta["gripper_zero_reference"], self.gripper_calibration["zero_reference"])
        self.assertEqual(meta["gripper_calibration"], self.gripper_calibration)
        self.assertEqual(meta["gripper_input_assumption"],
                         "missing sixth input treated as 0% calibrated physical closed")
        self.assertIsNone(closed_meta["gripper_input_assumption"])
        self.assertAlmostEqual(midpoint["gripper_angle_rad"], meta["gripper_open_rad"] / 2)
        self.assertAlmostEqual(with_percent["gripper_angle_rad"], meta["gripper_open_rad"])
        self.assertEqual(with_percent["omitted_links"], [])
        self.assertEqual(with_percent["link_count"], 10)
        closed_poses = draw.call_args_list[1].args[1]
        opened_poses = draw.call_args_list[3].args[1]
        for link in ("moving_jaw_so101_v1_link", "moving_soft_fin"):
            self.assertFalse(np.allclose(closed_poses[link], opened_poses[link]))

    def test_gripper_endpoints_and_midpoint_move_jaw_and_fixed_child(self):
        degrees = [12.0, -30.0, 45.0, 11.0, -13.0]
        gripper_pose = web.fk(self.overlay.model, degrees)
        closed, opened = self.overlay._gripper_endpoints()
        self.assertEqual(closed, 0.0)
        self.assertAlmostEqual(opened, 1435 * 2 * np.pi / 4095)
        self.assertGreater(opened, 1.8)
        transforms = [
            self.overlay._link_transforms(degrees, gripper_pose, closed),
            self.overlay._link_transforms(degrees, gripper_pose, (closed + opened) / 2),
            self.overlay._link_transforms(degrees, gripper_pose, opened),
        ]
        jaw = "moving_jaw_so101_v1_link"
        child = "moving_soft_fin"
        self.assertIn(jaw, self.overlay._meshes)
        self.assertIn(child, self.overlay._meshes)
        for name in (jaw, child):
            self.assertFalse(np.allclose(transforms[0][name], transforms[1][name]))
            self.assertFalse(np.allclose(transforms[1][name], transforms[2][name]))
        for name in ("shoulder_link", "upper_arm_link", "lower_arm_link",
                     "wrist_link", "gripper_link"):
            np.testing.assert_allclose(transforms[0][name], transforms[1][name], atol=1e-10)
            np.testing.assert_allclose(transforms[1][name], transforms[2][name], atol=1e-10)
        moving_child_joint = next(j for j in self.overlay.model["joints"]
                                  if j["name"] == "moving_soft_fin_joint")
        child_origin = np.asarray(moving_child_joint["origin"]).reshape(4, 4)
        for pose in transforms:
            np.testing.assert_allclose(pose[child], pose[jaw] @ child_origin, atol=1e-10)

    def test_invalid_gripper_percent_is_rejected(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        intrinsics = np.array([[10.0, 0, 10], [0, 10.0, 10], [0, 0, 1]])
        for percent in (-0.01, 100.01, float("nan"), float("inf")):
            with self.subTest(percent=percent), self.assertRaises(ValueError):
                self.overlay.render(image, [0, 0, 0, 0, 0, percent], np.eye(4), intrinsics)

    def test_missing_physical_calibration_does_not_fall_back_to_urdf_limit(self):
        overlay = ModelOverlay(self.model_path)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        intrinsics = np.array([[10.0, 0, 10], [0, 10.0, 10], [0, 0, 1]])
        with self.assertRaisesRegex(ValueError, "physical gripper calibration"):
            overlay.render(image, [0, 0, 0, 0, 0, 100.0], np.eye(4), intrinsics)

    def test_actual_per_link_poses_match_fk_path_and_require_all_links(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        degrees = [12.0, -30.0, 45.0, 11.0, -13.0]
        camera_from_base = np.eye(4)
        camera_from_base[:3, 3] = [0.1, 0.05, 2.0]
        intrinsics = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
        base_from_links = self.overlay._link_transforms(
            degrees, web.fk(self.overlay.model, degrees))
        camera_from_links = {
            name: camera_from_base @ base_from_links[name]
            for name in self.overlay._meshes
        }

        from_fk, fk_metadata = self.overlay.render(image, degrees, camera_from_base, intrinsics)
        from_tracker, tracker_metadata = self.overlay.render_poses(
            image, camera_from_links, intrinsics)

        np.testing.assert_array_equal(from_tracker, from_fk)
        self.assertEqual(tracker_metadata["rendered_links"], fk_metadata["rendered_links"])
        self.assertEqual(tracker_metadata["pose_source"],
                         "caller-provided actual M3T per-link transforms")
        self.assertIn("moving_jaw_so101_v1_link", tracker_metadata["rendered_links"])
        self.assertIn("moving_soft_fin", tracker_metadata["rendered_links"])
        self.assertNotIn("T_base_from_gripper", tracker_metadata)
        incomplete = dict(camera_from_links)
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaises(ValueError):
            self.overlay.render_poses(image, incomplete, intrinsics)

    def test_actual_pose_path_passes_each_moving_link_pose_unchanged(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        intrinsics = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
        supplied = {name: np.eye(4) for name in self.overlay._meshes}
        for index, name in enumerate(("moving_jaw_so101_v1_link", "moving_soft_fin"), 1):
            supplied[name][0, 3] = index * 0.1
            supplied[name][2, 3] = 1.0

        with patch.object(self.overlay, "_draw_camera_poses",
                          wraps=self.overlay._draw_camera_poses) as draw:
            _, metadata = self.overlay.render_poses(image, supplied, intrinsics)

        actual_camera_poses = draw.call_args.args[1]
        for name, expected_pose in supplied.items():
            np.testing.assert_array_equal(actual_camera_poses[name], expected_pose)
        self.assertEqual(metadata["rendered_links"], list(self.overlay._meshes))

    def test_contour_cache_reuses_geometry_but_uses_current_image_and_color(self):
        self.overlay._contour_cache.clear()
        degrees = [0.0] * 5
        camera_from_base = np.eye(4)
        camera_from_base[2, 3] = 1.0
        transforms = self.overlay._link_transforms(
            degrees, web.fk(self.overlay.model, degrees))
        camera_from_links = {
            name: camera_from_base @ transforms[name]
            for name in self.overlay._meshes
        }
        intrinsics = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
        first_image = np.zeros((100, 100, 3), dtype=np.uint8)
        first, _ = self.overlay.render_poses(
            first_image, camera_from_links, intrinsics, color=(255, 0, 0))
        self.assertEqual(len(self.overlay._contour_cache), 1)

        current_image = np.full((100, 100, 3), (7, 8, 9), dtype=np.uint8)
        second, _ = self.overlay.render_poses(
            current_image, camera_from_links, intrinsics, color=(0, 255, 0))
        self.assertEqual(len(self.overlay._contour_cache), 1)
        self.assertEqual(second[99, 99].tolist(), [7, 8, 9])
        self.assertGreater(np.count_nonzero(second[:, :, 1] > current_image[:, :, 1]), 0)
        np.testing.assert_array_equal(first_image, np.zeros_like(first_image))

    def test_rejects_invalid_camera_transform_or_intrinsics(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        q = [0, 0, 0, 0, 0]
        with self.assertRaises(ValueError):
            self.overlay.render(image, q, np.zeros((4, 4)), np.eye(3))
        bad_k = np.array([[0.0, 0, 10], [0, 10.0, 10], [0, 0, 1]])
        with self.assertRaises(ValueError):
            self.overlay.render(image, q, np.eye(4), bad_k)


if __name__ == "__main__":
    unittest.main()
