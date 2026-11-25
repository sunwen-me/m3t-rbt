"""Offline contract tests for read-only camera/base/gripper pose evidence."""
import copy
import json
import math
import unittest
from unittest.mock import Mock

import numpy as np

import m3t_preview as preview


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ARMS = ("so101_left", "so101_right")
MATRIX_FIELDS = (
    "T_camera_from_base_link",
    "T_base_link_from_gripper_link",
    "T_camera_from_gripper_link_fk",
    "T_camera_from_gripper_link_m3t",
)


def rotation(axis, radians):
    """Independent elementary rotations; do not call production FK/Rotation."""
    c, s = math.cos(radians), math.sin(radians)
    result = np.eye(4)
    result[:3, :3] = (
        [[1, 0, 0], [0, c, -s], [0, s, c]] if axis == "x" else
        [[c, 0, s], [0, 1, 0], [-s, 0, c]] if axis == "y" else
        [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    )
    return result


def rigid_pose(xyz, rpy):
    result = rotation("z", rpy[2]) @ rotation("y", rpy[1]) @ rotation("x", rpy[0])
    result[:3, 3] = xyz
    return result


class GripperPoseOutputTests(unittest.TestCase):
    def setUp(self):
        self.axes = ("z", "y", "x", "y", "z")
        self.origins = [
            rigid_pose(
                [0.03 + i * 0.01, -0.02 + i * 0.003, 0.04],
                [0.03 * (i + 1), -0.02 * i, 0.01 * i],
            )
            for i in range(5)
        ]
        joints = []
        parent = "base_link"
        for i, name in enumerate(JOINTS):
            child = "gripper_link" if i == 4 else f"link_{i}"
            axis = [int(self.axes[i] == component) for component in "xyz"]
            joints.append({
                "name": name, "type": "revolute", "parent": parent,
                "child": child, "origin": self.origins[i].reshape(-1).tolist(),
                "axis": axis, "lower": -3.0, "upper": 3.0,
            })
            parent = child
        self.model = {
            "root_link": "base_link", "joints": joints,
            "links": [{"name": name} for name in ["base_link"] + [j["child"] for j in joints]],
        }
        self.angles = {
            ARMS[0]: [0.41, -0.32, 0.23, 0.17, -0.29],
            ARMS[1]: [-0.36, 0.27, -0.19, 0.31, 0.22],
        }
        self.bases = {
            ARMS[0]: rigid_pose([0.18, -0.24, 0.62], [0.22, -0.16, 0.37]),
            ARMS[1]: rigid_pose([-0.21, 0.15, 0.74], [-0.18, 0.26, -0.43]),
        }
        self.fk_expected = {arm: self.independent_fk(q) for arm, q in self.angles.items()}
        self.native = {arm: self.bases[arm] @ self.fk_expected[arm] for arm in ARMS}
        self.estimate = {
            "event": "estimate", "frame_sequence": 47,
            "frame_id": "orbbec335/color_optical", "color_capture_ns": 123456789,
            "links": [
                {"arm": arm, "link": name, "pose": matrix.reshape(-1).tolist()}
                for arm in ARMS
                for name, matrix in (("base_link", self.bases[arm]), ("gripper_link", self.native[arm]))
            ],
            "joint_angles_rad": {
                arm: dict(zip(JOINTS, q), gripper=0.13) for arm, q in self.angles.items()
            },
        }
        # Bypass startup entirely: no mesh, hardware, subprocess, or network.
        self.experiment = preview.M3TExperiment.__new__(preview.M3TExperiment)
        self.experiment.runtime_model = self.model
        self.experiment._contexts = Mock(return_value=dict.fromkeys(ARMS, object()))

    def independent_fk(self, radians):
        result = np.eye(4)
        for origin, axis, q in zip(self.origins, self.axes, radians):
            result = result @ origin @ rotation(axis, q)
        return result

    def link(self, arm, name, estimate=None):
        value = self.estimate if estimate is None else estimate
        return next(row for row in value["links"] if (row["arm"], row["link"]) == (arm, name))

    def assert_rejected(self, estimate=None):
        value = self.estimate if estimate is None else estimate
        before = json.dumps(value, sort_keys=True)
        with self.assertRaises(ValueError):
            self.experiment.tcp_poses(value)
        self.assertEqual(json.dumps(value, sort_keys=True), before)

    def assert_pose_fields(self, actual, arm, native=None):
        expected = (self.bases[arm], self.fk_expected[arm], self.native[arm],
                    self.native[arm] if native is None else native)
        self.assertEqual(set(actual), set(MATRIX_FIELDS) | {"max_matrix_abs_difference"})
        for field, matrix in zip(MATRIX_FIELDS, expected):
            with self.subTest(arm=arm, field=field):
                self.assertIsInstance(actual[field], list)
                self.assertEqual(len(actual[field]), 16)
                np.testing.assert_allclose(actual[field], matrix.reshape(-1), atol=1e-12, rtol=0)
        difference = float(np.max(np.abs(expected[2] - expected[3])))
        self.assertIsInstance(actual["max_matrix_abs_difference"], float)
        self.assertTrue(math.isfinite(actual["max_matrix_abs_difference"]))
        self.assertAlmostEqual(actual["max_matrix_abs_difference"], difference, places=14)

    def test_dual_arm_fields_radian_angles_and_camera_from_gripper_direction(self):
        result = self.experiment.tcp_poses(self.estimate)
        self.assertEqual(set(result), set(ARMS))
        for arm in ARMS:
            self.assert_pose_fields(result[arm], arm)
            # Exercise noncommuting transforms, metric translations and radian input.
            self.assertFalse(np.allclose(self.native[arm], self.fk_expected[arm] @ self.bases[arm]))
            self.assertFalse(np.allclose(self.native[arm], np.linalg.inv(self.native[arm])))
            self.assertGreater(np.linalg.norm(self.native[arm][:3, 3]), 0.1)
            self.assertLess(np.linalg.norm(self.native[arm][:3, 3]), 2.0)
            wrong_units = self.bases[arm] @ self.independent_fk(np.deg2rad(self.angles[arm]))
            self.assertFalse(np.allclose(self.native[arm], wrong_units, atol=2e-4))
        self.assertFalse(np.allclose(self.native[ARMS[0]], self.native[ARMS[1]]))

    def test_single_arm_and_input_order_are_supported(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                estimate = copy.deepcopy(self.estimate)
                estimate["links"] = [row for row in reversed(estimate["links"]) if row["arm"] == arm]
                estimate["joint_angles_rad"] = {arm: estimate["joint_angles_rad"][arm]}
                self.experiment._contexts.return_value = {arm: object()}
                result = self.experiment.tcp_poses(estimate)
                self.assertEqual(set(result), {arm})
                self.assert_pose_fields(result[arm], arm)

    def test_valid_model_intermediate_link_does_not_change_tcp_output(self):
        self.estimate["links"].append({
            "arm": ARMS[0], "link": "link_0",
            "pose": (self.bases[ARMS[0]] @ self.origins[0]
                     @ rotation(self.axes[0], self.angles[ARMS[0]][0])).reshape(-1).tolist(),
        })
        self.assert_pose_fields(self.experiment.tcp_poses(self.estimate)[ARMS[0]], ARMS[0])

    def test_small_native_residual_is_reported_without_overwriting_either_pose(self):
        arm = ARMS[1]
        native = self.native[arm].copy()
        native[:3, 3] += [0.00012, -0.00007, 0.00003]
        self.link(arm, "gripper_link")["pose"] = native.reshape(-1).tolist()
        result = self.experiment.tcp_poses(self.estimate)
        self.assert_pose_fields(result[arm], arm, native)
        self.assertAlmostEqual(result[arm]["max_matrix_abs_difference"], 0.00012)
        self.assert_pose_fields(result[ARMS[0]], ARMS[0])

    def test_matrix_difference_limit_is_two_e_minus_four(self):
        for residual in (1.99999e-4, -1.99999e-4, 2.00001e-4, -2.00001e-4):
            with self.subTest(residual=residual):
                estimate = copy.deepcopy(self.estimate)
                self.link(ARMS[0], "gripper_link", estimate)["pose"][3] += residual
                if abs(residual) > 2e-4:
                    self.assert_rejected(estimate)
                else:
                    actual = self.experiment.tcp_poses(estimate)[ARMS[0]]
                    self.assertAlmostEqual(actual["max_matrix_abs_difference"], abs(residual), places=14)

    def test_rejects_wrong_transform_direction_and_rotation_mismatch(self):
        arm = ARMS[0]
        for native in (self.fk_expected[arm] @ self.bases[arm],
                       np.linalg.inv(self.native[arm]), self.native[arm] @ rotation("x", 0.002)):
            with self.subTest(native=native.tolist()):
                estimate = copy.deepcopy(self.estimate)
                self.link(arm, "gripper_link", estimate)["pose"] = native.reshape(-1).tolist()
                self.assert_rejected(estimate)

    def test_rejects_missing_and_duplicate_base_or_gripper_for_either_arm(self):
        for arm in ARMS:
            for name in ("base_link", "gripper_link"):
                for duplicate in (False, True):
                    with self.subTest(arm=arm, link=name, duplicate=duplicate):
                        estimate = copy.deepcopy(self.estimate)
                        row = self.link(arm, name, estimate)
                        if duplicate:
                            estimate["links"].append(copy.deepcopy(row))
                        else:
                            estimate["links"].remove(row)
                        self.assert_rejected(estimate)

    def test_rejects_unknown_arm_or_link_even_with_complete_valid_required_links(self):
        for arm, name in (("so101_typo", "base_link"), ("so101_typo", "gripper_link"),
                          (ARMS[0], "unknown_link")):
            with self.subTest(arm=arm, link=name):
                estimate = copy.deepcopy(self.estimate)
                estimate["links"].append({"arm": arm, "link": name, "pose": np.eye(4).reshape(-1).tolist()})
                self.assert_rejected(estimate)

    def test_rejects_joint_arm_set_mismatch(self):
        for arm in ARMS:
            for extra in (False, True):
                with self.subTest(arm=arm, extra=extra):
                    estimate = copy.deepcopy(self.estimate)
                    if extra:
                        estimate["joint_angles_rad"]["so101_typo"] = estimate["joint_angles_rad"][arm].copy()
                    else:
                        del estimate["joint_angles_rad"][arm]
                    self.assert_rejected(estimate)

    def test_rejects_missing_nonfinite_boolean_and_nonnumeric_five_axis_angles(self):
        for arm in ARMS:
            for name in JOINTS:
                for bad in (None, float("nan"), float("inf"), -float("inf"), True, False, "0.2"):
                    with self.subTest(arm=arm, joint=name, bad=bad):
                        estimate = copy.deepcopy(self.estimate)
                        if bad is None:
                            del estimate["joint_angles_rad"][arm][name]
                        else:
                            estimate["joint_angles_rad"][arm][name] = bad
                        self.assert_rejected(estimate)

    def test_rejects_nonrigid_nonfinite_and_malformed_link_matrices(self):
        for arm in ARMS:
            for name in ("base_link", "gripper_link"):
                for index, value in ((0, 2.0), (0, float("nan")), (3, float("inf")),
                                     (12, 0.1), (15, 0.0)):
                    with self.subTest(arm=arm, link=name, index=index, value=value):
                        estimate = copy.deepcopy(self.estimate)
                        self.link(arm, name, estimate)["pose"][index] = value
                        self.assert_rejected(estimate)
                estimate = copy.deepcopy(self.estimate)
                self.link(arm, name, estimate)["pose"] = [0.0] * 15
                self.assert_rejected(estimate)
                estimate = copy.deepcopy(self.estimate)
                reflection = np.eye(4)
                reflection[0, 0] = -1
                self.link(arm, name, estimate)["pose"] = reflection.reshape(-1).tolist()
                self.assert_rejected(estimate)

    def test_rejects_missing_runtime_model_wrong_root_and_broken_five_axis_chain(self):
        invalid = [None, {}, dict(self.model, root_link="world")]
        broken = copy.deepcopy(self.model)
        broken["joints"][-1]["parent"] = "missing_link"
        invalid.append(broken)
        missing_axis = copy.deepcopy(self.model)
        missing_axis["joints"][2]["type"] = "fixed"
        invalid.append(missing_axis)
        bad_origin = copy.deepcopy(self.model)
        bad_origin["joints"][2]["origin"][0] = 2.0
        invalid.append(bad_origin)
        bad_axis = copy.deepcopy(self.model)
        bad_axis["joints"][2]["axis"] = [2, 0, 0]
        invalid.append(bad_axis)
        for model in invalid:
            with self.subTest(model=model):
                self.experiment.runtime_model = model
                self.assert_rejected()

    def test_rejects_out_of_limit_joint_angle(self):
        self.estimate["joint_angles_rad"][ARMS[0]][JOINTS[2]] = 3.1
        self.assert_rejected()

    def test_success_and_failure_preserve_estimate_and_output_does_not_alias_input(self):
        before = copy.deepcopy(self.estimate)
        model_before = copy.deepcopy(self.model)
        first = self.experiment.tcp_poses(self.estimate)
        self.assertEqual(self.estimate, before)
        self.assertEqual(self.model, model_before)
        for field in MATRIX_FIELDS:
            first[ARMS[0]][field][3] += 10
        self.assertEqual(self.estimate, before)
        self.assertEqual(self.model, model_before)
        self.assert_pose_fields(self.experiment.tcp_poses(self.estimate)[ARMS[0]], ARMS[0])
        self.link(ARMS[1], "gripper_link")["pose"][3] += 0.01
        self.assert_rejected()
        self.assertEqual(self.model, model_before)


if __name__ == "__main__":
    unittest.main()
