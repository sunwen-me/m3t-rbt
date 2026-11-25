"""Synthetic offline checks. Run: python -m unittest discover -s <this directory>."""

import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import handeye


def transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def project(matrix, intrinsics):
    # Independent projection with explicitly ordered corners, not solver code.
    points = np.array([[-0.0145, 0.0145, 0, 1], [0.0145, 0.0145, 0, 1],
                       [0.0145, -0.0145, 0, 1], [-0.0145, -0.0145, 0, 1]])
    camera = (matrix @ points.T)[:3]
    pixels = intrinsics @ camera
    return (pixels[:2] / pixels[2]).T


def fixture(count=6, single_axis=False, noise=0.0, behind=False):
    rng = np.random.default_rng(20261101)
    truth = {
        "so101_left": (transform([0.12, -0.18, 0.07], [0.08, -0.04, -0.65 if behind else 0.65]),
                       transform([0.08, -0.14, 0.04], [0.008, 0.025, 0.045])),
        "so101_right": (transform([-0.08, 0.13, -0.12], [-0.12, 0.06, -0.75 if behind else 0.75]),
                        transform([-0.07, 0.09, 0.10], [-0.006, 0.035, 0.055])),
    }
    nominal = {arm: x @ transform([0.025, -0.02, 0.01], [0.004, -0.003, 0.005])
               for arm, (_, x) in truth.items()}
    angles = [[0, 0, 0], [0.5, 0.1, -0.2], [-0.3, 0.55, 0.15],
              [0.25, -0.45, 0.4], [-0.45, -0.25, -0.3], [0.35, 0.4, -0.4]]
    samples = []
    for index in range(count):
        arms = {}
        for arm, (b, x) in truth.items():
            rotation = [0, 0, index * 0.16] if single_axis else angles[index]
            f = transform(rotation, [0.035 * np.sin(index), 0.04 * np.cos(index), 0.015 * index])
            matrix = b @ f @ x
            # Each snapshot may have its own intrinsics.
            k = np.array([[820 + index * 3, 0, 320], [0, 805 + index * 2, 240], [0, 0, 1.]])
            pixels = project(matrix, k) + rng.normal(0, noise, (4, 2))
            seed = matrix @ transform([0.015, -0.01, 0.005], [0.001, -0.002, 0.002])
            arms[arm] = {"T_base_from_gripper": f.tolist(), "corners_px": pixels.tolist(),
                         "K": k.tolist(), "T_camera_from_marker_seed": seed.tolist(),
                         "unused_depth": 123, "joint_provenance": "ignored"}
        samples.append({"id": f"sample-{index}", "pose_group": f"pose-{index}", "arms": arms})
    return {"schema_version": 1, "setup_id": "synthetic-fixed-setup",
            "setup_confirmed_unchanged": True, "marker_side_m": 0.029,
            "nominal_T_gripper_from_marker": {arm: x.tolist() for arm, x in nominal.items()},
            "samples": samples, "provenance": {"unused": True}}, truth


def add_depth_planes(manifest, truth, noise_m=0.0):
    """Simulate external plane measurements from synthetic truth, never seeds."""
    rng = np.random.default_rng(301)
    for sample in manifest["samples"]:
        for arm, observation in sample["arms"].items():
            b, x = truth[arm]
            camera_from_marker = b @ np.array(observation["T_base_from_gripper"]) @ x
            normal = camera_from_marker[:3, 2]
            offset = -float(normal @ camera_from_marker[:3, 3])
            observation["depth_plane"] = {
                "normal_camera": normal.tolist(),
                "offset_m": offset + float(rng.normal(0, noise_m)), "sigma_m": 0.003}
    return manifest


class HandeyeTests(unittest.TestCase):
    def assert_candidate(self, report, status):
        self.assertEqual(report["status"], status)
        self.assertFalse(report["accepted"])
        for arm in report["arms"].values():
            self.assertEqual(arm["status"], "CANDIDATE_ONLY")
            self.assertIn("candidate", arm)
            self.assertNotIn("accepted", arm)
            for matrix in arm["candidate"].values():
                handeye._transform(matrix, "candidate")
        json.dumps(report, allow_nan=False)

    def test_recovers_both_transforms_and_predicts_unseen_pose(self):
        manifest, truth = fixture()
        report = handeye.solve_manifest(manifest)
        self.assertEqual(report["status"], "VALIDATED_MULTI_POSE")
        self.assertTrue(report["accepted"])
        unseen_f = transform([0.21, -0.31, -0.22], [-0.022, 0.031, 0.08])
        k = np.array([[900, 0, 315], [0, 890, 235], [0, 0, 1.]])
        for name, (b, x) in truth.items():
            arm = report["arms"][name]
            fitted_b = np.array(arm["accepted"]["T_camera_from_base"])
            fitted_x = np.array(arm["accepted"]["T_gripper_from_marker"])
            np.testing.assert_allclose(fitted_b, b, atol=2e-7)
            np.testing.assert_allclose(fitted_x, x, atol=2e-7)
            np.testing.assert_allclose(project(fitted_b @ unseen_f @ fitted_x, k),
                                       project(b @ unseen_f @ x, k), atol=1e-5)
            self.assertTrue(arm["jacobian"]["full_rank"])
            self.assertEqual(len(arm["jacobian"]["singular_values"]), 12)
            self.assertGreater(arm["jacobian"]["smallest_to_largest_ratio"], 1e-5)
            self.assertLess(arm["training"]["rms_px"], 1e-6)
            self.assertTrue(arm["validation"]["passed"])
            self.assertLess(arm["validation"]["max_prediction_frame_rms_px"], 1e-5)

    def test_single_arm_manifests_solve_and_gate_the_observed_arm(self):
        for name in handeye.ARMS:
            with self.subTest(arm=name):
                manifest, truth = fixture()
                absent = next(arm for arm in handeye.ARMS if arm != name)
                for sample in manifest["samples"]:
                    del sample["arms"][absent]
                del manifest["nominal_T_gripper_from_marker"][absent]

                report = handeye.solve_manifest(manifest)

                self.assertEqual(report["status"], "VALIDATED_MULTI_POSE")
                self.assertTrue(report["accepted"])
                self.assertEqual(set(report["arms"]), {name})
                arm = report["arms"][name]
                self.assertTrue(arm["validation"]["passed"])
                np.testing.assert_allclose(arm["accepted"]["T_camera_from_base"], truth[name][0], atol=2e-7)
                np.testing.assert_allclose(arm["accepted"]["T_gripper_from_marker"], truth[name][1], atol=2e-7)

    def test_single_arm_identity_nominal_handles_near_pi_marker_rotation(self):
        marker_mount = transform([np.pi - 0.01, 0.02, -0.04], [0.008, 0.025, 0.045])
        seed_bias = transform([0.015, -0.01, 0.005], [0.001, -0.002, 0.002])
        for name in handeye.ARMS:
            with self.subTest(arm=name):
                manifest, truth = fixture()
                absent = next(arm for arm in handeye.ARMS if arm != name)
                for sample in manifest["samples"]:
                    del sample["arms"][absent]
                del manifest["nominal_T_gripper_from_marker"][absent]
                base, _ = truth[name]
                truth[name] = (base, marker_mount)
                manifest["nominal_T_gripper_from_marker"][name] = np.eye(4).tolist()
                for sample in manifest["samples"]:
                    observation = sample["arms"][name]
                    fk = np.asarray(observation["T_base_from_gripper"])
                    camera_from_marker = base @ fk @ marker_mount
                    observation["corners_px"] = project(
                        camera_from_marker, np.asarray(observation["K"])).tolist()
                    observation["T_camera_from_marker_seed"] = (
                        camera_from_marker @ seed_bias).tolist()

                report = handeye.solve_manifest(manifest)

                self.assertEqual(report["status"], "VALIDATED_MULTI_POSE")
                self.assertTrue(report["accepted"])
                np.testing.assert_allclose(
                    report["arms"][name]["accepted"]["T_gripper_from_marker"],
                    marker_mount, atol=2e-7)

    def test_present_arm_requires_nominal_and_unknown_arm_names_are_rejected(self):
        manifest, _ = fixture(count=1)
        del manifest["nominal_T_gripper_from_marker"]["so101_left"]
        with self.assertRaisesRegex(ValueError, "required for observed arm"):
            handeye.solve_manifest(manifest)

        manifest, _ = fixture(count=1)
        manifest["samples"][0]["arms"]["so101_center"] = copy.deepcopy(
            manifest["samples"][0]["arms"]["so101_left"])
        with self.assertRaisesRegex(ValueError, "supported arms"):
            handeye.solve_manifest(manifest)

        manifest, _ = fixture(count=1)
        manifest["nominal_T_gripper_from_marker"]["so101_center"] = np.eye(4).tolist()
        with self.assertRaisesRegex(ValueError, "unsupported arm name"):
            handeye.solve_manifest(manifest)

    def test_low_pixel_noise_is_accepted(self):
        manifest, _ = fixture(noise=0.15)
        report = handeye.solve_manifest(manifest)
        self.assertEqual(report["status"], "VALIDATED_MULTI_POSE")
        for arm in report["arms"].values():
            self.assertLess(arm["training"]["rms_px"], 0.4)

    def test_single_and_two_pose_fits_remain_candidates(self):
        for count in (1, 2):
            with self.subTest(count=count):
                manifest, _ = fixture(count=count)
                report = handeye.solve_manifest(manifest)
                self.assert_candidate(report, "NEED_MORE_POSES")
                for arm in report["arms"].values():
                    self.assertIn("at_least_4_distinct_pose_groups_required", arm["reasons"])
                    self.assertFalse(arm["validation"]["passed"])

    def test_single_axis_rotation_refused(self):
        manifest, _ = fixture(single_axis=True)
        report = handeye.solve_manifest(manifest)
        self.assert_candidate(report, "NEED_MORE_POSES")
        for arm in report["arms"].values():
            self.assertGreater(arm["excitation"]["max_relative_rotation_deg"], 15)
            self.assertLess(arm["excitation"]["second_to_first_ratio"], 1e-10)
            self.assertIn("insufficient_nonparallel_rotation_excitation", arm["reasons"])

    def test_full_rank_fit_refused_when_a_holdout_subset_loses_rank(self):
        manifest, truth = fixture(single_axis=True)
        # Only the sixth pose changes the rotation axis. The full fit is
        # identifiable, but cannot predict that pose from the other five.
        for arm, (b, x) in truth.items():
            observation = manifest["samples"][-1]["arms"][arm]
            f = transform([0.6, -0.4, 0.1], [0.01, 0.02, 0.07])
            observation["T_base_from_gripper"] = f.tolist()
            observation["corners_px"] = project(b @ f @ x, np.array(observation["K"])).tolist()
            observation["T_camera_from_marker_seed"] = (b @ f @ x).tolist()
        report = handeye.solve_manifest(manifest)
        self.assert_candidate(report, "NEED_MORE_POSES")
        for arm in report["arms"].values():
            self.assertTrue(arm["excitation"]["sufficient"])
            self.assertTrue(arm["jacobian"]["full_rank"])
            self.assertLess(arm["training"]["rms_px"], 1e-5)
            self.assertFalse(arm["validation"]["folds"][-1]["jacobian"]["full_rank"])
            self.assertIn("leave_one_group_out_subset_insufficient_or_rank_deficient", arm["reasons"])

    def test_repeated_snapshots_do_not_increase_pose_count(self):
        manifest, _ = fixture(count=2)
        for index in range(8):
            sample = copy.deepcopy(manifest["samples"][index % 2])
            sample["id"] = f"repeat-{index}"
            manifest["samples"].append(sample)
        report = handeye.solve_manifest(manifest)
        self.assert_candidate(report, "NEED_MORE_POSES")
        for arm in report["arms"].values():
            self.assertEqual(arm["sample_count"], 10)
            self.assertEqual(arm["excitation"]["distinct_pose_groups"], 2)

    def test_new_labels_cannot_turn_repeated_fk_into_independent_poses(self):
        manifest, _ = fixture()
        # The right arm moves through six poses; the left visits only three.
        # Global labels alone would incorrectly give both arms six poses.
        for index in range(3, 6):
            manifest["samples"][index]["arms"]["so101_left"] = copy.deepcopy(
                manifest["samples"][index - 3]["arms"]["so101_left"])
        report = handeye.solve_manifest(manifest)
        self.assert_candidate(report, "NEED_MORE_POSES")
        left = report["arms"]["so101_left"]
        right = report["arms"]["so101_right"]
        self.assertEqual(left["excitation"]["distinct_pose_groups"], 6)
        self.assertEqual(left["excitation"]["materially_distinct_pose_count"], 3)
        self.assertEqual(left["excitation"]["duplicate_pose_groups"],
                         [["pose-0", "pose-3"], ["pose-1", "pose-4"], ["pose-2", "pose-5"]])
        self.assertIn("at_least_4_materially_distinct_fk_poses_required", left["reasons"])
        self.assertEqual(right["excitation"]["materially_distinct_pose_count"], 6)
        self.assertEqual(right["assessment"], "VALIDATED_MULTI_POSE")
        self.assertEqual(len(left["validation"]["folds"]), 6)

    def test_fk_diversity_tolerances_require_both_translation_and_rotation(self):
        observations = []
        for label, metres, degrees in (("a", 0, 0), ("b", 0.0009, 0.9),
                                       ("c", 0.01, 0.5), ("d", 0.0005, 5)):
            observations.append({"pose_group": label,
                                 "F": transform([0, 0, np.deg2rad(degrees)], [metres, 0, 0])})
        diversity = handeye._excitation(observations)
        self.assertEqual(diversity["materially_distinct_pose_count"], 3)
        self.assertEqual(diversity["duplicate_pose_groups"], [["a", "b"]])
        self.assertEqual(diversity, handeye._excitation(list(reversed(observations))))

    def test_whole_groups_heldout_and_seeds_do_not_leak(self):
        manifest, _ = fixture()
        duplicate = copy.deepcopy(manifest["samples"][0])
        duplicate["id"] = "duplicate-snapshot"
        manifest["samples"].append(duplicate)
        observations = handeye._observations(manifest, "so101_left")
        nominal = np.array(manifest["nominal_T_gripper_from_marker"]["so101_left"])
        original_fit = handeye._fit
        seen = []

        def record_fit(training, corners, initial_x):
            seen.append({obs["id"] for obs in training})
            return original_fit(training, corners, initial_x)

        with patch.object(handeye, "_fit", side_effect=record_fit):
            result = handeye._validate_groups(observations, handeye.marker_corners(0.029), nominal)
        for fold, training_ids in zip(result["folds"], seen):
            self.assertEqual(training_ids, set(fold["training_sample_ids"]))
            self.assertFalse(training_ids & set(fold["heldout_sample_ids"]))
        first = result["folds"][0]
        self.assertEqual(set(first["heldout_sample_ids"]), {"sample-0", "duplicate-snapshot"})
        altered = copy.deepcopy(observations)
        for obs in altered:
            if obs["pose_group"] == "pose-0":
                obs["seed"] = transform([1, 2, 1], [3, 2, 1])
        changed = handeye._validate_groups(altered, handeye.marker_corners(0.029), nominal)
        self.assertEqual(first["prediction"], changed["folds"][0]["prediction"])

    def test_high_noise_and_outlier_not_accepted(self):
        for noise, outlier in ((5.0, False), (0.15, True)):
            with self.subTest(noise=noise, outlier=outlier):
                manifest, _ = fixture(noise=noise)
                if outlier:
                    # Corrupt just one arm: even a good other arm is not promoted.
                    manifest["samples"][2]["arms"]["so101_left"]["corners_px"][0][0] += 35
                report = handeye.solve_manifest(manifest)
                self.assert_candidate(report, "REJECTED_FIT")
                self.assertIn("training_frame_rms_exceeds_3px",
                              report["arms"]["so101_left"]["reasons"])

    def test_negative_depth_is_rejected_even_with_exact_reprojection(self):
        manifest, _ = fixture(behind=True)
        report = handeye.solve_manifest(manifest)
        self.assert_candidate(report, "REJECTED_FIT")
        for arm in report["arms"].values():
            self.assertLess(arm["training"]["rms_px"], 1e-5)
            self.assertIn("nonpositive_corner_depth", arm["reasons"])

    def test_raw_pixel_rms_convention(self):
        manifest, truth = fixture(count=1)
        b, x = truth["so101_left"]
        observations = handeye._observations(manifest, "so101_left")
        observations[0]["pixels"] += [3, 4]
        metrics = handeye._metrics(np.r_[handeye._pack(b), handeye._pack(x)],
                                   observations, handeye.marker_corners(0.029))
        self.assertAlmostEqual(metrics["rms_px"], 5.0)

    def test_invalid_manifest_hard_checks(self):
        original, _ = fixture(count=1)
        cases = []
        for key, value in (("schema_version", True), ("setup_confirmed_unchanged", False),
                           ("setup_confirmed_unchanged", "true"), ("setup_id", " "),
                           ("marker_side_m", float("nan")), ("marker_side_m", 0)):
            bad = copy.deepcopy(original)
            bad[key] = value
            cases.append(bad)
        duplicate = copy.deepcopy(original)
        duplicate["samples"].append(copy.deepcopy(duplicate["samples"][0]))
        cases.append(duplicate)
        for key, row, column, value in (("T_base_from_gripper", 0, 0, -1),
                                         ("T_camera_from_marker_seed", 3, 3, 2),
                                         ("T_base_from_gripper", 0, 0, 2),
                                         ("T_base_from_gripper", 0, 3, float("inf")),
                                         ("K", 0, 0, -800), ("corners_px", 0, 0, float("nan"))):
            bad = copy.deepcopy(original)
            bad["samples"][0]["arms"]["so101_left"][key][row][column] = value
            cases.append(bad)
        bad = copy.deepcopy(original)
        bad["nominal_T_gripper_from_marker"]["so101_left"][2][2] = 0
        cases.append(bad)
        bad = copy.deepcopy(original)
        bad["samples"][0]["pose_group"] = ""
        cases.append(bad)
        for index, manifest in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                handeye.solve_manifest(manifest)

    def test_cli_existing_output_and_duplicate_json_keys_refused(self):
        with patch.object(Path, "exists", return_value=True), \
                patch.object(handeye, "solve_manifest") as solve, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(handeye.main(["--manifest", "input.json", "--output", "output.json"]), 2)
            solve.assert_not_called()
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            json.loads('{"pose_group": "a", "pose_group": "b"}',
                       object_pairs_hook=handeye._unique_json_object)

    def test_cli_uses_exclusive_creation_and_keeps_candidate(self):
        manifest, _ = fixture(count=2)

        class Output(io.StringIO):
            def close(self):
                pass

        output = Output()
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "open", side_effect=[io.StringIO(json.dumps(manifest)), output]) as op, \
                contextlib.redirect_stdout(io.StringIO()):
            code = handeye.main(["--manifest", "input.json", "--output", "output.json"])
        self.assertEqual(code, 0)
        self.assertEqual(op.call_args_list[-1].args, ("x",))
        self.assert_candidate(json.loads(output.getvalue()), "NEED_MORE_POSES")


class RGBDTests(unittest.TestCase):
    def test_exact_planes_recover_transforms_with_biased_pnp_seeds(self):
        manifest, truth = fixture()
        add_depth_planes(manifest, truth)
        for sample in manifest["samples"]:
            for observation in sample["arms"].values():
                observation["T_camera_from_marker_seed"][2][3] += 0.025
        report = handeye.solve_manifest(manifest)
        self.assertEqual(report["status"], "VALIDATED_MULTI_POSE")
        self.assertEqual(report["modality"], "RGB_D")
        for name, (b, x) in truth.items():
            arm = report["arms"][name]
            self.assertEqual(arm["depth_plane_count"], 6)
            self.assertEqual(arm["jacobian"]["modality"], "RGB_D")
            np.testing.assert_allclose(arm["accepted"]["T_camera_from_base"], b, atol=2e-7)
            np.testing.assert_allclose(arm["accepted"]["T_gripper_from_marker"], x, atol=2e-7)
            self.assertLess(arm["training"]["depth_plane"]["rms_mm"], 1e-5)
            self.assertLess(arm["validation"]["max_prediction_depth_plane_frame_rms_mm"], 1e-5)

    def test_noisy_planes_and_pixels_are_accepted(self):
        manifest, truth = fixture(noise=0.15)
        add_depth_planes(manifest, truth, noise_m=0.0008)
        report = handeye.solve_manifest(manifest)
        self.assertEqual(report["status"], "VALIDATED_MULTI_POSE")
        for arm in report["arms"].values():
            self.assertLess(arm["training"]["depth_plane"]["rms_mm"], 2)
            self.assertLess(arm["training"]["rms_px"], 0.5)

    def test_plane_weight_and_raw_pixels_and_millimetre_metrics(self):
        manifest, truth = fixture(count=1)
        add_depth_planes(manifest, truth)
        obs = handeye._observations(manifest, "so101_left")
        b, x = truth["so101_left"]
        parameters = np.r_[handeye._pack(b), handeye._pack(x)]
        corners = handeye.marker_corners(0.029)
        obs[0]["pixels"] += [3, 4]
        obs[0]["depth_plane"]["offset_m"] += 0.006
        residual = handeye._residual(parameters, obs, corners)
        self.assertEqual(residual.shape, (12,))
        np.testing.assert_allclose(residual[:8], [-3, -4] * 4, atol=1e-10)
        np.testing.assert_allclose(residual[8:], [1] * 4, atol=1e-10)
        self.assertAlmostEqual(float(residual[8:] @ residual[8:]), (0.006 / 0.003) ** 2)
        metrics = handeye._metrics(parameters, obs, corners)
        self.assertAlmostEqual(metrics["depth_plane"]["rms_mm"], 6)
        self.assertAlmostEqual(metrics["frames"][0]["depth_plane"]["max_abs_mm"], 6)
        # Simultaneously reversing normal and offset must describe the same plane.
        obs[0]["depth_plane"]["normal_camera"] = (-np.array(obs[0]["depth_plane"]["normal_camera"])).tolist()
        obs[0]["depth_plane"]["offset_m"] *= -1
        np.testing.assert_allclose(handeye._residual(parameters, obs, corners)[8:], [-1] * 4, atol=1e-10)

    def test_optional_planes_allow_mixed_coverage_and_rgb_arm(self):
        manifest, truth = fixture()
        add_depth_planes(manifest, truth)
        for index, sample in enumerate(manifest["samples"]):
            del sample["arms"]["so101_right"]["depth_plane"]
            if index % 2:
                del sample["arms"]["so101_left"]["depth_plane"]
        report = handeye.solve_manifest(manifest)
        self.assertEqual(report["status"], "VALIDATED_MULTI_POSE")
        left, right = report["arms"]["so101_left"], report["arms"]["so101_right"]
        self.assertEqual(left["depth_plane_count"], 3)
        self.assertEqual(left["training"]["depth_plane"]["frame_count"], 3)
        self.assertEqual(right["modality"], "RGB")
        self.assertNotIn("depth_plane", right["training"])
        self.assertNotIn("depth_plane", left["training"]["frames"][1])

    def test_inconsistent_depth_rejected_with_candidate_only(self):
        manifest, truth = fixture()
        add_depth_planes(manifest, truth)
        manifest["samples"][0]["arms"]["so101_left"]["depth_plane"]["offset_m"] += 0.030
        report = handeye.solve_manifest(manifest)
        self.assertEqual(report["status"], "REJECTED_FIT")
        self.assertFalse(report["accepted"])
        for arm in report["arms"].values():
            self.assertIn("candidate", arm)
            self.assertNotIn("accepted", arm)
        reasons = report["arms"]["so101_left"]["reasons"]
        self.assertIn("training_depth_plane_rms_exceeds_5mm", reasons)
        self.assertIn("training_depth_plane_frame_rms_exceeds_8mm", reasons)
        self.assertIn("leave_one_group_out_depth_plane_prediction_failed", reasons)

    def test_heldout_plane_is_not_used_for_fitting_but_is_gated(self):
        manifest, truth = fixture()
        add_depth_planes(manifest, truth)
        observations = handeye._observations(manifest, "so101_left")
        corners = handeye.marker_corners(0.029)
        nominal = np.array(manifest["nominal_T_gripper_from_marker"]["so101_left"])
        baseline = handeye._validate_groups(observations, corners, nominal)["folds"][0]
        observations[0]["depth_plane"]["offset_m"] += 0.012
        changed = handeye._validate_groups(observations, corners, nominal)["folds"][0]
        self.assertTrue(baseline["passed"])
        self.assertFalse(changed["passed"])
        self.assertEqual(changed["jacobian"], baseline["jacobian"])
        self.assertEqual(changed["prediction"]["rms_px"], baseline["prediction"]["rms_px"])
        self.assertAlmostEqual(changed["prediction"]["depth_plane"]["max_frame_rms_mm"], 12, places=5)

    def test_invalid_depth_plane_hard_checks(self):
        manifest, truth = fixture(count=1)
        add_depth_planes(manifest, truth)
        for key, value in (("normal_camera", [0, 0, 2]), ("normal_camera", [0, 0, 0]),
                           ("normal_camera", [0, float("nan"), 1]), ("normal_camera", [0, 1]),
                           ("offset_m", float("inf")), ("offset_m", None),
                           ("sigma_m", 0), ("sigma_m", -0.003), ("sigma_m", float("nan")),
                           ("sigma_m", True)):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(manifest)
                bad["samples"][0]["arms"]["so101_left"]["depth_plane"][key] = value
                with self.assertRaises(ValueError):
                    handeye.solve_manifest(bad)
        for value in (None, [], {}):
            bad = copy.deepcopy(manifest)
            bad["samples"][0]["arms"]["so101_left"]["depth_plane"] = value
            with self.assertRaises(ValueError):
                handeye.solve_manifest(bad)


if __name__ == "__main__":
    unittest.main()
