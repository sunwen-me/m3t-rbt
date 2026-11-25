import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import shared_board


def _pose(rotvec, translation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    result[:3, 3] = translation
    return result


def _pixels(points, transform, camera):
    rotation, _ = cv2.Rodrigues(transform[:3, :3])
    pixels, _ = cv2.projectPoints(points, rotation, transform[:3, 3],
                                  camera["K"], camera["distortion"])
    return pixels.reshape(-1, 2)


def _camera(serial, size, focal, distortion):
    width, height = size
    return {"camera_serial": serial, "image_size": list(size),
            "K": [[focal, 0.0, width / 2], [0.0, focal * 1.01, height / 2], [0.0, 0.0, 1.0]],
            "distortion": distortion}


def _fk_poses(same_axis=False, repeated=False):
    if repeated:
        return [_pose([0, 0, 0], [0.035, -0.015, 0.025])] * 8
    if same_axis:
        vectors = [[0.04 * i, 0, 0] for i in range(8)]
    else:
        vectors = [[0, 0, 0], [0.24, 0.02, 0], [0, 0.22, 0.03],
                   [0.02, 0, 0.23], [0.17, 0.14, -0.08],
                   [-0.16, 0.11, 0.18], [0.13, -0.20, 0.12],
                   [-0.11, -0.13, 0.21]]
    translations = [[0.035, -0.015, 0.025], [0.08, -0.025, 0.04],
                    [-0.03, 0.045, 0.055], [0.025, 0.055, -0.01],
                    [0.065, 0.035, 0.045], [-0.025, -0.04, 0.05],
                    [0.045, -0.055, 0.015], [-0.055, 0.015, -0.02]]
    return [_pose(vector, translation) for vector, translation in zip(vectors, translations)]


def _seed_noise(transform, index):
    delta = _pose([0.001 * (index + 1), -0.0007 * (index + 1), 0.0005 * (index + 1)],
                  [0.0004 * index, -0.0002 * index, 0.0003 * index])
    return transform @ delta


def _make_arm(arm="so101_left", wrist_config=None, same_axis=False, repeated=False,
              outlier_group=None, offset=0.0, board_offset=0.0):
    board_points = shared_board._board_corners()
    main_indices = [0, 1, 2, 3, 4, 5, 7, 8, 11, 12, 15, 18, 19, 22, 23, 27]
    wrist_indices = [0, 1, 4, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 26, 27]
    main = _camera("main-serial", (1280, 960), 900.0,
                   [0.012, -0.003, 0.0004, -0.0002, 0.0])
    if wrist_config is None:
        wrist = _camera(f"wrist-{arm}-serial", (640, 480), 200.0,
                        [0.009, -0.001, 0.0002, -0.0001, 0.0, 0.0, 0.0, 0.0])
    else:
        wrist = dict(wrist_config)
    main["K"] = np.asarray(main["K"], dtype=float)
    main["distortion"] = np.asarray(main["distortion"], dtype=float)
    wrist["K"] = np.asarray(wrist["K"], dtype=float)
    wrist["distortion"] = np.asarray(wrist["distortion"], dtype=float)
    b = _pose([0.05 + offset, -0.04, 0.02], [0.05 + offset, -0.02, 0.42])
    x = _pose([0.12, 0.05 + offset / 2, -0.08], [0.025, -0.015, 0.035])
    q = _pose([0.04, -0.06, 0.02], [0.12 + board_offset, 0.025, 1.12])
    samples = []
    for index, f in enumerate(_fk_poses(same_axis=same_axis, repeated=repeated)):
        main_objects = board_points[main_indices]
        wrist_objects = board_points[wrist_indices]
        wrist_from_board = np.linalg.inv(b @ f @ x) @ q
        main_pixels = _pixels(main_objects, q, main)
        wrist_pixels = _pixels(wrist_objects, wrist_from_board, wrist)
        if index == outlier_group:
            main_pixels = main_pixels + np.array([18.0, -12.0])
        main_obs = {**main, "object_points": main_objects.tolist(),
                    "image_points": main_pixels.tolist(),
                    "T_camera_from_board_seed": _seed_noise(q, index).tolist()}
        wrist_obs = {**wrist, "object_points": wrist_objects.tolist(),
                     "image_points": wrist_pixels.tolist(),
                     "T_camera_from_board_seed": _seed_noise(wrist_from_board, index).tolist()}
        samples.append({"id": f"{arm}-frame-{index}", "pose_group": f"pose-{index}",
                        "T_base_from_gripper": f.tolist(), "main": main_obs, "wrist": wrist_obs})
    return {"samples": samples}, (b, x, q)


def _manifest(arms=None):
    if arms is None:
        arm, truth = _make_arm()
        arms = {"so101_left": arm}
    else:
        truth = None
    return {"schema_version": 1, "setup_id": "fixed-bench-2026-11",
            "setup_confirmed_unchanged": True, "board": dict(shared_board.BOARD),
            "arms": arms}, truth


class SharedBoardTests(unittest.TestCase):
    def test_synthetic_dual_camera_recovery_with_distortion_and_corner_subsets(self):
        manifest, truth = _manifest()
        report = shared_board.solve_manifest(manifest)
        arm = report["arms"]["so101_left"]
        self.assertTrue(report["accepted"], arm["reasons"])
        self.assertEqual(report["status"], "VALIDATED_SHARED_BOARD")
        self.assertEqual(arm["validation"]["method"], "leave_one_pose_group_out")
        self.assertTrue(arm["validation"]["passed"])
        self.assertTrue(arm["jacobian"]["full_rank"])
        self.assertLess(arm["training"]["per_camera"]["main"]["rms_px"], 1e-8)
        for key, expected in zip(("T_main_from_base", "T_gripper_from_wrist", "T_main_from_board"), truth):
            np.testing.assert_allclose(arm["candidate"][key], expected, atol=2e-5, rtol=0)

    def test_each_arm_can_use_its_own_wrist_camera(self):
        left, _ = _make_arm("so101_left")
        right_wrist = _camera("wrist-right-unique", (800, 1000), 200.0,
                              [0.015, -0.004, 0.0003, 0.0001, 0.0])
        right, _ = _make_arm("so101_right", wrist_config=right_wrist, offset=0.18)
        manifest, _ = _manifest({"so101_left": left, "so101_right": right})
        report = shared_board.solve_manifest(manifest)
        self.assertTrue(report["accepted"], {arm: row["reasons"] for arm, row in report["arms"].items()})
        self.assertEqual(set(report["arms"]), {"so101_left", "so101_right"})
        self.assertTrue(all(row["accepted"] for row in report["arms"].values()))
        self.assertTrue(report["cross_arm_board_pose_consistency"]["passed"])
        # The fixed main camera remains one shared identity and calibration.
        left = report["arms"]["so101_left"]
        right = report["arms"]["so101_right"]
        np.testing.assert_allclose(left["candidate"]["T_main_from_board"],
                                   right["candidate"]["T_main_from_board"], atol=2e-5, rtol=0)

    def test_cross_arm_board_pose_mismatch_rejects_both_arm_results(self):
        left, _ = _make_arm("so101_left")
        right_wrist = _camera("wrist-right-unique", (800, 1000), 200.0,
                              [0.015, -0.004, 0.0003, 0.0001, 0.0])
        right, _ = _make_arm("so101_right", wrist_config=right_wrist, offset=0.18,
                             board_offset=0.03)
        manifest, _ = _manifest({"so101_left": left, "so101_right": right})
        report = shared_board.solve_manifest(manifest)
        self.assertFalse(report["accepted"])
        consistency = report["cross_arm_board_pose_consistency"]
        self.assertTrue(consistency["evaluated"])
        self.assertFalse(consistency["passed"])
        self.assertGreater(consistency["pairs"][0]["translation_delta_m"], 0.01)
        for arm in report["arms"].values():
            self.assertFalse(arm["accepted"])
            self.assertIn("shared_main_board_pose_inconsistent_between_arms", arm["reasons"])
            self.assertNotIn("validated_transforms", arm)

    def test_nonparallel_rotation_and_distinct_pose_gates_reject_degenerate_motion(self):
        for same_axis, repeated, reason in (
                (True, False, "insufficient_nonparallel_rotation_excitation"),
                (False, True, "too_few_materially_distinct_fk_poses")):
            with self.subTest(reason=reason):
                arm, _ = _make_arm(same_axis=same_axis, repeated=repeated)
                manifest, _ = _manifest({"so101_left": arm})
                report = shared_board.solve_manifest(manifest)["arms"]["so101_left"]
                self.assertFalse(report["accepted"])
                self.assertIn(reason, report["reasons"])
                self.assertIn("candidate", report)

    def test_heldout_group_outlier_fails_prediction_gate(self):
        arm, _ = _make_arm(outlier_group=7)
        manifest, _ = _manifest({"so101_left": arm})
        report = shared_board.solve_manifest(manifest)["arms"]["so101_left"]
        self.assertFalse(report["accepted"])
        self.assertIn("leave_one_group_out_prediction_failed", report["reasons"])
        heldout = next(fold for fold in report["validation"]["folds"]
                       if fold["heldout_pose_group"] == "pose-7")
        self.assertFalse(heldout["passed"])
        self.assertGreater(heldout["prediction"]["max_frame_rms_px"], 4.0)

    def test_malformed_transforms_intrinsics_duplicate_ids_and_board_points_fail(self):
        base, _ = _manifest()
        cases = []
        bad = copy.deepcopy(base)
        bad["arms"]["so101_left"]["samples"][0]["T_base_from_gripper"][0][0] = 2.0
        cases.append((bad, "expected a proper SE"))
        bad = copy.deepcopy(base)
        del bad["arms"]["so101_left"]["samples"][0]["main"]["K"]
        cases.append((bad, "missing required fields"))
        bad = copy.deepcopy(base)
        bad["arms"]["so101_left"]["samples"][1]["id"] = bad["arms"]["so101_left"]["samples"][0]["id"]
        cases.append((bad, "duplicate sample id"))
        bad = copy.deepcopy(base)
        bad["arms"]["so101_left"]["samples"][0]["wrist"]["object_points"][0][0] += 0.001
        cases.append((bad, "24 mm board internal corner"))
        bad = copy.deepcopy(base)
        bad["arms"]["so101_left"]["samples"][0]["main"]["corner_ids"] = [1] * 16
        cases.append((bad, "IDs must match"))
        for manifest, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                shared_board.solve_manifest(manifest)

    def test_main_and_wrist_cannot_claim_same_physical_camera(self):
        manifest, _ = _manifest()
        manifest["arms"]["so101_left"]["samples"][0]["wrist"]["camera_serial"] = "main-serial"
        with self.assertRaisesRegex(ValueError, "distinct serial identities"):
            shared_board.validate_manifest(manifest)

    def test_cli_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("keep", encoding="utf-8")
            with patch("sys.stderr"):
                code = shared_board.main(["--manifest", str(Path(directory) / "missing.json"),
                                          "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_json_duplicate_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            json.loads('{"schema_version": 1, "schema_version": 2}',
                       object_pairs_hook=shared_board._unique_json_object)


if __name__ == "__main__":
    unittest.main()
