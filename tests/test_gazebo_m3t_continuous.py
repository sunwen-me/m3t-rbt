#!/usr/bin/env python3
"""Bounded feedback-source checks; no Gazebo, hardware, or child processes."""

from collections import defaultdict
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_gazebo_m3t_continuous as servo
import verify_m3t_position_ik as native_ik
from prepare_gazebo_m3t_scene import CANONICAL_URDF, INITIAL_JOINTS, add_camera


class Kinematics:
    """Nonlinear FK/J stand-in so isolation tests need no installed SDK."""

    def __init__(self):
        self.solver = self

    def forward(self, degrees):
        q = np.deg2rad(degrees)
        pose = np.eye(4)
        pose[:3, 3] = [0.1 * math.sin(q[0]) + 0.2 * math.cos(q[1]),
                       0.3 * math.sin(q[1]) + 0.1 * math.sin(q[2]),
                       0.2 * math.cos(q[2]) + 0.1 * math.sin(q[3])]
        return pose

    def _fk_jac(self, degrees, _with_jacobian):
        q = np.deg2rad(degrees)
        jac = np.zeros((6, 5))
        jac[0, 0], jac[0, 1] = 0.1 * math.cos(q[0]), -0.2 * math.sin(q[1])
        jac[1, 1], jac[1, 2] = 0.3 * math.cos(q[1]), 0.1 * math.cos(q[2])
        jac[2, 2], jac[2, 3] = -0.2 * math.sin(q[2]), 0.1 * math.cos(q[3])
        return self.forward(degrees), jac


class Journal:
    def __init__(self, events=()):
        self.records = defaultdict(list, estimate=list(events))

    def update(self):
        pass


class NativeSampleTest(unittest.TestCase):
    def event(self, *, mount=True, sequence=1, shift=0.0):
        angles = {joint: .1 * index for index, joint in enumerate(native_ik.TRACKED_JOINTS)}
        event = {
            "event": "estimate", "schema_version": 1, "frame_id": native_ik.CAMERA_FRAME,
            "encoder_reference_links": True, "joints_locked_to_feedback": False,
            "frame_sequence": sequence, "color_capture_ns": 2_000_000_000 + sequence,
            "capture_clock_kind": "HostMonotonic", "capture_clock_source": "01" * 16,
            "capture_clock_generation": 0, "joint_angles_rad": {}, "links": [],
            "joint_priors": {
                "enabled": True, "status": "applied", "time_basis": "received",
                "capture_time_associated": False, "rejected_mappings": [], "prior_count": 12,
                "age_ns": 20_000_000, "sequence": sequence, "controller_id": "readonly-controller",
                "driver_session": "recorded-session", "priors": [],
            },
        }
        diagnostics = []
        for index, arm in enumerate(native_ik.ARMS):
            nominal = np.eye(4)
            nominal[:3, 3] = [.1 * index, -.2, .8]
            base = nominal.copy()
            if mount:
                angle = .06 + shift
                base[:3, :3] = [[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                base[:3, 3] += [.015 + shift, -.01, .005]
            event["joint_angles_rad"][arm] = {joint: value + .01 for joint, value in angles.items()}
            event["joint_priors"]["priors"].extend(
                {"arm": arm, "joint": joint, "angle_rad": value} for joint, value in angles.items())
            gripper = base @ Kinematics().forward(
                np.rad2deg([event["joint_angles_rad"][arm][joint] for joint in native_ik.JOINTS]))
            event["links"].extend(
                {"arm": arm, "link": name, "frame_id": f"{arm}/{name}", "pose": pose.ravel().tolist()}
                for name, pose in (("base_link", base), (native_ik.TOOL_LINK, gripper)))
            diagnostics.append({
                "arm": arm, "nominal_base_in_camera": nominal.ravel().tolist(),
                "refined_base_in_camera": base.ravel().tolist(),
                "translation_correction_m": float(np.linalg.norm(base[:3, 3] - nominal[:3, 3])),
                "rotation_correction_rad": .06 + shift,
            })
        if mount:
            event["mount_refinement"] = {"enabled": True, "joints_locked_to_feedback": False,
                                          "arms": diagnostics}
        return event

    def test_combined_mode_keeps_actual_moving_base_and_visual_joint_residuals(self):
        for arm in native_ik.ARMS:
            with self.subTest(arm=arm):
                event = self.event()
                sample = native_ik.native_sample(event, arm)
                diagnostic = next(entry for entry in event["mount_refinement"]["arms"] if entry["arm"] == arm)
                np.testing.assert_array_equal(sample.camera_from_base.ravel(), diagnostic["refined_base_in_camera"])
                self.assertFalse(np.array_equal(sample.camera_from_base.ravel(), diagnostic["nominal_base_in_camera"]))
                np.testing.assert_allclose(sample.visual_deg - sample.encoder_deg, np.rad2deg(.01), atol=1e-14)
                expected = sample.camera_from_base @ Kinematics().forward(sample.visual_deg)
                np.testing.assert_allclose(sample.camera_from_gripper, expected, atol=1e-15)

    def test_legacy_encoder_estimates_stay_accepted_and_locked_flags_are_rejected(self):
        event = self.event(mount=False)
        event.pop("joints_locked_to_feedback")
        native_ik.native_sample(event, native_ik.ARMS[0])
        for locked in (True, 0, "false", None):
            with self.subTest(locked=locked), self.assertRaises(ValueError):
                native_ik.native_sample({**event, "joints_locked_to_feedback": locked}, native_ik.ARMS[0])

    def test_rejects_incomplete_locked_contradictory_and_nonrigid_mount_diagnostics(self):
        edits = [
            lambda e: e.update(encoder_reference_links=False),
            lambda e: e.update(joints_locked_to_feedback=True),
            lambda e: e.update(mount_refinement=None),
            lambda e: e["mount_refinement"].update(enabled=False),
            lambda e: e["mount_refinement"].update(joints_locked_to_feedback=True),
            lambda e: e["mount_refinement"].update(joints_locked_to_feedback=0),
            lambda e: e["mount_refinement"].pop("joints_locked_to_feedback"),
            lambda e: e["mount_refinement"]["arms"].pop(),
            lambda e: e["mount_refinement"]["arms"].__setitem__(1, e["mount_refinement"]["arms"][0]),
            lambda e: e["mount_refinement"]["arms"][0].update(arm="unknown"),
            lambda e: e["mount_refinement"]["arms"][0]["refined_base_in_camera"].__setitem__(3, .9),
            lambda e: e["links"].pop(2),  # Other arm's base must be validated too.
            lambda e: e["links"][2].update(frame_id="wrong/base_link"),
            lambda e: e["links"].append(copy.deepcopy(e["links"][2])),
        ]
        for key in ("nominal_base_in_camera", "refined_base_in_camera"):
            edits.append(lambda e, key=key: e["mount_refinement"]["arms"][0].pop(key))
            edits.append(lambda e, key=key: e["mount_refinement"]["arms"][0].update({key: [1] * 15}))
            for position, value in ((0, float("nan")), (3, float("inf")), (0, True),
                                    (0, 2.0), (4, .25), (15, 0.0)):
                edits.append(lambda e, key=key, position=position, value=value:
                             e["mount_refinement"]["arms"][0][key].__setitem__(position, value))
        for key in ("translation_correction_m", "rotation_correction_rad"):
            edits.append(lambda e, key=key: e["mount_refinement"]["arms"][0].pop(key))
            for value in (float("nan"), float("inf"), True, -.01, 0.0, .9):
                edits.append(lambda e, key=key, value=value:
                             e["mount_refinement"]["arms"][0].update({key: value}))
        for index, edit in enumerate(edits):
            with self.subTest(case=index):
                event = self.event()
                edit(event)
                with self.assertRaises(ValueError):
                    native_ik.native_sample(event, native_ik.ARMS[0])

    def test_recording_retains_each_frames_current_refined_base(self):
        events = [self.event(sequence=1), self.event(sequence=2, shift=.02)]
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "native.jsonl"
            recording.write_text("\n".join(json.dumps(event) for event in events) + "\n")
            samples = native_ik.load_recording(recording, native_ik.ARMS[0])
        self.assertEqual(len(samples), 2)
        self.assertFalse(np.array_equal(samples[0].camera_from_base, samples[1].camera_from_base))
        for sample, event in zip(samples, events):
            np.testing.assert_array_equal(sample.camera_from_base.ravel(), event["links"][0]["pose"])


class FeedbackSourceTest(unittest.TestCase):
    def setUp(self):
        self.kinematics = Kinematics()
        self.base = add_camera(ET.Element("world"))
        self.metadata = {"arm": "so101_left", "camera_frame": "gazebo/color_optical",
                         "optical_from_base": self.base.ravel().tolist()}
        self.q = np.array([INITIAL_JOINTS[name] for name in servo.JOINTS])
        self.args = SimpleNamespace(period_sec=16.0, amplitude_x_m=0.010, amplitude_y_m=0.005)
        self.origin = (self.base @ self.kinematics.forward(np.rad2deg(self.q)))[:3, 3]

    def event(self, *, capture=2_000_000_000, sequence=1, prior_age=20_000_000, bias=None):
        q_encoder = self.q.copy()
        if bias is not None:
            q_encoder += bias
        angles = dict(zip(servo.JOINTS, self.q + 0.01))
        angles["gripper"] = INITIAL_JOINTS["gripper"]
        return {"event": "estimate", "schema_version": 1, "frame_id": self.metadata["camera_frame"],
                "capture_clock_kind": "Device", "capture_clock_source": "synthetic-clock",
                "capture_clock_generation": 0, "frame_sequence": sequence,
                "color_capture_ns": capture, "capture_ns": capture, "pair_skew_ns": 0,
                "encoder_reference_links": True,
                "joint_angles_rad": {"so101_left": angles},
                "links": [{"arm": "so101_left", "link": name, "pose": pose.ravel().tolist()}
                          for name, pose in (("base_link", self.base), (servo.TOOL_LINK, np.eye(4)))],
                "joint_priors": {"enabled": True, "status": "applied", "time_basis": "capture",
                                 "capture_time_associated": True, "controller_id": "gazebo-so101",
                                 "driver_session": "synthetic-session", "prior_count": 6,
                                 "rejected_mappings": [], "age_ns": prior_age, "sequence": sequence,
                                 "priors": [{"arm": "so101_left", "joint": name, "angle_rad": float(value)}
                                            for name, value in zip((*servo.JOINTS, "gripper"),
                                                                   (*q_encoder, INITIAL_JOINTS["gripper"]))]}}

    def consumed(self, event, source="encoder", now=2_100_000_000):
        reader = servo.VisionReader(Journal([event]), self.metadata, self.kinematics, io.StringIO(), source)
        return reader.consume(now, 1_000_000_000, self.origin, self.args)

    def command(self, selected, now=2_100_000_000):
        age = servo.feedback_sample_age(selected, now)
        _, feedforward = servo.target_at(self.origin, now, 1_000_000_000, self.args)
        return servo.resolved_rate(selected["jacobian"],
                                   feedforward + np.asarray(selected["feedback_correction_camera_m_s"]),
                                   selected["q_feedback_rad"], -np.ones(5) * 3, np.ones(5) * 3,
                                   age / 1e9)[0]

    def test_encoder_command_is_independent_of_visual_joints_and_poses(self):
        original = self.event()
        tampered = copy.deepcopy(original)
        tampered["joint_angles_rad"]["so101_left"].update(dict(zip(servo.JOINTS, self.q + 2.0)))
        changed_base = np.eye(4)
        changed_base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        changed_base[:3, 3] = [10, -20, 30]
        tampered["links"][0]["pose"] = changed_base.ravel().tolist()
        tampered["links"][1]["pose"] = changed_base.ravel().tolist()
        first, second = self.consumed(original), self.consumed(tampered)
        for field in ("q_feedback_rad", "feedback_camera_xyz_m", "jacobian", "feedback_correction_camera_m_s"):
            np.testing.assert_array_equal(first[field], second[field])
        np.testing.assert_array_equal(self.command(first), self.command(second))
        self.assertNotEqual(first["q_visual_rad"], second["q_visual_rad"])
        self.assertNotEqual(first["visual_camera_xyz_m"], second["visual_camera_xyz_m"])

    def test_encoder_bias_is_preserved_in_fk_and_jacobian(self):
        bias = np.array([0.0, math.radians(2), 0.0, 0.0, 0.0])
        original = self.consumed(self.event())
        biased = self.consumed(self.event(bias=bias))
        np.testing.assert_allclose(biased["q_feedback_rad"], self.q + bias, atol=1e-15)
        self.assertFalse(np.allclose(original["feedback_camera_xyz_m"], biased["feedback_camera_xyz_m"]))
        self.assertFalse(np.allclose(original["jacobian"], biased["jacobian"]))
        self.assertFalse(np.allclose(self.command(original), self.command(biased)))

    def test_default_m3t_feedback_matches_previous_fk_jacobian_and_command(self):
        event = self.event()
        visual, _encoder, base, _gripper = servo.sample_estimate(event, self.metadata)
        reader = servo.VisionReader(Journal([event]), self.metadata, self.kinematics, io.StringIO())
        selected = reader.consume(2_100_000_000, 1_000_000_000, self.origin, self.args)
        xyz = (base @ self.kinematics.forward(visual))[:3, 3]
        jac = servo.camera_jacobian(self.kinematics, visual, base)
        np.testing.assert_array_equal(selected["q_feedback_rad"], np.deg2rad(visual))
        np.testing.assert_array_equal(selected["feedback_camera_xyz_m"], xyz)
        np.testing.assert_array_equal(selected["jacobian"], jac)
        desired, _ = servo.target_at(self.origin, event["color_capture_ns"], 1_000_000_000, self.args)
        _, feedforward = servo.target_at(self.origin, 2_100_000_000, 1_000_000_000, self.args)
        previous, _, _ = servo.resolved_rate(jac, feedforward + servo.GAIN_PER_S * (desired - xyz),
                                            np.deg2rad(visual), -np.ones(5) * 3, np.ones(5) * 3, 0.1)
        np.testing.assert_array_equal(self.command(selected), previous)
        self.assertEqual(selected["feedback_source"], "m3t")
        self.assertEqual(selected["feedback_sample_ns"], event["color_capture_ns"])

    def test_m3t_feedback_uses_current_refined_base_for_fk_and_jacobian(self):
        event = self.event()
        refined = self.base.copy()
        angle = .15
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                             [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        refined[:3, :3] = rotation @ refined[:3, :3]
        refined[:3, 3] += [.015, -.01, .005]
        event["links"][0]["pose"] = refined.ravel().tolist()
        event["mount_refinement"] = {"enabled": True, "joints_locked_to_feedback": False, "arms": [{
            "arm": "so101_left", "nominal_base_in_camera": self.base.ravel().tolist(),
            "refined_base_in_camera": refined.ravel().tolist(),
            "translation_correction_m": float(np.linalg.norm(refined[:3, 3] - self.base[:3, 3])),
            "rotation_correction_rad": angle,
        }]}
        selected = self.consumed(event, "m3t")
        visual = np.rad2deg([event["joint_angles_rad"]["so101_left"][joint] for joint in servo.JOINTS])
        np.testing.assert_allclose(selected["feedback_camera_xyz_m"],
                                   (refined @ self.kinematics.forward(visual))[:3, 3])
        np.testing.assert_allclose(selected["jacobian"], servo.camera_jacobian(self.kinematics, visual, refined))
        self.assertFalse(np.allclose(selected["jacobian"],
                                     servo.camera_jacobian(self.kinematics, visual, self.base)))

    def test_encoder_sample_time_sets_error_time_and_staleness(self):
        event = self.event()
        selected = self.consumed(event)
        self.assertEqual(selected["feedback_sample_ns"], 1_980_000_000)
        self.assertEqual(selected["feedback_sample_age_ns"], 120_000_000)
        expected, _ = servo.target_at(self.origin, 1_980_000_000, 1_000_000_000, self.args)
        np.testing.assert_array_equal(selected["desired_camera_xyz_m"], expected)
        self.assertEqual(servo.feedback_sample_age(selected, 2_480_000_000), 500_000_000)
        for now in (1_979_999_999, 2_480_000_001):
            with self.assertRaises(RuntimeError):
                servo.feedback_sample_age(selected, now)
        self.assertIsNone(self.consumed(event, now=2_490_000_000))
        self.assertIsNotNone(self.consumed(event, "m3t", now=2_490_000_000))

    def test_invalid_and_regressing_prior_sample_time_is_rejected(self):
        for age in (-1, True, 40_000_001):
            with self.assertRaises(ValueError):
                self.consumed(self.event(prior_age=age))
        first = self.event()
        second = self.event(capture=2_001_000_000, sequence=2, prior_age=40_000_000)
        reader = servo.VisionReader(Journal([first, second]), self.metadata, self.kinematics, io.StringIO(), "encoder")
        with self.assertRaisesRegex(ValueError, "sample time regressed"):
            reader.consume(2_100_000_000)
        future = servo.VisionReader(Journal([first]), self.metadata, self.kinematics, io.StringIO(), "encoder")
        self.assertIsNone(future.consume(1_999_999_999))
        self.assertEqual(future.index, 0)

    def test_command_trace_keeps_visual_and_feedback_sources_distinct(self):
        selected = self.consumed(self.event())
        journal = Journal()
        result = {"command_trace": [], "feedback_source": "encoder"}
        stream = servo.CommandStream(io.StringIO(), journal, io.StringIO(), result)
        dq = self.command(selected)
        command_id = stream.send(2_100_000_000, dq, selected)
        journal.records["command"].append({"command_id": command_id, "stamp_ns": 2_101_000_000,
                                            "command_mode": "velocity", "dq_rad_s": dq.tolist(),
                                            "actuator_values": dq.tolist()})
        stream.acknowledge()
        row = result["command_trace"][0]
        self.assertEqual(row["feedback_source"], "encoder")
        self.assertEqual(row["source_capture_ns"], 2_000_000_000)
        self.assertEqual(row["feedback_sample_ns"], 1_980_000_000)
        self.assertEqual(row["feedback_sample_age_ns"], 120_000_000)
        self.assertEqual(row["actual_feedback_sample_age_ns"], 121_000_000)
        self.assertEqual(row["actual_publish_ns"], 2_101_000_000)
        self.assertNotEqual(row["q_visual_rad"], row["q_feedback_rad"])
        self.assertEqual(row["prior_sequence"], 1)

    def test_cli_feedback_source_defaults_to_m3t(self):
        required = ["--robot-model", "r", "--urdf", "u", "--build-dir", "b", "--composer", "c", "--output-dir", "o"]
        for extra, expected in (([], "m3t"), (["--feedback-source", "encoder"], "encoder")):
            with patch.object(sys, "argv", ["servo", *required, *extra]), patch.object(servo, "run", return_value=0) as run:
                self.assertEqual(servo.main(), 0)
                self.assertEqual(run.call_args.args[0].feedback_source, expected)

    def test_cli_tracker_warmup_timeout_defaults_and_is_passed_to_run(self):
        required = ["--robot-model", "r", "--urdf", "u", "--build-dir", "b", "--composer", "c", "--output-dir", "o"]
        for extra, expected in (([], 90.0), (["--tracker-warmup-timeout-sec", "180.25"], 180.25)):
            with patch.object(sys, "argv", ["servo", *required, *extra]), patch.object(servo, "run", return_value=0) as run:
                self.assertEqual(servo.main(), 0)
                self.assertEqual(run.call_args.args[0].tracker_warmup_timeout_sec, expected)

    def test_cli_tracker_warmup_timeout_rejects_nonpositive_and_nonfinite(self):
        required = ["--robot-model", "r", "--urdf", "u", "--build-dir", "b", "--composer", "c", "--output-dir", "o"]
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                with patch.object(sys, "argv", ["servo", *required, f"--tracker-warmup-timeout-sec={value}"]), \
                        patch.object(servo, "run") as run, patch.object(sys, "stderr", io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as raised:
                        servo.main()
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("--tracker-warmup-timeout-sec must be positive and finite", error.getvalue())
                    run.assert_not_called()

    def test_offline_feedback_truth_uses_prior_sample_time(self):
        metadata = {**self.metadata, "optical_from_base": np.eye(4).ravel().tolist()}
        events = [self.event(capture=2_000_000_000), self.event(capture=2_100_000_000, sequence=2)]
        gazebo, tracker = Journal(), Journal(events)
        gazebo.channel, tracker.channel = io.StringIO(), io.StringIO()
        for stamp in range(1_960_000_000, 2_140_000_001, 20_000_000):
            gazebo.records["pose"].append({"stamp_ns": stamp, "base_xyz_m": [0, 0, 0],
                                           "base_quat_xyzw": [0, 0, 0, 1],
                                           "gripper_xyz_m": [stamp / 1e10, 0, 0]})
            gazebo.records["state"].append({"stamp_ns": stamp, "q_rad": self.q.tolist(),
                                            "dq_rad_s": [0.01] * 5})
        args = SimpleNamespace(**vars(self.args), feedback_source="encoder")
        result = {"motion_start_ns": 1_990_000_000, "motion_end_requested_ns": 2_110_000_000,
                  "target_origin_camera_xyz_m": [0, 0, 0], "command_trace": []}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(servo, "Journal", side_effect=[gazebo, tracker]):
                servo.offline_analysis(Path(directory), result, metadata, self.kinematics, args)
            rows = [json.loads(line) for line in (Path(directory) / "offline_measurements.jsonl").read_text().splitlines()]
        self.assertTrue(result["measurement_complete"])
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["feedback_sample_ns"], row["capture_ns"] - 20_000_000)
            self.assertAlmostEqual(row["feedback_truth_camera_xyz_m"][0], row["feedback_sample_ns"] / 1e10)
            self.assertAlmostEqual(row["truth_camera_xyz_m"][0] - row["feedback_truth_camera_xyz_m"][0], .002)
            expected = np.linalg.norm(np.asarray(row["feedback_camera_xyz_m"]) - row["feedback_truth_camera_xyz_m"]) * 1000
            self.assertAlmostEqual(row["feedback_to_truth_error_mm"], expected)

    def test_installed_sdk_encoder_bias_changes_fk_and_jacobian(self):
        try:
            from so101_control.kinematics import ArmKinematics
        except ImportError:
            self.skipTest("installed SO101 SDK unavailable; run with Gazebo IK venv and PYTHONPATH")
        self.kinematics = ArmKinematics(str(CANONICAL_URDF), target_frame=servo.TOOL_LINK, joint_names=servo.JOINTS)
        first = self.consumed(self.event())
        bias = np.array([0.0, math.radians(2), 0.0, 0.0, 0.0])
        second = self.consumed(self.event(bias=bias))
        self.assertGreater(np.linalg.norm(np.asarray(first["feedback_camera_xyz_m"]) - second["feedback_camera_xyz_m"]), 1e-4)
        self.assertGreater(np.max(np.abs(first["jacobian"] - second["jacobian"])), 1e-4)
        self.assertGreater(np.max(np.abs(self.command(first) - self.command(second))), 1e-4)
        self.test_encoder_command_is_independent_of_visual_joints_and_poses()


if __name__ == "__main__":
    unittest.main()
