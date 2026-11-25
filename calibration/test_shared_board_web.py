"""Offline acceptance checks for the shared-board operator dashboard helpers."""
import email.message
import http.client
import io
import json
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import shared_board_web as web
from wrist_intrinsics import DEFAULT_BOARD, CharucoDetector


class PacketTests(unittest.TestCase):
    WIDTH = 640
    HEIGHT = 480

    def setUp(self):
        self.now_ns = 20_000_000_000
        self.source = "01" + "00" * 47
        self.session = {
            "schema": "rbt-stereo-capture-v1",
            "max_skew_ns": 20_000_000,
            "reference": {"width": self.WIDTH, "height": self.HEIGHT},
            "secondary": {"width": self.WIDTH, "height": self.HEIGHT},
        }

    def packet_bytes(self, *, ref_ns=None, sec_ns=None, ref_clock=None,
                     sec_clock=None, ref_generation=0, sec_generation=0,
                     padding=0):
        ref_ns = self.now_ns - 100_000_000 if ref_ns is None else ref_ns
        sec_ns = self.now_ns - 90_000_000 if sec_ns is None else sec_ns
        header = {
            "schema": "rbt-stereo-preview-v1",
            "reference": {
                "capture_ns": ref_ns, "clock_kind": "HostMonotonic",
                "clock_generation": ref_generation,
                "clock_source_hex": ref_clock or self.source,
                "sequence": 1,
            },
            "secondary": {
                "capture_ns": sec_ns, "clock_kind": "HostMonotonic",
                "clock_generation": sec_generation,
                "clock_source_hex": sec_clock or self.source,
                "sequence": 2,
            },
            "skew_ns": abs(ref_ns - sec_ns),
        }
        line = json.dumps(header, separators=(",", ":")).encode()
        line += b" " * padding
        rgb = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        rgb[0, 0] = (11, 23, 47)
        other = np.zeros_like(rgb)
        return line + b"\n" + rgb.tobytes() + other.tobytes()

    def test_packet_rgb_is_copied_and_reordered_to_bgr(self):
        packet, bgr = web.parse_packet(self.packet_bytes(), self.session, self.now_ns)
        self.assertEqual(packet["schema"], "rbt-stereo-preview-v1")
        self.assertEqual(bgr.shape, (self.HEIGHT, self.WIDTH, 3))
        self.assertEqual(bgr[0, 0].tolist(), [47, 23, 11])
        self.assertTrue(bgr.flags["OWNDATA"])

    def test_rejects_truncated_and_trailing_packet_bytes(self):
        payload = self.packet_bytes()
        for damaged in (payload[:-1], payload + b"\0"):
            with self.subTest(length=len(damaged)):
                with self.assertRaises(ValueError):
                    web.parse_packet(damaged, self.session, self.now_ns)

    def test_rejects_stale_future_and_mismatched_clock_packets(self):
        cases = (
            self.packet_bytes(ref_ns=self.now_ns - 1_600_000_000,
                              sec_ns=self.now_ns - 1_590_000_000),
            self.packet_bytes(ref_ns=self.now_ns + 1,
                              sec_ns=self.now_ns + 10_000_001),
            self.packet_bytes(ref_clock=self.source,
                              sec_clock="02" + "00" * 47),
            self.packet_bytes(ref_generation=1, sec_generation=0),
        )
        for payload in cases:
            with self.subTest(header=payload[:100]):
                with self.assertRaises(ValueError):
                    web.parse_packet(payload, self.session, self.now_ns)

    def test_rejects_unsupported_clock_skew_and_oversized_header(self):
        payload = self.packet_bytes()
        header, _, body = payload.partition(b"\n")
        value = json.loads(header)
        value["reference"]["clock_kind"] = "Device"
        wrong_clock = json.dumps(value, separators=(",", ":")).encode() + b"\n" + body
        with self.assertRaises(ValueError):
            web.parse_packet(wrong_clock, self.session, self.now_ns)

        value["reference"]["clock_kind"] = "HostMonotonic"
        value["skew_ns"] += 1
        wrong_skew = json.dumps(value, separators=(",", ":")).encode() + b"\n" + body
        with self.assertRaises(ValueError):
            web.parse_packet(wrong_skew, self.session, self.now_ns)

        with self.assertRaises(ValueError):
            web.parse_packet(self.packet_bytes(padding=8193), self.session, self.now_ns)


class ForwardKinematicsTests(unittest.TestCase):
    @staticmethod
    def origin(translation, euler):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler("xyz", euler).as_matrix()
        result[:3, 3] = translation
        return result

    def model(self, limits=(-3.0, 3.0)):
        axes = ([0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
        joints = []
        parent = "base_link"
        for index, name in enumerate(web.NAMES):
            child = "gripper_link" if index == 4 else f"link_{index}"
            origin = self.origin(
                [0.03 + index * 0.01, -0.02 + index * 0.003, 0.04],
                [0.03 * (index + 1), -0.02 * index, 0.01 * index],
            )
            joints.append({
                "name": name, "type": "revolute", "parent": parent,
                "child": child, "origin": origin.reshape(-1).tolist(),
                "axis": axes[index], "lower": limits[0], "upper": limits[1],
            })
            parent = child
        return {"root_link": "base_link", "joints": joints}

    def test_fk_composes_nonidentity_origins_and_joint_rotations(self):
        model = self.model()
        degrees = [31.0, -22.0, 14.0, 8.0, -17.0]
        expected = np.eye(4)
        for joint in model["joints"]:
            expected = expected @ np.asarray(joint["origin"]).reshape(4, 4)
            angle = np.deg2rad(degrees[web.NAMES.index(joint["name"])])
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(np.asarray(joint["axis"]) * angle).as_matrix()
            expected = expected @ motion
        actual = web.fk(model, degrees)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        self.assertGreater(np.linalg.norm(actual[:3, 3]), 0.05)
        self.assertGreater(np.linalg.norm(actual[:3, :3] - np.eye(3)), 0.1)

    def test_fk_rejects_joint_limits_and_bad_origins(self):
        with self.assertRaises(ValueError):
            web.fk(self.model(limits=(-1.0, 1.0)), [80, 0, 0, 0, 0])
        model = self.model()
        model["joints"][0]["origin"][0] = float("nan")
        with self.assertRaises(ValueError):
            web.fk(model, [0, 0, 0, 0, 0])


class PoseObservationTests(unittest.TestCase):
    def setUp(self):
        self.detector = CharucoDetector(DEFAULT_BOARD)
        self.lattice = self.detector.board_corners
        self.config = {
            "K": [[620.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]],
            "distortion": [0.0] * 8,
            "image_size": [640, 480],
            "camera_serial": "test-camera",
        }
        self.rvec = np.array([[0.16], [-0.11], [0.04]], dtype=float)
        self.tvec = np.array([[-0.055], [-0.085], [0.68]], dtype=float)
        projected = cv2.projectPoints(
            self.lattice, self.rvec, self.tvec,
            np.asarray(self.config["K"], float),
            np.asarray(self.config["distortion"], float),
        )[0]
        self.pixels = projected.reshape(-1, 2)

    def test_synthetic_charuco_points_produce_a_valid_pnp_seed(self):
        ids = np.arange(len(self.lattice), dtype=int)
        observation = web.pose_observation(self.config, ids, self.pixels, self.lattice)
        self.assertLess(observation["pnp_rms_px"], 1e-3)
        np.testing.assert_allclose(observation["object_points"], self.lattice, atol=1e-12)
        np.testing.assert_allclose(observation["T_camera_from_board_seed"][:3][0][:3],
                                   cv2.Rodrigues(self.rvec)[0][0], atol=1e-5)
        self.assertEqual(observation["corner_ids"], ids.tolist())

    def test_pose_observation_requires_six_noncollinear_corners(self):
        with self.assertRaises(ValueError):
            web.pose_observation(self.config, np.arange(5), self.pixels[:5], self.lattice)
        observation = web.pose_observation(
            self.config, np.arange(6), self.pixels[:6], self.lattice)
        self.assertEqual(len(observation["corner_ids"]), 6)


class WristIdentityAndReadinessTests(unittest.TestCase):
    CAMERA_CONFIG = {
        "K": [[620.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]],
        "distortion": [0.0] * 8,
        "image_size": [640, 480],
        "camera_serial": "stable-serial-123",
    }
    DEVICE_URI = "v4l2://by-id/usb-Example_Camera_123"

    def dashboard_for_wrist(self):
        dashboard = web.Dashboard.__new__(web.Dashboard)
        dashboard.adopted = {"contents": {
            "board": DEFAULT_BOARD,
            "board_geometry_sha256": "board-hash",
            "focus_value": 42.0,
            "device_uri": self.DEVICE_URI,
            "camera_serial": self.CAMERA_CONFIG["camera_serial"],
            "image_size": [640, 480],
            "solver_camera_config": self.CAMERA_CONFIG,
        }}
        return dashboard

    def observation(self, **changes):
        value = {
            "board": DEFAULT_BOARD,
            "board_geometry_sha256": "board-hash",
            "focus": {"locked": True, "autofocus": 0, "focus_value": 42.0},
            "device_binding": {
                "device_uri": self.DEVICE_URI,
                "camera_serial": self.CAMERA_CONFIG["camera_serial"],
                "resolved_device_node": "/dev/video2",
            },
            "width": 640,
            "height": 480,
            "read_completed_monotonic_seconds": time.monotonic(),
        }
        value.update(changes)
        return value

    def test_ephemeral_video_node_change_is_accepted_for_same_stable_camera(self):
        dashboard = self.dashboard_for_wrist()
        observation = self.observation(device_binding={
            "device_uri": self.DEVICE_URI,
            "camera_serial": self.CAMERA_CONFIG["camera_serial"],
            "resolved_device_node": "/dev/video3",
        })
        self.assertEqual(dashboard.verify_wrist(observation), self.CAMERA_CONFIG)

    def test_stable_camera_identity_path_focus_resolution_and_freshness_are_enforced(self):
        dashboard = self.dashboard_for_wrist()
        cases = (
            ("path", {"device_binding": {"device_uri": "v4l2://by-id/usb-Other_Camera_456",
                                           "camera_serial": self.CAMERA_CONFIG["camera_serial"],
                                           "resolved_device_node": "/dev/video2"}}),
            ("serial", {"device_binding": {"device_uri": self.DEVICE_URI,
                                             "camera_serial": "different-serial",
                                             "resolved_device_node": "/dev/video2"}}),
            ("focus", {"focus": {"locked": True, "autofocus": 0, "focus_value": 43.0}}),
            ("resolution", {"width": 1280}),
            ("stale", {"read_completed_monotonic_seconds": time.monotonic() - 1.1}),
        )
        for label, changes in cases:
            with self.subTest(rejected_change=label):
                with self.assertRaises(ValueError):
                    dashboard.verify_wrist(self.observation(**changes))

    @staticmethod
    def dashboard_for_state(frame_age):
        dashboard = web.Dashboard.__new__(web.Dashboard)
        dashboard.args = SimpleNamespace(arm="so101_left")
        dashboard.feedback = "test"
        dashboard.manifest = {"arms": {"so101_left": {"samples": []}}}
        dashboard.adopted = None
        dashboard.solving = False
        dashboard.error = None
        dashboard.report = None
        dashboard.main_frame = lambda: ({"reference": {"sequence": 17}}, np.zeros((2, 2, 3), dtype=np.uint8))
        dashboard.detect = lambda _image: (None, [1, 2, 3], None)
        dashboard.child = lambda _path: {
            "frame_id": 991,
            "frame_age_seconds": frame_age,
            "error": None,
        }
        return dashboard

    def test_wrist_readiness_requires_fresh_age_even_with_a_frame_id(self):
        stale = self.dashboard_for_state(1.01).state()
        self.assertEqual(stale["wrist_ready"], False)
        self.assertEqual(stale["wrist_error"], "画面已过期，正在等待相机恢复")

        fresh = self.dashboard_for_state(0.25).state()
        self.assertEqual(fresh["wrist_ready"], True)
        self.assertIsNone(fresh["wrist_error"])


class MutationSecurityTests(unittest.TestCase):
    def test_mutating_route_rejects_missing_csrf_without_touching_hardware(self):
        dashboard = SimpleNamespace(args=SimpleNamespace(public_origin=""))
        token = "test-only-token"
        server = web.Server(("127.0.0.1", 0), web.handler(dashboard, token))
        dashboard.args.public_origin = f"http://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request(
                "POST", "/api/handeye/capture",
                body=json.dumps({"fixed": True}),
                headers={
                    "Host": f"127.0.0.1:{server.server_port}",
                    "Origin": dashboard.args.public_origin,
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            self.assertIn("校验令牌", response.read().decode())
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class ProxyFailureTests(unittest.TestCase):
    class BrokenWriter:
        def __init__(self):
            self.calls = 0

        def write(self, _payload):
            self.calls += 1
            raise BrokenPipeError("test client disconnected")

    class RecordingWriter:
        def __init__(self):
            self.payloads = []

        def write(self, payload):
            self.payloads.append(payload)
            return len(payload)

    @staticmethod
    def bare_handler(dashboard, token, *, path, body=b"{}", host=None,
                     origin=None, writer=None):
        cls = web.handler(dashboard, token)
        instance = cls.__new__(cls)
        instance.path = path
        instance.close_connection = False
        instance.rfile = io.BytesIO(body)
        instance.wfile = writer or ProxyFailureTests.RecordingWriter()
        instance.headers = email.message.Message()
        instance.headers["Content-Type"] = "application/json"
        instance.headers["Content-Length"] = str(len(body))
        if host is not None:
            instance.headers["Host"] = host
        if origin is not None:
            instance.headers["Origin"] = origin
        instance.headers["X-CSRF-Token"] = token
        statuses = []
        instance.send_response = lambda status, *args: statuses.append(status)
        instance.send_header = lambda *args: None
        instance.end_headers = lambda: None
        return instance, statuses

    def test_disconnected_client_does_not_trigger_second_response_or_replay(self):
        actions = []
        dashboard = SimpleNamespace(
            args=SimpleNamespace(public_origin="http://127.0.0.1:18898"),
            lock=threading.Lock(),
            check_arm=lambda: (actions.append("check") or {"message": "read complete"}),
        )
        h, statuses = self.bare_handler(
            dashboard, "session-token", path="/api/handeye/check-arm",
            host="127.0.0.1:18898", origin="http://127.0.0.1:18898",
            writer=self.BrokenWriter(),
        )

        h.do_POST()

        self.assertEqual(actions, ["check"])
        self.assertEqual(statuses, [200])
        self.assertEqual(h.wfile.calls, 1)
        self.assertTrue(h.close_connection)

    def test_child_post_timeout_returns_503_without_retry(self):
        calls = []
        closes = []

        class TimedOutConnection:
            def __init__(self, host, port, timeout):
                calls.append(("open", host, port, timeout))

            def request(self, method, path, body, headers):
                calls.append((method, path, body, headers))
                raise socket.timeout("simulated child timeout")

            def close(self):
                closes.append("closed")

        dashboard = SimpleNamespace(
            args=SimpleNamespace(
                public_origin="http://127.0.0.1:18898", wrist_port=8097),
        )
        h, statuses = self.bare_handler(
            dashboard, "child-token", path="/api/action", host="127.0.0.1:18898",
            origin="http://127.0.0.1:18898",
        )

        with mock.patch.object(web.http.client, "HTTPConnection", TimedOutConnection):
            h.do_POST()

        self.assertEqual(len([call for call in calls if call[0] == "POST"]), 1)
        self.assertEqual(calls[0], ("open", "127.0.0.1", 8097, 180))
        self.assertEqual(closes, ["closed"])
        self.assertEqual(statuses, [503])
        body = b"".join(h.wfile.payloads).decode()
        self.assertIn("操作可能已经完成", body)


if __name__ == "__main__":
    unittest.main()
