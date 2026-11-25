"""Meaningful offline checks for the wrist intrinsic collector."""

import importlib
import http.client
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest import mock

import cv2
import numpy as np

import wrist_intrinsics as wi


class FakeCamera:
    def control_state(self):
        return 0.0, 37.0


class FakeDetector:
    def __init__(self, board_corners):
        self.board_corners = board_corners


class WristIntrinsicHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = wi.CharucoDetector(wi.DEFAULT_BOARD)
        cls.lattice = cls.detector.board_corners

    def test_charuco_ids_map_to_opencv_lattice_coordinates(self):
        ids = np.array([0, 2, 5, 8, 11, 14, 17, 20, 23, 27], dtype=np.int32)
        pixels = np.array([[100 + (i % 4) * 90, 80 + (i // 4) * 95]
                           for i in range(len(ids))], dtype=float)
        decoded_ids, objects, decoded_pixels = wi.charuco_correspondences(
            ids, pixels, self.lattice, 1280, 720)
        np.testing.assert_array_equal(decoded_ids, ids)
        np.testing.assert_array_equal(objects, self.lattice[ids])
        np.testing.assert_array_equal(decoded_pixels, pixels)
        self.assertEqual(objects.shape, (10, 3))

    def test_stale_duplicate_and_nonfinite_frames_are_rejected(self):
        ids = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11], dtype=int)
        pixels = np.array([[200 + (i % 4) * 180, 150 + (i // 4) * 180]
                           for i in range(len(ids))], dtype=float)
        frame = {"monotonic_time": 10.0, "corner_ids": ids, "image_points": pixels}
        with self.assertRaisesRegex(ValueError, "過期"):
            wi.validate_frame_candidate(frame, 11.01, 1920, 1080, self.lattice)
        with self.assertRaisesRegex(ValueError, "finite"):
            wi.charuco_correspondences(ids, pixels * np.nan, self.lattice, 1920, 1080)
        good = wi.validate_frame_candidate(frame, 10.5, 1920, 1080, self.lattice)
        with self.assertRaisesRegex(ValueError, "相似"):
            wi.reject_duplicate_view(good, [good])
        moved = dict(good, image_points=(np.asarray(good["image_points"]) + [20, 0]).tolist())
        wi.reject_duplicate_view(moved, [good])

    def test_board_json_is_fixed_and_module_import_does_not_open_hardware_or_bind(self):
        accepted = wi.normalize_board_config({"squares_x": 5, "legacy_pattern": False})
        self.assertEqual(accepted, wi.DEFAULT_BOARD)
        with self.assertRaises(ValueError):
            wi.normalize_board_config({"squares_x": 6})
        with mock.patch.object(cv2, "VideoCapture", side_effect=AssertionError("hardware opened")) as capture:
            with mock.patch.object(socket.socket, "bind", side_effect=AssertionError("socket bound")) as bind:
                importlib.reload(wi)
        capture.assert_not_called()
        bind.assert_not_called()

    def test_http_server_can_rebind_after_a_preview_request(self):
        handler = wi.make_handler(None, "test-csrf")
        handler.log_message = lambda *_args: None
        first = wi.LocalHTTPServer(("127.0.0.1", 0), handler)
        port = first.server_port
        thread = threading.Thread(target=first.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn("腕部相機內參收集".encode("utf-8"), response.read())
            connection.close()
        finally:
            first.shutdown()
            first.server_close()
            thread.join(timeout=2)
        restarted = wi.LocalHTTPServer(("127.0.0.1", port), handler)
        restarted.server_close()

    def test_disconnected_response_write_does_not_pollute_session_error(self):
        session = type("Session", (), {"last_error": "previous diagnostic"})()
        handler_class = wi.make_handler(session, "test-csrf")
        handler = handler_class.__new__(handler_class)
        handler.close_connection = False
        response_calls = []
        handler.send_response = lambda status: response_calls.append(status)
        handler.send_header = lambda *_args: None
        handler.end_headers = lambda: None

        class BrokenWriter:
            def write(self, _payload):
                raise BrokenPipeError("client closed")

        handler.wfile = BrokenWriter()
        handler._send(200, "application/json", b"{}")
        self.assertEqual(response_calls, [200])
        self.assertTrue(handler.close_connection)
        self.assertEqual(session.last_error, "previous diagnostic")

    def test_async_calibration_returns_promptly_and_only_starts_one_job(self):
        session = object.__new__(wi.IntrinsicsSession)
        session.job_lock = threading.Lock()
        session.solving = False
        session.last_error = "old error"
        session.last_calibration = None
        session.calibration_thread = None
        entered = threading.Event()
        release = threading.Event()
        starts = []

        def slow_calibrate():
            starts.append(1)
            entered.set()
            release.wait(2)
            session.last_calibration = {"status": "DONE", "message": "完成"}

        session.calibrate = slow_calibrate
        started_at = __import__("time").monotonic()
        first = session.start_calibration()
        elapsed = __import__("time").monotonic() - started_at
        self.assertTrue(first["pending"])
        self.assertLess(elapsed, 0.5)
        self.assertTrue(entered.wait(1))
        second = session.start_calibration()
        self.assertTrue(second["pending"])
        self.assertEqual(len(starts), 1)
        with self.assertRaisesRegex(ValueError, "計算中"):
            session.capture(1)
        with self.assertRaisesRegex(ValueError, "計算中"):
            session.lock_focus()
        session.last_error = "busy request rejected"
        release.set()
        session.calibration_thread.join(timeout=2)
        self.assertFalse(session.solving)
        self.assertIsNone(session.last_error)
        self.assertEqual(session.last_calibration["status"], "DONE")

    def test_synthetic_perspective_views_recover_intrinsics_and_repeated_views_fail(self):
        width, height = 1920, 1080
        true_k = np.array([[1220.0, 0, 951.0], [0, 1195.0, 543.0], [0, 0, 1]], dtype=float)
        true_dist = np.array([-0.075, 0.018, 0.0008, -0.0004, 0.002], dtype=float)
        object_points = np.asarray(self.lattice, dtype=float)
        board_center = (object_points[:, :3].min(axis=0) + object_points[:, :3].max(axis=0)) / 2

        def make_views(count):
            observations = []
            for index in range(count):
                rx = np.deg2rad([-22, -12, 11, 24][index % 4])
                ry = np.deg2rad([14, -20, 24, -11][(index * 3) % 4])
                rz = np.deg2rad((index % 5 - 2) * 7)
                rvec = np.array([rx, ry, rz], dtype=float)
                rotation, _ = cv2.Rodrigues(rvec)
                depth = 0.36 + 0.02 * (index % 3)
                target_x = [470, 760, 1050, 1340][index % 4]
                target_y = [300, 505, 710][(index // 4) % 3]
                rotated_center = rotation @ board_center
                tvec = np.array([
                    (target_x - true_k[0, 2]) * depth / true_k[0, 0] - rotated_center[0],
                    (target_y - true_k[1, 2]) * depth / true_k[1, 1] - rotated_center[1],
                    depth - rotated_center[2],
                ], dtype=float)
                pixels, _ = cv2.projectPoints(object_points, rvec, tvec, true_k, true_dist)
                pixels = pixels.reshape(-1, 2)
                observations.append({
                    "view_id": f"view_{index + 1:04d}",
                    "corner_ids": list(range(len(object_points))),
                    "object_points": object_points.tolist(),
                    "image_points": pixels.tolist(),
                    "image_size": [width, height],
                    "device_uri": "/dev/video-test",
                    "resolved_device_node": "/dev/video-test",
                    "camera_serial": "test-camera",
                    "focus_value": 37.0,
                })
            return observations

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fit-session"
            root.mkdir()
            session_meta = {
                "schema_version": 1, "device_uri": "/dev/video-test",
                "resolved_device_node": "/dev/video-test", "camera_serial": "test-camera",
                "image_size": [width, height],
                "board_geometry_sha256": wi.board_geometry_sha256(wi.DEFAULT_BOARD),
            }
            (root / "session.json").write_text(json.dumps(session_meta), encoding="utf-8")
            (root / "focus_lock.json").write_text(json.dumps({"focus_value": 37.0}), encoding="utf-8")
            views = make_views(16)
            for observation in views:
                view_dir = root / observation["view_id"]
                view_dir.mkdir()
                (view_dir / "raw.png").write_bytes(b"synthetic image placeholder")
                (view_dir / "observation.json").write_text(json.dumps(observation), encoding="utf-8")
            session = wi.IntrinsicsSession(root, "/dev/video-test", "/dev/video-test", "test-camera",
                                           width, height, wi.DEFAULT_BOARD,
                                           FakeDetector(self.lattice), FakeCamera())
            result = session.calibrate()
            self.assertTrue(result.get("ok"), result)
            self.assertEqual(result["status"], wi.CANDIDATE_STATUS)
            self.assertLess(abs(result["matrix"][0][0] - true_k[0, 0]) / true_k[0, 0], 0.03)
            self.assertLess(abs(result["matrix"][1][1] - true_k[1, 1]) / true_k[1, 1], 0.03)
            adopted = session.latest_candidate()
            self.assertIsNotNone(adopted)
            self.assertEqual(len(adopted["sha256"]), 64)
            self.assertTrue((root / adopted["result_file"]).is_file())
            resumed = wi.IntrinsicsSession(root, "/dev/video-test", "/dev/video-test", "test-camera",
                                            width, height, wi.DEFAULT_BOARD,
                                            FakeDetector(self.lattice), FakeCamera())
            self.assertEqual(resumed.last_calibration["result_file"], adopted["result_file"])
            self.assertIsNotNone(resumed.latest_candidate())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repeat-session"
            root.mkdir()
            session_meta["board_geometry_sha256"] = wi.board_geometry_sha256(wi.DEFAULT_BOARD)
            (root / "session.json").write_text(json.dumps(session_meta), encoding="utf-8")
            (root / "focus_lock.json").write_text(json.dumps({"focus_value": 37.0}), encoding="utf-8")
            repeated = make_views(1)[0]
            for index in range(wi.MIN_VIEWS):
                observation = dict(repeated, view_id=f"view_{index + 1:04d}")
                view_dir = root / observation["view_id"]
                view_dir.mkdir()
                (view_dir / "raw.png").write_bytes(b"synthetic image placeholder")
                (view_dir / "observation.json").write_text(json.dumps(observation), encoding="utf-8")
            session = wi.IntrinsicsSession(root, "/dev/video-test", "/dev/video-test", "test-camera",
                                           width, height, wi.DEFAULT_BOARD,
                                           FakeDetector(self.lattice), FakeCamera())
            result = session.calibrate()
            self.assertFalse(result.get("ok"))
            self.assertEqual(result["status"], "CALIBRATION_GATES_NOT_MET")
            self.assertFalse(list(root.glob("intrinsics_*.json")))



class CameraReconnectTests(unittest.TestCase):
    class Capture:
        def __init__(self, width=1920, autofocus=0, focus=37, refuse_focus=False):
            self.values={cv2.CAP_PROP_FRAME_WIDTH:width, cv2.CAP_PROP_FRAME_HEIGHT:1080,
                         cv2.CAP_PROP_FOURCC:cv2.VideoWriter_fourcc(*"MJPG"),
                         cv2.CAP_PROP_AUTOFOCUS:autofocus, cv2.CAP_PROP_FOCUS:focus}
            self.width=width; self.refuse_focus=refuse_focus; self.released=False
        def isOpened(self): return True
        def get(self,key): return self.values.get(key,0)
        def set(self,key,value):
            if key==cv2.CAP_PROP_FRAME_WIDTH and self.width!=1920: return False
            if key==cv2.CAP_PROP_FOCUS and self.refuse_focus: return False
            self.values[key]=value; return True
        def release(self): self.released=True

    def test_reopen_clears_cache_and_restores_locked_focus(self):
        with tempfile.TemporaryDirectory() as temp:
            device=Path(temp)/"stable-video"; device.touch()
            first=self.Capture(); second=self.Capture(autofocus=1,focus=9)
            with mock.patch.object(wi,"_camera_serial",return_value="usb-location:test"), mock.patch.object(cv2,"VideoCapture",side_effect=[first,second]):
                camera=wi.CameraWorker(device,1920,1080,None)
                camera.set_expected_focus(37)
                camera.frames.append(object())
                camera._disconnect("unplugged")
                self.assertEqual(len(camera.frames),0)
                self.assertTrue(first.released)
                camera._open_capture()
                self.assertTrue(camera._focus_verified)
                self.assertEqual(camera.control_state(),(0,37))
                camera.close()

    def test_reopen_rejects_different_identity_and_wrong_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            device=Path(temp)/"stable-video"; device.touch()
            with mock.patch.object(wi,"_camera_serial",side_effect=["one","one","two"]), mock.patch.object(cv2,"VideoCapture",return_value=self.Capture()) as opening:
                camera=wi.CameraWorker(device,1920,1080,None)
                camera._disconnect("unplugged")
                with self.assertRaisesRegex(RuntimeError,"身分改變"): camera._open_capture()
                self.assertEqual(opening.call_count,1)
                camera.close()
            bad=self.Capture(width=640)
            with mock.patch.object(wi,"_camera_serial",return_value="one"), mock.patch.object(cv2,"VideoCapture",return_value=bad):
                with self.assertRaisesRegex(RuntimeError,"模式不符"): wi.CameraWorker(device,1920,1080,None)
                self.assertTrue(bad.released)

    def test_reopen_refuses_unrestorable_focus(self):
        with tempfile.TemporaryDirectory() as temp:
            device=Path(temp)/"stable-video"; device.touch()
            first=self.Capture(); bad=self.Capture(autofocus=1,focus=9,refuse_focus=True)
            with mock.patch.object(wi,"_camera_serial",return_value="one"), mock.patch.object(cv2,"VideoCapture",side_effect=[first,bad]):
                camera=wi.CameraWorker(device,1920,1080,None); camera.set_expected_focus(37)
                camera._disconnect("unplugged")
                with self.assertRaisesRegex(RuntimeError,"鎖焦值"): camera._open_capture()
                self.assertEqual(len(camera.frames),0)
                camera.close()

    def test_usb_location_identity_survives_node_renumbering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); usb=root/"usb-parent"; usb.mkdir()
            for key,value in {"idVendor":"1bcf","idProduct":"2281","manufacturer":"XHH","product":"2M"}.items(): (usb/key).write_text(value)
            for node in ("video2","video3"):
                parent=root/node; parent.mkdir(); (parent/"device").symlink_to(usb,target_is_directory=True)
            a=wi._camera_serial("/dev/v4l/by-path/same-port","/dev/video2",root)
            b=wi._camera_serial("/dev/v4l/by-path/same-port","/dev/video3",root)
            self.assertEqual(a,b); self.assertTrue(a.startswith("usb-location:1bcf:2281:"))
            self.assertNotEqual(a,wi._camera_serial("/dev/v4l/by-path/another-port","/dev/video3",root))

if __name__ == "__main__":
    unittest.main()
