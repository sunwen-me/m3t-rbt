#!/usr/bin/env python3
"""Local wrist-camera ChArUco intrinsic collector.

The process opens only the explicitly named V4L2 device. It keeps raw frames
for exact-frame capture, serves a loopback-only Chinese UI, and writes each
view/result with exclusive-create semantics. A fitted K is always a candidate
that needs independent verification.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import ipaddress
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np


DEFAULT_BOARD = {
    "squares_x": 5,
    "squares_y": 8,
    "square_length_m": 0.024,
    "marker_length_m": 0.017,
    "dictionary": "DICT_4X4_50",
    "legacy_pattern": False,
}
MIN_CORNERS = 10
MIN_VIEWS = 12
MAX_FRAME_AGE_S = 1.0
MIN_CORNER_MOTION_PX = 12.0
MIN_HULL_FRACTION = 0.02
FRAME_CACHE_LIMIT = 32
CANDIDATE_STATUS = "CANDIDATE_INTRINSICS_REQUIRES_INDEPENDENT_VERIFICATION"


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _finite_float(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected a finite number")
    return result


def normalize_board_config(value=None):
    """Accept only the fixed shared-board geometry used by the solver."""
    if value is None:
        return dict(DEFAULT_BOARD)
    if not isinstance(value, dict) or set(value) - set(DEFAULT_BOARD):
        raise ValueError("board JSON must contain only supported board fields")
    board = dict(DEFAULT_BOARD)
    board.update(value)
    for key, expected in DEFAULT_BOARD.items():
        actual = board[key]
        if isinstance(expected, bool):
            good = type(actual) is bool and actual is expected
        elif isinstance(expected, int):
            good = type(actual) is int and actual == expected
        elif isinstance(expected, float):
            good = (not isinstance(actual, bool) and isinstance(actual, (int, float))
                    and math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12))
        else:
            good = type(actual) is str and actual == expected
        if not good:
            raise ValueError(f"board.{key} must be {expected!r}")
    return board


def board_geometry_sha256(board):
    return hashlib.sha256(_json_bytes(normalize_board_config(board))).hexdigest()


def charuco_correspondences(corner_ids, image_points, board_corners,
                            width=None, height=None):
    """Map decoded IDs through OpenCV's actual board lattice, never guessed order."""
    lattice = np.asarray(board_corners, dtype=np.float64)
    ids = np.asarray(corner_ids)
    pixels = np.asarray(image_points, dtype=np.float64)
    if lattice.ndim != 2 or lattice.shape[1] != 3 or not np.isfinite(lattice).all():
        raise ValueError("board lattice must be a finite Nx3 array")
    ids = ids.reshape(-1)
    if ids.size < MIN_CORNERS or pixels.shape != (len(ids), 2):
        raise ValueError(f"at least {MIN_CORNERS} corresponding ChArUco corners are required")
    if not np.issubdtype(ids.dtype, np.integer):
        if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
            raise ValueError("ChArUco corner IDs must be integers")
    ids = ids.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("duplicate ChArUco corner IDs")
    if (np.any(ids < 0) or np.any(ids >= len(lattice)) or not np.isfinite(pixels).all()):
        raise ValueError("corner IDs or image points are outside their valid finite range")
    if width and height and (np.any(pixels[:, 0] < 0) or np.any(pixels[:, 0] >= int(width))
                             or np.any(pixels[:, 1] < 0) or np.any(pixels[:, 1] >= int(height))):
        raise ValueError("ChArUco image points fall outside the captured image bounds")
    objects = lattice[ids]
    if np.linalg.matrix_rank(objects[:, :2] - objects[:, :2].mean(axis=0), tol=1e-9) < 2:
        raise ValueError("ChArUco corners are degenerate or collinear")
    hull_area = float(cv2.contourArea(cv2.convexHull(pixels.astype(np.float32))))
    if width and height and hull_area / (int(width) * int(height)) < MIN_HULL_FRACTION:
        raise ValueError("the detected board covers less than 2% of the image")
    return ids, objects, pixels


def validate_frame_candidate(frame, now_monotonic, width, height, board_corners):
    """Validate an exact cached frame before it is allowed to become a view."""
    if now_monotonic - float(frame["monotonic_time"]) > MAX_FRAME_AGE_S:
        raise ValueError("畫面已過期，請使用最新畫面再保存")
    if now_monotonic < float(frame["monotonic_time"]):
        raise ValueError("畫面時間戳無效")
    ids, objects, pixels = charuco_correspondences(
        frame["corner_ids"], frame["image_points"], board_corners, width, height)
    return {"corner_ids": ids.astype(int).tolist(),
            "object_points": objects.tolist(), "image_points": pixels.tolist()}


def reject_duplicate_view(new_view, previous_views):
    """Reject views whose shared corner pixels barely moved from a saved view."""
    new_ids = np.asarray(new_view["corner_ids"], dtype=int)
    new_pixels = np.asarray(new_view["image_points"], dtype=float)
    new_lookup = {int(cid): new_pixels[i] for i, cid in enumerate(new_ids)}
    for previous in previous_views:
        old_ids = np.asarray(previous["corner_ids"], dtype=int)
        old_pixels = np.asarray(previous["image_points"], dtype=float)
        old_lookup = {int(cid): old_pixels[i] for i, cid in enumerate(old_ids)}
        common = sorted(set(new_lookup) & set(old_lookup))
        if len(common) < MIN_CORNERS:
            continue
        movement = np.linalg.norm(np.asarray([new_lookup[i] - old_lookup[i] for i in common]), axis=1)
        if not np.isfinite(movement).all():
            raise ValueError("image points contain non-finite values")
        if float(np.median(movement)) < MIN_CORNER_MOTION_PX:
            raise ValueError("與已保存姿態太相似：共同角點移動小於 12 像素")


def estimate_board_tilt_degrees(object_points, image_points, width, height):
    """Estimate board-normal tilt from a homography using a rough image K."""
    source = np.asarray(object_points, dtype=np.float64)[:, :2]
    target = np.asarray(image_points, dtype=np.float64)
    homography, _ = cv2.findHomography(source, target, 0)
    if homography is None or not np.isfinite(homography).all():
        return None
    focal = float(max(width, height))
    rough_k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=float)
    normalized = np.linalg.inv(rough_k) @ homography
    first, second = normalized[:, 0], normalized[:, 1]
    scale = (np.linalg.norm(first) + np.linalg.norm(second)) / 2.0
    if not math.isfinite(float(scale)) or scale < 1e-12:
        return None
    r1 = first / np.linalg.norm(first)
    r2 = second - np.dot(second, r1) * r1
    norm_r2 = np.linalg.norm(r2)
    if norm_r2 < 1e-12:
        return None
    r2 /= norm_r2
    normal = np.cross(r1, r2)
    normal /= np.linalg.norm(normal)
    tilt = math.degrees(math.acos(float(np.clip(abs(normal[2]), 0, 1))))
    return tilt if math.isfinite(tilt) else None


def dataset_diversity(views, width, height):
    """Return spatial coverage and perspective-tilt gates for saved views."""
    if len(views) < MIN_VIEWS:
        return {"passed": False, "reason": f"至少需要 {MIN_VIEWS} 個不同姿態，目前 {len(views)} 個"}
    mask = np.zeros((height, width), dtype=np.uint8)
    centers = []
    tilts = []
    for view in views:
        pixels = np.asarray(view["image_points"], dtype=np.float32)
        hull = cv2.convexHull(pixels)
        cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
        centers.append(pixels.mean(axis=0))
        tilt = estimate_board_tilt_degrees(view["object_points"], pixels, width, height)
        if tilt is not None:
            tilts.append(tilt)
    coverage = float(cv2.countNonZero(mask)) / float(width * height)
    if len(centers) > 1:
        centers = np.asarray(centers, dtype=float)
        spread = float(np.max(np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)))
    else:
        spread = 0.0
    diagonal = math.hypot(width, height)
    tilted = [angle for angle in tilts if angle >= 8.0]
    tilt_spread = (max(tilts) - min(tilts)) if len(tilts) > 1 else 0.0
    failures = []
    if coverage < 0.20:
        failures.append(f"姿態聯集覆蓋率 {coverage:.1%}，需至少 20%")
    if spread < 0.10 * diagonal:
        failures.append(f"板中心最大跨度 {spread:.1f}px，需至少影像對角線的 10%")
    if len(tilted) < 3 or max(tilts, default=0.0) < 15.0 or tilt_spread < 5.0:
        failures.append("板傾角多樣性不足：至少 3 個姿態需傾斜 8°，最大傾角需達 15°，傾角跨度需達 5°")
    return {"passed": not failures, "coverage_fraction": coverage,
            "center_spread_px": spread, "tilts_degrees": tilts,
            "tilt_spread_degrees": tilt_spread, "failures": failures}


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_exclusive(path, payload):
    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _camera_serial(device_name, resolved_node, sysfs_root="/sys/class/video4linux"):
    """Return durable camera identity; never use /dev/videoN as identity."""
    video_name = Path(resolved_node).name
    device_link = Path(sysfs_root) / video_name / "device"
    try:
        usb_device = device_link.resolve(strict=True)
    except OSError:
        usb_device = device_link
    for parent in (usb_device, *usb_device.parents):
        try:
            vendor = (parent / "idVendor").read_text(encoding="utf-8").strip().lower()
            product_id = (parent / "idProduct").read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if not vendor or not product_id:
            continue
        try:
            serial = (parent / "serial").read_text(encoding="utf-8").strip()
        except OSError:
            serial = ""
        if serial:
            return f"usb:{vendor}:{product_id}:{serial}"
        try:
            manufacturer = (parent / "manufacturer").read_text(encoding="utf-8").strip()
        except OSError:
            manufacturer = ""
        try:
            product = (parent / "product").read_text(encoding="utf-8").strip()
        except OSError:
            product = ""
        stable_uri = str(Path(device_name).expanduser())
        material = "\0".join((stable_uri, manufacturer, product)).encode("utf-8")
        location = hashlib.sha256(material).hexdigest()[:20]
        return f"usb-location:{vendor}:{product_id}:{location}"
    return f"uri:{Path(device_name).expanduser()}"


class CharucoDetector:
    def __init__(self, board_config):
        aruco = cv2.aruco
        dictionary_id = getattr(aruco, board_config["dictionary"])
        dictionary = aruco.getPredefinedDictionary(dictionary_id)
        self.board = aruco.CharucoBoard(
            (board_config["squares_x"], board_config["squares_y"]),
            board_config["square_length_m"], board_config["marker_length_m"], dictionary)
        if hasattr(self.board, "setLegacyPattern"):
            self.board.setLegacyPattern(board_config["legacy_pattern"])
        elif hasattr(self.board, "legacyPattern"):
            self.board.legacyPattern = board_config["legacy_pattern"]
        self.board_corners = np.asarray(self.board.getChessboardCorners(), dtype=np.float64)
        self.dictionary = dictionary
        params = aruco.DetectorParameters()
        self.marker_detector = (aruco.ArucoDetector(dictionary, params)
                                if hasattr(aruco, "ArucoDetector") else None)
        self.charuco_detector = (aruco.CharucoDetector(self.board)
                                 if hasattr(aruco, "CharucoDetector") else None)

    def detect(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, charuco_corners, charuco_ids = [], None, None, None
        if self.charuco_detector is not None:
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
        else:
            if self.marker_detector is None:
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
            else:
                marker_corners, marker_ids, _ = self.marker_detector.detectMarkers(gray)
            if marker_ids is not None and len(marker_ids):
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.board)
        pixels = (np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
                  if charuco_corners is not None else np.empty((0, 2), dtype=float))
        ids = (np.asarray(charuco_ids, dtype=np.int64).reshape(-1)
               if charuco_ids is not None else np.empty((0,), dtype=np.int64))
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        return marker_count, ids, pixels


@dataclass
class Frame:
    sequence: int
    monotonic_time: float
    read_started_monotonic: float
    read_completed_monotonic: float
    wall_time: str
    bgr: np.ndarray
    marker_count: int
    corner_ids: np.ndarray
    image_points: np.ndarray


class CameraWorker:
    """Owns exactly one explicit V4L2 camera and a short fresh-frame cache."""
    def __init__(self, device, width, height, detector, camera_serial=None):
        self.device = str(device)
        self.width, self.height = int(width), int(height)
        self.detector = detector
        self.capture_lock = threading.RLock()
        self.cache_lock = threading.Lock()
        self.frames = deque(maxlen=FRAME_CACHE_LIMIT)
        self.sequence = 0
        self.resolved_node = str(Path(self.device).expanduser().resolve(strict=True))
        self.camera_serial = camera_serial or _camera_serial(self.device, self.resolved_node)
        self.expected_focus_value = None
        self.error = None
        self.connected = False
        self._focus_verified = False
        self._read_failures = 0
        self._reconnect_delay = 0.25
        self._next_reconnect_at = 0.0
        self.stopping = threading.Event()
        self.cap = None
        self._open_capture()
        self.thread = threading.Thread(target=self._read_loop, name="wrist-v4l2-frame-reader", daemon=True)

    @staticmethod
    def _fourcc_name(value):
        number = int(round(value))
        try:
            return "".join(chr((number >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")
        except (ValueError, OverflowError):
            return ""

    def start(self):
        self.thread.start()

    def _open_capture(self):
        resolved = str(Path(self.device).expanduser().resolve(strict=True))
        actual_identity = _camera_serial(self.device, resolved)
        if actual_identity != self.camera_serial:
            raise RuntimeError(
                f"指定相機身分改變：預期 {self.camera_serial}，實際 {actual_identity}")
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"無法開啟指定相機 {self.device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        actual_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fourcc = self._fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))
        if (actual_width != self.width or actual_height != self.height or actual_fourcc != "MJPG"):
            cap.release()
            raise RuntimeError(
                f"相機模式不符：要求 {self.width}x{self.height} MJPG，"
                f"實際 {actual_width}x{actual_height} {actual_fourcc or '未知'}")
        with self.capture_lock:
            self.cap = cap
            self.resolved_node = resolved
            self.connected = True
            self._read_failures = 0
        if self.expected_focus_value is not None:
            with self.capture_lock:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                if not math.isclose(float(cap.get(cv2.CAP_PROP_FOCUS)), self.expected_focus_value,
                                    rel_tol=0, abs_tol=1e-3):
                    cap.set(cv2.CAP_PROP_FOCUS, self.expected_focus_value)
        self._focus_verified = self._focus_matches_expected()
        self._reconnect_delay = 0.25
        if self._focus_verified:
            self.error = None
        else:
            raise RuntimeError("相機重連後無法恢復既有鎖焦值；拒絕使用畫面")

    def _focus_matches_expected(self):
        if self.expected_focus_value is None:
            return True
        try:
            autofocus, focus = self.control_state()
        except (cv2.error, OSError, ValueError):
            return False
        return (math.isfinite(autofocus) and abs(autofocus) <= 1e-6
                and math.isfinite(focus)
                and math.isclose(focus, self.expected_focus_value, rel_tol=0, abs_tol=1e-3))

    def set_expected_focus(self, value):
        self.expected_focus_value = None if value is None else float(value)
        self._focus_verified = self._focus_matches_expected()
        if not self._focus_verified:
            self.error = "相機焦距或自動對焦與工作階段鎖定值不符；拒絕使用畫面"

    def _clear_frames(self):
        with self.cache_lock:
            self.frames.clear()

    def _disconnect(self, message, drop_capture=True):
        self.error = str(message)
        self.connected = False
        self._focus_verified = False
        self._clear_frames()
        if drop_capture:
            with self.capture_lock:
                cap, self.cap = self.cap, None
                if cap is not None:
                    cap.release()
            self._next_reconnect_at = time.monotonic() + self._reconnect_delay
            self._reconnect_delay = min(2.0, self._reconnect_delay * 2.0)

    def _read_loop(self):
        while not self.stopping.is_set():
            if self.cap is None:
                delay = self._next_reconnect_at - time.monotonic()
                if delay > 0:
                    self.stopping.wait(min(delay, 0.25))
                    continue
                try:
                    self._open_capture()
                except Exception as exc:
                    self._disconnect(f"相機重新連線失敗：{type(exc).__name__}: {exc}")
                    continue
            read_started = time.monotonic()
            try:
                with self.capture_lock:
                    cap = self.cap
                    ok, bgr = (False, None) if cap is None else cap.read()
            except Exception as exc:
                ok, bgr = False, None
                read_error = f"相機讀取失敗：{type(exc).__name__}: {exc}"
            else:
                read_error = "相機讀取失敗"
            read_completed = time.monotonic()
            if not ok or bgr is None:
                self._read_failures += 1
                self._disconnect(read_error, drop_capture=self._read_failures >= 2)
                if self.cap is not None:
                    self.stopping.wait(0.03)
                continue
            if (bgr.ndim != 3 or bgr.shape[0] != self.height
                    or bgr.shape[1] != self.width or bgr.shape[2] != 3):
                self._disconnect(
                    f"影格尺寸或通道不符：預期 {self.width}x{self.height} BGR，"
                    f"實際 shape={getattr(bgr, 'shape', None)}")
                continue
            self._read_failures = 0
            self.connected = True
            self._focus_verified = self._focus_matches_expected()
            if not self._focus_verified:
                self._clear_frames()
                self.error = "自動對焦或焦距與工作階段鎖定值不符；拒絕使用畫面"
                self.stopping.wait(0.05)
                continue
            try:
                marker_count, ids, pixels = self.detector.detect(bgr)
                with self.cache_lock:
                    self.sequence += 1
                    self.frames.append(Frame(self.sequence, read_completed, read_started,
                                             read_completed, _utc_now(),
                                             bgr.copy(), marker_count, ids.copy(), pixels.copy()))
                self.error = None
            except Exception as exc:  # keep preview alive and expose a short diagnostic
                self.error = f"影像偵測錯誤：{type(exc).__name__}: {exc}"

    def latest(self):
        with self.cache_lock:
            frame = self.frames[-1] if self.frames else None
            if frame is not None and time.monotonic() - frame.monotonic_time > MAX_FRAME_AGE_S:
                self.frames.clear()
                self.error = "相機畫面已過期；等待重新連線"
                return None
            return frame

    def cached(self, sequence):
        with self.cache_lock:
            frame = next((frame for frame in self.frames if frame.sequence == sequence), None)
            if frame is not None and time.monotonic() - frame.monotonic_time > MAX_FRAME_AGE_S:
                self.frames.clear()
                self.error = "相機畫面已過期；等待重新連線"
                return None
            return frame

    def control_state(self):
        with self.capture_lock:
            if self.cap is None:
                return float("nan"), float("nan")
            autofocus = float(self.cap.get(cv2.CAP_PROP_AUTOFOCUS))
            focus = float(self.cap.get(cv2.CAP_PROP_FOCUS))
        return autofocus, focus

    def lock_autofocus(self):
        with self.capture_lock:
            if self.cap is None:
                raise ValueError("相機尚未連線")
            before = float(self.cap.get(cv2.CAP_PROP_FOCUS))
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            autofocus = float(self.cap.get(cv2.CAP_PROP_AUTOFOCUS))
            after = float(self.cap.get(cv2.CAP_PROP_FOCUS))
        if not math.isfinite(autofocus) or abs(autofocus) > 1e-6:
            raise ValueError("相機未回報自動對焦已關閉，不能保存或計算")
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError("相機沒有可驗證的焦距位置讀值")
        if before < 0 or after < 0:
            raise ValueError("相機未提供可用的焦距位置讀值")
        if not math.isclose(before, after, rel_tol=0, abs_tol=1e-3):
            raise ValueError("關閉自動對焦時焦距位置改變；本次操作未保存任何姿態")
        return after

    def close(self):
        self.stopping.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        with self.capture_lock:
            cap, self.cap = self.cap, None
            if cap is not None:
                cap.release()


class IntrinsicsSession:
    def __init__(self, output, device_uri, resolved_node, camera_serial,
                 width, height, board_config, detector, camera):
        self.output = Path(output).expanduser().absolute()
        self.width, self.height = int(width), int(height)
        self.board_config = board_config
        self.detector = detector
        self.camera = camera
        self.device_uri = str(device_uri)
        self.resolved_node = str(resolved_node)
        self.camera_serial = str(camera_serial)
        self.board_sha = board_geometry_sha256(board_config)
        self.lock = threading.RLock()
        self.job_lock = threading.Lock()
        self.solving = False
        self.calibration_thread = None
        self.last_error = None
        self.last_calibration = None
        self._prepare_output()
        set_expected_focus = getattr(self.camera, "set_expected_focus", None)
        if callable(set_expected_focus):
            set_expected_focus(self.locked_focus)
        self._restore_latest_candidate()

    def _current_resolved_node(self):
        return str(getattr(self.camera, "resolved_node", None) or self.resolved_node)

    def _metadata(self):
        autofocus, focus = self.camera.control_state()
        return {
            "schema_version": 1,
            "task": "wrist_camera_intrinsic_calibration_collection",
            "created_utc": _utc_now(),
            "device_uri": self.device_uri,
            "resolved_device_node": self.resolved_node,
            "camera_serial": self.camera_serial,
            "image_size": [self.width, self.height],
            "focus": {"startup_value": focus if math.isfinite(focus) else None,
                      "autofocus_on_at_start": bool(autofocus > 0.5) if math.isfinite(autofocus) else None,
                      "locked_value": None},
            "K": None,
            "board": self.board_config,
            "board_geometry_sha256": self.board_sha,
        }

    def _prepare_output(self):
        if self.output.exists():
            if not self.output.is_dir():
                raise ValueError("輸出路徑必須是工作階段目錄")
            meta_path = self.output / "session.json"
            if not meta_path.exists():
                if any(self.output.iterdir()):
                    raise ValueError("既有輸出目錄不是本工具的空白工作階段，拒絕覆寫")
                meta = self._metadata()
                _write_exclusive(meta_path, _json_bytes(meta))
            else:
                try:
                    existing = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError("既有 session.json 無法讀取") from exc
                expected = (self.device_uri, self.camera_serial,
                            [self.width, self.height], self.board_sha)
                actual = (existing.get("device_uri"), existing.get("camera_serial"),
                          existing.get("image_size"), existing.get("board_geometry_sha256"))
                if actual != expected:
                    raise ValueError("既有工作階段的裝置、解析度或板幾何不同；拒絕續用")
                if existing.get("schema_version") != 1:
                    raise ValueError("既有工作階段 schema 不相容")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            _write_exclusive(self.output / "session.json", _json_bytes(self._metadata()))
        self._validate_existing_views()
        lock_path = self.output / "focus_lock.json"
        existing_views = self._views()
        if lock_path.exists():
            focus_record = json.loads(lock_path.read_text(encoding="utf-8"))
            autofocus, focus = self.camera.control_state()
            if (not math.isfinite(autofocus) or abs(autofocus) > 1e-6
                    or not math.isfinite(focus)
                    or not math.isclose(focus, float(focus_record["focus_value"]), rel_tol=0, abs_tol=1e-3)):
                raise ValueError("既有工作階段焦距或自動對焦狀態不符；拒絕續用")
            self.locked_focus = float(focus_record["focus_value"])
            for view in existing_views:
                if (not math.isfinite(float(view.get("focus_value", float("nan"))))
                        or not math.isclose(float(view["focus_value"]), self.locked_focus,
                                            rel_tol=0, abs_tol=1e-3)):
                    raise ValueError("既有姿態焦距與工作階段鎖定值不同；拒絕續用")
        else:
            if existing_views:
                raise ValueError("既有姿態缺少焦距鎖定記錄；拒絕以未知焦距續用")
            self.locked_focus = None
        saved_sequences = [view["sequence_id"] for view in existing_views
                           if type(view.get("sequence_id")) is int]
        # Continue IDs across a matching session restart so saved frame IDs stay unique.
        self.camera.sequence = max(saved_sequences, default=0)

    def _validate_existing_views(self):
        for path in self.output.glob("view_*"):
            if not path.is_dir():
                raise ValueError(f"既有記錄不是姿態目錄：{path.name}")
            metadata = path / "observation.json"
            image = path / "raw.png"
            if not metadata.is_file() or not image.is_file():
                raise ValueError(f"姿態記錄不完整：{path.name}")
            value = json.loads(metadata.read_text(encoding="utf-8"))
            ids, objects, pixels = charuco_correspondences(
                value.get("corner_ids"), value.get("image_points"), self.detector.board_corners,
                self.width, self.height)
            if value.get("image_size") != [self.width, self.height]:
                raise ValueError(f"姿態解析度不符：{path.name}")
            if (value.get("device_uri") != self.device_uri
                    or value.get("camera_serial") != self.camera_serial):
                raise ValueError(f"姿態裝置綁定不符：{path.name}")
            if "sequence_id" in value and (type(value["sequence_id"]) is not int
                                             or value["sequence_id"] < 1):
                raise ValueError(f"姿態影格 ID 無效：{path.name}")

    def _views(self):
        views = []
        for path in sorted(self.output.glob("view_*")):
            if path.is_dir():
                value = json.loads((path / "observation.json").read_text(encoding="utf-8"))
                views.append(value)
        return views

    def _require_focus_locked(self):
        if self.locked_focus is None:
            raise ValueError("請先按「鎖定目前焦距」")
        autofocus, focus = self.camera.control_state()
        if not math.isfinite(autofocus) or abs(autofocus) > 1e-6:
            raise ValueError("自動對焦仍開啟；請重新鎖定焦距")
        if not math.isfinite(focus) or not math.isclose(
                focus, self.locked_focus, rel_tol=0, abs_tol=1e-3):
            raise ValueError("焦距位置與鎖定值不同；拒絕保存或計算")

    def lock_focus(self):
        if self.solving:
            raise ValueError("內參計算中，請稍候完成後再鎖定焦距")
        with self.lock:
            if self.solving:
                raise ValueError("內參計算中，請稍候完成後再鎖定焦距")
            value = self.camera.lock_autofocus()
            if self.locked_focus is not None:
                if not math.isclose(value, self.locked_focus, rel_tol=0, abs_tol=1e-3):
                    raise ValueError("既有工作階段的焦距鎖定值不同")
                return {"ok": True, "focus_value": value, "already_locked": True}
            record = {"schema_version": 1, "locked_utc": _utc_now(),
                      "focus_value": value, "autofocus": 0}
            _write_exclusive(self.output / "focus_lock.json", _json_bytes(record))
            self.locked_focus = value
            setter = getattr(self.camera, "set_expected_focus", None)
            if callable(setter):
                setter(value)
            return {"ok": True, "focus_value": value, "already_locked": False}

    def capture(self, sequence):
        if self.solving:
            raise ValueError("內參計算中，請稍候完成後再保存姿態")
        with self.lock:
            if self.solving:
                raise ValueError("內參計算中，請稍候完成後再保存姿態")
            self._require_focus_locked()
            if type(sequence) is not int or sequence < 1:
                raise ValueError("frame_id 必須是正整數")
            frame = self.camera.cached(sequence)
            if frame is None:
                raise ValueError("指定畫面已離開快取，請重新選取最新畫面")
            observation = validate_frame_candidate({
                "monotonic_time": frame.monotonic_time,
                "corner_ids": frame.corner_ids,
                "image_points": frame.image_points,
            }, time.monotonic(), self.width, self.height, self.detector.board_corners)
            existing = self._views()
            if any(view.get("sequence_id") == frame.sequence for view in existing):
                raise ValueError("指定影格已保存過，請選擇不同的新影格")
            reject_duplicate_view(observation, existing)
            next_index = max([int(path.name.removeprefix("view_")) for path in self.output.glob("view_*")
                              if path.is_dir() and path.name.removeprefix("view_").isdigit()] + [0]) + 1
            directory = self.output / f"view_{next_index:04d}"
            ok, encoded = cv2.imencode(".png", frame.bgr)
            if not ok:
                raise RuntimeError("無法編碼原始 BGR 畫面")
            directory.mkdir(exist_ok=False)
            observation.update({
                "schema_version": 1,
                "view_id": directory.name,
                "sequence_id": frame.sequence,
                "captured_utc": frame.wall_time,
                "captured_monotonic_seconds": frame.monotonic_time,
                "read_started_monotonic_seconds": frame.read_started_monotonic,
                "read_completed_monotonic_seconds": frame.read_completed_monotonic,
                "image_size": [self.width, self.height],
                "device_uri": self.device_uri,
                "resolved_device_node": self._current_resolved_node(),
                "camera_serial": self.camera_serial,
                "focus_value": self.locked_focus,
                "image_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
                "raw_image": "raw.png",
            })
            _write_exclusive(directory / "raw.png", encoded.tobytes())
            _write_exclusive(directory / "observation.json", _json_bytes(observation))
            return {"ok": True, "view_id": directory.name,
                    "view_count": len(existing) + 1, "sequence_id": frame.sequence}

    def observation_payload(self):
        frame = self.camera.latest()
        if frame is None:
            raise RuntimeError(self.camera.error or "尚無相機畫面")
        age = max(0.0, time.monotonic() - frame.monotonic_time)
        ids, objects, pixels = charuco_correspondences(
            frame.corner_ids, frame.image_points, self.detector.board_corners,
            self.width, self.height) if len(frame.corner_ids) >= MIN_CORNERS else (
                frame.corner_ids.astype(int), np.empty((0, 3)), frame.image_points)
        autofocus, focus = self.camera.control_state()
        ok, encoded = cv2.imencode(".png", frame.bgr)
        if not ok:
            raise RuntimeError("無法編碼原始 BGR 畫面")
        return {
            "frame_id": frame.sequence,
            "captured_utc": frame.wall_time,
            "captured_monotonic_seconds": frame.monotonic_time,
            "read_started_monotonic_seconds": frame.read_started_monotonic,
            "read_completed_monotonic_seconds": frame.read_completed_monotonic,
            "read_time_utc": _utc_now(),
            "frame_age_seconds": age,
            "width": self.width,
            "height": self.height,
            "raw_image_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "corner_ids": ids.astype(int).tolist(),
            "corners_px": pixels.tolist(),
            "object_points_m": objects.tolist(),
            "marker_count": frame.marker_count,
            "corner_count": int(len(ids)),
            "focus": {"autofocus": autofocus if math.isfinite(autofocus) else None,
                      "focus_value": focus if math.isfinite(focus) else None,
                      "locked": self.locked_focus is not None,
                      "locked_value": self.locked_focus},
            "device_binding": {"device_uri": self.device_uri,
                               "resolved_device_node": self._current_resolved_node(),
                               "camera_serial": self.camera_serial},
            "board": self.board_config,
            "board_geometry_sha256": self.board_sha,
            "K": None,
        }

    def start_calibration(self):
        """Start one background solve so proxy request timeouts cannot interrupt it."""
        with self.job_lock:
            if self.solving:
                return {"ok": True, "pending": True,
                        "status": "CALIBRATING", "message": "內參計算正在進行，請稍候。"}
            self.solving = True
            self.last_error = None
            self.calibration_thread = threading.Thread(
                target=self._run_calibration_job, name="wrist-intrinsics-solver", daemon=True)
            try:
                self.calibration_thread.start()
            except Exception:
                self.solving = False
                self.calibration_thread = None
                raise
        return {"ok": True, "pending": True, "status": "CALIBRATING",
                "message": "內參計算已在背景開始，完成後會顯示 RMS 和結果。"}

    def _run_calibration_job(self):
        try:
            self.calibrate()
        except Exception as exc:
            self.last_calibration = {
                "status": "CALIBRATION_FAILED",
                "message": f"內參求解失敗：{type(exc).__name__}: {exc}",
            }
        finally:
            with self.job_lock:
                self.last_error = None
                self.solving = False

    def calibrate(self):
        with self.lock:
            self._require_focus_locked()
            views = self._views()
            if len(views) < MIN_VIEWS:
                self.last_calibration = {
                    "status": "NEED_MORE_VIEWS",
                    "message": f"至少需要 {MIN_VIEWS} 個不同姿態，目前只有 {len(views)} 個；尚未計算內參。",
                    "view_count": len(views),
                }
                return {"ok": False, **self.last_calibration}
            diversity = dataset_diversity(views, self.width, self.height)
            if not diversity["passed"]:
                self.last_calibration = {"status": "CALIBRATION_GATES_NOT_MET",
                                         "message": "；".join(diversity["failures"]),
                                         "view_count": len(views), "diversity": diversity}
                return {"ok": False, **self.last_calibration}
            object_points = [np.asarray(view["object_points"], dtype=np.float32) for view in views]
            image_points = [np.asarray(view["image_points"], dtype=np.float32) for view in views]
            try:
                result = cv2.aruco  # fail clearly if the installed OpenCV lacks contrib APIs
                del result
                fit = cv2.calibrateCameraExtended(
                    object_points, image_points, (self.width, self.height), None, None, flags=0)
            except Exception as exc:
                self.last_calibration = {"status": "CALIBRATION_FAILED",
                                         "message": f"內參求解失敗：{type(exc).__name__}: {exc}"}
                return {"ok": False, **self.last_calibration}
            rms, matrix, distortion, rvecs, tvecs, std_intrinsics, std_extrinsics, per_view = fit
            matrix = np.asarray(matrix, dtype=float)
            distortion = np.asarray(distortion, dtype=float).reshape(-1)
            std_intrinsics = np.asarray(std_intrinsics, dtype=float).reshape(-1)
            per_view = np.asarray(per_view, dtype=float).reshape(-1)
            if (matrix.shape != (3, 3) or distortion.size != 5 or std_intrinsics.size < 9
                    or per_view.size != len(views)
                    or not np.isfinite(matrix).all() or not np.isfinite(distortion).all()
                    or not np.isfinite(std_intrinsics[:9]).all() or not np.isfinite(per_view).all()
                    or not math.isfinite(float(rms))):
                self.last_calibration = {"status": "CALIBRATION_FAILED",
                                         "message": "求解結果或不確定度包含非有限值，未建立候選檔案。"}
                return {"ok": False, **self.last_calibration}
            reasons = []
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
                reasons.append("焦距必須為正值")
            if not (-0.1 * self.width <= matrix[0, 2] <= 1.1 * self.width
                    and -0.1 * self.height <= matrix[1, 2] <= 1.1 * self.height):
                reasons.append("主點落在合理影像範圍之外")
            if float(rms) > 1.0:
                reasons.append(f"整體 RMS {float(rms):.3f}px 超過 1px")
            bad_views = [index + 1 for index, error in enumerate(per_view) if error > 2.0]
            if bad_views:
                reasons.append("單姿態 RMS 超過 2px：" + ", ".join(map(str, bad_views)))
            focus = self.locked_focus
            result = {
                "schema_version": 1,
                "created_utc": _utc_now(),
                "status": CANDIDATE_STATUS,
                "quality_gates_passed": not reasons,
                "quality_gate_failures": reasons,
                "opencv_model": "ordinary_pinhole_5_coefficients",
                "image_size": [self.width, self.height],
                "K": matrix.tolist(),
                "distortion": distortion[:5].tolist(),
                "intrinsic_uncertainties_1sigma": {
                    "fx_px": float(std_intrinsics[0]), "fy_px": float(std_intrinsics[1]),
                    "cx_px": float(std_intrinsics[2]), "cy_px": float(std_intrinsics[3]),
                    "distortion": std_intrinsics[4:9].tolist(),
                },
                "overall_rms_px": float(rms),
                "per_view_rms_px": [float(value) for value in per_view],
                "view_ids": [view["view_id"] for view in views],
                "diversity": diversity,
                "camera_serial": self.camera_serial,
                "device_uri": self.device_uri,
                "resolved_device_node": self.resolved_node,
                "focus_value": focus,
                "board": self.board_config,
                "board_geometry_sha256": self.board_sha,
                "solver_camera_config": {
                    "K": matrix.tolist(), "distortion": distortion[:5].tolist(),
                    "image_size": [self.width, self.height],
                    "camera_serial": self.camera_serial,
                    "device_uri": self.device_uri,
                    "resolved_device_node": self.resolved_node,
                    "focus_value": focus,
                    "opencv_model": "ordinary_pinhole_5_coefficients",
                },
            }
            path = self._write_result(result)
            self.last_calibration = {
                "status": CANDIDATE_STATUS,
                "message": ("候選內參已保存；須獨立驗證。" if not reasons else
                            "候選結果已保存，但品質門檻未通過：" + "；".join(reasons)),
                "result_file": path.name, "overall_rms_px": float(rms),
                "view_count": len(views), "quality_gates_passed": not reasons,
                "matrix": matrix.tolist(), "distortion": distortion[:5].tolist(),
            }
            return {"ok": not reasons, **self.last_calibration}

    def _write_result(self, result):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for increment in range(1, 10000):
            path = self.output / f"intrinsics_{stamp}_{increment:03d}.json"
            try:
                _write_exclusive(path, _json_bytes(result))
                return path
            except FileExistsError:
                continue
        raise RuntimeError("無法分配唯一的內參結果檔名")

    def latest_candidate(self):
        """Return the latest passing candidate with its content hash for adoption review."""
        report = self.last_calibration
        if (not isinstance(report, dict) or report.get("status") != CANDIDATE_STATUS
                or report.get("quality_gates_passed") is not True
                or not isinstance(report.get("result_file"), str)):
            return None
        path = self.output / report["result_file"]
        if not path.is_file():
            return None
        content = self._valid_candidate_file(path)
        if content is None:
            return None
        payload = path.read_bytes()
        return {"result_file": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "contents": content}

    def _valid_candidate_file(self, path):
        output = self.output.resolve()
        resolved = path.resolve()
        if resolved.parent != output or not path.name.startswith("intrinsics_") or path.suffix != ".json":
            return None
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(content, dict):
                return None
            matrix = np.asarray(content["K"], dtype=float)
            distortion = np.asarray(content["distortion"], dtype=float)
            size = content["image_size"]
            errors = np.asarray(content["per_view_rms_px"], dtype=float)
            view_ids = content["view_ids"]
            uncertainty = content["intrinsic_uncertainties_1sigma"]
            uncertainty_values = np.asarray([
                uncertainty["fx_px"], uncertainty["fy_px"], uncertainty["cx_px"],
                uncertainty["cy_px"], *uncertainty["distortion"]], dtype=float)
            candidate_focus = float(content["focus_value"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        if (type(content.get("schema_version")) is not int or content["schema_version"] != 1
                or content.get("status") != CANDIDATE_STATUS
                or content.get("quality_gates_passed") is not True
                or content.get("quality_gate_failures") != []
                or content.get("opencv_model") != "ordinary_pinhole_5_coefficients"
                or content.get("camera_serial") != self.camera_serial
                or content.get("device_uri") != self.device_uri
                or size != [self.width, self.height]
                or content.get("board") != self.board_config
                or content.get("board_geometry_sha256") != self.board_sha
                or self.locked_focus is None
                or not math.isclose(candidate_focus, self.locked_focus, rel_tol=0, abs_tol=1e-3)):
            return None
        if (not isinstance(view_ids, list) or any(not isinstance(value, str) for value in view_ids)
                or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1], atol=1e-8, rtol=0)
                or abs(matrix[0, 1]) > 1e-8 or abs(matrix[1, 0]) > 1e-8
                or not (-0.1 * self.width <= matrix[0, 2] <= 1.1 * self.width)
                or not (-0.1 * self.height <= matrix[1, 2] <= 1.1 * self.height)
                or distortion.shape != (5,) or not np.isfinite(distortion).all()
                or uncertainty_values.shape != (9,) or not np.isfinite(uncertainty_values).all()
                or len(view_ids) < MIN_VIEWS or len(set(view_ids)) != len(view_ids)
                or errors.shape != (len(view_ids),)
                or not np.isfinite(errors).all() or np.any(errors > 2.0)):
            return None
        try:
            overall_rms = float(content["overall_rms_px"])
        except (KeyError, TypeError, ValueError):
            return None
        diversity = content.get("diversity")
        if (not math.isfinite(overall_rms) or overall_rms < 0 or overall_rms > 1.0
                or not isinstance(diversity, dict) or diversity.get("passed") is not True
                or not isinstance(content.get("created_utc"), str)):
            return None
        try:
            coverage = float(diversity["coverage_fraction"])
            center_spread = float(diversity["center_spread_px"])
            tilts = np.asarray(diversity["tilts_degrees"], dtype=float)
            tilt_spread = float(diversity["tilt_spread_degrees"])
        except (KeyError, TypeError, ValueError):
            return None
        if (not math.isfinite(coverage) or coverage < 0.20
                or not math.isfinite(center_spread) or center_spread < 0.10 * math.hypot(self.width, self.height)
                or tilts.ndim != 1 or not np.isfinite(tilts).all()
                or np.count_nonzero(tilts >= 8.0) < 3
                or np.max(tilts, initial=0.0) < 15.0
                or not math.isfinite(tilt_spread) or tilt_spread < 5.0):
            return None
        solver = content.get("solver_camera_config")
        if not isinstance(solver, dict):
            return None
        try:
            solver_k = np.asarray(solver["K"], dtype=float)
            solver_dist = np.asarray(solver["distortion"], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        if (solver.get("image_size") != size
                or solver.get("camera_serial") != self.camera_serial
                or solver.get("device_uri") != self.device_uri
                or solver.get("focus_value") != candidate_focus
                or solver.get("opencv_model") != "ordinary_pinhole_5_coefficients"
                or solver_k.shape != (3, 3) or not np.array_equal(solver_k, matrix)
                or solver_dist.shape != (5,) or not np.array_equal(solver_dist, distortion)):
            return None
        return content

    def _restore_latest_candidate(self):
        for path in sorted(self.output.glob("intrinsics_*.json"), reverse=True):
            content = self._valid_candidate_file(path)
            if content is None:
                continue
            self.last_calibration = {
                "status": CANDIDATE_STATUS,
                "message": "已恢復本工作階段通過品質門檻的候選內參；仍須獨立驗證。",
                "result_file": path.name,
                "overall_rms_px": content["overall_rms_px"],
                "view_count": len(content["view_ids"]),
                "quality_gates_passed": True,
                "matrix": content["K"],
                "distortion": content["distortion"],
            }
            return

    def state(self):
        frame = self.camera.latest()
        autofocus, focus = self.camera.control_state()
        views = self._views()
        autofocus_on = (bool(autofocus > 0.5) if math.isfinite(autofocus) else None)
        return {
            "frame_id": None if frame is None else frame.sequence,
            "frame_age_seconds": None if frame is None else max(0.0, time.monotonic() - frame.monotonic_time),
            "connected": bool(getattr(self.camera, "connected", frame is not None) and frame is not None),
            "marker_count": 0 if frame is None else frame.marker_count,
            "corner_count": 0 if frame is None else len(frame.corner_ids),
            "cache_ids": self._cached_ids(),
            "focus_state": {"autofocus": autofocus if math.isfinite(autofocus) else None,
                            "autofocus_on": autofocus_on,
                            "focus_value": focus if math.isfinite(focus) else None,
                            "locked": self.locked_focus is not None,
                            "locked_value": self.locked_focus},
            "view_count": len(views),
            "minimum_views": MIN_VIEWS,
            "solving": self.solving,
            "error": self.last_error or self.camera.error,
            "last_calibration": self.last_calibration,
        }

    def _cached_ids(self):
        with self.camera.cache_lock:
            return [frame.sequence for frame in self.camera.frames]


class LocalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(session, csrf_token):
    class Handler(BaseHTTPRequestHandler):
        server_version = "WristIntrinsics/1"
        sys_version = ""

        def log_message(self, fmt, *args):
            # Do not log request bodies, headers, or the per-session CSRF token.
            super().log_message(fmt, *args)

        def _send(self, status, content_type, payload):
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def _local_request(self):
            try:
                return ipaddress.ip_address(self.client_address[0]).is_loopback
            except ValueError:
                return False

        def do_GET(self):
            if not self._local_request():
                self._send(403, "text/plain; charset=utf-8", b"loopback only")
                return
            path = urlparse(self.path).path
            if path == "/":
                token = json.dumps(csrf_token)
                html = HTML_PAGE.replace("__CSRF_TOKEN__", token).encode("utf-8")
                self._send(200, "text/html; charset=utf-8", html)
            elif path == "/api/state":
                try:
                    self._send(200, "application/json; charset=utf-8", _json_bytes(session.state()))
                except Exception as exc:
                    self._send(500, "application/json; charset=utf-8",
                               _json_bytes({"error": f"{type(exc).__name__}: {exc}"}))
            elif path == "/api/observation":
                try:
                    self._send(200, "application/json; charset=utf-8",
                               _json_bytes(session.observation_payload()))
                except Exception as exc:
                    self._send(503, "application/json; charset=utf-8",
                               _json_bytes({"error": f"{type(exc).__name__}: {exc}"}))
            elif path == "/api/candidate":
                try:
                    candidate = session.latest_candidate()
                    if candidate is None:
                        self._send(404, "application/json; charset=utf-8",
                                   _json_bytes({"error": "no passing candidate is available"}))
                    else:
                        self._send(200, "application/json; charset=utf-8", _json_bytes(candidate))
                except Exception as exc:
                    self._send(500, "application/json; charset=utf-8",
                               _json_bytes({"error": f"{type(exc).__name__}: {exc}"}))
            elif path == "/frame.jpg":
                query = urlparse(self.path).query
                requested = None
                if query.startswith("frame_id="):
                    try:
                        requested = int(query.split("=", 1)[1])
                    except ValueError:
                        requested = -1
                frame = session.camera.latest() if requested is None else session.camera.cached(requested)
                if frame is None:
                    self._send(404, "text/plain; charset=utf-8", b"frame unavailable")
                    return
                annotated = frame.bgr.copy()
                if len(frame.corner_ids):
                    cv2.aruco.drawDetectedCornersCharuco(
                        annotated, frame.image_points.astype(np.float32).reshape(-1, 1, 2),
                        frame.corner_ids.astype(np.int32).reshape(-1, 1), (30, 220, 30))
                cv2.putText(annotated,
                            f"frame {frame.sequence} | markers {frame.marker_count} | corners {len(frame.corner_ids)}",
                            (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    self._send(500, "text/plain; charset=utf-8", b"JPEG encoding failed")
                else:
                    self._send(200, "image/jpeg", encoded.tobytes())
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")

        def do_POST(self):
            if not self._local_request():
                self._send(403, "application/json; charset=utf-8", b'{"error":"loopback only"}')
                return
            expected_origin = f"http://127.0.0.1:{self.server.server_port}"
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin", "")
            if host != f"127.0.0.1:{self.server.server_port}" or origin != expected_origin:
                self._send(403, "application/json; charset=utf-8", b'{"error":"origin rejected"}')
                return
            if not hmac.compare_digest(self.headers.get("X-CSRF-Token", ""), csrf_token):
                self._send(403, "application/json; charset=utf-8", b'{"error":"csrf rejected"}')
                return
            if self.headers.get_content_type() != "application/json":
                self._send(415, "application/json; charset=utf-8", b'{"error":"JSON required"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise ValueError("request body size rejected")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("request body must be an object")
                action = body.get("action")
                if action == "lock_focus":
                    result = session.lock_focus()
                elif action == "capture":
                    result = session.capture(body.get("frame_id"))
                elif action == "calibrate":
                    result = session.start_calibration()
                else:
                    raise ValueError("未知操作")
                session.last_error = None
                self._send(200, "application/json; charset=utf-8", _json_bytes(result))
            except (ValueError, RuntimeError, OSError, cv2.error) as exc:
                session.last_error = str(exc)
                self._send(400, "application/json; charset=utf-8",
                           _json_bytes({"ok": False, "error": str(exc)}))
            except Exception as exc:
                session.last_error = f"{type(exc).__name__}: {exc}"
                self._send(500, "application/json; charset=utf-8",
                           _json_bytes({"ok": False, "error": session.last_error}))

    return Handler


HTML_PAGE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>腕部相機內參收集</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;margin:20px;color:#17202a;background:#f4f6f8}main{max-width:920px;margin:auto}
h1{font-size:1.5rem;margin:0 0 8px}.note{background:#fff;border-left:4px solid #2678b2;padding:12px 16px;margin:12px 0}
img{display:block;width:720px;max-width:100%;height:auto;background:#111;border:1px solid #9aa4ad}
.row{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}button{font:inherit;padding:9px 14px;border:0;border-radius:5px;background:#145da0;color:white;cursor:pointer}
button:disabled{opacity:.45;cursor:not-allowed}pre{white-space:pre-wrap;background:#fff;padding:12px;border-radius:5px;min-height:48px}
.muted{color:#52616b;font-size:.92rem}
</style></head><body><main>
<h1>腕部相機內參收集</h1>
<div class="note">此階段只移動標定板，相機與手臂保持靜止。先把標定板完整放進畫面，再按「鎖定目前焦距」。對焦鎖定後保存至少 12 個有位置與傾角變化的姿態。之後手眼標定階段會固定標定板並移動手臂。</div>
<img id="preview" alt="相機即時畫面" width="720">
<p id="status">正在連線相機…</p>
<div class="row"><button id="lock">鎖定目前焦距</button><button id="capture" disabled>保存這一姿態</button><button id="calibrate" disabled>計算內參</button></div>
<p class="muted">每次保存使用畫面上顯示的同一張快取影格；超過 1 秒或與舊姿態接近的畫面會被拒絕。原始 BGR PNG 會與角點 ID、座標和時間一併保存。</p>
<pre id="message"></pre>
</main><script>
const csrf=__CSRF_TOKEN__;let latest=null;let refreshing=false;
async function refresh(){if(refreshing)return;refreshing=true;try{const r=await fetch('/api/state',{cache:'no-store'});const s=await r.json();latest=s;
 document.getElementById('status').textContent=`影格 ${s.frame_id??'—'}｜標記 ${s.marker_count}｜角點 ${s.corner_count}｜已保存 ${s.view_count}/${s.minimum_views}｜AF ${s.focus_state.autofocus_on?'開啟':'關閉'}｜焦距 ${s.focus_state.focus_value}`;
 const fresh=s.connected&&s.frame_id&&s.frame_age_seconds<=1;if(fresh)document.getElementById('preview').src='/frame.jpg?frame_id='+s.frame_id;else document.getElementById('preview').removeAttribute('src');
 document.getElementById('lock').disabled=!!s.solving;
 document.getElementById('capture').disabled=!!s.solving||!s.focus_state.locked||!fresh;
 document.getElementById('calibrate').disabled=!!s.solving||!s.focus_state.locked||s.view_count<s.minimum_views;
 const msg=document.getElementById('message');
 if(s.solving)msg.textContent='內參計算中，請稍候…';
 else if(s.error)msg.textContent=s.error;
 else if(s.last_calibration){const c=s.last_calibration;const rms=Number.isFinite(c.overall_rms_px)?`（整體 RMS ${c.overall_rms_px.toFixed(3)} px）`:'';msg.textContent=(c.message||c.status||'')+rms}
 }catch(e){document.getElementById('status').textContent='無法讀取相機狀態：'+e}finally{refreshing=false}}
async function action(action,extra={}){const msg=document.getElementById('message');msg.textContent='處理中…';try{const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':csrf},body:JSON.stringify({action,...extra})});const data=await r.json();msg.textContent=data.message||data.error||(data.ok?(action==='lock_focus'?'焦距已锁定，可以开始保存姿态。':action==='capture'?`已保存第 ${data.view_count} 个姿态，请换一个位置或倾角。`:'操作完成。'):'操作未完成，请检查采集条件。');await refresh()}catch(e){msg.textContent=String(e)}}
document.getElementById('lock').onclick=()=>action('lock_focus');
document.getElementById('capture').onclick=()=>latest&&action('capture',{frame_id:latest.frame_id});
document.getElementById('calibrate').onclick=()=>action('calibrate');
refresh();setInterval(refresh,300);
</script></body></html>"""


def _load_board_argument(argument):
    if argument is None:
        return normalize_board_config()
    candidate = Path(argument).expanduser()
    try:
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
        else:
            value = json.loads(argument)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("--board 須為 board JSON 檔案路徑或 JSON 文字") from exc
    return normalize_board_config(value)


def _localhost_host(host):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("只允許 --host 127.0.0.1 或 localhost")
    return "127.0.0.1"


def build_parser():
    parser = argparse.ArgumentParser(description="Local wrist-camera ChArUco intrinsic collector")
    parser.add_argument("--device", required=True, help="one stable /dev/v4l/by-id/... camera path")
    parser.add_argument("--output", required=True, help="new or matching owned session directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--board", help="supported board JSON file or inline JSON")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        host = _localhost_host(args.host)
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be in 1..65535")
        if args.width <= 0 or args.height <= 0:
            raise ValueError("image dimensions must be positive")
        device_path = Path(args.device).expanduser()
        if not device_path.exists():
            raise ValueError("--device must name an existing explicit V4L device path")
        resolved_node = str(device_path.resolve(strict=True))
        board = _load_board_argument(args.board)
        detector = CharucoDetector(board)
        camera = CameraWorker(str(device_path), args.width, args.height, detector)
        serial = _camera_serial(str(device_path), resolved_node)
        session = IntrinsicsSession(args.output, str(device_path), resolved_node, serial,
                                    args.width, args.height, board, detector, camera)
        camera.start()
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 2
    csrf_token = secrets.token_urlsafe(32)
    server = None
    try:
        server = LocalHTTPServer((host, args.port), make_handler(session, csrf_token))
        print(f"UI: http://127.0.0.1:{server.server_port}/")
        print(f"Session: {session.output}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        print(f"server error: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 2
    finally:
        if server is not None:
            server.server_close()
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
