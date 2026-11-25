#!/usr/bin/env python3
"""Offline fixed-main-camera / moving-wrist-camera shared-board solver.

The manifest contains calibrated camera intrinsics and image observations of
one stationary 5-by-8 ChArUco board. ``T_base_from_gripper`` must be computed
with the official SO101+v2 forward kinematics for the matching arm. This tool
does not read a robot, load a runtime transform, or fit joint zero offsets.

For each arm the fitted transforms are B = T_main_from_base,
X = T_gripper_from_wrist, and Q = T_main_from_board. Every main and wrist
corner is optimized with OpenCV's distortion model. Acceptance measures
reprojection consistency only, not absolute calibration accuracy.

Usage: python shared_board.py --manifest samples.json --output report.json
The output is created exclusively and is never written into runtime config.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ARM_NAMES = ("so101_left", "so101_right")
BOARD = {"squares_x": 5, "squares_y": 8, "square_length_m": 0.024,
         "marker_length_m": 0.017, "dictionary": "DICT_4X4_50",
         "legacy_pattern": False}
TRANSLATION_SCALE_M = 0.1
RANK_RATIO_MIN = 1e-5
ROTATION_AXIS_RATIO_MIN = 0.05
POSE_TRANSLATION_TOLERANCE_M = 0.001
POSE_ROTATION_TOLERANCE_DEG = 1.0
MAX_NFEV = 1500
DEFAULT_GATES = {"training_rms_px_max": 2.0,
                 "training_max_frame_rms_px": 4.0,
                 "heldout_max_frame_rms_px": 4.0,
                 "minimum_pose_groups": 6,
                 "minimum_training_pose_groups": 4,
                 "jacobian_rank_ratio_min": RANK_RATIO_MIN,
                 "rotation_axis_ratio_min": ROTATION_AXIS_RATIO_MIN,
                 "shared_board_translation_delta_m_max": 0.010,
                 "shared_board_rotation_delta_deg_max": 2.0}


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a finite numeric array") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite array of shape {shape}")
    return result


def _transform(value, name):
    matrix = _array(value, (4, 4), name)
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)):
        raise ValueError(f"{name}: expected a proper SE(3) transform")
    return matrix


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected a nonempty string")
    return value


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: expected a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}: must be at least {minimum}")
    return float(value)


def _board_corners():
    """Return row-major internal-corner coordinates from the board's top-left."""
    return np.array([[0.024 * (column + 1), 0.024 * (row + 1), 0.0]
                     for row in range(7) for column in range(4)], dtype=float)


def _corner_id(point):
    nearest = _board_corners()
    distances = np.linalg.norm(nearest - point, axis=1)
    index = int(np.argmin(distances))
    if distances[index] > 1e-8:
        return None
    return index


def _camera_observation(value, prefix):
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}: expected a camera observation object")
    required = ("K", "distortion", "image_size", "camera_serial", "object_points",
                "image_points", "T_camera_from_board_seed")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"{prefix}: missing required fields: {', '.join(missing)}")
    serial = _label(value["camera_serial"], f"{prefix}.camera_serial")
    image_size = value["image_size"]
    if (not isinstance(image_size, list) or len(image_size) != 2
            or any(type(side) is not int or side <= 0 for side in image_size)):
        raise ValueError(f"{prefix}.image_size: expected positive integer [width, height]")
    k = _array(value["K"], (3, 3), f"{prefix}.K")
    if (k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1], atol=1e-10, rtol=0)
            or abs(k[0, 1]) > 1e-10 or abs(k[1, 0]) > 1e-10):
        raise ValueError(f"{prefix}.K: expected pinhole intrinsics with positive focal lengths")
    try:
        distortion = np.asarray(value["distortion"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}.distortion: expected finite OpenCV coefficients") from exc
    if (distortion.ndim != 1 or len(distortion) not in (0, 4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()):
        raise ValueError(f"{prefix}.distortion: expected 0, 4, 5, 8, 12, or 14 finite coefficients")
    object_points = np.asarray(value["object_points"], dtype=float)
    image_points = np.asarray(value["image_points"], dtype=float)
    if (object_points.ndim != 2 or object_points.shape[1:] != (3,)
            or image_points.ndim != 2 or image_points.shape != (len(object_points), 2)
            or len(object_points) < 6 or not np.isfinite(object_points).all()
            or not np.isfinite(image_points).all()):
        raise ValueError(f"{prefix}: expected at least 6 finite corresponding Nx3 and Nx2 point arrays")
    if (np.any(image_points[:, 0] < 0) or np.any(image_points[:, 0] >= image_size[0])
            or np.any(image_points[:, 1] < 0) or np.any(image_points[:, 1] >= image_size[1])):
        raise ValueError(f"{prefix}.image_points: points must lie inside the declared image size")
    ids = []
    for point in object_points:
        corner_id = _corner_id(point)
        if corner_id is None:
            raise ValueError(f"{prefix}.object_points: every point must be a 24 mm board internal corner")
        ids.append(corner_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{prefix}.object_points: duplicate board corner coordinates")
    if np.linalg.matrix_rank(object_points[:, :2] - object_points[:, :2].mean(axis=0), tol=1e-9) < 2:
        raise ValueError(f"{prefix}.object_points: at least 6 noncollinear board corners are required")
    explicit_ids = value.get("corner_ids")
    if explicit_ids is not None:
        if (not isinstance(explicit_ids, list) or len(explicit_ids) != len(ids)
                or any(type(item) is not int for item in explicit_ids)
                or explicit_ids != ids):
            raise ValueError(f"{prefix}.corner_ids: IDs must match the supplied row-major board coordinates")
    return {"serial": serial, "image_size": tuple(image_size), "K": k,
            "distortion": distortion, "object_points": object_points,
            "image_points": image_points,
            "seed": _transform(value["T_camera_from_board_seed"],
                               f"{prefix}.T_camera_from_board_seed"),
            "corner_ids": ids}


def _parse_gates(value):
    gates = dict(DEFAULT_GATES)
    if value is None:
        return gates
    if not isinstance(value, dict) or set(value) - set(DEFAULT_GATES):
        raise ValueError("gates: expected an object containing only supported gate names")
    for name, default in DEFAULT_GATES.items():
        if name not in value:
            continue
        if name.startswith("minimum_"):
            item = value[name]
            if type(item) is not int or item < default:
                raise ValueError(f"gates.{name}: must be an integer at least {default}")
            gates[name] = item
        else:
            item = _number(value[name], f"gates.{name}", minimum=0.0)
            if item == 0 or (name in ("jacobian_rank_ratio_min", "rotation_axis_ratio_min")
                             and item < default):
                raise ValueError(f"gates.{name}: may only tighten the default gate {default}")
            if name in ("training_rms_px_max", "training_max_frame_rms_px",
                        "heldout_max_frame_rms_px", "shared_board_translation_delta_m_max",
                        "shared_board_rotation_delta_deg_max") and item > default:
                raise ValueError(f"gates.{name}: may only tighten the default gate {default}")
            gates[name] = item
    return gates


def validate_manifest(manifest):
    """Validate the complete manifest and return normalized observations."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    _label(manifest.get("setup_id"), "setup_id")
    if manifest.get("setup_confirmed_unchanged") is not True:
        raise ValueError("setup_confirmed_unchanged must be true")
    board = manifest.get("board")
    if not isinstance(board, dict):
        raise ValueError("board must describe the shared ChArUco board")
    for key, expected in BOARD.items():
        actual = board.get(key)
        if isinstance(expected, float):
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
                    float(actual), expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"board.{key} must be {expected}")
        elif actual != expected or type(actual) is not type(expected):
            raise ValueError(f"board.{key} must be {expected!r}")
    gates = _parse_gates(manifest.get("gates"))
    arms = manifest.get("arms")
    if not isinstance(arms, dict) or not arms or set(arms) - set(ARM_NAMES):
        raise ValueError("arms must contain so101_left and/or so101_right")
    normalized = {}
    camera_identity = {"main": [], "wrist": {}}
    for arm, arm_data in arms.items():
        if not isinstance(arm_data, dict) or not isinstance(arm_data.get("samples"), list) or not arm_data["samples"]:
            raise ValueError(f"arms.{arm}.samples must be a nonempty list")
        samples, seen_ids = [], set()
        for raw in arm_data["samples"]:
            if not isinstance(raw, dict):
                raise ValueError(f"arms.{arm}: each sample must be an object")
            sample_id = _label(raw.get("id"), f"arms.{arm}.sample.id")
            if sample_id in seen_ids:
                raise ValueError(f"arms.{arm}: duplicate sample id {sample_id}")
            seen_ids.add(sample_id)
            group = _label(raw.get("pose_group"), f"{arm}.{sample_id}.pose_group")
            f = _transform(raw.get("T_base_from_gripper"),
                           f"{arm}.{sample_id}.T_base_from_gripper")
            cameras = {}
            for role in ("main", "wrist"):
                if role not in raw:
                    raise ValueError(f"{arm}.{sample_id}: missing {role} camera board observation")
                camera = _camera_observation(raw[role], f"{arm}.{sample_id}.{role}")
                if role == "main":
                    camera_identity[role].append((arm, sample_id, camera))
                else:
                    camera_identity[role].setdefault(arm, []).append((sample_id, camera))
                cameras[role] = camera
            if cameras["main"]["serial"] == cameras["wrist"]["serial"]:
                raise ValueError(f"{arm}.{sample_id}: main and wrist cameras must have distinct serial identities")
            samples.append({"id": sample_id, "pose_group": group, "F": f, **cameras})
        normalized[arm] = samples
    entries = camera_identity["main"]
    first = entries[0][2]
    for arm, sample_id, camera in entries[1:]:
        if (camera["serial"] != first["serial"]
                or camera["image_size"] != first["image_size"]
                or not np.array_equal(camera["K"], first["K"])
                or not np.array_equal(camera["distortion"], first["distortion"])):
            raise ValueError(f"{arm}.{sample_id}.main: camera identity, size, and intrinsics must stay fixed")
    for arm, entries in camera_identity["wrist"].items():
        first = entries[0][1]
        for sample_id, camera in entries[1:]:
            if (camera["serial"] != first["serial"]
                    or camera["image_size"] != first["image_size"]
                    or not np.array_equal(camera["K"], first["K"])
                    or not np.array_equal(camera["distortion"], first["distortion"])):
                raise ValueError(f"{arm}.{sample_id}.wrist: camera identity, size, and intrinsics must stay fixed")
    return normalized, gates


def _pack(transform):
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
                 transform[:3, 3] / TRANSLATION_SCALE_M]


def _unpack(parameters):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    matrix[:3, 3] = parameters[3:6] * TRANSLATION_SCALE_M
    return matrix


def _mean_transform(transforms):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(np.array([item[:3, :3] for item in transforms])).mean().as_matrix()
    result[:3, 3] = np.array([item[:3, 3] for item in transforms]).mean(axis=0)
    return result


def _initial_parameters(observations):
    q = _mean_transform([obs["main"]["seed"] for obs in observations])
    relatives = [obs["main"]["seed"] @ np.linalg.inv(obs["wrist"]["seed"])
                 for obs in observations]
    x = np.eye(4)
    initialized = False
    diagnostic = None
    if len(observations) >= 3:
        try:
            rotations, translation = cv2.calibrateHandEye(
                [obs["F"][:3, :3] for obs in observations],
                [obs["F"][:3, 3].reshape(3, 1) for obs in observations],
                [np.linalg.inv(y)[:3, :3] for y in relatives],
                [np.linalg.inv(y)[:3, 3].reshape(3, 1) for y in relatives],
                method=cv2.CALIB_HAND_EYE_PARK)
            candidate_x = np.eye(4)
            candidate_x[:3, :3] = np.asarray(rotations, dtype=float)
            candidate_x[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
            x = _transform(candidate_x, "OpenCV hand-eye initialization")
            initialized = True
        except (cv2.error, ValueError, np.linalg.LinAlgError, TypeError) as exc:
            diagnostic = f"OpenCV hand-eye seed unavailable: {exc}"
    b = _mean_transform([y @ np.linalg.inv(obs["F"] @ x)
                         for obs, y in zip(observations, relatives)])
    return np.r_[_pack(b), _pack(x), _pack(q)], {"opencv_handeye_used": initialized,
                                                  "warning": diagnostic}


def _project(points_camera, camera):
    z = points_camera[:, 2]
    safe_z = np.where(np.abs(z) < 1e-8, np.where(z < 0, -1e-8, 1e-8), z)
    normalized = points_camera[:, :2] / safe_z[:, None]
    normalized = np.clip(normalized, -1e4, 1e4)
    normalized_points = np.column_stack((normalized, np.ones(len(normalized))))
    pixels, _ = cv2.projectPoints(normalized_points, np.zeros(3), np.zeros(3),
                                  camera["K"], camera["distortion"])
    return pixels.reshape(-1, 2), z


def _camera_points(parameters, observation, role):
    b, x, q = (_unpack(parameters[:6]), _unpack(parameters[6:12]),
               _unpack(parameters[12:18]))
    if role == "main":
        transform = q
    else:
        # inv(T_main_from_wrist) @ T_main_from_board = T_wrist_from_board.
        main_from_wrist = b @ observation["F"] @ x
        transform = np.linalg.inv(main_from_wrist) @ q
    object_points = observation[role]["object_points"]
    return object_points @ transform[:3, :3].T + transform[:3, 3]


def _residual(parameters, observations):
    residuals = []
    for observation in observations:
        for role in ("main", "wrist"):
            camera = observation[role]
            points, _ = _project(_camera_points(parameters, observation, role), camera)
            residuals.append((points - camera["image_points"]).ravel())
    result = np.concatenate(residuals)
    if not np.isfinite(result).all():
        raise ValueError("projection produced nonfinite pixel residuals")
    return result


def _jacobian_report(parameters, observations, minimum_ratio=RANK_RATIO_MIN):
    columns = []
    for index in range(18):
        delta = np.zeros(18)
        step = 1e-6 * max(1.0, abs(parameters[index]))
        delta[index] = step
        columns.append((_residual(parameters + delta, observations)
                        - _residual(parameters - delta, observations)) / (2 * step))
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if len(singular) == 18 and singular[0] > 0 else 0.0
    return {"singular_values": singular.tolist(), "smallest_to_largest_ratio": ratio,
            "required_ratio_gt": minimum_ratio, "full_rank": ratio > minimum_ratio,
            "parameter_units": "rotation radians; translation / 0.1 m"}


def _fit(observations, minimum_ratio=RANK_RATIO_MIN):
    initial, initialization = _initial_parameters(observations)
    result = least_squares(_residual, initial, args=(observations,), loss="soft_l1",
                           f_scale=1.0, jac="3-point", max_nfev=MAX_NFEV,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    jacobian = _jacobian_report(result.x, observations, minimum_ratio)
    return result, jacobian, initialization


def _metrics(parameters, observations):
    frame_rows, squared_errors = [], []
    all_depths = []
    for observation in observations:
        for role in ("main", "wrist"):
            camera = observation[role]
            points, depth = _project(_camera_points(parameters, observation, role), camera)
            error = points - camera["image_points"]
            corner_error = np.linalg.norm(error, axis=1)
            frame_rms = float(np.sqrt(np.mean(corner_error ** 2)))
            squared_errors.extend(np.sum(error * error, axis=1).tolist())
            all_depths.extend(depth.tolist())
            frame_rows.append({"sample_id": observation["id"], "pose_group": observation["pose_group"],
                               "camera": role, "corner_count": int(len(depth)),
                               "rms_px": frame_rms, "minimum_depth_m": float(depth.min()),
                               "all_positive_depths": bool(np.all(depth > 0))})
    values = np.asarray(squared_errors, dtype=float)
    depths = np.asarray(all_depths, dtype=float)
    per_camera = {}
    for role in ("main", "wrist"):
        frames = [frame["rms_px"] for frame in frame_rows if frame["camera"] == role]
        role_sq = []
        for observation in observations:
            camera = observation[role]
            points, _ = _project(_camera_points(parameters, observation, role), camera)
            role_sq.extend(np.sum((points - camera["image_points"]) ** 2, axis=1).tolist())
        per_camera[role] = {"rms_px": float(np.sqrt(np.mean(role_sq))),
                            "max_frame_rms_px": float(max(frames))}
    return {"rms_px": float(np.sqrt(np.mean(values))),
            "max_frame_rms_px": float(max(frame["rms_px"] for frame in frame_rows)),
            "all_positive_depths": bool(np.all(depths > 0)), "frames": frame_rows,
            "per_camera": per_camera}


def _excitation(observations, axis_ratio_min=ROTATION_AXIS_RATIO_MIN):
    representatives = {}
    group_consistent = True
    for observation in observations:
        group = observation["pose_group"]
        if group not in representatives:
            representatives[group] = observation["F"]
        else:
            prior = representatives[group]
            translation = np.linalg.norm(prior[:3, 3] - observation["F"][:3, 3])
            angle = math.degrees(Rotation.from_matrix(prior[:3, :3].T @ observation["F"][:3, :3]).magnitude())
            group_consistent &= (translation <= POSE_TRANSLATION_TOLERANCE_M
                                 and angle <= POSE_ROTATION_TOLERANCE_DEG)
    labels = sorted(representatives)
    neighbors = {label: set() for label in labels}
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            a, b = representatives[left], representatives[right]
            translation = np.linalg.norm(a[:3, 3] - b[:3, 3])
            angle = math.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
            if translation <= POSE_TRANSLATION_TOLERANCE_M and angle <= POSE_ROTATION_TOLERANCE_DEG:
                neighbors[left].add(right)
                neighbors[right].add(left)
    components, unseen = [], set(labels)
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            label = pending.pop()
            if label in component:
                continue
            component.add(label)
            pending.extend(neighbors[label] - component)
        unseen -= component
        components.append(sorted(component))
    transforms = [representatives[group[0]] for group in components]
    vectors = []
    for index in range(1, len(transforms)):
        vector = Rotation.from_matrix(transforms[0][:3, :3].T @ transforms[index][:3, :3]).as_rotvec()
        if np.linalg.norm(vector) >= math.radians(1.0):
            vectors.append(vector)
    singular = np.zeros(3)
    if vectors:
        values = np.linalg.svd(np.asarray(vectors), compute_uv=False)
        singular[:len(values)] = values
    axis_ratio = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
    max_angle = math.degrees(max((np.linalg.norm(vector) for vector in vectors), default=0.0))
    return {"pose_group_count": len(labels), "materially_distinct_fk_pose_count": len(components),
            "duplicate_pose_groups": [group for group in components if len(group) > 1],
            "group_fk_consistent": bool(group_consistent),
            "duplicate_translation_tolerance_m": POSE_TRANSLATION_TOLERANCE_M,
            "duplicate_rotation_tolerance_deg": POSE_ROTATION_TOLERANCE_DEG,
            "max_rotation_from_reference_deg": max_angle,
            "relative_rotvec_singular_values": singular.tolist(),
            "second_to_first_axis_ratio": axis_ratio,
            "required_axis_ratio_gt": axis_ratio_min,
            "nonparallel_rotation_axes": axis_ratio > axis_ratio_min,
            "sufficient": (len(components) >= 2 and group_consistent
                           and axis_ratio > axis_ratio_min)}


def _validate_groups(observations, gates):
    groups = sorted({obs["pose_group"] for obs in observations})
    folds = []
    for group in groups:
        training = [obs for obs in observations if obs["pose_group"] != group]
        heldout = [obs for obs in observations if obs["pose_group"] == group]
        fold = {"heldout_pose_group": group, "training_group_count": len(groups) - 1,
                "training_sample_ids": [obs["id"] for obs in training],
                "heldout_sample_ids": [obs["id"] for obs in heldout]}
        if len(groups) - 1 < gates["minimum_training_pose_groups"]:
            fold.update(passed=False, reason="training_subset_has_too_few_pose_groups")
        else:
            try:
                result, jacobian, _ = _fit(training, gates["jacobian_rank_ratio_min"])
                excitation = _excitation(training, gates["rotation_axis_ratio_min"])
                prediction = _metrics(result.x, heldout)
                fold.update(optimizer_success=bool(result.success), training_jacobian=jacobian,
                            training_excitation=excitation, prediction=prediction,
                            passed=bool(result.success and jacobian["full_rank"]
                                        and excitation["sufficient"]
                                        and prediction["all_positive_depths"]
                                        and prediction["max_frame_rms_px"] <= gates["heldout_max_frame_rms_px"]))
            except (ValueError, cv2.error, np.linalg.LinAlgError, FloatingPointError,
                    RuntimeError, TypeError) as exc:
                fold.update(optimizer_success=False, passed=False,
                            reason=f"training_fit_or_prediction_failed: {exc}")
        folds.append(fold)
    maximum = max((fold["prediction"]["max_frame_rms_px"] for fold in folds
                   if "prediction" in fold), default=None)
    return {"method": "leave_one_pose_group_out", "folds": folds,
            "max_heldout_frame_rms_px": maximum,
            "passed": bool(folds) and all(fold["passed"] for fold in folds)}


def _candidate(parameters):
    return {"T_main_from_base": _unpack(parameters[:6]).tolist(),
            "T_gripper_from_wrist": _unpack(parameters[6:12]).tolist(),
            "T_main_from_board": _unpack(parameters[12:18]).tolist()}


def _solve_arm(arm, observations, gates):
    reasons, warnings = [], []
    groups = {obs["pose_group"] for obs in observations}
    excitation = _excitation(observations, gates["rotation_axis_ratio_min"])
    result = None
    try:
        result, jacobian, initialization = _fit(observations, gates["jacobian_rank_ratio_min"])
        training = _metrics(result.x, observations)
        candidate = _candidate(result.x)
    except (ValueError, cv2.error, np.linalg.LinAlgError, FloatingPointError,
            RuntimeError, TypeError) as exc:
        initial, initialization = _initial_parameters(observations)
        candidate = _candidate(initial)
        training = None
        jacobian = {"full_rank": False, "smallest_to_largest_ratio": 0.0,
                    "required_ratio_gt": gates["jacobian_rank_ratio_min"], "singular_values": []}
        warnings.append(f"optimizer or projection failed: {exc}")
    if not initialization["opencv_handeye_used"] and initialization["warning"]:
        warnings.append(initialization["warning"])
    if len(groups) < gates["minimum_pose_groups"]:
        reasons.append("too_few_distinct_pose_groups")
    if excitation["materially_distinct_fk_pose_count"] < gates["minimum_pose_groups"]:
        reasons.append("too_few_materially_distinct_fk_poses")
    if not excitation["group_fk_consistent"]:
        reasons.append("pose_group_contains_different_fk_poses")
    if not excitation["sufficient"]:
        reasons.append("insufficient_nonparallel_rotation_excitation")
    if not jacobian["full_rank"]:
        reasons.append("scaled_18_parameter_jacobian_not_full_rank")
    if result is None or not result.success:
        reasons.append("optimizer_did_not_converge")
    if training is not None:
        if not training["all_positive_depths"]:
            reasons.append("nonpositive_camera_corner_depth")
        if training["rms_px"] > gates["training_rms_px_max"]:
            reasons.append("training_rms_exceeds_gate")
        if training["max_frame_rms_px"] > gates["training_max_frame_rms_px"]:
            reasons.append("training_frame_rms_exceeds_gate")
        validation = _validate_groups(observations, gates)
    else:
        validation = {"method": "leave_one_pose_group_out", "folds": [], "passed": False,
                      "reason": "full_data_fit_failed"}
    if not validation["passed"]:
        if any(fold.get("training_group_count", 0) < gates["minimum_training_pose_groups"]
               or not fold.get("training_jacobian", {}).get("full_rank", False)
               or not fold.get("training_excitation", {}).get("sufficient", False)
               for fold in validation["folds"]):
            reasons.append("leave_one_group_out_training_subset_insufficient_or_rank_deficient")
        else:
            reasons.append("leave_one_group_out_prediction_failed")
    accepted = not reasons
    report = {"arm_id": arm, "scope": "one_arm", "assessment": (
                  "VALIDATED_SHARED_BOARD" if accepted else
                  "NEED_MORE_POSES" if any(reason in reasons for reason in (
                      "too_few_distinct_pose_groups", "too_few_materially_distinct_fk_poses",
                      "insufficient_nonparallel_rotation_excitation",
                      "scaled_18_parameter_jacobian_not_full_rank",
                      "leave_one_group_out_training_subset_insufficient_or_rank_deficient"))
                  else "REJECTED_FIT"),
              "accepted": accepted, "reasons": reasons, "warnings": warnings,
              "sample_count": len(observations), "pose_group_count": len(groups),
              "excitation": excitation, "jacobian": jacobian, "training": training,
              "validation": validation, "candidate": candidate,
              "optimizer": None if result is None else {
                  "success": bool(result.success), "message": str(result.message),
                  "nfev": int(result.nfev), "loss": "soft_l1", "f_scale_px": 1.0}}
    if accepted:
        report["validated_transforms"] = candidate
    return report


def solve_manifest(manifest):
    """Return a JSON-serializable candidate/acceptance report without I/O."""
    observations, gates = validate_manifest(manifest)
    reports = {arm: _solve_arm(arm, observations[arm], gates)
               for arm in sorted(observations)}
    cross_arm = {"applicable": len(reports) > 1, "evaluated": False,
                 "passed": None, "pairs": [],
                 "translation_delta_m_max": gates["shared_board_translation_delta_m_max"],
                 "rotation_delta_deg_max": gates["shared_board_rotation_delta_deg_max"]}
    if len(reports) == 1:
        cross_arm["reason"] = "single_arm_scope"
    elif all(report["accepted"] for report in reports.values()):
        cross_arm["evaluated"] = True
        names = sorted(reports)
        for index, left_name in enumerate(names):
            left = reports[left_name]["candidate"]["T_main_from_board"]
            for right_name in names[index + 1:]:
                right = reports[right_name]["candidate"]["T_main_from_board"]
                translation_delta = float(np.linalg.norm(np.asarray(left)[:3, 3]
                                                         - np.asarray(right)[:3, 3]))
                rotation_delta = math.degrees(Rotation.from_matrix(
                    np.asarray(left)[:3, :3].T @ np.asarray(right)[:3, :3]).magnitude())
                cross_arm["pairs"].append({"left_arm": left_name, "right_arm": right_name,
                                           "translation_delta_m": translation_delta,
                                           "rotation_delta_deg": rotation_delta,
                                           "passed": bool(translation_delta <= gates["shared_board_translation_delta_m_max"]
                                                          and rotation_delta <= gates["shared_board_rotation_delta_deg_max"])})
        cross_arm["passed"] = all(pair["passed"] for pair in cross_arm["pairs"])
        if not cross_arm["passed"]:
            for report in reports.values():
                report["accepted"] = False
                report["assessment"] = "REJECTED_FIT"
                report["reasons"].append("shared_main_board_pose_inconsistent_between_arms")
                report.pop("validated_transforms", None)
    else:
        cross_arm["reason"] = "pending_arm_fit"
    assessments = [report["assessment"] for report in reports.values()]
    status = ("NEED_MORE_POSES" if "NEED_MORE_POSES" in assessments else
              "REJECTED_FIT" if "REJECTED_FIT" in assessments else "VALIDATED_SHARED_BOARD")
    accepted = all(report["accepted"] for report in reports.values())
    return {"schema_version": 1, "setup_id": manifest["setup_id"],
            "setup_confirmed_unchanged": True, "board": dict(BOARD),
            "status": status, "accepted": accepted,
            "gates": gates, "cross_arm_board_pose_consistency": cross_arm,
            "accuracy_scope": "Reprojection and heldout consistency only; not absolute calibration accuracy.",
            "arms": reports}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        with args.manifest.open(encoding="utf-8") as stream:
            manifest = json.load(stream, object_pairs_hook=_unique_json_object)
        report = solve_manifest(manifest)
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except (OSError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        print(f"shared_board: {exc}", file=sys.stderr)
        return 2
    print(f"{report['status']}: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
