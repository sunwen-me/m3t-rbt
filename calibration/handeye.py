"""Offline fixed-camera hand-eye calibration from encoder FK and image corners.

Model: T_camera_from_marker = B @ T_base_from_gripper @ X, where
B = T_camera_from_base and X = T_gripper_from_marker are fitted per arm.
No tracker poses, hardware, files other than the explicit manifest, or historical
data are consulted. Marker poses in the manifest are initialization seeds only.

Corners must be in decoded ArUco order: (-s/2,+s/2), (+s/2,+s/2),
(+s/2,-s/2), (-s/2,-s/2), at marker z=0. Pixel coordinates and K must
describe the same pinhole/undistorted image (distortion is not fitted).
Translations are metres. Each sample is an independent saved snapshot; repeated
pose_group labels are allowed and are held out together. Groups with duplicate
FK (within 1 mm and 1 degree) never count as extra independent poses.
Validation measures prediction consistency, not absolute accuracy.

Optional depth_plane observations contain normal_camera (unit vector), offset_m,
and sigma_m > 0, with equation n dot p + offset = 0. They must come from an
independent registered depth patch (for example TLS), never from PnP seeds or
fitted B/X. The solver consumes these external measurements without deriving
planes. Each plane adds four corner distances / (sigma_m * sqrt(4)); pixels
remain eight raw residuals. Missing planes retain RGB-only behavior.

Usage: python handeye.py --manifest samples.json --output result.json
The output must not already exist. A nonvalidated numerical fit is explicitly a
candidate, and no accepted transforms are emitted unless every observed arm
passes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ARMS = ("so101_left", "so101_right")
TRANSLATION_SCALE_M = 0.1
RANK_RATIO_MIN = 1e-5
POSE_TRANSLATION_TOLERANCE_M = 0.001
POSE_ROTATION_TOLERANCE_DEG = 1.0
MAX_NFEV = 1000


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
            or not math.isclose(np.linalg.det(rotation), 1.0, abs_tol=1e-6)):
        raise ValueError(f"{name}: expected a proper SE(3) transform")
    return matrix


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected a nonempty string")
    return value


def _validate_plane(plane, name):
    if not isinstance(plane, dict):
        raise ValueError(f"{name}: expected a measured depth-plane object")
    normal = _array(plane.get("normal_camera"), (3,), f"{name}.normal_camera")
    if not math.isclose(float(np.linalg.norm(normal)), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError(f"{name}.normal_camera: expected a unit normal")
    for key in ("offset_m", "sigma_m"):
        value = plane.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name}.{key}: expected a finite number")
    if plane["sigma_m"] <= 0:
        raise ValueError(f"{name}.sigma_m: expected a positive value")


def validate_manifest(manifest):
    """Check inputs and optional measured planes; ignore unrelated fields."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if manifest.get("setup_confirmed_unchanged") is not True:
        raise ValueError("setup_confirmed_unchanged must be true")
    _label(manifest.get("setup_id"), "setup_id")
    side = manifest.get("marker_side_m")
    if isinstance(side, bool) or not isinstance(side, (int, float)) or not math.isfinite(side) or side <= 0:
        raise ValueError("marker_side_m must be finite and positive")
    nominal = manifest.get("nominal_T_gripper_from_marker")
    if not isinstance(nominal, dict):
        raise ValueError("nominal_T_gripper_from_marker must be an object")
    if set(nominal) - set(ARMS):
        raise ValueError("nominal_T_gripper_from_marker: unsupported arm name")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a nonempty list")
    ids = set()
    present_arms = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("each sample must be an object")
        sid = _label(sample.get("id"), "sample.id")
        if sid in ids:
            raise ValueError(f"duplicate sample id: {sid}")
        ids.add(sid)
        _label(sample.get("pose_group"), f"{sid}.pose_group")
        arms = sample.get("arms")
        if not isinstance(arms, dict) or not arms or set(arms) - set(ARMS):
            raise ValueError(f"{sid}.arms: expected observations for supported arms")
        for arm, observation in arms.items():
            prefix = f"{sid}.{arm}"
            if not isinstance(observation, dict):
                raise ValueError(f"{prefix}: expected an object")
            for key in ("T_base_from_gripper", "T_camera_from_marker_seed"):
                _transform(observation.get(key), f"{prefix}.{key}")
            _array(observation.get("corners_px"), (4, 2), f"{prefix}.corners_px")
            k = _array(observation.get("K"), (3, 3), f"{prefix}.K")
            if (k[0, 0] <= 0 or k[1, 1] <= 0
                    or not np.allclose(k[2], [0, 0, 1], atol=1e-10, rtol=0)
                    or not math.isclose(k[1, 0], 0, abs_tol=1e-10)):
                raise ValueError(f"{prefix}.K: expected pinhole intrinsics with positive focal lengths")
            if "depth_plane" in observation:
                _validate_plane(observation["depth_plane"], f"{prefix}.depth_plane")
            present_arms.add(arm)
    for arm in ARMS:
        if arm in present_arms:
            if arm not in nominal:
                raise ValueError(f"nominal_T_gripper_from_marker.{arm}: required for observed arm")
            _transform(nominal[arm], f"nominal_T_gripper_from_marker.{arm}")


def marker_corners(side):
    """Four canonical ArUco marker-frame corners, in metres."""
    h = side / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])


def _pack(transform):
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
                 transform[:3, 3] / TRANSLATION_SCALE_M]


def _unpack(parameters):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    matrix[:3, 3] = parameters[3:6] * TRANSLATION_SCALE_M
    return matrix


def _observations(manifest, arm):
    return [dict(id=sample["id"], pose_group=sample["pose_group"],
                 F=np.asarray(sample["arms"][arm]["T_base_from_gripper"], float),
                 seed=np.asarray(sample["arms"][arm]["T_camera_from_marker_seed"], float),
                 pixels=np.asarray(sample["arms"][arm]["corners_px"], float),
                 K=np.asarray(sample["arms"][arm]["K"], float),
                 depth_plane=sample["arms"][arm].get("depth_plane"))
            for sample in manifest["samples"] if arm in sample["arms"]]


def _seed(observations, nominal):
    # Estimate X from relative motions so an arbitrary nominal mount cannot
    # pull the optimizer into a poor basin. Only this fit's observations are
    # passed here, including training-only observations in heldout folds.
    relative = []
    for i, left in enumerate(observations):
        for right in observations[i + 1:]:
            a = np.linalg.inv(right["F"]) @ left["F"]
            c = np.linalg.inv(right["seed"]) @ left["seed"]
            a_vector = Rotation.from_matrix(a[:3, :3]).as_rotvec()
            c_vector = Rotation.from_matrix(c[:3, :3]).as_rotvec()
            if np.linalg.norm(a_vector) > 1e-6 and np.linalg.norm(c_vector) > 1e-6:
                relative.append((a, c, a_vector, c_vector))

    x_seed = nominal
    if len(relative) >= 2:
        a_vectors = np.asarray([item[2] for item in relative])
        c_vectors = np.asarray([item[3] for item in relative])
        if (np.linalg.matrix_rank(a_vectors, tol=1e-6) >= 2
                and np.linalg.matrix_rank(c_vectors, tol=1e-6) >= 2):
            rotation, _ = Rotation.align_vectors(a_vectors, c_vectors)
            lhs = np.concatenate([item[0][:3, :3] - np.eye(3) for item in relative])
            rhs = np.concatenate([rotation.apply(item[1][:3, 3]) - item[0][:3, 3]
                                  for item in relative])
            translation, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
            if rank == 3 and np.isfinite(translation).all():
                x_seed = np.eye(4)
                x_seed[:3, :3] = rotation.as_matrix()
                x_seed[:3, 3] = translation

    bases = np.array([obs["seed"] @ np.linalg.inv(x_seed) @ np.linalg.inv(obs["F"])
                      for obs in observations])
    base = np.eye(4)
    base[:3, :3] = Rotation.from_matrix(bases[:, :3, :3]).mean().as_matrix()
    base[:3, 3] = bases[:, :3, 3].mean(axis=0)
    return np.r_[_pack(base), _pack(x_seed)]


def _camera_points(parameters, observations, corners):
    b, x = _unpack(parameters[:6]), _unpack(parameters[6:])
    transforms = b @ np.array([obs["F"] for obs in observations]) @ x
    return corners @ transforms[:, :3, :3].transpose(0, 2, 1) + transforms[:, None, :3, 3]


def _predict(parameters, observations, corners):
    points = _camera_points(parameters, observations, corners)
    homogeneous = points @ np.array([obs["K"] for obs in observations]).transpose(0, 2, 1)
    depth = homogeneous[:, :, 2]
    # Keep optimization finite at the projection singularity. Negative depths
    # remain negative, and the acceptance gate independently rejects them.
    denominator = np.where(np.abs(depth) < 1e-9, np.where(depth < 0, -1e-9, 1e-9), depth)
    return homogeneous[:, :, :2] / denominator[:, :, None], points[:, :, 2]


def _residual(parameters, observations, corners):
    predicted, _ = _predict(parameters, observations, corners)
    pixels = (predicted - np.array([obs["pixels"] for obs in observations])).ravel()
    if not any(obs.get("depth_plane") is not None for obs in observations):
        return pixels
    points = _camera_points(parameters, observations, corners)
    planes = [distance / (obs["depth_plane"]["sigma_m"] * math.sqrt(4))
              for obs, distance in zip(observations, _plane_distances(points, observations))
              if distance is not None]
    return np.r_[pixels, np.concatenate(planes)]


def _plane_distances(points, observations):
    return [None if obs.get("depth_plane") is None else
            p @ np.asarray(obs["depth_plane"]["normal_camera"]) + obs["depth_plane"]["offset_m"]
            for p, obs in zip(points, observations)]


def _modality(observations):
    return "RGB_D" if any(obs.get("depth_plane") is not None for obs in observations) else "RGB"


def _jacobian_report(parameters, observations, corners):
    # Use the joint data Jacobian, not scipy's loss-reweighted Jacobian.
    # RGB rows are raw pixels; optional plane rows use sigma * sqrt(4).
    # Columns already use rotation radians and translation / 0.1 m units.
    columns = []
    for index in range(12):
        delta = np.zeros(12)
        step = 1e-6 * max(1.0, abs(parameters[index]))
        delta[index] = step
        columns.append((_residual(parameters + delta, observations, corners)
                        - _residual(parameters - delta, observations, corners)) / (2 * step))
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if len(singular) == 12 and singular[0] > 0 else 0.0
    return {"singular_values": singular.tolist(), "smallest_to_largest_ratio": ratio,
            "required_ratio_gt": RANK_RATIO_MIN, "full_rank": ratio > RANK_RATIO_MIN,
            "modality": _modality(observations),
            "parameter_units": "rotation radians; translation / 0.1 m"}


def _fit(observations, corners, nominal):
    result = least_squares(_residual, _seed(observations, nominal),
                           args=(observations, corners), loss="soft_l1", f_scale=1.0,
                           jac="3-point", max_nfev=MAX_NFEV,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return result, _jacobian_report(result.x, observations, corners)


def _metrics(parameters, observations, corners):
    predicted, depth = _predict(parameters, observations, corners)
    error = predicted - np.array([obs["pixels"] for obs in observations])
    # Pixel RMS is Euclidean corner error, not RMS over scalar x/y components.
    frame_rms = np.sqrt(np.mean(np.sum(error * error, axis=2), axis=1))
    frames = [{"id": obs["id"], "pose_group": obs["pose_group"],
               "rms_px": float(rms), "minimum_depth_m": float(z.min()),
               "all_positive_depths": bool((z > 0).all())}
              for obs, rms, z in zip(observations, frame_rms, depth)]
    metrics = {"rms_px": float(np.sqrt(np.mean(frame_rms ** 2))),
               "max_frame_rms_px": float(frame_rms.max()),
               "all_positive_depths": bool((depth > 0).all()), "frames": frames}
    if _modality(observations) == "RGB_D":
        distances = _plane_distances(_camera_points(parameters, observations, corners), observations)
        measured = []
        for frame, distance in zip(frames, distances):
            if distance is not None:
                mm = distance * 1000
                measured.append(mm)
                frame["depth_plane"] = {"rms_mm": float(np.sqrt(np.mean(mm ** 2))),
                                        "max_abs_mm": float(np.max(np.abs(mm))),
                                        "corner_distances_mm": mm.tolist()}
        metrics["depth_plane"] = {
            "frame_count": len(measured),
            "rms_mm": float(np.sqrt(np.mean(np.asarray(measured) ** 2))),
            "max_frame_rms_mm": max(f["depth_plane"]["rms_mm"] for f in frames if "depth_plane" in f),
            "max_abs_mm": float(np.max(np.abs(measured)))}
    return metrics


def _excitation(observations):
    # One measured FK per labeled group; then merge duplicate full poses.
    # Connected components make near-duplicate chains conservative and make
    # the distinct-pose count independent of input group ordering.
    representatives = {}
    for obs in observations:
        representatives.setdefault(obs["pose_group"], obs["F"])
    labels = sorted(representatives)
    neighbors = {label: set() for label in labels}
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            a, b = representatives[left], representatives[right]
            translation = np.linalg.norm(a[:3, 3] - b[:3, 3])
            angle = math.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
            if (translation <= POSE_TRANSLATION_TOLERANCE_M
                    and angle <= POSE_ROTATION_TOLERANCE_DEG):
                neighbors[left].add(right)
                neighbors[right].add(left)
    components = []
    unseen = set(labels)
    while unseen:
        pending = [min(unseen)]
        component = set()
        while pending:
            label = pending.pop()
            if label in component:
                continue
            component.add(label)
            pending.extend(neighbors[label] - component)
        unseen -= component
        components.append(sorted(component))
    rotations = [representatives[group[0]][:3, :3] for group in components]
    vectors = [Rotation.from_matrix(rotations[i].T @ rotations[j]).as_rotvec()
               for i in range(len(rotations)) for j in range(i + 1, len(rotations))]
    singular = np.zeros(3)
    if vectors:
        values = np.linalg.svd(np.asarray(vectors), compute_uv=False)
        singular[:len(values)] = values
    max_angle = math.degrees(max((np.linalg.norm(v) for v in vectors), default=0))
    ratio = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
    return {"distinct_pose_groups": len(labels),
            "materially_distinct_pose_count": len(components),
            "duplicate_pose_groups": [group for group in components if len(group) > 1],
            "duplicate_translation_tolerance_m": POSE_TRANSLATION_TOLERANCE_M,
            "duplicate_rotation_tolerance_deg": POSE_ROTATION_TOLERANCE_DEG,
            "duplicate_grouping": "connected components of duplicate representative FK poses",
            "max_relative_rotation_deg": max_angle,
            "relative_rotvec_singular_values": singular.tolist(),
            "second_to_first_ratio": ratio,
            "sufficient": max_angle >= 15 and ratio >= 0.1}


def _validate_groups(observations, corners, nominal):
    groups = sorted({obs["pose_group"] for obs in observations})
    folds = []
    for group in groups:
        training = [obs for obs in observations if obs["pose_group"] != group]
        heldout = [obs for obs in observations if obs["pose_group"] == group]
        fold = {"heldout_pose_group": group, "training_group_count": len(groups) - 1,
                "training_sample_ids": [obs["id"] for obs in training],
                "heldout_sample_ids": [obs["id"] for obs in heldout]}
        if len(groups) - 1 < 3:
            fold.update(passed=False, reason="subset_needs_at_least_3_pose_groups")
        else:
            # Reinitialize using only training observations: no heldout seeds,
            # corners, or full-data fitted parameters enter this fit.
            result, jacobian = _fit(training, corners, nominal)
            metrics = _metrics(result.x, heldout, corners)
            fold.update(optimizer_success=bool(result.success), jacobian=jacobian,
                        prediction=metrics,
                        passed=bool(result.success and jacobian["full_rank"]
                                    and metrics["all_positive_depths"]
                                    and metrics["max_frame_rms_px"] <= 4.0
                                    and metrics.get("depth_plane", {}).get("max_frame_rms_mm", 0) <= 10.0))
        folds.append(fold)
    maximum = max((fold["prediction"]["max_frame_rms_px"] for fold in folds
                   if "prediction" in fold), default=None)
    validation = {"method": "leave_one_pose_group_out", "folds": folds,
                  "max_prediction_frame_rms_px": maximum,
                  "passed": all(fold["passed"] for fold in folds)}
    depth_maxima = [fold["prediction"]["depth_plane"]["max_frame_rms_mm"] for fold in folds
                    if "depth_plane" in fold.get("prediction", {})]
    if depth_maxima:
        validation["max_prediction_depth_plane_frame_rms_mm"] = max(depth_maxima)
    return validation


def solve_manifest(manifest):
    """Return a JSON-serializable gated report without reading or writing files."""
    validate_manifest(manifest)
    corners = marker_corners(manifest["marker_side_m"])
    reports = {}
    transforms = {}
    present_arms = tuple(arm for arm in ARMS
                         if any(arm in sample["arms"] for sample in manifest["samples"]))
    for arm in present_arms:
        observations = _observations(manifest, arm)
        nominal = np.asarray(manifest["nominal_T_gripper_from_marker"][arm], float)
        result, jacobian = _fit(observations, corners, nominal)
        metrics = _metrics(result.x, observations, corners)
        excitation = _excitation(observations)
        validation = _validate_groups(observations, corners, nominal)
        need_more = []
        rejected = []
        if excitation["distinct_pose_groups"] < 4:
            need_more.append("at_least_4_distinct_pose_groups_required")
        if excitation["materially_distinct_pose_count"] < 4:
            need_more.append("at_least_4_materially_distinct_fk_poses_required")
        if not excitation["sufficient"]:
            need_more.append("insufficient_nonparallel_rotation_excitation")
        if not jacobian["full_rank"]:
            need_more.append("scaled_jacobian_not_full_rank")
        if not result.success:
            rejected.append("optimizer_did_not_converge")
        if not metrics["all_positive_depths"]:
            rejected.append("nonpositive_corner_depth")
        if metrics["rms_px"] > 2.0:
            rejected.append("training_rms_exceeds_2px")
        if metrics["max_frame_rms_px"] > 3.0:
            rejected.append("training_frame_rms_exceeds_3px")
        if metrics.get("depth_plane", {}).get("rms_mm", 0) > 5.0:
            rejected.append("training_depth_plane_rms_exceeds_5mm")
        if metrics.get("depth_plane", {}).get("max_frame_rms_mm", 0) > 8.0:
            rejected.append("training_depth_plane_frame_rms_exceeds_8mm")
        if not validation["passed"]:
            if any(fold["training_group_count"] < 3
                   or ("jacobian" in fold and not fold["jacobian"]["full_rank"])
                   for fold in validation["folds"]):
                need_more.append("leave_one_group_out_subset_insufficient_or_rank_deficient")
            else:
                rejected.append("leave_one_group_out_prediction_failed")
            if validation.get("max_prediction_depth_plane_frame_rms_mm", 0) > 10.0:
                rejected.append("leave_one_group_out_depth_plane_prediction_failed")
        assessment = ("NEED_MORE_POSES" if need_more else
                      "REJECTED_FIT" if rejected else "VALIDATED_MULTI_POSE")
        reports[arm] = {"assessment": assessment, "reasons": need_more + rejected,
                        "modality": _modality(observations),
                        "depth_plane_count": sum(obs.get("depth_plane") is not None for obs in observations),
                        "sample_count": len(observations), "excitation": excitation,
                        "jacobian": jacobian, "training": metrics, "validation": validation,
                        "optimizer": {"success": bool(result.success), "message": str(result.message),
                                      "nfev": int(result.nfev), "loss": "soft_l1", "f_scale_px": 1.0}}
        transforms[arm] = {"T_camera_from_base": _unpack(result.x[:6]).tolist(),
                           "T_gripper_from_marker": _unpack(result.x[6:]).tolist()}
    assessments = [report["assessment"] for report in reports.values()]
    status = ("NEED_MORE_POSES" if "NEED_MORE_POSES" in assessments else
              "REJECTED_FIT" if "REJECTED_FIT" in assessments else "VALIDATED_MULTI_POSE")
    accepted = status == "VALIDATED_MULTI_POSE"
    for arm, report in reports.items():
        report["status"] = "VALIDATED_MULTI_POSE" if accepted else "CANDIDATE_ONLY"
        report["accepted" if accepted else "candidate"] = transforms[arm]
    return {"schema_version": 1, "setup_id": manifest["setup_id"],
            "setup_confirmed_unchanged": True, "marker_side_m": manifest["marker_side_m"],
            "status": status, "accepted": accepted,
            "modality": "RGB_D" if any(a["modality"] == "RGB_D" for a in reports.values()) else "RGB",
            "accuracy_scope": "Reprojection and heldout consistency only; not calibrated absolute accuracy.",
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
        print(f"handeye: {exc}", file=sys.stderr)
        return 2
    print(f"{report['status']}: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
