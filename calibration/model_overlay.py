"""Project SO101 link-mesh contours into a camera image.

The OBJ files exported for the robot already contain each visual's scale and
origin in link coordinates.  This module therefore applies only the URDF joint
tree transforms.  Raw feedback rendering maps gripper percent over the
independently supplied physical encoder travel.  Caller-provided M3T link
poses are rendered directly.  Contours are drawn without hidden-surface
removal.
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:  # Support both direct script imports and package imports.
    from .shared_board_web import fk
except ImportError:  # pragma: no cover - used by calibration scripts
    from shared_board_web import fk


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
OMITTED_LINKS = ()
GRIPPER_CLOSED_RAD = 0.0
NEAR_M = 1e-4


def _transform(value, name):
    try:
        matrix = np.asarray(value, dtype=float).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 4x4 rigid transform") from exc
    r = matrix[:3, :3]
    if (not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
            or abs(np.linalg.det(r) - 1) > 1e-6):
        raise ValueError(f"{name} must be a finite rigid transform")
    return matrix


def _mesh(path):
    """Load a triangular OBJ and weld duplicate coordinates once at startup."""
    vertices, faces = [], []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
            elif fields[0] == "f" and len(fields) >= 4:
                ids = []
                for token in fields[1:]:
                    idx = int(token.split("/", 1)[0])
                    ids.append(idx - 1 if idx > 0 else len(vertices) + idx)
                for k in range(1, len(ids) - 1):
                    faces.append((ids[0], ids[k], ids[k + 1]))
    raw = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int32)
    if raw.ndim != 2 or raw.shape[1] != 3 or not len(tri) or not np.isfinite(raw).all():
        raise ValueError(f"invalid or empty OBJ mesh: {path}")
    if np.any(tri < 0) or np.any(tri >= len(raw)):
        raise ValueError(f"OBJ face index outside vertex table: {path}")

    # The exporter writes duplicate vertex records per visual triangle.  Six
    # decimal places are well below the mesh's geometric scale and weld seams.
    _, first, inverse = np.unique(np.round(raw, 6), axis=0,
                                  return_index=True, return_inverse=True)
    vertices = raw[first]
    triangles = inverse[tri].astype(np.int32, copy=False)

    return vertices, triangles


class ModelOverlay:
    """Render SO101 link contours for five arm angles and optional gripper percent.

    Raw-feedback rendering requires physical gripper calibration metadata.
    A missing sixth feedback value is treated as the calibrated closed endpoint.
    """

    omitted_links = OMITTED_LINKS

    def __init__(self, robot_path, gripper_calibration=None):
        self.robot_path = Path(robot_path)
        try:
            self.gripper_calibration = dict(gripper_calibration or {})
        except (TypeError, ValueError) as exc:
            raise ValueError("gripper_calibration must be a mapping") from exc
        self.model = json.loads(self.robot_path.read_text(encoding="utf-8"))
        if self.model.get("units") != "metres_radians" or self.model.get("root_link") != "base_link":
            raise ValueError("expected the official SO101 model in metres_radians")
        links = self.model.get("links")
        joints = self.model.get("joints")
        if not isinstance(links, list) or not isinstance(joints, list):
            raise ValueError("robot model must contain link and joint arrays")
        self._link_names = {link["name"] for link in links}
        self._children = {}
        for joint in joints:
            self._children.setdefault(joint["parent"], []).append(joint)
        self._meshes = {}
        for link in links:
            name, mesh_name = link["name"], link.get("mesh", "")
            if not mesh_name or name in self.omitted_links:
                continue
            mesh_path = Path(mesh_name)
            if not mesh_path.is_absolute():
                mesh_path = self.robot_path.parent / mesh_path
            self._meshes[name] = _mesh(mesh_path)
        if not self._meshes:
            raise ValueError("robot model contains no renderable meshes")
        self._contour_cache = OrderedDict()
        self._contour_cache_capacity = 4

    def _link_transforms(self, degrees, gripper_pose, gripper_angle_rad=GRIPPER_CLOSED_RAD):
        q = np.asarray(degrees, dtype=float)[:5]
        by_name = dict(zip(JOINT_NAMES, q))
        output = {self.model["root_link"]: np.eye(4)}
        pending = [self.model["root_link"]]
        while pending:
            parent = pending.pop()
            for joint in self._children.get(parent, ()):
                child = joint["child"]
                t = _transform(joint["origin"], f"{joint['name']}.origin").copy()
                if joint["type"] == "revolute":
                    axis = np.asarray(joint["axis"], dtype=float)
                    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis) - 1) > 1e-6:
                        raise ValueError(f"invalid axis for {joint['name']}")
                    if joint["name"] == "gripper":
                        angle_rad = float(gripper_angle_rad)
                        if not math.isfinite(angle_rad):
                            raise ValueError("gripper angle must be finite")
                    else:
                        angle_rad = math.radians(by_name[joint["name"]])
                    t[:3, :3] = t[:3, :3] @ Rotation.from_rotvec(axis * angle_rad).as_matrix()
                elif joint["type"] != "fixed":
                    raise ValueError(f"unsupported joint type: {joint['type']}")
                output[child] = output[parent] @ t
                pending.append(child)
        if "gripper_link" not in output or not np.allclose(output["gripper_link"], gripper_pose, atol=1e-8, rtol=0):
            raise ValueError("link transforms do not agree with SO101 FK")
        return output

    @staticmethod
    def _validate_intrinsics(K):
        try:
            k = np.asarray(K, dtype=float).reshape(3, 3)
        except (TypeError, ValueError) as exc:
            raise ValueError("K must be a 3x3 camera matrix") from exc
        if (not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1], atol=1e-10, rtol=0)
                or abs(np.linalg.det(k)) < 1e-12):
            raise ValueError("K contains invalid pinhole intrinsics")
        return k

    @staticmethod
    def _clip_near(polygon):
        """Clip one triangle polygon against z >= NEAR_M before projection."""
        result = []
        previous = polygon[-1]
        previous_inside = previous[2] >= NEAR_M
        for current in polygon:
            current_inside = current[2] >= NEAR_M
            if current_inside != previous_inside:
                t = (NEAR_M - previous[2]) / (current[2] - previous[2])
                result.append(previous + t * (current - previous))
            if current_inside:
                result.append(current)
            previous, previous_inside = current, current_inside
        return np.asarray(result, dtype=float)

    @staticmethod
    def _clip_polygon(polygon, width, height):
        """Clip a projected polygon to the image before integer conversion."""
        result = [np.asarray(p, dtype=float) for p in polygon]
        for axis, bound, keep_greater in ((0, 0.0, True), (0, width - 1.0, False),
                                          (1, 0.0, True), (1, height - 1.0, False)):
            if not result:
                break
            clipped = []
            previous = result[-1]
            previous_inside = previous[axis] >= bound if keep_greater else previous[axis] <= bound
            for current in result:
                current_inside = current[axis] >= bound if keep_greater else current[axis] <= bound
                if current_inside != previous_inside:
                    denom = current[axis] - previous[axis]
                    if denom != 0:
                        t = (bound - previous[axis]) / denom
                        clipped.append(previous + t * (current - previous))
                if current_inside:
                    clipped.append(current)
                previous, previous_inside = current, current_inside
            result = clipped
        return np.asarray(result, dtype=float)

    @staticmethod
    def _project(points, K):
        normalized = points[:, :2] / points[:, 2, None]
        homogeneous = np.column_stack((normalized, np.ones(len(points)))) @ K.T
        return homogeneous[:, :2] / homogeneous[:, 2, None]

    def render(self, image, degrees, B, K):
        """Render raw feedback using five arm angles and an optional gripper percent.

        ``B`` is T_camera_from_base.  The optional sixth value is gripper
        percent in [0, 100], mapped linearly over the calibrated physical
        gripper travel.  Omitting it assumes 0% at the calibrated closed
        endpoint and is identified in the returned metadata.
        """
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or image.dtype != np.uint8 or image.shape[0] < 1 or image.shape[1] < 1):
            raise ValueError("image must be a nonempty uint8 BGR image")
        try:
            q = np.asarray(degrees, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("degrees must contain five finite angles and an optional gripper percent") from exc
        if q.ndim != 1 or q.size not in (5, 6) or not np.isfinite(q[:5]).all():
            raise ValueError("degrees must contain five finite angles and an optional gripper percent")
        assumed_closed = q.size == 5
        gripper_percent = 0.0 if assumed_closed else float(q[5])
        if not math.isfinite(gripper_percent) or not 0.0 <= gripper_percent <= 100.0:
            raise ValueError("gripper percent must be finite and between 0 and 100")
        gripper_closed_rad, gripper_open_rad = self._gripper_endpoints()
        gripper_angle_rad = gripper_closed_rad + (gripper_percent / 100.0) * (
            gripper_open_rad - gripper_closed_rad
        )
        gripper_pose = fk(self.model, q[:5])  # also enforces the model's five joint limits
        t_camera_base = _transform(B, "T_camera_from_base")
        k = self._validate_intrinsics(K)
        transforms = self._link_transforms(q[:5], gripper_pose, gripper_angle_rad)
        camera_poses = {
            name: t_camera_base @ transforms[name]
            for name in self._meshes
        }
        metadata = self._render_camera_poses(image, camera_poses, k)
        metadata["T_base_from_gripper"] = gripper_pose.tolist()
        metadata.update({
            "gripper_percent": gripper_percent,
            "gripper_angle_rad": gripper_angle_rad,
            "gripper_closed_rad": gripper_closed_rad,
            "gripper_open_rad": gripper_open_rad,
            "gripper_mapping": self.gripper_calibration["mapping"],
            "gripper_calibration_sha256": self.gripper_calibration["calibration_sha256"],
            "gripper_scale_rad_per_percent": self.gripper_calibration["scale_rad_per_percent"],
            "gripper_zero_reference": self.gripper_calibration["zero_reference"],
            "gripper_calibration": dict(self.gripper_calibration),
            "gripper_input_assumption": (
                "missing sixth input treated as 0% calibrated physical closed"
                if assumed_closed else None
            ),
        })
        return self._draw_camera_poses(image, camera_poses, k, metadata)

    def _gripper_endpoints(self):
        """Return calibrated physical endpoints without consulting URDF limits."""
        calibration = self.gripper_calibration
        if not calibration:
            raise ValueError("physical gripper calibration metadata is required for rendering")
        try:
            closed = float(calibration["closed_rad"])
            opened = float(calibration["open_rad"])
            scale = float(calibration["scale_rad_per_percent"])
            calibration_sha256 = calibration["calibration_sha256"]
            mapping = calibration["mapping"]
            zero_reference = calibration["zero_reference"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("gripper calibration must contain physical endpoints and audit metadata") from exc
        if (not math.isfinite(closed) or not math.isfinite(opened) or opened <= closed
                or not math.isfinite(scale) or scale <= 0
                or not isinstance(calibration_sha256, str) or not calibration_sha256
                or not isinstance(mapping, str) or not mapping
                or not isinstance(zero_reference, str) or not zero_reference):
            raise ValueError("gripper calibration contains invalid physical endpoints or audit metadata")
        return closed, opened

    def render_poses(self, image, poses_by_link, K, color=(255, 255, 0)):
        """Render supplied M3T ``T_camera_from_link`` poses directly.

        The mapping must contain exactly every rendered rigid-link name,
        including the moving jaw and its soft fin.  Each supplied pose is used
        directly; this method does not reconstruct link poses from joint
        angles or the nominal raw-feedback gripper mapping.
        """
        if not isinstance(poses_by_link, dict):
            raise ValueError("poses_by_link must map rigid link names to camera poses")
        expected = set(self._meshes)
        supplied = set(poses_by_link)
        if supplied != expected:
            missing, extra = sorted(expected - supplied), sorted(supplied - expected)
            raise ValueError(f"poses_by_link must contain exactly rendered links; missing={missing}, extra={extra}")
        camera_poses = {
            name: _transform(poses_by_link[name], f"T_camera_from_{name}")
            for name in self._meshes
        }
        k = self._validate_intrinsics(K)
        metadata = self._render_camera_poses(image, camera_poses, k)
        metadata["pose_source"] = "caller-provided actual M3T per-link transforms"
        return self._draw_camera_poses(image, camera_poses, k, metadata, color)

    def _render_camera_poses(self, image, camera_poses, K):
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or image.dtype != np.uint8 or image.shape[0] < 1 or image.shape[1] < 1):
            raise ValueError("image must be a nonempty uint8 BGR image")
        if set(camera_poses) != set(self._meshes):
            raise ValueError("camera pose map must match the loaded renderable links exactly")
        # Revalidate here so both public entry points share the same checks.
        camera_poses = {name: _transform(camera_poses[name], f"T_camera_from_{name}")
                        for name in self._meshes}
        k = self._validate_intrinsics(K)
        return {
            "link_count": len(self._meshes),
            "rendered_links": list(self._meshes),
            "omitted_links": list(self.omitted_links),
            "contour_count": 0,
        }

    def _draw_camera_poses(self, image, camera_poses, K, metadata, color=(255, 255, 0)):
        height, width = image.shape[:2]
        cache_key = (
            width,
            height,
            np.ascontiguousarray(K, dtype=np.float64).tobytes(),
            tuple((name, np.ascontiguousarray(camera_poses[name], dtype=np.float64).tobytes())
                  for name in self._meshes),
        )
        contours = self._contour_cache.get(cache_key)
        if contours is None:
            contours = self._projected_contours(width, height, camera_poses, K)
            self._contour_cache[cache_key] = contours
            while len(self._contour_cache) > self._contour_cache_capacity:
                self._contour_cache.popitem(last=False)
        else:
            self._contour_cache.move_to_end(cache_key)

        output = image.copy()
        if contours:
            cv2.polylines(output, contours, True, color, 1, lineType=cv2.LINE_AA)
        metadata["contour_count"] = len(contours)
        return output, metadata

    def _projected_contours(self, width, height, camera_poses, K):
        contours = []

        for name, (vertices, triangles) in self._meshes.items():
            t_camera_link = camera_poses[name]
            points = vertices @ t_camera_link[:3, :3].T + t_camera_link[:3, 3]
            tri_points = points[triangles]
            normals = np.cross(tri_points[:, 1] - tri_points[:, 0],
                               tri_points[:, 2] - tri_points[:, 0])
            centers = tri_points.mean(axis=1)
            front_facing = np.einsum("ij,ij->i", normals, -centers) >= 0
            tri_points = tri_points[front_facing]
            mask = np.zeros((height, width), dtype=np.uint8)
            positive = np.all(tri_points[:, :, 2] >= NEAR_M, axis=1)
            if np.any(positive):
                projected = self._project(tri_points[positive].reshape(-1, 3), K).reshape(-1, 3, 2)
                finite = np.isfinite(projected).all(axis=(1, 2))
                projected = projected[finite]
                if len(projected):
                    lo, hi = projected.min(axis=1), projected.max(axis=1)
                    inside = np.all((lo >= [0, 0]) & (hi <= [width - 1, height - 1]), axis=1)
                    for polygon in projected[inside]:
                        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
                    partial = (~inside & (hi[:, 0] >= 0) & (lo[:, 0] <= width - 1)
                               & (hi[:, 1] >= 0) & (lo[:, 1] <= height - 1))
                    for polygon in projected[partial]:
                        clipped = self._clip_polygon(polygon, width, height)
                        if len(clipped) >= 3:
                            cv2.fillConvexPoly(mask, np.rint(clipped).astype(np.int32), 255)
            # Only triangles crossing the near plane need the scalar clipper.
            crossing = (~positive & np.any(tri_points[:, :, 2] >= NEAR_M, axis=1))
            for triangle in tri_points[crossing]:
                clipped_3d = self._clip_near(triangle)
                if len(clipped_3d) < 3:
                    continue
                projected = self._project(clipped_3d, K)
                if not np.isfinite(projected).all():
                    continue
                clipped_2d = self._clip_polygon(projected, width, height)
                if len(clipped_2d) >= 3:
                    cv2.fillConvexPoly(mask, np.rint(clipped_2d).astype(np.int32), 255)
            if not np.any(mask):
                continue
            found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(found[-2])
        return tuple(contours)
