#!/usr/bin/env python3
"""Build the pinned LeRobot SO-101 follower with reviewed soft-fin geometry.

This produces a visual tracking model. It does not establish physical calibration,
soft-fin contact geometry, collision safety, or dynamics parameters.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
import tempfile
import xml.etree.ElementTree as ET


MODEL_DIR = Path(__file__).resolve().parent
URDF_NAME = "so101_lerobot_soft_fin.urdf"
LOCK_NAME = "soft_fin.lock.json"
UPSTREAM_LOCK_NAME = "upstream.lock.json"
OFFICIAL_URDF_NAME = "sources/so101_new_calib.urdf"
LEGACY_URDF_NAME = "sources/soft_fin_reference.urdf"
CUSTOM_ASSETS = (
    "fixed_palm.stl",
    "moving_jaw_body.stl",
    "fixed_fin.stl",
    "moving_fin.stl",
    "bracket_34.stl",
)
REQUIRED_INPUTS = {
    "sources/soft_fin_reference.urdf",
    "sources/prior_assembly.json",
    "sources/prior_assembly_decisions.json",
    "sources/prior_assembly_before_bracket_flip.json",
    "sources/prior_model_provenance.json",
    *(f"assets/soft_fin/{name}" for name in CUSTOM_ASSETS),
}
ANCHOR_TOLERANCE = 1e-6
IDENTITY_TOLERANCE = 1e-6
# User-approved tracking limits covering v2 recorded travel and wrist full turn.
JOINT_LIMIT_OVERRIDES = {
    "shoulder_lift": (-math.radians(110), math.radians(110)),
    "elbow_flex": (-1.69, math.radians(100)),
    "wrist_flex": (-math.radians(110), math.radians(110)),
    "wrist_roll": (-math.pi, math.pi),
}
JOINT_LIMIT_OVERRIDE_REASONS = {
    "shoulder_lift": "User-approved expansion for v2 recorded shoulder/wrist travel",
    "elbow_flex": "User-approved upper-limit expansion for recorded right elbow travel of 97.758 degrees",
    "wrist_flex": "User-approved expansion for v2 recorded shoulder/wrist travel",
    "wrist_roll": (
        "User-requested visual-model expansion for v2 wrist full-turn compatibility: "
        "both arms use raw range [0, 4095], normalized as "
        "(raw - midpoint) * 360 / 4095 to [-180, 180] degrees; "
        "covers the recorded right wrist_roll angle of -168.66 degrees"
    ),
}
WARNING = (
    "Visual tracking model only. Official inertials are preserved but were not "
    "recomputed for the changed geometry; accessory and fin inertials are not "
    "supplied. Do not use this model for dynamics, contact safety, or collision safety."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON input {path}: {exc}") from exc


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe path in {label}: {value!r}")
    return path.as_posix()


def _inside(root: Path, rel: str, label: str) -> Path:
    safe_rel = _relative_path(rel, label)
    path = root / safe_rel
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes model directory in {label}: {rel!r}")
    return path


def _check_hash_record(path: Path, sha: object, size: object | None = None) -> bytes:
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"Invalid SHA-256 pin for {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Missing pinned input {path}: {exc}") from exc
    if sha256_bytes(data) != sha:
        raise ValueError(f"Pinned input checksum mismatch: {path}")
    if size is not None and (not isinstance(size, int) or size != len(data)):
        raise ValueError(f"Pinned input byte count mismatch: {path}")
    return data


def _vector(value: str | None, default: tuple[float, float, float], label: str) -> tuple[float, float, float]:
    if value is None:
        return default
    try:
        result = tuple(float(part) for part in value.split())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if len(result) != 3 or any(not math.isfinite(part) for part in result):
        raise ValueError(f"Invalid {label}: expected three finite values")
    return result  # type: ignore[return-value]


def transform_from_origin(element: ET.Element | None, label: str) -> list[list[float]]:
    xyz = _vector(element.get("xyz") if element is not None else None, (0.0, 0.0, 0.0), f"{label} xyz")
    rpy = _vector(element.get("rpy") if element is not None else None, (0.0, 0.0, 0.0), f"{label} rpy")
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF fixed-axis roll, pitch, yaw: Rz(yaw) * Ry(pitch) * Rx(roll).
    rotation = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], xyz[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], xyz[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _inverse_rigid(matrix: list[list[float]]) -> list[list[float]]:
    rotation = [row[:3] for row in matrix[:3]]
    inverse = [[rotation[j][i] for j in range(3)] for i in range(3)]
    translation = [matrix[i][3] for i in range(3)]
    result = [row + [0.0] for row in inverse]
    for i in range(3):
        result[i][3] = -sum(inverse[i][j] * translation[j] for j in range(3))
    result.append([0.0, 0.0, 0.0, 1.0])
    return result


def _max_difference(a: list[list[float]], b: list[list[float]]) -> float:
    return max(abs(a[i][j] - b[i][j]) for i in range(4) for j in range(4))


def _matrix_to_xyz_rpy(matrix: list[list[float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    cp = math.cos(pitch)
    if abs(cp) > 1e-10:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        # Gimbal lock: choose roll=0 and preserve the combined yaw/roll angle.
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1]) if pitch > 0 else math.atan2(matrix[0][1], matrix[1][1])
    return (matrix[0][3], matrix[1][3], matrix[2][3]), (roll, pitch, yaw)


def _format_vector(values: tuple[float, float, float]) -> str:
    return " ".join("0" if abs(value) < 5e-18 else format(value, ".17g") for value in values)


def _origin_text(matrix: list[list[float]]) -> str:
    xyz, rpy = _matrix_to_xyz_rpy(matrix)
    return f'xyz="{_format_vector(xyz)}" rpy="{_format_vector(rpy)}"'


def xml_signature(element: ET.Element):
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(xml_signature(child) for child in element),
    )


def _source_asset_hashes(prior_provenance: dict) -> dict[str, str]:
    records = prior_provenance.get("original_assets_copied")
    if not isinstance(records, list):
        raise ValueError("Prior provenance lacks copied original-asset hash records")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid copied original-asset record")
        path = _relative_path(record.get("relative_path"), "prior original asset path")
        name = PurePosixPath(path).name
        if name in result:
            raise ValueError(f"Duplicate original-asset provenance entry: {name}")
        result[name] = record.get("sha256")
    return result


def verify_inputs():
    lock_path = MODEL_DIR / LOCK_NAME
    lock = read_json(lock_path)
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported soft_fin.lock.json schema")
    if lock.get("official_lock") != {
        "path": UPSTREAM_LOCK_NAME,
        "sha256": lock.get("official_lock", {}).get("sha256") if isinstance(lock.get("official_lock"), dict) else None,
    }:
        raise ValueError("soft_fin.lock.json must pin the local upstream.lock.json")
    upstream_lock_path = _inside(MODEL_DIR, UPSTREAM_LOCK_NAME, "official lock path")
    upstream_lock_bytes = _check_hash_record(upstream_lock_path, lock["official_lock"].get("sha256"))
    upstream = read_json(upstream_lock_path)
    if upstream.get("schema_version") != 1:
        raise ValueError("Unsupported upstream.lock.json schema")
    sources = upstream.get("sources")
    if not isinstance(sources, dict) or not isinstance(sources.get("urdf"), dict) or not isinstance(sources.get("assets"), list):
        raise ValueError("Malformed official source inventory in upstream.lock.json")

    input_records = lock.get("inputs")
    if not isinstance(input_records, list):
        raise ValueError("soft_fin.lock.json inputs must be a list")
    input_bytes: dict[str, bytes] = {}
    for record in input_records:
        if not isinstance(record, dict):
            raise ValueError("Invalid soft-fin input record")
        rel = _relative_path(record.get("path"), "soft-fin input path")
        if rel in input_bytes:
            raise ValueError(f"Duplicate soft-fin input pin: {rel}")
        input_bytes[rel] = _check_hash_record(_inside(MODEL_DIR, rel, "soft-fin input path"), record.get("sha256"))
    if set(input_bytes) != REQUIRED_INPUTS:
        missing = sorted(REQUIRED_INPUTS - set(input_bytes))
        extra = sorted(set(input_bytes) - REQUIRED_INPUTS)
        raise ValueError(f"soft_fin.lock.json input coverage mismatch; missing={missing}, extra={extra}")

    custom_license = lock.get("custom_geometry_license", {})
    if custom_license.get("status") != "not specified by source project files":
        raise ValueError("Custom geometry license must remain explicitly unspecified")

    official_urdf_record = sources["urdf"]
    official_urdf_rel = _relative_path(official_urdf_record.get("local_path"), "official URDF path")
    if official_urdf_rel != OFFICIAL_URDF_NAME:
        raise ValueError("Official URDF pin must use the vendored source path")
    official_urdf_path = _inside(MODEL_DIR, official_urdf_rel, "official URDF path")
    official_urdf_bytes = _check_hash_record(official_urdf_path, official_urdf_record.get("sha256"), official_urdf_record.get("bytes"))

    assets_by_uri: dict[str, dict] = {}
    official_asset_bytes: dict[str, bytes] = {}
    official_asset_hashes: dict[str, str] = {}
    for record in sources["assets"]:
        if not isinstance(record, dict):
            raise ValueError("Invalid official asset lock record")
        uri = _relative_path(record.get("asset_uri"), "official mesh URI")
        rel = _relative_path(record.get("local_path"), "official mesh asset path")
        source_path = _relative_path(record.get("source_path"), "official source mesh path")
        name = PurePosixPath(source_path).name
        if not uri.startswith("assets/") or uri != f"assets/{name}" or not source_path.startswith("so101/assets/"):
            raise ValueError(f"Official mesh mapping is inconsistent: {uri!r} -> {source_path!r}")
        if rel != f"assets/official/{name}" or uri in assets_by_uri:
            raise ValueError(f"Unexpected or duplicate local official mesh mapping: {uri}")
        asset_path = _inside(MODEL_DIR, rel, "official mesh asset path")
        data = _check_hash_record(asset_path, record.get("sha256"), record.get("bytes"))
        assets_by_uri[uri] = record
        official_asset_bytes[rel] = data
        official_asset_hashes[rel] = record["sha256"]

    # The original mesh frames behind these visual-origin anchors are trusted only
    # when the old model provenance and pinned official lock prove identical bytes.
    prior_provenance = read_json(MODEL_DIR / "sources/prior_model_provenance.json")
    old_asset_hashes = _source_asset_hashes(prior_provenance)
    anchor_mesh_names = (
        "sts3215_03a_v1.stl",
        "wrist_roll_follower_so101_v1.stl",
        "moving_jaw_so101_v1.stl",
    )
    for name in anchor_mesh_names:
        old_hash = old_asset_hashes.get(name)
        official_record = assets_by_uri.get(f"assets/{name}")
        if not isinstance(old_hash, str) or official_record is None or old_hash != official_record.get("sha256"):
            raise ValueError(f"Legacy and official anchor mesh bytes do not match: {name}")

    legacy_generated = prior_provenance.get("generated_files_sha256", {})
    expected_prior_files = {
        "so101_soft_fin_with_handeye.urdf": LEGACY_URDF_NAME,
        **{f"assets/soft_fin/{name}": f"assets/soft_fin/{name}" for name in CUSTOM_ASSETS},
    }
    for provenance_path, input_path in expected_prior_files.items():
        if legacy_generated.get(provenance_path) != sha256_bytes(input_bytes[input_path]):
            raise ValueError(f"Prior build provenance does not match frozen input: {input_path}")

    license_record = sources.get("license")
    if not isinstance(license_record, dict):
        raise ValueError("Official upstream lock lacks its license record")
    license_rel = _relative_path(license_record.get("local_path"), "official license path")
    license_data = _check_hash_record(_inside(MODEL_DIR, license_rel, "official license path"), license_record.get("sha256"), license_record.get("bytes"))

    return {
        "soft_lock": lock,
        "soft_lock_bytes": lock_path.read_bytes(),
        "upstream_lock": upstream,
        "upstream_lock_bytes": upstream_lock_bytes,
        "official_urdf_bytes": official_urdf_bytes,
        "official_urdf_record": official_urdf_record,
        "official_urdf_text": official_urdf_bytes.decode("utf-8"),
        "assets_by_uri": assets_by_uri,
        "official_asset_bytes": official_asset_bytes,
        "official_asset_hashes": official_asset_hashes,
        "license_rel": license_rel,
        "license_bytes": license_data,
        "license_record": license_record,
        "input_bytes": input_bytes,
        "prior_provenance": prior_provenance,
    }


def _visual_origin(root: ET.Element, link_name: str, mesh_basename: str) -> tuple[list[list[float]], ET.Element]:
    link = root.find(f"link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Missing anchor link {link_name}")
    matches = []
    for visual in link.findall("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is not None and PurePosixPath(mesh.get("filename", "")).name == mesh_basename:
            matches.append((visual, mesh))
    if len(matches) != 1:
        raise ValueError(f"Expected one visual anchor for {mesh_basename} on {link_name}, found {len(matches)}")
    visual, mesh = matches[0]
    scale = _vector(mesh.get("scale"), (1.0, 1.0, 1.0), f"{mesh_basename} scale")
    if any(abs(value - 1.0) > 1e-12 for value in scale):
        raise ValueError(f"Scaled visual anchors are unsupported: {link_name}/{mesh_basename}")
    return transform_from_origin(visual.find("origin"), f"{link_name}/{mesh_basename} visual origin"), visual


def _derive_anchor(
    official_root: ET.Element,
    legacy_root: ET.Element,
    *,
    name: str,
    official_link: str,
    legacy_link: str,
    official_mesh: str,
    legacy_mesh: str,
    official_hash: str,
    legacy_source_hash: str,
) -> dict:
    official_origin, _ = _visual_origin(official_root, official_link, official_mesh)
    legacy_origin, _ = _visual_origin(legacy_root, legacy_link, legacy_mesh)
    transform = _matmul(official_origin, _inverse_rigid(legacy_origin))
    return {
        "name": name,
        "official_link": official_link,
        "legacy_link": legacy_link,
        "official_mesh": official_mesh,
        "legacy_mesh": legacy_mesh,
        "official_original_mesh_sha256": official_hash,
        "legacy_original_mesh_sha256": legacy_source_hash,
        "official_visual_origin": official_origin,
        "legacy_visual_origin": legacy_origin,
        "T_official_from_legacy_from_anchor": transform,
    }


def derive_frames(input_state: dict) -> dict:
    official_root = ET.fromstring(input_state["official_urdf_bytes"])
    legacy_bytes = input_state["input_bytes"][LEGACY_URDF_NAME]
    legacy_root = ET.fromstring(legacy_bytes)
    old_hashes = _source_asset_hashes(input_state["prior_provenance"])
    anchors = {
        "fixed": [
            _derive_anchor(
                official_root, legacy_root,
                name="gripper_motor_sts3215",
                official_link="gripper_link", legacy_link="gripper",
                official_mesh="sts3215_03a_v1.stl", legacy_mesh="sts3215_03a_v1.stl",
                official_hash=input_state["assets_by_uri"]["assets/sts3215_03a_v1.stl"]["sha256"],
                legacy_source_hash=old_hashes["sts3215_03a_v1.stl"],
            ),
            _derive_anchor(
                official_root, legacy_root,
                name="fixed_palm_wrist_roll_anchor",
                official_link="gripper_link", legacy_link="gripper",
                official_mesh="wrist_roll_follower_so101_v1.stl", legacy_mesh="fixed_palm.stl",
                official_hash=input_state["assets_by_uri"]["assets/wrist_roll_follower_so101_v1.stl"]["sha256"],
                legacy_source_hash=old_hashes["wrist_roll_follower_so101_v1.stl"],
            ),
        ],
        "moving": [
            _derive_anchor(
                official_root, legacy_root,
                name="moving_jaw_short_jaw_anchor",
                official_link="moving_jaw_so101_v1_link", legacy_link="jaw",
                official_mesh="moving_jaw_so101_v1.stl", legacy_mesh="moving_jaw_body.stl",
                official_hash=input_state["assets_by_uri"]["assets/moving_jaw_so101_v1.stl"]["sha256"],
                legacy_source_hash=old_hashes["moving_jaw_so101_v1.stl"],
            ),
        ],
    }
    transforms = {}
    for group, members in anchors.items():
        candidate = members[0]["T_official_from_legacy_from_anchor"]
        agreement = max(_max_difference(candidate, item["T_official_from_legacy_from_anchor"]) for item in members)
        identity_delta = _max_difference(candidate, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        if agreement > ANCHOR_TOLERANCE:
            raise ValueError(f"{group} visual anchors disagree by {agreement:.9g}; refusing frame transfer")
        if identity_delta > IDENTITY_TOLERANCE:
            raise ValueError(f"{group} frame mapping is not near identity ({identity_delta:.9g})")
        transforms[group] = {
            "T_official_from_legacy": candidate,
            "anchors": members,
            "anchor_agreement_max_abs": agreement,
            "identity_delta_max_abs": identity_delta,
            "anchor_tolerance": ANCHOR_TOLERANCE,
            "identity_tolerance": IDENTITY_TOLERANCE,
        }

    legacy_mount = legacy_root.find("joint[@name='gripper_mount']")
    if legacy_mount is None or legacy_mount.get("type") != "fixed":
        raise ValueError("Legacy reference lacks the fixed gripper_mount accessory transform")
    parent = legacy_mount.find("parent")
    child = legacy_mount.find("child")
    if parent is None or parent.get("link") != "gripper" or child is None or child.get("link") != "gripper_camera_mount":
        raise ValueError("Legacy gripper_mount no longer attaches the bracket under gripper")
    old_gripper_from_mount = transform_from_origin(legacy_mount.find("origin"), "legacy gripper_mount origin")
    old_from_mount = _matmul(transforms["fixed"]["T_official_from_legacy"], old_gripper_from_mount)
    transforms["bracket"] = {
        "T_legacy_gripper_from_legacy_mount": old_gripper_from_mount,
        "T_official_from_legacy_mount": old_from_mount,
        "source_joint": "gripper_mount",
        "parent_link": "gripper_link",
        "child_link": "soft_fin_camera_mount_link",
    }
    return transforms


def _mesh_replacement_text(source_text: str, source_root: ET.Element, input_state: dict) -> str:
    text = source_text
    expected_counts: dict[str, int] = {}
    for mesh in source_root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename not in input_state["assets_by_uri"]:
            raise ValueError(f"Official URDF has an unpinned mesh path: {filename!r}")
        expected_counts[filename] = expected_counts.get(filename, 0) + 1
    for uri, count in expected_counts.items():
        old = f'filename="{uri}"'
        if text.count(old) != count:
            raise ValueError(f"Source URDF text and parsed mesh references disagree for {uri}")
        official_rel = input_state["assets_by_uri"][uri]["local_path"]
        text = text.replace(old, f'filename="{official_rel}"')

    for old_name, new_name in (
        ("wrist_roll_follower_so101_v1.stl", "assets/soft_fin/fixed_palm.stl"),
        ("moving_jaw_so101_v1.stl", "assets/soft_fin/moving_jaw_body.stl"),
    ):
        official_ref = f'filename="assets/official/{old_name}"'
        if text.count(official_ref) != 2:
            raise ValueError(f"Expected exactly visual and collision references for {old_name}")
        text = text.replace(official_ref, f'filename="{new_name}"')
    return text


def _addition_xml(transforms: dict) -> str:
    fixed_origin = _origin_text(transforms["fixed"]["T_official_from_legacy"])
    moving_origin = _origin_text(transforms["moving"]["T_official_from_legacy"])
    bracket_origin = _origin_text(transforms["bracket"]["T_official_from_legacy_mount"])
    return f'''  <!-- Added reviewed visual geometry; link inertials are not recomputed. -->
  <link name="fixed_soft_fin">
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="assets/soft_fin/fixed_fin.stl"/></geometry><material name="3d_printed"/></visual>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="assets/soft_fin/fixed_fin.stl"/></geometry></collision>
  </link>
  <joint name="fixed_soft_fin_joint" type="fixed">
    <origin {fixed_origin}/><parent link="gripper_link"/><child link="fixed_soft_fin"/>
  </joint>
  <link name="moving_soft_fin">
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="assets/soft_fin/moving_fin.stl"/></geometry><material name="3d_printed"/></visual>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="assets/soft_fin/moving_fin.stl"/></geometry></collision>
  </link>
  <joint name="moving_soft_fin_joint" type="fixed">
    <origin {moving_origin}/><parent link="moving_jaw_so101_v1_link"/><child link="moving_soft_fin"/>
  </joint>
  <link name="soft_fin_camera_mount_link">
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="assets/soft_fin/bracket_34.stl"/></geometry><material name="3d_printed"/></visual>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="assets/soft_fin/bracket_34.stl"/></geometry></collision>
  </link>
  <joint name="soft_fin_camera_mount_joint" type="fixed">
    <origin {bracket_origin}/><parent link="gripper_link"/><child link="soft_fin_camera_mount_link"/>
  </joint>'''


def graph_check(root: ET.Element) -> dict:
    links = [link.get("name") for link in root.findall("link")]
    joints = root.findall("joint")
    joint_names = [joint.get("name") for joint in joints]
    if None in links or None in joint_names or len(set(links)) != len(links) or len(set(joint_names)) != len(joint_names):
        raise ValueError("URDF contains missing or duplicate link/joint names")
    parent_of = {}
    for joint in joints:
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint lacks parent/child: {joint.get('name')}")
        p, c = parent.get("link"), child.get("link")
        if p not in links or c not in links or c in parent_of:
            raise ValueError(f"Invalid or multiply parented joint: {joint.get('name')}")
        parent_of[c] = p
    roots = set(links) - set(parent_of)
    if len(roots) != 1 or len(joints) != len(links) - 1:
        raise ValueError("URDF must be one connected rooted tree")
    for start in links:
        seen = set()
        node = start
        while node in parent_of:
            if node in seen:
                raise ValueError("URDF graph contains a cycle")
            seen.add(node)
            node = parent_of[node]
        if node not in roots:
            raise ValueError("URDF graph contains a disconnected component")
    return {"root": next(iter(roots)), "links": len(links), "joints": len(joints)}


def apply_joint_limit_overrides(text: str) -> str:
    for name, (lower, upper) in JOINT_LIMIT_OVERRIDES.items():
        pattern = rf'(<joint\b(?=[^>]*\btype=)[^>]*\bname="{re.escape(name)}"[^>]*>)(.*?)(</joint>)'
        def replace_joint(match):
            def replace_limit(limit_match):
                tag = limit_match.group(0)
                for key, value in (("lower", lower), ("upper", upper)):
                    tag, count = re.subn(rf'\b{key}="[^"]*"', f'{key}="{value:.16g}"', tag)
                    if count != 1:
                        raise ValueError(f"Missing or duplicate {name} {key} limit")
                return tag
            body, count = re.subn(r'<limit\b[^>]*/>', replace_limit, match.group(2))
            if count != 1:
                raise ValueError(f"Expected one limit for {name}")
            return match.group(1) + body + match.group(3)
        text, count = re.subn(pattern, replace_joint, text, flags=re.S)
        if count != 1:
            raise ValueError(f"Expected one joint named {name}")
    return text


def verify_preserved(baseline: ET.Element, output: ET.Element) -> dict:
    baseline_links = {node.get("name"): node for node in baseline.findall("link")}
    output_links = {node.get("name"): node for node in output.findall("link")}
    replacements = {
        "gripper_link": {
            "assets/official/wrist_roll_follower_so101_v1.stl": "assets/soft_fin/fixed_palm.stl",
        },
        "moving_jaw_so101_v1_link": {
            "assets/official/moving_jaw_so101_v1.stl": "assets/soft_fin/moving_jaw_body.stl",
        },
    }
    if not set(baseline_links).issubset(output_links):
        raise ValueError("An official link was removed")
    for name, original in baseline_links.items():
        expected = copy.deepcopy(original)
        for mesh in expected.findall(".//mesh"):
            filename = mesh.get("filename", "")
            if filename.startswith("assets/"):
                filename = f"assets/official/{PurePosixPath(filename).name}"
            mesh.set("filename", replacements.get(name, {}).get(filename, filename))
        if xml_signature(expected) != xml_signature(output_links[name]):
            raise ValueError(f"Official link semantics changed outside authorized mesh replacement: {name}")

    baseline_joints = {node.get("name"): node for node in baseline.findall("joint")}
    output_joints = {node.get("name"): node for node in output.findall("joint")}
    added_joints = {"fixed_soft_fin_joint", "moving_soft_fin_joint", "soft_fin_camera_mount_joint"}
    if set(output_joints) != set(baseline_joints) | added_joints:
        raise ValueError("Output changed the official joint inventory or added unexpected joints")
    limit_changes = {}
    for name, node in baseline_joints.items():
        expected = copy.deepcopy(node)
        if name in JOINT_LIMIT_OVERRIDES:
            limit = expected.find("limit")
            if limit is None:
                raise ValueError(f"Missing official limit: {name}")
            lower, upper = JOINT_LIMIT_OVERRIDES[name]
            limit_changes[name] = {"old_lower": float(limit.get("lower")),
                                   "old_upper": float(limit.get("upper")),
                                   "new_lower": lower, "new_upper": upper,
                                   "reason": JOINT_LIMIT_OVERRIDE_REASONS[name]}
            limit.set("lower", f"{lower:.16g}")
            limit.set("upper", f"{upper:.16g}")
        if xml_signature(expected) != xml_signature(output_joints[name]):
            raise ValueError(f"Official joint XML changed outside authorized limits: {name}")
    for tag in ("transmission",):
        old = {node.get("name"): node for node in baseline.findall(tag)}
        new = {node.get("name"): node for node in output.findall(tag)}
        if set(old) != set(new) or any(xml_signature(node) != xml_signature(new[name]) for name, node in old.items()):
            raise ValueError(f"Official {tag} semantics changed")
    frame_link = baseline.find("link[@name='gripper_frame_link']")
    frame_joint = baseline.find("joint[@name='gripper_frame_joint']")
    if frame_link is None or frame_joint is None:
        raise ValueError("Official gripper frame link/joint is missing")
    if xml_signature(frame_link) != xml_signature(output_links["gripper_frame_link"]):
        raise ValueError("Official gripper_frame_link changed")
    if xml_signature(frame_joint) != xml_signature(output_joints["gripper_frame_joint"]):
        raise ValueError("Official gripper_frame_joint changed")
    return {
        "official_joint_names": list(baseline_joints),
        "official_joints_semantically_unchanged": False,
        "official_joint_kinematics_unchanged": True,
        "authorized_joint_limit_overrides": limit_changes,
        "official_links_semantically_unchanged_except_two_mesh_filenames": True,
        "official_inertials_and_visual_origins_unchanged": True,
        "gripper_frame_link_and_joint_semantically_unchanged": True,
    }


def _validate_runtime_meshes(root: ET.Element, available: set[str]) -> dict:
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        rel = _relative_path(filename, "runtime mesh URI")
        if rel not in available:
            raise ValueError(f"Unresolved or unpinned runtime mesh reference: {filename}")
    expected = {
        ("gripper_link", "visual", "assets/soft_fin/fixed_palm.stl"),
        ("gripper_link", "collision", "assets/soft_fin/fixed_palm.stl"),
        ("moving_jaw_so101_v1_link", "visual", "assets/soft_fin/moving_jaw_body.stl"),
        ("moving_jaw_so101_v1_link", "collision", "assets/soft_fin/moving_jaw_body.stl"),
        ("fixed_soft_fin", "visual", "assets/soft_fin/fixed_fin.stl"),
        ("fixed_soft_fin", "collision", "assets/soft_fin/fixed_fin.stl"),
        ("moving_soft_fin", "visual", "assets/soft_fin/moving_fin.stl"),
        ("moving_soft_fin", "collision", "assets/soft_fin/moving_fin.stl"),
        ("soft_fin_camera_mount_link", "visual", "assets/soft_fin/bracket_34.stl"),
        ("soft_fin_camera_mount_link", "collision", "assets/soft_fin/bracket_34.stl"),
    }
    found = set()
    for link in root.findall("link"):
        for role in ("visual", "collision"):
            for mesh in link.findall(f"{role}/geometry/mesh"):
                found.add((link.get("name"), role, mesh.get("filename")))
    if not expected.issubset(found):
        raise ValueError(f"Required replacement/addition visual or collision meshes are missing: {expected - found}")
    if any(link.get("name") in {"camera_module", "handeye_cam"} for link in root.findall("link")):
        raise ValueError("Legacy camera geometry must not be added to the official model")
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    expected_parents = {
        "fixed_soft_fin_joint": ("gripper_link", "fixed_soft_fin"),
        "moving_soft_fin_joint": ("moving_jaw_so101_v1_link", "moving_soft_fin"),
        "soft_fin_camera_mount_joint": ("gripper_link", "soft_fin_camera_mount_link"),
    }
    for name, (parent_name, child_name) in expected_parents.items():
        joint = joints[name]
        parent, child = joint.find("parent"), joint.find("child")
        if joint.get("type") != "fixed" or parent is None or parent.get("link") != parent_name or child is None or child.get("link") != child_name:
            raise ValueError(f"Invalid fixed accessory/fin parenting: {name}")
    return {"all_runtime_mesh_references_relative_and_pinned": True, "camera_links_added": False}


def _copy_inputs(input_state: dict) -> dict[str, bytes]:
    copies = {
        LOCK_NAME: input_state["soft_lock_bytes"],
        UPSTREAM_LOCK_NAME: input_state["upstream_lock_bytes"],
        OFFICIAL_URDF_NAME: input_state["official_urdf_bytes"],
        input_state["license_rel"]: input_state["license_bytes"],
    }
    copies.update(input_state["input_bytes"])
    copies.update(input_state["official_asset_bytes"])
    return copies


def _build_payloads(input_state: dict) -> tuple[dict[str, bytes], dict]:
    source_text = input_state["official_urdf_text"]
    baseline = ET.fromstring(source_text)
    transforms = derive_frames(input_state)
    text = _mesh_replacement_text(source_text, baseline, input_state)
    newline = "\r\n" if "\r\n" in source_text else "\n"
    closing = "</robot>"
    if text.count(closing) != 1:
        raise ValueError("Official URDF must have one robot closing tag")
    additions = _addition_xml(transforms)
    text = text.replace(closing, newline + additions + newline + closing, 1)
    text = apply_joint_limit_overrides(text)
    output_root = ET.fromstring(text)

    preservation = verify_preserved(baseline, output_root)
    structure = graph_check(output_root)
    copies = _copy_inputs(input_state)
    mesh_check = _validate_runtime_meshes(output_root, set(copies) | {URDF_NAME, "model_provenance.json"})

    generated_hashes = {
        URDF_NAME: sha256_bytes(text.encode("utf-8")),
        **{f"assets/soft_fin/{name}": sha256_bytes(copies[f"assets/soft_fin/{name}"]) for name in CUSTOM_ASSETS},
    }
    prior = input_state["prior_provenance"]
    anchor_meshes = {
        "gripper_motor_sts3215": "assets/sts3215_03a_v1.stl",
        "fixed_palm_wrist_roll_anchor": "assets/wrist_roll_follower_so101_v1.stl",
        "moving_jaw_short_jaw_anchor": "assets/moving_jaw_so101_v1.stl",
    }
    official_anchor_hashes = {
        label: input_state["assets_by_uri"][uri]["sha256"] for label, uri in anchor_meshes.items()
    }
    provenance = {
        "schema_version": 1,
        "status": "VISUAL_TRACKING_GEOMETRY_ONLY_NOT_DYNAMICS_NOT_DEPLOYED",
        "model": "LeRobot SO101 follower with soft fins and fixed #34 accessory bracket",
        "official_source": {
            "repository": input_state["upstream_lock"]["sources"]["urdf"]["repository"],
            "revision": input_state["upstream_lock"]["sources"]["urdf"]["revision"],
            "urdf_path": OFFICIAL_URDF_NAME,
            "urdf_sha256": sha256_bytes(input_state["official_urdf_bytes"]),
            "upstream_lock_sha256": sha256_bytes(input_state["upstream_lock_bytes"]),
            "license": input_state["license_record"].get("spdx"),
            "assets": input_state["official_asset_hashes"],
        },
        "custom_geometry_source": {
            "project": input_state["soft_lock"].get("source_project"),
            "prior_reference_urdf_sha256": sha256_bytes(input_state["input_bytes"][LEGACY_URDF_NAME]),
            "prior_assembly_sha256": sha256_bytes(input_state["input_bytes"]["sources/prior_assembly.json"]),
            "prior_assembly_decisions_sha256": sha256_bytes(input_state["input_bytes"]["sources/prior_assembly_decisions.json"]),
            "prior_model_provenance_sha256": sha256_bytes(input_state["input_bytes"]["sources/prior_model_provenance.json"]),
            "assets_sha256": {f"assets/soft_fin/{name}": sha256_bytes(copies[f"assets/soft_fin/{name}"]) for name in CUSTOM_ASSETS},
            "license": input_state["soft_lock"]["custom_geometry_license"],
            "robot_commands_issued": False,
        },
        "mesh_frame_derivation": {
            "method": "Match each pinned common-asset visual origin as T_official_from_mesh * inverse(T_legacy_from_mesh); no ICP, registration, or geometry transform was applied to mesh bytes.",
            "fixed_soft_fin": {
                "parent_link": "gripper_link",
                "mesh_frame_source": "legacy gripper link frame",
                "joint_origin_used": transforms["fixed"]["T_official_from_legacy"],
                **{key: value for key, value in transforms["fixed"].items() if key != "T_official_from_legacy"},
            },
            "moving_soft_fin": {
                "parent_link": "moving_jaw_so101_v1_link",
                "mesh_frame_source": "legacy jaw link frame",
                "joint_origin_used": transforms["moving"]["T_official_from_legacy"],
                **{key: value for key, value in transforms["moving"].items() if key != "T_official_from_legacy"},
            },
        },
        "bracket_accessory": {
            **transforms["bracket"],
            "mesh": "assets/soft_fin/bracket_34.stl",
            "mesh_bytes_preserved": True,
            "camera_module_or_handeye_pose_added": False,
        },
        "preservation_checks": {
            **preservation,
            "structure": structure,
            "runtime_meshes": mesh_check,
            "official_anchor_mesh_bytes_equal_to_legacy_originals": official_anchor_hashes,
        },
        "tracking_frame_note": "The official gripper_frame_link remains unchanged and is the official nominal tool point; it is not a calibrated soft-fin contact tip.",
        "dynamics_caveat": WARNING,
        "generated_files_sha256": generated_hashes,
        "pinned_inputs_sha256": {
            UPSTREAM_LOCK_NAME: sha256_bytes(input_state["upstream_lock_bytes"]),
            **{path: sha256_bytes(data) for path, data in sorted(input_state["input_bytes"].items())},
            **input_state["official_asset_hashes"],
        },
    }
    payloads = dict(copies)
    payloads[URDF_NAME] = text.encode("utf-8")
    payloads["model_provenance.json"] = (json.dumps(provenance, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return payloads, provenance


def _safe_output(output: Path, payloads: dict[str, bytes]) -> Path:
    output = Path(output).expanduser().resolve()
    canonical = MODEL_DIR.resolve()
    if output != canonical and (output.is_relative_to(canonical) or canonical.is_relative_to(output)):
        raise ValueError("Alternate output must not overlap the pinned model input directory")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {output}")
    if output != canonical and output.exists():
        expected = set(payloads)
        expected_dirs = {
            parent.as_posix()
            for rel in expected
            for parent in PurePosixPath(rel).parents
            if parent.as_posix() != "."
        }
        found = {p.relative_to(output).as_posix() for p in output.rglob("*")}
        extras = sorted(found - expected - expected_dirs)
        if extras:
            raise ValueError(f"Refusing to write into a nonempty unrelated output directory; unexpected files: {extras[:8]}")
        for rel, data in payloads.items():
            path = output / rel
            if path.exists() and (not path.is_file() or path.read_bytes() != data):
                raise ValueError(f"Refusing to overwrite a different existing output file: {path}")
    return output


def _write_payloads(output: Path, payloads: dict[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for rel, data in payloads.items():
        dst = _inside(output, rel, "output path")
        if dst.exists() and dst.is_file() and dst.read_bytes() == data:
            continue
        if dst.exists() and not dst.is_file():
            raise ValueError(f"Output target is not a regular file: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{dst.name}.", dir=dst.parent, delete=False) as tmp:
                tmp.write(data)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
            os.replace(tmp_name, dst)
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)


def build(output: Path = MODEL_DIR) -> Path:
    """Build and return the generated URDF path.

    Pinned input failures and unsafe output conflicts raise ``ValueError``.
    """
    input_state = verify_inputs()
    payloads, _ = _build_payloads(input_state)
    output = _safe_output(Path(output), payloads)
    _write_payloads(output, payloads)
    # Validate that the published tree still resolves every runtime mesh.
    root = ET.parse(output / URDF_NAME).getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        path = _inside(output, filename, "runtime mesh path")
        if not path.is_file():
            raise ValueError(f"Unresolved runtime mesh after build: {filename}")
    return output / URDF_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL_DIR, help="Output model directory; defaults to this pinned model")
    args = parser.parse_args()
    path = build(args.output)
    print(path)


if __name__ == "__main__":
    main()
