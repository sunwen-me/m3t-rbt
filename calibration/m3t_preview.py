"""Validate exact-image M3T preview packets from managed and independent trackers."""
from __future__ import annotations
import json
import copy
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
import time
import numpy as np
from shared_board_web import NAMES, encoded, exclusive, fk, sha, transform


JOINT_OUTPUTS = (*NAMES, 'gripper')
GRIPPER_MAPPING_ASSUMPTION = (
    'v2 encoder endpoint span mapped linearly to the nominal model zero reference; '
    'endpoint alignment is not independently measured'
)


MOUNT_REFINEMENT_DESCRIPTION = 'CAD 底座软约束 · 编码器关节固定'
ENCODER_REFERENCE_MOUNT_DESCRIPTION = 'CAD 底座软约束 · 五关节 + 夹爪实时软先验'
END_EFFECTOR_PRIORITY_DESCRIPTION = '夹爪刚性结构优先'
END_EFFECTOR_VISUAL_LINK_WEIGHTS = {'gripper_link': 5.0, 'moving_jaw_so101_v1_link': 5.0}
TCP_MAX_MATRIX_ABS_DIFFERENCE = 2e-4
PREVIEW_MAX_AGE_NS = 2_000_000_000


def mount_refinement_config(translation_std_m, rotation_std_deg):
    if (type(translation_std_m) not in (int, float)
            or type(rotation_std_deg) not in (int, float)
            or not math.isfinite(translation_std_m) or not math.isfinite(rotation_std_deg)
            or not .0001 <= translation_std_m <= .2
            or not .0001 <= math.radians(rotation_std_deg) <= .785398):
        raise ValueError('M3T mount sigma 必须有限，平移在 [0.0001, 0.2] m、旋转在 [0.0001, 0.785398] rad 内')
    return {'enabled': True, 'translation_standard_deviation_m': translation_std_m,
            'rotation_standard_deviation_rad': math.radians(rotation_std_deg)}


def mount_refinement_diagnostics(estimate, nominal_bases, root_link="base_link", *,
                                 joints_locked_to_feedback=True):
    """Validate mount poses and joint mode; None centers belong to an external producer."""
    error = 'M3T CAD 底座软约束诊断缺失或无效，请检查 native binary'
    report = estimate.get('mount_refinement')
    if (not isinstance(report, dict) or report.get('enabled') is not True
            or report.get('joints_locked_to_feedback') is not joints_locked_to_feedback
            or estimate.get('joints_locked_to_feedback', joints_locked_to_feedback) is not joints_locked_to_feedback
            or estimate.get('encoder_reference_links', False) is not (not joints_locked_to_feedback)
            or not isinstance(report.get('arms'), list)
            or len(report['arms']) != len(nominal_bases)):
        raise ValueError(error)
    link_arms = {link['arm'] for link in estimate['links']}
    if link_arms != set(nominal_bases) or set(estimate['joint_angles_rad']) != set(nominal_bases):
        raise ValueError(error)
    seen = set()
    for entry in report['arms']:
        if not isinstance(entry, dict):
            raise ValueError(error)
        arm = entry.get('arm')
        if not isinstance(arm, str) or arm not in nominal_bases or arm in seen:
            raise ValueError(error)
        seen.add(arm)
        matrices = []
        for key in ('nominal_base_in_camera', 'refined_base_in_camera'):
            pose = entry.get(key)
            if (not isinstance(pose, list) or len(pose) != 16
                    or any(type(value) not in (int, float) for value in pose)):
                raise ValueError(error)
            matrices.append(transform(pose))
        nominal, refined = matrices
        bases = [link['pose'] for link in estimate['links']
                 if link['arm'] == arm and link['link'] == root_link]
        if (len(bases) != 1
                or (nominal_bases[arm] is not None
                    and not np.allclose(nominal, nominal_bases[arm], rtol=0, atol=1e-6))
                or not np.allclose(refined, transform(bases[0]), rtol=0, atol=1e-6)):
            raise ValueError(error)
        translation = float(np.linalg.norm(refined[:3, 3] - nominal[:3, 3]))
        relative_rotation = refined[:3, :3] @ nominal[:3, :3].T
        # atan2 stays stable near zero with native float matrices.
        sine = .5 * float(np.linalg.norm([
            relative_rotation[2, 1] - relative_rotation[1, 2],
            relative_rotation[0, 2] - relative_rotation[2, 0],
            relative_rotation[1, 0] - relative_rotation[0, 1]]))
        cosine = float(np.clip((np.trace(relative_rotation) - 1.) / 2., -1., 1.))
        rotation = math.atan2(sine, cosine)
        for key, expected, tolerance in (('translation_correction_m', translation, 1e-4),
                                         ('rotation_correction_rad', rotation, 1e-4)):
            value = entry.get(key)
            if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or not math.isclose(value, expected, rel_tol=0, abs_tol=tolerance)):
                raise ValueError(error)
    for prior in estimate['joint_priors']['priors']:
        angle = prior.get('angle_rad')
        if type(angle) not in (int, float) or not math.isfinite(angle):
            raise ValueError('M3T 实时编码器先验角度无效')
        if (joints_locked_to_feedback
                and abs(estimate['joint_angles_rad'][prior['arm']][prior['joint']] - angle) > 1e-4):
            raise ValueError('M3T 编码器关节固定校验失败：输出角度与 applied prior 不一致')


def gripper_calibration_values(calibration):
    """Validate the physical encoder calibration supplied by the preview renderer."""
    if not isinstance(calibration, dict):
        raise ValueError('缺少 v2 夹爪编码器标定信息')
    try:
        closed = float(calibration['closed_rad'])
        opened = float(calibration['open_rad'])
        scale = float(calibration['scale_rad_per_percent'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('v2 夹爪标定端点或比例无效') from exc
    calibration_sha = calibration.get('calibration_sha256')
    zero_reference = calibration.get('zero_reference')
    expected_scale = (opened - closed) / 100.0
    if (not all(math.isfinite(value) for value in (closed, opened, scale))
            or not math.isclose(closed, 0.0, rel_tol=0.0, abs_tol=1e-12)
            or opened <= closed or scale <= 0.0
            or not math.isclose(scale, expected_scale, rel_tol=1e-12, abs_tol=1e-12)
            or not isinstance(calibration_sha, str) or len(calibration_sha) != 64
            or any(char not in '0123456789abcdefABCDEF' for char in calibration_sha)
            or not isinstance(zero_reference, str) or not zero_reference.strip()):
        raise ValueError('v2 夹爪标定未提供有效的物理端点和零位假设')
    return closed, opened, scale


def gripper_angle_from_percent(percent, calibration):
    closed, _, scale = gripper_calibration_values(calibration)
    try:
        value = float(percent)
    except (TypeError, ValueError) as exc:
        raise ValueError('夹爪反馈百分比无效') from exc
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError('夹爪反馈百分比必须在 0 到 100 之间')
    return closed + value * scale


def prior_mappings(arm, calibration):
    """Build the five degree mappings and the calibrated gripper percent mapping."""
    closed, _, gripper_scale = gripper_calibration_values(calibration)
    if arm not in ('so101_left', 'so101_right'):
        raise ValueError('M3T 先验机械臂身份无效')
    source_arm = arm.removeprefix('so101_')
    mappings = [
        {'arm': arm, 'joint': name, 'source_arm': source_arm, 'source_joint': name,
         'scale_rad_per_unit': math.pi / 180.0, 'offset_rad': 0.0,
         'standard_deviation_rad': 0.1}
        for name in NAMES
    ]
    mappings.append({'arm': arm, 'joint': 'gripper', 'source_arm': source_arm,
        'source_joint': 'gripper', 'scale_rad_per_unit': gripper_scale,
        'offset_rad': closed,
        'standard_deviation_rad': 0.01})
    return mappings


def averaged_feedback(before, after):
    """Validate and average stable five-joint plus gripper feedback samples."""
    values = []
    for reading in (before, after):
        try:
            sample = np.asarray(reading['degrees_and_gripper_percent'], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('关节反馈必须包含五轴角度和夹爪百分比') from exc
        if sample.shape != (6,) or not np.isfinite(sample).all():
            raise ValueError('关节反馈必须是六个有限数值')
        if not 0.0 <= sample[5] <= 100.0:
            raise ValueError('夹爪反馈百分比必须在 0 到 100 之间')
        values.append(sample)
    first, second = values
    if np.max(np.abs(first[:5] - second[:5])) > 0.3:
        raise ValueError('请先停稳机械臂，再启动 M3T')
    if abs(float(first[5] - second[5])) > 0.5:
        raise ValueError('夹爪正在开合；请停稳后启动 M3T')
    return (first + second) / 2.0


def runtime_robot_model(source_path, calibration):
    """Clone a robot model for this run, widening only gripper limits and resolving meshes."""
    closed, opened, _ = gripper_calibration_values(calibration)
    source = Path(source_path).resolve()
    try:
        original_bytes = source.read_bytes()
        model = json.loads(original_bytes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError('无法读取 M3T 机器人模型以创建本次运行副本') from exc
    if not isinstance(model, dict) or not isinstance(model.get('links'), list) or not isinstance(model.get('joints'), list):
        raise ValueError('M3T 机器人模型结构无效')
    source_urdf = model.get('source_urdf')
    if not isinstance(source_urdf, str) or not Path(source_urdf).is_absolute():
        raise ValueError('M3T 模型 source_urdf 必须保持为绝对路径')
    matches = [joint for joint in model['joints'] if joint.get('name') == 'gripper']
    if len(matches) != 1 or matches[0].get('type') != 'revolute':
        raise ValueError('M3T 模型必须包含一个 revolute gripper 关节')
    joint = matches[0]
    try:
        original_lower = float(joint['lower'])
        original_upper = float(joint['upper'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('M3T gripper 关节限位无效') from exc
    if (not math.isfinite(original_lower) or not math.isfinite(original_upper)
            or original_lower > original_upper):
        raise ValueError('M3T gripper 关节限位无效')
    new_lower = min(original_lower, closed)
    new_upper = max(original_upper, opened)
    runtime = copy.deepcopy(model)
    runtime_joint = next(item for item in runtime['joints'] if item.get('name') == 'gripper')
    runtime_joint['lower'] = new_lower
    runtime_joint['upper'] = new_upper
    for link in runtime['links']:
        mesh = link.get('mesh')
        if mesh:
            mesh_path = Path(mesh)
            if not mesh_path.is_absolute():
                mesh_path = source.parent / mesh_path
            link['mesh'] = str(mesh_path.resolve())
    audit = {
        'schema_version': 1,
        'purpose': 'per-run M3T gripper endpoint limit accommodation',
        'source_robot_json': str(source),
        'source_robot_json_sha256': sha(source),
        'source_urdf': source_urdf,
        'gripper_joint': 'gripper',
        'original_limits_rad': {'lower': original_lower, 'upper': original_upper},
        'runtime_limits_rad': {'lower': new_lower, 'upper': new_upper},
        'gripper_calibration': copy.deepcopy(calibration),
        'mapping_assumption': GRIPPER_MAPPING_ASSUMPTION,
        'canonical_model_modified': False,
    }
    return runtime, audit


def fresh_capture(stamp, now_ns):
    return type(stamp) is int and 0 <= now_ns - stamp <= PREVIEW_MAX_AGE_NS


def parse_packet(payload, now_ns, config, arm, required_links):
    line,sep,raw=payload.partition(b'\n')
    if not sep or len(line)>262144: raise ValueError('M3T 预览包头无效')
    header=json.loads(line)
    if (not isinstance(header, dict) or header.get('schema')!='rbt-m3t-preview-v1'
            or header.get('pixel_format')!='BGR8'):
        raise ValueError('M3T 预览格式不支持')
    w,h=header.get('width'),header.get('height')
    if [w,h]!=config['image_size'] or type(w) is not int or type(h) is not int or len(raw)!=w*h*3:
        raise ValueError('M3T 图像与主相机尺寸不一致')
    estimate=header.get('estimate')
    if (not isinstance(estimate, dict) or estimate.get('schema_version')!=1
            or estimate.get('event')!='estimate'):
        raise ValueError('不是 M3T 跟踪结果')
    if estimate.get('modalities')!=['region','depth'] or estimate.get('frame_id')!='orbbec335/color_optical':
        raise ValueError('M3T 相机或模态不匹配')
    if estimate.get('capture_clock_kind') not in ('HostMonotonic','host_monotonic'):
        raise ValueError('M3T 采集时钟不支持')
    source=estimate.get('capture_clock_source')
    if not isinstance(source,str) or not source or len(source)>256:
        raise ValueError('M3T 时钟身份无效')
    try: source_bytes=bytes.fromhex(source)
    except ValueError as exc: raise ValueError('M3T 时钟身份无效') from exc
    if not any(source_bytes): raise ValueError('M3T 时钟身份无效')
    if type(estimate.get('frame_sequence')) is not int or estimate['frame_sequence']<0:
        raise ValueError('M3T 跟踪序号无效')
    if type(estimate.get('capture_clock_generation')) is not int or estimate['capture_clock_generation']<0:
        raise ValueError('M3T 时钟代次无效')
    stamp=estimate.get('color_capture_ns')
    if not fresh_capture(stamp, now_ns):
        raise ValueError('M3T 图像或结果已过期')
    multiple = not isinstance(arm, str)
    if multiple and not isinstance(arm, (tuple, list)):
        raise ValueError('M3T 机械臂身份无效')
    arms = tuple(arm) if multiple else (arm,)
    if (not arms or any(not isinstance(name, str) or not name for name in arms)
            or len(set(arms)) != len(arms)):
        raise ValueError('M3T 机械臂身份无效或重复')
    poses = {name: {} for name in arms}
    links = estimate.get('links', [])
    if not isinstance(links, list):
        raise ValueError('M3T 跟踪 link 不完整')
    for link in links:
        if not isinstance(link, dict):
            raise ValueError('M3T 跟踪 link 无效')
        name, link_name = link.get('arm'), link.get('link')
        if (not isinstance(name, str) or name not in poses
                or not isinstance(link_name, str) or link_name in poses[name]):
            raise ValueError('M3T 机械臂身份或 link 重复')
        poses[name][link_name] = transform(link['pose'])
    angles_by_arm = estimate.get('joint_angles_rad', {})
    if not isinstance(angles_by_arm, dict) or set(angles_by_arm) != set(arms):
        raise ValueError('M3T 关节输出机械臂身份不匹配')
    selected = {}
    for name in arms:
        required = required_links[name] if multiple and isinstance(required_links, dict) else required_links
        if not poses[name] or not set(required) <= set(poses[name]):
            raise ValueError('M3T 跟踪 link 不完整')
        angles = angles_by_arm[name]
        if (not isinstance(angles, dict) or set(angles) != set(JOINT_OUTPUTS)
                or any(type(angles[n]) not in (int, float) or not math.isfinite(angles[n])
                       for n in JOINT_OUTPUTS)):
            raise ValueError('M3T 关节输出不完整')
        selected[name] = {n: poses[name][n] for n in required}
    image = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    return estimate, image, selected if multiple else selected[arm]


class M3TPreview:
    def _contexts(self):
        p = self.preview
        contexts = getattr(p, 'arm_previews', None)
        if contexts is None:
            return {p.dashboard.args.arm: p}
        if (not isinstance(contexts, dict) or set(contexts) != {'so101_left', 'so101_right'}
                or any(child.dashboard.args.arm != arm for arm, child in contexts.items())):
            raise ValueError('M3T 双臂上下文身份不匹配')
        return {arm: contexts[arm] for arm in ('so101_left', 'so101_right')}

    def _read_packet(self, contexts):
        multiple=len(contexts)>1
        selected=tuple(contexts) if multiple else self.preview.dashboard.args.arm
        required={arm:child.renderer._meshes for arm,child in contexts.items()} if multiple else self.preview.renderer._meshes
        estimate,image,poses=parse_packet(self.packet_path.read_bytes(),time.monotonic_ns(),
            self.preview.dashboard.main_config,selected,required)
        if self.prior_service:
            prior=estimate.get('joint_priors',{})
            reported=prior.get('priors',[]) if isinstance(prior,dict) else None
            expected={(arm,joint) for arm in contexts for joint in JOINT_OUTPUTS}
            error='M3T 双臂十二项实时先验尚未全部生效或已过期' if multiple else 'M3T 实时五轴与夹爪六项先验尚未全部生效或已过期'
            if not isinstance(reported,list):raise ValueError(error)
            identities=[(entry.get('arm'),entry.get('joint')) for entry in reported if isinstance(entry,dict)]
            if (prior.get('status')!='applied' or type(prior.get('prior_count')) is not int
                or prior['prior_count']!=len(expected) or len(reported)!=len(expected)
                or any(not isinstance(arm,str) or not isinstance(joint,str) for arm,joint in identities)
                or set(identities)!=expected
                or prior.get('controller_id')!=self.prior_controller or prior.get('time_basis')!='received'
                or type(prior.get('age_ns')) is not int or not 0<=prior['age_ns']<=100_000_000):
                raise ValueError(error)
        return estimate,image,poses

    def tcp_poses(self, estimate):
        """Build read-only TCP pose evidence from the M3T estimate."""
        model=self.runtime_model
        if not isinstance(model,dict) or model.get('root_link')!='base_link':
            raise ValueError('M3T runtime model root_link 必须是 base_link')
        contexts=self._contexts()
        arms=tuple(contexts)
        links=estimate.get('links') if isinstance(estimate,dict) else None
        if not isinstance(links,list):
            raise ValueError('M3T TCP 位姿缺少 links')
        by_arm={arm:{} for arm in arms}
        model_links={link['name'] for link in model['links']}
        for link in links:
            if not isinstance(link,dict):
                raise ValueError('M3T TCP link 无效')
            arm=link.get('arm'); name=link.get('link')
            if not isinstance(arm,str) or arm not in by_arm or not isinstance(name,str) or name not in model_links:
                raise ValueError('M3T TCP link 机械臂或 link 身份无效')
            if name in ('base_link','gripper_link'):
                if name in by_arm[arm]:
                    raise ValueError('M3T TCP link 重复')
                by_arm[arm][name]=transform(link.get('pose'))
        angles_by_arm=estimate.get('joint_angles_rad') if isinstance(estimate,dict) else None
        if not isinstance(angles_by_arm,dict) or set(angles_by_arm)!=set(arms):
            raise ValueError('M3T TCP 关节输出机械臂身份不匹配')
        result={}
        for arm in arms:
            if set(by_arm[arm])!={'base_link','gripper_link'}:
                raise ValueError('M3T TCP 位姿缺少 base_link 或 gripper_link')
            angles=angles_by_arm[arm]
            if (not isinstance(angles,dict)
                    or any(name not in angles or type(angles[name]) not in (int,float)
                           or not math.isfinite(angles[name]) for name in NAMES)):
                raise ValueError('M3T TCP 五轴角度无效')
            fk_pose=fk(model,[math.degrees(angles[name]) for name in NAMES])
            camera_from_fk=by_arm[arm]['base_link']@fk_pose
            camera_from_m3t=by_arm[arm]['gripper_link']
            difference=float(np.max(np.abs(camera_from_fk-camera_from_m3t)))
            if not math.isfinite(difference) or difference<0:
                raise ValueError('M3T TCP 位姿差异无效')
            if difference>TCP_MAX_MATRIX_ABS_DIFFERENCE:
                raise ValueError(f'M3T TCP 位姿差异过大：{difference:.9g}')
            result[arm]={'T_camera_from_base_link':by_arm[arm]['base_link'].reshape(-1).tolist(),
                'T_base_link_from_gripper_link':fk_pose.reshape(-1).tolist(),
                'T_camera_from_gripper_link_fk':camera_from_fk.reshape(-1).tolist(),
                'T_camera_from_gripper_link_m3t':camera_from_m3t.reshape(-1).tolist(),
                'max_matrix_abs_difference':difference}
        return result


class M3TPacketSubscriber(M3TPreview):
    """Read the current atomic native packet without controlling its producer."""

    attached = True

    def __init__(self, preview, packet_path, *, prior_service, prior_controller, publish_tf=False):
        self.preview = preview
        self.packet_path = Path(packet_path)
        self.prior_service = prior_service
        self.prior_controller = prior_controller
        self.publish_tf = publish_tf
        self.runtime_model = preview.dashboard.model
        contexts = self._contexts()
        if len(contexts) != 2 or not prior_service or not prior_controller:
            raise ValueError('独立 M3T 预览需要双臂和实时编码器先验')
        self.gripper_ranges = {arm: gripper_calibration_values(child.renderer.gripper_calibration)
                               for arm, child in contexts.items()}

    def read_sample(self):
        contexts = self._contexts()
        try:
            estimate, image, poses = self._read_packet(contexts)
        except FileNotFoundError as exc:
            raise ValueError('尚未收到独立 M3T 预览') from exc
        except (OSError, KeyError, TypeError, OverflowError) as exc:
            raise ValueError('独立 M3T 预览无效或不可读取') from exc
        if (estimate.get('encoder_reference_links') is not True
                or estimate.get('joints_locked_to_feedback', False) is not False):
            raise ValueError('独立 M3T 预览需要编码器参考链接与关节软先验')
        if 'mount_refinement' in estimate:
            # The independent tracker owns its initialization/prior centers;
            # dashboard CAD overlays can use a different nominal reference.
            mount_refinement_diagnostics(estimate, {arm: None for arm in contexts},
                self.runtime_model['root_link'], joints_locked_to_feedback=False)
        tcp = self.tcp_poses(estimate)
        priors = {(item['arm'], item['joint']): item.get('angle_rad')
                  for item in estimate['joint_priors']['priors']}
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in priors.values()):
            raise ValueError('独立 M3T 编码器先验角度无效')
        feedback = {}
        for arm in contexts:
            closed, _, scale = self.gripper_ranges[arm]
            gripper_percent = (priors[arm, 'gripper'] - closed) / scale
            # Native JSON rounds float angles, including the encoder endpoints.
            if not -1e-4 <= gripper_percent <= 100.0001:
                raise ValueError('独立 M3T 夹爪编码器先验超出标定行程')
            feedback[arm] = [math.degrees(priors[arm, name]) for name in NAMES]
            feedback[arm].append(min(100., max(0., gripper_percent)))
        return estimate, image, poses, tcp, feedback

    def state(self):
        stamp = None
        error = None
        estimate = None
        try:
            estimate = self.read_sample()[0]
            stamp = estimate['color_capture_ns']
        except ValueError as exc:
            error = str(exc)
        now = time.monotonic_ns()
        fresh = fresh_capture(stamp, now)
        refining = estimate is not None and 'mount_refinement' in estimate
        return {
            'attached': True, 'enabled': True, 'running': fresh, 'packet_ready': fresh,
            'packet_capture_age_seconds': (now - stamp) / 1e9 if stamp is not None else None,
            'packet_error': error, 'fixed_base': not refining if estimate is not None else None,
            'encoder_reference_links': True, 'joints_locked_to_feedback': False,
            'mount_refinement': copy.deepcopy(estimate['mount_refinement']) if refining else {'enabled': False},
            'mode_description': ENCODER_REFERENCE_MOUNT_DESCRIPTION if refining else
                ('底座固定 · 五关节 + 夹爪实时软先验' if estimate is not None else '等待原生跟踪模式'),
            'joint_priors_enabled': True, 'joint_priors_count': 6 * len(self.gripper_ranges),
            'native_tf_enabled': self.publish_tf,
            'native_tf_capability': 'perception/M3tTf' if self.publish_tf else None,
            'native_tf_frame_prefix': '' if self.publish_tf else None,
            'native_tf_topology': ('cad_mount_correction' if refining else 'cad_mount_fixed_base')
                if self.publish_tf and estimate is not None else None,
        }


class M3TExperiment(M3TPreview):
    attached=False

    def __init__(self,preview,command,run_root,cache_directory,prior_service=None,prior_controller=None,
                 *,mount_refinement=False,mount_translation_std_m=.02,mount_rotation_std_deg=5.,
                 end_effector_priority=False,correspondence_iterations=3,
                 modeled_occlusions=False,publish_tf=False):
        if end_effector_priority and not mount_refinement:
            raise ValueError('M3T end effector priority 需要 mount refinement')
        if (type(correspondence_iterations) is not int
                or not 1 <= correspondence_iterations <= 20):
            raise ValueError('M3T correspondence iterations 必须是 1..20 的整数')
        if type(modeled_occlusions) is not bool:
            raise ValueError('M3T modeled occlusions 必须是布尔值')
        if type(publish_tf) is not bool:
            raise ValueError('M3T publish TF 必须是布尔值')
        self.publish_tf=publish_tf
        self.end_effector_priority=end_effector_priority
        self.correspondence_iterations=correspondence_iterations
        self.modeled_occlusions=modeled_occlusions
        self.visual_link_weights=END_EFFECTOR_VISUAL_LINK_WEIGHTS.copy() if end_effector_priority else {}
        self.preview=preview;self.command=Path(command);self.run_root=Path(run_root)
        self.cache_directory=Path(cache_directory)
        self.prior_service=prior_service;self.prior_controller=prior_controller
        self.process=None;self.run=None;self.packet_path=None;self.enabled=False
        self.started=None;self.clock=None;self.last_sequence=None
        self.gripper_calibration=None
        self.gripper_calibrations={}
        self.mount_refinement=mount_refinement
        self.mount_config=mount_refinement_config(mount_translation_std_m,mount_rotation_std_deg)
        self.nominal_bases={}
        self.runtime_model=None

    def _mount_metadata(self):
        metadata={'end_effector_priority':self.end_effector_priority,
            'visual_link_weights':self.visual_link_weights.copy(),
            'correspondence_iterations':self.correspondence_iterations,
            'modeled_occlusions':self.modeled_occlusions,
            'native_tf_enabled':self.publish_tf,
            'native_tf_capability':'perception/M3tTf' if self.publish_tf else None,
            'native_tf_frame_prefix':'' if self.publish_tf else None,
            'native_tf_topology':'cad_mount_correction' if self.publish_tf else None}
        if self.mount_refinement:
            description=MOUNT_REFINEMENT_DESCRIPTION
            if self.end_effector_priority:description+=' · '+END_EFFECTOR_PRIORITY_DESCRIPTION
            metadata.update({'mount_refinement':copy.deepcopy(self.mount_config),
                'mode_description':description,'joints_locked_to_feedback':True})
        return metadata

    def start(self):
        if self.process is not None and self.process.poll() is None:
            raise ValueError('M3T 已在运行；需要时先停止再按当前反馈重启')
        p=self.preview;d=p.dashboard;p.check_sources();pitch=p.read_pitch()
        contexts=self._contexts();multiple=len(contexts)>1
        if self.mount_refinement and (not self.prior_service
                or getattr(d.args,'m3t_prior_service',None)!=self.prior_service
                or p.mount_source.get('nominal') is not True or p.pitch_source!='assumed_fixed'
                or any(child.mount_source.get('nominal') is not True for child in contexts.values())):
            raise ValueError('M3T mount refinement 需要 nominal mount、显式固定 pitch 与共用 RBT 实时反馈')
        if not self.command.is_file() or not os.access(self.command,os.X_OK):raise ValueError('M3T 实验启动器尚未准备好')
        primary=next(iter(contexts.values()))
        source_hash=sha(primary.dashboard.args.robot_model)
        calibrations={};source_hashes={}
        # Validate every source before reading feedback or creating run files.
        for arm, child in contexts.items():
            if child is not p:
                child.check_sources()
                if not math.isclose(child.read_pitch(),pitch,rel_tol=0,abs_tol=1e-9):
                    raise ValueError('M3T 双臂相机 pitch 假设不一致')
            source_hashes[arm]=sha(child.dashboard.args.robot_model)
            if source_hashes[arm]!=source_hash:
                raise ValueError('M3T 双臂必须使用同一 canonical 机器人模型')
            calibration=copy.deepcopy(child.renderer.gripper_calibration)
            gripper_calibration_values(calibration)
            calibrations[arm]=calibration
        runtime_model,audit=runtime_robot_model(primary.dashboard.args.robot_model,
            calibrations[next(iter(contexts))])
        if self.end_effector_priority:
            mesh_links={link['name'] for link in runtime_model['links']
                if link.get('mesh') and Path(link['mesh']).is_file()}
            if not set(self.visual_link_weights)<=mesh_links:
                raise ValueError('M3T 夹爪刚性结构优先需要 gripper_link 与 moving_jaw_so101_v1_link 的 mesh')
        if multiple:
            joint=next(item for item in runtime_model['joints'] if item.get('name')=='gripper')
            for calibration in calibrations.values():
                closed,opened,_=gripper_calibration_values(calibration)
                joint['lower']=min(joint['lower'],closed)
                joint['upper']=max(joint['upper'],opened)
            audit['runtime_limits_rad']={'lower':joint['lower'],'upper':joint['upper']}
            audit['per_arm_gripper_calibration']=copy.deepcopy(calibrations)
            audit['per_arm_source_robot_json_sha256']=source_hashes
            audit['per_arm_calibration_sha256']={arm:c['calibration_sha256'] for arm,c in calibrations.items()}
        arms=[];arms_source={};mappings=[];gripper_provenances={}
        for arm, child in contexts.items():
            dashboard=child.dashboard;calibration=calibrations[arm]
            closed,opened,_=gripper_calibration_values(calibration)
            before=dashboard.read_arm();time.sleep(.15);after=dashboard.read_arm()
            feedback=averaged_feedback(before,after)
            q=feedback[:5].tolist();fk(dashboard.model,q)
            jaw_percent=float(feedback[5])
            jaw_angle=gripper_angle_from_percent(jaw_percent,calibration)
            angles={n:math.radians(v) for n,v in zip(NAMES,q)};angles['gripper']=jaw_angle
            arms.append({'name':arm,'fixed_base':not self.mount_refinement,
                'base_in_camera':child.B.reshape(-1).tolist(),'joint_angles':angles})
            gripper_provenance={'calibration_sha256':calibration['calibration_sha256'],
                'feedback_percent':jaw_percent,'seed_angle_rad':jaw_angle,
                'closed_rad':closed,'open_rad':opened,
                'zero_reference':calibration['zero_reference'],
                'mapping_assumption':GRIPPER_MAPPING_ASSUMPTION}
            gripper_provenances[arm]=gripper_provenance
            source={'mount_source':copy.deepcopy(child.mount_source),'joint_before':before,
                'joint_after':after,'gripper_seed':gripper_provenance}
            if multiple:
                source['source_robot_json_sha256']=source_hashes[arm]
            if child.audit_path is not None:
                key='legacy_binding_audit_sha256' if child.mount_source['nominal'] else 'mount_audit_sha256'
                source[key]=sha(child.audit_path)
            arms_source[arm]=source
            mappings.extend(prior_mappings(arm,calibration))
        self.run_root.mkdir(parents=True,exist_ok=True)
        run=Path(tempfile.mkdtemp(prefix='session-',dir=self.run_root))
        component='bimanual' if multiple else d.args.arm.removeprefix('so101_')
        packet_dir=Path(tempfile.mkdtemp(prefix=f'rbt-m3t-{component}-',dir='/dev/shm'))
        packet_path=packet_dir/'latest.bin'
        # One shared model and mesh snapshot feeds every arm in the native Tracker.
        (run/'meshes').mkdir()
        mesh_audit={}
        for link in runtime_model['links']:
            if link.get('mesh'):
                source_mesh=Path(link['mesh'])
                mesh_name=f"meshes/{link['name']}.obj"
                shutil.copyfile(source_mesh,run/mesh_name)
                mesh_audit[mesh_name]=sha(run/mesh_name)
                link['mesh']=mesh_name
        audit['mesh_sha256']=mesh_audit
        exclusive(run/'robot.json',encoded(runtime_model))
        self.runtime_model=copy.deepcopy(runtime_model)
        exclusive(run/'runtime-model-audit.json',encoded(audit))
        # OpenCV FileStorage rejects JSON nulls even in unused provenance.
        mount_metadata={key:value for key,value in self._mount_metadata().items() if value is not None}
        provenance={'purpose':'user-authorized M3T correction experiment',
            'mount_source':copy.deepcopy(p.mount_source),'pitch_source':p.pitch_source,
            'pitch_deg':p.expected_pitch,'solver_accepted':False,'fixed_base':not self.mount_refinement,
            **mount_metadata}
        if multiple:
            provenance['arms_source']=arms_source
        else:
            provenance.update(arms_source[d.args.arm])
        init={'schema_version':1,'coordinate_frame':'orbbec335/color_optical',
            'serial':d.main_config['camera_serial'],'arms':arms,'provenance':provenance}
        exclusive(run/'initialization.json',encoded(init))
        config={'schema_version':1,'robot_model':'robot.json',
            'initialization':'initialization.json','cache_directory':os.path.relpath(self.cache_directory,run),
            'model_divides':2,'model_points':200,'model_image_size':512,
            'correspondence_iterations':self.correspondence_iterations,'update_iterations':2,
            'modalities':{'region':True,'depth':True,'texture':False},
            'occlusion':{'measured':True,'modeled':self.modeled_occlusions,
                'region_checking':False,'image_size':512}}
        if self.prior_service:
            config['joint_priors']={'enabled':True,'service_name':self.prior_service,
                'controller_id':self.prior_controller,'required':True,'initialize_from_measurement':True,
                'time_basis':'received','maximum_age_ms':100,'capacity':128,
                'mappings':mappings}
        if self.mount_refinement:config['mount_refinement']=copy.deepcopy(self.mount_config)
        if self.end_effector_priority:config['visual_link_weights']=self.visual_link_weights.copy()
        exclusive(run/'tracking.json',encoded(config))
        command=[str(self.command),'run','--config',str(run/'tracking.json'),
            '--provider','orbbec335','--component-id',f'm3t-{component}-web-experiment',
            '--duration-sec','86400','--preview-packet',str(packet_path)]
        if self.publish_tf:command.append('--publish-tf')
        with (run/'tracker.log').open('ab') as log:
            process=subprocess.Popen(command,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        self.process=process;self.run=run;self.packet_path=packet_path;self.enabled=True
        self.nominal_bases={arm:child.B.copy() for arm,child in contexts.items()}
        self.mount_root_link=runtime_model['root_link']
        self.gripper_calibrations=calibrations
        self.gripper_calibration=calibrations[d.args.arm]
        self.started=time.monotonic();self.clock=None;self.last_sequence=None
        launch={'pid':process.pid,'command':str(self.command),
            'packet':str(packet_path),'motor_commands':False,'fixed_base':not self.mount_refinement,
            **self._mount_metadata(),
            'mount_source':copy.deepcopy(p.mount_source),'pitch_source':p.pitch_source,
            'pitch_deg':p.expected_pitch,'solver_accepted':False,
            'joint_priors_enabled':bool(self.prior_service),
            'joint_priors_count':len(mappings) if self.prior_service else 0,
            'gripper_prior_provenance':gripper_provenances[d.args.arm] if self.prior_service else None,
            'runtime_model_audit':'runtime-model-audit.json'}
        if multiple:
            launch['arms_source']=arms_source
            launch['per_arm_gripper_prior_provenance']=gripper_provenances if self.prior_service else None
        exclusive(run/'launch.json',encoded(launch))
        # A comparison frame must be produced after M3T starts, not reused FK cache.
        with p.lock:p.sample=None;p.error=None;p.enabled=True
        p.event.set()
        if self.mount_refinement:
            message='M3T 已启动；'+self._mount_metadata()['mode_description']+'。'
        elif self.prior_service:
            message='M3T 已启动；双臂各六项、共十二项实时软先验，图像继续修正姿态。' if multiple else 'M3T 已启动；实时五轴与夹爪共六项角度作为软先验，图像继续修正姿态。'
        else:
            label='双臂' if multiple else ('右臂' if d.args.arm=='so101_right' else '左臂')
            message=f'M3T 已启动，首次会生成 Region/Depth 模型缓存；请保持{label}静止。'
        return {'message':message}

    def stop(self):
        self.enabled=False
        process=self.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
        with self.preview.lock:self.preview.sample=None
        return {'message':'M3T 已停止，返回原始反馈叠加。'}

    def read_frame(self):
        if not self.enabled:raise ValueError('M3T 未启动')
        if self.process is None or self.process.poll() is not None:
            raise ValueError('M3T 进程已退出，请查看运行日志后重启')
        if self.packet_path is None or not self.packet_path.exists():raise ValueError('M3T 正在初始化或生成模型缓存…')
        contexts=self._contexts()
        for arm, calibration in getattr(self,'gripper_calibrations',{}).items():
            child=contexts[arm]
            child.check_sources()
            if (child.renderer.gripper_calibration!=calibration
                    or sha(child.dashboard.args.calibration)!=calibration['calibration_sha256']):
                raise ValueError('M3T 夹爪标定来源已变化，请按当前反馈重启跟踪')
        estimate,image,poses=self._read_packet(contexts)
        if getattr(self,'mount_refinement',False):
            mount_refinement_diagnostics(estimate,self.nominal_bases,self.mount_root_link)
        if getattr(self,'end_effector_priority',False):
            weights=estimate.get('visual_link_weights')
            if not isinstance(weights,dict) or weights!=self.visual_link_weights:
                raise ValueError('M3T 夹爪刚性结构视觉权重缺失或不匹配，请检查 native binary')
        identity=(estimate['capture_clock_source'],estimate.get('capture_clock_generation'))
        if self.clock is None:self.clock=identity
        if identity!=self.clock:raise ValueError('M3T 相机时钟已变化，请按当前反馈重启跟踪')
        if self.last_sequence is not None and estimate['frame_sequence']<self.last_sequence:
            raise ValueError('M3T 跟踪序号回退')
        self.last_sequence=estimate['frame_sequence']
        return estimate,image,poses

    def state(self):
        running=self.process is not None and self.process.poll() is None
        contexts=self._contexts()
        def gripper_source(calibration):
            return {key:calibration[key] for key in ('calibration_sha256','closed_rad','open_rad',
                'scale_rad_per_percent','zero_reference')} | {'mapping_assumption':GRIPPER_MAPPING_ASSUMPTION}
        result={'enabled':self.enabled,'running':running,'run_directory':str(self.run) if self.run else None,
            'exit_code':None if self.process is None else self.process.poll(),
            'fixed_base':not self.mount_refinement,**self._mount_metadata(),
            'joint_priors_enabled':bool(self.prior_service),
            'mount_source':copy.deepcopy(self.preview.mount_source),'pitch_source':self.preview.pitch_source,
            'pitch_deg':self.preview.expected_pitch,'solver_accepted':False,
            'joint_priors_count':6*len(contexts) if self.prior_service else 0,
            'gripper_prior_provenance':None if not self.prior_service or self.gripper_calibration is None else gripper_source(self.gripper_calibration),
            'packet_ready':bool(self.packet_path and self.packet_path.exists())}
        if len(contexts)>1:
            result['per_arm_gripper_prior_provenance']={arm:gripper_source(c) for arm,c in self.gripper_calibrations.items()} if self.prior_service else None
        return result
