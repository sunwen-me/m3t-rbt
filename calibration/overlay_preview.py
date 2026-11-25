"""Read-only stationary overlay worker; no tracking or pose correction."""
from __future__ import annotations
import json
from pathlib import Path
import threading
import time
import urllib.request
import cv2
import numpy as np
from shared_board_web import MODEL_SHA, encoded, exclusive, fk, sha, transform
from handeye import marker_corners
from m3t_preview import fresh_capture
from model_overlay import ModelOverlay
from gripper_calibration import load_gripper_calibration


def load_nominal_mount_prior(path, arm, static_config, robot_model):
    """Bind a nominal camera-from-base seed to the local config and model bytes."""
    path=Path(path).resolve()
    prior=json.loads(path.read_text())
    try:
        if (type(prior['schema_version']) is not int or prior['schema_version']!=1
                or prior['record_type']!='nominal-mount-prior-not-runtime-initialization'
                or prior['solver_accepted'] is not False or prior['units']['translation']!='m'
                or prior['frames']['camera']!='orbbec335/color_optical'
                or prior['frames']['arm_base']!='official SO101 base_link'):
            raise ValueError('CAD 粗初值类型、坐标系或单位不匹配')
        pitch=prior['capture_pitch_deg']
        if type(pitch) not in (int,float) or not np.isfinite(pitch):
            raise ValueError('CAD 粗初值 pitch 无效')
        source=prior['source']
        if source['static_config']['sha256_as_recorded']!=sha(static_config):
            raise ValueError('CAD 粗初值源静态配置 hash 不匹配')
        model=json.loads(Path(robot_model).read_text())
        model_sha=sha(robot_model)
        if (source['urdf']['sha256']!=MODEL_SHA or model.get('source_sha256')!=MODEL_SHA
                or model.get('root_link')!='base_link'
                or source['model_json']['sha256']!=model_sha):
            raise ValueError('CAD 粗初值源模型 hash 或 base_link 不匹配')
        matches=[entry for entry in prior['arms'] if entry['name']==arm]
        if len(matches)!=1:
            raise ValueError('CAD 粗初值必须唯一匹配机械臂 name')
        entry=matches[0]
        pose=np.asarray(entry['pose4x4'],dtype=float)
        rowmajor=np.asarray(entry['rowmajor16'],dtype=float)
        if (pose.shape!=(4,4) or rowmajor.shape!=(16,)
                or not np.allclose(pose.reshape(-1),rowmajor,rtol=0,atol=1e-12)):
            raise ValueError('CAD 粗初值 pose4x4 / rowmajor16 不一致')
        B=transform(pose)
        if (not np.allclose(B[3],[0,0,0,1],rtol=0,atol=1e-8)
                or not np.allclose(B[:3,:3].T@B[:3,:3],np.eye(3),rtol=0,atol=1e-6)
                or B[2,3]<=0):
            raise ValueError('CAD 粗初值必须为刚体变换且具有正光学深度')
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('CAD 粗初值缺少有效的来源或位姿字段') from exc
    source_info={'kind':'nominal_cad','path':str(path),'sha256':sha(path),
        'nominal':True,'solver_accepted':False,'arm':arm,'capture_pitch_deg':float(pitch),
        'source_static_config_sha256':sha(static_config),'source_model_json_sha256':model_sha}
    return B,float(pitch),source_info


class OverlayPreview:
    def __init__(self,dashboard,audit_path,report_path,static_config,pitch_url,
                 nominal_mount_prior=None,fixed_pitch_assumption_deg=None,*,start_worker=True):
        if fixed_pitch_assumption_deg is not None and nominal_mount_prior is None:
            raise ValueError('固定 pitch 假设仅允许在 nominal prior 模式使用')
        self.dashboard=dashboard
        if bool(audit_path)!=bool(report_path):
            raise ValueError('legacy audit/report 必须同时提供')
        self.audit_path=Path(audit_path) if audit_path else None
        self.report_path=Path(report_path) if report_path else None
        self.static_config=Path(static_config)
        self.hashes={str(self.static_config):sha(self.static_config)}
        self.X=None; self.B=None; self.expected_pitch=None
        self.pitch_url=pitch_url; self.pitch_source='live_controller'
        self.audit=None; self.report=None; self.mount_source=None
        if self.audit_path is not None:
            self.audit=json.loads(self.audit_path.read_text()); self.report=json.loads(self.report_path.read_text())
            if self.audit['model_urdf_sha256']!=MODEL_SHA or self.audit['source_report_sha256']!=sha(self.report_path):
                raise ValueError('叠加外参来源与当前模型不一致')
            self.hashes.update({str(p):sha(p) for p in (self.audit_path,self.report_path)})
            if self.hashes[str(self.static_config)]!=self.audit['config_after_sha256']:
                raise ValueError('静态 TF 配置与本次部署记录不一致')
            self.B=transform(self.audit['T_camera_from_newbase'])
            candidate=self.report['arms'][dashboard.args.arm]['candidate']
            if not np.allclose(self.B,transform(candidate['T_camera_from_base']),atol=1e-12):
                raise ValueError('已部署初值与报告不一致')
            self.X=transform(candidate['T_gripper_from_marker'])
            self.expected_pitch=float(self.audit['capture_pitch_deg_assumed'])
            self.mount_source={'kind':'legacy_audit','path':str(self.audit_path.resolve()),
                'sha256':self.hashes[str(self.audit_path)],'nominal':False}
        elif nominal_mount_prior is None:
            raise ValueError('需要 legacy 安装记录或 CAD 名义安装初值')
        if nominal_mount_prior is not None:
            self.B,self.expected_pitch,self.mount_source=load_nominal_mount_prior(
                nominal_mount_prior,dashboard.args.arm,self.static_config,dashboard.args.robot_model)
            self.X=None
            self.hashes[self.mount_source['path']]=self.mount_source['sha256']
            self.hashes[str(Path(dashboard.args.robot_model))]=self.mount_source['source_model_json_sha256']
            if fixed_pitch_assumption_deg is not None:
                if (type(fixed_pitch_assumption_deg) not in (int,float)
                        or not np.isfinite(fixed_pitch_assumption_deg)
                        or not np.isclose(fixed_pitch_assumption_deg,self.expected_pitch,rtol=0,atol=1e-9)):
                    raise ValueError('固定 pitch 假设与 CAD 粗初值采集 pitch 不匹配')
                self.expected_pitch=float(fixed_pitch_assumption_deg)
                self.pitch_source='assumed_fixed'
        self.renderer=ModelOverlay(dashboard.args.robot_model,load_gripper_calibration(dashboard.args.calibration))
        self.lock=threading.Lock(); self.enabled=False; self.sample=None; self.error=None
        self.layers={'encoder':True,'m3t':True,'predicted':self.X is not None,'observed':self.X is not None}
        self.event=threading.Event()
        self.m3t=None
        self.thread=threading.Thread(target=self.run,daemon=True)
        if start_worker: self.thread.start()

    def set_enabled(self,value):
        if type(value) is not bool: raise ValueError('启用状态无效')
        with self.lock:
            self.enabled=value; self.sample=None; self.error=None
        self.event.set()
        return {'message':'实时叠加已开启；只手动移动左臂，停稳后判断重合情况。' if value else '实时叠加已停止。'}

    def set_layers(self,value):
        expected={'encoder','m3t','predicted','observed'}
        if type(value) is not dict or set(value)!=expected or any(type(flag) is not bool for flag in value.values()):
            raise ValueError('显示图层无效')
        layers=value.copy()
        if self.X is None and (layers['predicted'] or layers['observed']):
            raise ValueError('粗初值未标定贴码外参，贴码图层不可用')
        with self.lock:
            self.layers=layers; self.sample=None; self.error=None
        self.event.set()
        return {'message':'叠加显示图层已更新。'}

    def check_sources(self):
        self.dashboard.check_binding()
        if any(sha(p)!=value for p,value in self.hashes.items()):
            raise ValueError('模型或安装配置已变化，请重新启动验证服务')

    def read_pitch(self):
        if self.pitch_source=='assumed_fixed':
            return self.expected_pitch
        with urllib.request.urlopen(self.pitch_url,timeout=2) as response:
            data=json.loads(response.read(8192))
        angle=data.get('angle')
        if data.get('online') is not True or type(angle) not in (int,float) or not np.isfinite(angle):
            raise ValueError('云台角度反馈不在线')
        if abs(angle-self.expected_pitch)>.5:
            raise ValueError(f'云台当前 {angle:.2f}°，请保持在 {self.expected_pitch:.0f}° 附近；本页不会移动云台')
        return float(angle)

    def acquire(self):
        with self.lock: layers=self.layers.copy()
        self.check_sources(); pitch=self.read_pitch(); d=self.dashboard
        if self.m3t is not None and self.m3t.enabled: self.m3t.read_frame()
        before=d.read_arm()
        deadline=time.monotonic()+2
        tracked=None; tracked_poses=None
        while True:
            if self.m3t is not None and self.m3t.enabled:
                tracked,image,tracked_poses=self.m3t.read_frame()
                packet={'schema':'rbt-m3t-preview-v1','estimate':tracked}
                stamp=tracked['color_capture_ns']/1e9
            else:
                packet,image=d.main_frame()
                stamp=packet['reference']['capture_ns']/1e9
            if stamp>=before['read_completed']: break
            if time.monotonic()>deadline: raise ValueError('尚无反馈读取之后的新鲜图像；请停稳后重试')
            time.sleep(.025)
        after=d.read_arm()
        a=np.asarray(before['degrees_and_gripper_percent'][:5],float)
        b=np.asarray(after['degrees_and_gripper_percent'][:5],float)
        if np.max(np.abs(a-b))>.3: raise ValueError('机械臂正在移动；请停稳后观察叠加')
        if not before['read_completed']<=stamp<=after['read_started']:
            raise ValueError('图像与反馈时间区间不符，请停稳后重试')
        q=((a+b)/2).tolist(); F=fk(d.model,q)
        jaw_before=float(before['degrees_and_gripper_percent'][5])
        jaw_after=float(after['degrees_and_gripper_percent'][5])
        if not np.isfinite([jaw_before,jaw_after]).all() or abs(jaw_before-jaw_after)>.5:
            raise ValueError('夹爪正在开合；请停稳后观察叠加')
        feedback=q+[(jaw_before+jaw_after)/2]
        encoder_result,metadata=self.renderer.render(image.copy(),feedback,self.B,np.asarray(d.main_config['K']))
        result=encoder_result if layers['encoder'] else image.copy()
        tracked_metadata=None
        tcp_poses=None
        if tracked_poses is not None:
            tcp_poses=self.m3t.tcp_poses(tracked)
            tracked_result,tracked_metadata=self.renderer.render_poses(result,tracked_poses,np.asarray(d.main_config['K']),color=(0,165,255))
            if layers['m3t']: result=tracked_result
        marker_rms=None; marker_error='粗初值未标定贴码外参' if self.X is None else None; tracked_marker_rms=None
        if self.X is not None:
            predicted=self.B@F@self.X; k=np.asarray(d.main_config['K'],float)
            corners=marker_corners(.029)@predicted[:3,:3].T+predicted[:3,3]
            if np.any(corners[:,2]<=0): raise ValueError('预测贴码位于相机后方')
            px=(corners@k.T); px=px[:,:2]/px[:,2,None]
            if np.isfinite(px).all() and np.max(np.abs(px))<100000:
                if layers['predicted']:
                    cv2.polylines(result,[np.round(px).astype(np.int32)],True,(255,80,255),2,cv2.LINE_AA)
            from aruco_web import detect_marker
            try:
                actual=detect_marker(image)
                if layers['observed']:
                    cv2.polylines(result,[np.round(actual).astype(np.int32)],True,(80,255,80),2,cv2.LINE_AA)
                marker_rms=float(np.sqrt(np.mean(np.sum((px-actual)**2,axis=1))))
                if tracked_poses is not None:
                    tracked_marker=tracked_poses['gripper_link']@self.X
                    points=marker_corners(.029)@tracked_marker[:3,:3].T+tracked_marker[:3,3]
                    if np.all(points[:,2]>0):
                        uv=points@k.T; uv=uv[:,:2]/uv[:,2,None]
                        tracked_marker_rms=float(np.sqrt(np.mean(np.sum((uv-actual)**2,axis=1))))
                        if layers['m3t'] and np.isfinite(uv).all() and np.max(np.abs(uv))<100000:
                            cv2.polylines(result,[np.round(uv).astype(np.int32)],True,(0,165,255),2,cv2.LINE_AA)
            except ValueError as exc: marker_error=str(exc)
        receipt={'joint_before':before,'joint_after':after,'main_packet':packet,
                 'pitch_deg':pitch,'pitch_source':self.pitch_source,'mount_source':self.mount_source.copy(),
                 'pitch_measurement_semantics':('Explicit fixed pitch assumption; not live feedback'
                     if self.pitch_source=='assumed_fixed' else 'Read from existing controller status, not exposure synchronized'),
                 'timing_semantics':'host publication entry and encoder read interval, not exposure synchronization',
                 'degrees':q,'marker_rms_px':marker_rms,'marker_error':marker_error,
                 'renderer':metadata,
                 'T_camera_from_base':self.B.tolist(),'unmodified_encoder_fk':True,'m3t_enabled':tracked is not None,
                 'm3t_estimate':tracked,'m3t_renderer':tracked_metadata,'m3t_marker_rms_px':tracked_marker_rms,
                 'display_layers':layers.copy()}
        if tcp_poses is not None:
            receipt['tcp_poses']=tcp_poses
        audit_key='legacy_binding_audit_sha256' if self.mount_source['nominal'] else 'audit_sha256'
        if self.audit_path is not None: receipt[audit_key]=self.hashes[str(self.audit_path)]
        return {'created':time.monotonic(),'raw':image,'overlay':result,'receipt':receipt}

    def run(self):
        while True:
            with self.lock: enabled=self.enabled
            if not enabled:
                self.event.wait(.5); self.event.clear(); continue
            # Share the dashboard's hardware lock with capture/check operations.
            if not self.dashboard.lock.acquire(timeout=.1): continue
            try:
                sample=self.acquire()
                with self.lock:
                    if self.enabled and sample['receipt']['display_layers']==self.layers:
                        self.sample=sample; self.error=None
            except Exception as exc:
                with self.lock: self.sample=None; self.error=str(exc)
            finally: self.dashboard.lock.release()
            self.event.wait(.15); self.event.clear()

    def snapshot(self):
        with self.lock:
            age=None if self.sample is None else time.monotonic()-self.sample['created']
            fresh=self.enabled and age is not None and 0<=age<=2
            m3t=self.m3t.state() if self.m3t is not None else None
            if fresh and m3t and m3t.get('attached') is True:
                tracked=self.sample['receipt'].get('m3t_estimate')
                if tracked is not None:
                    fresh=m3t['running'] and fresh_capture(tracked['color_capture_ns'],time.monotonic_ns())
            return self.sample if fresh else None, {'enabled':self.enabled,'ready':bool(fresh),
                'error':self.error or (None if fresh or not self.enabled else '正在读取反馈和图像…'),
                'age_seconds':age,'expected_pitch_deg':self.expected_pitch,
                'pitch_source':self.pitch_source,'mount_source':self.mount_source.copy(),
                'marker_available':self.X is not None,
                'receipt':self.sample['receipt'] if fresh else None,'layers':self.layers.copy(),
                'm3t':m3t}

    def save(self):
        sample,_=self.snapshot()
        if sample is None: raise ValueError('尚无新鲜的静止验证画面')
        path=self.dashboard.output/'verification'/f'view-{time.time_ns()}'
        path.mkdir(parents=True,exist_ok=False)
        exclusive(path/'main.png',cv2.imencode('.png',sample['raw'])[1].tobytes())
        exclusive(path/'overlay.png',cv2.imencode('.png',sample['overlay'])[1].tobytes())
        exclusive(path/'receipt.json',encoded(sample['receipt']))
        return {'message':'验证画面与关节反馈已保存：'+path.name}
