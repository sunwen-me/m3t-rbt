"""Render both arms from one image and one native M3T estimate packet."""
from __future__ import annotations
import threading
import time
import numpy as np
from overlay_preview import OverlayPreview

ARMS=('so101_left','so101_right')
M3T_COLORS={'so101_left':(0,165,255),'so101_right':(235,90,205)}


class BimanualOverlayPreview(OverlayPreview):
    def __init__(self,left,right,static_config,pitch_url,nominal_mount_prior,fixed_pitch_assumption_deg):
        self.dashboard=left
        self.arm_previews={arm:OverlayPreview(d,None,None,static_config,pitch_url,
            nominal_mount_prior=nominal_mount_prior,
            fixed_pitch_assumption_deg=fixed_pitch_assumption_deg,start_worker=False)
            for arm,d in zip(ARMS,(left,right))}
        self.renderer=self.arm_previews[ARMS[0]].renderer
        self.B=self.arm_previews[ARMS[0]].B
        self.X=None; self.audit_path=None; self.report_path=None; self.audit=None; self.report=None
        self.static_config=static_config;self.pitch_url=pitch_url
        self.expected_pitch=self.arm_previews[ARMS[0]].expected_pitch
        self.pitch_source=self.arm_previews[ARMS[0]].pitch_source
        if any(p.expected_pitch!=self.expected_pitch or p.pitch_source!=self.pitch_source
               for p in self.arm_previews.values()):
            raise ValueError('双臂必须共享相机 pitch 来源和固定角度')
        self.mount_source={'kind':'nominal_cad','nominal':True,'solver_accepted':False,
            'arms':{arm:p.mount_source.copy() for arm,p in self.arm_previews.items()},
            'capture_pitch_deg':self.expected_pitch}
        self.hashes={k:v for p in self.arm_previews.values() for k,v in p.hashes.items()}
        self.lock=threading.Lock(); self.enabled=False; self.sample=None; self.error=None
        self.layers={'encoder':False,'m3t':True,'predicted':False,'observed':False}
        self.event=threading.Event();self.m3t=None
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def check_sources(self):
        for p in self.arm_previews.values():p.check_sources()

    def read_pitch(self):
        values=[p.read_pitch() for p in self.arm_previews.values()]
        if not np.allclose(values,values[0],rtol=0,atol=.5):raise ValueError('双臂 pitch 来源不一致')
        return float(values[0])

    def set_enabled(self,value):
        result=super().set_enabled(value)
        if value:result['message']='双臂只读叠加已开启；使用同一相机帧和各臂反馈。'
        return result

    def acquire(self):
        with self.lock:layers=self.layers.copy()
        self.check_sources();pitch=self.read_pitch()
        if getattr(self.m3t,'attached',False) is True:
            return self._acquire_attached(layers,pitch)
        if self.m3t is not None and self.m3t.enabled:self.m3t.read_frame()
        before={arm:p.dashboard.read_arm() for arm,p in self.arm_previews.items()}
        deadline=time.monotonic()+2
        tracked=None;tracked_poses=None
        while True:
            if self.m3t is not None and self.m3t.enabled:
                tracked,image,tracked_poses=self.m3t.read_frame()
                packet={'schema':'rbt-m3t-preview-v1','estimate':tracked}
                stamp=tracked['color_capture_ns']/1e9
            else:
                packet,image=self.dashboard.main_frame()
                stamp=packet['reference']['capture_ns']/1e9
            if stamp>=max(r['read_completed'] for r in before.values()):break
            if time.monotonic()>deadline:raise ValueError('尚无双臂读取之后的新鲜图像；请停稳后重试')
            time.sleep(.025)
        after={arm:p.dashboard.read_arm() for arm,p in self.arm_previews.items()}
        feedback={};receipts={}
        for arm,p in self.arm_previews.items():
            a=np.asarray(before[arm]['degrees_and_gripper_percent'],float)
            b=np.asarray(after[arm]['degrees_and_gripper_percent'],float)
            if a.shape!=(6,) or b.shape!=(6,) or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f'{arm} 关节反馈不完整')
            if np.max(np.abs(a[:5]-b[:5]))>.3 or abs(float(a[5]-b[5]))>.5:
                raise ValueError(f'{arm} 正在移动；请停稳后观察叠加')
            if not before[arm]['read_completed']<=stamp<=after[arm]['read_started']:
                raise ValueError(f'{arm} 图像与反馈时间区间不符')
            feedback[arm]=((a+b)/2).tolist()
            receipts[arm]={'joint_before':before[arm],'joint_after':after[arm],
                'degrees':feedback[arm][:5],'gripper_percent':feedback[arm][5],
                'T_camera_from_base':p.B.tolist(),'mount_source':p.mount_source.copy()}
        tcp=None if tracked is None else self.m3t.tcp_poses(tracked)
        return self._render_sample(image,packet,pitch,layers,feedback,receipts,
            tracked,tracked_poses,tcp,
            'Both arms bracket one exact camera image; software receipt timing, not exposure synchronization')

    def _acquire_attached(self,layers,pitch):
        try:
            tracked,image,poses,tcp,feedback=self.m3t.read_sample()
        except ValueError:
            packet,image=self.dashboard.main_frame()
            return self._render_sample(image,packet,pitch,layers,{}, {},None,None,None,
                'Independent color preview; no associated joint state')
        packet={'schema':'rbt-m3t-preview-v1','estimate':tracked}
        receipts={arm:{'degrees':feedback[arm][:5],'gripper_percent':feedback[arm][5],
            'T_camera_from_base':p.B.tolist(),'mount_source':p.mount_source.copy()}
            for arm,p in self.arm_previews.items()}
        return self._render_sample(image,packet,pitch,layers,feedback,receipts,tracked,poses,tcp,
            'Native M3T estimate and its exact color image; encoder priors selected by received time')

    def _render_sample(self,image,packet,pitch,layers,feedback,receipts,tracked,tracked_poses,tcp,timing):
        encoder_result=image.copy();encoder_meta={}
        for arm,angles in feedback.items():
            p=self.arm_previews[arm]
            encoder_result,encoder_meta[arm]=p.renderer.render(
                encoder_result,angles,p.B,np.asarray(self.dashboard.main_config['K']))
        result=encoder_result if layers['encoder'] else image.copy()
        tracked_meta={}
        if tracked_poses is not None:
            if set(tracked_poses)!=set(ARMS):raise ValueError('双臂 M3T packet 缺少完整的双臂位姿')
            tracked_result=result.copy()
            for arm,p in self.arm_previews.items():
                tracked_result,tracked_meta[arm]=p.renderer.render_poses(tracked_result,tracked_poses[arm],
                    np.asarray(self.dashboard.main_config['K']),color=M3T_COLORS[arm])
            if layers['m3t']:result=tracked_result
        left=receipts.get(ARMS[0],{})
        receipt={'arms':receipts,'joint_before':left.get('joint_before'),'joint_after':left.get('joint_after'),
            'main_packet':packet,'pitch_deg':pitch,'pitch_source':self.pitch_source,
            'mount_source':self.mount_source.copy(),
            'pitch_measurement_semantics':('Explicit fixed pitch assumption; not live feedback'
                if self.pitch_source=='assumed_fixed' else 'Read from controller, not exposure synchronized'),
            'timing_semantics':timing,
            'degrees':left.get('degrees'),
            'marker_rms_px':None,'marker_error':'粗初值未标定贴码外参','m3t_marker_rms_px':None,
            'renderer':encoder_meta,'T_camera_from_bases':{a:p.B.tolist() for a,p in self.arm_previews.items()},
            'unmodified_encoder_fk':True,'m3t_enabled':tracked is not None,
            'm3t_estimate':tracked,'m3t_renderer':tracked_meta,'display_layers':layers.copy()}
        if tcp is not None:
            receipt['tcp_poses']=tcp
        return {'created':time.monotonic(),'raw':image,'overlay':result,'receipt':receipt}
