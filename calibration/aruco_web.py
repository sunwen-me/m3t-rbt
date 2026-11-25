#!/usr/bin/env python3
"""Loopback dashboard for a rigid gripper marker and fixed main camera."""
from __future__ import annotations
import argparse
import copy
import hmac
import json
import math
from pathlib import Path
import secrets
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from shared_board_web import (Dashboard as SharedDashboard, MODEL_SHA, Server,
    encoded, exclusive, fk, main_intrinsics, sha, transform)
from handeye import marker_corners, solve_manifest


def detect_marker(image, marker_id=1):
    parameters=cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),parameters)
    corners,ids,_=detector.detectMarkers(image)
    ids=[] if ids is None else ids.flatten().tolist()
    matches=[np.asarray(c).reshape(4,2) for c,i in zip(corners,ids) if i==marker_id]
    if len(matches)!=1:
        raise ValueError('未看到夹爪 ID 1，请让码完整朝向主相机' if not matches else '出现多个 ID 1，请移走后面的 ChArUco 板或其他同号码')
    pixels=matches[0]
    edges=np.linalg.norm(pixels-np.roll(pixels,1,axis=0),axis=1)
    if min(edges)<25: raise ValueError('码太小或倾角太大，请让每条边至少达到 25 像素')
    return pixels


def observation(image, config, side):
    pixels=detect_marker(image)
    objects=marker_corners(side)
    k=np.asarray(config['K'],float)
    ok,r,t=cv2.solvePnP(objects,pixels,k,None,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise ValueError('码位姿求解失败')
    projected=cv2.projectPoints(objects,r,t,k,None)[0].reshape(-1,2)
    rms=float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1))))
    T=np.eye(4); T[:3,:3]=cv2.Rodrigues(r)[0]; T[:3,3]=t.reshape(3)
    if not np.isfinite(T).all() or np.any((objects@T[:3,:3].T+T[:3,3])[:,2]<=0) or rms>2:
        raise ValueError('码位姿质量不足，请停稳并避免反光或遮挡')
    return {'corners_px':pixels.tolist(),'K':config['K'],
            'T_camera_from_marker_seed':T.tolist(),'pnp_rms_px':rms}


class Dashboard(SharedDashboard):
    def __init__(self,args):
        self.args=args; self.output=Path(args.output); self.output.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()
        self.session=json.loads(Path(args.main_session).read_text()); self.main_config=main_intrinsics(self.session)
        self.model=json.loads(Path(args.robot_model).read_text())
        if self.model.get('source_sha256')!=MODEL_SHA or self.model.get('root_link')!='base_link':
            raise ValueError('请使用当前官方 SO101 软鳍版模型')
        self.hashes={k:sha(getattr(args,k)) for k in ('main_session','robot_model','calibration')}
        self.feedback_reader=None
        if getattr(args,'m3t_prior_service',None):
            from joint_feedback import RbtFeedbackReader
            self.feedback_reader=RbtFeedbackReader(args.m3t_prior_service,args.joint_feedback_controller,
                source_arm=args.arm.removeprefix('so101_'),python_executable=args.driver_python)
        self.binding={'schema_version':1,'mode':'fixed-main-gripper-aruco','hashes':self.hashes,
            'arm':args.arm,'port':args.arm_port,'dictionary':'DICT_4X4_50','marker_id':1,'marker_side_m':.029}
        binding=self.output/'binding.json'
        if binding.exists():
            if json.loads(binding.read_text())!=self.binding: raise ValueError('配置已变化，请新建采集目录')
        else:
            if any(self.output.iterdir()): raise ValueError('输出目录必须为空')
            exclusive(binding,encoded(self.binding))
        self.manifest_path=self.output/'manifest.json'
        self.manifest={'schema_version':1,'setup_id':self.output.name,'setup_confirmed_unchanged':True,
            'marker_side_m':.029,'nominal_T_gripper_from_marker':{args.arm:np.eye(4).tolist()},'samples':[]}
        if self.manifest_path.exists(): self.manifest=json.loads(self.manifest_path.read_text())
        self.report=None; self.solving=False; self.error=None; self.feedback='尚未检查'; self.arm_ready=False
        reports=sorted(self.output.glob('result-*.json'))
        if reports:
            path=reports[-1]; report=json.loads(path.read_text())
            if report.get('arms',{}).get(args.arm,{}).get('sample_count')==len(self.manifest['samples']):
                self.report={'filename':path.name,'contents':report}

    def read_arm(self):
        if getattr(self,'feedback_reader',None) is not None:
            return self.feedback_reader.read()
        return super().read_arm()

    def check_arm(self):
        self.arm_ready=False
        if self.feedback_reader is not None:
            self.check_binding()
            try:
                value=self.read_arm()
                fk(self.model,value['degrees_and_gripper_percent'][:5])
            except Exception as exc:
                self.feedback='反馈未通过：'+str(exc)
                raise
            self.feedback='v2 实时双臂反馈检查通过'
            self.arm_ready=True
            return {'message':self.feedback}
        if not Path(self.args.arm_port).exists():
            self.feedback='左臂 USB 串口未连接，请检查 USB 和供电后重试'
            raise ValueError(self.feedback)
        result=super().check_arm(); self.arm_ready=True
        return result

    def capture(self,fixed):
        if fixed is not True: raise ValueError('请确认主相机、底座和贴码位置已固定')
        if self.solving: raise ValueError('正在计算，请稍后采集')
        self.check_binding()
        # Three read-only measurements span at least one second before the frame.
        # A fresh frame is then bracketed by encoder reads; this is not exposure sync.
        reads=[]
        for i in range(3):
            reads.append(self.read_arm())
            if i<2: time.sleep(.5)
        deadline=time.monotonic()+2
        while True:
            packet,image=self.main_frame()
            stamp=packet['reference']['capture_ns']/1e9
            if stamp>=reads[-1]['read_completed']: break
            if time.monotonic()>deadline: raise ValueError('主相机没有产生新的画面，请检查相机服务')
            time.sleep(.04)
        reads.append(self.read_arm())
        qs=np.asarray([r['degrees_and_gripper_percent'][:5] for r in reads],float)
        if np.max(np.ptp(qs,axis=0))>.3: raise ValueError('机械臂还在移动，请停稳一秒后再保存')
        if not reads[-2]['read_completed']<=stamp<=reads[-1]['read_started']:
            raise ValueError('画面不在反馈读取区间，请停稳后重试')
        F=fk(self.model,qs.mean(axis=0).tolist())
        for sample in self.manifest['samples']:
            old=transform(sample['arms'][self.args.arm]['T_base_from_gripper'])
            angle=Rotation.from_matrix(old[:3,:3].T@F[:3,:3]).magnitude()
            if np.linalg.norm(old[:3,3]-F[:3,3])<.001 and angle<math.radians(1):
                raise ValueError('这个姿态已经保存，请换一个姿态')
        obs=observation(image,self.main_config,.029); obs['T_base_from_gripper']=F.tolist()
        ident=f'pose_{len(self.manifest["samples"])+1:04d}'
        sample={'id':ident,'pose_group':ident,'arms':{self.args.arm:obs}}
        directory=self.output/ident; directory.mkdir(exist_ok=False)
        exclusive(directory/'main.png',cv2.imencode('.png',image)[1].tobytes())
        exclusive(directory/'observation.json',encoded(sample))
        exclusive(directory/'provenance.json',encoded({'binding':self.binding,'main_packet':packet,
            'joint_reads':reads,'timing_semantics':'software receipt, not exposure synchronization'}))
        self.manifest['samples'].append(sample)
        tmp=self.output/'manifest.tmp'; tmp.write_bytes(encoded(self.manifest)); tmp.replace(self.manifest_path)
        self.report=None
        return {'message':f'已保存第 {len(self.manifest["samples"])} 个姿态，码重投影误差 {obs["pnp_rms_px"]:.2f} px'}

    def solve(self):
        if self.solving: raise ValueError('正在计算')
        if len(self.manifest['samples'])<6: raise ValueError('至少保存 6 个不同姿态，建议 10–15 个')
        self.check_binding(); data=json.loads(json.dumps(self.manifest)); self.solving=True; self.error=None
        def run():
            try:
                report=solve_manifest(data); name=f'result-{time.time_ns()}.json'
                exclusive(self.output/name,encoded(report)); self.report={'filename':name,'contents':report}
            except Exception as exc: self.error=str(exc)
            finally: self.solving=False
        threading.Thread(target=run,daemon=True).start()
        return {'message':'正在计算外参和留一姿态验证…'}

    def state(self):
        result={'samples':len(self.manifest['samples']),'feedback':self.feedback,'arm_ready':self.arm_ready,
                'solving':self.solving,'error':self.error,'report':self.report,'marker_ready':False}
        try:
            _,image=self.main_frame(); obs=observation(image,self.main_config,.029)
            result.update(marker_ready=True,pnp_rms_px=obs['pnp_rms_px'])
        except Exception as exc: result['camera_error']=str(exc)
        return result


def verification_page(dashboard,token):
    attached=bool(getattr(dashboard.args,'m3t_preview_packet',None))
    page=VERIFY_PAGE.replace('__TOKEN__',json.dumps(token)).replace('__ATTACHED__',json.dumps(attached))
    if attached:
        page=page.replace('<button id="start">开始只读叠加</button><button id="stop" disabled>停止</button>','')
        page=page.replace('<button id="m3tstart">启动 M3T 修正</button><button id="m3tstop" disabled>停止 M3T</button>','')
    if getattr(dashboard.args,'dual_arms',False):
        body,separator,script=page.partition('<script>')
        page=(body.replace('左臂','双臂')+separator+script).replace('五关节 + 夹爪实时软先验','双臂十二项实时软先验')
        start=page.index('<p><span class="cyan">青色：模型轮廓</span>')
        end=page.index('</p>',start)+4
        page=page[:start]+'<p><span class="cyan">青色：双臂编码器模型</span> · <span style="color:#ffa500">橙色：左臂 M3T</span> · <span style="color:#d65aeb">紫色：右臂 M3T</span></p>'+page[end:]
        page=page.replace('橙色 · M3T 修正','橙/紫 · 双臂 M3T').replace('青色/橙色轮廓','双臂轮廓')
    return page


def gripper_pose_response(overlay):
    """Export the latest fresh, image-associated gripper poses; never a calibrated contact TCP."""
    sample,state=overlay.snapshot()
    if sample is None or not state.get('ready') or state.get('error'):
        raise ValueError('末端位姿不可用：没有新鲜的有效跟踪画面')
    m3t=state.get('m3t') or {}
    receipt=sample['receipt'];estimate=receipt.get('m3t_estimate') or {}
    poses=receipt.get('tcp_poses')
    if (not m3t.get('running') or not receipt.get('m3t_enabled')
            or not isinstance(poses,dict) or not poses):
        raise ValueError('末端位姿不可用：M3T 未运行或位姿尚未输出')
    return {'schema':'rbt-m3t-gripper-poses-v1','coordinate_frame':estimate['frame_id'],
        'units':{'translation':'m','rotation':'rotation_matrix'},
        'matrix_layout':'row-major 16; T_A_from_B maps B coordinates to A',
        'target_link':'gripper_link','physical_contact_tcp_calibrated':False,
        'frame_sequence':estimate['frame_sequence'],
        'color_capture_ns':estimate['color_capture_ns'],
        'capture_clock_kind':estimate['capture_clock_kind'],
        'capture_clock_source':estimate['capture_clock_source'],
        'capture_clock_generation':estimate['capture_clock_generation'],
        'preview_age_seconds':state['age_seconds'],
        'timing_semantics':receipt['timing_semantics'],
        'pitch_deg':receipt['pitch_deg'],'pitch_source':receipt['pitch_source'],
        'joints_locked_to_feedback':estimate.get('mount_refinement',{}).get('joints_locked_to_feedback',False),
        'encoder_reference_links':estimate.get('encoder_reference_links') is True,
        'joint_priors':estimate.get('joint_priors',{}),'arms':poses}


def handler(dashboard,token):
    attached=bool(getattr(getattr(dashboard,'args',None),'m3t_preview_packet',None))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,status,kind,data):
            try:
                self.send_response(status); self.send_header('Content-Type',kind)
                self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError): self.close_connection=True
        def json(self,status,data): self.send(status,'application/json; charset=utf-8',encoded(data))
        def do_GET(self):
            try:
                path=urlsplit(self.path).path
                if path=='/': self.send(200,'text/html; charset=utf-8',PAGE.replace('__TOKEN__',json.dumps(token)).encode())
                elif path=='/api/state': self.json(200,dashboard.state())
                elif path=='/verify' and getattr(dashboard,'overlay',None):
                    self.send(200,'text/html; charset=utf-8',verification_page(dashboard,token).encode())
                elif path=='/api/overlay/state' and getattr(dashboard,'overlay',None):
                    self.json(200,dashboard.overlay.snapshot()[1])
                elif path=='/api/overlay/gripper-poses' and getattr(dashboard,'overlay',None):
                    self.json(200,gripper_pose_response(dashboard.overlay))
                elif path in ('/api/overlay/frame.jpg','/api/overlay/raw.jpg') and getattr(dashboard,'overlay',None):
                    sample,_=dashboard.overlay.snapshot()
                    if sample is None: raise ValueError('没有新鲜的静止叠加画面')
                    im=sample['raw' if path.endswith('raw.jpg') else 'overlay']
                    self.send(200,'image/jpeg',cv2.imencode('.jpg',im)[1].tobytes())
                elif path=='/frame.jpg':
                    _,image=dashboard.main_frame()
                    try:
                        corners=detect_marker(image).astype(np.float32).reshape(1,4,2)
                        cv2.aruco.drawDetectedMarkers(image,[corners],np.array([[1]],np.int32))
                    except ValueError: pass
                    self.send(200,'image/jpeg',cv2.imencode('.jpg',image)[1].tobytes())
                elif path=='/api/report' and dashboard.report: self.json(200,dashboard.report['contents'])
                else: self.json(404,{'error':'未找到内容'})
            except Exception as exc: self.json(503,{'error':str(exc)})
        def do_POST(self):
            try:
                expected=dashboard.args.public_origin
                if (self.headers.get('Origin')!=expected or self.headers.get('Host')!=urlsplit(expected).netloc
                    or not hmac.compare_digest(self.headers.get('X-CSRF-Token',''),token)):
                    self.json(403,{'error':'请求来源或校验令牌无效'}); return
                n=int(self.headers.get('Content-Length','0'))
                if self.headers.get_content_type()!='application/json' or not 0<n<=4096: raise ValueError('请求无效')
                data=json.loads(self.rfile.read(n))
                with dashboard.lock:
                    path=urlsplit(self.path).path
                    if path=='/api/check-arm': result=dashboard.check_arm()
                    elif path=='/api/capture': result=dashboard.capture(data.get('fixed'))
                    elif path=='/api/solve': result=dashboard.solve()
                    elif path=='/api/overlay/enable' and not attached and getattr(dashboard,'overlay',None): result=dashboard.overlay.set_enabled(data.get('enabled'))
                    elif path=='/api/overlay/layers' and getattr(dashboard,'overlay',None): result=dashboard.overlay.set_layers(data.get('layers'))
                    elif path=='/api/overlay/save' and getattr(dashboard,'overlay',None): result=dashboard.overlay.save()
                    elif path=='/api/overlay/m3t-start' and not attached and getattr(dashboard,'overlay',None) and dashboard.overlay.m3t: result=dashboard.overlay.m3t.start()
                    elif path=='/api/overlay/m3t-stop' and not attached and getattr(dashboard,'overlay',None) and dashboard.overlay.m3t: result=dashboard.overlay.m3t.stop()
                    else: raise ValueError('未知操作')
                self.json(200,result)
            except Exception as exc: self.json(400,{'error':str(exc)})
    return Handler


PAGE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>左臂 ArUco 标定 · RBT</title><style>
*{box-sizing:border-box}body{margin:0;background:#101720;color:#e6edf4;font:16px/1.65 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:28px}h1{font-size:28px;margin:0}h2{font-size:19px;margin:0 0 12px}p{color:#b5c4d3}.tag{color:#65d5bc;font-size:13px}.card{background:#19232f;border:1px solid #304052;border-radius:14px;padding:22px;margin-top:20px}.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:24px}img{width:100%;background:#080d13;border-radius:8px}button,a{display:inline-block;color:#08241e;background:#6ee0bd;border:0;border-radius:7px;padding:10px 16px;font:inherit;font-weight:600;cursor:pointer;text-decoration:none;margin:8px 8px 8px 0}button:disabled{opacity:.35;cursor:not-allowed}label{display:block}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}#notice{padding:12px 18px;background:#243c48;border-radius:8px;margin-top:20px}.warn{color:#f5ca7d}[hidden]{display:none!important}@media(max-width:800px){main{padding:14px}.grid{grid-template-columns:1fr}}
</style><main><span class="tag">RBT · 官方 SO101 + 软鳍夹爪 · v2</span><h1>左臂夹爪贴码标定</h1><p>固定主相机 → 观察夹爪 ArUco · ID 1 · 黑色外框 29 mm</p><a href="/verify">打开模型叠加验证 →</a><div id="notice" role="status">正在连接…</div><p id="mountsource"></p><section class="card grid"><div><h2>主相机实时画面</h2><img id="frame" alt="主相机识别左臂夹爪码的实时画面"><p id="camera">正在检查画面…</p></div><div><h2>01 / 准备</h2><p>移走背景 ChArUco 板，画面中只能有一个 ID 1。码固定在硬质夹爪壳体，不能随指头开合而移动。</p><button id="check">检查 v2 机械臂反馈</button><p id="feedback">尚未检查</p><label><input id="fixed" type="checkbox"> 主相机、底座和贴码位置已固定</label><h2 style="margin-top:24px">02 / 采集</h2><p>只移动机械臂，停稳一秒再保存。建议 10–15 个不同位置与倾角，包含绕不同轴转动，保持码完整可见。</p><button id="capture" disabled>保存当前姿态</button><p id="count">已采集 0 个姿态</p><p class="warn">网页只读取反馈，不发送机械臂运动指令。</p></div></section><section class="card"><h2>03 / 计算与验证</h2><button id="solve" disabled>计算标定</button><a id="download" href="/api/report" download="left-aruco-result.json" hidden>下载结果</a><p id="summary">至少采集 6 个不同姿态后可以计算。</p><pre id="report" hidden></pre><p>结果检查重投影误差、运动多样性和留一姿态预测。只标定左臂，结果不会自动替换 M3T 初始化。</p></section></main><script>
const token=__TOKEN__;let busy=false,refreshing=false,state={};const $=s=>document.querySelector(s);
function buttons(){$('#capture').disabled=busy||state.solving||!state.marker_ready||!state.arm_ready||!$('#fixed').checked;$('#solve').disabled=busy||state.solving||state.samples<6;$('#check').disabled=busy||state.solving}
async function action(name,body={}){if(busy)return;busy=true;buttons();$('#notice').textContent='处理中，请保持机械臂静止…';try{let r=await fetch('/api/'+name,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(body)});let d=await r.json();$('#notice').textContent=d.message||d.error}catch(e){$('#notice').textContent='连接中断，请先查看保存数量再决定是否重试。'}finally{busy=false;await refresh();buttons()}}
$('#check').onclick=()=>action('check-arm');$('#capture').onclick=()=>action('capture',{fixed:$('#fixed').checked});$('#solve').onclick=()=>action('solve');$('#fixed').onchange=buttons;
async function refresh(){if(refreshing)return;refreshing=true;try{const r=await fetch('/api/state');if(!r.ok)throw Error('服务暂时不可用');state=await r.json();$('#feedback').textContent=state.feedback;$('#camera').textContent=state.marker_ready?'ID 1 已识别 · 29 mm · 单帧误差 '+state.pnp_rms_px.toFixed(2)+' px':state.camera_error;$('#count').textContent='已采集 '+state.samples+' 个姿态';$('#frame').src='/frame.jpg?t='+Date.now();if(state.solving)$('#summary').textContent='正在计算和留一姿态验证…';else if(state.error)$('#summary').textContent=state.error;else if(state.report){const d=state.report.contents,a=d.arms.so101_left;$('#summary').textContent=(d.accepted?'验证通过':'尚未通过：'+a.assessment)+' · 训练误差 '+a.training.rms_px.toFixed(2)+' px · 留一姿态最大误差 '+(a.validation.max_prediction_frame_rms_px??0).toFixed(2)+' px';$('#report').hidden=false;$('#report').textContent=JSON.stringify(d,null,2);$('#download').hidden=false}else{$('#summary').textContent='至少采集 6 个不同姿态，建议 10–15 个。';$('#report').hidden=true;$('#download').hidden=true}buttons()}catch(e){state.marker_ready=false;$('#camera').textContent='连接失败：'+e;$('#frame').removeAttribute('src');buttons()}finally{refreshing=false}}
$('#notice').textContent='先检查 v2 反馈，确认固定后开始采集。';refresh();setInterval(refresh,1500);
</script></html>'''


VERIFY_PAGE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>左臂模型叠加验证 · RBT</title><style>
*{box-sizing:border-box}body{margin:0;background:#101720;color:#e6edf4;font:16px/1.65 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:28px;margin:0}.tag{color:#65d5bc;font-size:13px}p{color:#b5c4d3}.card{background:#19232f;border:1px solid #304052;border-radius:14px;padding:20px;margin-top:18px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:22px}img{width:100%;border-radius:8px;background:#080d13;min-height:200px}button,a{display:inline-block;color:#08241e;background:#6ee0bd;border:0;border-radius:7px;padding:10px 16px;font:inherit;cursor:pointer;text-decoration:none;margin:8px 8px 8px 0}button:disabled{opacity:.35;cursor:not-allowed}label{display:block;margin:8px 0}#notice{padding:12px 16px;background:#243c48;border-radius:8px;margin-top:16px}.cyan{color:#50e6ff}.pink{color:#ff70ff}.green{color:#70ff70}pre{white-space:pre-wrap;font-size:13px}.warn{color:#f5ca7d}.layers{display:flex;flex-wrap:wrap;gap:4px 18px;border:1px solid #304052;border-radius:8px;margin-top:12px;padding:8px 12px}.layers legend{color:#b5c4d3}.layers label{margin:3px 0;cursor:pointer}@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:12px}}
</style><main><span class="tag">RBT · 左臂 · 官方软鳍模型 + v2 · 反馈 / M3T 对比</span><h1>模型叠加验证</h1><p id="setup-instructions">第一步 / 云台保持 −61°，手动改变左臂姿态，停稳后观察模型与实物是否贴合。</p><div id="notice" role="status">正在连接…</div><p id="mountsource"></p><section class="card grid"><div><img id="frame" alt="左臂模型与实物实时叠加"><p><span class="cyan">青色：模型轮廓</span> · <span class="pink">粉色：预测码框</span> · <span class="green">绿色：实测码框</span> · <span style="color:#ffa500">橙色：M3T 修正</span></p><fieldset class="layers"><legend>显示图层</legend><label class="cyan"><input type="checkbox" id="layer-encoder" checked> 青色 · 关节反馈模型</label><label style="color:#ffa500"><input type="checkbox" id="layer-m3t" checked> 橙色 · M3T 修正</label><label class="pink"><input type="checkbox" id="layer-predicted" checked> 粉色 · 预测码框</label><label class="green"><input type="checkbox" id="layer-observed" checked> 绿色 · 实测码框</label></fieldset><p style="font-size:13px;margin:6px 0">取消勾选即可隐藏对应轮廓，跟踪继续运行。</p><label><input type="checkbox" id="show" checked> 显示叠加（取消查看原图）</label><p id="quality">等待反馈</p></div><div><h2>实时验证</h2><button id="start">开始只读叠加</button><button id="stop" disabled>停止</button><button id="save" disabled>保存验证画面</button><button id="m3tstart">启动 M3T 修正</button><button id="m3tstop" disabled>停止 M3T</button><label><input type="checkbox" id="m3t-tf-configured" disabled> RBT TF（配置启用）</label><p id="m3tstatus">M3T 未开启</p><p id="status">尚未开启</p><p id="pitch">云台要求：−61° ±0.5°</p><p id="age"></p><pre id="joints"></pre><p>末端参考：各臂 gripper_link（接触 TCP 未标定）</p><pre id="gripper-poses">等待末端位姿</pre><p id="observation-instructions">选几个未参与标定的姿态，分别看底座、肘部、手腕和固定软鳍。码暂时不可见时，仍能看模型轮廓。</p><p id="tracking-instructions" class="warn">机械臂移动中会暂停叠加，停稳后恢复。此页面不会控制机械臂或云台。M3T 使用实时 v2 五关节与夹爪开度软先验配合图像优化；大幅换姿态或丢失后，停止并按当前反馈重新启动。</p><p>两边夹指和软鳍均显示。青色夹爪按 v2 实测编码器行程换算，不按 URDF 上限缩放；闭合端点暂作模型零位，对齐尚未独立验证；橙色使用 M3T 实际 link 位姿。模型轮廓允许透视内部遮挡，不能用遮挡线判断贴合。当前外参是用户选用的实验初值；−61°采集角度是换算假设。</p><a href="/">返回标定采集</a></div></section></main><script>
const token=__TOKEN__,attached=__ATTACHED__,$=s=>document.querySelector(s);let busy=false,polling=false,last={},mountMode=null,frameLoading=false,frameLoadedAt=0,frameStale=true;
const layerNames=['encoder','m3t','predicted','observed'];
function updateFrameAge(){if(!frameLoadedAt){$('#age').textContent='暂无成功画面';return}const shown=((Date.now()-frameLoadedAt)/1000).toFixed(1);$('#age').textContent=frameStale?'上一张成功画面可能已过期 · 页面已显示 '+shown+' 秒':'画面处理后 '+Number(last.age_seconds||0).toFixed(1)+' 秒 · 页面已显示 '+shown+' 秒'}
function retainFrame(){frameStale=true;updateFrameAge()}
function requestFrame(path){if(frameLoading)return;frameLoading=true;const visible=$('#frame'),candidate=new Image();candidate.onload=()=>{frameLoading=false;candidate.id='frame';candidate.alt=visible.alt;visible.replaceWith(candidate);frameLoadedAt=Date.now();frameStale=!last.ready||path!==($('#show').checked?'/api/overlay/frame.jpg':'/api/overlay/raw.jpg');updateFrameAge()};candidate.onerror=()=>{frameLoading=false;retainFrame()};candidate.src=path+'?t='+Date.now()}
function lockLayers(value){for(const name of layerNames)$('#layer-'+name).disabled=value||(last.marker_available===false&&(name==='predicted'||name==='observed'))}
function feedbackText(d){if(attached&&!d.m3t_enabled)return '原图画面无配对关节反馈';const names=['肩转','肩抬','肘部','腕弯','腕转'];function line(v,g){return names.map((n,i)=>n+' '+v[i].toFixed(1)+'°').join('\n')+'\n夹爪开度 '+g.toFixed(1)+'%'}if(d.arms)return Object.entries(d.arms).map(([arm,r])=>(arm==='so101_left'?'左臂':'右臂')+'\n'+line(r.degrees,r.gripper_percent)).join('\n\n');return line(d.degrees,d.joint_after.degrees_and_gripper_percent[5])}
function mountCorrectionText(estimate){const report=estimate?.mount_refinement;if(report?.enabled!==true||!Array.isArray(report.arms))return '等待底座修正诊断';return report.arms.map(a=>(a.arm==='so101_left'?'左':'右')+' 底座修正 '+(a.translation_correction_m*1000).toFixed(1)+' mm / '+(a.rotation_correction_rad*180/Math.PI).toFixed(1)+'°').join(' · ')}
function trackingModeText(m){const joints=m?.joints_locked_to_feedback===true?'编码器关节固定':m?.joint_priors_enabled===true?'五关节 + 夹爪实时软先验':'无实时先验';const base=m?.mount_refinement?.enabled===true?'CAD 底座软约束':m?.fixed_base===true?'底座固定':'等待底座状态';return base+' · '+joints}
function pitchText(value){return String(value).replace('-', '−')+'°'}
function showMount(s){
if(attached){
 const assumed=s.pitch_source==='assumed_fixed';
 $('#mountsource').textContent='M3T · '+trackingModeText(s.m3t);
 $('#setup-instructions').textContent=assumed?'云台假设固定 '+pitchText(s.expected_pitch_deg):'原生跟踪预览';
 $('#observation-instructions').textContent='青色为同一 M3T 帧应用的编码器先验，使用 CAD 名义安装位姿。';
 $('#tracking-instructions').textContent=s.m3t?.mount_refinement?.enabled===true?'底座在 CAD 参考附近以软约束修正，关节在实时编码器软先验附近由图像继续优化。页面只读订阅原生 M3T。':'页面只读订阅原生 M3T。预览缺失或过期时显示相机原图。';
 if(mountMode===null){$('#notice').textContent='已订阅原生 M3T 预览';mountMode=true}
 lockLayers(busy);return;
}
const nominal=s.mount_source?.nominal===true,assumed=s.pitch_source==='assumed_fixed',refining=s.m3t?.mount_refinement?.enabled===true;$('#mountsource').textContent=nominal?'CAD粗初值 · '+(assumed?'假设固定':'反馈守卫 ')+pitchText(s.expected_pitch_deg)+' · '+trackingModeText(s.m3t)+' · 求解未验收':'';if(assumed)$('#pitch').textContent='云台假设固定：'+pitchText(s.expected_pitch_deg)+'（无实时反馈）';if(nominal){$('#setup-instructions').textContent='固定 −61° 相机 · CAD 名义安装初值 + 实时编码器；当前用于 M3T 初始化';$('#observation-instructions').textContent='先观察当前静止姿态的青色/橙色轮廓；本轮不要求重新标定或多姿态测试。';$('#quality').textContent=refining?'等待底座修正诊断':'粗初值未标定贴码外参'}if(refining)$('#tracking-instructions').textContent=s.m3t?.joints_locked_to_feedback===true?'此模式将关节固定为实时 v2 反馈，只优化 CAD 参考附近的底座位姿；页面只读，不控制机械臂或云台。停稳后观察；大幅换姿态或丢失后，按当前反馈重新启动。':'底座在 CAD 参考附近以软约束修正，关节在实时编码器软先验附近由图像继续优化。页面只读，不控制机械臂或云台。';if(mountMode!==nominal){$('#notice').textContent=nominal?'只读关节与图像；云台采用显式固定角度假设':'点击开始，只读取关节和云台反馈。';mountMode=nominal}lockLayers(busy)}
for(const name of layerNames)$('#layer-'+name).onchange=()=>action('layers',{layers:Object.fromEntries(layerNames.map(n=>[n,$('#layer-'+n).checked]))});
async function action(name,body={}){if(busy)return;busy=true;lockLayers(true);try{let r=await fetch('/api/overlay/'+name,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(body)});let d=await r.json();$('#notice').textContent=d.message||d.error}catch(e){$('#notice').textContent='操作连接中断，请先查看状态。'}finally{busy=false;lockLayers(false);refresh()}}
if(!attached){$('#start').onclick=()=>action('enable',{enabled:true});$('#stop').onclick=()=>action('enable',{enabled:false});$('#m3tstart').onclick=()=>action('m3t-start');$('#m3tstop').onclick=()=>action('m3t-stop')}
$('#save').onclick=()=>action('save');$('#show').onchange=()=>{if(last.ready)requestFrame($('#show').checked?'/api/overlay/frame.jpg':'/api/overlay/raw.jpg')};
async function refresh(){if(polling)return;polling=true;try{let r=await fetch('/api/overlay/state');if(!r.ok)throw Error('服务连接失败');last=await r.json();showMount(last);if(!busy&&last.layers)for(const name of layerNames)$('#layer-'+name).checked=last.layers[name];$('#save').disabled=busy||!last.ready;
let m=last.m3t;$('#m3t-tf-configured').checked=m?.native_tf_enabled===true;
if(!attached){
 $('#start').disabled=busy||last.enabled;$('#stop').disabled=busy||!last.enabled;
 $('#m3tstart').disabled=busy||!m||m.running;$('#m3tstop').disabled=busy||!m||!m.enabled;
}
$('#m3tstatus').textContent=attached?(m?.running?'M3T Region + Depth · 原生实时预览 · '+trackingModeText(m):(m?.packet_error||'等待原生 M3T 预览')):!m||!m.enabled?'M3T 未开启':(m.running?(m.packet_ready?'M3T Region + Depth · '+trackingModeText(m):'M3T 正在加载 / 生成模型缓存…'):'M3T 已退出，代码 '+m.exit_code);if(m?.mount_refinement?.enabled===true&&m.end_effector_priority===true)$('#m3tstatus').textContent+=' · 夹爪刚性结构优先';$('#status').textContent=last.ready?(attached?(last.receipt.m3t_enabled?'原生跟踪画面':'相机原图'):'静止画面 · 叠加已更新'):(last.error||(last.enabled?'正在读取…':'尚未开启'));if(last.ready){let d=last.receipt;requestFrame($('#show').checked?'/api/overlay/frame.jpg':'/api/overlay/raw.jpg');$('#pitch').textContent=(d.pitch_source==='assumed_fixed'?'云台假设固定：':'云台反馈：')+d.pitch_deg.toFixed(2)+'°'+(d.pitch_source==='assumed_fixed'?'（无实时反馈）':'');let jp=d.m3t_estimate?.joint_priors;if(jp)$('#m3tstatus').textContent+=' · 已应用 '+jp.prior_count+'/'+(m.joint_priors_count||6)+' · '+(jp.age_ns/1e6).toFixed(1)+' ms';updateFrameAge();$('#joints').textContent=feedbackText(d);$('#gripper-poses').textContent=d.tcp_poses?Object.entries(d.tcp_poses).map(([arm,p])=>{let t=p.T_camera_from_gripper_link_m3t;return (arm==='so101_left'?'左臂':'右臂')+' 相机系 XYZ (mm): '+[t[3],t[7],t[11]].map(v=>(v*1000).toFixed(1)).join(', ')}).join('\n'):'等待末端位姿';$('#quality').textContent=m?.mount_refinement?.enabled===true?mountCorrectionText(d.m3t_estimate):d.mount_source?.nominal?'粗初值未标定贴码外参':d.marker_rms_px===null?'码不可测：'+d.marker_error:'当前预测码框误差：'+d.marker_rms_px.toFixed(2)+' px'+(d.m3t_enabled?' → M3T '+(d.m3t_marker_rms_px===null?'不可测':d.m3t_marker_rms_px.toFixed(2)+' px'):'（未做视觉修正）')}else{retainFrame();$('#quality').textContent=m?.mount_refinement?.enabled===true?'等待底座修正诊断':last.mount_source?.nominal?'粗初值未标定贴码外参':'等待新鲜、静止的反馈和图像';$('#joints').textContent='';$('#gripper-poses').textContent='末端位姿不可用'}}catch(e){$('#status').textContent=String(e);retainFrame();$('#save').disabled=true;$('#gripper-poses').textContent='末端位姿不可用'}finally{polling=false}}
$('#notice').textContent=attached?'正在连接原生 M3T 预览…':'点击开始，只读取关节和图像。';refresh();setInterval(refresh,700);
</script></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('main-session','main-packet','robot-model','calibration','output','driver-python','driver-path','arm-port'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--arm',choices=['so101_left','so101_right'],default='so101_left')
    parser.add_argument('--dual-arms',action='store_true')
    parser.add_argument('--right-calibration')
    parser.add_argument('--right-arm-port',default='/dev/so101_right')
    parser.add_argument('--port',type=int,default=8098)
    parser.add_argument('--joint-feedback-controller',default='so101-v2-readonly-m3t')
    parser.add_argument('--m3t-prior-service',help='RBT 双臂状态流，网页和 M3T 共用')
    parser.add_argument('--m3t-mount-refinement',action='store_true')
    parser.add_argument('--m3t-end-effector-priority',action='store_true')
    parser.add_argument('--m3t-correspondence-iterations',type=int,choices=range(1,21),default=3)
    parser.add_argument('--m3t-modeled-occlusions',action='store_true')
    parser.add_argument('--m3t-publish-tf',action='store_true',
        help='configure native RBT TF output or report its configuration in attached mode')
    parser.add_argument('--m3t-mount-translation-std-m',type=float,default=.02)
    parser.add_argument('--m3t-mount-rotation-std-deg',type=float,default=5.)
    parser.add_argument('--m3t-command'); parser.add_argument('--m3t-run-root'); parser.add_argument('--m3t-cache-directory')
    parser.add_argument('--m3t-preview-packet',type=Path,
        help='subscribe to an independent native M3T packet (requires --dual-arms)')
    parser.add_argument('--overlay-audit'); parser.add_argument('--overlay-report')
    parser.add_argument('--overlay-static-config')
    parser.add_argument('--nominal-mount-prior')
    parser.add_argument('--fixed-pitch-assumption-deg',type=float)
    parser.add_argument('--overlay-pitch-status-url',default='http://192.168.1.232:8784/api/status')
    parser.add_argument('--public-origin',default='http://127.0.0.1:18898')
    args=parser.parse_args(); u=urlsplit(args.public_origin)
    if u.scheme!='http' or u.hostname not in ('127.0.0.1','localhost') or u.path or u.query or u.fragment or u.username:
        raise ValueError('public origin 仅允许本机 HTTP 地址')
    from m3t_preview import mount_refinement_config
    try:mount_refinement_config(args.m3t_mount_translation_std_m,args.m3t_mount_rotation_std_deg)
    except ValueError as exc:parser.error(str(exc))
    if args.m3t_end_effector_priority and not args.m3t_mount_refinement:
        parser.error('--m3t-end-effector-priority 需要 --m3t-mount-refinement')
    if args.m3t_mount_refinement and (not args.nominal_mount_prior
            or args.fixed_pitch_assumption_deg is None or not math.isfinite(args.fixed_pitch_assumption_deg)
            or not args.m3t_prior_service):
        parser.error('--m3t-mount-refinement 需要 --nominal-mount-prior、显式 --fixed-pitch-assumption-deg 与 --m3t-prior-service')
    overlay_paths=(args.overlay_audit,args.overlay_report,args.overlay_static_config)
    if args.fixed_pitch_assumption_deg is not None and not args.nominal_mount_prior:
        parser.error('--fixed-pitch-assumption-deg 仅允许与 --nominal-mount-prior 一起使用')
    if bool(args.overlay_audit)!=bool(args.overlay_report):
        parser.error('legacy overlay audit/report 必须成对提供')
    if args.nominal_mount_prior and not args.overlay_static_config:
        parser.error('nominal prior 需要绑定 static-config')
    if not args.nominal_mount_prior and any(overlay_paths) and not all(overlay_paths):
        parser.error('legacy 叠加验证需要完整 audit/report/static-config')
    if args.arm=='so101_right' and (not args.m3t_prior_service or not args.nominal_mount_prior):
        parser.error('右臂需要显式 nominal prior 与 RBT 实时反馈')
    if args.dual_arms and (args.arm!='so101_left' or not args.right_calibration
            or not args.m3t_prior_service or not args.nominal_mount_prior):
        parser.error('双臂需要 left 入口、right-calibration、nominal prior 与 RBT 实时反馈')
    if args.dual_arms and Path(args.right_arm_port).resolve()==Path(args.arm_port).resolve():
        parser.error('左右臂串口必须不同')
    if args.m3t_preview_packet:
        if not args.dual_arms:
            parser.error('--m3t-preview-packet 需要 --dual-arms')
        if any((args.m3t_command,args.m3t_run_root,args.m3t_cache_directory,
                args.m3t_mount_refinement,args.m3t_end_effector_priority)):
            parser.error('--m3t-preview-packet 不能与网页跟踪启动选项同时使用')
    dashboard=Dashboard(args)
    if args.dual_arms:
        right_args=copy.copy(args)
        right_args.arm='so101_right';right_args.calibration=args.right_calibration
        right_args.arm_port=args.right_arm_port;right_args.output=str(Path(args.output)/'right-source')
        right_dashboard=Dashboard(right_args)
        from dual_overlay_preview import BimanualOverlayPreview
        dashboard.overlay=BimanualOverlayPreview(dashboard,right_dashboard,args.overlay_static_config,
            args.overlay_pitch_status_url,args.nominal_mount_prior,args.fixed_pitch_assumption_deg)
    elif any(overlay_paths) or args.nominal_mount_prior:
        from overlay_preview import OverlayPreview
        dashboard.overlay=OverlayPreview(dashboard,*overlay_paths,args.overlay_pitch_status_url,
            nominal_mount_prior=args.nominal_mount_prior,
            fixed_pitch_assumption_deg=args.fixed_pitch_assumption_deg)
    if getattr(dashboard,'overlay',None):
        if args.m3t_preview_packet:
            from m3t_preview import M3TPacketSubscriber
            dashboard.overlay.m3t=M3TPacketSubscriber(dashboard.overlay,args.m3t_preview_packet,
                prior_service=args.m3t_prior_service,prior_controller=args.joint_feedback_controller,
                publish_tf=args.m3t_publish_tf)
            dashboard.overlay.set_enabled(True)
        elif args.m3t_command:
            if not args.m3t_run_root or not args.m3t_cache_directory: raise ValueError('M3T 需要独立运行和缓存目录')
            from m3t_preview import M3TExperiment
            dashboard.overlay.m3t=M3TExperiment(dashboard.overlay,args.m3t_command,args.m3t_run_root,args.m3t_cache_directory,
                prior_service=args.m3t_prior_service,prior_controller=args.joint_feedback_controller,
                mount_refinement=args.m3t_mount_refinement,
                mount_translation_std_m=args.m3t_mount_translation_std_m,
                mount_rotation_std_deg=args.m3t_mount_rotation_std_deg,
                end_effector_priority=args.m3t_end_effector_priority,
                correspondence_iterations=args.m3t_correspondence_iterations,
                modeled_occlusions=args.m3t_modeled_occlusions,
                publish_tf=args.m3t_publish_tf)
    server=Server(('127.0.0.1',args.port),handler(dashboard,secrets.token_urlsafe(32)))
    print('ArUco dashboard ready on loopback port',args.port,flush=True)
    def shutdown(_signum,_frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,shutdown)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        if not args.m3t_preview_packet and getattr(dashboard,'overlay',None) and dashboard.overlay.m3t:
            dashboard.overlay.m3t.stop()
        server.server_close()

if __name__=='__main__': main()
