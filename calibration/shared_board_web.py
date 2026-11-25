#!/usr/bin/env python3
"""Local-only SO101 shared-board operator dashboard. Never writes motor goals."""
from __future__ import annotations
import argparse
import base64
import hashlib
import hmac
import http.client
import json
import math
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from wrist_intrinsics import DEFAULT_BOARD, CharucoDetector
from shared_board import solve_manifest

MODEL_SHA = '3b2b8cf4e0432659d4a7082528d972479e2bdcc81849e9036bf417626f0decb6'
NAMES = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')

def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exclusive(path, data):
    with Path(path).open('xb') as out:
        out.write(data)

def transform(value):
    a=np.asarray(value,dtype=float).reshape(4,4)
    if not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1],atol=1e-8) or not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-6) or abs(np.linalg.det(a[:3,:3])-1)>1e-6:
        raise ValueError('无效的刚体变换')
    return a

def fk(model, degrees):
    if len(degrees)<5 or not np.isfinite(degrees[:5]).all():
        raise ValueError('关节反馈不完整')
    bychild={j['child']:j for j in model['joints']}
    chain=[]; link='gripper_link'
    while link!=model['root_link']:
        if len(chain)>len(bychild) or link not in bychild: raise ValueError('模型链不完整')
        joint=bychild[link]; chain.append(joint); link=joint['parent']
    answer=np.eye(4)
    seen=[]
    for j in reversed(chain):
        answer=answer@transform(j['origin'])
        if j['type']=='fixed': continue
        if j['type']!='revolute' or j['name'] not in NAMES: raise ValueError('模型关节不支持')
        seen.append(j['name']); q=math.radians(float(degrees[NAMES.index(j['name'])]))
        if not j['lower']-1e-6<=q<=j['upper']+1e-6:
            raise ValueError(f"关节超出当前模型范围：{j['name']}，反馈 {math.degrees(q):.2f}°，允许 {math.degrees(j['lower']):.1f}°～{math.degrees(j['upper']):.1f}°")
        axis=np.asarray(j['axis'],dtype=float)
        if not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-6: raise ValueError('模型轴无效')
        r=np.eye(4); r[:3,:3]=Rotation.from_rotvec(axis*q).as_matrix(); answer=answer@r
    if set(seen)!=set(NAMES): raise ValueError('模型缺少五个机械臂关节')
    return answer

def parse_packet(payload, session, now_ns):
    if session.get('schema')!='rbt-stereo-capture-v1': raise ValueError('主相机 session schema 不支持')
    line, sep, raw=payload.partition(b'\n')
    if not sep or len(line)>8192: raise ValueError('相机预览包头无效')
    packet=json.loads(line)
    if packet.get('schema')!='rbt-stereo-preview-v1': raise ValueError('相机预览 schema 不支持')
    sizes=[]
    for role in ('reference','secondary'):
        c=session[role]; stamp=packet[role]
        w,h=c['width'],c['height']
        if type(w) is not int or type(h) is not int or not 0<w<=8192 or not 0<h<=8192: raise ValueError('相机尺寸无效')
        sizes.append(w*h*3)
        if stamp['clock_kind']!='HostMonotonic' or type(stamp['capture_ns']) is not int: raise ValueError('相机时钟不支持')
        clock=bytes.fromhex(stamp['clock_source_hex'])
        if len(clock)!=48 or not any(clock): raise ValueError('相机时钟来源无效')
        if type(stamp['clock_generation']) is not int or stamp['clock_generation']<0: raise ValueError('相机时钟代数无效')
        if not 0<=now_ns-stamp['capture_ns']<=1_500_000_000: raise ValueError('主相机预览已过期')
    a,b=packet['reference'],packet['secondary']
    if (a['clock_source_hex'],a['clock_generation'])!=(b['clock_source_hex'],b['clock_generation']): raise ValueError('相机时钟来源不一致')
    skew=abs(a['capture_ns']-b['capture_ns'])
    if packet['skew_ns']!=skew or skew>min(20_000_000,session['max_skew_ns']): raise ValueError('主相机预览配对时间异常')
    if len(raw)!=sum(sizes): raise ValueError('相机预览字节数不符')
    c=session['reference']
    image=np.frombuffer(raw[:sizes[0]],np.uint8).reshape(c['height'],c['width'],3)[:,:,::-1].copy()
    return packet,image

def main_intrinsics(session):
    c=session['reference']
    if c['serial']!='CP15641000AT' or c['distortion_order']!='k1,k2,k3,k4,k5,k6,p1,p2': raise ValueError('主相机身份或畸变定义不支持')
    d=np.asarray(c['distortion'],float)
    # Only the verified rectified stream is supported; do not guess a nonzero SDK model.
    if d.shape!=(8,) or not np.isfinite(d).all() or np.any(d!=0): raise ValueError('主相机应为已验证的无畸变图像')
    k=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1]],float)
    if not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0: raise ValueError('主相机内参无效')
    return {'K':k.tolist(),'distortion':[0.]*8,'image_size':[c['width'],c['height']],'camera_serial':c['serial']}

def pose_observation(config, ids, pixels, lattice):
    ids=np.asarray(ids,dtype=int).reshape(-1); pixels=np.asarray(pixels,float).reshape(-1,2)
    if len(ids)<6 or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=len(lattice)) or len(ids)!=len(pixels): raise ValueError('两台相机都至少需要 6 个有效角点')
    objects=np.asarray(lattice,float)[ids]
    w,h=config['image_size']
    if not np.isfinite(pixels).all() or np.any(pixels<0) or np.any(pixels[:,0]>=w) or np.any(pixels[:,1]>=h) or np.linalg.matrix_rank(objects[:,:2]-objects[:,:2].mean(0))<2: raise ValueError('角点越界或退化')
    k=np.asarray(config['K'],float); d=np.asarray(config['distortion'],float)
    ok,r,t=cv2.solvePnP(objects,pixels,k,d,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise ValueError('板子位姿求解失败')
    rot=cv2.Rodrigues(r)[0]; points=objects@rot.T+t.reshape(3)
    projected=cv2.projectPoints(objects,r,t,k,d)[0].reshape(-1,2)
    rms=float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1))))
    if np.any(points[:,2]<=0) or not math.isfinite(rms) or rms>2: raise ValueError('板子位姿质量不足，重投影误差应小于 2 px')
    T=np.eye(4); T[:3,:3]=rot; T[:3,3]=t.reshape(3)
    return {**{key:config[key] for key in ('K','distortion','image_size','camera_serial')},'object_points':objects.tolist(),'image_points':pixels.tolist(),'corner_ids':ids.tolist(),'T_camera_from_board_seed':T.tolist(),'pnp_rms_px':rms}

READ_SCRIPT=r'''
import sys,json,time,subprocess
sys.path.insert(0,sys.argv[1])
from so101_rbt.backend import FeetechSTS3215Backend
from so101_rbt.calibration import load_arm_calibration,MOTOR_NAMES
port,path=sys.argv[2:4]
r=subprocess.run(['fuser',port],capture_output=True)
if r.returncode!=1: raise RuntimeError('机械臂串口已被其他程序占用，或无法验证占用情况')
cal=load_arm_calibration(path)
b=FeetechSTS3215Backend(port,cal,allow_goal_writes=False,write_timeout_s=.2,num_read_retries=2)
try:
 for name in MOTOR_NAMES:
  c=cal.motors[name]
  for reg,expected in [('Homing_Offset',c.homing_offset),('Min_Position_Limit',c.range_min),('Max_Position_Limit',c.range_max)]:
   if b._read_register(c.motor_id,reg)!=expected: raise RuntimeError('硬件电机标定与 v2 文件不一致：'+name+'/'+reg)
 start=time.monotonic(); raw=b.read_raw(); end=time.monotonic()
 print(json.dumps({'degrees_and_gripper_percent':cal.normalize_raw(raw),'raw':raw,'read_started':start,'read_completed':end}))
finally: b.close()
'''

class Dashboard:
    def __init__(self,args):
        self.args=args; self.output=Path(args.output); self.output.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock(); self.detector_lock=threading.Lock(); self.detector=CharucoDetector(DEFAULT_BOARD)
        self.session=json.loads(Path(args.main_session).read_text()); self.main_config=main_intrinsics(self.session)
        self.model=json.loads(Path(args.robot_model).read_text())
        if self.model.get('source_sha256')!=MODEL_SHA or self.model.get('root_link')!='base_link': raise ValueError('请使用已确认的官方 SO101 软鳍版模型')
        self.hashes={k:sha(getattr(args,k)) for k in ('main_session','robot_model','calibration')}
        self.manifest_path=self.output/'manifest.json'; self.manifest={'schema_version':1,'setup_id':self.output.name,'setup_confirmed_unchanged':True,'board':DEFAULT_BOARD,'arms':{args.arm:{'samples':[]}}}
        self.adopted=None; self.report=None; self.solving=False; self.error=None; self.feedback='尚未检查'
        self.binding={'schema_version':1,'hashes':self.hashes,'arm':args.arm,'port':args.arm_port,'board':DEFAULT_BOARD}
        binding=self.output/'binding.json'
        if binding.exists():
            if json.loads(binding.read_text())!=self.binding: raise ValueError('既有手眼工作阶段的模型、v2 或相机设置不同，请新建目录')
        else:
            if any(self.output.iterdir()): raise ValueError('输出目录并非本工具拥有的空白目录')
            exclusive(binding,encoded(self.binding))
        if self.manifest_path.exists():
            self.manifest=json.loads(self.manifest_path.read_text())
        adoption=self.output/'adopted-intrinsics.json'
        if adoption.exists(): self.adopted=json.loads(adoption.read_text())

    def child(self,path):
        connection=http.client.HTTPConnection('127.0.0.1',self.args.wrist_port,timeout=8)
        try:
            connection.request('GET',path); r=connection.getresponse(); data=r.read()
            if r.status!=200:
                if path=='/api/candidate': raise ValueError('还没有通过检查的内参，请先保存至少 12 个板姿态并计算内参。')
                raise ValueError(json.loads(data).get('error','夹爪相机服务尚未准备好'))
            return json.loads(data)
        finally: connection.close()

    def check_binding(self):
        if any(sha(getattr(self.args,k))!=v for k,v in self.hashes.items()): raise ValueError('模型、v2 或主相机设置已变化，请新建工作阶段')

    def main_frame(self):
        self.check_binding()
        return parse_packet(Path(self.args.main_packet).read_bytes(),self.session,time.monotonic_ns())

    def detect(self, image):
        with self.detector_lock:
            return self.detector.detect(image)

    def read_arm(self):
        try:
            result=subprocess.run([self.args.driver_python,'-c',READ_SCRIPT,self.args.driver_path,self.args.arm_port,self.args.calibration],capture_output=True,text=True,timeout=12)
        except subprocess.TimeoutExpired as exc:
            raise ValueError('机械臂读取超时，请检查供电、USB 和串口连接。') from exc
        if result.returncode:
            if hasattr(self,'output'):
                exclusive(self.output/f'feedback-error-{time.time_ns()}.log',result.stderr.encode())
            if 'comm=' in result.stderr:
                raise ValueError('机械臂电机未应答，请检查机械臂供电和 USB，恢复后再点“检查 v2 机械臂反馈”。')
            detail=result.stderr.strip().splitlines()[-1] if result.stderr.strip() else '未知读取错误'
            raise ValueError('机械臂反馈读取失败：'+detail[:250])
        return json.loads(result.stdout)

    def check_arm(self):
        self.check_binding()
        try:
            value=self.read_arm()
            fk(self.model,value['degrees_and_gripper_percent'])
            self.feedback='v2 寄存器与五关节反馈检查通过'
            return {'message':self.feedback}
        except Exception as exc:
            self.feedback='反馈未通过：'+str(exc)
            raise

    def verify_wrist(self,obs):
        if self.adopted is None: raise ValueError('请先计算并采用夹爪相机内参')
        c=self.adopted['contents']; f=obs['focus']
        if obs['board']!=DEFAULT_BOARD or obs['board_geometry_sha256']!=c['board_geometry_sha256']: raise ValueError('标定板参数变化')
        if not f['locked'] or f['autofocus']!=0 or abs(f['focus_value']-c['focus_value'])>1e-3: raise ValueError('夹爪相机焦距变化，请重新标定')
        # /dev/videoN is a receipt-time detail, not a persistent camera identity.
        for key in ('device_uri','camera_serial'):
            if obs['device_binding'][key]!=c[key]: raise ValueError('夹爪相机来源变化')
        if [obs['width'],obs['height']]!=c['image_size']: raise ValueError('夹爪相机分辨率变化')
        age=time.monotonic()-obs['read_completed_monotonic_seconds']
        if not 0<=age<=1: raise ValueError('夹爪相机画面已过期')
        return c['solver_camera_config']

    def adopt(self):
        if self.adopted is not None: raise ValueError('本阶段已固定内参；更换时请新建阶段')
        candidate=self.child('/api/candidate'); c=candidate['contents']
        if c.get('quality_gates_passed') is not True or c.get('status')!='CANDIDATE_INTRINSICS_REQUIRES_INDEPENDENT_VERIFICATION': raise ValueError('内参质量检查尚未通过')
        if c['board']!=DEFAULT_BOARD: raise ValueError('标定板参数不匹配')
        self.adopted=candidate
        try: self.verify_wrist(self.child('/api/observation'))
        except Exception:
            self.adopted=None; raise
        exclusive(self.output/'adopted-intrinsics.json',encoded(candidate))
        return {'message':'已固定这份内参。现在固定板子、主相机和机械臂底座，再开始移动机械臂采集。'}

    def capture(self,fixed):
        if fixed is not True: raise ValueError('请确认标定板、主相机、云台和底座已固定')
        if self.solving: raise ValueError('正在求解，请稍后采集')
        self.check_binding()
        if self.adopted is None: raise ValueError('请先计算并采用夹爪相机内参')
        before=self.read_arm()
        # Obtain wrist first: full-resolution PNG encoding is part of this receipt delay.
        obs=self.child('/api/observation')
        if obs['read_completed_monotonic_seconds'] < before['read_completed']:
            obs=self.child('/api/observation')
        config=self.verify_wrist(obs)
        packet,image=self.main_frame()
        stamp=packet['reference']['capture_ns']/1e9; wrist_stamp=obs['read_completed_monotonic_seconds']
        if abs(stamp-wrist_stamp)>.250: raise ValueError('两相机软件接收时间差超过 250 ms，请重试')
        after=self.read_arm()
        a=np.asarray(before['degrees_and_gripper_percent'][:5],float); b=np.asarray(after['degrees_and_gripper_percent'][:5],float)
        if np.max(np.abs(a-b))>.3: raise ValueError('机械臂仍在移动，请停稳后再保存')
        if not before['read_completed']<=wrist_stamp<=after['read_started']: raise ValueError('夹爪帧不在关节读取前后区间，请重试')
        if not before['read_completed']-.25<=stamp<=after['read_started']+.25: raise ValueError('主相机帧不在关节读取区间，请重试')
        F=fk(self.model,((a+b)/2).tolist())
        samples=self.manifest['arms'][self.args.arm]['samples']
        for sample in samples:
            old=transform(sample['T_base_from_gripper']); angle=Rotation.from_matrix(old[:3,:3].T@F[:3,:3]).magnitude()
            if np.linalg.norm(old[:3,3]-F[:3,3])<.001 and angle<math.radians(1): raise ValueError('这个机械臂姿态已采集，请换一个姿态')
        _,ids,pixels=self.detect(image)
        main=pose_observation(self.main_config,ids,pixels,self.detector.board_corners)
        wrist=pose_observation(config,obs['corner_ids'],obs['corners_px'],self.detector.board_corners)
        raw_wrist=base64.b64decode(obs['raw_image_png_base64'],validate=True)
        decoded=cv2.imdecode(np.frombuffer(raw_wrist,np.uint8),cv2.IMREAD_COLOR)
        if decoded is None or [decoded.shape[1],decoded.shape[0]]!=config['image_size']: raise ValueError('夹爪原始 PNG 无效')
        ident=f'pose_{len(samples)+1:04d}'; directory=self.output/ident; directory.mkdir(exist_ok=False)
        sample={'id':ident,'pose_group':ident,'T_base_from_gripper':F.tolist(),'main':main,'wrist':wrist}
        provenance={'software_receipt_skew_ms':abs(stamp-wrist_stamp)*1000,'timing_semantics':'software receipt, not exposure synchronization','main_packet':packet,'wrist_frame_id':obs['frame_id'],'wrist_read_started':obs['read_started_monotonic_seconds'],'wrist_read_completed':wrist_stamp,'joint_before':before,'joint_after':after,'hashes':self.hashes,'intrinsic_sha256':self.adopted['sha256']}
        exclusive(directory/'main.png',cv2.imencode('.png',image)[1].tobytes()); exclusive(directory/'wrist.png',raw_wrist)
        exclusive(directory/'observation.json',encoded(sample)); exclusive(directory/'provenance.json',encoded(provenance))
        samples.append(sample)
        tmp=self.output/'manifest.tmp'; tmp.write_bytes(encoded(self.manifest)); tmp.replace(self.manifest_path)
        return {'message':f'已保存第 {len(samples)} 个机械臂姿态。'}

    def solve(self):
        if self.solving: raise ValueError('正在求解')
        if len(self.manifest['arms'][self.args.arm]['samples'])<6: raise ValueError('至少需要 6 个不同机械臂姿态，建议 10–15 个')
        self.check_binding(); data=json.loads(json.dumps(self.manifest)); self.solving=True; self.error=None
        def run():
            try:
                report=solve_manifest(data); name=f'result-{time.time_ns()}.json'; exclusive(self.output/name,encoded(report)); self.report={'filename':name,'contents':report}
            except Exception as exc: self.error=str(exc)
            finally: self.solving=False
        threading.Thread(target=run,daemon=True).start()
        return {'message':'正在后台求解和留一姿态验证，请耐心等待。'}

    def state(self):
        result={'arm':self.args.arm,'feedback':self.feedback,'samples':len(self.manifest['arms'][self.args.arm]['samples']),'intrinsics_adopted':self.adopted is not None,'solving':self.solving,'error':self.error,'report':self.report,'main_ready':False}
        try:
            packet,image=self.main_frame(); _,ids,_=self.detect(image)
            result.update(main_ready=True,main_corners=len(ids),main_sequence=packet['reference']['sequence'])
        except Exception as exc: result['main_error']=str(exc)
        result['wrist_ready']=False
        try:
            wrist=self.child('/api/state')
            age=wrist.get('frame_age_seconds')
            result['wrist_ready']=(wrist.get('frame_id') is not None and isinstance(age,(int,float))
                                   and 0<=age<=1)
            result['wrist_error']=wrist.get('error') or ('画面已过期，正在等待相机恢复' if not result['wrist_ready'] else None)
        except Exception:
            result['wrist_error']='相机服务暂时未连接'
        return result

class Server(ThreadingHTTPServer):
    daemon_threads=True
    allow_reuse_address=True

def handler(dashboard,token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,status,kind,data):
            try:
                self.send_response(status); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # A disconnected preview client does not invalidate a saved action.
                self.close_connection=True
        def json(self,status,value): self.send(status,'application/json; charset=utf-8',encoded(value))
        def origin_ok(self):
            expected=dashboard.args.public_origin
            return self.headers.get('Origin')==expected and self.headers.get('Host')==urlsplit(expected).netloc
        def proxy(self,method):
            allowed={'/intrinsics','/api/state','/api/observation','/api/candidate','/frame.jpg','/api/action'}
            path=urlsplit(self.path).path
            if path not in allowed or (method=='POST' and (path!='/api/action' or not self.origin_ok())): self.json(403,{'error':'请求来源不允许'}); return
            length=int(self.headers.get('Content-Length','0'))
            if length<0 or length>4096: raise ValueError('请求过大')
            body=self.rfile.read(length) if method=='POST' else None
            headers={'Host':f'127.0.0.1:{dashboard.args.wrist_port}','Origin':f'http://127.0.0.1:{dashboard.args.wrist_port}'}
            for key in ('Content-Type','X-CSRF-Token'):
                if self.headers.get(key): headers[key]=self.headers[key]
            # POSTs are never retried: the child may already have saved the action.
            connection=http.client.HTTPConnection('127.0.0.1',dashboard.args.wrist_port,timeout=180 if method=='POST' else 10)
            try:
                connection.request(method,'/' if path=='/intrinsics' else self.path,body,headers); response=connection.getresponse(); self.send(response.status,response.getheader('Content-Type','application/octet-stream'),response.read())
            finally: connection.close()
        def do_GET(self):
            try:
                path=urlsplit(self.path).path
                if path=='/': self.send(200,'text/html; charset=utf-8',PAGE.replace('__TOKEN__',json.dumps(token)).encode())
                elif path=='/api/handeye/state': self.json(200,dashboard.state())
                elif path=='/api/main/frame.jpg':
                    _,im=dashboard.main_frame(); _,ids,px=dashboard.detect(im)
                    if len(ids): cv2.aruco.drawDetectedCornersCharuco(im,px.astype(np.float32).reshape(-1,1,2),ids.astype(np.int32).reshape(-1,1))
                    self.send(200,'image/jpeg',cv2.imencode('.jpg',im)[1].tobytes())
                elif path=='/api/handeye/report':
                    if dashboard.report is None: raise ValueError('尚无求解报告')
                    self.json(200,dashboard.report['contents'])
                else: self.proxy('GET')
            except (http.client.HTTPException, socket.timeout, ConnectionError):
                self.json(503,{'error':'相机服务连接暂时中断，请稍后刷新。'})
            except Exception as exc: self.json(503,{'error':str(exc)})
        def do_POST(self):
            try:
                if urlsplit(self.path).path=='/api/action': self.proxy('POST'); return
                if not self.origin_ok() or not hmac.compare_digest(self.headers.get('X-CSRF-Token',''),token): self.json(403,{'error':'请求来源或校验令牌无效'}); return
                if self.headers.get_content_type()!='application/json': raise ValueError('需要 JSON 请求')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096: raise ValueError('请求大小无效')
                data=json.loads(self.rfile.read(length))
                with dashboard.lock:
                    path=urlsplit(self.path).path
                    if path=='/api/handeye/check-arm': result=dashboard.check_arm()
                    elif path=='/api/handeye/adopt': result=dashboard.adopt()
                    elif path=='/api/handeye/capture': result=dashboard.capture(data.get('fixed'))
                    elif path=='/api/handeye/solve': result=dashboard.solve()
                    else: raise ValueError('未知操作')
                self.json(200,{'ok':True,**result})
            except (http.client.HTTPException, socket.timeout, ConnectionError):
                self.json(503,{'ok':False,'error':'相机服务连接中断，操作可能已经完成。请先查看保存数量或计算结果，勿重复点击。'})
            except Exception as exc: self.json(400,{'ok':False,'error':str(exc)})
    return Handler

PAGE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SO101 双相机标定台</title><style>
[hidden]{display:none!important}*{box-sizing:border-box}body{margin:0;background:#101720;color:#e6edf4;font:15px/1.65 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:28px;margin:0}h2{font-size:19px;margin:0 0 12px}p{margin:8px 0;color:#b5c4d3}.tag{color:#65d5bc;font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:#19232f;border:1px solid #304052;border-radius:14px;padding:20px;margin-top:20px}img{width:100%;background:#080d13;border-radius:8px}iframe{width:100%;height:890px;border:0;border-radius:8px;background:white}button,a.button{display:inline-block;color:#08241e;background:#6ee0bd;border:0;border-radius:7px;padding:10px 16px;font:inherit;font-weight:600;cursor:pointer;text-decoration:none;margin:8px 8px 8px 0}button:disabled{opacity:.35;cursor:not-allowed}label{display:block;padding:8px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#b7dbf5}#notice{position:sticky;top:0;background:#243c48;padding:12px 18px;border-radius:8px;z-index:2}.warn{color:#f5ca7d}@media(max-width:800px){main{padding:14px}.grid{grid-template-columns:1fr}iframe{height:1000px}}
</style><main><span class="tag">RBT · 官方 SO101 + 软鳍夹爪 · v2</span><h1>双相机标定台</h1><p>当前左臂 · 主相机与夹爪相机看同一块板 · 手动摆姿态</p><div id="notice">正在连接…</div>
<section class="card"><h2>01 / 夹爪相机内参</h2><details id="intrinsics-details" open><summary>展开 / 收起内参采集</summary><p>先让手臂保持静止。移动板子到画面不同位置，并改变倾角，至少保存 12 个姿态。先锁焦，再采集，最后计算内参。</p><iframe src="/intrinsics" title="夹爪相机内参采集"></iframe><button id="adopt">采用这份内参，进入手眼采集</button><p id="adoption"></p></details></section>
<section class="card"><h2>02 / 两相机共同观测</h2><p class="warn">从这一步开始，板子、主相机、云台和机械臂底座必须保持固定；只移动机械臂。每次停稳后再保存。</p><div class="grid"><div><p id="mainstatus">主相机</p><img id="main" alt="主相机真实画面"></div><div><p id="wriststatus">左臂夹爪相机</p><img id="wrist" src="/frame.jpg" alt="夹爪相机真实画面"></div></div><button id="checkarm">检查 v2 机械臂反馈</button><p id="feedback">尚未检查</p><label><input type="checkbox" id="fixed"> 我已固定板子、主相机、云台和底座</label><button id="capture">保存当前机械臂姿态</button><p id="count">已采集 0 个姿态</p><p>建议采集 10–15 个姿态，包含绕不同轴的转动与位置变化。网页只读取 v2 电机反馈，不发送运动指令。软件接收时间配对不代表曝光同步。</p></section>
<section class="card"><h2>03 / 求解与检查</h2><button id="solve">计算手眼标定</button><a class="button" id="download" href="/api/handeye/report" download="shared-board-result.json" hidden>下载结果 JSON</a><pre id="report">等待采集。结果会显示重投影误差、留一姿态验证和运动多样性检查。</pre><p>结果保存为候选，不会自动替换 M3T 初始化。左臂通过不代表右臂已标定。</p></section></main>
<script>const token=__TOKEN__;let busy=false;let refreshing=false;let phase=null;async function action(name,body={}){if(busy)return;busy=true;const n=document.querySelector('#notice');n.textContent='处理中…';try{const r=await fetch('/api/handeye/'+name,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(body)});const d=await r.json();n.textContent=d.message||d.error}catch(e){n.textContent=String(e)}finally{busy=false;refresh()}}document.querySelector('#checkarm').onclick=()=>action('check-arm');document.querySelector('#adopt').onclick=()=>action('adopt');document.querySelector('#capture').onclick=()=>action('capture',{fixed:document.querySelector('#fixed').checked});document.querySelector('#solve').onclick=()=>action('solve');async function refresh(){if(refreshing)return;refreshing=true;try{let r=await fetch('/api/handeye/state');let s=await r.json();if(phase!==s.intrinsics_adopted){phase=s.intrinsics_adopted;document.querySelector('#intrinsics-details').open=!phase;document.querySelector('#notice').textContent=phase?'当前是手眼采集：板子、主相机和底座保持固定，只移动机械臂。':'先完成第 1 步：锁焦并采集夹爪相机内参。'}document.querySelector('#feedback').textContent=s.feedback;document.querySelector('#mainstatus').textContent=s.main_ready?'主相机 · 已连接 · '+s.main_corners+' 个角点':'主相机 · '+s.main_error;document.querySelector('#adoption').textContent=s.intrinsics_adopted?'内参已固定，可以开始手眼采集。':'请先在上方计算内参，然后点击采用。';document.querySelector('#adopt').disabled=s.intrinsics_adopted;document.querySelector('#capture').disabled=!s.intrinsics_adopted||!s.wrist_ready||!s.main_ready||s.solving||busy;document.querySelector('#solve').disabled=s.samples<6||s.solving||busy;document.querySelector('#count').textContent='已采集 '+s.samples+' 个机械臂姿态';if(s.solving)document.querySelector('#report').textContent='正在求解和留一姿态验证…';if(s.report){document.querySelector('#report').textContent=JSON.stringify(s.report.contents,null,2);document.querySelector('#download').hidden=false}if(s.error)document.querySelector('#report').textContent=s.error;if(s.main_ready)document.querySelector('#main').src='/api/main/frame.jpg?t='+Date.now();document.querySelector('#wriststatus').textContent=s.wrist_ready?'左臂夹爪相机 · 实时画面':'左臂夹爪相机 · '+s.wrist_error;if(s.wrist_ready)document.querySelector('#wrist').src='/frame.jpg?t='+Date.now();else document.querySelector('#wrist').removeAttribute('src')}catch(e){document.querySelector('#mainstatus').textContent='连接失败：'+e}finally{refreshing=false}}document.querySelector('#notice').textContent='先完成第 1 步：锁焦并采集夹爪相机内参。';refresh();setInterval(refresh,1500);</script></html>'''

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('main-session','main-packet','robot-model','calibration','output','driver-python','driver-path','arm-port'): p.add_argument('--'+key,required=True)
    p.add_argument('--arm',choices=['so101_left','so101_right'],default='so101_left')
    p.add_argument('--port',type=int,default=8098); p.add_argument('--wrist-port',type=int,default=8097)
    p.add_argument('--public-origin',default='http://127.0.0.1:18898')
    return p

def main():
    args=parser().parse_args()
    u=urlsplit(args.public_origin)
    if u.scheme!='http' or u.hostname not in ('127.0.0.1','localhost') or u.path or u.query or u.fragment or u.username: raise ValueError('public origin 仅允许本机 HTTP 地址')
    dashboard=Dashboard(args)
    server=Server(('127.0.0.1',args.port),handler(dashboard,secrets.token_urlsafe(32)))
    print('Dashboard ready on loopback port',args.port,flush=True)
    try: server.serve_forever()
    finally: server.server_close()

if __name__=='__main__': main()
