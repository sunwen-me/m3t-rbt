"""Reject stale/misbound packets and preserve exact tracked-frame pairing."""
import copy
import hashlib
import json
import unittest
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import m3t_preview as web


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.config={'image_size':[4,3]};self.now=12_000_000_000
        self.estimate={'schema_version':1,'event':'estimate','modalities':['region','depth'],
            'frame_id':'orbbec335/color_optical','color_capture_ns':self.now-100_000_000,
            'capture_clock_kind':'HostMonotonic','capture_clock_source':'12'*16,
            'capture_clock_generation':0,'frame_sequence':3,
            'links':[{'arm':'so101_left','link':'gripper_link','pose':np.eye(4).reshape(-1).tolist()}],
            'joint_angles_rad':{'so101_left':dict.fromkeys(web.JOINT_OUTPUTS,0.)}}
        self.image=np.arange(36,dtype=np.uint8).reshape(3,4,3)
    def packet(self,estimate=None,raw=None):
        head={'schema':'rbt-m3t-preview-v1','width':4,'height':3,'pixel_format':'BGR8',
              'estimate':estimate if estimate is not None else self.estimate}
        return json.dumps(head).encode()+b'\n'+(self.image.tobytes() if raw is None else raw)
    def parse(self,payload):return web.parse_packet(payload,self.now,self.config,'so101_left',['gripper_link'])
    def test_exact_image_and_output_pose_preserved(self):
        e=copy.deepcopy(self.estimate);e['links'][0]['pose'][3]=.123
        estimate,image,poses=self.parse(self.packet(e))
        np.testing.assert_array_equal(image,self.image)
        self.assertEqual(poses['gripper_link'][0,3],.123)
        self.assertEqual(estimate['frame_sequence'],3)
    def test_rejects_stale_wrong_camera_and_clock(self):
        for key,value in [('color_capture_ns',self.now-3_000_000_000),('color_capture_ns',self.now+1),
                          ('frame_id','chest_camera/color_optical'),('capture_clock_kind','Device'),
                          ('capture_clock_source','00'*16),('modalities',['region'])]:
            with self.subTest(key=key,value=value):
                e=copy.deepcopy(self.estimate);e[key]=value
                with self.assertRaises(ValueError):self.parse(self.packet(e))
    def test_rejects_partial_image_and_wrong_arm_or_link(self):
        with self.assertRaises(ValueError):self.parse(self.packet(raw=b'x'))
        for changed in [{'arm':'so101_right'},{'link':'wrist_link'},{'pose':[0.]*16}]:
            e=copy.deepcopy(self.estimate);e['links'][0].update(changed)
            with self.assertRaises(ValueError):self.parse(self.packet(e))
    def test_rejects_nonfinite_joint_and_duplicate_link(self):
        e=copy.deepcopy(self.estimate);e['joint_angles_rad']['so101_left']['shoulder_pan']=float('nan')
        with self.assertRaises(ValueError):self.parse(self.packet(e))
        e=copy.deepcopy(self.estimate);del e['joint_angles_rad']['so101_left']['gripper']
        with self.assertRaises(ValueError):self.parse(self.packet(e))
        e=copy.deepcopy(self.estimate);e['joint_angles_rad']['so101_left']['gripper']=float('nan')
        with self.assertRaises(ValueError):self.parse(self.packet(e))
        e=copy.deepcopy(self.estimate);e['links'].append(e['links'][0])
        with self.assertRaises(ValueError):self.parse(self.packet(e))

    def dual_estimate(self):
        estimate=copy.deepcopy(self.estimate)
        estimate['links'].append(copy.deepcopy(estimate['links'][0]))
        estimate['links'][1]['arm']='so101_right'
        estimate['links'][1]['pose'][3]=.234
        estimate['joint_angles_rad']['so101_right']=dict.fromkeys(web.JOINT_OUTPUTS,.5)
        return estimate

    def test_bimanual_packet_returns_both_arms_from_the_exact_image(self):
        estimate=self.dual_estimate()
        for arms in (('so101_left','so101_right'),['so101_left','so101_right']):
            with self.subTest(arms=arms):
                actual,image,poses=web.parse_packet(self.packet(estimate),self.now,self.config,
                    arms,{arm:['gripper_link'] for arm in arms})
                np.testing.assert_array_equal(image,self.image)
                self.assertEqual(actual,estimate)
                self.assertEqual(set(poses),set(arms))
                self.assertEqual(poses['so101_left']['gripper_link'][0,3],0.)
                self.assertEqual(poses['so101_right']['gripper_link'][0,3],.234)
                self.assertEqual(actual['frame_sequence'],3)

    def test_bimanual_packet_rejects_missing_unknown_duplicate_or_misbound_arms(self):
        edits=[
            lambda e:e['links'].pop(),
            lambda e:e['links'][1].update(arm='other'),
            lambda e:e['links'][1].update(arm='so101_left'),
            lambda e:e['links'].append(copy.deepcopy(e['links'][0])),
            lambda e:e['joint_angles_rad'].pop('so101_right'),
            lambda e:e['joint_angles_rad'].update(other=dict.fromkeys(web.JOINT_OUTPUTS,0.)),
            lambda e:e['joint_angles_rad']['so101_right'].pop('gripper'),
            lambda e:e['joint_angles_rad']['so101_right'].update(gripper=float('nan')),
            lambda e:e['links'][1].update(link='unrelated_link'),
        ]
        for i,edit in enumerate(edits):
            with self.subTest(case=i):
                estimate=self.dual_estimate();edit(estimate)
                with self.assertRaises(ValueError):
                    web.parse_packet(self.packet(estimate),self.now,self.config,
                        ('so101_left','so101_right'),['gripper_link'])
        for arms in ([],['so101_left','so101_left']):
            with self.subTest(arms=arms),self.assertRaises(ValueError):
                web.parse_packet(self.packet(self.dual_estimate()),self.now,self.config,arms,['gripper_link'])

    def test_single_arm_parser_rejects_other_arm_joint_outputs_and_links(self):
        with self.assertRaises(ValueError):self.parse(self.packet(self.dual_estimate()))
        estimate=copy.deepcopy(self.estimate)
        estimate['joint_angles_rad']['so101_right']=dict.fromkeys(web.JOINT_OUTPUTS,0.)
        with self.assertRaises(ValueError):self.parse(self.packet(estimate))
        estimate=copy.deepcopy(self.estimate)
        estimate['links'][0]['arm']='so101_right'
        estimate['joint_angles_rad']={'so101_right':estimate['joint_angles_rad']['so101_left']}
        _,image,poses=web.parse_packet(self.packet(estimate),self.now,self.config,'so101_right',['gripper_link'])
        np.testing.assert_array_equal(image,self.image)
        self.assertEqual(set(poses),{'gripper_link'})


def calibration():
    span=2.201799979439
    return {'closed_rad':0.0,'open_rad':span,'scale_rad_per_percent':span/100.0,
        'calibration_sha256':'ab'*32,
        'zero_reference':'physical closed endpoint assigned nominal model zero; alignment not independently measured'}


class GripperMappingTests(unittest.TestCase):
    def test_percent_endpoints_midpoint_and_prior_mapping(self):
        c=calibration();span=c['open_rad']
        self.assertAlmostEqual(web.gripper_angle_from_percent(0,c),0.0)
        self.assertAlmostEqual(web.gripper_angle_from_percent(50,c),span/2)
        self.assertAlmostEqual(web.gripper_angle_from_percent(100,c),span)
        mappings=web.prior_mappings('so101_left',c)
        self.assertEqual(len(mappings),6)
        self.assertEqual([m['joint'] for m in mappings],list(web.NAMES)+['gripper'])
        for mapping in mappings[:5]:
            self.assertAlmostEqual(mapping['scale_rad_per_unit'],np.pi/180)
            self.assertEqual(mapping['offset_rad'],0.0)
            self.assertEqual(mapping['standard_deviation_rad'],0.1)
        self.assertAlmostEqual(mappings[-1]['scale_rad_per_unit'],span/100)
        self.assertAlmostEqual(mappings[-1]['offset_rad'],0.0)
        self.assertEqual(mappings[-1]['standard_deviation_rad'],0.01)

    def test_right_prior_mapping_keeps_six_calibrated_priors_and_sigmas(self):
        left=web.prior_mappings('so101_left',calibration())
        right=web.prior_mappings('so101_right',calibration())
        self.assertEqual(len(right),6)
        for l,r in zip(left,right):
            self.assertEqual(l['source_arm'],'left')
            self.assertEqual(r['source_arm'],'right')
            self.assertEqual(r['arm'],'so101_right')
            self.assertEqual({k:v for k,v in l.items() if k not in ('arm','source_arm')},
                {k:v for k,v in r.items() if k not in ('arm','source_arm')})
        with self.assertRaises(ValueError):web.prior_mappings('other',calibration())

    def test_physical_span_does_not_come_from_runtime_urdf_upper_limit(self):
        c=calibration()
        robot={'schema_version':1,'name':'fixture','root_link':'base_link','units':'metres_radians',
            'source_urdf':'/source/robot.urdf','links':[{'name':'base_link','mesh':''}],
            'joints':[{'name':'gripper','type':'revolute','lower':-1.0,'upper':1.0,
                'parent':'base_link','child':'jaw'}], 'unrecognized_model_metadata':{'kept':True}}
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'robot.json';source.write_text(json.dumps(robot))
            original=source.read_bytes()
            runtime,audit=web.runtime_robot_model(source,c)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(runtime['unrecognized_model_metadata'],{'kept':True})
            self.assertEqual(runtime['source_urdf'],robot['source_urdf'])
            joint=runtime['joints'][0]
            self.assertEqual(joint['lower'],-1.0)
            self.assertEqual(joint['upper'],c['open_rad'])
            self.assertEqual(audit['source_robot_json_sha256'],hashlib.sha256(original).hexdigest())
            self.assertAlmostEqual(web.prior_mappings('so101_left',c)[-1]['scale_rad_per_unit'],
                c['open_rad']/100)
            robot['joints'][0]['upper']=100.0
            source.write_text(json.dumps(robot))
            wide_runtime,_=web.runtime_robot_model(source,c)
            self.assertEqual(wide_runtime['joints'][0]['upper'],100.0)
            self.assertAlmostEqual(web.prior_mappings('so101_left',c)[-1]['scale_rad_per_unit'],
                c['open_rad']/100)

    def test_rejects_invalid_percent_and_moving_feedback(self):
        for bad in (float('nan'),-0.01,100.01):
            with self.subTest(percent=bad), self.assertRaises(ValueError):
                web.gripper_angle_from_percent(bad,calibration())
            sample={'degrees_and_gripper_percent':[0,0,0,0,0,bad]}
            with self.subTest(feedback=bad), self.assertRaises(ValueError):
                web.averaged_feedback(sample,sample)
        before={'degrees_and_gripper_percent':[0,0,0,0,0,25]}
        after={'degrees_and_gripper_percent':[0,0,0,0,0,25.51]}
        with self.assertRaises(ValueError):web.averaged_feedback(before,after)

class PriorGateTests(unittest.TestCase):
    def test_requires_six_fresh_actual_priors_including_gripper(self):
        prior={'status':'applied','prior_count':6,'controller_id':'real','time_basis':'received',
               'age_ns':10_000_000,'priors':[{'arm':'so101_left','joint':n} for n in web.JOINT_OUTPUTS]}
        manager=web.M3TExperiment.__new__(web.M3TExperiment)
        manager.enabled=True;manager.process=SimpleNamespace(poll=lambda:None)
        manager.packet_path=SimpleNamespace(exists=lambda:True,read_bytes=lambda:b'packet')
        manager.preview=SimpleNamespace(dashboard=SimpleNamespace(main_config={},args=SimpleNamespace(arm='so101_left')),renderer=SimpleNamespace(_meshes={}))
        manager.prior_service='stream';manager.prior_controller='real';manager.clock=None;manager.last_sequence=None
        estimate={'joint_priors':prior,'capture_clock_source':'clock','capture_clock_generation':0,'frame_sequence':1}
        with patch.object(web,'parse_packet',return_value=(estimate,None,{})):
            self.assertIs(manager.read_frame()[0],estimate)
            for key,bad in [('status','missing'),('prior_count',5),('controller_id','other'),('time_basis','capture'),('age_ns',100_000_001),('age_ns',-1)]:
                with self.subTest(key=key):
                    value=prior[key];prior[key]=bad
                    with self.assertRaises(ValueError):manager.read_frame()
                    prior[key]=value
            prior['priors']=prior['priors'][:-1]
            with self.assertRaises(ValueError):manager.read_frame()
            prior['prior_count']=5
            prior['priors']=[{'arm':'so101_left','joint':n} for n in web.NAMES]
            with self.assertRaises(ValueError):manager.read_frame()

class InitializationTests(unittest.TestCase):
    def test_nominal_B_fresh_joints_and_sources_reach_actual_initialization(self):
        from test_overlay_preview import nominal_preview
        import overlay_preview
        for nominal in (True,False):
            with self.subTest(nominal=nominal),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                p=nominal_preview(folder)
                if not nominal:
                    with patch.object(overlay_preview,'ModelOverlay'),patch.object(overlay_preview.threading.Thread,'start'):
                        p=overlay_preview.OverlayPreview(p.dashboard,p.audit_path,p.report_path,p.static_config,p.pitch_url)
                p.renderer.gripper_calibration=calibration()
                before={'degrees_and_gripper_percent':[10,20,30,40,50,25]}
                after={'degrees_and_gripper_percent':[10.2,20.2,30.2,40.2,50.2,25.2]}
                from unittest.mock import Mock
                p.dashboard.read_arm=Mock(side_effect=[before,after])
                command=root/'fake-m3t';command.write_text('test only');command.chmod(0o700)
                manager=web.M3TExperiment(p,command,root/'runs',root/'cache',prior_service='stream',prior_controller='real')
                real_mkdtemp=web.tempfile.mkdtemp
                def local_mkdtemp(*args,**kwargs):
                    if kwargs.get('dir')=='/dev/shm':kwargs['dir']=folder
                    return real_mkdtemp(*args,**kwargs)
                with patch.object(web.subprocess,'Popen',return_value=SimpleNamespace(pid=123,poll=lambda:None)) as process, \
                        patch.object(web,'fk',return_value=np.eye(4)), \
                        patch.object(web.time,'sleep'),patch.object(web.tempfile,'mkdtemp',side_effect=local_mkdtemp), \
                        patch.object(p,'read_pitch',wraps=p.read_pitch if nominal else lambda:-61.0) as pitch:
                    manager.start()
                process.assert_called_once();pitch.assert_called_once()
                self.assertEqual(p.dashboard.read_arm.call_count,2)
                init=json.loads((manager.run/'initialization.json').read_text())
                self.assertEqual(len(init['arms']),1)
                arm=init['arms'][0]
                self.assertEqual(arm['name'],'so101_left')
                self.assertIs(arm['fixed_base'],True)
                np.testing.assert_array_equal(arm['base_in_camera'],p.B.reshape(-1))
                for name,angle in zip(web.NAMES,[10.1,20.1,30.1,40.1,50.1]):
                    self.assertAlmostEqual(arm['joint_angles'][name],np.deg2rad(angle))
                self.assertAlmostEqual(arm['joint_angles']['gripper'],web.gripper_angle_from_percent(25.1,calibration()))
                provenance=init['provenance']
                self.assertEqual(provenance['joint_before'],before)
                self.assertEqual(provenance['joint_after'],after)
                self.assertEqual(provenance['mount_source'],p.mount_source)
                self.assertEqual(provenance['pitch_source'],'assumed_fixed' if nominal else 'live_controller')
                self.assertEqual(provenance['pitch_deg'],-61.0)
                self.assertIs(provenance['solver_accepted'],False)
                if nominal:
                    self.assertNotIn('mount_audit_sha256',provenance)
                    self.assertEqual(provenance['legacy_binding_audit_sha256'],web.sha(p.audit_path))
                    self.assertEqual(provenance['mount_source']['sha256'],web.sha(root/'prior.json'))
                    self.assertEqual(provenance['mount_source']['path'],str(root/'prior.json'))
                else:self.assertEqual(provenance['mount_audit_sha256'],web.sha(p.audit_path))
                launch=json.loads((manager.run/'launch.json').read_text())
                for data in (launch,manager.state()):
                    self.assertEqual(data['mount_source'],p.mount_source)
                    self.assertEqual(data['pitch_source'],p.pitch_source)
                    self.assertIs(data['fixed_base'],True)
                    self.assertIs(data['solver_accepted'],False)
                tracking=json.loads((manager.run/'tracking.json').read_text())
                self.assertNotIn('mount_refinement',tracking)
                self.assertNotIn('visual_link_weights',tracking)
                self.assertEqual(len(tracking['joint_priors']['mappings']),6)
                self.assertTrue(all(m['arm']=='so101_left' and m['source_arm']=='left'
                    for m in tracking['joint_priors']['mappings']))

    def test_moving_feedback_rejected_before_process_or_run_creation(self):
        from test_overlay_preview import nominal_preview
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder:
            p=nominal_preview(folder);p.renderer.gripper_calibration=calibration()
            p.dashboard.read_arm=Mock(side_effect=[
                {'degrees_and_gripper_percent':[0,0,0,0,0,25]},
                {'degrees_and_gripper_percent':[1,0,0,0,0,25]}])
            root=Path(folder);command=root/'fake-m3t';command.write_text('test only');command.chmod(0o700)
            manager=web.M3TExperiment(p,command,root/'runs',root/'cache')
            with patch.object(web.subprocess,'Popen') as process,patch.object(web.time,'sleep'):
                with self.assertRaisesRegex(ValueError,'停稳'):manager.start()
                process.assert_not_called()
            self.assertFalse((root/'runs').exists())


def bimanual_preview(folder):
    """Two independent feedback contexts, one canonical model and one image source."""
    from gripper_calibration import load_gripper_calibration
    root=Path(folder)
    mesh=root/'canonical.obj';mesh.write_text('v 0 0 0\n')
    model={'schema_version':1,'name':'fixture','root_link':'base_link',
        'source_urdf':str(root/'canonical.urdf'),
        'links':[{'name':'base_link','mesh':str(mesh)}],
        'joints':[{'name':'gripper','type':'revolute','lower':-.1,'upper':.2}]}
    model_path=root/'canonical.json';model_path.write_text(json.dumps(model))
    contexts={}
    for i,arm in enumerate(('so101_left','so101_right')):
        path=root/(arm+'-v2.json')
        path.write_text(json.dumps({'gripper':{'id':6,'range_min':1000,'range_max':2000+500*i,'drive_mode':i}}))
        B=np.eye(4);B[:3,3]=[.1 if i==0 else -.2,.3,.5]
        before={'degrees_and_gripper_percent':[10+10*i,20+10*i,30+10*i,40+10*i,50+10*i,25+25*i],
            'read_started':1.+i,'read_completed':1.01+i,'source':'actual readonly '+arm}
        after={'degrees_and_gripper_percent':[v+.2 for v in before['degrees_and_gripper_percent']],
            'read_started':1.2+i,'read_completed':1.21+i,'source':'actual readonly '+arm}
        d=SimpleNamespace(args=SimpleNamespace(arm=arm,calibration=str(path),robot_model=str(model_path)),
            model=model,main_config={'camera_serial':'fixture','image_size':[4,3]},
            read_arm=Mock(side_effect=[before,after]))
        contexts[arm]=SimpleNamespace(dashboard=d,B=B,
            renderer=SimpleNamespace(gripper_calibration=load_gripper_calibration(path),_meshes={'gripper_link':None}),
            mount_source={'kind':'nominal_cad','nominal':True,'arm':arm,'sha256':arm+'-prior'},
            audit_path=None,check_sources=Mock(),read_pitch=Mock(return_value=-61.))
    return SimpleNamespace(arm_previews=dict(reversed(list(contexts.items()))),
        dashboard=contexts['so101_left'].dashboard,
        mount_source={'kind':'nominal_cad','nominal':True,'arms':{arm:c.mount_source for arm,c in contexts.items()},'solver_accepted':False},
        audit_path=None,expected_pitch=-61.,pitch_source='assumed_fixed',
        check_sources=Mock(),read_pitch=Mock(return_value=-61.),lock=threading.Lock(),event=threading.Event())


def start_bimanual(preview,folder,prior_service='stream',**options):
    root=Path(folder);command=root/'fake-m3t';command.write_text('test only');command.chmod(0o700)
    manager=web.M3TExperiment(preview,command,root/'runs',root/'cache',prior_service=prior_service,prior_controller='real',**options)
    real_mkdtemp=web.tempfile.mkdtemp
    def local_mkdtemp(*args,**kwargs):
        if kwargs.get('dir')=='/dev/shm':kwargs['dir']=folder
        return real_mkdtemp(*args,**kwargs)
    with patch.object(web.subprocess,'Popen',return_value=SimpleNamespace(pid=123,poll=lambda:None)) as process, \
            patch.object(web,'fk',return_value=np.eye(4)),patch.object(web.time,'sleep'), \
            patch.object(web.tempfile,'mkdtemp',side_effect=local_mkdtemp), \
            patch.object(web.shutil,'copyfile',wraps=web.shutil.copyfile) as copy_mesh:
        manager.start()
    return manager,process,copy_mesh


class AttachedPacketTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / 'latest.bin'
        self.preview = bimanual_preview(folder.name)
        self.fixture = PacketTests()
        self.fixture.setUp()
        self.now = self.fixture.now
        model = self.preview.dashboard.model
        model['links'] = [{'name': 'base_link'}]
        model['joints'] = []
        parent = 'base_link'
        for i, name in enumerate(web.NAMES):
            child = 'gripper_link' if i == 4 else f'joint_link_{i}'
            model['links'].append({'name': child})
            model['joints'].append({'name': name, 'type': 'revolute', 'parent': parent,
                'child': child, 'origin': np.eye(4).reshape(-1).tolist(),
                'axis': [0, 0, 1], 'lower': -2., 'upper': 2.})
            parent = child
        self.subscriber = web.M3TPacketSubscriber(self.preview, self.path,
            prior_service='stream', prior_controller='real', publish_tf=True)
        estimate = self.fixture.dual_estimate()
        estimate.update(encoder_reference_links=True, links=[], joint_priors={
            'status': 'applied', 'prior_count': 12, 'controller_id': 'real', 'time_basis': 'received',
            'age_ns': 10_000_000, 'priors': [],
        })
        for arm, child in self.preview.arm_previews.items():
            base = child.B.copy()
            base[0, 3] += .02
            angles = dict.fromkeys(web.JOINT_OUTPUTS, 0.)
            estimate['joint_angles_rad'][arm] = angles
            for link in ['base_link', 'gripper_link']:
                estimate['links'].append({'arm': arm, 'link': link, 'pose': base.reshape(-1).tolist()})
            for joint in angles:
                value = (web.gripper_angle_from_percent(25., child.renderer.gripper_calibration)
                         if joint == 'gripper' else .1)
                estimate['joint_priors']['priors'].append({'arm': arm, 'joint': joint, 'angle_rad': value})
        self.estimate = estimate

    def publish(self, estimate):
        path = self.path.with_suffix('.tmp')
        path.write_bytes(self.fixture.packet(estimate))
        path.replace(self.path)

    def refined_estimate(self, base_shift=0.):
        estimate = copy.deepcopy(self.estimate)
        entries = []
        for arm, child in self.preview.arm_previews.items():
            # The independent tracker's prior center differs from the dashboard
            # overlay's CAD reference, and its fused joints differ from feedback.
            nominal = child.B.copy()
            nominal[1, 3] -= .03
            refined = child.B.copy()
            refined[0, 3] += .02 + base_shift
            angle = .05
            refined[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                               [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
            estimate['joint_angles_rad'][arm][web.NAMES[0]] = -.15
            gripper = refined @ web.fk(self.preview.dashboard.model,
                [np.rad2deg(estimate['joint_angles_rad'][arm][name]) for name in web.NAMES])
            for link in estimate['links']:
                if link['arm'] == arm:
                    link['pose'] = (refined if link['link'] == 'base_link' else gripper).reshape(-1).tolist()
            entries.append({'arm': arm, 'nominal_base_in_camera': nominal.reshape(-1).tolist(),
                'refined_base_in_camera': refined.reshape(-1).tolist(),
                'translation_correction_m': float(np.linalg.norm(refined[:3, 3] - nominal[:3, 3])),
                'rotation_correction_rad': angle})
        estimate['mount_refinement'] = {'enabled': True, 'joints_locked_to_feedback': False, 'arms': entries}
        return estimate

    def test_soft_base_and_visual_joints_are_preserved_with_distinct_prior_centers(self):
        with patch.object(web.time, 'monotonic_ns', return_value=self.now):
            previous = None
            for shift in (0., .01):
                packet = self.refined_estimate(shift)
                self.publish(packet)
                estimate, _, poses, tcp, feedback = self.subscriber.read_sample()
                self.assertEqual(estimate, packet)
                state = self.subscriber.state()
                self.assertTrue(state['running'])
                self.assertIs(state['fixed_base'], False)
                self.assertIs(state['joints_locked_to_feedback'], False)
                self.assertEqual(state['native_tf_topology'], 'cad_mount_correction')
                self.assertEqual(state['mount_refinement'], packet['mount_refinement'])
                self.assertEqual(state['mode_description'], web.ENCODER_REFERENCE_MOUNT_DESCRIPTION)
                for arm, child in self.preview.arm_previews.items():
                    gripper = np.asarray(tcp[arm]['T_camera_from_gripper_link_m3t']).reshape(4, 4)
                    base = np.asarray(tcp[arm]['T_camera_from_base_link']).reshape(4, 4)
                    np.testing.assert_array_equal(gripper, poses[arm]['gripper_link'])
                    encoder = base @ web.fk(self.preview.dashboard.model, feedback[arm][:5])
                    self.assertFalse(np.allclose(gripper, encoder))
                    self.assertAlmostEqual(base[0, 3], child.B[0, 3] + .02 + shift)
                    if previous is not None:
                        self.assertAlmostEqual(gripper[0, 3] - previous[arm], .01)
                previous = {arm: tcp[arm]['T_camera_from_gripper_link_m3t'][3] for arm in tcp}

    def test_combined_mode_rejects_contradictory_flags_or_mount_diagnostics(self):
        edits = [
            lambda e: e['mount_refinement'].pop('joints_locked_to_feedback'),
            lambda e: e['mount_refinement'].update(joints_locked_to_feedback=True),
            lambda e: e['mount_refinement'].update(joints_locked_to_feedback=0),
            lambda e: e.update(joints_locked_to_feedback=True),
            lambda e: e.update(encoder_reference_links=False),
            lambda e: e['mount_refinement'].update(enabled=False),
            lambda e: e['mount_refinement']['arms'].pop(),
            lambda e: e['mount_refinement']['arms'][0].update(arm='other'),
            lambda e: e['mount_refinement']['arms'].__setitem__(0, e['mount_refinement']['arms'][1]),
            lambda e: e['mount_refinement']['arms'][0]['nominal_base_in_camera'].__setitem__(0, 2.),
            lambda e: e['mount_refinement']['arms'][0]['refined_base_in_camera'].__setitem__(3, .9),
            lambda e: e['mount_refinement']['arms'][0].update(translation_correction_m=0.),
            lambda e: e['mount_refinement']['arms'][0].update(rotation_correction_rad=0.),
        ]
        with patch.object(web.time, 'monotonic_ns', return_value=self.now):
            for index, edit in enumerate(edits):
                estimate = self.refined_estimate()
                edit(estimate)
                self.publish(estimate)
                with self.subTest(case=index):
                    state = self.subscriber.state()
                    self.assertFalse(state['running'])
                    self.assertIsNotNone(state['packet_error'])
                    self.assertIsNone(state['fixed_base'])

    def test_running_requires_a_fresh_valid_capture_and_reconnects_after_replacement(self):
        with patch.object(web.time, 'monotonic_ns', side_effect=lambda: self.now):
            self.assertFalse(self.subscriber.state()['running'])
            self.publish(self.estimate)
            estimate, image, poses, tcp, feedback = self.subscriber.read_sample()
            state = self.subscriber.state()
            self.assertTrue(state['running'])
            self.assertTrue(state['packet_ready'])
            self.assertEqual(state['packet_capture_age_seconds'], .1)
            self.assertEqual(state['native_tf_topology'], 'cad_mount_fixed_base')
            self.assertEqual(estimate, self.estimate)
            np.testing.assert_array_equal(image, self.fixture.image)
            for arm, child in self.preview.arm_previews.items():
                self.assertAlmostEqual(poses[arm]['gripper_link'][0, 3], child.B[0, 3] + .02)
                self.assertEqual(tcp[arm]['max_matrix_abs_difference'], 0.)
                self.assertAlmostEqual(feedback[arm][5], 25.)
                self.assertNotEqual(feedback[arm][0], estimate['joint_angles_rad'][arm][web.NAMES[0]])
            self.now += web.PREVIEW_MAX_AGE_NS
            self.assertFalse(self.subscriber.state()['running'])
            self.assertFalse(self.subscriber.state()['packet_ready'])
            replacement = dict(self.estimate, frame_sequence=0, color_capture_ns=self.now,
                               capture_clock_generation=1)
            self.publish(replacement)
            self.assertTrue(self.subscriber.state()['running'])
            self.path.unlink()
            self.assertFalse(self.subscriber.state()['running'])

    def test_rejects_wrong_mode_controller_partial_priors_and_bad_gripper_pose(self):
        edits = [
            lambda estimate: estimate.pop('encoder_reference_links'),
            lambda estimate: estimate.update(mount_refinement={}),
            lambda estimate: estimate['joint_priors'].update(controller_id='other'),
            lambda estimate: estimate['joint_priors']['priors'].pop(),
            lambda estimate: estimate['joint_priors']['priors'][0].pop('angle_rad'),
            lambda estimate: estimate['links'][1]['pose'].__setitem__(3, 0.),
        ]
        with patch.object(web.time, 'monotonic_ns', return_value=self.now):
            for edit in edits:
                with self.subTest(edit=edit):
                    estimate = copy.deepcopy(self.estimate)
                    edit(estimate)
                    self.publish(estimate)
                    self.assertFalse(self.subscriber.state()['running'])
            self.path.write_bytes(b'null\n' + self.fixture.image.tobytes())
            self.assertFalse(self.subscriber.state()['running'])

    def test_overlay_uses_native_image_and_poses_then_falls_back_to_raw(self):
        from aruco_web import gripper_pose_response
        from dual_overlay_preview import BimanualOverlayPreview

        preview = BimanualOverlayPreview.__new__(BimanualOverlayPreview)
        vars(preview).update(vars(self.preview))
        preview.m3t = self.subscriber
        preview.layers = {'encoder': False, 'm3t': True, 'predicted': False, 'observed': False}
        preview.enabled, preview.error, preview.X = True, None, None
        preview.dashboard.main_config['K'] = np.eye(3)
        raw = np.zeros_like(self.fixture.image)
        preview.dashboard.main_frame = Mock(return_value=({'reference': {'capture_ns': self.now}}, raw))
        for child in preview.arm_previews.values():
            child.renderer.render = Mock(side_effect=lambda image, *args: (image.copy(), {}))
            child.renderer.render_poses = Mock(side_effect=lambda image, *args, **kwargs: (image.copy(), {}))
        with patch.object(web.time, 'monotonic_ns', side_effect=lambda: self.now):
            self.publish(self.estimate)
            preview.sample = preview.acquire()
            np.testing.assert_array_equal(preview.sample['raw'], self.fixture.image)
            preview.dashboard.main_frame.assert_not_called()
            self.assertEqual(set(gripper_pose_response(preview)['arms']), set(preview.arm_previews))
            self.now += web.PREVIEW_MAX_AGE_NS
            with self.assertRaises(ValueError):
                gripper_pose_response(preview)
            preview.sample = preview.acquire()
            np.testing.assert_array_equal(preview.sample['raw'], raw)
            self.assertFalse(preview.sample['receipt']['m3t_enabled'])
            self.assertNotIn('tcp_poses', preview.sample['receipt'])
            self.assertTrue(preview.snapshot()[1]['ready'])
        for child in preview.arm_previews.values():
            child.dashboard.read_arm.assert_not_called()


class NativeTfTests(unittest.TestCase):
    def test_native_flag_and_metadata_are_opt_in(self):
        for enabled in (False,True):
            with self.subTest(enabled=enabled),tempfile.TemporaryDirectory() as folder:
                preview=bimanual_preview(folder)
                options={'publish_tf':True} if enabled else {}
                manager,process,_=start_bimanual(preview,folder,**options)
                argv=process.call_args.args[0]
                self.assertEqual(argv.count('--publish-tf'),int(enabled))
                initialization=json.loads((manager.run/'initialization.json').read_text())
                launch=json.loads((manager.run/'launch.json').read_text())
                provenance=initialization['provenance']
                for metadata in (provenance,launch,manager.state()):
                    self.assertIs(metadata['native_tf_enabled'],enabled)
                    self.assertEqual(metadata.get('native_tf_capability'),'perception/M3tTf' if enabled else None)
                    self.assertEqual(metadata.get('native_tf_frame_prefix'),'' if enabled else None)
                    self.assertEqual(metadata.get('native_tf_topology'),'cad_mount_correction' if enabled else None)
                if not enabled:
                    for key in ('native_tf_capability','native_tf_frame_prefix','native_tf_topology'):
                        self.assertNotIn(key,provenance)

    def test_configured_tf_does_not_imply_a_running_process(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=web.M3TExperiment(bimanual_preview(folder),'tracker','runs','cache',publish_tf=True)
            state=manager.state()
            self.assertIs(state['native_tf_enabled'],True)
            self.assertIs(state['running'],False)
            self.assertIs(state['enabled'],False)

    def test_publish_tf_requires_bool(self):
        for value in (None,0,1,'false','true'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'布尔值'):
                web.M3TExperiment(None,'tracker','runs','cache',publish_tf=value)


class BimanualExperimentTests(unittest.TestCase):
    def test_explicit_tracking_tuning_reaches_config_provenance_launch_and_status(self):
        with tempfile.TemporaryDirectory() as folder:
            p=bimanual_preview(folder)
            manager,_,_=start_bimanual(p,folder,correspondence_iterations=11,
                modeled_occlusions=True)
            config=json.loads((manager.run/'tracking.json').read_text())
            self.assertEqual(config['correspondence_iterations'],11)
            self.assertIs(config['occlusion']['modeled'],True)
            init=json.loads((manager.run/'initialization.json').read_text())
            launch=json.loads((manager.run/'launch.json').read_text())
            for data in (init['provenance'],launch,manager.state()):
                self.assertEqual(data['correspondence_iterations'],11)
                self.assertIs(data['modeled_occlusions'],True)

    def test_one_process_uses_both_fresh_seeds_and_twelve_priors_without_changing_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            p=bimanual_preview(folder)
            original=Path(p.dashboard.args.robot_model).read_bytes()
            manager,process,copy_mesh=start_bimanual(p,folder)
            process.assert_called_once();copy_mesh.assert_called_once()
            argv=process.call_args.args[0]
            self.assertEqual(argv[argv.index('--component-id')+1],'m3t-bimanual-web-experiment')
            self.assertTrue(manager.packet_path.parent.name.startswith('rbt-m3t-bimanual-'))
            self.assertEqual(Path(p.dashboard.args.robot_model).read_bytes(),original)
            init=json.loads((manager.run/'initialization.json').read_text())
            self.assertEqual([a['name'] for a in init['arms']],['so101_left','so101_right'])
            self.assertEqual(init['provenance']['pitch_deg'],-61.)
            self.assertEqual(init['provenance']['mount_source'],p.mount_source)
            self.assertNotIn('legacy_binding_audit_sha256',init['provenance'])
            self.assertNotIn('mount_audit_sha256',init['provenance'])
            for i,arm in enumerate(init['arms']):
                name=arm['name'];child=p.arm_previews[name]
                self.assertTrue(arm['fixed_base'])
                np.testing.assert_array_equal(arm['base_in_camera'],child.B.reshape(-1))
                source=init['provenance']['arms_source'][name]
                self.assertEqual(source['mount_source'],child.mount_source)
                self.assertNotIn('legacy_binding_audit_sha256',source)
                self.assertNotIn('mount_audit_sha256',source)
                np.testing.assert_allclose([arm['joint_angles'][n] for n in web.NAMES],
                    np.deg2rad([10+10*i+.1,20+10*i+.1,30+10*i+.1,40+10*i+.1,50+10*i+.1]))
                c=child.renderer.gripper_calibration
                self.assertAlmostEqual(arm['joint_angles']['gripper'],web.gripper_angle_from_percent(25+25*i+.1,c))
                self.assertEqual(source['gripper_seed']['calibration_sha256'],c['calibration_sha256'])
                self.assertEqual(source['joint_before']['source'],'actual readonly '+name)
                self.assertEqual(source['joint_after']['source'],'actual readonly '+name)
                self.assertEqual(child.dashboard.read_arm.call_count,2)
                child.check_sources.assert_called_once();child.read_pitch.assert_called_once()
            config=json.loads((manager.run/'tracking.json').read_text())
            self.assertEqual({k:v for k,v in config.items() if k!='joint_priors'},
                {'schema_version':1,'robot_model':'robot.json','initialization':'initialization.json',
                 'cache_directory':web.os.path.relpath(Path(folder)/'cache',manager.run),
                 'model_divides':2,'model_points':200,'model_image_size':512,
                 'correspondence_iterations':3,'update_iterations':2,
                 'modalities':{'region':True,'depth':True,'texture':False},
                 'occlusion':{'measured':True,'modeled':False,'region_checking':False,'image_size':512}})
            priors=config['joint_priors']
            self.assertEqual(len(priors['mappings']),12)
            self.assertEqual({k:v for k,v in priors.items() if k!='mappings'},
                {'enabled':True,'service_name':'stream','controller_id':'real','required':True,
                 'initialize_from_measurement':True,'time_basis':'received','maximum_age_ms':100,'capacity':128})
            expected=[m for arm in ('so101_left','so101_right') for m in web.prior_mappings(arm,p.arm_previews[arm].renderer.gripper_calibration)]
            self.assertEqual(priors['mappings'],expected)
            audit=json.loads((manager.run/'runtime-model-audit.json').read_text())
            self.assertEqual(audit['per_arm_gripper_calibration'],manager.gripper_calibrations)
            self.assertEqual(set(audit['per_arm_calibration_sha256']),set(p.arm_previews))
            self.assertEqual(set(audit['per_arm_source_robot_json_sha256'].values()),{web.sha(p.dashboard.args.robot_model)})
            self.assertEqual(audit['runtime_limits_rad']['lower'],-.1)
            self.assertEqual(audit['runtime_limits_rad']['upper'],manager.gripper_calibrations['so101_right']['open_rad'])
            runtime=json.loads((manager.run/'robot.json').read_text())
            self.assertEqual(runtime['joints'][0]['upper'],audit['runtime_limits_rad']['upper'])
            self.assertEqual(manager.gripper_calibration,manager.gripper_calibrations['so101_left'])
            launch=json.loads((manager.run/'launch.json').read_text())
            for state in (launch,manager.state()):
                self.assertEqual(state['joint_priors_count'],12)
                self.assertTrue(state['fixed_base'])
                self.assertFalse(state['solver_accepted'])
                self.assertEqual(state['mount_source'],p.mount_source)
                self.assertEqual(set(state['per_arm_gripper_prior_provenance']),set(p.arm_previews))
                for arm,c in manager.gripper_calibrations.items():
                    self.assertEqual(state['per_arm_gripper_prior_provenance'][arm]['calibration_sha256'],c['calibration_sha256'])

    def test_default_tracking_tuning_remains_legacy_compatible(self):
        manager=web.M3TExperiment(None,'tracker','runs','cache')
        self.assertEqual(manager.correspondence_iterations,3)
        self.assertIs(manager.modeled_occlusions,False)

    def test_correspondence_iterations_range_is_rejected(self):
        for value in (0,21,True,3.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                web.M3TExperiment(None,'tracker','runs','cache',
                    correspondence_iterations=value)

    def test_moving_right_feedback_and_different_canonical_models_prevent_launch(self):
        for defect in ('moving_right','different_model','wrong_identity','different_pitch','invalid_right_calibration'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as folder:
                p=bimanual_preview(folder);right=p.arm_previews['so101_right']
                if defect=='moving_right':
                    right.dashboard.read_arm=Mock(side_effect=[
                        {'degrees_and_gripper_percent':[20,30,40,50,60,50]},
                        {'degrees_and_gripper_percent':[21,30,40,50,60,50]}])
                elif defect=='different_model':
                    path=Path(folder)/'other.json';path.write_text('{}')
                    right.dashboard.args.robot_model=str(path)
                elif defect=='wrong_identity':right.dashboard.args.arm='so101_left'
                elif defect=='different_pitch':right.read_pitch.return_value=-60.
                else:right.renderer.gripper_calibration['scale_rad_per_percent']=0.
                command=Path(folder)/'fake-m3t';command.write_text('test only');command.chmod(0o700)
                manager=web.M3TExperiment(p,command,Path(folder)/'runs',Path(folder)/'cache')
                with patch.object(web.subprocess,'Popen') as process,patch.object(web.time,'sleep'), \
                        patch.object(web,'fk',return_value=np.eye(4)),self.assertRaises(ValueError):
                    manager.start()
                process.assert_not_called()
                self.assertFalse((Path(folder)/'runs').exists())

    def test_dual_read_frame_requires_exact_twelve_priors_and_current_calibrations(self):
        with tempfile.TemporaryDirectory() as folder:
            p=bimanual_preview(folder);manager,_,_=start_bimanual(p,folder)
            packet=PacketTests();packet.setUp();estimate=packet.dual_estimate()
            prior={'status':'applied','prior_count':12,'controller_id':'real','time_basis':'received',
                'age_ns':10_000_000,'priors':[{'arm':a,'joint':n} for a in p.arm_previews for n in web.JOINT_OUTPUTS]}
            estimate['joint_priors']=prior
            manager.packet_path.write_bytes(packet.packet(estimate))
            with patch.object(web.time,'monotonic_ns',return_value=packet.now):
                actual,image,poses=manager.read_frame()
            self.assertEqual(actual,estimate)
            np.testing.assert_array_equal(image,packet.image)
            self.assertEqual(set(poses),set(p.arm_previews))
            bad_priors=[
                {**prior,'prior_count':6}, {**prior,'prior_count':True},
                {**prior,'priors':prior['priors'][:-1]},
                {**prior,'priors':[dict(e,arm='so101_left') for e in prior['priors']]},
                {**prior,'priors':prior['priors'][:-1]+[prior['priors'][0]]},
                {**prior,'priors':prior['priors'][:-1]+[{'arm':'other','joint':'gripper'}]},
                {**prior,'priors':prior['priors'][:-1]+[{'arm':[],'joint':'gripper'}]},
                {**prior,'controller_id':'other'}, {**prior,'status':'missing'},
                {**prior,'time_basis':'capture'}, {**prior,'age_ns':100_000_001},
            ]
            for bad in bad_priors:
                estimate['joint_priors']=bad;manager.packet_path.write_bytes(packet.packet(estimate))
                with self.subTest(prior=bad),patch.object(web.time,'monotonic_ns',return_value=packet.now),self.assertRaises(ValueError):
                    manager.read_frame()
            estimate['joint_priors']=prior;manager.packet_path.write_bytes(packet.packet(estimate))
            right=p.arm_previews['so101_right']
            right.renderer.gripper_calibration['open_rad']+=.1
            with self.assertRaisesRegex(ValueError,'标定来源已变化'):manager.read_frame()
            right.renderer.gripper_calibration=copy.deepcopy(manager.gripper_calibrations['so101_right'])
            Path(right.dashboard.args.calibration).write_text('changed')
            with self.assertRaisesRegex(ValueError,'标定来源已变化'):manager.read_frame()

    def test_single_nominal_without_legacy_audit_preserves_single_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            dual=bimanual_preview(folder);p=dual.arm_previews['so101_left']
            p.pitch_source='assumed_fixed';p.expected_pitch=-61.;p.lock=threading.Lock();p.event=threading.Event()
            manager,process,_=start_bimanual(p,folder)
            argv=process.call_args.args[0]
            self.assertEqual(argv[argv.index('--component-id')+1],'m3t-left-web-experiment')
            self.assertTrue(manager.packet_path.parent.name.startswith('rbt-m3t-left-'))
            init=json.loads((manager.run/'initialization.json').read_text())
            self.assertEqual(len(init['arms']),1)
            self.assertIn('joint_before',init['provenance'])
            self.assertIn('gripper_seed',init['provenance'])
            self.assertNotIn('arms_source',init['provenance'])
            self.assertNotIn('legacy_binding_audit_sha256',init['provenance'])
            self.assertNotIn('mount_audit_sha256',init['provenance'])
            self.assertEqual(manager.state()['joint_priors_count'],6)
            self.assertNotIn('per_arm_gripper_prior_provenance',manager.state())



class MountRefinementTests(unittest.TestCase):
    def test_diagnostics_use_model_root_and_reject_unknown_arm_sets(self):
        with tempfile.TemporaryDirectory() as folder:
            manager,_,_=start_bimanual(self.preview(folder),folder,mount_refinement=True)
            packet=PacketTests();packet.setUp();estimate=self.exact_estimate(packet,manager)
            for link in estimate['links']:
                if link['link']=='base_link':link['link']='physical_root'
            web.mount_refinement_diagnostics(estimate,manager.nominal_bases,'physical_root')
            for field in ('links','joint_angles_rad'):
                invalid=copy.deepcopy(estimate)
                if field=='links':invalid[field].append({'arm':'other','link':'physical_root'})
                else:invalid[field]['other']={}
                with self.subTest(field=field),self.assertRaises(ValueError):
                    web.mount_refinement_diagnostics(invalid,manager.nominal_bases,'physical_root')

    def preview(self,folder):
        p=bimanual_preview(folder)
        p.dashboard.args.m3t_prior_service='stream'
        return p

    def exact_estimate(self,packet,manager):
        estimate=packet.dual_estimate()
        entries=[]
        for arm,nominal in manager.nominal_bases.items():
            refined=nominal.copy();refined[0,3]+=.004
            angle=.025;c=web.math.cos(angle);s=web.math.sin(angle)
            refined[:3,:3]=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]]) @ nominal[:3,:3]
            estimate['links'].append({'arm':arm,'link':'base_link','pose':refined.reshape(-1).tolist()})
            entries.append({'arm':arm,'nominal_base_in_camera':nominal.reshape(-1).tolist(),
                'refined_base_in_camera':refined.reshape(-1).tolist(),
                'translation_correction_m':.004,'rotation_correction_rad':angle})
        estimate['mount_refinement']={'enabled':True,'joints_locked_to_feedback':True,'arms':entries}
        estimate['joint_priors']={'status':'applied','prior_count':12,'controller_id':'real',
            'time_basis':'received','age_ns':10_000_000,
            'priors':[{'arm':arm,'joint':joint,'angle_rad':value}
                for arm,angles in estimate['joint_angles_rad'].items() for joint,value in angles.items()]}
        return estimate

    def test_soft_bases_complete_config_and_auditable_mode_keep_twelve_fresh_priors(self):
        for options in ({},{'mount_translation_std_m':.03,'mount_rotation_std_deg':7.}):
            with self.subTest(options=options),tempfile.TemporaryDirectory() as folder:
                p=self.preview(folder)
                manager,process,_=start_bimanual(p,folder,mount_refinement=True,**options)
                process.assert_called_once()
                init=json.loads((manager.run/'initialization.json').read_text())
                config=json.loads((manager.run/'tracking.json').read_text())
                expected={'enabled':True,'translation_standard_deviation_m':options.get('mount_translation_std_m',.02),
                    'rotation_standard_deviation_rad':np.deg2rad(options.get('mount_rotation_std_deg',5.))}
                self.assertEqual(config['mount_refinement'],expected)
                self.assertNotIn('visual_link_weights',config)
                self.assertEqual(len(init['arms']),2)
                for seed in init['arms']:
                    self.assertIs(seed['fixed_base'],False)
                    np.testing.assert_array_equal(seed['base_in_camera'],p.arm_previews[seed['name']].B.reshape(-1))
                    self.assertEqual(set(seed['joint_angles']),set(web.JOINT_OUTPUTS))
                priors=config['joint_priors']
                self.assertEqual(len(priors['mappings']),12)
                self.assertIs(priors['required'],True)
                self.assertIs(priors['initialize_from_measurement'],True)
                self.assertEqual(priors['maximum_age_ms'],100)
                self.assertEqual({(m['arm'],m['joint']) for m in priors['mappings']},
                    {(arm,joint) for arm in p.arm_previews for joint in web.JOINT_OUTPUTS})
                launch=json.loads((manager.run/'launch.json').read_text())
                for audit in (init['provenance'],launch,manager.state()):
                    self.assertIs(audit['fixed_base'],False)
                    self.assertIs(audit['end_effector_priority'],False)
                    self.assertEqual(audit['visual_link_weights'],{})
                    self.assertEqual(audit['mount_refinement'],expected)
                    self.assertEqual(audit['mode_description'],'CAD 底座软约束 · 编码器关节固定')
                    self.assertIs(audit['joints_locked_to_feedback'],True)
                self.assertIs(launch['motor_commands'],False)
                for child in p.arm_previews.values():self.assertEqual(child.dashboard.read_arm.call_count,2)

    def test_mode_needs_nominal_fixed_pitch_and_shared_priors_before_feedback(self):
        for missing in ('nominal','pitch','feedback','service'):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as folder:
                p=self.preview(folder)
                if missing=='nominal':p.mount_source['nominal']=False
                if missing=='pitch':p.pitch_source='live_controller'
                if missing=='feedback':p.dashboard.args.m3t_prior_service='different-stream'
                with patch.object(web.subprocess,'Popen') as process,self.assertRaises(ValueError):
                    start_bimanual(p,folder,prior_service=None if missing=='service' else 'stream',mount_refinement=True)
                process.assert_not_called()
                for child in p.arm_previews.values():child.dashboard.read_arm.assert_not_called()

    def test_exact_packet_accepts_refined_bases_and_all_feedback_locked_angles(self):
        with tempfile.TemporaryDirectory() as folder:
            manager,_,_=start_bimanual(self.preview(folder),folder,mount_refinement=True)
            packet=PacketTests();packet.setUp();estimate=self.exact_estimate(packet,manager)
            # Native floating point noise within 1e-4 rad is allowed for every joint.
            for angles in estimate['joint_angles_rad'].values():
                for name in angles:angles[name]+=.00009
            manager.packet_path.write_bytes(packet.packet(estimate))
            with patch.object(web.time,'monotonic_ns',return_value=packet.now):
                actual,image,poses=manager.read_frame()
            self.assertEqual(actual,estimate)
            np.testing.assert_array_equal(image,packet.image)
            self.assertEqual(set(poses),set(manager.nominal_bases))

    def test_magnitudes_match_float_poses_and_reject_zero_or_wrong_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.preview(folder)
            # A rotated CAD frame ensures corrections use the relative rotation.
            angle=.4;c=web.math.cos(angle);s=web.math.sin(angle)
            for child in p.arm_previews.values():
                child.B[:3,:3]=[[1.,0.,0.],[0.,c,-s],[0.,s,c]]
            manager,_,_=start_bimanual(p,folder,mount_refinement=True)
            packet=PacketTests();packet.setUp();original=self.exact_estimate(packet,manager)
            for entry in original['mount_refinement']['arms']:
                for key in ('nominal_base_in_camera','refined_base_in_camera'):
                    entry[key]=np.asarray(entry[key],dtype=np.float32).astype(float).tolist()
                for link in original['links']:
                    if link['arm']==entry['arm'] and link['link']=='base_link':
                        link['pose']=entry['refined_base_in_camera'][:]
            manager.packet_path.write_bytes(packet.packet(original))
            with patch.object(web.time,'monotonic_ns',return_value=packet.now):manager.read_frame()
            for index in range(2):
                for key in ('translation_correction_m','rotation_correction_rad'):
                    for reported in (0.,original['mount_refinement']['arms'][index][key]+.001):
                        estimate=copy.deepcopy(original)
                        estimate['mount_refinement']['arms'][index][key]=reported
                        manager.packet_path.write_bytes(packet.packet(estimate))
                        with self.subTest(arm=index,key=key,reported=reported), \
                                patch.object(web.time,'monotonic_ns',return_value=packet.now),self.assertRaises(ValueError):
                            manager.read_frame()

    def test_rejects_old_or_malformed_native_diagnostics_and_any_joint_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            manager,_,_=start_bimanual(self.preview(folder),folder,mount_refinement=True)
            packet=PacketTests();packet.setUp();original=self.exact_estimate(packet,manager)
            edits=[
                lambda e:e.pop('mount_refinement'),
                lambda e:e.update(mount_refinement=None),
                lambda e:e.update(joint_priors=None),
                lambda e:e['joint_priors'].update(priors=None),
                lambda e:e['joint_priors']['priors'].__setitem__(0,None),
                lambda e:e['mount_refinement'].pop('joints_locked_to_feedback'),
                lambda e:e['mount_refinement'].update(enabled=False),
                lambda e:e['mount_refinement'].update(joints_locked_to_feedback=1),
                lambda e:e['mount_refinement']['arms'].pop(),
                lambda e:e['mount_refinement']['arms'][0].update(arm='other'),
                lambda e:e['mount_refinement']['arms'][1].update(arm='so101_left'),
                lambda e:e['links'].pop(),
            ]
            for key in ('nominal_base_in_camera','refined_base_in_camera'):
                for value in (None,[0.]*16,[0.]*15,[float('nan')]*16,[float('inf')]*16):
                    edits.append(lambda e,key=key,value=value:e['mount_refinement']['arms'][0].update({key:value}))
                edits.append(lambda e,key=key:e['mount_refinement']['arms'][0][key].__setitem__(3,.9))
            for key in ('translation_correction_m','rotation_correction_rad'):
                for value in (None,float('nan'),float('inf'),-.1,True):
                    edits.append(lambda e,key=key,value=value:e['mount_refinement']['arms'][0].update({key:value}))
            for i in range(12):
                for value in (None,float('nan'),float('inf'),True):
                    edits.append(lambda e,i=i,value=value:e['joint_priors']['priors'][i].update(angle_rad=value))
                arm,joint=list((arm,joint) for arm in manager.nominal_bases for joint in web.JOINT_OUTPUTS)[i]
                edits.append(lambda e,arm=arm,joint=joint:e['joint_angles_rad'][arm].update(
                    {joint:e['joint_angles_rad'][arm][joint]+.00011}))
            for i,edit in enumerate(edits):
                estimate=copy.deepcopy(original);edit(estimate)
                manager.packet_path.write_bytes(packet.packet(estimate))
                with self.subTest(case=i),patch.object(web.time,'monotonic_ns',return_value=packet.now),self.assertRaises(ValueError):
                    manager.read_frame()

    def priority_preview(self,folder):
        p=self.preview(folder)
        path=Path(p.dashboard.args.robot_model);model=json.loads(path.read_text())
        for name in ('gripper_link','moving_jaw_so101_v1_link','soft_fin_link'):
            model['links'].append({'name':name,'mesh':str(Path(folder)/'canonical.obj')})
        path.write_text(json.dumps(model))
        return p

    def test_end_effector_priority_config_preserves_geometry_and_records_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.priority_preview(folder)
            source=Path(p.dashboard.args.robot_model).read_bytes()
            manager,process,_=start_bimanual(p,folder,mount_refinement=True,end_effector_priority=True)
            process.assert_called_once()
            expected={'gripper_link':5.0,'moving_jaw_so101_v1_link':5.0}
            tracking=json.loads((manager.run/'tracking.json').read_text())
            self.assertEqual(tracking['visual_link_weights'],expected)
            self.assertEqual(len(tracking['joint_priors']['mappings']),12)
            runtime=json.loads((manager.run/'robot.json').read_text())
            original=json.loads(source)
            self.assertEqual([l['name'] for l in runtime['links']],[l['name'] for l in original['links']])
            self.assertTrue(all(l['mesh'] for l in runtime['links']))
            self.assertNotIn('soft_fin_link',tracking['visual_link_weights'])
            self.assertEqual(Path(p.dashboard.args.robot_model).read_bytes(),source)
            init=json.loads((manager.run/'initialization.json').read_text())
            launch=json.loads((manager.run/'launch.json').read_text())
            for audit in (init['provenance'],launch,manager.state()):
                self.assertIs(audit['end_effector_priority'],True)
                self.assertEqual(audit['visual_link_weights'],expected)
                self.assertIn('夹爪刚性结构优先',audit['mode_description'])
                self.assertIs(audit['fixed_base'],False)

    def test_end_effector_priority_requires_mount_and_both_mesh_links(self):
        with self.assertRaisesRegex(ValueError,'mount refinement'):
            web.M3TExperiment(None,'tracker','runs','cache',end_effector_priority=True)
        for name in ('gripper_link','moving_jaw_so101_v1_link'):
            for defect in ('missing_link','empty_mesh','missing_mesh_file'):
                with self.subTest(name=name,defect=defect),tempfile.TemporaryDirectory() as folder:
                    p=self.priority_preview(folder)
                    path=Path(p.dashboard.args.robot_model);model=json.loads(path.read_text())
                    if defect=='missing_link':model['links']=[l for l in model['links'] if l['name']!=name]
                    else:
                        link=next(l for l in model['links'] if l['name']==name)
                        link['mesh']='' if defect=='empty_mesh' else str(Path(folder)/'missing.obj')
                    path.write_text(json.dumps(model))
                    with patch.object(web.subprocess,'Popen') as process,self.assertRaisesRegex(ValueError,'mesh'):
                        start_bimanual(p,folder,mount_refinement=True,end_effector_priority=True)
                    process.assert_not_called()
                    self.assertFalse((Path(folder)/'runs').exists())
                    for child in p.arm_previews.values():child.dashboard.read_arm.assert_not_called()

    def test_end_effector_priority_requires_exact_native_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            manager,_,_=start_bimanual(self.priority_preview(folder),folder,
                mount_refinement=True,end_effector_priority=True)
            packet=PacketTests();packet.setUp();estimate=self.exact_estimate(packet,manager)
            bad_weights=[None,{},[],{'gripper_link':5.0},
                {'gripper_link':1.0,'moving_jaw_so101_v1_link':5.0},
                {'gripper_link':5.0,'moving_jaw_so101_v1_link':float('nan')},
                {'gripper_link':5.0,'moving_jaw_so101_v1_link':float('inf')},
                {'gripper_link':5.0,'moving_jaw_so101_v1_link':5.0,'soft_fin_link':1.0}]
            for value in ['missing']+bad_weights:
                invalid=copy.deepcopy(estimate)
                if value!='missing':invalid['visual_link_weights']=value
                manager.packet_path.write_bytes(packet.packet(invalid))
                with self.subTest(weights=value),patch.object(web.time,'monotonic_ns',return_value=packet.now), \
                        self.assertRaisesRegex(ValueError,'权重'):
                    manager.read_frame()
            estimate['visual_link_weights']={'gripper_link':5.0,'moving_jaw_so101_v1_link':5.0}
            manager.packet_path.write_bytes(packet.packet(estimate))
            with patch.object(web.time,'monotonic_ns',return_value=packet.now):
                actual,image,_=manager.read_frame()
            self.assertEqual(actual,estimate)
            np.testing.assert_array_equal(image,packet.image)

if __name__=='__main__':unittest.main()
