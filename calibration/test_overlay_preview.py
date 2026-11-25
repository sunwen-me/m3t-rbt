"""Verification preview provenance, pitch guard and stale-frame checks."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import overlay_preview as web


class PreviewTests(unittest.TestCase):
    def preview(self,folder):
        root=Path(folder);audit=root/'audit.json';report=root/'report.json';config=root/'robot.yaml'
        config.write_text('test config')
        report.write_text(json.dumps({'arms':{'so101_left':{'candidate':{'T_camera_from_base':np.eye(4).tolist(),'T_gripper_from_marker':np.eye(4).tolist()}}}}))
        audit.write_text(json.dumps({'model_urdf_sha256':web.MODEL_SHA,'source_report_sha256':web.sha(report),
            'config_after_sha256':web.sha(config),'T_camera_from_newbase':np.eye(4).tolist(),'capture_pitch_deg_assumed':-61}))
        calibration=root/'v2.json'
        calibration.write_text(json.dumps({'gripper':{'id':6,'range_min':2025,'range_max':3460,'drive_mode':0}}))
        d=SimpleNamespace(args=SimpleNamespace(arm='so101_left',robot_model='unused',calibration=str(calibration)),check_binding=mock.Mock())
        with mock.patch.object(web,'ModelOverlay'),mock.patch.object(web.threading.Thread,'start'):
            return web.OverlayPreview(d,audit,report,config,'http://127.0.0.1/status')

    def test_changed_static_config_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.preview(folder);p.check_sources();p.static_config.write_text('changed')
            with self.assertRaisesRegex(ValueError,'配置已变化'):p.check_sources()

    def test_wrong_pitch_or_offline_refused_without_motor_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.preview(folder)
            for data in ({'online':True,'angle':-30},{'online':False,'angle':-61}):
                response=mock.MagicMock();response.__enter__.return_value.read.return_value=json.dumps(data).encode()
                with mock.patch.object(web.urllib.request,'urlopen',return_value=response) as request:
                    with self.assertRaises(ValueError):p.read_pitch()
                    self.assertEqual(request.call_args.args[0],'http://127.0.0.1/status')
            response.__enter__.return_value.read.return_value=b'{"online":true,"angle":-60.99}'
            with mock.patch.object(web.urllib.request,'urlopen',return_value=response):self.assertAlmostEqual(p.read_pitch(),-60.99)

    def test_stale_or_disabled_frame_not_exposed(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.preview(folder);p.enabled=True;p.sample={'created':10.,'receipt':{}}
            with mock.patch.object(web.time,'monotonic',return_value=10.5):self.assertTrue(p.snapshot()[1]['ready'])
            with mock.patch.object(web.time,'monotonic',return_value=13.):
                self.assertIsNone(p.snapshot()[0]);self.assertFalse(p.snapshot()[1]['ready'])
            p.set_enabled(False);self.assertIsNone(p.snapshot()[0])

    def test_layer_updates_validate_atomically_and_are_reported_as_copies(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.preview(folder)
            original={'encoder':True,'m3t':True,'predicted':True,'observed':True}
            p.sample={'created':10.,'receipt':{}}
            p.error='old error'
            p.event.clear()
            invalid=(None,{}, {'encoder':True,'m3t':True,'predicted':True},
                     {**original,'unknown':True}, {**original,'m3t':1},
                     {**original,'observed':np.bool_(False)})
            for layers in invalid:
                with self.subTest(layers=layers), self.assertRaises(ValueError):p.set_layers(layers)
                self.assertEqual(p.layers,original)
                self.assertIsNotNone(p.sample)
                self.assertEqual(p.error,'old error')
                self.assertFalse(p.event.is_set())

            updated={'encoder':False,'m3t':True,'predicted':False,'observed':True}
            result=p.set_layers(updated)
            self.assertIn('message',result)
            self.assertEqual(p.layers,updated)
            self.assertIsNone(p.sample)
            self.assertIsNone(p.error)
            self.assertTrue(p.event.is_set())
            state=p.snapshot()[1]
            self.assertEqual(state['layers'],updated)
            state['layers']['encoder']=True
            self.assertEqual(p.layers,updated)

    def test_all_hidden_layers_return_raw_image_without_changing_marker_error(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.preview(folder)
            raw=np.zeros((100,100,3),dtype=np.uint8)
            before={'read_completed':2.0,'degrees_and_gripper_percent':[0,0,0,0,0,25]}
            after={'read_started':2.2,'degrees_and_gripper_percent':[0,0,0,0,0,25]}
            packet={'reference':{'capture_ns':2_100_000_000}}
            d=p.dashboard
            d.model={}
            d.main_config={'K':np.array([[100.,0,50.],[0,100.,50.],[0,0,1.]])}
            d.read_arm=mock.Mock(side_effect=[before,after,before,after])
            d.main_frame=mock.Mock(return_value=(packet,raw.copy()))
            p.check_sources=mock.Mock()
            p.read_pitch=mock.Mock(return_value=-61.)
            p.renderer.render.side_effect=lambda image,*args,**kwargs:(image.copy(),{'rendered':'encoder'})
            predicted=np.eye(4);predicted[2,3]=1.
            detected=np.array([[48.,48.],[52.,48.],[52.,52.],[48.,52.]])
            with mock.patch.object(web,'fk',return_value=predicted), \
                    mock.patch('aruco_web.detect_marker',return_value=detected):
                visible=p.acquire()
                hidden_layers={name:False for name in ('encoder','m3t','predicted','observed')}
                p.set_layers(hidden_layers)
                hidden=p.acquire()

            self.assertFalse(np.array_equal(visible['overlay'],visible['raw']))
            np.testing.assert_array_equal(hidden['overlay'],hidden['raw'])
            self.assertEqual(hidden['receipt']['marker_rms_px'],visible['receipt']['marker_rms_px'])
            self.assertEqual(hidden['receipt']['display_layers'],hidden_layers)
            self.assertEqual(hidden['receipt']['renderer'],{'rendered':'encoder'})
            self.assertEqual(p.renderer.render.call_count,2)

    def test_save_refuses_stale_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.preview(folder)
            with self.assertRaisesRegex(ValueError,'新鲜'):p.save()

def nominal_preview(folder, assumed=-61.0, edit=None, legacy_edit=None, model_edit=None):
    """Local source-bound fixture; never starts the overlay thread."""
    legacy=PreviewTests().preview(folder)
    root=Path(folder)
    model={'schema_version':1,'name':'fixture','root_link':'base_link',
        'source_sha256':web.MODEL_SHA,'source_urdf':str(root/'official.urdf'),
        'links':[{'name':'base_link','mesh':''}],
        'joints':[{'name':'gripper','type':'revolute','lower':0.0,'upper':1.0}]}
    if model_edit:model_edit(model)
    model_path=root/'robot.json';model_path.write_text(json.dumps(model))
    prior=json.loads((Path(__file__).parent/'fixed_pitch61_nominal_mounts.json').read_text())
    prior['source']['urdf']['sha256']=web.MODEL_SHA
    prior['source']['static_config']['sha256_as_recorded']=web.sha(legacy.static_config)
    prior['source']['model_json']['sha256']=web.sha(model_path)
    if edit: edit(prior)
    path=root/'prior.json';path.write_text(json.dumps(prior))
    if legacy_edit: legacy_edit(legacy)
    d=legacy.dashboard;d.args.robot_model=str(model_path);d.model=model
    d.main_config={'camera_serial':'CP15641000AT','K':np.eye(3),'image_size':[100,100]}
    with mock.patch.object(web,'ModelOverlay'),mock.patch.object(web.threading.Thread,'start'):
        return web.OverlayPreview(d,legacy.audit_path,legacy.report_path,legacy.static_config,
            'http://127.0.0.1/status',nominal_mount_prior=path,fixed_pitch_assumption_deg=assumed)


class NominalPreviewTests(unittest.TestCase):
    def test_selects_named_B_and_fixed_assumption_without_HTTP(self):
        with tempfile.TemporaryDirectory() as folder:
            p=nominal_preview(folder,edit=lambda prior:prior['arms'].reverse())
            prior=json.loads((Path(folder)/'prior.json').read_text())
            expected=next(a['pose4x4'] for a in prior['arms'] if a['name']=='so101_left')
            np.testing.assert_array_equal(p.B,expected)
            self.assertIsNone(p.X)
            with mock.patch.object(web.urllib.request,'urlopen') as request:
                self.assertEqual(p.read_pitch(),-61.0)
                request.assert_not_called()
            state=p.snapshot()[1]
            self.assertEqual(state['pitch_source'],'assumed_fixed')
            self.assertEqual(state['mount_source']['kind'],'nominal_cad')
            self.assertIs(state['mount_source']['solver_accepted'],False)
            self.assertFalse(state['marker_available'])
            self.assertFalse(state['layers']['predicted'])
            self.assertFalse(state['layers']['observed'])
            with self.assertRaisesRegex(ValueError,'贴码图层不可用'):
                p.set_layers({name:True for name in p.layers})
            p.check_sources()
            (Path(folder)/'prior.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'配置已变化'):p.check_sources()

    def test_wrong_pitch_and_nonfinite_assumptions_are_rejected(self):
        for angle in (-60.0,61.0,float('nan'),float('inf'),True):
            with self.subTest(angle=angle),tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(ValueError,'pitch'):
                    nominal_preview(folder,assumed=angle)
        with self.assertRaisesRegex(ValueError,'仅允许'):
            web.OverlayPreview(None,None,None,None,None,fixed_pitch_assumption_deg=-61)

    def test_nominal_without_assumption_still_requires_live_pitch(self):
        with tempfile.TemporaryDirectory() as folder:
            p=nominal_preview(folder,assumed=None)
            response=mock.MagicMock()
            with mock.patch.object(web.urllib.request,'urlopen',return_value=response) as request:
                for angle in (-60.,-61.):
                    response.__enter__.return_value.read.return_value=json.dumps({'online':True,'angle':angle}).encode()
                    if angle==-60.:
                        with self.assertRaises(ValueError):p.read_pitch()
                    else:self.assertEqual(p.read_pitch(),-61.)
                self.assertEqual(request.call_count,2)
            self.assertEqual(p.pitch_source,'live_controller')

    def test_source_identity_and_invalid_pose_are_rejected(self):
        def pose_edit(prior, value):
            prior['arms'][0]['pose4x4']=value.tolist()
            prior['arms'][0]['rowmajor16']=value.reshape(-1).tolist()
        reflection=np.eye(4);reflection[0,0]=-1;reflection[2,3]=1
        behind=np.eye(4);behind[2,3]=-1
        nonfinite=np.eye(4);nonfinite[0,0]=float('nan');nonfinite[2,3]=1
        nonorthogonal=np.diag([1+4e-6,1-4e-6,1,1]);nonorthogonal[2,3]=1
        edits=[
            lambda p:p.update(record_type='runtime-initialization'),
            lambda p:p.update(capture_pitch_deg=float('nan')),
            lambda p:p['frames'].update(camera='wrong'),
            lambda p:p['units'].update(translation='mm'),
            lambda p:p['source']['static_config'].update(sha256_as_recorded='00'*32),
            lambda p:p['source']['urdf'].update(sha256='00'*32),
            lambda p:p['source']['model_json'].update(sha256='00'*32),
            lambda p:p.update(arms=p['arms'][1:]),
            lambda p:p['arms'].append(p['arms'][0]),
            lambda p:p['arms'][0]['rowmajor16'].__setitem__(3,999),
            lambda p:pose_edit(p,reflection),lambda p:pose_edit(p,behind),
            lambda p:pose_edit(p,nonfinite),lambda p:pose_edit(p,nonorthogonal),
        ]
        for i,edit in enumerate(edits):
            with self.subTest(case=i),tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):nominal_preview(folder,edit=edit)

    def test_model_semantics_and_loaded_source_changes_are_rejected(self):
        for edit in (lambda m:m.update(source_sha256='00'*32),lambda m:m.update(root_link='wrong')):
            with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(ValueError,'源模型'):
                nominal_preview(folder,model_edit=edit)
        for filename in ('robot.json','robot.yaml'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as folder:
                p=nominal_preview(folder)
                (Path(folder)/filename).write_text('changed')
                with self.assertRaisesRegex(ValueError,'配置已变化'):p.check_sources()

    def test_legacy_binding_still_checked_before_nominal_override(self):
        def break_audit(p):
            audit=json.loads(p.audit_path.read_text())
            audit['T_camera_from_newbase'][0][3]=1
            p.audit_path.write_text(json.dumps(audit))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'已部署初值与报告不一致'):
                nominal_preview(folder,legacy_edit=break_audit)

    def test_no_old_marker_prediction_or_detection_even_with_behind_fk(self):
        with tempfile.TemporaryDirectory() as folder:
            p=nominal_preview(folder);d=p.dashboard
            before={'read_completed':2.0,'degrees_and_gripper_percent':[1,2,3,4,5,25]}
            after={'read_started':2.2,'degrees_and_gripper_percent':[1,2,3,4,5,25]}
            d.read_arm=mock.Mock(side_effect=[before,after])
            d.main_frame=mock.Mock(return_value=({'reference':{'capture_ns':2_100_000_000}},np.zeros((100,100,3),np.uint8)))
            p.renderer.render.side_effect=lambda image,*args,**kwargs:(image,{'rendered':'encoder'})
            behind=np.eye(4);behind[2,3]=-100
            with mock.patch.object(web,'fk',return_value=behind), \
                    mock.patch('aruco_web.detect_marker') as detector, \
                    mock.patch.object(web.urllib.request,'urlopen') as request:
                sample=p.acquire()
                detector.assert_not_called();request.assert_not_called()
            receipt=sample['receipt']
            self.assertIsNone(receipt['marker_rms_px'])
            self.assertIsNone(receipt['m3t_marker_rms_px'])
            self.assertEqual(receipt['marker_error'],'粗初值未标定贴码外参')
            self.assertNotIn('audit_sha256',receipt)
            self.assertEqual(receipt['legacy_binding_audit_sha256'],web.sha(p.audit_path))
            self.assertEqual(receipt['pitch_source'],'assumed_fixed')
            self.assertIn('not live feedback',receipt['pitch_measurement_semantics'])
            np.testing.assert_array_equal(p.renderer.render.call_args.args[2],p.B)


if __name__=='__main__':unittest.main()
