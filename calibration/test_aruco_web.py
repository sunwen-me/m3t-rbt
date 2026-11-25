"""Marker ambiguity and stationary/fresh capture acceptance checks."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import cv2
import numpy as np
import aruco_web as web


def image(count=1, size=100):
    im=np.full((480,640,3),255,np.uint8)
    marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),1,size)
    for i in range(count): im[80:80+size,60+i*250:60+i*250+size]=marker[:,:,None]
    return im


class GripperPoseResponseTests(unittest.TestCase):
    def overlay(self):
        e={'frame_id':'orbbec335/color_optical','frame_sequence':42,
           'color_capture_ns':123,'capture_clock_kind':'host_monotonic',
           'capture_clock_source':'abcd','capture_clock_generation':1,
           'mount_refinement':{'joints_locked_to_feedback':True},'joint_priors':{'prior_count':12}}
        receipt={'m3t_estimate':e,'m3t_enabled':True,'tcp_poses':{'so101_left':{'pose':1}},
                 'timing_semantics':'software receipt','pitch_deg':-61,'pitch_source':'assumed_fixed'}
        sample={'receipt':receipt};state={'ready':True,'error':None,'age_seconds':.1,'m3t':{'running':True}}
        return SimpleNamespace(snapshot=mock.Mock(return_value=(sample,state))),sample,state

    def test_fresh_pose_has_frame_timing_and_uncalibrated_contact_label(self):
        overlay,sample,state=self.overlay();response=web.gripper_pose_response(overlay)
        self.assertEqual(response['arms'],sample['receipt']['tcp_poses'])
        self.assertFalse(response['physical_contact_tcp_calibrated'])
        self.assertEqual(response['target_link'],'gripper_link')
        self.assertEqual(response['frame_sequence'],42)
        self.assertEqual(response['units']['translation'],'m')
        self.assertTrue(response['joints_locked_to_feedback'])

    def test_soft_base_does_not_imply_encoder_locked_joints(self):
        overlay, sample, _ = self.overlay()
        estimate = sample['receipt']['m3t_estimate']
        estimate['encoder_reference_links'] = True
        estimate['mount_refinement'].update(enabled=True, joints_locked_to_feedback=False)
        response = web.gripper_pose_response(overlay)
        self.assertIs(response['joints_locked_to_feedback'], False)
        self.assertIs(response['encoder_reference_links'], True)

    def test_http_endpoint_returns_json_then_503_when_sample_expires(self):
        import threading
        import urllib.request
        import urllib.error
        overlay,sample,state=self.overlay()
        dashboard=SimpleNamespace(overlay=overlay)
        server=web.Server(('127.0.0.1',0),web.handler(dashboard,'test-only'))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_address[1]}/api/overlay/gripper-poses'
            with urllib.request.urlopen(url,timeout=2) as response:
                self.assertEqual(response.status,200)
                self.assertEqual(json.load(response)['schema'],'rbt-m3t-gripper-poses-v1')
            overlay.snapshot.return_value=(None,state)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url,timeout=2)
            self.assertEqual(error.exception.code,503)
        finally:
            server.shutdown();thread.join(timeout=2);server.server_close()

    def test_stale_stopped_or_encoder_only_never_returns_old_pose(self):
        for case in ('stale','error','stopped','encoder','missing'):
            overlay,sample,state=self.overlay()
            if case=='stale':overlay.snapshot.return_value=(None,state)
            elif case=='error':state['error']='source changed'
            elif case=='stopped':state['m3t']['running']=False
            elif case=='encoder':sample['receipt']['m3t_enabled']=False
            else:del sample['receipt']['tcp_poses']
            with self.subTest(case=case),self.assertRaises(ValueError):
                web.gripper_pose_response(overlay)


class MarkerTests(unittest.TestCase):
    def test_single_marker_and_pose(self):
        obs=web.observation(image(),{'K':[[460,0,320],[0,460,240],[0,0,1]]},.029)
        self.assertLess(obs['pnp_rms_px'],.1)
        self.assertGreater(obs['T_camera_from_marker_seed'][2][3],0)
    def test_duplicate_id_refused(self):
        with self.assertRaisesRegex(ValueError,'多个 ID 1'): web.detect_marker(image(2))
    def test_missing_and_small_refused(self):
        for im in (image(0),image(size=20)):
            with self.assertRaises(ValueError): web.detect_marker(im)


class CaptureTests(unittest.TestCase):
    def dashboard(self,folder, moving=False):
        d=web.Dashboard.__new__(web.Dashboard); d.output=Path(folder)
        d.manifest={'samples':[]}; d.manifest_path=d.output/'manifest.json'
        d.args=SimpleNamespace(arm='so101_left'); d.solving=False
        d.binding={}; d.report=None
        d.check_binding=mock.Mock(); d.main_config={'K':[[460,0,320],[0,460,240],[0,0,1]]}
        d.model={}
        reads=[]
        for i in range(4):
            reads.append({'degrees_and_gripper_percent':[1 if moving and i==1 else 0]*5+[0],
                          'read_started':i*2.,'read_completed':i*2.+.1})
        d.read_arm=mock.Mock(side_effect=reads)
        d.main_frame=mock.Mock(return_value=({'reference':{'capture_ns':5_000_000_000}},image()))
        return d
    def test_stable_capture_persists_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            d=self.dashboard(folder)
            with mock.patch.object(web.time,'sleep'), mock.patch.object(web,'fk',return_value=np.eye(4)):
                d.capture(True)
            saved=json.loads(d.manifest_path.read_text())
            self.assertEqual(len(saved['samples']),1)
            self.assertEqual(len(json.loads((d.output/'pose_0001/provenance.json').read_text())['joint_reads']),4)
            self.assertTrue((d.output/'pose_0001/main.png').exists())
    def test_mid_interval_movement_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            d=self.dashboard(folder,True)
            with mock.patch.object(web.time,'sleep'), self.assertRaisesRegex(ValueError,'还在移动'):
                d.capture(True)
            self.assertFalse(d.manifest_path.exists())
    def test_unconfirmed_fixed_refused_without_hardware_read(self):
        with tempfile.TemporaryDirectory() as folder:
            d=self.dashboard(folder)
            with self.assertRaises(ValueError): d.capture(False)
            d.read_arm.assert_not_called()

class NominalCLITests(unittest.TestCase):
    def arguments(self):
        keys=('main-session','main-packet','robot-model','calibration','output','driver-python','driver-path','arm-port')
        return ['aruco_web.py']+[part for key in keys for part in ('--'+key,'unused')]

    def test_fixed_pitch_requires_nominal_and_nominal_requires_legacy_binding(self):
        for options in (['--fixed-pitch-assumption-deg','-61'],['--nominal-mount-prior','prior.json']):
            with self.subTest(options=options),mock.patch('sys.argv',self.arguments()+options), \
                    mock.patch.object(web,'Dashboard') as dashboard,mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:web.main()
                self.assertEqual(error.exception.code,2)
                dashboard.assert_not_called()

    def test_cli_passes_prior_and_assumption_and_keeps_legacy_default(self):
        for nominal in (True,False):
            options=['--overlay-audit','audit','--overlay-report','report','--overlay-static-config','config']
            if nominal:options+=['--nominal-mount-prior','prior.json','--fixed-pitch-assumption-deg','-61']
            with self.subTest(nominal=nominal),mock.patch('sys.argv',self.arguments()+options), \
                    mock.patch.object(web,'Dashboard') as dashboard,mock.patch.object(web,'Server') as server, \
                    mock.patch('overlay_preview.OverlayPreview') as overlay, \
                    mock.patch.object(web.signal,'signal'),mock.patch('builtins.print'):
                overlay.return_value.m3t=None
                web.main()
                kwargs=overlay.call_args.kwargs
                self.assertEqual(kwargs['nominal_mount_prior'],'prior.json' if nominal else None)
                self.assertEqual(kwargs['fixed_pitch_assumption_deg'],-61.0 if nominal else None)
                self.assertEqual(dashboard.call_args.args[0].arm,'so101_left')
                server.return_value.serve_forever.assert_called_once()

    def test_verify_labels_use_explicit_source_and_disable_unavailable_marker_layers(self):
        self.assertIn('CAD粗初值',web.VERIFY_PAGE)
        self.assertIn("s.pitch_source==='assumed_fixed'",web.VERIFY_PAGE)
        self.assertIn('假设固定',web.VERIFY_PAGE)
        self.assertIn('云台反馈：',web.VERIFY_PAGE)
        self.assertIn('粗初值未标定贴码外参',web.VERIFY_PAGE)
        self.assertIn('last.marker_available===false',web.VERIFY_PAGE)

    def test_verify_frame_keeps_last_loaded_image_when_stale_or_load_fails(self):
        node=shutil.which('node')
        if node is None:self.skipTest('Node.js is unavailable')
        start=web.VERIFY_PAGE.index('function updateFrameAge()')
        end=web.VERIFY_PAGE.index('\nfunction lockLayers',start)
        helpers=web.VERIFY_PAGE[start:end]
        script='''
const assert=require('node:assert/strict');
let visible={id:'frame',alt:'retained',src:'/old.jpg',replaceWith(next){visible=next}};
const age={textContent:''},show={checked:true},images=[];
const $=selector=>selector==='#frame'?visible:selector==='#show'?show:age;
class Image {set src(value){this.url=value;images.push(this)}}
let frameLoading=false,frameLoadedAt=0,frameStale=true,last={ready:true,age_seconds:.2};
'''+helpers+'''
requestFrame('/api/overlay/frame.jpg');
assert.equal(visible.src,'/old.jpg','candidate stays hidden while loading');
images[0].onload();
assert.equal(visible,images[0],'loaded candidate replaces the retained frame');
last.ready=false;retainFrame();
assert.equal(visible,images[0],'not-ready state retains the last successful frame');
assert.match(age.textContent,/可能已过期/);
last.ready=true;requestFrame('/api/overlay/frame.jpg');images[1].onerror();
assert.equal(visible,images[0],'failed candidate does not replace the last successful frame');
assert.match(age.textContent,/可能已过期/);
'''
        subprocess.run([node,'-e',script],check=True,capture_output=True,text=True)




class BimanualCLITests(unittest.TestCase):
    arguments=NominalCLITests.arguments
    def test_dual_requires_right_calibration_and_rbt_feedback(self):
        options=['--dual-arms','--nominal-mount-prior','prior','--overlay-static-config','config']
        for more in ([],['--right-calibration','right'],['--m3t-prior-service','priors']):
            with mock.patch('sys.argv',self.arguments()+options+more),mock.patch.object(web,'Dashboard') as dashboard,mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit):web.main()
                dashboard.assert_not_called()

    def test_dual_creates_two_sources_and_one_tracker(self):
        options=['--dual-arms','--right-calibration','right-v2',
            '--nominal-mount-prior','prior','--overlay-static-config','config',
            '--fixed-pitch-assumption-deg','-61','--m3t-command','tracker','--m3t-run-root','runs',
            '--m3t-cache-directory','cache','--m3t-prior-service','priors']
        left=SimpleNamespace();right=SimpleNamespace()
        with mock.patch('sys.argv',self.arguments()+options),mock.patch.object(web,'Dashboard',side_effect=[left,right]) as dashboard, \
                mock.patch.object(web,'Server'),mock.patch('dual_overlay_preview.BimanualOverlayPreview') as overlay, \
                mock.patch('m3t_preview.M3TExperiment') as tracker,mock.patch.object(web.signal,'signal'),mock.patch('builtins.print'):
            web.main()
        self.assertEqual(dashboard.call_count,2)
        self.assertEqual(dashboard.call_args_list[1].args[0].arm,'so101_right')
        self.assertEqual(dashboard.call_args_list[1].args[0].calibration,'right-v2')
        overlay.assert_called_once();tracker.assert_called_once()

    def test_dual_page_names_both_arms_and_distinct_tracking_colors(self):
        page=web.verification_page(SimpleNamespace(args=SimpleNamespace(dual_arms=True)),'csrf')
        self.assertIn('双臂模型叠加验证',page)
        self.assertIn('橙色：左臂 M3T',page);self.assertIn('紫色：右臂 M3T',page)
        self.assertIn("arm==='so101_left'?'左臂':'右臂'",page)
        self.assertIn('m.joint_priors_count||6',page)


class AttachedDashboardTests(unittest.TestCase):
    arguments=NominalCLITests.arguments

    def options(self):
        return ['--dual-arms','--right-calibration','right-v2','--nominal-mount-prior','prior',
                '--overlay-static-config','config','--fixed-pitch-assumption-deg','-61',
                '--m3t-prior-service','priors','--joint-feedback-controller','controller',
                '--m3t-preview-packet','native/latest.bin','--m3t-publish-tf']

    def test_attached_cli_enables_local_preview_and_only_closes_web_on_shutdown(self):
        left, right = SimpleNamespace(), SimpleNamespace()
        with mock.patch('sys.argv', self.arguments()+self.options()), \
                mock.patch.object(web, 'Dashboard', side_effect=[left, right]), \
                mock.patch.object(web, 'Server') as server, \
                mock.patch('dual_overlay_preview.BimanualOverlayPreview') as overlay, \
                mock.patch('m3t_preview.M3TPacketSubscriber', autospec=True) as subscriber, \
                mock.patch('m3t_preview.M3TExperiment') as owner, \
                mock.patch.object(web.signal, 'signal'), mock.patch('builtins.print'):
            server.return_value.serve_forever.side_effect = KeyboardInterrupt
            web.main()
        subscriber.assert_called_once_with(overlay.return_value, Path('native/latest.bin'),
            prior_service='priors', prior_controller='controller', publish_tf=True)
        self.assertIs(left.overlay.m3t, subscriber.return_value)
        left.overlay.set_enabled.assert_called_once_with(True)
        owner.assert_not_called()
        server.return_value.server_close.assert_called_once()

    def test_attached_cli_refuses_simultaneous_managed_tracking_options(self):
        options = self.options()+['--m3t-command', 'tracker']
        with mock.patch('sys.argv', self.arguments()+options), \
                mock.patch.object(web, 'Dashboard') as dashboard, mock.patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                web.main()
        dashboard.assert_not_called()

    def test_attached_page_has_no_start_or_stop_controls(self):
        args = SimpleNamespace(dual_arms=True, m3t_preview_packet=Path('latest.bin'))
        page = web.verification_page(SimpleNamespace(args=args), 'csrf')
        self.assertIn('attached=true', page)
        self.assertIn('id="save"', page)
        for name in ('start','stop','m3tstart','m3tstop'):
            self.assertNotIn(f'id="{name}"', page)

    def test_attached_http_rejects_lifecycle_requests_with_valid_nonce(self):
        import threading
        import urllib.error
        import urllib.request

        dashboard = SimpleNamespace(args=SimpleNamespace(m3t_preview_packet=Path('latest.bin')),
                                    overlay=mock.Mock(), lock=threading.Lock())
        server = web.Server(('127.0.0.1',0), web.handler(dashboard, 'test-only'))
        origin = f'http://127.0.0.1:{server.server_address[1]}'
        dashboard.args.public_origin = origin
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for action in ('enable','m3t-start','m3t-stop'):
                request = urllib.request.Request(f'{origin}/api/overlay/{action}', b'{"enabled":false}',
                    {'Content-Type':'application/json', 'Origin':origin, 'X-CSRF-Token':'test-only'})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request, timeout=2)
                self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()
        self.assertEqual(dashboard.overlay.mock_calls, [])


class MountRefinementCLITests(unittest.TestCase):
    arguments=NominalCLITests.arguments

    def options(self):
        return ['--nominal-mount-prior','prior','--overlay-static-config','config',
            '--fixed-pitch-assumption-deg','-61','--m3t-prior-service','priors',
            '--m3t-command','tracker',
            '--m3t-run-root','runs','--m3t-cache-directory','cache']

    def test_mode_requires_nominal_mount_finite_pitch_and_rbt_feedback(self):
        for key in ('--nominal-mount-prior','--fixed-pitch-assumption-deg','--m3t-prior-service'):
            options=self.options();index=options.index(key);del options[index:index+2]
            with self.subTest(missing=key),mock.patch('sys.argv',self.arguments()+options+['--m3t-mount-refinement']), \
                    mock.patch.object(web,'Dashboard') as dashboard,mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:web.main()
                self.assertEqual(error.exception.code,2);dashboard.assert_not_called()
        for value in ('nan','inf','-inf'):
            options=self.options();options[options.index('--fixed-pitch-assumption-deg')+1]=value
            with mock.patch('sys.argv',self.arguments()+options+['--m3t-mount-refinement']), \
                    mock.patch.object(web,'Dashboard') as dashboard,mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit):web.main()
                dashboard.assert_not_called()

    def test_rejects_nonfinite_and_out_of_range_sigmas(self):
        for key,values in (('--m3t-mount-translation-std-m',('nan','inf','0.00009','0.20001')),
                           ('--m3t-mount-rotation-std-deg',('nan','inf','0.005','45'))):
            for value in values:
                with self.subTest(key=key,value=value),mock.patch('sys.argv',self.arguments()+self.options()+[
                        '--m3t-mount-refinement',key,value]),mock.patch.object(web,'Dashboard') as dashboard,mock.patch('sys.stderr'):
                    with self.assertRaises(SystemExit):web.main()
                    dashboard.assert_not_called()

    def test_cli_passes_mode_and_default_or_explicit_sigmas(self):
        for mode,extra,translation,rotation in ((False,[],.02,5.),(True,[],.02,5.),
                (True,['--m3t-mount-translation-std-m','.04','--m3t-mount-rotation-std-deg','8'],.04,8.)):
            options=self.options()+extra+(['--m3t-mount-refinement'] if mode else [])
            with self.subTest(mode=mode,extra=extra),mock.patch('sys.argv',self.arguments()+options), \
                    mock.patch.object(web,'Dashboard'),mock.patch.object(web,'Server'), \
                    mock.patch('overlay_preview.OverlayPreview'),mock.patch('m3t_preview.M3TExperiment') as tracker, \
                    mock.patch.object(web.signal,'signal'),mock.patch('builtins.print'):
                web.main()
                self.assertIs(tracker.call_args.kwargs['mount_refinement'],mode)
                self.assertIs(tracker.call_args.kwargs['end_effector_priority'],False)
                self.assertEqual(tracker.call_args.kwargs['mount_translation_std_m'],translation)
                self.assertEqual(tracker.call_args.kwargs['mount_rotation_std_deg'],rotation)

    def test_native_tf_cli_is_opt_in_and_passed_to_tracker(self):
        for enabled in (False,True):
            options=self.options()+(['--m3t-publish-tf'] if enabled else [])
            with self.subTest(enabled=enabled),mock.patch('sys.argv',self.arguments()+options), \
                    mock.patch.object(web,'Dashboard'),mock.patch.object(web,'Server'), \
                    mock.patch('overlay_preview.OverlayPreview'),mock.patch('m3t_preview.M3TExperiment') as tracker, \
                    mock.patch.object(web.signal,'signal'),mock.patch('builtins.print'):
                web.main()
                self.assertIs(tracker.call_args.kwargs['publish_tf'],enabled)

    def test_native_tf_ui_switch_reports_configuration_only(self):
        self.assertIn('id="m3t-tf-configured" disabled> RBT TF（配置启用）',web.VERIFY_PAGE)
        self.assertIn("$('#m3t-tf-configured').checked=m?.native_tf_enabled===true",web.VERIFY_PAGE)
        self.assertNotIn('RBT TF 正在发布',web.VERIFY_PAGE)
        self.assertIn("!m||!m.enabled?'M3T 未开启'",web.VERIFY_PAGE)
        self.assertIn("'M3T 已退出，代码 '+m.exit_code",web.VERIFY_PAGE)

    def test_cli_passes_m3t_tuning_and_rejects_out_of_range_iterations(self):
        options=self.options()+['--m3t-correspondence-iterations','11','--m3t-modeled-occlusions']
        with mock.patch('sys.argv',self.arguments()+options),mock.patch.object(web,'Dashboard'), \
                mock.patch.object(web,'Server'),mock.patch('overlay_preview.OverlayPreview'), \
                mock.patch('m3t_preview.M3TExperiment') as tracker,mock.patch.object(web.signal,'signal'), \
                mock.patch('builtins.print'):
            web.main()
        self.assertEqual(tracker.call_args.kwargs['correspondence_iterations'],11)
        self.assertIs(tracker.call_args.kwargs['modeled_occlusions'],True)
        for value in ('0','21'):
            with self.subTest(value=value),mock.patch('sys.argv',self.arguments()+self.options()+[
                    '--m3t-correspondence-iterations',value]), \
                    mock.patch.object(web,'Dashboard') as dashboard,mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:web.main()
                self.assertEqual(error.exception.code,2)
                dashboard.assert_not_called()

    def test_sigma_boundaries(self):
        from m3t_preview import mount_refinement_config
        for translation in (.0001,.2):
            for radians in (.0001,.785398):
                config=mount_refinement_config(translation,web.math.degrees(radians))
                self.assertAlmostEqual(config['rotation_standard_deviation_rad'],radians)

    def test_ui_uses_mode_for_mount_and_m3t_status_and_keeps_legacy_labels(self):
        self.assertIn('refining=s.m3t?.mount_refinement?.enabled===true',web.VERIFY_PAGE)
        self.assertIn("'M3T · '+trackingModeText(s.m3t)",web.VERIFY_PAGE)
        self.assertIn("'M3T Region + Depth · 原生实时预览 · '+trackingModeText(m)",web.VERIFY_PAGE)
        self.assertIn('m?.joints_locked_to_feedback===true',web.VERIFY_PAGE)
        self.assertIn('关节在实时编码器软先验附近由图像继续优化',web.VERIFY_PAGE)
        self.assertIn('五关节 + 夹爪实时软先验',web.VERIFY_PAGE)

    def test_rendered_mode_labels_distinguish_all_three_tracking_modes(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node.js required to execute page mode labels')
        start = web.VERIFY_PAGE.index('function trackingModeText(')
        end = web.VERIFY_PAGE.index('\nfunction pitchText(', start)
        modes = [
            {'mount_refinement': {'enabled': True}, 'fixed_base': False,
             'joints_locked_to_feedback': False, 'joint_priors_enabled': True},
            {'mount_refinement': {'enabled': True}, 'fixed_base': False,
             'joints_locked_to_feedback': True, 'joint_priors_enabled': True},
            {'fixed_base': True, 'joints_locked_to_feedback': False, 'joint_priors_enabled': True},
            {},
        ]
        script = web.VERIFY_PAGE[start:end] + '\nprocess.stdout.write(JSON.stringify(' + json.dumps(modes) + '.map(trackingModeText)));'
        result = subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout), [
            'CAD 底座软约束 · 五关节 + 夹爪实时软先验',
            'CAD 底座软约束 · 编码器关节固定',
            '底座固定 · 五关节 + 夹爪实时软先验',
            '等待底座状态 · 无实时先验'])

    def test_quality_displays_native_per_arm_corrections_only_for_new_mode(self):
        self.assertIn("m?.mount_refinement?.enabled===true?mountCorrectionText(d.m3t_estimate):d.mount_source?.nominal?",web.VERIFY_PAGE)
        self.assertIn('const report=estimate?.mount_refinement',web.VERIFY_PAGE)
        self.assertIn("report.arms.map(a=>(a.arm==='so101_left'?'左':'右')+' 底座修正 '",web.VERIFY_PAGE)
        self.assertIn("(a.translation_correction_m*1000).toFixed(1)+' mm / '",web.VERIFY_PAGE)
        self.assertIn("(a.rotation_correction_rad*180/Math.PI).toFixed(1)+'°'",web.VERIFY_PAGE)
        self.assertIn("'当前预测码框误差：'+d.marker_rms_px.toFixed(2)+' px'",web.VERIFY_PAGE)

    def test_end_effector_priority_requires_mount_and_is_passed_to_tracker(self):
        with mock.patch('sys.argv',self.arguments()+self.options()+['--m3t-end-effector-priority']), \
                mock.patch.object(web,'Dashboard') as dashboard,mock.patch('sys.stderr'):
            with self.assertRaises(SystemExit) as error:web.main()
            self.assertEqual(error.exception.code,2)
            dashboard.assert_not_called()
        with mock.patch('sys.argv',self.arguments()+self.options()+['--m3t-mount-refinement','--m3t-end-effector-priority']), \
                mock.patch.object(web,'Dashboard'),mock.patch.object(web,'Server'), \
                mock.patch('overlay_preview.OverlayPreview'),mock.patch('m3t_preview.M3TExperiment') as tracker, \
                mock.patch.object(web.signal,'signal'),mock.patch('builtins.print'):
            web.main()
            self.assertIs(tracker.call_args.kwargs['end_effector_priority'],True)
            self.assertIs(tracker.call_args.kwargs['mount_refinement'],True)

    def test_priority_status_preserves_tracking_instructions_and_base_corrections(self):
        self.assertIn("m?.mount_refinement?.enabled===true&&m.end_effector_priority===true",web.VERIFY_PAGE)
        self.assertIn("$('#m3tstatus').textContent+=' · 夹爪刚性结构优先'",web.VERIFY_PAGE)
        self.assertIn('id="tracking-instructions"',web.VERIFY_PAGE)
        self.assertIn('此模式将关节固定为实时 v2 反馈，只优化 CAD 参考附近的底座位姿',web.VERIFY_PAGE)
        self.assertIn('mountCorrectionText(d.m3t_estimate)',web.VERIFY_PAGE)

if __name__=='__main__': unittest.main()
