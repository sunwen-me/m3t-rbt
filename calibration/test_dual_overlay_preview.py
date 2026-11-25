"""Two arms must be rendered against one exact native image/estimate."""
from pathlib import Path
from types import SimpleNamespace
import copy
import tempfile
import threading
import unittest
from unittest import mock
import numpy as np
from dual_overlay_preview import ARMS,BimanualOverlayPreview
from test_overlay_preview import nominal_preview
import overlay_preview


class DualOverlayTests(unittest.TestCase):
    def preview(self,tracked=True):
        p=BimanualOverlayPreview.__new__(BimanualOverlayPreview)
        p.lock=threading.Lock();p.layers={'encoder':False,'m3t':True,'predicted':False,'observed':False}
        p.pitch_source='assumed_fixed';p.expected_pitch=-61.;p.mount_source={'nominal':True}
        p.check_sources=mock.Mock();p.read_pitch=mock.Mock(return_value=-61.)
        im=np.zeros((12,16,3),np.uint8)
        estimate={'color_capture_ns':2_100_000_000,'frame_sequence':7,'joint_priors':{'prior_count':12}}
        poses={arm:{'gripper_link':np.eye(4)*(i+1)} for i,arm in enumerate(ARMS)}
        p.arm_previews={}
        for i,arm in enumerate(ARMS):
            q=[float(i)]*5+[float(i*20)]
            before={'degrees_and_gripper_percent':q,'read_started':1.9,'read_completed':2.}
            after={'degrees_and_gripper_percent':q,'read_started':2.2,'read_completed':2.3}
            d=SimpleNamespace(read_arm=mock.Mock(side_effect=[before,after]),
                main_config={'K':np.eye(3)},main_frame=mock.Mock(return_value=({'reference':{'capture_ns':2_100_000_000}},im)))
            renderer=mock.Mock()
            renderer.render.side_effect=lambda image,*a,**kw:(image.copy(),{})
            renderer.render_poses.side_effect=lambda image,*a,**kw:(image.copy(),{})
            p.arm_previews[arm]=SimpleNamespace(dashboard=d,renderer=renderer,B=np.eye(4),mount_source={'arm':arm})
        p.dashboard=p.arm_previews[ARMS[0]].dashboard
        p.m3t=SimpleNamespace(enabled=True,read_frame=mock.Mock(return_value=(estimate,im,poses)),
            tcp_poses=mock.Mock(return_value={arm:{'sentinel':arm} for arm in ARMS})) if tracked else None
        return p

    def test_both_arms_use_one_native_frame_and_their_own_feedback_and_poses(self):
        p=self.preview();sample=p.acquire();r=sample['receipt']
        self.assertEqual(set(r['arms']),set(ARMS));self.assertEqual(r['m3t_estimate']['frame_sequence'],7)
        self.assertEqual(r['arms'][ARMS[0]]['degrees'],[0.]*5)
        self.assertEqual(r['arms'][ARMS[1]]['degrees'],[1.]*5)
        for i,arm in enumerate(ARMS):
            child=p.arm_previews[arm]
            child.dashboard.main_frame.assert_not_called()
            np.testing.assert_array_equal(child.renderer.render_poses.call_args.args[1]['gripper_link'],np.eye(4)*(i+1))
        self.assertIsNone(r['marker_rms_px']);self.assertEqual(r['pitch_source'],'assumed_fixed')
        self.assertEqual(set(r['tcp_poses']),set(ARMS))
        p.m3t.tcp_poses.assert_called_once_with(r['m3t_estimate'])

    def test_encoder_only_uses_one_camera_packet_for_both(self):
        p=self.preview(False);sample=p.acquire()
        p.dashboard.main_frame.assert_called_once()
        p.arm_previews[ARMS[1]].dashboard.main_frame.assert_not_called()
        self.assertFalse(sample['receipt']['m3t_enabled'])
        self.assertNotIn('tcp_poses',sample['receipt'])
        for child in p.arm_previews.values():child.renderer.render.assert_called_once()

    def test_right_motion_and_non_bracketed_frame_are_rejected(self):
        for moving in (True,False):
            p=self.preview();reader=p.arm_previews[ARMS[1]].dashboard.read_arm
            q=[1.]*5+[20.]
            before={'degrees_and_gripper_percent':q,'read_started':1.9,'read_completed':2.}
            after={'degrees_and_gripper_percent':([4.]*5+[20.]) if moving else q,
                   'read_started':2.2 if moving else 2.05,'read_completed':2.3}
            reader.side_effect=[before,after]
            with self.assertRaises(ValueError):p.acquire()

    def test_missing_right_native_pose_is_rejected(self):
        p=self.preview();est,im,poses=p.m3t.read_frame.return_value;poses.pop(ARMS[1])
        with self.assertRaisesRegex(ValueError,'双臂'):p.acquire()

    def test_nominal_only_binding_needs_no_legacy_report(self):
        with tempfile.TemporaryDirectory() as folder:
            p=nominal_preview(folder)
            with mock.patch.object(overlay_preview,'ModelOverlay'),mock.patch.object(overlay_preview.threading.Thread,'start'):
                q=overlay_preview.OverlayPreview(p.dashboard,None,None,p.static_config,p.pitch_url,
                    nominal_mount_prior=p.mount_source['path'],fixed_pitch_assumption_deg=-61,start_worker=False)
            np.testing.assert_allclose(q.B,p.B);self.assertIsNone(q.audit_path);self.assertIsNone(q.X)
            q.check_sources()

if __name__=='__main__':unittest.main()
