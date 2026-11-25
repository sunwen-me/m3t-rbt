"""Publisher ownership, calibration mismatch and failure handling."""
import argparse
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import joint_feedback_feed as feed


class FeedTests(unittest.TestCase):
    def test_one_actual_snapshot_and_read_failure_close_owned_resources(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);stop=threading.Event();opened=[];published=[]
                names=('a','b','c','d','e','gripper')
                cal=SimpleNamespace(motors={n:SimpleNamespace(motor_id=i,homing_offset=0,range_min=1,range_max=4095) for i,n in enumerate(names)},normalize_raw=lambda raw:tuple(float(raw[n]) for n in names))
                class Backend:
                    def __init__(self,port,calibration,**kwargs):
                        self.closed=False;opened.append(self)
                        self.assert_readonly=kwargs['allow_goal_writes'] is False
                    def _read_register(self,motor,register):return {'Homing_Offset':0,'Min_Position_Limit':1,'Max_Position_Limit':4095}[register]
                    def read_raw(self):
                        if fail:raise OSError('USB disconnected')
                        return dict.fromkeys(names,42)
                    def close(self):self.closed=True
                class Channels:
                    def __init__(self,**kwargs):
                        assert kwargs['publish_states'] and not kwargs['subscribe_commands']
                        self.closed=False;opened.append(self)
                    def publish_state(self,state,sequence):published.append(state);stop.set()
                    def close(self):self.closed=True
                class Service:
                    def __init__(self,left,right,**kwargs):self.left=left;self.right=right;assert kwargs['mode']=='readonly'
                    def read_snapshot(self):
                        arms={'left':list(self.left.read_state()),'right':list(self.right.read_state())}
                        return {'protocol':feed.STATE_PROTOCOL,'kind':'state','units':feed.STATE_UNITS,'mode':'readonly','controller_id':'test','sequence':1,'driver_session':'real-session','capture_time_known':False,'arms':arms,'read_intervals':{a:{'start_ns':1,'end_ns':2} for a in arms}}
                for arm in ('left','right'):(root/arm).write_text('{}')
                args=argparse.Namespace(poll_hz=30,output=root/'cache.json',left_port='/left',right_port='/right',left_calibration=root/'left',right_calibration=root/'right',driver_path='unused',channel_prefix='test',controller_id='test')
                api=feed.DriverApi(Backend,lambda path:cal,names,Service,Channels)
                with patch.object(feed,'atomic_write_cache',wraps=feed.atomic_write_cache) as write:
                    if fail:
                        with self.assertRaises(OSError):feed.run_feed(args,api=api,stop_event=stop,fuser_run=lambda *a,**k:SimpleNamespace(returncode=1))
                        write.assert_not_called();self.assertEqual(published,[])
                    else:
                        feed.run_feed(args,api=api,stop_event=stop,fuser_run=lambda *a,**k:SimpleNamespace(returncode=1))
                        self.assertEqual(published[0]['arms']['left'],[42.]*6)
                self.assertTrue(all(o.closed for o in opened))
                self.assertTrue(all(o.assert_readonly for o in opened if isinstance(o,Backend)))

    def test_register_mismatch_refused(self):
        cal=SimpleNamespace(motors={'a':SimpleNamespace(motor_id=1,homing_offset=0)})
        backend=SimpleNamespace(_read_register=lambda *args:99)
        with self.assertRaisesRegex(RuntimeError,'hardware/calibration mismatch'):
            feed.verify_hardware_calibration(backend,cal,('a',),'left')

    def test_atomic_cache_preserves_unknown_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache.json';path.write_text('{"other":true}')
            with self.assertRaises(ValueError):feed.atomic_write_cache(path,state={},calibration_sha256={},raw={})
            self.assertEqual(json.loads(path.read_text()),{'other':True})

if __name__=='__main__':unittest.main()
