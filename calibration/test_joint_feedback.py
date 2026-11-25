"""Real-feedback cache identity and age gates."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from joint_feedback import FeedbackReader, parse_snapshot
from shared_board_web import NAMES


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.now=12_000_000_000
        self.sample={'schema':'rbt-so101-readonly-feedback-v1','calibration_sha256':{'left':'verified-v2'},
            'written_monotonic_ns':self.now-10_000_000,'raw':{'left':dict.fromkeys((*NAMES,'gripper'),2048)},
            'state':{'protocol':'rbt.so101.dual.v1','kind':'state','units':'joint_deg_gripper_percent',
                'mode':'readonly','controller_id':'real-controller','fault':None,'driver_session':'session1','sequence':2,
                'arms':{'left':[10.,20.,30.,40.,50.,1.]},
                'read_intervals':{'left':{'start_ns':self.now-20_000_000,'end_ns':self.now-15_000_000}}}}
    def parse(self,sample):return parse_snapshot(sample,self.now,'verified-v2','real-controller')
    def test_keeps_measured_angles_raw_and_interval(self):
        r=self.parse(self.sample)
        self.assertEqual(r['degrees_and_gripper_percent'],[10.,20.,30.,40.,50.,1.])
        self.assertEqual(r['feedback_sequence'],2)
        self.assertEqual(r['source_arm'],'left')
        self.assertEqual(r,self.parse_arm(self.sample,'left','verified-v2'))
        self.assertAlmostEqual(r['read_started'],11.98)
    def test_refuses_wrong_v2_writable_identity_or_stale(self):
        variants=[]
        for key,value in [('mode','hardware'),('controller_id','other'),('kind','fault')]:
            s=copy.deepcopy(self.sample);s['state'][key]=value;variants.append(s)
        s=copy.deepcopy(self.sample);s['calibration_sha256']['left']='old';variants.append(s)
        s=copy.deepcopy(self.sample);s['written_monotonic_ns']=self.now-200_000_000;variants.append(s)
        for s in variants:
            with self.assertRaises(ValueError):self.parse(s)
    def test_refuses_invalid_interval_and_nonfinite_angle(self):
        s=copy.deepcopy(self.sample);s['state']['arms']['left'][0]=float('nan')
        with self.assertRaises(ValueError):self.parse(s)
        s=copy.deepcopy(self.sample);s['state']['read_intervals']['left']['end_ns']=self.now+1
        with self.assertRaises(ValueError):self.parse(s)

    def dual_sample(self):
        sample=copy.deepcopy(self.sample)
        sample['calibration_sha256']['right']='right-v2'
        sample['state']['arms']['right']=[-10.,-20.,-30.,-40.,-50.,2.]
        sample['state']['read_intervals']['right']={'start_ns':self.now-18_000_000,'end_ns':self.now-12_000_000}
        sample['raw']['right']=dict.fromkeys((*NAMES,'gripper'),1024)
        return sample

    def parse_arm(self,sample,arm='right',calibration='right-v2'):
        return parse_snapshot(sample,self.now,calibration,'real-controller',source_arm=arm)

    def test_right_uses_its_own_angles_raw_hash_and_interval(self):
        sample=self.dual_sample()
        result=self.parse_arm(sample)
        self.assertEqual(result['source_arm'],'right')
        self.assertEqual(result['degrees_and_gripper_percent'],sample['state']['arms']['right'])
        self.assertEqual(result['raw'],sample['raw']['right'])
        self.assertAlmostEqual(result['read_started'],11.982)
        self.assertAlmostEqual(result['read_completed'],11.988)
        del sample['calibration_sha256']['left']
        del sample['state']['arms']['left']
        del sample['state']['read_intervals']['left']
        del sample['raw']['left']
        self.assertEqual(result,self.parse_arm(sample))

    def test_left_hash_cannot_authenticate_right(self):
        with self.assertRaisesRegex(ValueError,'当前右臂'):
            self.parse_arm(self.dual_sample(),calibration='verified-v2')

    def test_missing_right_data_never_falls_back_to_left(self):
        for section in ('calibration_sha256','arms','read_intervals','raw'):
            with self.subTest(section=section):
                sample=self.dual_sample()
                container=sample if section in ('calibration_sha256','raw') else sample['state']
                del container[section]['right']
                with self.assertRaisesRegex(ValueError,'右臂'):
                    self.parse_arm(sample)

    def test_old_right_interval_rejected_with_fresh_left_and_cache(self):
        sample=self.dual_sample()
        sample['state']['read_intervals']['right']={'start_ns':self.now-220_000_000,'end_ns':self.now-200_000_000}
        with self.assertRaisesRegex(ValueError,'右臂读取时间'):
            self.parse_arm(sample)
        self.parse(sample)

    def test_right_keeps_identity_readonly_sequence_and_clock_checks(self):
        changes=[('protocol','other'),('kind','fault'),('units','radians'),
                 ('mode','hardware'),('controller_id','other'),('fault','fault'),
                 ('sequence',0),('driver_session','')]
        for key,value in changes:
            with self.subTest(key=key):
                sample=self.dual_sample();sample['state'][key]=value
                with self.assertRaises(ValueError):self.parse_arm(sample)
        for written in (self.now-150_000_001,self.now+1):
            with self.subTest(written=written):
                sample=self.dual_sample();sample['written_monotonic_ns']=written
                with self.assertRaises(ValueError):self.parse_arm(sample)
        for interval in ({'start_ns':self.now-11_000_000,'end_ns':self.now-9_000_000},
                         {'start_ns':self.now-12_000_000,'end_ns':self.now-18_000_000}):
            with self.subTest(interval=interval):
                sample=self.dual_sample();sample['state']['read_intervals']['right']=interval
                with self.assertRaisesRegex(ValueError,'右臂读取时间'):self.parse_arm(sample)

    def test_right_incomplete_angles_and_raw_rejected(self):
        for angles in ([1.], [float('nan')]*6, [True]*6):
            sample=self.dual_sample();sample['state']['arms']['right']=angles
            with self.assertRaisesRegex(ValueError,'右臂反馈不完整'):self.parse_arm(sample)
        sample=self.dual_sample();del sample['raw']['right']['gripper']
        with self.assertRaisesRegex(ValueError,'右臂电机原始反馈不完整'):self.parse_arm(sample)

    def test_invalid_arm_rejected_by_parser_and_reader(self):
        for arm in ('both','LEFT','',None,0):
            with self.subTest(arm=arm):
                with self.assertRaisesRegex(ValueError,'source_arm'):
                    self.parse_arm(self.dual_sample(),arm)
                with self.assertRaisesRegex(ValueError,'source_arm'):
                    FeedbackReader('/unused/feedback.json','right-v2','real-controller',source_arm=arm)

    def test_two_readers_select_independent_arms_from_same_file(self):
        sample=self.dual_sample()
        left=FeedbackReader('/unused/feedback.json','verified-v2','real-controller')
        right=FeedbackReader(left.path,'right-v2','real-controller',source_arm='right')
        self.assertEqual(left.source_arm,'left')
        self.assertEqual(right.source_arm,'right')
        with patch.object(Path,'read_bytes',return_value=json.dumps(sample).encode()) as read_file, \
             patch('joint_feedback.time.monotonic_ns',side_effect=[self.now-30_000_000,self.now]*2), \
             patch('joint_feedback.time.monotonic',side_effect=[0.,.1]*2), \
             patch('joint_feedback.time.sleep') as sleep:
            self.assertEqual(left.read(),self.parse(sample))
            self.assertIsNone(right.last)
            self.assertEqual(right.read(),self.parse_arm(sample))
        self.assertEqual(read_file.call_count,2)
        self.assertEqual(left.last,right.last)
        sleep.assert_not_called()

    def test_reader_keeps_request_time_and_duplicate_gates(self):
        for arm,calibration in (('left','verified-v2'),('right','right-v2')):
            for gate in ('requested','duplicate'):
                with self.subTest(arm=arm,gate=gate):
                    reader=FeedbackReader('/unused/feedback.json',calibration,'real-controller',source_arm=arm)
                    if gate=='duplicate':reader.last=('session1',2)
                    requested=self.now if gate=='requested' else self.now-30_000_000
                    with patch.object(Path,'read_bytes',return_value=json.dumps(self.dual_sample()).encode()), \
                         patch('joint_feedback.time.monotonic_ns',side_effect=[requested,self.now]), \
                         patch('joint_feedback.time.monotonic',side_effect=[0.,.1,1.]), \
                         patch('joint_feedback.time.sleep'):
                        with self.assertRaisesRegex(ValueError,'只读关节反馈不可用'):reader.read()

if __name__=='__main__':unittest.main()

