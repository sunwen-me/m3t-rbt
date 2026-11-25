import json
import math
from pathlib import Path
import tempfile
import unittest
from gripper_calibration import load_gripper_calibration

class GripperCalibrationTests(unittest.TestCase):
    def load(self,lo=2025,hi=3460,mode=0):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'v2.json';p.write_text(json.dumps({'gripper':{'id':6,'range_min':lo,'range_max':hi,'drive_mode':mode}}))
            return load_gripper_calibration(p)
    def test_real_encoder_span_does_not_use_urdf_limits(self):
        c=self.load()
        self.assertAlmostEqual(math.degrees(c['open_rad']),126.15384615384616)
        self.assertAlmostEqual(c['scale_rad_per_percent']*50,math.radians(63.07692307692308))
        self.assertEqual((c['closed_raw'],c['open_raw']),(2025,3460))
        self.assertIn('not independently measured',c['zero_reference'])
    def test_reverse_drive_preserves_opening_travel_with_swapped_raw_endpoints(self):
        c=self.load(mode=1)
        self.assertEqual((c['closed_raw'],c['open_raw']),(3460,2025))
        self.assertAlmostEqual(c['open_rad'],self.load()['open_rad'])
    def test_invalid_endpoints_rejected(self):
        for args in [(2025,2025,0),(-1,3460,0),(2025,4096,0),(2025,3460,2)]:
            with self.subTest(args=args),self.assertRaises(ValueError):self.load(*args)

if __name__=='__main__':unittest.main()
