"""Gripper angular travel from the installed v2 encoder calibration."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ENCODER_RESOLUTION = 4095  # installed SO101 STS3215 driver convention


def load_gripper_calibration(path):
    payload=Path(path).read_bytes();motor=json.loads(payload)['gripper']
    lo,hi,mode=motor['range_min'],motor['range_max'],motor['drive_mode']
    if (motor.get('id')!=6 or type(lo) is not int or type(hi) is not int
        or not 0<=lo<hi<=ENCODER_RESOLUTION or type(mode) is not int or mode not in (0,1)):
        raise ValueError('invalid v2 gripper encoder endpoints')
    span=(hi-lo)*2*math.pi/ENCODER_RESOLUTION
    return {'calibration_sha256':hashlib.sha256(payload).hexdigest(),
        'range_min':lo,'range_max':hi,'drive_mode':mode,
        'closed_raw':hi if mode else lo,'open_raw':lo if mode else hi,
        'encoder_resolution':ENCODER_RESOLUTION,
        'closed_rad':0.,'open_rad':span,'scale_rad_per_percent':span/100,
        'mapping':'v2 calibrated encoder travel; no URDF limit scaling',
        'zero_reference':'physical closed endpoint assigned nominal model zero; alignment not independently measured'}
