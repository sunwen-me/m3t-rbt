import copy
import json
import subprocess
import sys
import time
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from joint_feedback import RbtFeedbackReader, _parse_rbt_state


def state_at(start):
    return {
        'protocol': 'rbt.so101.dual.v1',
        'kind': 'state',
        'units': 'joint_deg_gripper_percent',
        'mode': 'readonly',
        'controller_id': 'dual-v2',
        'driver_session': 'driver-session',
        'sequence': 10,
        'monotonic_ns': start + 4,
        'fault': None,
        'arms': {'left': [1, 2, 3, 4, 5, 25], 'right': [-1, -2, -3, -4, -5, 60]},
        'read_intervals': {
            'left': {'start_ns': start, 'end_ns': start + 1},
            'right': {'start_ns': start + 2, 'end_ns': start + 3},
        },
    }


class RbtJointFeedbackTest(unittest.TestCase):
    def test_uses_driver_angles_and_timestamps(self):
        for mode in ('readonly', 'hardware'):
            state = state_at(1_000_000_000)
            state['mode'] = mode
            actual = _parse_rbt_state(state, 10, 1_000_000_002, 1_000_000_005, 'dual-v2', 'right')
            self.assertEqual(actual['degrees_and_gripper_percent'], [-1, -2, -3, -4, -5, 60])
            self.assertEqual(actual['read_started'], 1.000000002)
            self.assertEqual(actual['read_completed'], 1.000000003)
            self.assertEqual(actual['driver_session'], 'driver-session')
            self.assertEqual(actual['controller_id'], 'dual-v2')
            self.assertEqual(actual['feedback_sequence'], 10)
            self.assertNotIn('raw', actual)

    def test_rejects_incomplete_unsafe_or_stale_state(self):
        start = 1_000_000_000
        state = state_at(start)
        right = copy.deepcopy(state)
        right['arms']['right'][5] = float('nan')
        interval = copy.deepcopy(state)
        interval['read_intervals']['right']['end_ns'] = start + 5
        cases = [
            ({**state, 'mode': 'mock'}, 10, start, start + 5),
            ({**state, 'fault': 'communication error'}, 10, start, start + 5),
            ({**state, 'controller_id': 'other'}, 10, start, start + 5),
            ({**state, 'driver_session': ''}, 10, start, start + 5),
            ({**state, 'arms': {'left': [0] * 6}}, 10, start, start + 5),
            (state, 11, start, start + 5),
            (right, 10, start, start + 5),
            (interval, 10, start, start + 5),
            (state, 10, start + 1, start + 5),
            (state, 10, start, start + 150_000_005),
        ]
        for value, sequence, requested, now in cases:
            with self.subTest(value=value, sequence=sequence, requested=requested, now=now):
                with self.assertRaises(ValueError):
                    _parse_rbt_state(value, sequence, requested, now, 'dual-v2', 'left')

    def test_subscriber_rejects_wrong_clock_and_releases_both_payload_and_endpoint(self):
        clock = object()
        buffers = []

        class Subscriber:
            closed = False

            def receive(self, timeout):
                start = time.monotonic_ns() - 4
                received = SimpleNamespace(clock=clock if buffers else object())
                view = memoryview(json.dumps(state_at(start)).encode())
                buffers.append(view)
                return SimpleNamespace(sequence=10, timing=SimpleNamespace(received=received),
                                       payload_view=lambda: view)

            def close(self):
                self.closed = True

        subscriber = Subscriber()
        descriptor = object()
        rbt = ModuleType('rbt')
        rbt.RbtError = RuntimeError
        rbt.StreamProfile = SimpleNamespace(SENSOR='sensor')
        rbt_time = ModuleType('rbt.time')
        rbt_time.HostClock = lambda: SimpleNamespace(monotonic_identity=clock)
        rbt_time.TimeError = RuntimeError
        rbt_stream = ModuleType('rbt.time_stream')
        rbt_stream.stamped_bytes_descriptor = lambda service, profile: descriptor
        rbt_stream.StampedBytesSubscriber = SimpleNamespace(open=lambda requested: subscriber)
        with patch.dict(sys.modules, {'rbt': rbt, 'rbt.time': rbt_time, 'rbt.time_stream': rbt_stream}):
            actual = RbtFeedbackReader('so101/state/v1', 'dual-v2').read()
        self.assertEqual(actual['feedback_sequence'], 10)
        self.assertEqual(len(buffers), 2)
        self.assertTrue(subscriber.closed)
        with self.assertRaises(ValueError):
            bytes(buffers[-1])
        buffers[0].release()

    def test_external_reader_uses_configured_python_and_returns_json(self):
        reader = RbtFeedbackReader('so101/state/v1', 'dual-v2', source_arm='right',
                                   python_executable='/python/with/rbt')
        with patch('joint_feedback.subprocess.run', return_value=SimpleNamespace(
                stdout='{"feedback_sequence":42}', returncode=0)) as run:
            self.assertEqual(reader.read(), {'feedback_sequence': 42})
            command, = run.call_args.args
            self.assertEqual(command[0], '/python/with/rbt')
            self.assertTrue(command[1].endswith('/joint_feedback.py'))
            self.assertEqual(command[2:], ['so101/state/v1', 'dual-v2', 'right'])
            self.assertEqual(run.call_args.kwargs['timeout'], 1.5)
        with patch('joint_feedback.subprocess.run', side_effect=subprocess.TimeoutExpired('reader', 1.5)):
            with self.assertRaisesRegex(ValueError, '超时'):
                reader.read()


if __name__ == '__main__':
    unittest.main()
