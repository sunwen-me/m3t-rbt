"""Synthetic, offline regression tests; no hardware or saved target inputs."""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import joint_angle_audit as tool


class JointAngleAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = {key: self.root / (key + '.json') for key in
                      ('feedback', 'left_calibration', 'right_calibration', 'robot_model',
                       'tracking_config', 'initialization')}
        self.data = {}
        hashes, raw, states, seeds, provenance, mappings = {}, {}, {}, [], {}, []
        for arm in tool.ARMS:
            target = 'so101_' + arm
            cal = {name: {'id': i + 1, 'range_min': 859, 'range_max': 3065,
                          'homing_offset': -1734 if arm == 'right' else -1908, 'drive_mode': 0}
                   for i, name in enumerate(tool.NAMES)}
            self.data[arm + '_calibration'] = cal
            self.save(arm + '_calibration')
            hashes[arm] = hashlib.sha256(self.paths[arm + '_calibration'].read_bytes()).hexdigest()
            raw[arm] = {name: 1962 for name in tool.NAMES}
            if arm == 'right':
                raw[arm]['elbow_flex'] = 3074
            states[arm] = [(raw[arm][n] - 1962) * 360 / 4095 for n in tool.NAMES] + [20.]
            # Historical samples differ from current feedback, and from each other.
            history = [10., 20., 30., 40., 50., 20.]
            provenance[target] = {
                'joint_before': {'source_arm': arm, 'degrees_and_gripper_percent': history},
                'joint_after': {'source_arm': arm,
                                'degrees_and_gripper_percent': [v + .2 for v in history]}}
            seeds.append({'name': target, 'joint_angles':
                          {n: math.radians(history[i] + .1) for i, n in enumerate(tool.NAMES)}})
            mappings.extend({'arm': target, 'joint': n, 'source_arm': arm, 'source_joint': n,
                             'scale_rad_per_unit': math.pi / 180, 'offset_rad': 0.}
                            for n in tool.NAMES)
        self.data['feedback'] = {'schema': 'rbt-so101-readonly-feedback-v1',
            'calibration_sha256': hashes, 'raw': raw,
            'state': {'protocol': 'rbt.so101.dual.v1', 'kind': 'state', 'mode': 'readonly',
                      'units': 'joint_deg_gripper_percent', 'controller_id': 'test', 'arms': states}}
        self.data['robot_model'] = {'schema_version': 1, 'units': 'metres_radians',
            'source_sha256': 'a' * 64, 'joints':
            [{'name': n, 'lower': -1.69, 'upper': 1.69} for n in tool.NAMES]}
        self.data['tracking_config'] = {'schema_version': 1, 'joint_priors':
            {'enabled': True, 'controller_id': 'test', 'mappings': mappings}}
        self.data['initialization'] = {'schema_version': 1, 'arms': seeds,
                                      'provenance': {'arms_source': provenance}}
        self.save_all()

    def save(self, key):
        self.paths[key].write_text(json.dumps(self.data[key]))

    def save_all(self):
        for key in self.paths:
            self.save(key)

    def run_audit(self, **kwargs):
        self.save_all()
        return tool.audit(**self.paths, **kwargs)

    def cli(self, **kwargs):
        args = [v for k, p in self.paths.items() for v in ('--' + k.replace('_', '-'), str(p))]
        args.extend(['--output', str(kwargs.pop('output', self.root / 'report.json'))])
        args.extend(kwargs.pop('extra', []))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = tool.main(args)
        return code, err.getvalue()

    def test_right_elbow_regression_and_no_homing_subtraction(self):
        report = self.run_audit()
        self.assertEqual(report['errors'], [])
        row = report['arms']['right']['joints']['elbow_flex']
        self.assertEqual(row['midpoint'], 1962)
        self.assertAlmostEqual(row['degree_expected'], 97.75824175824175)
        self.assertEqual(row['degree_expected'], row['degree_state'])
        self.assertFalse(report['homing_offset_subtracted'])
        self.assertEqual(report['source_urdf_sha256'], 'a' * 64)
        self.assertEqual(report['zero_reference']['status'], 'not_provided')
        self.assertEqual(report['conclusion'], 'numerically_consistent_but_physical_reference_unverified')

    def test_user_confirmed_zero(self):
        report = self.run_audit(v2_zero_is_model_zero=True)
        self.assertEqual(report['conclusion'], 'numerically_consistent_with_user_confirmed_zero')
        self.assertEqual(report['zero_reference'],
                         {'status': 'user_confirmed_v2_zero_is_urdf_zero', 'independently_measured': False})
        self.assertFalse(report['positive_direction']['independently_measured'])
        self.assertEqual(self.cli(extra=['--v2-zero-is-model-zero'])[0], 0)

    def test_raw_and_model_excursions_are_independent_and_not_clamped(self):
        report = self.run_audit()
        row = report['arms']['right']['joints']['elbow_flex']
        self.assertTrue(row['raw_outside_calibration'])
        self.assertEqual(row['raw_excess_counts'], 9)
        self.assertTrue(row['model_limits']['outside_limits'])
        self.assertAlmostEqual(row['model_limits']['upper_deg'], math.degrees(1.69))
        self.assertAlmostEqual(row['model_limits']['excess_deg'], 97.75824175824175 - math.degrees(1.69))
        self.assertEqual(self.cli()[0], 0)
        runtime = self.root / 'runtime.json'
        model = json.loads(json.dumps(self.data['robot_model']))
        for joint in model['joints']:
            joint.update(lower=-math.radians(100), upper=math.radians(100))
        runtime.write_text(json.dumps(model))
        report = self.run_audit(runtime_robot_model=runtime)
        row = report['arms']['right']['joints']['elbow_flex']
        self.assertTrue(row['model_limits']['outside_limits'])
        self.assertFalse(row['runtime_model_limits']['outside_limits'])
        self.assertTrue(row['raw_outside_calibration'])
        self.assertAlmostEqual(row['degree_expected'], 97.75824175824175)
        self.assertFalse(report['runtime_model']['used_for_calibration_acceptance'])

    def test_swapped_calibration_sha(self):
        hashes = self.data['feedback']['calibration_sha256']
        hashes['left'], hashes['right'] = hashes['right'], hashes['left']
        report = self.run_audit(v2_zero_is_model_zero=True)
        self.assertEqual(report['conclusion'], 'software_mismatch')
        self.assertEqual(sum('SHA256' in e for e in report['errors']), 2)
        self.assertEqual(self.cli()[0], 2)

    def test_nonidentity_mapping_even_at_zero_and_compensating_offset(self):
        mapping = self.data['tracking_config']['joint_priors']['mappings'][0]
        for scale, offset in ((-math.pi / 180, 0), (1., 0), (math.pi / 180, .1)):
            with self.subTest(scale=scale, offset=offset):
                mapping.update(scale_rad_per_unit=scale, offset_rad=offset)
                self.assertTrue(any('nonidentity' in e for e in self.run_audit()['errors']))

    def test_seed_and_history_are_checked_independently_of_current_feedback(self):
        self.assertEqual(self.run_audit()['errors'], [])
        self.data['initialization']['arms'][0]['joint_angles']['shoulder_pan'] += .1
        report = self.run_audit()
        self.assertTrue(any('seed mismatch' in e for e in report['errors']))
        self.assertEqual(self.cli()[0], 2)

    def test_wrong_source_identity(self):
        mapping = self.data['tracking_config']['joint_priors']['mappings'][0]
        mapping['source_arm'] = 'right'
        self.assertTrue(any('source_arm' in e for e in self.run_audit()['errors']))
        mapping['source_arm'] = 'left'
        mapping['source_joint'] = 'elbow_flex'
        self.assertTrue(any('source_joint' in e for e in self.run_audit()['errors']))
        mapping['source_joint'] = 'shoulder_pan'
        sample = self.data['initialization']['provenance']['arms_source']['so101_left']['joint_before']
        sample['source_arm'] = 'right'
        self.assertTrue(any('joint_before.source_arm' in e for e in self.run_audit()['errors']))

    def test_feedback_identity_and_inactive_priors(self):
        for container, key, bad in ((self.data['feedback'], 'schema', 'bad'),
                (self.data['feedback']['state'], 'mode', 'command'),
                (self.data['feedback']['state'], 'units', 'radians'),
                (self.data['feedback']['state'], 'protocol', 'bad'),
                (self.data['tracking_config']['joint_priors'], 'enabled', False)):
            with self.subTest(key=key):
                old = container[key]
                container[key] = bad
                self.assertEqual(self.run_audit()['conclusion'], 'software_mismatch')
                container[key] = old

    def test_bool_nonfinite_missing_names_and_duplicate_mapping(self):
        raw = self.data['feedback']['raw']['left']
        for bad in (True, float('nan'), float('inf')):
            with self.subTest(value=bad):
                raw['shoulder_pan'] = bad
                with self.assertRaises(ValueError):
                    self.run_audit()
                self.assertEqual(self.cli()[0], 2)
        del raw['shoulder_pan']
        with self.assertRaisesRegex(ValueError, 'shoulder_pan'):
            self.run_audit()
        raw['shoulder_pan'] = 1962
        mappings = self.data['tracking_config']['joint_priors']['mappings']
        mappings.append(dict(mappings[0]))
        with self.assertRaisesRegex(ValueError, 'duplicate prior'):
            self.run_audit()
        mappings.pop()
        mappings.pop()
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_malformed_json_and_output_cannot_overwrite_input(self):
        path = self.paths['feedback']
        for content in ('{', '[]', '{"schema":1,"schema":2}'):
            path.write_text(content)
            code, error = self.cli()
            self.assertEqual(code, 2)
            self.assertIn('joint angle audit failed', error)
        self.save('feedback')
        before = {key: p.read_bytes() for key, p in self.paths.items()}
        self.assertEqual(self.cli(output=path)[0], 2)
        self.assertEqual(before, {key: p.read_bytes() for key, p in self.paths.items()})
        self.assertEqual(self.cli()[0], 0)
        self.assertEqual(before, {key: p.read_bytes() for key, p in self.paths.items()})
        link = self.root / 'input-hardlink.json'
        link.hardlink_to(path)
        self.assertEqual(self.cli(output=link)[0], 2)


if __name__ == '__main__':
    unittest.main()
