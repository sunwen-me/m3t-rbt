#!/usr/bin/env python3
"""Offline SO101 v2 angle audit. Reads inputs; writes only the requested report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
ARMS = ('left', 'right')


def number(value, label, *, integer=False):
    if type(value) not in ((int,) if integer else (int, float)):
        raise ValueError(f'{label}: expected {"integer" if integer else "number"}, not bool/null')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'{label}: expected finite number')
    return value

def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load(path):
    try:
        payload = Path(path).read_bytes()
        value = json.loads(payload, object_pairs_hook=pairs)
        def finite_tree(node):
            if type(node) in (int, float):
                number(node, str(path))
            elif isinstance(node, dict):
                for child in node.values():
                    finite_tree(child)
            elif isinstance(node, list):
                for child in node:
                    finite_tree(child)
        finite_tree(value)
        if not isinstance(value, dict):
            raise ValueError('expected JSON object')
        return value, hashlib.sha256(payload).hexdigest()
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f'{path}: {exc}') from exc


def named(items, key, label):
    if not isinstance(items, list):
        raise ValueError(f'{label}: expected array')
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get(key), str):
            raise ValueError(f'{label}: missing {key}')
        if item[key] in result:
            raise ValueError(f'{label}: duplicate {item[key]}')
        result[item[key]] = item
    return result

def vector(values, label):
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError(f'{label}: expected five joint degrees and gripper percent')
    return [number(v, f'{label}[{i}]') for i, v in enumerate(values)]


def limits(model):
    if model.get('units') != 'metres_radians':
        raise ValueError('robot model units must be metres_radians')
    joints = named(model['joints'], 'name', 'model joints')
    result = {}
    for name in NAMES:
        joint = joints[name]
        lo, hi = (number(joint[k], f'{name}.{k}') for k in ('lower', 'upper'))
        if lo > hi:
            raise ValueError(f'{name}: model lower exceeds upper')
        result[name] = (lo, hi)
    return result


def limit_report(degree, bounds):
    lo, hi = (number(math.degrees(v), 'model limit degrees') for v in bounds)
    excess = max(lo - degree, degree - hi, 0.0)
    return {'lower_rad': bounds[0], 'upper_rad': bounds[1],
            'lower_deg': lo, 'upper_deg': hi, 'outside_limits': excess > 0,
            'excess_deg': excess}


def audit(feedback, left_calibration, right_calibration, robot_model, tracking_config,
          initialization, *, v2_zero_is_model_zero=False, runtime_robot_model=None):
    """Audit six input paths; malformed input raises ValueError, mismatches are reported.

    Historical provenance is compared only to its seed and prior mapping. Model
    limit excursions are observations, not software failures or zero diagnoses.
    """
    paths = dict(feedback=feedback, left_calibration=left_calibration,
                 right_calibration=right_calibration, robot_model=robot_model,
                 tracking_config=tracking_config, initialization=initialization)
    documents = {key: load(path) for key, path in paths.items()}
    f, model, config, init = (documents[k][0] for k in
                             ('feedback', 'robot_model', 'tracking_config', 'initialization'))
    errors = []
    def check(ok, label):
        if not ok:
            errors.append(label)
        return bool(ok)
    def close(a, b):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-8)
    report = {'schema': 'rbt-so101-joint-angle-audit-v1', 'errors': errors,
              'zero_reference': {'status': 'user_confirmed_v2_zero_is_urdf_zero'
                                if v2_zero_is_model_zero else 'not_provided',
                                'independently_measured': False},
              'positive_direction': {'software_mapping': 'positive_degree_to_radian',
                                     'independently_measured': False},
              'physical_acceptance': False,
              'interpretation': 'Identity conversion is a software check, not physical acceptance. '
                                'Limit excursions alone do not establish a zero error.',
              'formula': '(raw - (range_min + range_max) / 2) * 360 / 4095',
              'homing_offset_subtracted': False,
              'inputs': {k: {'path': str(Path(p).resolve()), 'sha256': documents[k][1]}
                         for k, p in paths.items()}, 'arms': {}}
    try:
        for key, doc in (('robot_model', model), ('tracking_config', config), ('initialization', init)):
            if type(doc.get('schema_version')) is not int or doc['schema_version'] != 1:
                raise ValueError(f'{key}: expected schema_version 1')
        bounds = limits(model)
        report['source_urdf_sha256'] = model.get('source_sha256')
        runtime_bounds = None
        if runtime_robot_model is not None:
            runtime, digest = load(runtime_robot_model)
            runtime_bounds = limits(runtime)
            report['runtime_model'] = {'path': str(Path(runtime_robot_model).resolve()),
                                       'sha256': digest, 'used_for_calibration_acceptance': False}
        check(f.get('schema') == 'rbt-so101-readonly-feedback-v1', 'feedback.schema mismatch')
        state = f['state']
        for key, expected in dict(protocol='rbt.so101.dual.v1', kind='state', mode='readonly',
                                  units='joint_deg_gripper_percent').items():
            check(state.get(key) == expected, f'feedback.state.{key} mismatch')
        check(state.get('fault') is None, 'feedback.state.fault is present')
        priors = config['joint_priors']
        if type(priors.get('enabled', True)) is not bool:
            raise ValueError('joint_priors.enabled: expected bool')
        check(priors.get('enabled', True), 'joint_priors disabled')
        check(bool(state.get('controller_id')) and
              priors.get('controller_id') == state.get('controller_id'), 'controller_id mismatch')
        mappings = {}
        for mapping in priors['mappings']:
            key = (mapping['arm'], mapping['joint'])
            if key in mappings:
                raise ValueError(f'duplicate prior mapping: {key}')
            mappings[key] = mapping
        seeds = named(init['arms'], 'name', 'initialization arms')
        for arm in ARMS:
            target = 'so101_' + arm
            calibration, digest = documents[arm + '_calibration']
            binding = check(f['calibration_sha256'][arm] == digest, f'{arm}: calibration SHA256 mismatch')
            current = vector(state['arms'][arm], f'{arm}.state')
            source = init['provenance']['arms_source'][target]
            historical = []
            for key in ('joint_before', 'joint_after'):
                sample = source[key]
                check(sample.get('source_arm') == arm, f'{target}.{key}.source_arm mismatch')
                historical.append(vector(sample['degrees_and_gripper_percent'], f'{target}.{key}'))
            arm_report = {'calibration_sha256': digest, 'calibration_binding_matches': binding, 'joints': {}}
            report['arms'][arm] = arm_report
            for i, name in enumerate(NAMES):
                motor = calibration[name]
                lo, hi = (number(motor[k], f'{arm}.{name}.{k}', integer=True)
                          for k in ('range_min', 'range_max'))
                if not 0 <= lo < hi <= 4095:
                    raise ValueError(f'{arm}.{name}: invalid v2 calibration endpoints')
                if 'homing_offset' in motor:
                    number(motor['homing_offset'], f'{arm}.{name}.homing_offset', integer=True)
                raw = number(f['raw'][arm][name], f'{arm}.{name}.raw', integer=True)
                midpoint = (lo + hi) / 2
                degree = number((raw - midpoint) * 360 / 4095, f'{arm}.{name}.degree')
                q = math.radians(degree)
                mapping = mappings[(target, name)]
                identity = check(mapping.get('source_arm') == arm and mapping.get('source_joint') == name,
                                 f'{target}.{name}: prior source_arm/source_joint mismatch')
                scale = number(mapping['scale_rad_per_unit'], f'{target}.{name}.scale')
                offset = number(mapping['offset_rad'], f'{target}.{name}.offset')
                direct = check(math.isclose(scale, math.pi / 180, rel_tol=0, abs_tol=1e-12)
                               and abs(offset) <= 1e-12, f'{target}.{name}: nonidentity mapping')
                mapped = number(current[i] * scale + offset, f'{target}.{name}.mapped')
                average = historical[0][i] / 2 + historical[1][i] / 2
                expected_seed = math.radians(average)
                prior_seed = number(average * scale + offset, f'{target}.{name}.prior_seed')
                seed = number(seeds[target]['joint_angles'][name], f'{target}.{name}.seed')
                row = {'raw': raw, 'calibration_min': lo, 'calibration_max': hi, 'midpoint': midpoint,
                       'raw_outside_calibration': not lo <= raw <= hi,
                       'raw_excess_counts': max(lo - raw, raw - hi, 0),
                       'degree_expected': degree, 'degree_state': current[i],
                       'degree_matches': check(close(degree, current[i]), f'{target}.{name}: degree mismatch'),
                       'model_limits': limit_report(degree, bounds[name]),
                       'prior': {'source_identity_matches': identity, 'identity_conversion': direct,
                                 'q_rad': mapped, 'direct_rad': q,
                                 'matches_direct': check(close(mapped, q), f'{target}.{name}: prior q mismatch')},
                       'initialization': {'historical_mean_deg': average, 'seed_rad': seed,
                           'expected_seed_rad': expected_seed, 'prior_mapped_rad': prior_seed,
                           'seed_matches_history': check(close(seed, expected_seed), f'{target}.{name}: seed mismatch'),
                           'prior_matches_history': check(close(prior_seed, expected_seed), f'{target}.{name}: historical prior mismatch'),
                           'seed_matches_prior': check(close(seed, prior_seed), f'{target}.{name}: seed/prior mismatch')}}
                if runtime_bounds is not None:
                    row['runtime_model_limits'] = limit_report(degree, runtime_bounds[name])
                arm_report['joints'][name] = row
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f'malformed input: missing or invalid field {exc}') from exc
    report['conclusion'] = ('software_mismatch' if errors else
                            'numerically_consistent_with_user_confirmed_zero' if v2_zero_is_model_zero else
                            'numerically_consistent_but_physical_reference_unverified')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('feedback', 'left-calibration', 'right-calibration', 'robot-model',
                 'tracking-config', 'initialization', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--runtime-robot-model', help='optional runtime limits, reported separately')
    parser.add_argument('--v2-zero-is-model-zero', action='store_true')
    args = vars(parser.parse_args(argv))
    output = Path(args.pop('output'))
    try:
        for key, path in args.items():
            if key != 'v2_zero_is_model_zero' and path is not None:
                if output.resolve() == Path(path).resolve() or (output.exists() and output.samefile(path)):
                    raise ValueError('output must not overwrite any input file')
        report = audit(**args)
    except (ValueError, OSError) as exc:
        print(f'joint angle audit failed: {exc}', file=sys.stderr)
        return 2
    try:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError) as exc:
        print(f'cannot write audit report: {exc}', file=sys.stderr)
        return 2
    return 2 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
