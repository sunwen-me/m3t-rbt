"""Read live and archived SO101 joint feedback."""
from __future__ import annotations
import json
import math
from pathlib import Path
import subprocess
import time


def parse_snapshot(value,now_ns,expected_calibration,controller_id,*,source_arm='left'):
    from shared_board_web import NAMES

    if source_arm not in ('left','right'):raise ValueError('source_arm 只允许 left/right')
    arm_label='左臂' if source_arm=='left' else '右臂'
    if value.get('schema')!='rbt-so101-readonly-feedback-v1':raise ValueError('关节反馈缓存格式不匹配')
    if value.get('calibration_sha256',{}).get(source_arm)!=expected_calibration:
        raise ValueError(f'关节反馈来源不是当前{arm_label} v2')
    s=value['state']
    if (s.get('protocol')!='rbt.so101.dual.v1' or s.get('kind')!='state'
        or s.get('units')!='joint_deg_gripper_percent' or s.get('mode')!='readonly'
        or s.get('controller_id')!=controller_id or s.get('fault') is not None):
        raise ValueError('只读关节反馈身份或状态无效')
    written=value.get('written_monotonic_ns')
    if type(written) is not int or not 0<=now_ns-written<=150_000_000:
        raise ValueError('实时关节反馈已过期')
    seq=s.get('sequence');session=s.get('driver_session')
    if type(seq) is not int or seq<1 or not isinstance(session,str) or not session:
        raise ValueError('关节反馈序号无效')
    angles=s.get('arms',{}).get(source_arm)
    interval=s.get('read_intervals',{}).get(source_arm)
    if (not isinstance(angles,list) or len(angles)!=6
        or any(type(v) not in (int,float) or not math.isfinite(v) for v in angles)):
        raise ValueError(f'{arm_label}反馈不完整')
    if not isinstance(interval,dict):raise ValueError(f'{arm_label}读取时间无效或过期')
    start,end=interval.get('start_ns'),interval.get('end_ns')
    if (type(start) is not int or type(end) is not int or not 0<=start<=end<=written
        or now_ns-end>150_000_000):raise ValueError(f'{arm_label}读取时间无效或过期')
    raw=value.get('raw',{}).get(source_arm)
    if not isinstance(raw,dict) or any(type(raw.get(n)) is not int for n in (*NAMES,'gripper')):
        raise ValueError(f'{arm_label}电机原始反馈不完整')
    return {'degrees_and_gripper_percent':angles,'raw':raw,'read_started':start/1e9,'read_completed':end/1e9,
            'feedback_sequence':seq,'driver_session':session,'controller_id':controller_id,
            'source':'actual read-only RBT state publisher','source_arm':source_arm}


class FeedbackReader:
    def __init__(self,path,calibration_sha256,controller_id,*,source_arm='left'):
        if source_arm not in ('left','right'):raise ValueError('source_arm 只允许 left/right')
        self.source_arm=source_arm
        self.path=Path(path);self.calibration_sha256=calibration_sha256;self.controller_id=controller_id
        self.last=None

    def read(self):
        requested=time.monotonic_ns();deadline=time.monotonic()+.75;error='尚未收到实时关节反馈'
        while time.monotonic()<deadline:
            try:
                result=parse_snapshot(json.loads(self.path.read_bytes()),time.monotonic_ns(),
                    self.calibration_sha256,self.controller_id,source_arm=self.source_arm)
                key=(result['driver_session'],result['feedback_sequence'])
                if result['read_started']*1e9>=requested and key!=self.last:
                    self.last=key;return result
            except (OSError,ValueError,KeyError,TypeError) as exc:error=str(exc)
            time.sleep(.01)
        raise ValueError('只读关节反馈不可用：'+error)


_MAX_STATE_AGE_NS = 150_000_000
_ARMS = ('left', 'right')


def _parse_rbt_state(value, sequence, requested_ns, now_ns, controller_id, source_arm):
    if (not isinstance(value, dict) or value.get('protocol') != 'rbt.so101.dual.v1'
            or value.get('kind') != 'state' or value.get('units') != 'joint_deg_gripper_percent'
            or value.get('mode') not in ('readonly', 'hardware')
            or value.get('controller_id') != controller_id or value.get('fault') is not None):
        raise ValueError('RBT 关节反馈身份或状态无效')
    session = value.get('driver_session')
    if (type(value.get('sequence')) is not int or value['sequence'] != sequence or sequence < 1
            or not isinstance(session, str) or not session.strip()):
        raise ValueError('RBT 关节反馈会话或序号无效')

    stamp = value.get('monotonic_ns')
    if type(stamp) is not int or not 0 <= now_ns - stamp <= _MAX_STATE_AGE_NS:
        raise ValueError('RBT 实时关节反馈已过期或时间无效')
    arms, intervals = value.get('arms'), value.get('read_intervals')
    if (not isinstance(arms, dict) or set(arms) != set(_ARMS)
            or not isinstance(intervals, dict) or set(intervals) != set(_ARMS)):
        raise ValueError('RBT 双臂关节反馈不完整')

    starts, ends = {}, {}
    for arm in _ARMS:
        angles, interval = arms[arm], intervals[arm]
        if (not isinstance(angles, list) or len(angles) != 6
                or any(type(angle) not in (int, float) or not math.isfinite(angle) for angle in angles)
                or not 0 <= angles[5] <= 100):
            raise ValueError(f'RBT {arm} 关节或夹爪反馈无效')
        if not isinstance(interval, dict):
            raise ValueError(f'RBT {arm} 读取时间无效')
        start, end = interval.get('start_ns'), interval.get('end_ns')
        if (type(start) is not int or type(end) is not int
                or not 0 < start <= end <= stamp or now_ns - start > _MAX_STATE_AGE_NS):
            raise ValueError(f'RBT {arm} 读取时间无效或已过期')
        starts[arm], ends[arm] = start, end
    if starts[source_arm] < requested_ns:
        raise ValueError('尚未收到请求后的实时 RBT 关节反馈')

    return {
        'degrees_and_gripper_percent': arms[source_arm],
        'read_started': starts[source_arm] / 1e9,
        'read_completed': ends[source_arm] / 1e9,
        'feedback_sequence': sequence,
        'driver_session': session,
        'controller_id': controller_id,
        'source': 'SO101 RBT state publisher',
        'source_arm': source_arm,
    }


class RbtFeedbackReader:
    """Read a new dual-arm state from a subscriber local to this call."""

    def __init__(self, service, controller_id, *, source_arm='left', python_executable=None):
        if source_arm not in _ARMS:
            raise ValueError('source_arm 只允许 left/right')
        self.service = service
        self.controller_id = controller_id
        self.source_arm = source_arm
        self.python_executable = python_executable

    def read(self):
        if self.python_executable is None:
            return self._read_direct()
        try:
            result = subprocess.run([
                self.python_executable, str(Path(__file__).resolve()), self.service,
                self.controller_id, self.source_arm,
            ], capture_output=True, text=True, timeout=1.5)
        except subprocess.TimeoutExpired as exc:
            raise ValueError('RBT 关节反馈读取超时') from exc
        except OSError as exc:
            raise ValueError(f'无法启动 RBT 关节反馈读取：{exc}') from exc
        if result.returncode:
            detail = result.stderr.strip().splitlines()
            raise ValueError(detail[-1][:300] if detail else 'RBT 关节反馈读取失败')
        try:
            return json.loads(result.stdout)
        except (UnicodeError, ValueError) as exc:
            raise ValueError('RBT 关节反馈进程输出无效') from exc

    def _read_direct(self):
        requested = time.monotonic_ns()
        deadline = requested + 750_000_000
        try:
            from rbt import RbtError, StreamProfile
            from rbt.time import HostClock, TimeError
            from rbt.time_stream import StampedBytesSubscriber, stamped_bytes_descriptor
        except ImportError as exc:
            raise ValueError('RBT 关节反馈不可用：缺少 RBT Python SDK') from exc

        descriptor = stamped_bytes_descriptor(self.service, profile=StreamProfile.SENSOR)
        expected_clock = HostClock().monotonic_identity
        subscriber = None
        error = '尚未收到实时 RBT 关节反馈'
        try:
            while (remaining := (deadline - time.monotonic_ns()) / 1e9) > 0:
                if subscriber is None:
                    try:
                        subscriber = StampedBytesSubscriber.open(descriptor)
                    except RbtError as exc:
                        error = str(exc)
                        time.sleep(min(0.02, remaining))
                    continue
                try:
                    sample = subscriber.receive(timeout=remaining)
                    if sample is None:
                        continue
                    try:
                        sequence = sample.sequence
                        if sample.timing.received.clock != expected_clock:
                            raise ValueError('RBT 关节反馈时钟不是本机单调时钟')
                        with sample.payload_view() as payload:
                            value = json.loads(bytes(payload))
                    finally:
                        del sample
                    return _parse_rbt_state(value, sequence, requested, time.monotonic_ns(),
                                            self.controller_id, self.source_arm)
                except (RbtError, TimeError, ValueError, OverflowError) as exc:
                    error = str(exc)
        finally:
            if subscriber is not None:
                subscriber.close()
        raise ValueError('RBT 关节反馈不可用：' + error)


def _main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Read one fresh SO101 RBT state sample')
    parser.add_argument('service')
    parser.add_argument('controller_id')
    parser.add_argument('source_arm', choices=_ARMS)
    args = parser.parse_args()
    try:
        result = RbtFeedbackReader(args.service, args.controller_id, source_arm=args.source_arm).read()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
