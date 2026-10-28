"""Version-one facts, unknowns and explicitly scoped response observations."""
import hashlib
import json
import math
import re
from datetime import datetime, timezone

STATUSES = {'completed', 'physical_failure', 'preflight_rejected', 'planning_rejected',
            'interrupted', 'infrastructure_error', 'unknown', 'running', 'proposal'}
BOOLS = ('instruction_achieved', 'task_success', 'native_success',
         'approved_alternative_success', 'safety_stop', 'execution_complete',
         'replay_verified', 'training_ready')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def new_record(source_identity, *, task=None, producer='unknown'):
    ident = digest({'source_identity': source_identity})
    return {'schema_version': 1, 'attempt_id': ident, 'source_identity': source_identity,
            'task_id': task, 'producer': producer, 'experiment_id': None,
            'record_kind': 'attempt', 'status': 'unknown',
            'context': {'robot': None, 'hand': None, 'interface': None,
                        'controller_version': None, 'stage': None, 'scene': {},
                        'method': 'unknown', 'original_actions_accessible': None},
            'orchestration': [], 'control': [],
            'outcome': {key: None for key in BOOLS}, 'metrics': {}, 'criteria': {},
            'timing': {'started_at': None, 'ended_at': None, 'planning_s': None,
                       'simulation_s': None, 'rendering_s': None, 'total_s': None,
                       'unknown_reason': 'not available unless explicitly recorded'},
            'relations': {'family': ident, 'parent': None, 'replay_of': None,
                          'checkpoint_source': None, 'shared_plan': None},
            'evidence': [], 'evidence_level': 'reported', 'missing': [], 'notes': []}


def validate_record(r):
    if r.get('schema_version') != 1 or not re.fullmatch(r'[a-f0-9]{64}', r.get('attempt_id', '')):
        raise ValueError('Invalid schema/attempt ID')
    if r['attempt_id'] != digest({'source_identity': r['source_identity']}):
        raise ValueError('Source identity mismatch')
    if r.get('status') not in STATUSES:
        raise ValueError('Invalid outcome status')
    for key in BOOLS:
        if r['outcome'].get(key) is not None and type(r['outcome'][key]) is not bool:
            raise ValueError('Outcome must be boolean or unknown: '+key)
    for kind in ('orchestration', 'control'):
        if not isinstance(r[kind], list):
            raise ValueError('Events must be arrays')
    encoded(r)  # Nonfinite and non-JSON values cannot enter authoritative records.


def classify_response(*, desired, measured, frame, unit, window_s, tolerance):
    """Scalar delta in ONE declared frame/window; never infers causal effects."""
    values = (desired, measured, window_s, tolerance)
    if not frame or not unit or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        return 'unknown'
    if window_s <= 0 or tolerance <= 0 or abs(desired) <= tolerance:
        return 'unknown'
    if abs(measured) <= tolerance:
        return 'no_observed_response'
    if desired * measured < 0:
        return 'opposite'
    if abs(desired-measured) <= tolerance:
        return 'tracked'
    return 'insufficient' if abs(measured) < abs(desired) else 'overshoot'


def summary(r):
    """Deterministic, bounded summary; no retrospective invented reasoning."""
    outcome = ', '.join(f'{k}={v}' for k, v in r['outcome'].items() if v is not None) or 'unknown'
    return (f"任务 / task: {r['task_id'] or 'unknown'}\n"
            f"实验: {r['experiment_id'] or 'unknown'}; 状态: {r['status']}\n"
            f"接口: {r['context'].get('interface') or 'unknown'}; 方法: {r['context'].get('method')}\n"
            f"结果: {outcome}\n编排事件: {len(r['orchestration'])}; 控制事件: {len(r['control'])}\n"
            f"缺失: {', '.join(r['missing']) or 'see source'}\n"
            "范围: 单次记录，不能推断通用规律或成功概率；物理成功不等于视觉训练就绪。\n")
