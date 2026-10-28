"""Versioned workflow data. No simulator import and no task-specific defaults."""
import hashlib
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo


class WorkflowError(ValueError):
    def __init__(self, code, detail=''):
        self.code, self.detail = code, detail
        super().__init__(f'{code}: {detail}')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, TypeError) as exc:
        raise WorkflowError('invalid_input', 'Expected finite JSON data') from exc


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def positive(value, name, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise WorkflowError('invalid_input', name)
    return value


def validate_strategy(s):
    canonical(s)
    fields = {'schema_version','parent_id','kind','adapter','context','bindings','phases',
              'command_pose_offset','source_frame_count','policy','proposal','evidence','provenance'}
    if not isinstance(s, dict) or set(s) != fields or type(s['schema_version']) is not int or s['schema_version'] != 1:
        raise WorkflowError('invalid_input', 'Strategy schema mismatch')
    if s['kind'] != 'trajectory_retargeting' or s['provenance'].get('original_sequences_accessible') is not True:
        raise WorkflowError('invalid_input', 'Source access must be declared')
    if not isinstance(s['adapter'], str) or not s['adapter']:
        raise WorkflowError('invalid_input', 'Missing adapter')
    for name in ('context','bindings','policy','proposal','evidence','provenance'):
        if not isinstance(s[name], dict):raise WorkflowError('invalid_input', name)
    hands=s['bindings'].get('hands')
    if not isinstance(hands,list) or not hands or len(set(hands))!=len(hands) or not set(hands)<={'left','right'}:
        raise WorkflowError('unsupported_capability','Expected named active hands')
    if not isinstance(s['bindings'].get('object'),str) or not s['bindings']['object']:
        raise WorkflowError('invalid_input','Object binding required')
    n=s['source_frame_count'];offset=s['command_pose_offset']
    if type(n) is not int or n<2 or type(offset) is not int or offset not in (-1,0):
        raise WorkflowError('invalid_input','Explicit source alignment required')
    phases=s['phases'];previous=None;names=set()
    if not isinstance(phases,list) or not phases:raise WorkflowError('invalid_input','No phases')
    for p in phases:
        if not isinstance(p,dict) or set(p)!={'id','start','end','requires_contact','requires_clearance'}:
            raise WorkflowError('invalid_input','Phase schema mismatch')
        if not isinstance(p['id'],str) or not p['id'] or p['id'] in names:
            raise WorkflowError('invalid_input','Unique phase identifiers required')
        names.add(p['id'])
        if any(type(p[x]) is not int for x in ('start','end')) or not 0<=p['start']<p['end']<=n or p['start']+offset<0:
            raise WorkflowError('invalid_input','Invalid half-open source interval')
        if previous is not None and p['start']!=previous:raise WorkflowError('invalid_input','Contact phases must be contiguous')
        if any(type(p[x]) is not bool for x in ('requires_contact','requires_clearance')):
            raise WorkflowError('invalid_input','Phase conditions must be booleans')
        if p['requires_clearance'] and not p['requires_contact']:raise WorkflowError('invalid_input','Clearance phase needs contact')
        previous=p['end']
    return s


def phase_at(phases, frame):
    for phase in phases:
        if phase['start'] <= frame < phase['end']:return phase
    raise WorkflowError('invalid_input','Source frame outside phase schedule')
