"""Versioned job inputs. Reading and validation never advance physics."""
import json
import os
from pathlib import Path
import re

from ..workflow.contracts import WorkflowError, canonical, file_hash, positive, validate_strategy
from ..workflow.profile import validate_policy

FIELDS = {'schema_version', 'request_id', 'baseline_request', 'conditions', 'budgets',
          'output_root', 'render', 'export', 'stance_hint_frame'}


def require(condition, detail, code='invalid_input'):
    if not condition:
        raise WorkflowError(code, detail)


def validate_shape(job):
    canonical(job)
    require(isinstance(job, dict) and FIELDS <= set(job) <= FIELDS | {'motion_noise'}, 'Job fields')
    require(type(job['schema_version']) is int and job['schema_version'] == 1, 'Job schema version')
    require(isinstance(job['request_id'], str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', job['request_id']), 'Request ID')
    for key in ('baseline_request', 'output_root'):
        require(isinstance(job[key], str) and bool(job[key]), key)
    for key in ('render', 'export'):
        require(type(job[key]) is bool, key)
    require(job['stance_hint_frame'] in ('world', 'object_relative'), 'Explicit stance hint frame')
    if 'motion_noise' in job:
        from .noise import validate_noise
        validate_noise(job['motion_noise'])
    b = job['budgets']
    require(isinstance(b, dict) and set(b) == {'max_cases', 'wall_s'}, 'Budget fields')
    require(type(b['max_cases']) is int and b['max_cases'] > 0, 'Case budget')
    positive(b['wall_s'], 'Wall budget')
    return job


def load_job(path):
    path = Path(path).resolve()
    try:
        job = json.loads(path.read_text())
        validate_shape(job)
        for key in ('baseline_request', 'output_root'):
            job[key] = os.path.abspath(path.parent / job[key])
        return job
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError('invalid_input', 'Unreadable job: ' + type(exc).__name__) from None


def baseline_request(job):
    path = Path(os.path.abspath(job['baseline_request']))
    try:
        request = json.loads(path.read_text())
        require(request['decision'] == {'mode': 'deterministic'}, 'Deterministic execution required')
        require(request['strategy']['adapter'] == 'g1_sonic_native_v1', 'Unsupported adapter')
        require('grasp_trigger' not in request['strategy']['policy'], 'Runtime model grasp trigger disabled')
        validate_strategy(request['strategy'])
        validate_policy(request['strategy']['policy'])
        request['files'] = {k: os.path.abspath(path.parent / v) for k, v in request['files'].items()}
        for name in ('model', 'initial', 'targets', 'reference_states', 'task', 'record', 'alignment'):
            require(name in request['files'], 'Missing evidence ' + name)
        return request
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError('invalid_input', 'Unreadable baseline: ' + type(exc).__name__) from None


def verify_inputs(hashes):
    for path, expected in hashes.items():
        require(Path(path).is_file() and file_hash(path) == expected,
                'Input changed: ' + path, 'integrity_error')


def validate_job(job):
    from .conditions import sample_conditions
    cases = sample_conditions(job)
    request = baseline_request(job)
    files = {os.path.abspath(job['baseline_request']), *request['files'].values()}
    # Bind nested evidence and resources before queueing. Runtime loading remains prepare's responsibility.
    try:
        record = json.loads(Path(request['files']['record']).read_text())
        existing = record['preflight']['evidence_sha256']
        verify_inputs(existing)
        files.update(existing)
        files.add(record['physics']['physics_parameters_manifest'])
        task = json.loads(Path(request['files']['task']).read_text())
        from ..workflow.native import sonic_files
        files.update(sonic_files(task))
        hashes = {os.path.abspath(p): file_hash(p) for p in sorted(files)}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError('invalid_input', 'Incomplete evidence: ' + type(exc).__name__) from None
    return {'status': 'validated', 'physics_steps': 0, 'case_count': len(cases),
            'input_hashes': hashes, 'sampling': {'algorithm': 'PCG64/uniform', 'numpy_version': __import__('numpy').__version__}}
