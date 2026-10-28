import json
import subprocess
import sys
from pathlib import Path

import pytest

from humanoid_mimicgen.augmentation.provider import ResponsesProvider
from humanoid_mimicgen.workflow.contracts import WorkflowError


def config():
    return dict(protocol='responses', base_url='https://example.test/v1', model='gpt-6-astra',
                key_env='ABSENT_TRANSPORT_TEST_KEY', timeout_s=5, max_rounds=1,
                max_output_tokens=128, wall_s=10, transport_python=sys.executable)


def test_external_transport_does_not_forward_recursive_interpreter_setting(monkeypatch):
    seen = []
    def run(argv, **kwargs):
        payload = json.loads(kwargs['input'])
        seen.append((argv, payload))
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(dict(ok=True,
            response={'model': 'gpt-6-astra', 'status': 'completed', 'output': []},
            transport_runtime={'python': 'fixture', 'ssl': 'fixture'})).encode(), stderr=b'')
    monkeypatch.setattr(subprocess, 'run', run)
    p = ResponsesProvider(config())
    assert p.respond([], [], 'test')['model'] == 'gpt-6-astra'
    assert 'transport_python' not in seen[0][1]['config']
    assert seen[0][0][:3] == [sys.executable, '-m', 'humanoid_mimicgen.augmentation.provider_transport']
    assert p.last_transport_runtime['python'] == 'fixture'


def test_external_transport_reported_model_still_checked(monkeypatch):
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(dict(ok=True,
            response={'model': 'different'}, transport_runtime={})).encode(), stderr=b'')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(WorkflowError):
        ResponsesProvider(config()).respond([], [], 'test')


def test_transport_worker_missing_credentials_is_structured_and_secret_free(monkeypatch):
    import os
    env = os.environ.copy()
    env.pop('ABSENT_TRANSPORT_TEST_KEY', None)
    cfg = config()
    cfg.pop('transport_python')
    r = subprocess.run([sys.executable, '-m', 'humanoid_mimicgen.augmentation.provider_transport'],
                       input=json.dumps(dict(config=cfg, input=[], tools=[], instructions='test')).encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=10)
    value = json.loads(r.stdout)
    assert not value['ok']
    assert value['reason_code'] == 'api_unavailable'
    assert 'Authorization' not in r.stdout.decode()
