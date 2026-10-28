import json
from pathlib import Path

import pytest

from humanoid_mimicgen.augmentation.api_pool import RotatingResponsesProvider
from humanoid_mimicgen.workflow.contracts import WorkflowError


def config(model='gpt-6-astra'):
    return dict(protocol='responses', base_url='https://example.test/v1', model=model,
                key_env='TEST_KEY', timeout_s=5, max_rounds=10, max_output_tokens=128,
                wall_s=60, reasoning_effort='low')


class Slot:
    def __init__(self, label, fail=False):
        self.label = label
        self.fail = fail
        self.config = config()
        self.calls = 0

    def respond(self, *args):
        self.calls += 1
        if self.fail:
            raise WorkflowError('api_unavailable', 'fixture')
        return {'model': 'gpt-6-astra', 'status': 'completed', 'output': []}


def test_every_request_rotates_including_failed_requests_and_restart(tmp_path):
    a, b, c = Slot('api9', True), Slot('api8'), Slot('api7')
    pool = RotatingResponsesProvider(config(), [('api9', a), ('api8', b), ('api7', c)], tmp_path/'rotation.json')
    with pytest.raises(WorkflowError): pool.respond([], [], 'test')
    assert pool.last_provider_label == 'api9'
    pool.respond([], [], 'test')
    assert pool.last_provider_label == 'api8'
    resumed = RotatingResponsesProvider(config(), [('api9', a), ('api8', b), ('api7', c)], tmp_path/'rotation.json')
    resumed.respond([], [], 'test')
    assert resumed.last_provider_label == 'api7'
    assert [a.calls, b.calls, c.calls] == [1, 1, 1]
    assert json.loads((tmp_path/'rotation.json').read_text())['next_sequence'] == 3


def test_pool_cannot_change_model_between_credentials(tmp_path):
    slot = Slot('different')
    slot.config['model'] = 'another-model'
    with pytest.raises(WorkflowError):
        RotatingResponsesProvider(config(), [('different', slot)], tmp_path/'rotation.json')


def test_empty_existing_routing_record_cannot_reset_rotation(tmp_path):
    slot = Slot('api9')
    path = tmp_path/'rotation.json'
    path.write_text('')
    pool = RotatingResponsesProvider(config(), [('api9', slot)], path)
    with pytest.raises(WorkflowError):
        pool.respond([], [], 'test')
    assert slot.calls == 0
