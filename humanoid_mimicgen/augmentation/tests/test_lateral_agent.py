import json

import pytest

from humanoid_mimicgen.augmentation.lateral_agent import apply_response, run_session, normalize_json_action
from humanoid_mimicgen.augmentation.provider import ResponsesProvider
from humanoid_mimicgen.workflow.contracts import WorkflowError, file_hash


class Session:
    def __init__(self):
        self.calls = []
        self.state = {'prepared': {}}

    def dispatch(self, name, args, provenance):
        self.calls.append((name, args))
        return {'recorded': True}

    def complete(self):
        return False


def response(status='completed'):
    return {'status': status, 'output': [{'type': 'function_call', 'name': 'inspect_scene',
                                         'call_id': 'same_id', 'arguments': '{}'}]}


def test_incomplete_reply_never_authorizes_tools(tmp_path):
    s = Session()
    with pytest.raises(WorkflowError):
        apply_response(response('incomplete'), s, tmp_path, {}, [])
    assert not s.calls


def test_malformed_later_call_cannot_follow_an_already_dispatched_execution(tmp_path):
    s = Session()
    raw = response()
    raw['output'][0].update(name='execute_grasp', arguments='{"prepared_id":"p","initial_review":"clear"}')
    raw['output'].append({'type': 'function_call', 'name': 'reflect_grasp',
                          'call_id': 'later', 'arguments': '{'})
    with pytest.raises(WorkflowError):
        apply_response(raw, s, tmp_path, {}, [])
    assert not s.calls


def test_duplicate_call_is_replayed_but_changed_arguments_rejected(tmp_path):
    s = Session()
    apply_response(response(), s, tmp_path, {}, [])
    apply_response(response(), s, tmp_path, {}, [])
    assert s.calls == [('inspect_scene', {})]
    bad = response()
    bad['output'][0]['arguments'] = '{"different":true}'
    with pytest.raises(WorkflowError):
        apply_response(bad, s, tmp_path, {}, [])


def test_final_text_cannot_bypass_execution_and_reflection(tmp_path):
    with pytest.raises(WorkflowError):
        apply_response({'status': 'completed', 'output': []}, Session(), tmp_path, {}, [])


def test_explicit_json_protocol_keeps_native_and_normalized_calls_distinct(tmp_path):
    raw = {'status': 'completed', 'id': 'real-response', 'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text':
         '{"tool":"inspect_scene","arguments":{}}'}]}]}
    normalized = normalize_json_action(raw)
    assert raw['output'][0]['type'] == 'message'
    assert normalized['normalization'] == 'explicit_json_action_protocol'
    assert normalized['output'][0]['name'] == 'inspect_scene'
    assert normalized['output'][0]['call_id'].startswith('json_')
    s = Session()
    apply_response(normalized, s, tmp_path, {}, [])
    assert s.calls == [('inspect_scene', {})]


def test_json_finish_cannot_bypass_physical_evidence(tmp_path):
    raw = {'status': 'completed', 'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': '{"tool":"finish","arguments":{"summary":"done"}}'}]}]}
    with pytest.raises(WorkflowError):
        apply_response(normalize_json_action(raw), Session(), tmp_path, {}, [])


def test_json_refusal_cannot_be_dropped_during_normalization(tmp_path):
    raw = {'status': 'completed', 'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': '{"tool":"inspect_scene","arguments":{}}'},
        {'type': 'refusal', 'refusal': 'Refused'}]}]}
    s = Session()
    with pytest.raises(WorkflowError):
        apply_response(normalize_json_action(raw), s, tmp_path, {}, [])
    assert not s.calls


def test_no_tools_request_omits_native_tool_routing_fields(monkeypatch):
    import urllib.request
    captured = []
    class Reply:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return b'{"model":"gpt-6-astra","status":"completed","output":[]}'
    class Opener:
        def open(self, request, timeout):
            captured.append(json.loads(request.data))
            return Reply()
    monkeypatch.setenv('LATERAL_TEST_KEY', 'fixture-only-key')
    monkeypatch.setattr(urllib.request, 'build_opener', lambda *args: Opener())
    provider = ResponsesProvider(dict(protocol='responses', base_url='https://example.test/v1',
        model='gpt-6-astra', key_env='LATERAL_TEST_KEY', timeout_s=5, max_rounds=1,
        max_output_tokens=100, wall_s=10))
    provider.respond([{'role': 'user', 'content': 'test'}], [], 'test')
    assert 'tools' not in captured[0]
    assert 'parallel_tool_calls' not in captured[0]


def test_json_protocol_sends_role_messages_and_no_native_tool_definitions(tmp_path):
    class JSONProvider(FakeProvider):
        def respond(self, items, definitions, instructions):
            assert definitions == []
            assert all('role' in item for item in items)
            assert 'JSON' in instructions
            return {'status': 'completed', 'output': [{'type': 'message', 'content': [
                {'type': 'output_text', 'text': '{"tool":"finish","arguments":{"summary":"recorded"}}'}]}]}
    s = Session()
    s.complete = lambda: True
    result = run_session(tmp_path, {'max_rounds': 1, 'wall_s': 10, 'decision_protocol': 'json_action'}, s, JSONProvider())
    assert result['decision_protocol'] == 'json_action'
    assert not result['native_function_calling']


def test_role_history_can_keep_actual_native_function_calling(tmp_path):
    class RoleProvider(FakeProvider):
        def respond(self, items, definitions, instructions):
            assert definitions
            assert all('role' in x for x in items)
            return super().respond(items, definitions, instructions)
    s = Session()
    s.complete = lambda: True
    result = run_session(tmp_path, {'max_rounds': 1, 'wall_s': 10, 'history_protocol': 'role_messages'}, s, RoleProvider())
    assert result['native_function_calling']
    assert result['history_protocol'] == 'role_messages'


def test_large_local_integrity_index_stays_out_of_model_context(tmp_path):
    class InspectProvider(FakeProvider):
        def respond(self, items, definitions, instructions):
            assert 'UNNEEDED_LOCAL_HASH_PATH' not in json.dumps(items)
            return super().respond(items, definitions, instructions)
    s = Session()
    s.complete = lambda: True
    run_session(tmp_path, {'max_rounds': 1, 'wall_s': 10,
        'input_sha256': {'UNNEEDED_LOCAL_HASH_PATH': 'hash'}}, s, InspectProvider())


class FakeProvider:
    def __init__(self):
        self.config = {'model': 'fixture'}

    def respond(self, items, definitions, instructions):
        return {'status': 'completed', 'output': [], 'usage': {'input_tokens': 1, 'output_tokens': 0, 'total_tokens': 1}}


def test_injected_provider_is_never_reported_as_live_api(tmp_path):
    s = Session()
    s.complete = lambda: True
    result = run_session(tmp_path, {'max_rounds': 1, 'wall_s': 10}, s, FakeProvider())
    assert result['transport'] == 'injected'
    assert result['live_api_verified'] is False
    assert result['api_calls'] == 1


def test_cached_pool_response_requires_bound_origin(tmp_path):
    s = Session()
    s.complete = lambda: True
    cfg = {'max_rounds': 1, 'wall_s': 10}
    run_session(tmp_path, cfg, s, FakeProvider())
    (tmp_path/'agent_result.json').unlink()
    (tmp_path/'api/0000/response_origin.json').unlink()
    provider = FakeProvider()
    provider.slots = [None]
    with pytest.raises(WorkflowError):
        run_session(tmp_path, cfg, s, provider)


def bootstrap_fixture(path, name='read_skill'):
    from humanoid_mimicgen.augmentation.lateral_tools import tool_definitions
    path.mkdir()
    (path/'request.json').write_text(json.dumps({'provider': {'model': 'fixture'}, 'input': [
        {'role': 'user', 'content': 'Transport test: read skill only'}], 'tools': tool_definitions(),
        'instructions': 'Transport check only'}))
    raw = {'status': 'completed', 'model': 'fixture', 'usage': {'input_tokens': 5, 'output_tokens': 2, 'total_tokens': 7},
           'output': [{'type': 'function_call', 'name': name, 'call_id': 'real_prefix', 'arguments': '{}'}]}
    (path/'response.json').write_text(json.dumps(raw))
    (path/'result.json').write_text(json.dumps({'status': 'completed', 'started_at': '2026-10-27T15:10:00+08:00',
                                              'ended_at': '2026-10-27T15:10:01+08:00', 'wall_s': 1}))


def test_audited_prefix_returns_actual_pending_tool_result_before_next_model_call(tmp_path):
    bootstrap_fixture(tmp_path/'probe')
    class PrefixProvider(FakeProvider):
        def respond(self, items, definitions, instructions):
            assert any(x.get('type') == 'function_call_output' and x.get('call_id') == 'real_prefix' for x in items)
            assert 'authorized physical' in items[-1]['content']
            return super().respond(items, definitions, instructions)
    s = Session()
    s.complete = lambda: True
    result = run_session(tmp_path, {'max_rounds': 1, 'wall_s': 10,
        'bootstrap_read_skill_probe': str(tmp_path/'probe')}, s, PrefixProvider())
    assert s.calls == [('read_skill', {})]
    assert result['api_calls'] == 2
    assert not result['live_api_verified']


def test_bootstrap_can_never_import_a_physical_call(tmp_path):
    bootstrap_fixture(tmp_path/'probe', name='execute_grasp')
    s = Session()
    with pytest.raises(WorkflowError):
        run_session(tmp_path, {'max_rounds': 1, 'wall_s': 10,
            'bootstrap_read_skill_probe': str(tmp_path/'probe')}, s, FakeProvider())
    assert not s.calls
