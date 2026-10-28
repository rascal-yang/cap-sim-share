import json
from pathlib import Path
import pytest
from humanoid_mimicgen.augmentation import agent, tools
from humanoid_mimicgen.augmentation.jobs import JobStore
from humanoid_mimicgen.augmentation.provider import ResponsesProvider
from humanoid_mimicgen.workflow.contracts import WorkflowError
from humanoid_mimicgen.augmentation.tests.test_conditions import job


def call(ident, name, arguments):
    return {'type': 'function_call', 'call_id': ident, 'name': name, 'arguments': json.dumps(arguments)}


class Scripted:
    def __init__(self, replies): self.replies = iter(replies); self.inputs = []
    def respond(self, input_items, tools, instructions):
        self.inputs.append(json.loads(json.dumps(input_items)))
        return next(self.replies)


def reply(*items): return {'status': 'completed', 'model': 'test-model', 'output': list(items)}


def setup(tmp_path, monkeypatch):
    j = job(); j['output_root'] = str(tmp_path/'output')
    path = tmp_path/'job.json'; path.write_text(json.dumps(j))
    config = {'protocol': 'responses', 'base_url': 'https://api.example.test/v1', 'model': 'test-model',
              'key_env': 'AUGMENT_TEST_KEY', 'timeout_s': 5, 'max_rounds': 6, 'max_output_tokens': 1000, 'wall_s': 20}
    p = tmp_path/'provider.json'; p.write_text(json.dumps(config))
    store = JobStore(tmp_path/'store')
    monkeypatch.setattr(tools, 'validate_job', lambda j: {'status': 'validated', 'input_hashes': {}, 'physics_steps': 0})
    def complete(store, ident):
        store.claim(ident, {})
        store.transition(ident, 'running', 'completed', {'successes': 1, 'case_count': 1})
        return {'job_id': ident, 'status': 'completed'}
    monkeypatch.setattr(tools, 'start_job', complete)
    return path, p, store


def test_model_reads_skill_and_submits_job(tmp_path, monkeypatch):
    path, config, store = setup(tmp_path, monkeypatch)
    transport = Scripted([reply(call('read', 'read_skill', {'name': 'augmentation'})),
                          reply(call('submit', 'submit_job', {})), reply()])
    result = agent.run_agent(path, config, store, transport=transport)
    assert result['status'] == 'completed' and result['job']['details']['successes'] == 1
    assert result['transport'] == 'injected' and result['api_calls'] == 3
    # The consumer receives actual skill content, not a path inaccessible to the model.
    outputs = [x for x in transport.inputs[1] if x.get('type') == 'function_call_output']
    loaded = json.loads(outputs[0]['output'])
    assert len(loaded['content']) > 100 and loaded['sha256']


def test_repeated_call_reuses_result(tmp_path, monkeypatch):
    path, config, store = setup(tmp_path, monkeypatch)
    transport = Scripted([reply(call('read', 'read_skill', {'name': 'augmentation'})),
                          reply(call('submit', 'submit_job', {})),
                          reply(call('submit', 'submit_job', {})), reply()])
    result = agent.run_agent(path, config, store, transport=transport)
    assert result['job']['details']['successes'] == 1
    assert len(list((store.root/'agents').iterdir())) == 1
    # Restart consumes persisted replies and tool results, without another provider call.
    again = agent.run_agent(path, config, store, transport=Scripted([]))
    assert again['job']['job_id'] == result['job']['job_id']


def test_unknown_tool_rejected(tmp_path, monkeypatch):
    path, config, store = setup(tmp_path, monkeypatch)
    with pytest.raises(WorkflowError, match='unsupported_tool'):
        agent.run_agent(path, config, store, transport=Scripted([reply(call('x', 'shell', {'cmd': 'echo bad'}))]))


@pytest.mark.parametrize('raw', [{'status': 'incomplete', 'output': []},
    reply({'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'no'}]})])
def test_refusal_and_incomplete_response(tmp_path, monkeypatch, raw):
    path, config, store = setup(tmp_path, monkeypatch)
    with pytest.raises(WorkflowError): agent.run_agent(path, config, store, transport=Scripted([raw]))


def test_missing_usage_stays_unknown(tmp_path, monkeypatch):
    path, config, store = setup(tmp_path, monkeypatch)
    transport = Scripted([reply(call('read', 'read_skill', {'name': 'augmentation'})), reply(call('submit', 'submit_job', {})), reply()])
    result = agent.run_agent(path, config, store, transport=transport)
    assert result['usage'] is None


def test_credentials_absent_from_logs(tmp_path, monkeypatch):
    path, config, store = setup(tmp_path, monkeypatch)
    secret = 'test-secret-do-not-log'; monkeypatch.setenv('AUGMENT_TEST_KEY', secret)
    agent.run_agent(path, config, store, transport=Scripted([
        reply(call('read', 'read_skill', {'name': 'augmentation'})), reply(call('submit', 'submit_job', {})), reply()]))
    assert all(secret not in p.read_text() for p in (store.root/'agents').rglob('*.json'))


def test_no_runtime_model_polling(tmp_path, monkeypatch):
    path, config, store = setup(tmp_path, monkeypatch)
    result = agent.run_agent(path, config, store, transport=Scripted([
        reply(call('read', 'read_skill', {'name': 'augmentation'})), reply(call('submit', 'submit_job', {})), reply()]))
    assert result['api_calls'] == 3


def test_empty_model_success_does_not_mean_task_success(tmp_path, monkeypatch):
    path, config, store = setup(tmp_path, monkeypatch)
    with pytest.raises(WorkflowError, match='incomplete_workflow'):
        agent.run_agent(path, config, store, transport=Scripted([reply()]))


def test_provider_rejects_insecure_endpoint_and_absent_key(monkeypatch):
    config = {'protocol': 'responses', 'base_url': 'http://bad', 'model': 'test-model', 'key_env': 'NO_API_TEST_KEY',
              'timeout_s': 1, 'max_rounds': 2, 'max_output_tokens': 100, 'wall_s': 2}
    with pytest.raises(WorkflowError): ResponsesProvider(config)
    config['base_url'] = 'https://api.example.test/v1'
    monkeypatch.delenv('NO_API_TEST_KEY', raising=False)
    with pytest.raises(WorkflowError, match='api_unavailable'):
        ResponsesProvider(config).respond([], [], 'test')


def test_cached_final_rechecks_input_evidence(tmp_path, monkeypatch):
    path,config,store=setup(tmp_path,monkeypatch)
    evidence=tmp_path/'input'; evidence.write_text('original')
    from humanoid_mimicgen.workflow.contracts import file_hash
    monkeypatch.setattr(tools,'validate_job',lambda j:{'status':'validated','input_hashes':{str(evidence):file_hash(evidence)},'physics_steps':0})
    script=Scripted([reply(call('read','read_skill',{'name':'augmentation'})),reply(call('submit','submit_job',{})),reply()])
    agent.run_agent(path,config,store,transport=script)
    evidence.write_text('changed')
    with pytest.raises(WorkflowError,match='integrity_error'):
        agent.run_agent(path,config,store,transport=Scripted([]))


def test_timeout_attempt_usage_remains_unknown_after_retry(tmp_path, monkeypatch):
    path,config,store=setup(tmp_path,monkeypatch)
    class TimeoutAfterRead:
        count=0
        def respond(self,*args):
            self.count+=1
            if self.count==1:
                return dict(reply(call('read','read_skill',{'name':'augmentation'})),usage={'input_tokens':10,'output_tokens':2,'total_tokens':12})
            raise TimeoutError('test')
    with pytest.raises(TimeoutError): agent.run_agent(path,config,store,transport=TimeoutAfterRead())
    valid=lambda *items:dict(reply(*items),usage={'input_tokens':10,'output_tokens':2,'total_tokens':12})
    result=agent.run_agent(path,config,store,transport=Scripted([valid(call('submit','submit_job',{})),valid()]))
    assert result['api_calls']==4
    assert result['usage'] is None


def test_provider_sends_explicit_reasoning_effort(monkeypatch):
    from humanoid_mimicgen.augmentation import provider
    config = {'protocol': 'responses', 'base_url': 'https://api.example.test/v1', 'model': 'gpt-6-astra',
              'key_env': 'AUGMENT_TEST_KEY', 'timeout_s': 5, 'max_rounds': 6,
              'max_output_tokens': 2048, 'wall_s': 60, 'reasoning_effort': 'low'}
    monkeypatch.setenv('AUGMENT_TEST_KEY', 'test-only-secret')
    captured = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return json.dumps({'model': 'gpt-6-astra', 'status': 'completed', 'output': []}).encode()
    class Opener:
        def open(self, request, timeout):
            captured.append(json.loads(request.data))
            return Response()
    monkeypatch.setattr(provider.urllib.request, 'build_opener', lambda *args: Opener())
    ResponsesProvider(config).respond([], [], 'test')
    assert captured[0]['reasoning'] == {'effort': 'low'}
    assert 'test-only-secret' not in json.dumps(captured)
    config['reasoning_effort'] = 'made-up'
    with pytest.raises(WorkflowError): ResponsesProvider(config)


def test_provider_streams_terminal_response_and_identifies_client(monkeypatch):
    import io
    from humanoid_mimicgen.augmentation import provider
    config = {'protocol': 'responses', 'base_url': 'https://api.example.test/v1', 'model': 'gpt-6-astra',
              'key_env': 'AUGMENT_TEST_KEY', 'timeout_s': 5, 'max_rounds': 6,
              'max_output_tokens': 2048, 'wall_s': 60, 'reasoning_effort': 'low', 'stream': True, 'headers_env': {'x-openai-actor-authorization': 'AUGMENT_TEST_ACTOR'}}
    monkeypatch.setenv('AUGMENT_TEST_ACTOR', 'test-actor')
    monkeypatch.setenv('AUGMENT_TEST_KEY', 'test-only-secret')
    result = {'model': 'gpt-6-astra', 'status': 'completed', 'output': [], 'usage': {'total_tokens': 7}}
    events = ['event: response.created\ndata: '+json.dumps({'type':'response.created'})+'\n\n',
              'event: response.completed\ndata: '+json.dumps({'type':'response.completed', 'response':result})+'\n\n']
    class Opener:
        def open(self, request, timeout):
            assert request.get_header('X-openai-actor-authorization') == 'test-actor'
            assert request.get_header('User-agent') == 'cap-sim-augmentation/1.0'
            assert json.loads(request.data)['stream'] is True
            return io.BytesIO(''.join(events).encode())
    monkeypatch.setattr(provider.urllib.request, 'build_opener', lambda *args: Opener())
    assert ResponsesProvider(config).respond([], [], 'test') == result
    events.pop()
    with pytest.raises(WorkflowError, match='invalid_api_response'):
        ResponsesProvider(config).respond([], [], 'test')
