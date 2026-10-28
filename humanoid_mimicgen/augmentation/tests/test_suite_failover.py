import json
from types import SimpleNamespace
import pytest
from humanoid_mimicgen.augmentation import quality_suite as q
from humanoid_mimicgen.workflow.contracts import WorkflowError


def provider(outcome):
    class Stub:
        config={'model':'test','reasoning_effort':'low','base_url':'https://example.test/v1'}
        def respond(self,*args):
            if isinstance(outcome,Exception):raise outcome
            return outcome
    return Stub()


def success():
    d=dict(action='run',rationale='Evidence',hypothesis='Test',episode=0,tail_s=0,wrist_delta_m=[0,0,0],contact_start=-1,contact_end=-1)
    return {'status':'completed','model':'test','usage':{'total_tokens':10},'output':[{'type':'function_call','name':'decide','call_id':'c1','arguments':json.dumps(d)}]}


def test_six_primary_requests_then_secondary_with_complete_audit(tmp_path):
    from humanoid_mimicgen.augmentation.suite_provider import FailoverProvider
    chain=FailoverProvider(provider(WorkflowError('api_unavailable','TimeoutError')),provider(success()))
    q.API_CURSOR.clear()
    d,p=q.api_decide(chain,tmp_path,{'stage':'source'},[0])
    attempts=[json.loads(f.read_text()) for f in sorted(p.parent.glob('attempt_*.json'))]
    assert [a['provider_slot'] for a in attempts]==['primary']*6+['secondary']
    assert [a['status'] for a in attempts]==['error']*6+['completed']
    assert json.loads(p.read_text())['origin']['provider_slot']=='secondary'
    assert q.verify_origin(p)['decision']==d
    # Successful failover stays selected for subsequent uncached stages.
    d,p=q.api_decide(chain,tmp_path,{'stage':'saved_replay'},[0])
    assert json.loads((p.parent/'attempt_0.json').read_text())['provider_slot']=='secondary'


def test_both_providers_exhaust_without_authorizing_physics(tmp_path):
    from humanoid_mimicgen.augmentation.suite_provider import FailoverProvider
    fail=provider(WorkflowError('api_unavailable','TimeoutError'))
    q.API_CURSOR.clear()
    with pytest.raises(WorkflowError):q.api_decide(FailoverProvider(fail,fail),tmp_path,{'stage':'source'},[0])
    assert len(list(tmp_path.rglob('attempt_*.json')))==12
    assert not list(tmp_path.rglob('decision.json'))
    assert not list(tmp_path.rglob('physics.started.json'))


def test_invalid_strategy_does_not_spend_secondary_credentials(tmp_path):
    from humanoid_mimicgen.augmentation.suite_provider import FailoverProvider
    raw=success();raw['output'][0]['arguments']='{}'
    q.API_CURSOR.clear()
    with pytest.raises(ValueError):q.api_decide(FailoverProvider(provider(raw),provider(success())),tmp_path,{'stage':'source'},[0])
    ats=[json.loads(p.read_text()) for p in tmp_path.rglob('attempt_*.json')]
    assert len(ats)==3 and all(a['provider_slot']=='primary' for a in ats)


def test_completed_cached_decision_does_not_use_failover(tmp_path):
    from humanoid_mimicgen.augmentation.suite_provider import FailoverProvider
    q.API_CURSOR.clear();q.api_decide(provider(success()),tmp_path,{'stage':'source'},[0])
    q.API_CURSOR.clear();fail=provider(AssertionError('Cache attempted network'))
    q.api_decide(FailoverProvider(fail,fail),tmp_path,{'stage':'source'},[0])
    assert len(list(tmp_path.rglob('attempt_*.json')))==1


def test_resume_only_archives_service_errors_and_preserves_caches(tmp_path):
    from humanoid_mimicgen.augmentation.suite_batch import resume_service_errors
    for name,status,reason in [('apple','error','RuntimeError: Incomplete API response'),('button','error','WorkflowError: api_unavailable: TimeoutError'),('box','blocked','quality failed'),('tool','error','ValueError: bad contract')]:
        d=tmp_path/name;d.mkdir();(d/'result.json').write_text(json.dumps({'status':status,'reason':reason}));(d/'physics.result.json').write_text('cached')
    resumed=resume_service_errors(tmp_path,['apple','button','box','tool'])
    assert resumed==['apple','button']
    assert not (tmp_path/'apple/result.json').exists()
    assert len(list((tmp_path/'apple').glob('result.service_error_*.json')))==1
    assert (tmp_path/'box/result.json').exists() and (tmp_path/'tool/result.json').exists()
    assert all((tmp_path/n/'physics.result.json').read_text()=='cached' for n in ['apple','button','box','tool'])


def test_launcher_loads_distinct_credentials_without_serializing_secrets(tmp_path):
    import subprocess
    for n in (1,2):
        (tmp_path/f'{n}.toml').write_text(f'''[model_providers.OpenAI]\nbase_url="https://endpoint{n}.test/v1"\nexperimental_bearer_token="secret{n}"\n[model_providers.OpenAI.http_headers]\nx-openai-actor-authorization="actor{n}"\n''')
    code='''import json,os,sys
from scripts.run_quality_batch import configure_credentials
c=configure_credentials(sys.argv[1],sys.argv[2],{'model':'gpt-6-astra','reasoning_effort':'low','base_url':'https://endpoint1.test/v1'})
assert os.environ['AUGMENTATION_API_KEY']=='secret1'
assert os.environ['AUGMENTATION_SECONDARY_API_KEY']=='secret2'
assert os.environ['AUGMENTATION_SECONDARY_ACTOR_HEADER']=='actor2'
print(json.dumps(c))
'''
    proc=subprocess.run(['python','-c',code,str(tmp_path/'1.toml'),str(tmp_path/'2.toml')],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    c=json.loads(proc.stdout)
    assert c['base_url']=='https://endpoint2.test/v1' and c['model']=='gpt-6-astra'
    assert 'secret' not in proc.stdout and 'actor2' not in proc.stdout


def test_rotation_is_shared_across_tasks_and_retries(tmp_path):
    from humanoid_mimicgen.augmentation.suite_provider import AlternatingProvider
    (tmp_path/'a').mkdir();(tmp_path/'b').mkdir()
    route=tmp_path/'routing.json'
    first=AlternatingProvider(provider(WorkflowError('api_unavailable','TimeoutError')),provider(success()),route)
    q.API_CURSOR.clear();_,p=q.api_decide(first,tmp_path/'a',{'stage':'source'},[0])
    ats=[json.loads(f.read_text()) for f in sorted(p.parent.glob('attempt_*.json'))]
    assert [a['provider_slot'] for a in ats]==['primary','secondary']
    assert [a['routing_sequence'] for a in ats]==[0,1]
    other=AlternatingProvider(provider(success()),provider(success()),route)
    _,p=q.api_decide(other,tmp_path/'b',{'stage':'source'},[0])
    assert json.loads((p.parent/'attempt_0.json').read_text())['provider_slot']=='primary'
    _,p=q.api_decide(first,tmp_path/'a',{'stage':'saved_replay'},[0])
    assert json.loads((p.parent/'attempt_0.json').read_text())['provider_slot']=='secondary'


def test_alternating_outage_has_initial_plus_five_retries(tmp_path):
    from humanoid_mimicgen.augmentation.suite_provider import AlternatingProvider
    fail=provider(WorkflowError('api_unavailable','TimeoutError'))
    q.API_CURSOR.clear()
    (tmp_path/'task').mkdir()
    with pytest.raises(WorkflowError):q.api_decide(AlternatingProvider(fail,fail,tmp_path/'route.json'),tmp_path/'task',{'stage':'source'},[0])
    ats=[json.loads(p.read_text()) for p in sorted((tmp_path/'task').rglob('attempt_*.json'))]
    assert [a['provider_slot'] for a in ats]==['primary','secondary']*3
    assert not list((tmp_path/'task').rglob('decision.json'))


def test_shared_route_allocations_are_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from humanoid_mimicgen.augmentation.suite_provider import AlternatingProvider
    def allocate(_):
        p=AlternatingProvider(provider(success()),provider(success()),tmp_path/'route.json')
        slot,_=next(p.attempt_providers())
        return p.routing_sequence,slot
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(allocate,range(24)))
    assert sorted(rows)==[(i,'primary' if i%2==0 else 'secondary') for i in range(24)]


def test_three_endpoints_rotate_and_keep_six_attempt_cap(tmp_path):
    from humanoid_mimicgen.augmentation.suite_provider import AlternatingProvider
    fail=provider(WorkflowError('api_unavailable','TimeoutError'))
    chain=AlternatingProvider(fail,fail,tmp_path/'route.json',fail)
    assert [slot for slot,_ in chain.attempt_providers()]==['primary','secondary','tertiary']*2
    assert next(chain.attempt_providers())[0]=='primary'


def test_third_provider_success_has_audited_origin(tmp_path):
    from humanoid_mimicgen.augmentation.suite_provider import AlternatingProvider
    fail=provider(WorkflowError('api_unavailable','TimeoutError'))
    q.API_CURSOR.clear()
    _,p=q.api_decide(AlternatingProvider(fail,fail,tmp_path/'route.json',provider(success())),tmp_path,{'stage':'source'},[0])
    assert q.verify_origin(p)['origin']['provider_slot']=='tertiary'
