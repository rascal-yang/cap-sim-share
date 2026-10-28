import json
import pytest
from humanoid_mimicgen.workflow.batch import validate_manifest, load_request, summarize_run
from humanoid_mimicgen.workflow.contracts import WorkflowError


def test_batch_rejects_ambiguous_cases_and_requires_budget():
    base={'experiment_root':'/tmp','max_cases':2,'render':False,'cases':[{'id':'a','request':'a.json'}]}
    assert validate_manifest(base)['max_cases']==2
    for patch in [{'max_cases':0},{'cases':base['cases']*2},{'cases':[{'id':'../escape','request':'a.json'}]},{'render':'yes'}]:
        with pytest.raises(WorkflowError):validate_manifest(dict(base,**patch))


def test_batch_refuses_live_model_decisions(tmp_path):
    f=tmp_path/'r.json';f.write_text(json.dumps({'decision':{'mode':'responses'}}))
    with pytest.raises(WorkflowError):load_request(f)


def test_summary_keeps_first_and_final_navigation_errors(tmp_path):
    p=tmp_path/'execution';p.mkdir()
    (p/'execution_audit.json').write_text(json.dumps({'sim_seconds':20,'simulation_wall_s':8,'online_planning_wall_s':1,'navigation_corrections':1,'api_calls_during_execution':0}))
    (p/'events.json').write_text(json.dumps([{'event':'navigation_settled','error_m':.08,'time_s':4},{'event':'navigation_settled','error_m':.02,'time_s':8}]))
    r=summarize_run(tmp_path,{'status':'completed','physical_task_completed':True,'reason_code':None})
    assert r['first_settled_error_m']==.08 and r['final_settled_error_m']==.02
    assert r['navigation_corrections']==1 and r['success']


def test_batch_blocks_changed_referenced_input_before_prepare(tmp_path,monkeypatch):
    from humanoid_mimicgen.workflow import batch,service
    initial=tmp_path/'initial';initial.write_text('original')
    request=tmp_path/'request.json';request.write_text(json.dumps({'decision':{'mode':'deterministic'},'strategy':{'policy':{}},'files':{'initial':str(initial)},'strategy_root':str(tmp_path/'strategies'),'experiment_root':str(tmp_path)}))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'experiment_root':str(tmp_path),'max_cases':1,'cases':[{'id':'a','request':str(request)}]}))
    monkeypatch.setattr(service,'implementation_files',lambda:{})
    def mutate(r):
        initial.write_text('changed');return {'strategy_id':'test'}
    monkeypatch.setattr(service,'plan',mutate)
    monkeypatch.setattr(service,'prepare',lambda *a:pytest.fail('Changed input must block preparation'))
    result=batch.run_batch(manifest)
    assert result['rows'][0]['reason']=='integrity_error' and not result['rows'][0]['success']


def test_batch_preserves_calls_when_execute_raises_after_model_work(tmp_path,monkeypatch):
    from humanoid_mimicgen.workflow import batch,service
    request=tmp_path/'request.json';request.write_text(json.dumps({'decision':{'mode':'deterministic'},'strategy':{'policy':{}},'files':{},'strategy_root':str(tmp_path/'strategies'),'experiment_root':str(tmp_path)}))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'experiment_root':str(tmp_path),'max_cases':1,'cases':[{'id':'a','request':str(request)}]}))
    experiment=tmp_path/'physical';(experiment/'execution').mkdir(parents=True)
    monkeypatch.setattr(service,'implementation_files',lambda:{})
    monkeypatch.setattr(service,'plan',lambda r:{'strategy_id':'test'})
    monkeypatch.setattr(service,'prepare',lambda *args:{'experiment_path':str(experiment)})
    def fail(path):
        (path/'execution/execution_audit.json').write_text(json.dumps({'api_calls_during_execution':3}))
        raise WorkflowError('integrity_error','post-execution verification failure')
    monkeypatch.setattr(service,'execute',fail)
    result=batch.run_batch(manifest)
    assert result['api_calls']==3 and result['rows'][0]['api_calls']==3
    assert result['rows'][0]['reason']=='integrity_error' and not result['rows'][0]['success']
