import json
from pathlib import Path
from humanoid_mimicgen.experience_dataset.store import Dataset
from humanoid_mimicgen.experience_dataset.ingest import ingest, adapt, inventory, Resolver


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def suite(root, name='case'):
    case = root/name
    write(case/'experiment.json', {'experiment_id': name, 'robot': {'name': 'G1'}, 'pipeline': {'git_commit_and_dirty_patch': {'commit': 'abc'}}})
    write(case/'compiled_plan.json', {'decision': {'action': 'run', 'wrist_delta_m': [0,0,.05], 'rationale': '抬升不足', 'api_key': 'DO_NOT_COPY'}, 'origin': {'kind': 'api_compiled'}})
    write(case/'result.json', {'experiment_id': name, 'task': 'G1WholebodyTableApplePickTeleop', 'physics_steps': 8, 'success': False, 'native_success': False, 'failure': 'contact_entry_failed', 'mode': 'token_prefix_mode1', 'decision_path': str(case/'compiled_plan.json')})
    return case


def test_suite_import_unknowns_redaction_and_idempotence(tmp_path):
    root = tmp_path/'experiments'; case = suite(root)
    ds = Dataset(tmp_path/'dataset')
    first = ingest(ds, [root]); second = ingest(ds, [root])
    assert first['counts']['imported'] == 1
    assert second['counts']['unchanged'] == 1
    r = adapt(case, Resolver([root]))
    assert r['status'] == 'physical_failure'
    assert r['outcome']['replay_verified'] is None
    assert r['orchestration'][0]['instruction']['wrist_delta_m'] == [0,0,.05]
    assert 'DO_NOT_COPY' not in json.dumps(r)
    assert r['control'][0]['instruction_id'] == r['orchestration'][0]['instruction_id']


def test_core_workflow_and_unknown_tasks_are_covered(tmp_path):
    root = tmp_path/'exp'
    write(root/'generic'/'task.json', {'name': 'CustomTask', 'backend': 'demo'})
    write(root/'generic'/'attempts'/'001'/'result.json', {'attempt': 1, 'acceptance': {'success': True}, 'metrics': {}, 'failure': None})
    write(root/'workflow'/'experiment.json', {'experiment_id': 'workflow', 'workflow': {'strategy': {}}, 'inputs': {}})
    write(root/'workflow'/'result.json', {'physical_task_completed': True, 'native': {'success': False}, 'actual_modes': [1]})
    write(root/'odd'/'result.json', {'mystery': 4})
    ds = Dataset(tmp_path/'ds'); report = ingest(ds, [root])
    assert report['discovered'] == 3
    assert report['counts']['partial'] >= 1
    assert ds.stats()['attempts'] == 3
    core = adapt(root/'generic'/'attempts'/'001', Resolver([root]))
    assert core['task_id'] == 'CustomTask'
    assert core['outcome']['task_success'] is True
    workflow = adapt(root/'workflow', Resolver([root]))
    assert workflow['outcome']['native_success'] is False
    assert workflow['outcome']['task_success'] is True


def test_replay_duplicates_and_infrastructure_not_physics_failure(tmp_path):
    root = tmp_path/'exp'; case = suite(root)
    write(root/'summary.json', {'result': json.loads((case/'result.json').read_text())})
    write(root/'success'/'001'/'result.json', json.loads((case/'result.json').read_text()))
    write(root/'rejected'/'preflight_failure.json', {'reason_code': 'missing_model'})
    write(root/'broken'/'result.json', {'status': 'error', 'error_type': 'TimeoutError'})
    paths = inventory([root])
    assert len(paths) == 3
    reject = adapt(root/'rejected', Resolver([root]))
    assert reject['status'] == 'preflight_rejected'
    assert reject['outcome']['task_success'] is None
    assert adapt(root/'broken', Resolver([root]))['status'] == 'infrastructure_error'


def test_symlink_escape_not_read_and_truncated_json_retained(tmp_path):
    root = tmp_path/'exp'; root.mkdir()
    outside = write(tmp_path/'private'/'result.json', {'secret': 'hidden'})
    (root/'escape').symlink_to(outside.parent, target_is_directory=True)
    write(root/'bad'/'result.json', {})
    (root/'bad'/'result.json').write_text('{')
    ds = Dataset(tmp_path/'ds'); report = ingest(ds, [root])
    assert report['discovered'] == 1
    assert report['counts']['error'] == 1
    assert report['entries'][0]['error_type'] == 'JSONDecodeError'
    assert Resolver([root]).resolve(outside) is None


def test_new_stage_artifact_produces_revision_without_duplicate(tmp_path):
    root = tmp_path/'exp'; case = suite(root); ds = Dataset(tmp_path/'ds')
    ingest(ds, [root])
    write(case/'execution'/'metrics.json', {'max_lift_m': .04})
    report = ingest(ds, [root])
    assert report['counts']['imported'] == 1
    assert ds.stats()['attempts'] == 1


def test_dense_metrics_are_references_and_observation_is_preserved(tmp_path):
    root=tmp_path/'exp'; case=suite(root)
    write(case/'request.json', {'observation': {'wrist_z': .7}, 'interface': {'name':'mode1'}})
    write(case/'execution'/'metrics.json', [{'frame':i,'force':i*.1} for i in range(10000)])
    r=adapt(case,Resolver([root]))
    assert len(json.dumps(r)) < 25000
    assert r['orchestration'][0]['observation_before'] == {'wrist_z':.7}
    assert any(e['role']=='execution/metrics.json' for e in r['evidence'])


def test_workflow_strategy_has_orchestration_events(tmp_path):
    root=tmp_path/'exp'; case=root/'workflow'
    write(case/'experiment.json', {'workflow':{'strategy':{'policy':{'navigation':{'speed_m_s':.5}},'phases':[{'id':'grasp'}]}}})
    write(case/'task.json', {'name':'Apple','robot':{'name':'G1','hand':'Dex3'}})
    write(case/'result.json', {'physical_task_completed':True,'actual_modes':[1]})
    r=adapt(case,Resolver([root]))
    assert r['orchestration']
    assert r['context']['robot']=='G1'


def test_summary_only_history_is_partial_not_invented_execution(tmp_path):
    root=tmp_path/'exp'
    write(root/'legacy'/'summary.json', {'success':True,'note':'old report'})
    ds=Dataset(tmp_path/'ds'); report=ingest(ds,[root])
    assert report['discovered']==1
    r=adapt(root/'legacy',Resolver([root]))
    assert r['record_kind']=='report'
    assert r['outcome']['task_success'] is None


def test_instruction_rejection_is_not_physical_failure(tmp_path):
    root=tmp_path/'exp'; case=root/'proposal'
    write(case/'decision.json', {'decision':{'action':'reject','rationale':'unsupported foot control'}})
    r=adapt(case,Resolver([root]))
    assert r['status']=='planning_rejected'
    assert r['outcome']['task_success'] is None


def test_robot_and_task_metadata_are_bound_to_evidence(tmp_path):
    root=tmp_path/'exp'; case=root/'run/attempts/001'
    write(root/'run/task.json', {'name':'Custom','backend':'demo'})
    write(case/'result.json', {'attempt':1,'acceptance':{'success':True}})
    r=adapt(case,Resolver([root]))
    assert any(e['path']==str(root/'run/task.json') for e in r['evidence'])


def test_null_native_result_and_request_only_failure_are_retained(tmp_path):
    root=tmp_path/'exp'
    write(root/'workflow/result.json', {'physical_task_completed':False,'native':None})
    write(root/'planner/request.json', {'task':{'name':'custom'},'observation':{'problem':'initial'}})
    report=ingest(Dataset(tmp_path/'ds'),[root])
    assert report['discovered']==2
    assert report['counts'].get('error',0)==0


def test_measured_evidence_is_bound_and_mutation_during_read_rejected(tmp_path, monkeypatch):
    import humanoid_mimicgen.experience_dataset.ingest as module
    root=tmp_path/'exp'; case=suite(root)
    result=json.loads((case/'result.json').read_text())
    result['evidence']={'observations':'execution/observations.json'}
    write(case/'result.json',result); write(case/'execution/observations.json',[{'force':1}])
    r=adapt(case,Resolver([root]))
    assert any(e['role']=='referenced:observations' for e in r['evidence'])
    original=module.read
    def changed(path,resolver,**kwargs):
        value=original(path,resolver,**kwargs)
        if Path(path)==case/'result.json':
            write(case/'result.json',dict(result,success=True))
        return value
    monkeypatch.setattr(module,'read',changed)
    import pytest
    with pytest.raises(ValueError,match='changed'):
        adapt(case,Resolver([root]))


def test_rescan_cannot_revalidate_deleted_or_changed_terminal_support(tmp_path):
    from humanoid_mimicgen.experience_dataset.retrieval import evidence_valid
    root=tmp_path/'exp'; case=suite(root); ds=Dataset(tmp_path/'ds')
    support=case/'execution/trajectory.npz'; support.parent.mkdir(); support.write_bytes(b'original')
    first=ingest(ds,[root])['entries'][0]
    assert evidence_valid(ds.get(first['attempt_id']))
    support.write_bytes(b'changed')
    ingest(ds,[root]); assert not evidence_valid(ds.get(first['attempt_id']))
    support.unlink()
    ingest(ds,[root]); assert not evidence_valid(ds.get(first['attempt_id']))


def test_missing_reference_creation_is_detected(tmp_path):
    root=tmp_path/'exp'; case=suite(root); ds=Dataset(tmp_path/'ds')
    result=json.loads((case/'result.json').read_text()); result['evidence']={'measurements':'later.json'}
    write(case/'result.json',result)
    first=ingest(ds,[root])['entries'][0]
    assert 'evidence:referenced:measurements' in ds.get(first['attempt_id'])['missing']
    write(case/'later.json',{'force':1})
    second=ingest(ds,[root])
    assert second['counts'].get('unchanged',0)==0
    assert 'evidence:referenced:measurements' not in ds.get(first['attempt_id'])['missing']


def test_controller_identity_uses_frozen_code_not_just_commit(tmp_path):
    root=tmp_path/'exp'; case=suite(root)
    c=json.loads((case/'experiment.json').read_text())
    c['pipeline']['source_snapshot_and_hashes']={'controller.py':'a'*64}
    write(case/'experiment.json',c); first=adapt(case,Resolver([root]))
    c['pipeline']['source_snapshot_and_hashes']={'controller.py':'b'*64}
    write(case/'experiment.json',c); second=adapt(case,Resolver([root]))
    assert first['context']['controller_version']!=second['context']['controller_version']


def test_legacy_string_contract_sections_remain_partial_records(tmp_path):
    root=tmp_path/'exp'; case=root/'legacy'
    write(case/'experiment.json', {'physics':'historical runtime', 'pipeline':'old script', 'runtime':None})
    report=ingest(Dataset(tmp_path/'ds'),[root])
    assert report['counts'].get('error',0)==0
    assert report['counts']['partial']==1


def test_core_stage_measurements_link_to_actual_feedback_decision(tmp_path):
    root=tmp_path/'exp'; case=root/'core/attempts/001'
    write(case/'result.json', {'attempt':1,'acceptance':{'success':False}})
    write(case/'request.json', {'observation':{'time':0}})
    write(case/'decision.json', {'actions':[{'stage':'approach'}]})
    write(case/'feedback_001_request.json', {'observation':{'time':1},'executed_actions':[{'stage':'approach'}]})
    write(case/'feedback_001_decision.json', {'actions':[{'stage':'lift'}]})
    write(case/'diagnosis_002.json', {'stage':'lift','targets':{'right_wrist_position_m':[0,0,1]},'measurements':{'right_wrist_position_m':[0,0,.7]},'start_s':1,'end_s':2})
    r=adapt(case,Resolver([root]))
    assert r['orchestration'][1]['observation_before']=={'time':1}
    control=next(c for c in r['control'] if c.get('stage')=='lift')
    assert control['instruction_id']==r['orchestration'][1]['instruction_id']
    assert control['desired']['right_wrist_position_m']==[0,0,1]
    assert control['measurements']['right_wrist_position_m']==[0,0,.7]


def test_crash_before_ledger_cannot_reset_terminal_evidence(tmp_path,monkeypatch):
    import pytest
    import humanoid_mimicgen.experience_dataset.ingest as module
    from humanoid_mimicgen.experience_dataset.retrieval import evidence_valid
    root=tmp_path/'exp'; case=suite(root); ds=Dataset(tmp_path/'ds')
    support=case/'execution/trajectory.npz'; support.parent.mkdir(); support.write_bytes(b'original')
    original=module.atomic
    def crash(path,value):
        if Path(path).name=='sources.json': raise OSError('simulated ledger write failure')
        return original(path,value)
    monkeypatch.setattr(module,'atomic',crash)
    with pytest.raises(OSError): ingest(ds,[root])
    support.write_bytes(b'changed')
    monkeypatch.setattr(module,'atomic',original)
    report=ingest(ds,[root]); ident=report['entries'][0]['attempt_id']
    assert not evidence_valid(ds.get(ident))
