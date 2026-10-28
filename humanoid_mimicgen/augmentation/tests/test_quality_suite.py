import pytest
from humanoid_mimicgen.augmentation.quality_suite import validate_decision, gate


def decision(action='accept', **extra):
    return dict(action=action, rationale='Native evidence supports this', hypothesis='none', episode=0, tail_s=0., wrist_delta_m=[0.,0.,0.], contact_start=0, contact_end=1, **extra)


def test_failed_replay_cannot_enter_annotation():
    with pytest.raises(ValueError, match='quality'):
        gate('replay', {'success':False,'coverage':1.,'independent_success':False}, decision())


def test_unsupported_strategy_is_not_silently_executed():
    with pytest.raises(ValueError):
        validate_decision(decision('change_physics'), [0])


def test_tail_requires_explicit_bounded_time():
    with pytest.raises(ValueError):
        validate_decision(decision() | {'tail_s':float('nan')}, [0])
    with pytest.raises(ValueError):
        validate_decision(decision() | {'tail_s':10}, [0])


def test_annotations_need_ordered_measured_endpoints():
    with pytest.raises(ValueError):
        gate('annotation', {'success':True,'frames':10,'contact_frames':[3,4,5]}, decision() | {'contact_start':5,'contact_end':3})
    with pytest.raises(ValueError):
        gate('annotation', {'success':True,'frames':10,'contact_frames':[3,4,5]}, decision() | {'contact_start':0,'contact_end':1})


def test_success_without_full_coverage_or_independent_check_rejected():
    for r in [{'success':True,'coverage':.5,'independent_success':True}, {'success':True,'coverage':1.,'independent_success':False}]:
        with pytest.raises(ValueError): gate('zero', r, decision())


def test_strategy_must_match_completed_api_function_call(tmp_path):
    import json
    from humanoid_mimicgen.augmentation.quality_suite import verify_origin
    from humanoid_mimicgen.workflow.contracts import file_hash
    request=tmp_path/'request.json';request.write_text('{}')
    response=tmp_path/'response.json';d=decision('run')
    response.write_text(json.dumps({'status':'completed','output':[{'name':'decide','call_id':'c1','arguments':json.dumps(d)}]}))
    plan=tmp_path/'decision.json'
    record={'decision':d,'origin':{'kind':'api','response':str(response),'response_sha256':file_hash(response),'request':str(request),'request_sha256':file_hash(request),'call_id':'c1'}}
    plan.write_text(json.dumps(record));assert verify_origin(plan)['decision']==d
    record['decision']['tail_s']=1.;plan.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='differs'):verify_origin(plan)


def test_task_metric_adapters_handle_keyword_only_metrics():
    from types import SimpleNamespace
    from humanoid_mimicgen.augmentation.suite_physics import native_metrics
    class Task:
        def _dolly_metrics(self,mujoco_env=None):return {'displacement':.1}
    backend=SimpleNamespace(contacts=lambda:{'robot_object_N':2.,'events':[]},e=SimpleNamespace(task=Task(),mujoco=object()),d=SimpleNamespace(time=0.),object_ids={})
    assert native_metrics(backend,False)['_dolly_metrics']['displacement']==.1


def test_completed_decision_resume_does_not_call_provider_again(tmp_path,monkeypatch):
    from humanoid_mimicgen.augmentation import quality_suite as q
    class Provider:
        config={'model':'test','reasoning_effort':'low'}
        calls=0
        def respond(self,*args):
            import json
            self.calls+=1
            return {'status':'completed','model':'test','usage':{'total_tokens':1},'output':[{'type':'function_call','name':'decide','call_id':'x','arguments':json.dumps(decision('run'))}]}
    provider=Provider();q.API_CURSOR.clear()
    d,p=q.api_decide(provider,tmp_path,{'stage':'source'},[0])
    q.API_CURSOR.clear()
    d2,p2=q.api_decide(provider,tmp_path,{'stage':'source'},[0])
    assert provider.calls==1 and d==d2 and p==p2
    q.API_CURSOR.clear()
    with pytest.raises(ValueError,match='context changed'):
        q.api_decide(provider,tmp_path,{'stage':'another'},[0])


def test_completed_physics_resume_never_integrates_twice(tmp_path,monkeypatch):
    import json
    from humanoid_mimicgen.augmentation import quality_suite as q
    p=tmp_path/'decision.json';p.write_text('{}')
    monkeypatch.setattr(q,'verify_origin',lambda path:None)
    calls=[]
    def run(*args,**kwargs):
        calls.append(1);return {'mode':'original_token','object_delta_xy_m':[0,0]}
    monkeypatch.setattr(q.physics,'physical_case',run)
    q.execute('dataset',tmp_path,'replay',decision('run'),p)
    q.execute('dataset',tmp_path,'replay',decision('run'),p)
    assert len(calls)==1
    with pytest.raises(ValueError,match='parameters changed'):
        q.execute('dataset',tmp_path,'zero',decision('run'),p)


def test_only_effective_control_parameters_must_match():
    from humanoid_mimicgen.augmentation.quality_suite import same_execution
    a=decision('run');b=a|{'contact_start':5,'contact_end':8}
    assert same_execution(a,b)  # Zero wrist translation has no interval-dependent effect.
    assert not same_execution(a,b|{'tail_s':1.})
    a=a|{'wrist_delta_m':[.01,0,0]};b=b|{'wrist_delta_m':[.01,0,0]}
    assert not same_execution(a,b)


def test_unannotated_placeholder_is_allowed_only_without_wrist_correction():
    from humanoid_mimicgen.augmentation.quality_suite import validate_decision
    d=decision('run')|{'contact_start':-1,'contact_end':-1}
    validate_decision(d,[0])
    with pytest.raises(ValueError):validate_decision(d|{'wrist_delta_m':[.01,0,0]},[0])


def test_explicit_safety_guard_is_bounded_and_part_of_execution_identity():
    from humanoid_mimicgen.augmentation.quality_suite import same_execution
    d=decision('run')|{'safety_guard':{'root_height_min_m':.3,'root_tilt_max_deg':75}}
    validate_decision(d,[0])
    assert not same_execution(d,decision('run'))
    with pytest.raises(ValueError):validate_decision(d|{'safety_guard':{'root_height_min_m':.3,'root_tilt_max_deg':float('inf')}},[0])
