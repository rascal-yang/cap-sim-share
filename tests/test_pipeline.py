import copy
import pytest
from cap_sim.io import read, write
from test_core import Backend, task

class Planner:
    def __init__(self, fail_recovery=False):
        self.requests=[]
        self.fail_recovery=fail_recovery
    def plan(self, request):
        self.requests.append(request)
        fail=self.fail_recovery and request['task'].get('generation_kind')=='reference_recovery'
        return {'analysis':{'evidence':['measured state'],'hypothesis':'test recovery gate'},
                'actions':[{'name':'bad' if fail else 'win'}]}

def setup(tmp_path):
    source=task();source['budget']['optimization_attempts']=0
    ref=tmp_path/'ref.json';write(ref,{'source_acceptance':{'success':True},'actions':[]})
    target=copy.deepcopy(source);target['scene']={'offset':.1};target['references']=[str(ref)]
    return source,target

def test_failed_recovery_blocks_generation_despite_source_success(tmp_path):
    from cap_sim.pipeline import run_pipeline
    source,target=setup(tmp_path);p=Planner(True)
    result=run_pipeline(target,lambda out:p,lambda t:Backend(),tmp_path/'run',reference_task=source)
    assert result['status']=='reference_recovery_failed'
    assert not (tmp_path/'run/generation').exists()
    assert all(q['task']['scene']==source['scene'] for q in p.requests)

def test_recovery_success_precedes_changed_scene_and_supplies_new_baseline(tmp_path):
    from cap_sim.pipeline import run_pipeline
    source,target=setup(tmp_path);p=Planner()
    result=run_pipeline(target,lambda out:p,lambda t:Backend(),tmp_path/'run',reference_task=source)
    assert result['success']
    assert [q['task']['scene'] for q in p.requests]==[{}, {'offset':.1}]
    recovered=p.requests[1]['references'][0]
    assert recovered['acceptance']['success'] and recovered['task']['scene']=={}
    assert recovered['actions']==[{'name':'win'}]
    assert read(tmp_path/'run/reference_recovery/summary.json')['success']

@pytest.mark.parametrize('field,value',[('backend','other'),('acceptance',{'lift_m':.01,'hold_s':1}),('resources',{'controller':'other'})])
def test_mismatch_rejected_before_execution(tmp_path,field,value):
    from cap_sim.pipeline import run_pipeline
    source,target=setup(tmp_path);source[field]=value;p=Planner()
    with pytest.raises(ValueError,match='same'):
        run_pipeline(target,lambda out:p,lambda t:Backend(),tmp_path/'run',reference_task=source)
    assert not p.requests

def test_reference_requires_explicit_original_scene(tmp_path):
    from cap_sim.pipeline import run_pipeline
    _,target=setup(tmp_path)
    with pytest.raises(ValueError,match='reference_task'):
        run_pipeline(target,lambda out:Planner(),lambda t:Backend(),tmp_path/'run')

def test_augmentation_recovers_once_before_all_variants(tmp_path):
    from cap_sim.pipeline import augment_pipeline
    source,target=setup(tmp_path);p=Planner()
    result=augment_pipeline(target,[{'offset':.1},{'offset':.2}],lambda out:p,lambda t:Backend(),tmp_path/'run',reference_task=source)
    assert result['success_count']==2
    assert [q['task']['scene'] for q in p.requests]==[{}, {'offset':.1}, {'offset':.2}]

def test_resume_rechecks_recovered_evidence_before_generation(tmp_path):
    from cap_sim.pipeline import run_pipeline
    source,target=setup(tmp_path);p=Planner();out=tmp_path/'run'
    run_pipeline(target,lambda out:p,lambda t:Backend(),out,reference_task=source)
    write(out/'reference_recovery/attempts/001/samples.json',[])
    resumed=Planner()
    with pytest.raises(RuntimeError,match='evidence does not match'):
        run_pipeline(target,lambda out:resumed,lambda t:Backend(),out,reference_task=source,resume=True)
    assert not resumed.requests

def test_failed_recovery_blocks_all_variants(tmp_path):
    from cap_sim.pipeline import augment_pipeline
    source,target=setup(tmp_path);p=Planner(True)
    r=augment_pipeline(target,[{'offset':.1}],lambda out:p,lambda t:Backend(),tmp_path/'run',reference_task=source)
    assert r['variants']==[] and r['success_count']==0
    assert r['reference_recovery']['attempts']==source['budget']['attempts']
    assert not (tmp_path/'run/variant_000').exists()

def test_resume_preserves_success_without_reexecuting(tmp_path):
    from cap_sim.pipeline import run_pipeline
    source,target=setup(tmp_path);p=Planner();out=tmp_path/'run'
    first=run_pipeline(target,lambda out:p,lambda t:Backend(),out,reference_task=source)
    resumed=Planner()
    second=run_pipeline(target,lambda out:resumed,lambda t:Backend(),out,reference_task=source,resume=True)
    assert second['success'] and first['recovered_reference']==second['recovered_reference']
    assert not resumed.requests
