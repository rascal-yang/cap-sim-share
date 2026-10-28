import numpy as np
import pytest
from humanoid_mimicgen.workflow.contracts import WorkflowError
from humanoid_mimicgen.workflow.grasp_trigger import validate_judgment, trigger_targets, query_due


def test_judgment_binds_evidence_and_cannot_invent_actions():
    evidence={'measured':{},'geometry':{}}
    p={'action':'close_now','reason':'Opposing fingers surround object','evidence_ids':['geometry']}
    assert validate_judgment(p,evidence)==p
    with pytest.raises(WorkflowError):validate_judgment(dict(p,evidence_ids=['future_success']),evidence)
    with pytest.raises(WorkflowError):validate_judgment(dict(p,action='teleport'),evidence)


def test_trigger_holds_measured_wrist_then_rejoins_relative_lift_without_changing_fingers():
    poses=np.tile([0.,0.,0.,1.,0.,0.,0.],(10,1));poses[:,2]=np.arange(10)*.01
    targets={h+'_pose':poses.copy() for h in ('left','right','torso')};targets['fingers']=np.arange(140).reshape(10,14)
    actual={h:np.array([1.,2.,3.,1.,0.,0.,0.]) for h in ('left','right','torso')}
    out=trigger_targets(targets,actual,['right'],start=2,lift=5,end=10,offset=-1)
    np.testing.assert_allclose(out['right_pose'][1:5,:3],np.tile([1,2,3],(4,1)))
    np.testing.assert_allclose(out['right_pose'][8,:3],[1,2,3.04])
    np.testing.assert_array_equal(out['fingers'],targets['fingers'])
    np.testing.assert_array_equal(targets['right_pose'],poses)


def test_queries_are_bounded_and_sim_time_scheduled():
    opts={'max_calls':4,'interval_s':.5,'approach_lookahead_s':1.}
    assert not query_due('approach',1,3,None,0,opts)
    assert query_due('approach',2,3,None,0,opts)
    assert not query_due('grasp_probe',2.1,3,2.,1,opts)
    assert query_due('grasp_probe',2.5,3,2.,1,opts)
    assert not query_due('grasp_probe',3,3,2.,4,opts)


def test_approval_requires_close_now_at_exact_measured_time():
    from humanoid_mimicgen.workflow.grasp_trigger import validate_approval
    evidence={'measured':{'time_s':2.,'retry_index':0},'geometry':{}}
    proof={'status':'completed','model_requested':'gpt-6-astra','evidence_sha256':__import__('humanoid_mimicgen.workflow.contracts',fromlist=['digest']).digest(evidence)}
    response={'action':'close_now','reason':'ready','evidence_ids':['geometry']}
    event={'retry_index':0,'time_s':2.,'trigger':'gpt_privileged'}
    assert validate_approval(proof,response,evidence,event,'gpt-6-astra')
    with pytest.raises(WorkflowError):validate_approval(proof,dict(response,action='continue'),evidence,event,'gpt-6-astra')
    with pytest.raises(WorkflowError):validate_approval(proof,response,evidence,dict(event,time_s=3.),'gpt-6-astra')
