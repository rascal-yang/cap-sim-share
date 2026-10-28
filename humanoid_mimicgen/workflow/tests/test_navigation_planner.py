import numpy as np
from humanoid_mimicgen.workflow.navigation.planner import plan_stances
from humanoid_mimicgen.workflow.tests.test_navigation_prior import example
from humanoid_mimicgen.workflow.navigation.prior import extract_prior


def setup():
    p=extract_prior(*example())
    return p, {'task_frame_xyzyaw':[4,3,1,np.pi/2]}, [0,0,0], {
        'offsets_xyyaw':[[0,0,0],[.1,0,0],[-.1,0,0]],'max_candidates':4,'planning_timeout_s':2.}


def check(candidate):
    return {'feasible':candidate['id']!='current','reason':None,'travel_m':1.,
            'navigation_path':[[0,0,0],candidate['xyyaw']], 'operation_checked':True}


def test_transfer_candidates_in_task_frame_and_bound_work():
    p,s,c,o=setup();o['max_candidates']=2
    result=plan_stances(p,s,c,o,check)
    assert len(result['candidates'])==2
    np.testing.assert_allclose(result['selected']['xyyaw'],[4,2,np.pi/2],atol=1e-9)
    assert result['physics_steps']==0


def test_current_usable_skips_navigation_and_further_search():
    p,s,c,o=setup()
    result=plan_stances(p,s,c,o,lambda r:dict(check(r),feasible=True))
    assert result['selected']['id']=='current'
    assert len(result['candidates'])==1


def test_entry_only_evidence_cannot_be_selected():
    p,s,c,o=setup()
    result=plan_stances(p,s,c,o,lambda r:dict(check(r),operation_checked=False))
    assert result['selected'] is None
    assert all(r['reason']=='operation_unchecked' for r in result['candidates'])


def test_failed_prior_and_timeout_are_explicit():
    p,s,c,o=setup();p['status']='unverified_source'
    assert plan_stances(p,s,c,o,check)['status']=='unverified_source'
    p['status']='completed'
    ticks=iter([0.,3.,3.])
    result=plan_stances(p,s,c,o,check,clock=lambda:next(ticks))
    assert result['status']=='planning_budget_exhausted'
    assert result['selected'] is None


def test_preserve_concrete_rejection_when_operation_failed():
    p,s,c,o=setup()
    def failed(candidate):
        return {'feasible':False,'operation_checked':False,'reason':'operation_collision','failed_source_index':80}
    result=plan_stances(p,s,c,o,failed)
    assert result['candidates'][0]['reason']=='operation_collision'
