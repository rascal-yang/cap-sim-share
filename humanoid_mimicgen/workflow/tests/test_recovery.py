import numpy as np
from humanoid_mimicgen.workflow.feedback import IntegralFeedback, can_recover


def test_feedback_is_continuous_bounded_and_uses_actual_error():
    f=IntegralFeedback(['right'],gain_per_s=.5,cap_m=.02)
    np.testing.assert_allclose(f.update('right',[1,0,0],[0,0,0],.02),[.01,0,0])
    np.testing.assert_allclose(f.update('right',[1,0,0],[0,0,0],.02),[.02,0,0])
    np.testing.assert_allclose(f.update('right',[1,0,0],[0,0,0],.02),[.02,0,0])
    assert f.update('right',[0,0,0],[1,0,0],.02)[0]<.02


def test_recovery_never_overrides_collision_or_unsupported_object():
    ob={'contacts':{'support_N':2.,'robot_obstacle_N':0.}}
    assert can_recover('contact_lost',ob,0,2,1.)
    assert not can_recover('collision',ob,0,2,1.)
    assert not can_recover('contact_lost',ob,2,2,1.)
    ob['contacts']['support_N']=0
    assert not can_recover('contact_lost',ob,0,2,1.)


def test_retreat_plans_only_wrists_from_measured_state(monkeypatch):
    from humanoid_mimicgen.workflow import planning
    poses={h:np.array([0,0,0,1,0,0,0.]) for h in ('left','right','torso')}
    def fake(model,q,goals,policy):
        assert set(goals)=={'left','right'}
        np.testing.assert_array_equal(q,[1,2])
        return 'checked-retreat'
    monkeypatch.setattr(planning,'measured_approach',fake)
    assert planning.measured_retreat(None,np.array([1,2]),poses,{})=='checked-retreat'


def test_verifier_rejects_unknown_retry_ids_and_budget_overrun():
    from humanoid_mimicgen.workflow.verification import checked_retry_groups
    commands=[{'retry_index':0,'source_index':1},{'retry_index':2,'source_index':1}]
    _,errors=checked_retry_groups(commands,0,2)
    assert errors
    _,errors=checked_retry_groups(commands,3,2)
    assert errors
    groups,errors=checked_retry_groups([{'retry_index':0,'source_index':1},{'retry_index':1,'source_index':1}],1,2)
    assert not errors and list(map(len,groups))==[1,1]
