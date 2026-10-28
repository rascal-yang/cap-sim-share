import numpy as np

def test_idle_feed_is_distinct_from_height_conditioned_mode():
    from humanoid_mimicgen.standing_handoff import standing_feed
    f=standing_feed(np.zeros((4,36)),.1,17)
    assert f['mode'].item()==0
    assert f['target_vel'].item()==-1
    assert f['height'].item()==-1
    assert np.array_equal(f['movement_direction'],np.zeros((1,3)))


def test_standing_gate_requires_position_velocity_and_both_feet():
    from humanoid_mimicgen.standing_handoff import stable_sample
    assert stable_sample(.01,.01,.05,.01,.1,[20,30])
    assert not stable_sample(.04,.01,.05,.01,.1,[20,30])
    assert not stable_sample(.01,.1,.05,.01,.1,[20,30])
    assert not stable_sample(.01,.01,.05,.01,.1,[0,30])


def test_hold_body_reference_has_no_future_motion():
    from humanoid_mimicgen.standing_handoff import body_future
    a=np.r_[np.zeros(3),1.,np.zeros(32)];b=a.copy();b[7:]=1
    program=[{'stage':'native_prefix','tokens':250},{'duration_s':4.,'body_keyframes':[[0,*a],[4,*b],[5,*b]]}]
    ref,future=body_future(program,8.,True)
    assert ref.shape==(10,36)
    assert np.array_equal(ref,future)
    assert np.allclose(ref[:,7:],.75)
