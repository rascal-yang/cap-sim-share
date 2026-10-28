import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_mimicgen.torso_feedback import TorsoFeedback

def test_feedback_increases_undertracked_pitch_with_bounded_correction():
    feedback=TorsoFeedback(gain=.8,limit=.5)
    identity=np.array([1.,0,0,0]);desired=np.roll(Rotation.from_euler('y',.4).as_quat(),1)
    commanded=feedback.update(desired,identity,.02)
    assert Rotation.from_quat(np.roll(commanded,-1)).as_euler('xyz')[1]>.4
    for _ in range(1000):commanded=feedback.update(desired,identity,.02)
    assert np.linalg.norm(feedback.correction)<=.5000001
    assert np.isclose(np.linalg.norm(commanded),1)

def test_zero_error_leaves_target_unchanged():
    f=TorsoFeedback();q=np.array([1.,0,0,0]);assert np.allclose(f.update(q,q,.02),q)
