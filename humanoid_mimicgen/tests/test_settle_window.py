import numpy as np
from humanoid_mimicgen.settle_window import recent_root_drift

def test_braking_history_does_not_block_current_standing():
    rows=[{'time_s':0.,'root_pose':[0.,0.]},{'time_s':1.,'root_pose':[.04,0.]},{'time_s':1.3,'root_pose':[.041,0.]}]
    assert np.isclose(recent_root_drift(rows, 1., .3), .001)

def test_recent_motion_is_not_standing():
    rows=[{'time_s':1.,'root_pose':[0.,0.]},{'time_s':1.3,'root_pose':[.04,0.]}]
    assert recent_root_drift(rows, 1., .3) > .03

def test_incomplete_window_is_not_stationary():
    assert np.isinf(recent_root_drift([{'time_s':1.,'root_pose':[0.,0.]}],1.,.3))
