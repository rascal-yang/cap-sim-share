import numpy as np
from humanoid_mimicgen.wrist_feedback import position_bias

def test_high_actual_wrist_commands_down():
    np.testing.assert_allclose(position_bias(np.zeros(3),[0,0,.8],[0,0,.81],.02),[0,0,-.0001])

def test_saturated_bias_is_bounded_and_unwinds():
    a=position_bias([0,0,-.02],[0,0,.8],[0,0,.9],.02)
    assert np.linalg.norm(a)<=.02+1e-12
    b=position_bias(a,[0,0,.8],[0,0,.79],.02)
    assert b[2]>a[2]
