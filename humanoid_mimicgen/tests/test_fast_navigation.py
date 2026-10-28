import numpy as np
from humanoid_mimicgen.fast_navigation import navigation_command

def test_cruise_speed_is_half_meter_not_proportional():
    v, yaw, stop = navigation_command([0,0,0], [.3,.1,.2])
    assert np.isclose(np.linalg.norm(v), .5)
    assert np.allclose(v / np.linalg.norm(v), np.array([.3,.1])/np.hypot(.3,.1))
    assert not stop and yaw == .2

def test_small_offset_requests_stop_before_upper_body_planning():
    v, yaw, stop = navigation_command([0,0,0], [.08,0,.4])
    assert stop and np.array_equal(v, [0,0])

def test_no_motion_at_same_position():
    v, _, stop = navigation_command([0,0,0], [0,0,0])
    assert stop and np.isfinite(v).all()
