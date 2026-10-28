import numpy as np
from humanoid_mimicgen.planned_approach import timed_joint_reference


def test_time_scaling_keeps_all_waypoints_and_bounds_joint_speed():
    path=np.array([[0.,0.],[.2,.1],[.1,.4]])
    times,points=timed_joint_reference(path,joint_speed=.3,minimum_duration=1.)
    assert times[0]==0 and np.all(np.diff(times)>0)
    np.testing.assert_array_equal(points[0],path[0]);np.testing.assert_array_equal(points[-1],path[-1])
    for p in path:assert np.min(np.linalg.norm(points-p,axis=1))<1e-12
    assert np.max(abs(np.diff(points,axis=0))/np.diff(times)[:,None])<=.300001
    np.testing.assert_array_equal(path,[[0,.0],[.2,.1],[.1,.4]])
