import numpy as np
import pytest
from humanoid_mimicgen.workflow.navigation.control import RouteController
from humanoid_mimicgen.workflow.contracts import WorkflowError


def limits():
    return dict(speed_m_s=.5,acceleration_m_s2=.5,response_s=.4,stop_radius_m=.02,
                waypoint_radius_m=.02,corridor_radius_m=.08,yaw_speed_rad_s=.5,
                yaw_acceleration_rad_s2=1.,yaw_tolerance_rad=.02,settle_speed_m_s=.03,
                settle_yaw_speed_rad_s=.03,timeout_s=10.,max_dt_s=.1)


def test_acceleration_and_braking_from_measured_velocity():
    c=RouteController([[0,0,0],[2,0,0]],limits());previous=np.zeros(2)
    for _ in range(20):
        out=c.step([0,0,0],[0,0,0],.02);v=np.array(out['velocity_world_xy_m_s'])
        assert np.linalg.norm(v-previous)<=.01000001;previous=v
    out=c.step([1.96,0,0],[.3,0,0],.02)
    assert out['state']=='BRAKE'
    assert out['velocity_world_xy_m_s'][0]<previous[0]
    assert not out['arrived']


def test_zero_distance_requires_yaw_and_low_speed():
    c=RouteController([[0,0,0],[0,0,np.pi/2]],limits())
    out=c.step([0,0,0],[0,0,0],.02)
    assert not out['arrived']
    assert 0<out['facing_yaw_rad']<=.000401
    assert RouteController([[0,0,0],[0,0,0]],limits()).step([0,0,0],[0,0,0],.02)['arrived']


def test_corner_does_not_switch_at_high_speed_or_cut_obstacle():
    c=RouteController([[0,0,0],[1,0,0],[1,1,0]],limits())
    out=c.step([.99,0,0],[.4,0,0],.02)
    assert out['waypoint']==1
    assert out['velocity_world_xy_m_s'][1]==0
    out=c.step([1,0,0],[0,0,0],.02)
    assert out['waypoint']==2
    assert c.step([.8,.2,0],[0,0,0],.02)['corridor_error_m']>0


def test_yaw_shortest_turn_timeout_and_bad_state():
    c=RouteController([[0,0,3.13],[1,0,-3.13]],limits())
    out=c.step([0,0,3.13],[0,0,0],.02)
    assert abs(out['facing_yaw_rad']-3.13)<.001
    with pytest.raises(WorkflowError):c.step([np.nan,0,0],[0,0,0],.02)
    o=limits();o['timeout_s']=.03;c=RouteController([[0,0,0],[1,0,0]],o)
    c.step([0,0,0],[0,0,0],.02)
    assert c.step([0,0,0],[0,0,0],.02)['elapsed_s']>.03


def test_straight_intermediate_waypoint_does_not_force_stop():
    c=RouteController([[0,0,0],[1,0,0],[2,0,0]],limits())
    out=c.step([.99,0,0],[.3,0,0],.02)
    assert out['waypoint']==2


def test_finite_route_completes_under_simple_measured_integrator():
    c=RouteController([[0,0,0],[.3,0,0]],limits());pose=np.zeros(3);v=np.zeros(3)
    for _ in range(499):
        out=c.step(pose,v,.02)
        if out['arrived']:break
        v[:2]=out['velocity_world_xy_m_s'];pose[:2]+=.02*v[:2]
    assert out['arrived']


def test_small_turn_never_overshoots_checked_yaw_interval():
    o=limits();c=RouteController([[0,0,0],[0,0,.03]],o);pose=np.zeros(3);previous_rate=0.
    for _ in range(20):
        out=c.step(pose,[0,0,previous_rate],.1)
        heading=out['facing_yaw_rad'];rate=(heading-pose[2])/.1
        assert 0<=heading<=.03+1e-12
        assert abs(rate-previous_rate)<=.100000001
        pose[2]=heading;previous_rate=rate


def test_missing_or_wrong_scope_calibration_cannot_enable_execution():
    from humanoid_mimicgen.workflow.navigation.control import validate_route_calibration
    with pytest.raises(WorkflowError):validate_route_calibration({}, {'robot':'g1'}, {'minimum_samples':2,'max_residual_m':.01}, limits())
    with pytest.raises(WorkflowError):validate_route_calibration({'schema_version':2,'kind':'zero_intercept_velocity_response','scope':{'robot':'other'}}, {'robot':'g1'}, {'minimum_samples':2,'max_residual_m':.01}, limits())


def test_off_route_robot_continues_toward_goal_instead_of_aborting():
    c=RouteController([[0,0,0],[1,0,0]],limits())
    out=c.step([.2,.15,0],[0,0,0],.02)
    assert out['corridor_error_m']>.08
    assert out['velocity_world_xy_m_s'][0]>0 and out['velocity_world_xy_m_s'][1]<0
