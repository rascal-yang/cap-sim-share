import numpy as np
import pytest
from humanoid_mimicgen.workflow.planning import rank_candidates, navigate_command, validate_source
from humanoid_mimicgen.workflow.contracts import WorkflowError


def test_reachability_precedes_facing_without_magic_weights():
    rows=[{'id':'bad','feasible':False,'travel_m':0,'facing_error_rad':0},
          {'id':'far','feasible':True,'travel_m':2,'facing_error_rad':0},
          {'id':'near','feasible':True,'travel_m':1,'facing_error_rad':2}]
    assert rank_candidates(rows)['id']=='near'


def test_navigation_uses_declared_speed_and_tolerance():
    path=np.array([[0,0,0],[1,0,1.]])
    vel,heading,reached,index=navigate_command([0,0,0],path,1,{'speed_m_s':.25,'waypoint_radius_m':.01,'stop_radius_m':.03})
    np.testing.assert_allclose(vel,[.25,0]);assert heading==1 and not reached
    assert navigate_command([.98,0,0],path,1,{'speed_m_s':.25,'waypoint_radius_m':.01,'stop_radius_m':.03})[2]


def test_source_commands_are_not_measured_fingers():
    strategy={'source_frame_count':5,'phases':[{'start':2,'end':5}],'command_pose_offset':-1}
    poses=np.tile([0,0,0,1,0,0,0],(5,1))
    source={'left_pose':poses,'right_pose':poses,'torso_pose':poses,'fingers':np.zeros((5,14))}
    with pytest.raises(WorkflowError):validate_source(source,strategy,{'finger_source':'measured_qpos'})
    validate_source(source,strategy,{'finger_source':'executed_commands','command_dt_s':.005,'pose_alignment':'explicit_offset'})


def test_nominal_stance_is_full_object_frame_transform():
    from humanoid_mimicgen.workflow.planning import nominal_stance
    root=[1,0,1,1,0,0,0];source=[0,0,0,1,0,0,0]
    target=[2,3,0,np.cos(np.pi/4),0,0,np.sin(np.pi/4)]
    np.testing.assert_allclose(nominal_stance(root,source,target),[2,4,np.pi/2],atol=1e-10)
