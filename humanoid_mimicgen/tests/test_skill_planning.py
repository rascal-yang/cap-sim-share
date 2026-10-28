import numpy as np
import pytest
from humanoid_mimicgen.skill_planning import skill_batches, adapt_poses, rollout_action, plan_joint_path


def test_concurrent_skills_wait_for_both_predecessors():
    skills={'prepare': 'left', 'pick_l':'left','pick_r':'right','place_l':'left','place_r':'right'}
    result=skill_batches(skills,[('prepare','pick_l'),('pick_l','place_l'),('pick_r','place_r')],[('pick_l','pick_r'),('place_l','place_r')])
    assert result==[['prepare'],['pick_l','pick_r'],['place_l','place_r']]


@pytest.mark.parametrize('skills,pre,con',[
 ({'a':'right','b':'right'},[],[]),
 ({'a':'left','b':'right'},[('a','b'),('b','a')],[]),
 ({'a':'left','b':'right'},[('a','b')],[('a','b')]),
 ({'a':'left'},[('a','missing')],[]),
])
def test_reject_impossible_skill_annotations(skills,pre,con):
    with pytest.raises(ValueError):skill_batches(skills,pre,con)


def test_adaptation_preserves_object_relative_pose_under_rotation():
    source=np.array([1,0,0,1,0,0,0.]);target=np.array([0,2,0,2**-.5,0,0,2**-.5]);poses=np.array([[2,0,0,1,0,0,0],[2,0,1,1,0,0,0.]])
    adapted=adapt_poses(poses,source,target)
    np.testing.assert_allclose(adapted[:,:3],[[0,3,0],[0,3,1]],atol=1e-12)
    np.testing.assert_allclose(adapted[:,3:],np.tile(target[3:],(2,1)))
    np.testing.assert_array_equal(poses[:,2],[0,1])


def test_noise_changes_executed_action_not_expert_label():
    a=np.array([.2,.3]);x=rollout_action(a,.01,seed=12);y=rollout_action(a,.01,seed=12)
    np.testing.assert_array_equal(x['label_action'],a);np.testing.assert_array_equal(x['executed_action'],y['executed_action'])
    assert not np.array_equal(x['executed_action'],a)
    with pytest.raises(ValueError):rollout_action(a,-1,seed=0)


def test_joint_planner_routes_around_obstacle_and_checks_all_edges():
    valid=lambda q:not(abs(q[0])<.3 and abs(q[1])<.3)
    path,info=plan_joint_path(np.array([-1.,0]),np.array([1.,0]),np.array([[-1.5,1.5],[-1.5,1.5]]),valid,seed=7,max_iterations=3000,resolution=.03)
    assert info['method']=='rrt' and path is not None
    np.testing.assert_array_equal(path[0],[-1,0]);np.testing.assert_array_equal(path[-1],[1,0])
    assert all(valid(q) for q in path)
    assert np.max(abs(np.diff(path,axis=0)))<=.03000001


def test_joint_planner_does_not_invent_solution_for_blocked_start():
    path,info=plan_joint_path(np.array([0.]),np.array([1.]),np.array([[-2.,2.]]),lambda q:q[0]>.5)
    assert path is None and info['reason']=='invalid_start'
