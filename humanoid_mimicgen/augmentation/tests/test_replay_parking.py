import copy
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def fixture():
    from humanoid_mimicgen.augmentation.direct_grasp import rot
    calls=[]
    q=np.r_[0.,0.,.8,np.roll(Rotation.from_euler('y',.1).as_quat(),1),np.zeros(12)]
    data=SimpleNamespace(qpos=q,qvel=np.zeros(18),time=0.)
    sonic=SimpleNamespace(qids=np.arange(7,19),blend_to_standing=lambda *args,**kw:calls.append((args,kw)))
    poses={s:np.r_[0.,0.,1.,1.,0.,0.,0.] for s in ['left','right','torso']}
    adapter=SimpleNamespace(d=data,m=None,sonic=sonic,poses=lambda:copy.deepcopy(poses),observe=lambda:{'foot_ground_N':[20.,20.],'contacts':{'events':[]}})
    policy=dict(navigation={},settle={'speed_m_s':.03,'yaw_speed_rad_s':.15,'foot_force_N':10.,'minimum_s':1.,'continuous_s':.3},planning={},approach={},budgets={})
    cfg=dict(strategy={'policy':policy,'bindings':{}},navigation={'route_limits':{'speed_m_s':.5},'parking':{'acceleration_m_s2':.5,'response_s':.6,'arrival_radius_m':.1,'settle_s':3.,'max_corrections':2,'tolerance_m':.03}})
    return adapter,cfg,calls,q.copy()


def test_parking_decelerates_and_follows_intermediate_waypoints():
    from humanoid_mimicgen.augmentation.replay_parking import ReplayParking
    a,cfg,calls,q=fixture();route=np.array([[0,0,0],[0,.3,0],[.4,.3,0]])
    parking=ReplayParking(a,cfg,route,q)
    first=parking.navigation_command(.02)
    assert first['velocity_world_xy_m_s'][0]==0 and first['velocity_world_xy_m_s'][1]>0
    far=None
    for _ in range(100):far=parking.navigation_command(.02)
    a.d.qpos[:2]=[0,.28];near=parking.navigation_command(.02)
    assert near['waypoint']==2 and near['velocity_world_xy_m_s'][0]>0
    a.d.qpos[:2]=[.29,.3]
    for _ in range(80):near=parking.navigation_command(.02)
    assert np.linalg.norm(near['velocity_world_xy_m_s'])<np.linalg.norm(far['velocity_world_xy_m_s'])


def test_upright_handoff_only_changes_reference_and_soft_settle_expires(monkeypatch):
    from humanoid_mimicgen.augmentation.replay_parking import ReplayParking
    a,cfg,calls,q=fixture();p=ReplayParking(a,cfg,np.array([[0,0,0],[.2,0,0]]),q)
    p.begin();np.testing.assert_array_equal(a.d.qpos,q)
    anchor=calls[0][0][0]
    np.testing.assert_allclose(Rotation.from_quat(np.roll(anchor[3:7],-1)).as_euler('xyz')[:2],[0,0],atol=1e-12)
    a.d.qvel[0]=.2;a.d.time=3.1
    monkeypatch.setattr(p,'correct',lambda:False)
    assert p.update(.02) and p.done
    assert p.events[-1]['event']=='parking_complete' and not p.events[-1]['settled']


def test_failed_correction_proceeds_to_operation(monkeypatch):
    from humanoid_mimicgen.augmentation import replay_parking as module
    a,cfg,calls,q=fixture();p=module.ReplayParking(a,cfg,np.array([[0,0,0],[.2,0,0]]),q)
    monkeypatch.setattr(module,'planted_stance_reference',lambda *args:(None,{'accepted':False}))
    p.begin();a.d.time=1.5
    for _ in range(20):p.update(.02)
    assert p.done and any(x['event']=='correction_skipped' for x in p.events)


def test_recovery_restores_arms_and_waist_under_current_root_and_legs():
    import mujoco
    from humanoid_mimicgen.augmentation.replay_parking import ReplayRecovery
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1" mass="1"/><body><joint name="hip_pitch_joint"/><geom size=".1" mass="1"/><body><joint name="waist_yaw_joint"/><geom size=".1" mass="1"/><body><joint name="right_shoulder_pitch_joint"/><geom size=".1" mass="1"/></body></body></body></body></worldbody></mujoco>')
    current=model.qpos0.copy();current[:3]=[.3,-.2,.8];current[7:]=[.4,.5,.6];normal=model.qpos0.copy();original=current.copy()
    p=ReplayRecovery(model,current,normal,np.zeros(14),{'release_s':.3,'restore_s':1.,'joint_speed_rad_s':2.,'settle_s':1.,'joint_tolerance_rad':.13})
    start=p.sample(0.,current);end=p.sample(.3+p.duration,current)
    np.testing.assert_array_equal(start['qpos'],current)
    np.testing.assert_array_equal(end['qpos'][:8],current[:8])
    np.testing.assert_allclose(end['qpos'][8:],normal[8:],atol=1e-12)
    assert not end['completed'] and p.sample(.3+p.duration+1.01,current)['completed']
    assert not p.restored
    np.testing.assert_array_equal(current,original)


def test_retry_translation_uses_current_object_position_without_rotating_grasp():
    from humanoid_mimicgen.augmentation.replay_parking import retry_translation
    source=np.array([1.,2.,3.,1.,0,0,0]);current=np.array([1.02,1.97,3.01,0.,0,0,1.])
    np.testing.assert_allclose(retry_translation(source,current),[.02,-.03,.01])


@pytest.mark.parametrize('simulation_budget_s,target_reachable',[(.3,True),(.145,True),(.08,True),(.3,False)])
def test_in_run_retry_keeps_time_continuous_and_restarts_source_hand_rows(tmp_path,monkeypatch,simulation_budget_s,target_reachable):
    import json,mujoco
    from humanoid_mimicgen.augmentation import replay_navigation as module
    from humanoid_mimicgen.augmentation import replay_parking as helper
    monkeypatch.setattr(helper,'retry_reachable',lambda *args:(target_reachable,{'reachable':target_reachable}))
    from humanoid_mimicgen.augmentation.tests.test_replay_navigation import contract_fixture
    contract_fixture(tmp_path)
    m=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005"/><worldbody><body name="pelvis"><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    np.savez(tmp_path/'reference.npz',qpos=np.tile(d.qpos,(3,1)),fingers=np.repeat(np.array([0.,.1,.2])[:,None],14,axis=1),source_rows=np.array([10,11,12]))
    (tmp_path/'offline_plan.json').write_text(json.dumps({'selected':{'navigation_path':[[0,0,0],[0,0,0]]}}))
    sonic=SimpleNamespace(records=[],qids=np.array([],int),tick=0)
    a=SimpleNamespace(m=m,d=d,dt=.005,control_stride=4,sonic=sonic,last_target=np.zeros(29),providers={},model_files={})
    a.poses=lambda:{s:np.r_[d.xpos[1],d.xquat[1]].copy() for s in ['left','right','torso']}
    a.observe=lambda:dict(object_pose=[0.,.03 if d.time>=.1 else 0.,.7,1.,0,0,0],clearance_m=.1,contacts={'hand_object_N':float(d.time<.13 or d.time>.15),'support_N':0.,'robot_obstacle_N':0.})
    def command(*args,**kw):
        if kw['policy_tick']:sonic.records.append({'mode':1,'target':[0.]*29});sonic.tick+=1
    a.command=command;a.step=lambda:setattr(d,'time',d.time+.005);a.native_success=lambda:d.time>.1;a.close=lambda:None
    monkeypatch.setattr(module,'save_action_bundle',lambda *args,**kwargs:{'coverage_complete':True})
    cfg=dict(navigation={'route_limits':{'stop_radius_m':.025,'speed_m_s':.25},'transition_s':.02,'retry':{'max_retries':1,'release_s':.01,'restore_s':.01,'settle_s':.01,'joint_speed_rad_s':2.,'joint_tolerance_rad':.13}},strategy={'bindings':{'wrists':{'left':'pelvis','right':'pelvis'},'torso':'pelvis','feet':{'left':'pelvis','right':'pelvis'}},'policy':{'control_dt_s':.02}},feedback_gain=2.,wrist_z_offset_m=0.,simulation_budget_s=simulation_budget_s,hold_s=.02,offset_m=.05)
    cfg['navigation']['wrist_integral']={'gain_per_s':.5,'cap_m':.02}
    result=module.run(tmp_path,(a,cfg));commands=json.loads((tmp_path/'execution/commands.json').read_text());trace=np.load(tmp_path/'execution/trajectory.npz')
    assert result['retry_count']==1 and len(result['audit']['attempts'])==2
    assert result['navigation_arrived']
    assert not result['first_attempt_native_success']
    if not target_reachable:
        assert result['failure']=='retry_target_unreachable' and not result['audit']['attempts'][-1]['operation_started']
        assert not any(c['stage']=='transition' and c['retry_index']==1 for c in commands)
        return
    if simulation_budget_s>=.2:
        assert result['success']
        assert [next(c['hand_source_row'] for c in commands if c['retry_index']==i and c['hand_source_row']>=0) for i in [0,1]]==[10,10]
    else:
        assert not result['audit']['attempts'][-1]['completed']
        assert result['audit']['attempts'][-1]['operation_started']==(simulation_budget_s>=.1)
        assert result['retained_grasp']==bool(simulation_budget_s<.13 or simulation_budget_s>.15)
    if simulation_budget_s>=.1:
        first_retry=next(c for c in commands if c['retry_index']==1 and c['stage']=='transition')
        np.testing.assert_allclose(first_retry['encoder_pose']['right'],first_retry['source_desired']['right'])
        np.testing.assert_array_equal(first_retry['wrist_integral_bias_m'],np.zeros(3))
        if simulation_budget_s>=.2:assert commands[-1]['wrist_integral_bias_m'][1]>0
    np.testing.assert_array_equal(trace['state'][1:],trace['next_state'][:-1])


def test_large_parking_residual_is_corrected_in_small_steps(monkeypatch):
    from humanoid_mimicgen.augmentation import replay_parking as module
    a,cfg,calls,q=fixture();p=module.ReplayParking(a,cfg,np.array([[0,0,0],[.08,0,0]]),q)
    def fit(model,current,goal,*args):
        assert np.linalg.norm(goal[:2]-current[:2])<=.03+1e-12
        return current.copy(),{'accepted':True,'duration_s':.6}
    monkeypatch.setattr(module,'planted_stance_reference',fit)
    assert p.correct()


def test_support_poses_follow_measured_root_without_moving_task_wrist():
    from humanoid_mimicgen.augmentation.replay_navigation import measured_support_poses
    poses={s:np.array([2.,0.,1.,1.,0.,0.,0.]) for s in ['left','right','torso']}
    reference=np.array([1.,0.,0.,1.,0.,0.,0.]);measured=reference.copy();measured[0]=3.
    result=measured_support_poses(poses,reference,measured)
    np.testing.assert_array_equal(result['right'],poses['right'])
    assert result['left'][0]==4 and result['torso'][0]==4 and poses['left'][0]==2


def test_stance_is_blended_before_resuming_full_source_body_replay():
    from humanoid_mimicgen.augmentation.replay_navigation import replay_body_enabled
    assert not replay_body_enabled({'parking':{'leg_mode':'standing'}},'parking')
    assert replay_body_enabled({'parking':{'leg_mode':'source_replay'}},'transition')
    assert replay_body_enabled({'parking':{'leg_mode':'source_replay'}},'operation')
    assert not replay_body_enabled({'parking':{'leg_mode':'source_replay'}},'recovery_restore')


