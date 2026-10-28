import numpy as np
import mujoco
from humanoid_mimicgen.augmentation.replay_navigation import ReplayProgram


def test_navigation_keeps_walking_speed_until_it_reaches_the_goal():
    from humanoid_mimicgen.augmentation.replay_navigation import navigation_command
    for distance in [.4,.2,.05,.026]:
        cmd=navigation_command(np.array([0.,0.]),np.array([distance,0.,-.1]),.25)
        np.testing.assert_allclose(cmd['velocity_world_xy_m_s'],[.25,0.])
        assert cmd['facing_yaw_rad']==-.1


def test_pose_feedback_corrects_both_position_and_orientation_without_mutating_desired():
    from scipy.spatial.transform import Rotation
    from humanoid_mimicgen.augmentation.replay_navigation import compensate_pose
    desired=np.r_[.2,0,0,np.roll(Rotation.from_euler('z',.2).as_quat(),1)]
    measured=np.r_[.1,0,0,np.roll(Rotation.from_euler('z',.1).as_quat(),1)]
    corrected=compensate_pose(desired,measured,2.,1.)
    assert abs(corrected[0]-.4)<1e-12 and desired[0]==.2
    assert abs(Rotation.from_quat(np.roll(corrected[3:],-1)).as_euler('xyz')[2]-.3)<1e-12


def test_direct_program_runs_transition_replay_and_hold_without_readiness_gates():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>')
    initial=model.qpos0.copy();reference=np.tile(initial,(3,1));reference[:,0]=[.1,.12,.15]
    fingers=np.arange(42).reshape(3,14)*.01
    program=ReplayProgram(model,initial,reference,fingers,np.array([10,11,12]),.005,1.,.1)
    first=program.sample(0.)
    np.testing.assert_array_equal(first['qpos'],initial)
    assert first['stage']=='transition'
    for index in range(3):
        row=program.sample(1.+index*.005)
        np.testing.assert_array_equal(row['qpos'],reference[index])
        np.testing.assert_array_equal(row['fingers'],fingers[index])
        assert row['source_row']==10+index and row['stage']=='operation'
    row=program.sample(1.02)
    assert row['stage']=='hold' and not row['completed']
    np.testing.assert_array_equal(row['fingers'],fingers[-1])
    assert program.sample(1.116)['completed']
    np.testing.assert_array_equal(initial,model.qpos0)


def test_terminal_hold_starts_when_last_body_target_is_actually_sent():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>')
    q=np.tile(model.qpos0,(3,1));p=ReplayProgram(model,q[0],q,np.zeros((3,14)),np.arange(3),.005,1.,.1)
    p.sample(1.,policy_tick=True)
    p.sample(1.01,policy_tick=False)
    p.sample(1.02,policy_tick=True)
    assert not p.sample(1.116,policy_tick=False)['completed']
    assert p.sample(1.121,policy_tick=False)['completed']


def test_hand_change_is_available_between_body_policy_ticks():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>')
    q=np.tile(model.qpos0,(3,1));hands=np.zeros((3,14));hands[2,0]=.5
    p=ReplayProgram(model,q[0],q,hands,np.arange(3),.005,1.,.1)
    assert p.sample(1.01,policy_tick=False)['fingers'][0]==.5


def contract_fixture(path):
    import json
    from humanoid_mimicgen.workflow.contracts import file_hash
    (path/'execution').mkdir()
    (path/'experiment.json').write_text(json.dumps({'preflight':{'evidence_sha256':{}}}))
    (path/'experiment.sha256').write_text(file_hash(path/'experiment.json'))


def test_started_case_cannot_repeat_physics(tmp_path,monkeypatch):
    import pytest
    from humanoid_mimicgen.augmentation import replay_navigation as module
    contract_fixture(tmp_path);(tmp_path/'execution/started.json').write_text('{}')
    monkeypatch.setattr(module.direct,'initialize',lambda case:pytest.fail('Repeated physics initialization'))
    with pytest.raises(FileExistsError):module.run(tmp_path)


def test_first_step_and_token_export_error_still_save_result_and_close(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from humanoid_mimicgen.augmentation import replay_navigation as module
    from humanoid_mimicgen.workflow.contracts import WorkflowError
    from humanoid_mimicgen.workflow.tests.test_navigation_control import limits
    contract_fixture(tmp_path)
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="pelvis"><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>');data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    q=np.tile(data.qpos,(3,1));np.savez(tmp_path/'reference.npz',qpos=q,fingers=np.zeros((3,14)),source_rows=np.arange(3))
    (tmp_path/'offline_plan.json').write_text(json.dumps({'selected':{'navigation_path':[[0,0,0],[1,0,0]]}}))
    cfg=dict(navigation={'route_limits':limits(),'transition_s':1.},strategy={'bindings':{'wrists':{'left':'pelvis','right':'pelvis'},'torso':'pelvis'},'policy':{'control_dt_s':.02}},simulation_budget_s=.1,hold_s=.1,offset_m=.05)
    a=SimpleNamespace(m=model,d=data,dt=.005,control_stride=4,sonic=SimpleNamespace(records=[]),last_target=np.zeros(29),providers={},model_files={})
    a.poses=lambda:{s:np.r_[data.xpos[model.body('pelvis').id],data.xquat[model.body('pelvis').id]].copy() for s in ['left','right','torso']}
    a.command=lambda *args,**kwargs:a.sonic.records.append(dict(mode=1))
    def fail():raise RuntimeError('first step failure')
    a.step=fail;closed=[];a.close=lambda:closed.append(True)
    def bad_record(*args,**kwargs):raise WorkflowError('incomplete_sonic_record','no executed tokens')
    monkeypatch.setattr(module,'save_action_bundle',bad_record)
    result=module.run(tmp_path,(a,cfg))
    assert closed==[True] and result['physics_steps']==0
    assert not result['audit']['sonic_action_record']['coverage_complete']
    assert (tmp_path/'result.json').exists() and (tmp_path/'execution/execution_audit.json').exists()
