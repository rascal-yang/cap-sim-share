import copy
import numpy as np
import pytest
from humanoid_mimicgen.workflow.contracts import WorkflowError
from humanoid_mimicgen.workflow.navigation import operation


def config():
    return dict(transition_s=.2, position_m=.02, orientation_rad=.1,
                continuous_s=.04, timeout_s=.2, object_force_stop_N=100., open_fingers_rad=[0.]*14,
                alignment_bodies=['pelvis','right_wrist'],
                contact_gates=[dict(frame=4, groups=[['thumb'],['index']], minimum_force_N=1.)])


def observed(aligned=True, contacts=True):
    return dict(aligned=aligned, stable=True, object_force_N=5., contacts=[{'bodies':['thumb','apple'],'force_N':2.},
        {'bodies':['index','apple'],'force_N':3.}] if contacts else [], object_bodies=['apple'])


def test_clock_waits_before_any_source_hand_command_and_at_lift():
    clock=operation.PhaseClock(2,6,config())
    for _ in range(3):assert clock.step(.02,observed(False))['frame'] is None
    assert clock.step(.02,observed())['frame'] is None
    assert clock.step(.02,observed())['frame']==2
    assert clock.step(.02,observed())['frame']==3
    for _ in range(3):assert clock.step(.02,observed(contacts=False))['frame']==3
    assert clock.step(.02,observed())['frame']==3
    assert clock.step(.02,observed())['frame']==4
    assert clock.step(.02,observed())['frame']==5
    assert clock.step(.02,observed())['completed']


def test_one_finger_or_non_object_contact_cannot_release_lift():
    clock=operation.PhaseClock(2,6,config())
    for _ in range(3):clock.step(.02,observed())
    ob=observed();ob['contacts'][1]['bodies']=['index','table']
    for _ in range(5):assert clock.step(.02,ob)['frame']==3
    ob=observed();ob['contacts'][0]['force_N']=float('nan')
    with pytest.raises(WorkflowError):clock.step(.02,ob)


def test_wait_is_bounded_and_requires_continuous_geometry():
    clock=operation.PhaseClock(2,6,config())
    clock.step(.02,observed());clock.step(.02,observed(False))
    assert clock.step(.02,observed())['frame'] is None
    with pytest.raises(WorkflowError,match='operation_gate_timeout'):
        for _ in range(15):clock.step(.02,observed(False))


@pytest.mark.parametrize('change',[
    {'continuous_s':0}, {'timeout_s':.01}, {'alignment_bodies':[]},
    {'contact_gates':[dict(frame=6,groups=[['thumb']],minimum_force_N=1.)]},
    {'contact_gates':[dict(frame=4,groups=[['thumb'],['thumb']],minimum_force_N=1.)]},
])
def test_invalid_operation_rejected(change):
    cfg=config();cfg.update(change)
    with pytest.raises(WorkflowError):operation.PhaseClock(2,6,cfg)


def test_align_poses_requires_every_body_and_rotation():
    expected={'pelvis':np.array([0,0,1,1,0,0,0.]),'right_wrist':np.array([1,0,1,1,0,0,0.])}
    actual=copy.deepcopy(expected)
    assert operation.poses_aligned(actual,expected,config())
    actual['pelvis'][0]+=.03
    assert not operation.poses_aligned(actual,expected,config())
    actual=copy.deepcopy(expected);actual['right_wrist'][3:]=[0,0,0,1]
    assert not operation.poses_aligned(actual,expected,config())


def test_whole_body_reference_uses_same_robot_transform_without_moving_live_state():
    import mujoco
    from humanoid_mimicgen.workflow.navigation.whole_body import BodyReference
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="pelvis" pos="0 0 1"><freejoint/><geom size=".1" mass="1"/><body name="right_wrist" pos=".3 0 0"><joint name="arm"/><geom size=".05" mass=".1"/></body></body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data);before=data.qpos.copy()
    ref=BodyReference(model,{'left':'right_wrist','right':'right_wrist','torso':'pelvis'},['pelvis','right_wrist'])
    q=before.copy();q[0]=.1;q[7]=.2
    poses,alignment=ref.poses(q)
    assert alignment['pelvis'][0]==pytest.approx(.1)
    assert poses['right'][0]==pytest.approx(.4)
    np.testing.assert_array_equal(data.qpos,before)


def test_sonic_reference_does_not_reset_history_or_allow_planner_overwrite():
    from types import SimpleNamespace
    from humanoid_mimicgen.workflow.navigation.whole_body import set_body_reference
    s=SimpleNamespace(qids=np.array([7,8]),tick=37,records=['history'])
    q=np.array([.2,.1,.8,1,0,0,0,.3,.4])
    set_body_reference(s,q)
    assert s.tick==37 and s.records==['history']
    assert s.navigating is False and s.handoff_until_tick>37
    np.testing.assert_array_equal(s.ref[0],q)
    assert s.ref_tick==37


def test_operation_binding_requires_full_range_body_coverage_and_contact_gates():
    from humanoid_mimicgen.workflow.navigation.integration import validate_operation_binding
    cfg=config();cfg['alignment_bodies']=['pelvis','left_foot','right_foot','right_wrist']
    strategy=dict(command_pose_offset=0,phases=[dict(start=2,end=4,requires_contact=False),dict(start=4,end=6,requires_contact=True)],
        policy={'whole_body_operation':cfg},bindings={'root_body':'pelvis','torso':'pelvis','feet':{'left':'left_foot','right':'right_foot'}})
    plan={'prior':{'operation_interval':[2,5],'bindings':{'operation_bodies':['right_wrist']}}}
    validate_operation_binding(plan,strategy)
    strategy['phases'][-1]['end']=7
    with pytest.raises(WorkflowError):validate_operation_binding(plan,strategy)
    strategy['phases'][-1]['end']=6;cfg['contact_gates']=[]
    with pytest.raises(WorkflowError):validate_operation_binding(plan,strategy)


def test_operation_inputs_reject_resampling_and_closed_entry(tmp_path):
    import json
    from humanoid_mimicgen.workflow.navigation.integration import validate_operation_inputs
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'state_key':'state'}))
    source=np.zeros((6,15));source[:,0]=np.arange(6)*.005
    np.savez(tmp_path/'source.npz',state=source);np.savez(tmp_path/'targets.npz',fingers=np.zeros((6,14)))
    files={'reference_states':str(tmp_path/'source.npz'),'targets':str(tmp_path/'targets.npz')}
    s={'phases':[{'start':2,'end':6}], 'policy':{'whole_body_operation':config()}}
    plan={'source_manifest':str(manifest)}
    validate_operation_inputs(plan,files,s,.005)
    with pytest.raises(WorkflowError,match='resampling'):validate_operation_inputs(plan,files,s,.02)
    h=np.ones((6,14));np.savez(tmp_path/'targets.npz',fingers=h)
    with pytest.raises(WorkflowError,match='open'):validate_operation_inputs(plan,files,s,.005)


def test_hand_closure_waits_for_geometry_even_after_entry_passed():
    cfg=config();cfg['geometry_gates']=[3]
    clock=operation.PhaseClock(2,6,cfg)
    clock.step(.02,observed());assert clock.step(.02,observed())['frame']==2
    for _ in range(3):assert clock.step(.02,observed(False))['frame']==2
    assert clock.step(.02,observed())['frame']==2
    assert clock.step(.02,observed())['frame']==3


def test_detect_each_hand_leaving_open_and_preserve_reopening():
    h=np.zeros((8,14));h[2:5,7:]=1;h[4:6,:7]=1;h[7,7:]=1
    assert operation.closure_frames(h,[0.]*14,0,8)==[2,4,7]


def test_excessive_object_force_stops_even_between_gates():
    clock=operation.PhaseClock(2,6,config())
    clock.step(.02,observed());clock.step(.02,observed())
    ob=observed();ob['object_force_N']=129.
    with pytest.raises(WorkflowError,match='operation_object_force_limit'):clock.step(.02,ob)


def test_native_bridge_runs_phase_clock_on_fk_copies_without_physics(tmp_path):
    import json,mujoco
    from types import SimpleNamespace
    from humanoid_mimicgen.workflow.tests.test_navigation_archive import archive
    from humanoid_mimicgen.workflow.tests.test_navigation_native import fixture
    from humanoid_mimicgen.workflow.navigation.whole_body import WholeBodyOperation
    from humanoid_mimicgen.privileged import no_physics
    manifest=archive(tmp_path);doc=json.loads(manifest.read_text())
    doc['bindings'].update(operation_bodies=['pelvis'],contact_policy=[]);manifest.write_text(json.dumps(doc))
    model=mujoco.MjModel.from_binary_path(str(tmp_path/'scene.mjb'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    _,_,_,_,options=fixture()
    plan=dict(source_manifest=str(manifest),prior={'source_stance_xyyaw':[0,0,0],
        'source_task_frame_xyzyaw':[1,0,.1,0],'operation_interval':[20,29]},
        scene={'task_frame_xyzyaw':[1,0,.1,0]},native_options=options)
    cfg=config();cfg.update(alignment_bodies=['pelvis'],contact_gates=[dict(frame=28,groups=[['pelvis']],minimum_force_N=1.)])
    strategy=dict(command_pose_offset=0,phases=[dict(start=20,end=30)],bindings={'wrists':{'left':'pelvis','right':'pelvis'},'torso':'pelvis'},
        policy={'control_dt_s':.02,'whole_body_operation':cfg,'approach':dict(joint_speed_rad_s=1.,wrist_speed_m_s=1.,wrist_speed_rad_s=1.)})
    adapter=SimpleNamespace(m=model,d=data,object_bodies={model.body('object').id},observe=lambda:{'contacts':{'events':[],'robot_object_N':0.}})
    h=np.zeros((30,14));h[24:,7:]=1;source=np.load(tmp_path/'trajectory.npz')['state'];before=data.qpos.copy()
    with no_physics():
        op=WholeBodyOperation(adapter,plan,strategy,source,h)
        rows=[]
        for _ in range(23):rows.append(op.command(.02,True))
    assert op.clock.config['geometry_gates']==[24]
    assert any(x['frame']==24 and x['fingers'][7]==1 for x in rows)
    assert rows[-1]['frame']==27 and rows[-1]['waiting']
    np.testing.assert_array_equal(data.qpos,before)
    # Run the actual bridge under the runner's 200 Hz / 50 Hz schedule.
    cfg['contact_gates']=[];h[25:]=0
    with no_physics():
        op=WholeBodyOperation(adapter,plan,strategy,source,h)
        sent=[];terminal_tick=None
        for step in range(150):
            tick=step%4==0;cmd=op.command(.005,True,policy_tick=tick)
            if cmd['completed']:
                assert terminal_tick is not None and step-terminal_tick>=4
                break
            if tick:
                sent.append((step,op.applied_frame,cmd['fingers'].copy()))
                if op.applied_frame==29 and terminal_tick is None:terminal_tick=step
        else:pytest.fail('operation did not complete')
    assert any(frame==24 and fingers[7]==1 for _,frame,fingers in sent)
    assert any(frame==25 and fingers[7]==0 for _,frame,fingers in sent)
    assert all(b[0]-a[0]==4 for a,b in zip(sent,sent[1:]))
    np.testing.assert_array_equal(data.qpos,before)


def test_real_apple_failed_preclose_error_is_rejected_by_new_gate():
    import json
    from pathlib import Path
    root=Path('/home/kinetra-4090-01/code/zijian/experiments/2026-10/26/20261026_2248_four_task_10cm_api')
    audit=root/'FINAL_AUDIT.json'
    if not audit.exists():pytest.skip('Local frozen apple comparison unavailable')
    case=Path(json.loads(audit.read_text())['results'][0]['case'])
    with np.load(case/'execution/hand_commands.npz') as z:h=np.c_[z['left'],z['right']]
    with np.load(case/'execution/trajectory.npz') as z:error=z['wrist_error_m'][340]
    cfg=config();cfg['geometry_gates']=operation.closure_frames(h,[0.]*14,338,343)
    cfg['contact_gates']=[]
    assert cfg['geometry_gates']==[340] and max(error)>.1
    clock=operation.PhaseClock(338,343,cfg)
    clock.step(.02,observed());clock.step(.02,observed());clock.step(.02,observed())
    assert clock.frame==339
    assert clock.step(.02,observed(aligned=bool(max(error)<=cfg['position_m'])))['frame']==339
