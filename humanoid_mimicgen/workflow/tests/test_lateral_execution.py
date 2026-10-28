import copy
import numpy as np
import mujoco
import pytest
from types import SimpleNamespace
from humanoid_mimicgen.workflow.contracts import WorkflowError
from humanoid_mimicgen.workflow.tests.test_lateral_planning import mechanism
from humanoid_mimicgen.workflow.navigation.reference_bundle import save_reference_bundle


def operation_fixture(tmp_path):
    m,q,p,b,o,c=mechanism();d=mujoco.MjData(m);d.qpos[:]=q[0];mujoco.mj_forward(m,d)
    generated=q.copy();generated[:,m.jnt_qposadr[m.joint('rw_y').id]]+=.04
    bridge=np.array([q[0],generated[0]])
    ref=save_reference_bundle(tmp_path/'generated',dict(qpos=generated,time_s=np.arange(5)*.005,source_rows=np.arange(5),
         fingers=np.zeros((5,14)),bridge_qpos=bridge,bridge_time_s=np.array([0.,2.])),dict(operation_interval=[0,4]))
    plan=dict(reference_mode='apple_lateral_v1',generated_reference=ref,prior=dict(p,bindings=b),evidence_sha256={})
    config=dict(transition_s=2.,position_m=.003,orientation_rad=.03,continuous_s=.01,timeout_s=1.,object_force_stop_N=100.,
        open_fingers_rad=[0.]*14,alignment_bodies=['pelvis','lf','rf','rw'],contact_gates=[])
    strategy=dict(command_pose_offset=0,phases=[dict(start=0,end=5)],bindings=dict(wrists=b['wrists'],torso='pelvis'),
        policy=dict(control_dt_s=.02,whole_body_operation=config,approach=dict(joint_speed_rad_s=10.,wrist_speed_m_s=1.,wrist_speed_rad_s=1.)))
    adapter=SimpleNamespace(m=m,d=d,object_bodies={m.body('apple').id},observe=lambda:dict(contacts=dict(events=[],robot_object_N=0.)))
    source=np.c_[np.arange(5)*.005,q,np.zeros((5,m.nv))]
    from humanoid_mimicgen.workflow.navigation.reference_bundle import load_reference_bundle
    from humanoid_mimicgen.workflow.navigation.lateral_execution import plan_identity
    arrays=load_reference_bundle(ref);arrays.pop('metadata')
    ref=save_reference_bundle(tmp_path/'derived',arrays,dict(operation_interval=[0,4],source_evidence_sha256={},
        parent_plan_sha256=plan_identity(plan),derivation='measured_support_handoff',measured_handoff_qpos=q[0].tolist()))
    return adapter,plan,strategy,source,ref,generated


def test_operation_uses_generated_references_and_checked_bridge(tmp_path):
    from humanoid_mimicgen.workflow.navigation.whole_body import WholeBodyOperation
    adapter,plan,strategy,source,ref,generated=operation_fixture(tmp_path)
    before=adapter.d.qpos.copy()
    op=WholeBodyOperation(adapter,plan,strategy,source,np.zeros((5,14)),generated_reference=ref)
    np.testing.assert_array_equal(op.references[3],generated[3])
    cmd=op.command(1.,True)
    assert cmd['qpos'][adapter.m.jnt_qposadr[adapter.m.joint('rw_y').id]]==pytest.approx(.02)
    np.testing.assert_array_equal(adapter.d.qpos,before)


@pytest.mark.parametrize('change',['missing','fingers','start','timing'])
def test_generated_operation_rejects_stale_or_mismatched_input(tmp_path,change):
    from humanoid_mimicgen.workflow.navigation.whole_body import WholeBodyOperation
    adapter,plan,strategy,source,ref,_=operation_fixture(tmp_path);hands=np.zeros((5,14))
    if change=='missing':ref=None
    if change=='fingers':hands[3,8]=1
    if change=='start':adapter.d.qpos[0]+=.1
    if change=='timing':strategy['policy']['whole_body_operation']['transition_s']=1.
    with pytest.raises(WorkflowError):WholeBodyOperation(adapter,plan,strategy,source,hands,generated_reference=ref)


def test_binding_rejects_generated_phase_and_fingers_mismatch(tmp_path):
    from humanoid_mimicgen.workflow.navigation.lateral_execution import validate_generated_binding
    adapter,plan,strategy,source,ref,_=operation_fixture(tmp_path)
    plan['prior']['operation_interval']=[0,4]
    validate_generated_binding(plan,strategy)
    strategy['command_pose_offset']=1
    with pytest.raises(WorkflowError):validate_generated_binding(plan,strategy)


def test_measured_object_movement_invalidates_plan():
    from humanoid_mimicgen.workflow.navigation.lateral_execution import check_measured_object
    m,q,p,b,o,c=mechanism();p['bindings']=b
    plan=dict(prior=p,scene_change=dict(delta_world_m=[0,0,0]),lateral_options=o)
    check_measured_object(m,q[0],plan)
    moved=q[0].copy();moved[-6]+=.02
    with pytest.raises(WorkflowError,match='object_moved'):check_measured_object(m,moved,plan)


def test_foreign_generated_bundle_and_wrong_source_timing_are_rejected(tmp_path):
    from humanoid_mimicgen.workflow.navigation.whole_body import WholeBodyOperation
    from humanoid_mimicgen.workflow.navigation.reference_bundle import load_reference_bundle
    adapter,plan,strategy,source,ref,_=operation_fixture(tmp_path)
    data=load_reference_bundle(ref);meta=data.pop('metadata');data['time_s']=np.arange(5)*10.
    foreign=save_reference_bundle(tmp_path/'foreign',data,dict(operation_interval=[0,4],source_evidence_sha256={}))
    with pytest.raises(WorkflowError):WholeBodyOperation(adapter,plan,strategy,source,np.zeros((5,14)),generated_reference=foreign)
