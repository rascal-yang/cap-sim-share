"""Validate generated inputs and regenerate once from measured landing support."""
import time
import hashlib
from pathlib import Path
import numpy as np
import mujoco
from ..contracts import WorkflowError, positive, canonical
from .archive import verify_evidence
from .prior import yaw
from .lateral_ik import LateralSolver,rotation
from .lateral_prior import load_lateral_source
from .reference_bundle import load_reference_bundle,save_reference_bundle


def validate_generated_binding(plan,strategy):
    bundle=load_reference_bundle(plan['generated_reference'])
    if strategy['command_pose_offset']!=0:
        raise WorkflowError('unsupported_capability','Generated references require explicit zero source-row offset')
    interval=[strategy['phases'][0]['start'],strategy['phases'][-1]['end']-1]
    if interval!=bundle['metadata']['operation_interval'] or interval!=plan['prior']['operation_interval']:
        raise WorkflowError('context_changed','Generated operation phase mismatch')
    config=strategy['policy']['whole_body_operation']
    if not np.isclose(config['transition_s'],bundle['bridge_time_s'][-1],atol=1e-9,rtol=0):
        raise WorkflowError('context_changed','Generated bridge timing mismatch')
    if not np.allclose(bundle['fingers'][0],config['open_fingers_rad'],atol=1e-7,rtol=0):
        raise WorkflowError('invalid_operation_entry','Generated entry is not open')
    return bundle


def plan_identity(plan):
    return hashlib.sha256(canonical(plan)).hexdigest()


def validate_derived_reference(plan,reference,source,current):
    bundle=load_reference_bundle(reference);meta=bundle['metadata'];rows=bundle['source_rows']
    if (meta.get('parent_plan_sha256')!=plan_identity(plan) or meta.get('derivation')!='measured_support_handoff' or
        meta.get('source_evidence_sha256')!=plan.get('evidence_sha256') or
        not np.array_equal(meta.get('measured_handoff_qpos'),current) or
        rows[-1]>=len(source) or not np.allclose(bundle['time_s'],source[rows,0],atol=1e-9,rtol=0)):
        raise WorkflowError('context_changed','Derived reference is not bound to this plan, measured state and source time')
    return bundle


def check_measured_object(model,current,plan):
    d=mujoco.MjData(model);d.qpos[:]=current;mujoco.mj_forward(model,d)
    obj=model.body(plan['prior']['bindings']['object']).id
    expected=np.asarray(plan['prior']['source_initial_object_pose']).copy()
    expected[:3]+=plan['scene_change']['delta_world_m'];o=plan['lateral_options']['ik']
    if (np.linalg.norm(d.xpos[obj]-expected[:3])>o['position_m'] or
        (rotation(expected[3:]).inv()*rotation(d.xquat[obj])).magnitude()>o['angle_rad']):
        raise WorkflowError('object_moved','Apple changed during navigation; no stale retargeting')
    return d


def regenerate_handoff(plan,current_qpos,output_dir):
    started=time.monotonic();timeout=plan['lateral_options']['handoff_timeout_s'];positive(timeout,'Handoff budget');deadline=started+timeout
    verify_evidence(plan['evidence_sha256'])
    trace,prior,fingers,evidence=load_lateral_source(plan['source_manifest'],plan['lateral_options'],deadline=deadline)
    prior['surface']=plan['prior']['surface'];m=mujoco.MjModel.from_binary_path(plan['scene_files']['model'])
    current=np.asarray(current_qpos,float).copy();d=check_measured_object(m,current,plan);b=prior['bindings']
    feet={n:np.r_[d.xpos[m.body(n).id],d.xquat[m.body(n).id]].tolist() for n in b['feet']}
    candidate=dict(id='measured_handoff',feet=feet,root_pose=current[:7].tolist(),
                   xyyaw=[*current[:2],float(yaw(current[:7]))],seed_qpos=current.tolist(),requires_navigation=False)
    options=dict(plan['lateral_options']);options['ik']=dict(options['ik'])
    remaining=timeout-(time.monotonic()-started)
    if remaining<=0:raise WorkflowError('planning_budget_exhausted','Handoff source loading budget')
    options['ik']['timeout_s']=min(options['ik']['timeout_s'],remaining)
    solver=LateralSolver(m,trace['qpos'],trace['time_s'],prior,b,options,deadline=deadline)
    remaining=timeout-(time.monotonic()-started)
    if remaining<=0:raise WorkflowError('planning_budget_exhausted','Handoff FK budget')
    solver.o['timeout_s']=min(solver.o['timeout_s'],remaining)
    result=solver.solve(candidate,np.asarray(plan['scene_change']['delta_world_m']),current)
    audit={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    if time.monotonic()-started>timeout:audit.update(feasible=False,reason='planning_budget_exhausted')
    if audit['feasible']:
        arrays={k:result[k] for k in ('qpos','time_s','source_rows','bridge_qpos','bridge_time_s')}
        arrays['fingers']=fingers[arrays['source_rows']]
        audit['generated_reference']=save_reference_bundle(output_dir,arrays,dict(operation_interval=prior['operation_interval'],
            source_evidence_sha256=plan['evidence_sha256'],candidate=candidate,measured_handoff_qpos=current.tolist(),
            source_control=prior['source_control'],original_token_steps=0,parent_plan_sha256=plan_identity(plan),
            derivation='measured_support_handoff',verification=result['verification'],solver_audit=result['solver_audit']))
    if time.monotonic()>deadline:audit.update(feasible=False,reason='planning_budget_exhausted',detail='Handoff artifact completion exceeded deadline')
    audit.update(planning_pause_wall_s=time.monotonic()-started,physics_steps=0,measured_qpos=current.tolist(),
                 support_kind='measured_feet_geometry_runtime_stability_gate_required')
    return audit
