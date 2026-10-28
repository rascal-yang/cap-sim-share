"""Measured support windows and explicitly timed hand commands. No simulation."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import WorkflowError, file_hash, positive
from .prior import finite, task_frame, yaw
from .archive import load_archive, verify_evidence


def align_hand_commands(state_times, control_times, hands, control_dt_s):
    t=np.asarray(state_times,float);c=np.asarray(control_times,float)
    positive(control_dt_s,'Control interval')
    if (t.ndim!=1 or c.ndim!=1 or not len(t) or not len(c) or
        not np.isfinite(t).all() or not np.isfinite(c).all() or np.any(np.diff(t)<=0) or
        (len(c)>1 and not np.allclose(np.diff(c),control_dt_s,atol=1e-8,rtol=0))):
        raise WorkflowError('needs_annotation','Noncontinuous command timestamps')
    h=finite(hands,(len(c),14),'Commanded fingers')
    # Small numerical tolerance only; no extrapolation beyond one command hold.
    rows=np.searchsorted(c,t+1e-9,side='right')-1
    if np.any(rows<0) or np.any(t-c[np.maximum(rows,0)]>=control_dt_s-1e-9):
        raise WorkflowError('needs_annotation','Hand command coverage missing')
    return h[rows].copy(),rows


def extract_support_prior(trace, bindings, options, body_poses, fingers):
    t=np.asarray(trace['time_s'],float);n=len(t)
    if t.ndim!=1 or n<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise WorkflowError('invalid_input','Increasing measured times required')
    entry,end=trace['entry_index'],trace['end_index']
    if type(entry) is not int or type(end) is not int or not 0<entry<end<n:
        raise WorkflowError('needs_annotation','Operation interval')
    feet=bindings['feet']
    if len(feet)!=2 or len(set(feet))!=2:raise WorkflowError('needs_annotation','Two distinct feet required')
    poses={k:finite(body_poses[k],(n,7),k) for k in feet}
    for value in poses.values():yaw(value)
    root=finite(trace['root_pose'],(n,7),'Measured root');yaw(root)
    fingers=finite(fingers,(n,14),'Aligned hand commands')
    opened=finite(options['open_fingers_rad'],(14,),'Open hands')
    for key in ('window_s','lookback_s','foot_drift_m','foot_angle_rad','finger_tolerance_rad'):
        positive(options[key],key)
    if type(options['max_windows']) is not int or options['max_windows']<1:raise WorkflowError('invalid_input','Window budget')
    support=np.asarray(trace['support_ok'])
    if support.shape!=(n,) or support.dtype!=bool:raise WorkflowError('needs_annotation','Support evidence')
    stages=bindings['operation_stages']
    if (set(stages)!={'entry','closure','first_contact','lift','end'} or
        stages['entry']!=entry or stages['end']!=end or
        any(type(i) is not int or not entry<=i<=end for i in stages.values()) or
        stages['lift']<max(stages['first_contact'],stages['closure'])):
        raise WorkflowError('needs_annotation','Explicit entry/contact/closure/lift/end indices')
    base=dict(schema_version=1,kind='apple_lateral_support_prior',status='unverified_source',
              operation_interval=[entry,end],bindings=bindings,options=options,
              success_criterion=trace.get('success_criterion'),support_windows=[],rejected_windows=[],
              key_indices=sorted(set(stages.values())),physics_steps=0,api_calls=0,
              operation_kind='trajectory_retargeting',original_sequences_accessible=True,
              planner_access_to_original_actions=True,dynamic_execution_verified=False)
    if trace.get('success') is not True or not trace.get('success_criterion'):return base
    closed=np.any(abs(fingers-opened)>options['finger_tolerance_rad'],axis=1)
    if closed[entry] or not closed[entry:end+1].any():return dict(base,status='invalid_operation_entry')
    closure=entry+int(np.flatnonzero(closed[entry:end+1])[0])
    if closure!=stages['closure']:raise WorkflowError('needs_annotation','Closure row does not match commands')
    first=int(np.searchsorted(t,t[entry]-options['lookback_s']))
    for right in range(entry,first,-1):
        left=int(np.searchsorted(t,t[right]-options['window_s']+1e-9,side='right'))-1
        if left<first or left<0:continue
        reason=None
        # Check through entry: a stable pause followed by another step is ineligible.
        for name,p in poses.items():
            pos=p[left:entry+1,:3];r=Rotation.from_quat(np.roll(p[left:entry+1,3:],-1,axis=1))
            if np.max(np.linalg.norm(pos-pos[0],axis=1))>options['foot_drift_m']:reason='foot_motion'
            if np.max((r[0].inv()*r).magnitude())>options['foot_angle_rad']:reason='foot_rotation'
        if not support[left:entry+1].all():reason='missing_support'
        if reason:
            base['rejected_windows'].append(dict(indices=[left,right],reason=reason));continue
        if base['support_windows'] and t[base['support_windows'][-1]['indices'][1]]-t[right]<options['window_s']-1e-9:continue
        # Representative actual frame retains coherent quaternion and joint posture.
        mid=(left+right)//2;fp={k:p[mid].tolist() for k,p in poses.items()}
        base['support_windows'].append(dict(indices=[left,right],time_s=[float(t[left]),float(t[right])],
            representative_row=mid,feet=fp,root_pose=root[mid].tolist(),qpos=np.asarray(trace['qpos'][mid]).tolist(),
            foot_xyz_range_m={k:np.ptp(p[left:right+1,:3],axis=0).tolist() for k,p in poses.items()},
            pelvis_xyz_range_m=np.ptp(root[left:right+1,:3],axis=0).tolist(),
            foot_separation_m=float(np.linalg.norm(poses[feet[0]][mid,:3]-poses[feet[1]][mid,:3])),
            root_relative_to_feet_center_m=(root[mid,:3]-np.mean([p[mid,:3] for p in poses.values()],axis=0)).tolist()))
        if len(base['support_windows'])>=options['max_windows']:break
    if not base['support_windows']:return dict(base,status='unsupported_entry_pattern')
    w=base['support_windows'][0];rp=np.asarray(w['root_pose']);frame=task_frame(trace['object_pose'][entry],bindings['frame'])
    base.update(status='completed',source_stance_xyyaw=[*rp[:2],float(yaw(rp))],source_root_height_m=float(rp[2]),
                source_task_frame_xyzyaw=frame,source_initial_object_pose=np.asarray(trace['object_pose'][0]).tolist(),
                environment_relations=trace.get('environment_relations',{}))
    return base


def load_lateral_source(manifest_path, options, deadline=None):
    trace,bindings,evidence=load_archive(manifest_path,deadline=deadline)
    files=trace['files'];declaration=trace['manifest'].get('hand_alignment',{})
    if not {'hands','controller'}<=files.keys() or declaration.get('kind')!='controller_timestamps':
        raise WorkflowError('needs_annotation','Paired hand/controller timestamp evidence required')
    with np.load(files['hands'],allow_pickle=False) as h: hands=np.c_[h['left'],h['right']]
    with np.load(files['controller'],allow_pickle=False) as c:times=c['records'][:,0]
    fingers,rows=align_hand_commands(trace['time_s'],times,hands,declaration['control_dt_s'])
    if 'command_hand_joints' in bindings:
        import mujoco
        model=mujoco.MjModel.from_binary_path(files['model']);names=bindings['command_hand_joints']
        if len(names)!=14 or len(set(names))!=14:raise WorkflowError('needs_annotation','Explicit 14-joint hand command mapping required')
        for index,name in enumerate(names):
            j=model.joint(name).id;lo,hi=model.jnt_range[j]
            if not model.jnt_limited[j] or np.any(fingers[:,index]<lo-1e-7) or np.any(fingers[:,index]>hi+1e-7):
                raise WorkflowError('joint_limit_violation','Commanded finger '+name)
    prior=extract_support_prior(trace,bindings,options['extraction'],trace['body_poses'],fingers)
    result=json.loads(Path(files['result']).read_text())
    dataset_shift=None
    if 'task' in files:dataset_shift=json.loads(Path(files['task']).read_text()).get('scene',{}).get('translation')
    prior.update(source_manifest=str(Path(manifest_path).resolve()),evidence_sha256=evidence,
                 source_trajectory_sha256=file_hash(files['trajectory']),command_row_map=rows.tolist(),
                 support_evidence_kind=trace['support_evidence_kind'],
                 source_control={k:result.get(k) for k in ('original_token_steps','encoder_modes','control_variant')},
                 source_dataset_translation_m=dataset_shift)
    verify_evidence(evidence)
    return trace,prior,fingers,evidence
