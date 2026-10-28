"""Extract task-frame stance relations from measured replay, without physics."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import WorkflowError, positive


def wrap(angle):
    return (np.asarray(angle)+np.pi)%(2*np.pi)-np.pi


def finite(value, shape, name):
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.isfinite(a).all():
        raise WorkflowError('invalid_input',name)
    return a


def yaw(poses):
    a=np.asarray(poses,float)
    if not np.allclose(np.linalg.norm(a[...,3:7],axis=-1),1,atol=1e-5):
        raise WorkflowError('invalid_input','Nonunit quaternion')
    return Rotation.from_quat(np.roll(a[...,3:7],-1,axis=-1)).as_euler('xyz')[...,2]


def rotation2(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])


def task_frame(object_pose, declaration):
    pose=finite(object_pose,(7,),'Object pose');yaw(pose)
    kind=declaration.get('kind')
    if kind=='world_heading':
        if not declaration.get('evidence'):
            raise WorkflowError('needs_annotation','Heading needs scene evidence')
        angle=float(declaration['yaw_rad'])
    elif kind=='object_axis':
        axis=finite(declaration['axis'],(3,),'Semantic object axis')
        direction=Rotation.from_quat(np.roll(pose[3:],-1)).apply(axis)
        if np.linalg.norm(direction[:2])<1e-6:
            raise WorkflowError('needs_annotation','Ambiguous horizontal object axis')
        angle=float(np.arctan2(direction[1],direction[0]))
    else:
        raise WorkflowError('needs_annotation','Explicit semantic or environment frame required')
    return finite([*pose[:3],angle],(4,),'Task frame').tolist()


def transfer_stance(prior, target_frame):
    target=finite(target_frame,(4,),'Target frame xyz/yaw')
    relative=finite(prior['root_in_task_xyyaw'],(3,),'Relative stance')
    return np.r_[target[:2]+rotation2(target[3])@relative[:2],wrap(target[3]+relative[2])].tolist()


def extract_prior(trace, bindings, options):
    """Latest stable pre-entry window; never infer success or semantic object identity."""
    t=np.asarray(trace['time_s'],float);n=len(t)
    if t.ndim!=1 or n<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise WorkflowError('invalid_input','Strictly increasing time required')
    root=finite(trace['root_pose'],(n,7),'Root poses')
    obj=finite(trace['object_pose'],(n,7),'Object poses')
    vel=finite(trace['root_velocity'],(n,6),'Measured root velocity')
    heading=yaw(root);yaw(obj)
    support=np.asarray(trace['support_ok'])
    if support.shape!=(n,) or support.dtype!=np.dtype(bool):
        raise WorkflowError('needs_annotation','Boolean support evidence required')
    entry=trace.get('entry_index');end=trace.get('end_index')
    if type(entry) is not int or type(end) is not int or not 0<entry<=end<n:
        raise WorkflowError('needs_annotation','Explicit operation phase indices required')
    if not bindings.get('object'):
        raise WorkflowError('needs_annotation','Operation object binding required')
    for key in ('window_s','lookback_s','speed_m_s','yaw_speed_rad_s','drift_m'):
        positive(options[key],key)
    frame=task_frame(obj[entry],bindings['frame'])
    base={'schema_version':1,'kind':'replay_navigation_prior','task':trace.get('task'),
          'success_criterion':trace.get('success_criterion'),'bindings':bindings,'options':options,
          'operation_interval':[entry,end],'source_task_frame_xyzyaw':frame,
          'physics_steps':0,'api_calls':0,'dynamic_execution_verified':False,
          'original_sequences_accessible':True,'operation_kind':'trajectory_retargeting'}
    if trace.get('success') is not True or not trace.get('success_criterion'):
        return dict(base,status='unverified_source',reason='Explicit successful acceptance evidence required')
    stable=(np.linalg.norm(vel[:,:2],axis=1)<=options['speed_m_s']) & (abs(vel[:,5])<=options['yaw_speed_rad_s']) & support
    first=int(np.searchsorted(t,t[entry]-options['lookback_s']))
    selected=None
    for right in range(entry,first,-1):
        left=int(np.searchsorted(t,t[right]-options['window_s'],side='right'))-1
        if left<first or left<0 or not stable[left:right+1].all():continue
        xy=root[left:right+1,:2]
        if np.max(np.linalg.norm(xy-xy.mean(axis=0),axis=1))>options['drift_m']:continue
        # A quiet interval before another walking segment is not an operation stance.
        if np.max(np.linalg.norm(root[right:entry+1,:2]-xy.mean(axis=0),axis=1))>options['drift_m']:continue
        selected=(left,right);break
    if selected is None:
        return dict(base,status='unsupported_entry_pattern',reason='No recent stable supported pre-entry window')
    left,right=selected;poses=root[left:right+1];xy=np.median(poses[:,:2],axis=0)
    angle=float(np.arctan2(np.mean(np.sin(heading[left:right+1])),np.mean(np.cos(heading[left:right+1]))))
    rel=rotation2(-frame[3])@(xy-np.asarray(frame[:2]))
    keys=sorted(set([entry,end]+list(bindings.get('key_indices',[]))))
    if any(type(k) is not int or not entry<=k<=end for k in keys):
        raise WorkflowError('invalid_input','Key frame outside operation interval')
    return dict(base,status='completed',root_in_task_xyyaw=[*rel.tolist(),float(wrap(angle-frame[3]))],
                source_root_height_m=float(np.median(poses[:,2])),source_stance_xyyaw=[*xy.tolist(),angle],
                stable_window={'indices':[left,right],'time_s':[float(t[left]),float(t[right])],
                               'max_xy_deviation_m':float(np.max(np.linalg.norm(poses[:,:2]-xy,axis=1)))},
                key_indices=keys,environment_relations=trace.get('environment_relations',{}))
