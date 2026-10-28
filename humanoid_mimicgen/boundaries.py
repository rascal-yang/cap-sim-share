"""Explicit semantic boundary states for reusable interaction clips.

Closed state ranges retain both endpoints. Adjacent clips may share a boundary
state. Stored command rows keep their original alignment; these arrays are not
an executable transition stream and must not be blindly concatenated/replayed.
"""
import numpy as np


def boundary_clip(evidence, start, end, entry_condition, completion_condition):
    t=np.asarray(evidence['time_s'])
    if (t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0)):
        raise ValueError('Invalid source timestamps')
    if (type(start) is not int or type(end) is not int or not 0<=start<end<len(t)):
        raise ValueError('Need two actual ordered source boundary frames')
    if not isinstance(entry_condition,str) or not entry_condition.strip() or not isinstance(completion_condition,str) or not completion_condition.strip():
        raise ValueError('Entry and completion semantics are required')
    arrays={k:np.asarray(v) for k,v in evidence.items()}
    if any(a.ndim==0 or len(a)!=len(t) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Evidence must consist of aligned finite arrays')
    def boundary(i,condition):
        return {'frame_index':i,'time_s':float(t[i]),'condition':condition,
                'state':{k:v[i].tolist() for k,v in arrays.items()
                         if k in ('qpos','qvel','object_pose','left_pose','right_pose','relative_position','relative_quaternion','clearance_m','left_contact','right_contact','support_contact')}}
    return ({k:v[start:end+1].copy() for k,v in arrays.items()},
            {'schema_version':2,'state_range_convention':'closed [start_frame,end_frame]',
             'start':boundary(start,entry_condition),'end':boundary(end,completion_condition),
             'duration_s':float(t[end]-t[start]),'sample_count':end-start+1,
             'shared_boundary_rule':'Keep shared state; command temporal alignment must be resolved by executor, never duplicate a control tick',
             'new_physical_execution_verified':False})
