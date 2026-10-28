"""Strict fixed-root target translation comparison for button generalization."""
import json
from pathlib import Path
import numpy as np


def compare_arrays(current,baseline,target,delta):
    if set(current)!=set(baseline):raise ValueError('Scene model fields changed')
    for key,value in current.items():
        expected=baseline[key].copy()
        if key=='body_pos':expected[target]+=np.asarray(delta)
        if not np.allclose(value,expected,rtol=0,atol=1e-14,equal_nan=True):
            raise ValueError('Undeclared scene model change: '+key)


def numeric_fields(obj):
    out={}
    for k in dir(obj):
        if k.startswith('_'):continue
        value=getattr(obj,k)
        if isinstance(value,(np.ndarray,int,float,np.number,bool)):out[k]=np.asarray(value)
    return out


def validate_scene(case,baseline,target=None,branch=False):
    import mujoco
    case=Path(case);baseline=Path(baseline)
    read=lambda p:json.loads(p.read_text())
    if read(case/'physics.json')!=read(baseline/'physics.json'):raise ValueError('Physics changed')
    task=read(case/'task.json');prior=read(baseline/'task.json')
    if task['name']!=prior['name']:raise ValueError('Baseline task mismatch')
    delta=np.asarray(task['scene']['translation'])-np.asarray(prior['scene']['translation'])
    a=mujoco.MjModel.from_binary_path(str(case/'scene.mjb'));b=mujoco.MjModel.from_binary_path(str(baseline/'scene.mjb'))
    if target is None:
        # Task target identity is bound by native source/runtime, passed by runtime when available.
        raise ValueError('Target body identity required')
    jid=int(a.body_jntadr[target]);free=jid>=0 and a.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE
    arrays=lambda m:{k:getattr(m,k) for k in dir(m) if isinstance(getattr(m,k),np.ndarray)}
    compare_arrays(numeric_fields(a),numeric_fields(b),target,np.zeros(3) if free else delta)
    compare_arrays(numeric_fields(a.opt),numeric_fields(b.opt),target,np.zeros(3))
    if not branch:
        with np.load(case/'initial.npz') as x,np.load(baseline/'initial.npz') as y:
            if set(x.files)!=set(y.files):raise ValueError('Initial fields changed')
            for k in x.files:
                expected=y[k].copy()
                if free and k=='qpos':
                    adr=int(a.jnt_qposadr[jid]);expected[adr:adr+3]+=delta
                if not np.allclose(x[k],expected,rtol=0,atol=1e-14,equal_nan=True):raise ValueError('Initial state changed: '+k)
    return {'baseline':str(baseline),'world_target_delta_m':delta.tolist(),'target_body_id':int(target),'free_target':bool(free),'all_model_arrays_checked':True,'initial_state_exact_except_declared_target':not branch,'branch_initial_checked_by_checkpoint_manifest':branch}

validate_button_scene=validate_scene
