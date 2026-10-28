"""Versioned snapshots for stateful preencoder controllers; no physics mutations."""
import copy
import numpy as np
from .retry_state import pack,unpack


def snapshot(leg=None,wrist=None,fingers=None):
    values={}
    if leg is not None:values['leg']={'offset':leg['offset'].copy()}
    if wrist is not None:values['wrist']={'feedback':wrist.feedback.copy()}
    if fingers is not None:values['fingers']={'integral':fingers.integral.copy(),'previous':None if fingers.previous is None else fingers.previous.copy()}
    return {'version':1,'controllers':pack(values)}


def restore_extensions(saved,leg=None,wrist=None,fingers=None):
    expected={k for k,v in [('leg',leg),('wrist',wrist),('fingers',fingers)] if v is not None}
    if saved.get('version')!=1 or set(saved.get('controllers',{}))!=expected:raise ValueError('Missing or mismatched controller checkpoint')
    values=unpack(saved['controllers'])
    for name,fields in values.items():
        for key,value in fields.items():
            shape=(3,) if name in ('leg','wrist') else (7,)
            if value is None and name=='fingers' and key=='previous':continue
            if not isinstance(value,np.ndarray) or value.shape!=shape or not np.isfinite(value).all():raise ValueError('Malformed controller state')
    if leg is not None:leg['offset']=values['leg']['offset'].copy()
    if wrist is not None:wrist.feedback=values['wrist']['feedback'].copy()
    if fingers is not None:
        fingers.integral=values['fingers']['integral'].copy();v=values['fingers']['previous'];fingers.previous=None if v is None else v.copy()


def validate_branch(parent,child,frame):
    """A changed reference cannot retroactively inherit actions it would change."""
    if type(frame) is not int or frame<0:raise ValueError('Invalid checkpoint frame')
    ignored={'rationale','hypothesis','action','resume','strategy'}
    for key in set(parent)|set(child):
        if key in ignored or parent.get(key)==child.get(key):continue
        if key=='finger_control':boundary=min(parent[key]['start_frame'],child[key]['start_frame'])
        elif key in ('wrist_delta_m','leg_response','preencoder'):boundary=min(parent['contact_start'],child['contact_start'])
        elif key=='tail_s':
            # Changing horizon changes encoder future preview, so use start boundary.
            boundary=min(parent['contact_start'],child['contact_start'])
        else:raise ValueError('Unapproved checkpoint policy change: '+key)
        if frame>boundary:raise ValueError('Checkpoint inherits an affected prefix: '+key)
