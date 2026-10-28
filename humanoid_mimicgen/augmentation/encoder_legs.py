"""Isolated six-joint reference adaptation before mode1 encoding; no actuator edits."""
import copy
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .agent import save
from ..workflow.contracts import file_hash

JOINTS=tuple(side+'_'+joint+'_joint' for side in ('left','right') for joint in ('hip_pitch','knee','ankle_pitch'))

def validate(c):
    if not isinstance(c,dict) or set(c)!={'blend','offset_rad','ramp_s'}:raise ValueError('Invalid leg reference fields')
    blend=np.asarray(c['blend'],float);offset=np.asarray(c['offset_rad'],float)
    if blend.shape!=(6,) or offset.shape!=(6,) or not np.isfinite(blend).all() or not np.isfinite(offset).all():raise ValueError('Six finite coefficients required')
    if np.any(blend<0) or np.any(blend>1) or np.any(np.abs(offset)>.3):raise ValueError('Leg blend/offset exceeds pilot capability')
    if type(c['ramp_s']) not in (int,float) or not np.isfinite(c['ramp_s']) or not .2<=c['ramp_s']<=2:raise ValueError('Invalid ramp')
    return c

def transform(q,v,reference,indices,blend,offset,start,ramp,dt=.02):
    validate({'blend':blend,'offset_rad':offset,'ramp_s':ramp})
    q=np.asarray(q,float);v=np.asarray(v,float);reference=np.asarray(reference,float);ids=np.asarray(indices)
    if q.ndim!=2 or v.shape!=q.shape or reference.shape!=q.shape or len(q)<3 or not np.isfinite([q,v,reference]).all():raise ValueError('Invalid leg arrays')
    if ids.shape!=(6,) or len(set(ids.tolist()))!=6 or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=q.shape[1]):raise ValueError('Invalid joint mapping')
    if type(start) is not int or not 0<=start<len(q) or dt<=0:raise ValueError('Invalid activation')
    u=np.clip((np.arange(len(q))-start)*dt/ramp,0,1);w=10*u**3-15*u**4+6*u**5
    delta=w[:,None]*(np.asarray(blend)*(reference[:,ids]-q[:,ids])+np.asarray(offset))
    dv=np.gradient(delta,dt,axis=0);dv[:start]=0
    if np.max(np.abs(delta))>.5+1e-12 or np.max(np.abs(dv))>2+1e-12:raise ValueError('Correction exceeds 0.5rad/2rad_s pilot bounds')
    out=q.copy();vel=v.copy();out[:,ids]+=delta;vel[:,ids]+=dv
    return out,vel

def tracking_rmse(actual,reference,start):
    if len(actual)<=start:return None
    return np.sqrt(np.mean((actual[start:]-reference[start:])**2,axis=0)).tolist()

def execution_metadata(result):
    result.update(control_variant='mode1_preencoder_six_legs_fixed_pd',post_decoder_residual_calls=0,contact_observer_only=True,token_alone_replays_execution='pending_independent_replay',planner_calls=0)

def observer_contract(c):
    if c['contact_control']['joints']:raise ValueError('Observer must have empty joint list')
    c=copy.deepcopy(c)
    c.update(interface='contact observer and unchanged safety only; identity decoder targets and PD',postdecoder_corrections=False,token_alone_replays_execution='pending_independent_replay',reference_source='unused joint reference; retained native criteria provenance',bounds='unchanged selected-contact-force stop; no joint correction')
    return c

@contextmanager
def reference_scope(config,plan,enabled=True):
    """Scoped hooks; prepare arrays before first physics step, freeze all evidence."""
    from . import suite_physics as ph
    from .saved_token_replay import verify_source_contract
    old_source,old_write=ph.source_poses,ph.write_new
    ref=Path(config['leg_reference_case']);verify_source_contract(ref)
    result=json.loads((ref/'result.json').read_text())
    if result['task']!='G1WholebodyChairSitTeleop' or not result['success'] or result['coverage']!=1 or result['failure'] or result['episode']!=plan['episode']:raise ValueError('Successful full matching chair reference required')
    with np.load(ref/'execution/trajectory.npz') as f:reference=f['commands'][::4,1:30].copy()
    prepared={}
    def source(b,df):
        q,v,poses=old_source(b,df)
        if not enabled:return q,v,poses
        names=[b.m.joint(int(j)).name for j in b.robot.body_joint_index];ids=[names.index(n) for n in JOINTS]
        if len(names)!=29 or any(i>=12 for i in ids):raise ValueError('Unexpected body joint mapping')
        # Verify reference scene joint addresses match the live model before indexing commands.
        import mujoco
        rm=mujoco.MjModel.from_binary_path(str(ref/'scene.mjb'))
        if rm.nq!=b.m.nq or any(rm.joint(n).qposadr!=b.m.joint(n).qposadr for n in names):raise ValueError('Reference joint layout mismatch')
        qids=b.m.jnt_qposadr[b.robot.body_joint_index];vids=b.m.jnt_dofadr[b.robot.body_joint_index]
        target=reference[np.minimum(np.arange(len(q)),len(reference)-1)]
        qc,vc=transform(q[:,qids],v[:,vids],target,ids,plan['leg_reference']['blend'],plan['leg_reference']['offset_rad'],plan['contact_start'],plan['leg_reference']['ramp_s'])
        ranges=b.m.jnt_range[b.robot.body_joint_index[ids]]
        if np.any(qc[:,ids]<ranges[:,0]) or np.any(qc[:,ids]>ranges[:,1]):raise ValueError('Corrected leg reference outside joint limits')
        prepared.update(names=names,indices=ids,source_q=q[:,qids].copy(),source_dq=v[:,vids].copy(),reference_q=target,corrected_q=qc,corrected_dq=vc)
        q=q.copy();v=v.copy();q[:,qids]=qc;v[:,vids]=vc
        return q,v,poses
    def write(path,value):
        path=Path(path)
        if enabled and path.name in ('result.json','execution_audit.json'):execution_metadata(value)
        if enabled and path.name=='contact_contract.json':value=observer_contract(value)
        if path.name=='experiment.json':
            value=copy.deepcopy(value);folder=path.parent
            evidence=[Path(config['config_path']),ref/'result.json',ref/'experiment.json',ref/'execution/trajectory.npz',ref/'scene.mjb']
            if enabled:
                if not prepared:raise ValueError('Leg references not prepared before freeze')
                artifact=folder/'leg_reference.npz';np.savez_compressed(artifact,**prepared);evidence.append(artifact)
                value['control']['stage_interfaces'][-1]['lower_body_interface']='six-joint corrected q/dq references into SONIC mode1; original token prefix unchanged'
                stage=value['control']['stage_interfaces'][-1]
                stage['joint_residual_interface']=observer_contract(stage['joint_residual_interface'])
                value['purpose']['planned_changes']=[observer_contract(x) if isinstance(x,dict) and 'contact_control' in x else x for x in value['purpose']['planned_changes']]
                value['control']['stage_interfaces'][-1]['preencoder_legs']={'joint_order':list(JOINTS),'parameters':plan['leg_reference'],'active_start_frame':plan['contact_start'],'active_end':'episode including fixed tail','source':str(ref),'source_type':'historically executed commanded PD targets; not measured posture','velocity_rule':'original dq plus central finite difference of smooth position correction; prefix correction zero','future_sampling':'10 frames with stride5 at50Hz, indices clamped at final frame','postdecoder_corrections':False,'fixed_PD':True,'explicit_pelvis_target':False}
                value['purpose']['planned_changes'].append({'encoder_leg_reference':plan['leg_reference']})
            value['pipeline'].update(entrypoint='encoder_leg_study.run',feedback_and_reflection_rules='API selects static preencoder leg reference; no online API, no postdecoder residual; native acceptance and unchanged contact observer/safety',allowed_retry_changes=['only six-joint preencoder blend/offset/ramp; fixed physical plan'])
            for p in evidence:
                value['preflight']['evidence_paths'].append(str(p));value['preflight']['evidence_sha256'][str(p)]=file_hash(p)
            shutil.copy2(config['config_path'],folder/'source_snapshot/encoder_leg_config.json')
        return old_write(path,value)
    ph.source_poses=source;ph.write_new=write
    try:yield prepared
    finally:ph.source_poses=old_source;ph.write_new=old_write
