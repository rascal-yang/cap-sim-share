"""Measured local leg response and bounded reference feedback before SONIC encoding."""
import copy,json,shutil
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from .agent import save
from .encoder_legs import JOINTS,observer_contract
from ..workflow.contracts import file_hash

def paired(x):return np.tile(np.asarray(x,float),2)

def identify(plus,minus,epsilon):
    p=np.asarray(plus,float);m=np.asarray(minus,float)
    if p.shape!=(3,6) or m.shape!=p.shape or not np.isfinite([p,m]).all() or not np.isfinite(epsilon) or epsilon<=0:raise ValueError('Invalid finite-difference data')
    j=((p-m)/(2*epsilon)).T;u,s,vt=np.linalg.svd(j,full_matrices=False);keep=s>max(1e-8,s[0]*.01);rec=np.zeros_like(s);rec[keep]=1/s[keep];inverse=(vt.T*rec)@u.T
    return {'jacobian':j.tolist(),'inverse':inverse.tolist(),'singular_values':s.tolist(),'rank':int(sum(keep)),'relative_svd_cutoff':.01}

def advance(value,error,inverse,tau,limit,rate_limit,dt):
    value=np.asarray(value,float);error=np.asarray(error,float);inverse=np.asarray(inverse,float)
    if value.shape!=(3,) or error.shape!=(6,) or inverse.shape!=(3,6) or not all(np.isfinite(x).all() for x in [value,error,inverse]) or min(tau,limit,rate_limit,dt)<=0:raise ValueError('Invalid feedback input')
    rate=np.clip(inverse@error/tau,-rate_limit,rate_limit);out=np.clip(value+dt*rate,-limit,limit)
    return out,(out-value)/dt

def predict(value,rate,times,limit):
    raw=np.asarray(value)[None,:]+np.asarray(times)[:,None]*np.asarray(rate)[None,:];q=np.clip(raw,-limit,limit);v=np.broadcast_to(rate,q.shape).copy();v[(raw>=limit)|(raw<=-limit)]=0
    return q,v

def forecast(value,rate,index,length,limit):
    ids=np.minimum(index+np.arange(10)*5,length-1)
    q,v=predict(value,rate,(ids-index)*.02,limit)
    v[ids==length-1]=0
    return ids,q,v

@contextmanager
def scope(config,plan,variant,model=None,replay=False):
    from . import suite_physics as ph
    from .saved_token_replay import verify_source_contract
    old_source,old_encode,old_write=ph.source_poses,ph.encode_reference,ph.write_new
    goal=Path(config['measured_replay_case']);verify_source_contract(goal);gr=json.loads((goal/'result.json').read_text());gt=np.load(goal/'execution/trajectory.npz');params=plan['leg_response'];cache={};logs=[];offset=np.zeros(3)
    if gr['mode']!='original_token' or not gr['success'] or gr['coverage']!=1 or gr['failure'] or gr['task']!='G1WholebodyChairSitTeleop' or gr['episode']!=plan['episode']:raise ValueError('Verified successful original-token physics replay required')
    def source(b,df):
        q,v,poses=old_source(b,df)
        if replay:return q,v,poses
        import mujoco
        gm=mujoco.MjModel.from_binary_path(str(goal/'scene.mjb'));names=[b.m.joint(int(j)).name for j in b.robot.body_joint_index];indices=[names.index(n) for n in JOINTS]
        if any(i>=12 for i in indices) or gm.nq!=b.m.nq or any(gm.joint(n).qposadr!=b.m.joint(n).qposadr for n in names):raise ValueError('Leg joint mapping differs')
        qids=b.m.jnt_qposadr[b.robot.body_joint_index];vids=b.m.jnt_dofadr[b.robot.body_joint_index];states=gt['state'][::4];gq=states[:,1:1+b.m.nq][:,qids][:,indices];gv=states[:,1+b.m.nq:][:,vids][:,indices]
        # Use final POST-state for padding, never a stale pre-control state/velocity.
        if len(df)>len(states):
            final=gt['next_state'][-1,1:1+b.m.nq][qids][indices];gq=np.vstack([gq,np.repeat(final[None],len(df)-len(states),axis=0)]);gv=np.vstack([gv,np.zeros((len(df)-len(states),6))])
        cache.update(names=names,indices=np.asarray(indices),qids=qids,vids=vids,goal_q=gq[:len(df)],goal_dq=gv[:len(df)],ranges=b.m.jnt_range[b.robot.body_joint_index][indices])
        return q,v,poses
    def encode(s,d,q,v,poses,index,shift,pose_frame=None):
        nonlocal offset
        ids=cache['indices'];qids=cache['qids'][ids];vids=cache['vids'][ids];error=cache['goal_q'][index]-d.qpos[qids];times=np.arange(10)*.1;future=np.minimum(index+np.arange(10)*5,len(q)-1)
        if variant['kind']=='probe':
            a=plan['contact_start'];u=np.clip(((future-a)*.02)/params['ramp_s'],0,1);w=10*u**3-15*u**4+6*u**5;wd=(30*u**2-60*u**3+30*u**4)/params['ramp_s'];vector=np.asarray(variant['vector']);delta=w[:,None]*vector;dv=wd[:,None]*vector
            # Clamped future frames represent held terminal reference.
            dv[future==len(q)-1]=0
            value=delta[0];rate=dv[0]
        else:
            value,rate=advance(offset,error,model['inverse'],params['feedback_timescale_s'],params['max_offset_rad'],params['max_rate_rad_s'],.02);offset=value
            future,delta,dv=forecast(value,rate,index,len(q),params['max_offset_rad'])
        qq=q.copy();vv=v.copy();after=q[future][:,qids]+np.tile(delta,(1,2));vel=v[future][:,vids]+np.tile(dv,(1,2));ranges=cache['ranges'];clipped=np.clip(after,ranges[:,0],ranges[:,1]);vel[clipped!=after]=0
        # Duplicate terminal sample IDs must all use the terminal (first occurrence) forecast.
        _,first=np.unique(future,return_index=True)
        for j in first:qq[future[j],qids]=clipped[j];vv[future[j],vids]=vel[j]
        token,e=old_encode(s,d,qq,vv,poses,index,shift,pose_frame)
        logs.append({'frame':index,'goal_q':cache['goal_q'][index].tolist(),'goal_dq':cache['goal_dq'][index].tolist(),'measured_q':d.qpos[qids].tolist(),'measured_dq':d.qvel[vids].tolist(),'error_rad':error.tolist(),'offset_paired_rad':value.tolist(),'offset_rate_rad_s':rate.tolist(),'saturated':bool(np.any(abs(value)>=params['max_offset_rad']-1e-10)),'joint_limit_clipped':bool(np.any(clipped!=after)),'future_frames':future.tolist(),'encoder_q':e[661:781].reshape(10,12)[:,ids].tolist(),'encoder_dq':e[781:901].reshape(10,12)[:,ids].tolist()})
        return token,e
    def write(path,value):
        path=Path(path)
        if not replay and path.name=='contact_contract.json':value=observer_contract(value)
        if path.name=='experiment.json':
            value=copy.deepcopy(value);folder=path.parent;bound=[Path(config['config_path']),goal/'result.json',goal/'experiment.json',goal/'scene.mjb',goal/'execution/trajectory.npz']
            if model:
                for case in model['source_cases']:
                    verify_source_contract(Path(case))
                    bound.extend(Path(case)/n for n in ['result.json','response_audit.json','execution/trajectory.npz'])
            variant_file=folder/'response_variant.json';save(variant_file,{'variant':variant,'parameters':params,'model':model,'replay':replay});bound.append(variant_file)
            if not replay:
                artifact=folder/'measured_goal.npz';np.savez_compressed(artifact,q=cache['goal_q'],dq=cache['goal_dq'],joint_names=JOINTS);bound.append(artifact)
                st=value['control']['stage_interfaces'][-1];st['joint_residual_interface']=observer_contract(st['joint_residual_interface']);st['lower_body_interface']='SONIC mode1 source leg reference plus measured-response probe or closed-loop reference correction';st['preencoder_leg_response']={'variant':variant,'parameters':params,'measured_replay_source':str(goal),'reference_shift':gr['object_delta_xy_m'],'tested_shift':config['task']['source']['object_delta_xy_m'],'paired_joint_order':['bilateral hip_pitch','bilateral knee','bilateral ankle_pitch'],'feedback_error':'same-frame successful replay measured q minus current actual q; final post-state hold after replay end','preview':'0..0.9s linear correction forecast, bounded, dq includes forecast rate; duplicate terminal samples held','postdecoder_corrections':False,'explicit_pelvis_target':False}
                value['purpose']['planned_changes']=[observer_contract(x) if isinstance(x,dict) and 'contact_control' in x else x for x in value['purpose']['planned_changes']];value['purpose']['planned_changes'].append(st['preencoder_leg_response'])
            value['pipeline'].update(entrypoint='leg_response_study.run',feedback_and_reflection_rules='API chooses bounded calibration/feedback parameters; local Python deterministic finite differences and reference feedback; no runtime API; no PD/decoder edits',allowed_retry_changes=['response parameters only; each case variant and response model frozen'])
            for p in bound:value['preflight']['evidence_paths'].append(str(p));value['preflight']['evidence_sha256'][str(p)]=file_hash(p)
            shutil.copy2(config['config_path'],folder/'source_snapshot/leg_response_config.json')
        if not replay and path.name in ('result.json','execution_audit.json'):
            value.update(control_variant='mode1_measured_replay_leg_response_fixed_pd',post_decoder_residual_calls=0,contact_observer_only=True,token_alone_replays_execution='pending_independent_replay',planner_calls=0,leg_response_variant=variant)
            # Written once, before result is published; repeated suite audit/result writes share this list.
            out=Path(value['experiment_path'])/'execution/leg_response.json'
            if not out.exists():save(out,logs)
        return old_write(path,value)
    ph.source_poses=source;ph.write_new=write
    if not replay:ph.encode_reference=encode
    try:yield logs
    finally:ph.source_poses=old_source;ph.encode_reference=old_encode;ph.write_new=old_write
