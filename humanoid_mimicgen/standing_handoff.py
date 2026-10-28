"""Original-scene standing/hand-off probes; explicit controller cases, no fallback.

All cases retain the exact torque prefix. State writes after initialization are
forbidden. Frozen leg references constrain policy inputs, never simulator feet.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Slerp
from cap_sim.backends.native_sonic import Sonic,rot,quat
from .action_replay import sha,write,now,initialize_source,measurement
from .privileged import no_physics

CASES=('standing_idle','anchored_legs','source_mode0')


def standing_feed(context,yaw,seed):
    return {'context_mujoco_qpos':np.asarray(context)[None].astype(np.float32),'target_vel':np.array([-1],np.float32),
        'mode':np.array([0],np.int64),'movement_direction':np.zeros((1,3),np.float32),
        'facing_direction':np.array([[np.cos(yaw),np.sin(yaw),0]],np.float32),'height':np.array([-1],np.float32),
        'random_seed':np.array([seed],np.int64),'has_specific_target':np.zeros((1,1),np.int64),
        'specific_target_positions':np.zeros((1,4,3),np.float32),'specific_target_headings':np.zeros((1,4),np.float32),
        'allowed_pred_num_tokens':np.array([[0]*10+[1]],np.int64)}


def stable_sample(position_error,speed,angular_speed,wrist_error,wrist_angle,foot_forces):
    return bool(position_error<=.03 and speed<=.03 and angular_speed<=.15 and wrist_error<=.025
                and wrist_angle<=.2 and min(foot_forces)>10)


def body_future(program,t,hold=False):
    start=5.
    for action in program[1:]:
        if t<start+action['duration_s']-1e-8:break
        start+=action['duration_s']
    else:
        start-=action['duration_s']
    rows=np.asarray(action['body_keyframes']);local=max(0,t-start)
    def sample(offset):
        ts=np.clip(local+offset,rows[0,0],rows[-1,0])
        vals=np.stack([np.interp(ts,rows[:,0],rows[:,i]) for i in range(1,37)],axis=-1)
        vals[:,3:7]=quat(Slerp(rows[:,0],rot(rows[:,4:8]))(ts));return vals
    if hold:
        a=sample(np.zeros(10));return a,a.copy()
    return sample(np.arange(10)*.1),sample(np.arange(10)*.1+.01)


class StandingSonic(Sonic):
    def __init__(self,*args,case,**kwargs):
        super().__init__(*args,**kwargs);self.case=case;self.planner_records=[];self.anchor=None
    def plan_legs(self,d,velocity,height,facing_yaw=None):
        if self.anchor is None:self.anchor=np.r_[d.qpos[:7],d.qpos[self.qids]].copy()
        if self.case=='anchored_legs':
            self.ref=np.repeat(self.anchor[None],100,axis=0);self.ref_tick=self.tick
            self.planner_records.append({'time':float(d.time),'type':'fixed_measured_leg_reference','tick':self.tick,'reference':self.anchor.tolist()})
            return
        if self.case!='standing_idle':raise ValueError('Unexpected planner request')
        if self.ref is None:context=np.repeat(np.r_[d.qpos[:7],d.qpos[self.qids]][None],4,axis=0)
        else:
            context=self.sample((self.tick-self.ref_tick)*.6+np.arange(4));context[:,:2]+=d.qpos[:2]-context[0,:2]
        yaw=rot(d.qpos[3:7]).as_euler('xyz')[2] if facing_yaw is None else facing_yaw
        feed=standing_feed(context,yaw,17+self.serial)
        raw,n=self.planner.run(None,feed);self.ref=raw[0,:int(n.item())].copy();self.ref_tick=self.tick;self.serial+=1
        self.planner_records.append({'time':float(d.time),'type':'standing_idle','feed':{k:v.tolist() for k,v in feed.items()},'output_reference':self.ref.tolist()})


def check_contract(path):
    path=Path(path);r=json.loads(path.read_text())
    if sha(path)!=(path.parent/'experiment.sha256').read_text().strip():raise ValueError('Not frozen')
    if r['record_status']!='frozen_ready' or not r['preflight']['passed']:raise ValueError('Not ready')
    if r['control']['case'] not in CASES or r['control']['fallback_allowed'] or r['budget']['attempts_per_case']!=1:raise ValueError('Invalid case')
    hashes=r['preflight']['evidence_sha256']
    if str(Path(__file__).resolve()) not in hashes:raise ValueError('Unbound executor')
    for key in ['model','initial','controls','reference_states','reference_evidence','targets','task','source_controller','program']:
        if r['inputs'][key] not in hashes:raise ValueError('Unbound '+key)
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Evidence changed '+p)
    runtime=json.loads(Path(r['runtime']['actual_loaded_session_evidence']).read_text())
    if not runtime['passed'] or not runtime['sessions_loaded'] or len(runtime['actual_session_providers'])!=3:raise ValueError('Missing sessions')
    if any(x!=['CPUExecutionProvider'] for x in runtime['actual_session_providers'].values()):raise ValueError('Wrong providers')
    return r


def extended_measurement(b):
    ob=measurement(b.m,b.d);ob['root_velocity']=b.d.qvel[:6].tolist()
    ob['feet']={side:np.r_[b.d.xpos[b.m.body(side+'_ankle_roll_link').id],b.d.xquat[b.m.body(side+'_ankle_roll_link').id]].tolist() for side in ('left','right')}
    ob['foot_ground_N']=[sum(e['force_N'] for e in ob['contacts']['events'] if 'ground' in e['geoms'] and any(side+'_ankle' in x for x in e['bodies'])) for side in ('left','right')]
    ob['leg_velocity_rad_s']=b.d.qvel[b.vids[:12]].tolist()
    return ob


def run(contract):
    import mujoco
    from cap_sim.backends.humano import Backend
    from humano.core.action import ActionCmd
    from humano.mt.base import MotorCommand
    r=check_contract(contract);root=Path(contract).parent;out=root/'execution';out.mkdir(exist_ok=False);case=r['control']['case']
    task=json.loads(Path(r['inputs']['task']).read_text());b=Backend()
    with no_physics():b.describe(task)
    m,d=b.m,b.d;phys=json.loads(Path(r['physics']['physics_parameters_manifest']).read_text())
    for key,val in phys['arrays'].items():
        if not np.array_equal(getattr(m,key),np.asarray(val)):raise ValueError('Physics mismatch '+key)
    if abs(m.opt.timestep-.005)>1e-12:raise ValueError('dt changed')
    sonic=StandingSonic(m,b.robot,task['resources']['sonic_models'],b.initial,case=case)
    providers={k:getattr(sonic,k).get_providers() for k in ('encoder','decoder','planner')}
    if any(v!=['CPUExecutionProvider'] for v in providers.values()):raise ValueError('Provider mismatch')
    write(out/'actual_providers.json',providers)
    source=np.load(r['inputs']['reference_states'])['samples'];ctrl=np.load(r['inputs']['controls'])['ctrl'];targets=np.load(r['inputs']['targets'])
    old=json.loads(Path(r['inputs']['source_controller']).read_text());program=json.loads(Path(r['inputs']['program']).read_text())
    initialize_source(m,d,np.load(r['inputs']['initial']))
    np.savez_compressed(out/'actual_initial.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qacc_warmstart=d.qacc_warmstart.copy())
    states=[];obs=[];commands=[];tracking=[];events=[];failure=None;exception=None;steps=0;stage='prefix';si=1599;contact_started=None;settled=0.;bad=lost=handtime=0.;longest_stable=0.;anchor=None;last_target=None
    start=now();clock=time.perf_counter();limit=round(r['budget']['simulation_seconds_per_attempt']/.005)
    try:
        while steps<limit:
            if steps<1600:
                if steps%4==0:
                    sonic.history.append({'omega':d.qvel[3:6].copy(),'q':(d.qpos[b.qids]-sonic.default)[sonic.order],'dq':d.qvel[b.vids][sonic.order].copy(),'last':sonic.last.copy(),'gravity':rot(d.qpos[3:7]).inv().apply([0,0,-1])})
                    last_target=np.asarray(old[steps//4]['target']);sonic.last=((last_target-sonic.default)/sonic.scale)[sonic.order];sonic.tick+=1
                d.ctrl[:]=ctrl[steps];source_index=steps
            else:
                if steps==1600:
                    err=float(max(abs(d.qpos-source[1599,1:58]).max(),abs(d.qvel-source[1599,58:113]).max()))
                    if err>1e-10:raise ValueError('Prefix mismatch')
                    anchor=d.qpos[:7].copy();stage='standing';events.append({'event':'handoff','time_s':float(d.time),'prefix_error':err,'root_anchor':anchor.tolist()})
                if stage=='standing':source_index=1599;j=1599
                else:source_index=si;j=si-1
                if stage!='standing':
                    if si==1880:
                        ok=handtime>=.12 and obs[-1]['contacts']['hand_object_N']>.1;events.append({'event':'lift_gate','time_s':float(d.time),'passed':ok})
                        if not ok:failure='lift_contact_gate';break
                    if si==2434:
                        ok=obs[-1]['clearance_m']>=.03 and obs[-1]['contacts']['hand_object_N']>.1;events.append({'event':'hold_gate','time_s':float(d.time),'passed':ok})
                        if not ok:failure='hold_entry_gate';break
                    stage='grasp' if si<=1879 else ('lift' if si<=2433 else ('hold' if si<=2654 else 'source_tail'))
                if steps%4==0:
                    wrists=[(targets[side+'_pose'][j,:3],targets[side+'_pose'][j,3:]) for side in ('left','right')]
                    torso=targets['torso_pose'][j];rp=targets['root_pose'][j]
                    bf=body_future(program,8. if stage=='standing' else float(source[j,0]),stage=='standing') if case=='source_mode0' else None
                    target=sonic.step(d,wrists,(torso[:3],torso[3:]),np.zeros(2),float(rp[2]),bf,float(rot(rp[3:]).as_euler('xyz')[2]))
                    expected=0 if case=='source_mode0' else 1
                    if sonic.records[-1]['mode']!=expected:raise ValueError('Mode deviation')
                    row={'time_s':float(d.time),'stage':stage,'source_index':j,'mode':expected,'target_jump_leg_max_rad':float(abs(target[:12]-last_target[:12]).max()),'target_jump_body_max_rad':float(abs(target-last_target).max())}
                    for side,bid,(p,q) in zip(('left','right'),(b.left,b.right),wrists):
                        row[side]={'target_pose':np.r_[p,q].tolist(),'actual_pose':np.r_[d.xpos[bid],d.xquat[bid]].tolist(),'position_error_m':float(np.linalg.norm(d.xpos[bid]-p)),'orientation_error_rad':float((rot(q).inv()*rot(d.xquat[bid])).magnitude())}
                    tracking.append(row);last_target=target.copy()
                fingers=targets['fingers'][source_index]
                b.robot.apply_action(ActionCmd('wbc_torque',low_cmd=MotorCommand(q_target=target,kp=sonic.kp,kd=sonic.kd),use_sensor=False,left_hand_cmd=MotorCommand.hand_command(fingers[:7]),right_hand_cmd=MotorCommand.hand_command(fingers[7:])))
                commands.append({'time_s':float(d.time),'stage':stage,'source_index':source_index,'body_target_rad':target.tolist(),'fingers_rad':fingers.tolist(),'ctrl_Nm':d.ctrl.tolist()})
            mujoco.mj_step(m,d);steps+=1;mujoco.mj_forward(m,d)
            ob=extended_measurement(b);ob.update(time_s=float(d.time),stage=stage,source_poststep_frame=int(source_index));states.append(np.r_[d.time,d.qpos,d.qvel,d.ctrl]);obs.append(ob)
            c=ob['contacts'];handtime=handtime+.005 if c['hand_object_N']>.1 else 0.;bad=bad+.005 if c['robot_table_N']>5 else 0.;lost=lost+.005 if stage in ('lift','hold','source_tail') and c['hand_object_N']<=.1 else 0.
            _,x,y,_=d.qpos[3:7];tilt=np.arccos(np.clip(1-2*(x*x+y*y),-1,1))
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():failure='nonfinite'
            elif d.qpos[2]<.3 or tilt>np.deg2rad(50):failure='fall'
            elif c['robot_table_N']>50 or bad>=.05-1e-10:failure='table_contact'
            elif lost>=.03-1e-10:failure='contact_lost'
            elif any(w.number for w in d.warning):failure='simulation_warning'
            if failure:break
            if stage=='standing':
                errs=[];angles=[]
                for side,bid in [('left',b.left),('right',b.right)]:
                    p=targets[side+'_pose'][1599];errs.append(np.linalg.norm(d.xpos[bid]-p[:3]));angles.append((rot(p[3:]).inv()*rot(d.xquat[bid])).magnitude())
                stable=stable_sample(np.linalg.norm(d.qpos[:2]-anchor[:2]),np.linalg.norm(d.qvel[:2]),abs(d.qvel[5]),max(errs),max(angles),ob['foot_ground_N'])
                settled=settled+.005 if stable else 0.;longest_stable=max(longest_stable,settled);ob.update(stable_sample=stable,continuous_stable_s=settled)
                if steps>=1800 and steps%4==0 and settled>=.3-1e-10:
                    contact_started=float(d.time);events.append({'event':'standing_pass','time_s':contact_started,'continuous_stable_s':settled,'root_pose':d.qpos[:7].tolist()});stage='grasp';si=1600
                elif steps>=2000:failure='standing_timeout';break
            elif steps>1600:
                if si==2679:break
                si+=1
        if not failure and (contact_started is None or si<2679):failure='budget_exhausted'
    except Exception as e:failure='executor_exception';exception=repr(e)
    finally:
        np.savez_compressed(out/'states.npz',samples=np.asarray(states));write(out/'observations.json',obs);write(out/'contacts.json',[o['contacts'] for o in obs]);write(out/'commands.json',commands);write(out/'tracking.json',tracking);write(out/'events.json',events);write(out/'controller.json',sonic.records);write(out/'planner.json',sonic.planner_records)
        audit={'experiment_id':r['experiment_id'],'case':case,'started_at':start,'ended_at':now(),'contract_sha256':sha(contract),'physics_steps':steps,'recorded_steps':len(states),'missing_measurement_steps':steps-len(states),'failure':failure,'exception':exception,'completed':failure is None and si==2679,'contact_started_at_s':contact_started,'longest_stable_s':longest_stable,'actual_encoder_modes':sorted(set(x['mode'] for x in sonic.records)),'planner_inference_calls':sonic.serial,'encoder_calls':len(sonic.records),'decoder_calls':len(sonic.records),'online_ik_calls':0,'midrun_state_resets':0,'simulation_wall_s':time.perf_counter()-clock,'sim_seconds':float(d.time),'fallback':False}
        write(out/'execution_audit.json',audit);b.close()
    return audit


def verify(contract):
    from cap_sim.backends.humano import Backend
    r=check_contract(contract);root=Path(contract).parent;out=root/'execution';a=json.loads((out/'execution_audit.json').read_text());ob=json.loads((out/'observations.json').read_text());tr=json.loads((out/'tracking.json').read_text());s=np.load(out/'states.npz')['samples']
    b=Backend()
    try:
        with no_physics():b.describe(json.loads(Path(r['inputs']['task']).read_text()));native=b.verify(out)
    finally:b.close()
    post=ob[1600:];anchor=np.asarray(ob[1599]['root_pose']);times=np.array([o['time_s'] for o in ob]);p=np.array([o['object_pose'] for o in ob]);wr=np.array([o['right_pose'] for o in ob]);rel=rot(wr[:,3:]).inv().apply(p[:,:3]-wr[:,:3]);rq=(rot(wr[:,3:]).inv()*rot(p[:,3:])).as_quat();velocity=np.gradient(p[:,2],times)
    indices=[i for i,o in enumerate(ob) if o['stage']!='standing' and 2433<=o['source_poststep_frame']<=2654];longest=runlength=0;holdmin=None
    if indices:
        lo=indices[0];holdmin=min(ob[i]['clearance_m'] for i in indices)
        for i in indices:
            good=ob[i]['clearance_m']>=.03 and ob[i]['contacts']['hand_object_N']>.1 and ob[i]['contacts']['support_N']<.1 and abs(velocity[i])<=.04 and np.linalg.norm(rel[i]-rel[lo])<=.02 and 2*np.arccos(np.clip(abs(rq[i]@rq[lo]),0,1))<=.25
            runlength=runlength+1 if good else 0;longest=max(longest,runlength)
    hold=max(0,longest-1)*.005
    def stat(values):
        if not values:return None
        x=np.asarray(values);return {'rmse':float(np.sqrt(np.mean(x*x))),'p95':float(np.percentile(x,95)),'max':float(x.max())}
    foot={side:{'max_horizontal_displacement_m':float(max(np.linalg.norm(np.asarray(o['feet'][side][:2])-np.asarray(ob[1599]['feet'][side][:2])) for o in post)), 'max_rise_m':float(max(o['feet'][side][2]-ob[1599]['feet'][side][2] for o in post))} for side in ('left','right')}
    result={'experiment_id':r['experiment_id'],'case':r['control']['case'],'standing_pass':a['contact_started_at_s'] is not None,'success':bool(a['completed'] and native['success'] and hold>=1 and max(o['contacts']['robot_table_N'] for o in ob)<=5),'failure':a['failure'],'native':native,'stable_hold_s':hold,'hold_min_clearance_m':holdmin,'max_clearance_m':max(o['clearance_m'] for o in post),'max_hand_N':max(o['contacts']['hand_object_N'] for o in post),'max_table_N':max(o['contacts']['robot_table_N'] for o in post),'table_contact_gt_0p1_s':sum(o['contacts']['robot_table_N']>.1 for o in post)*.005,'max_root_xy_drift_m':float(max(np.linalg.norm(np.asarray(o['root_pose'][:2])-anchor[:2]) for o in post)),'feet':foot,'first_leg_target_jump_rad':tr[0]['target_jump_leg_max_rad'],'max_leg_target_jump_rad':max(t['target_jump_leg_max_rad'] for t in tr),'leg_velocity_rad_s':stat([x for o in post for x in o['leg_velocity_rad_s']]),'wrist_tracking':{side:{'position_m':stat([t[side]['position_error_m'] for t in tr]),'angle_rad':stat([t[side]['orientation_error_rad'] for t in tr])} for side in ('left','right')},'phase_coverage':{phase:sum(o['stage']==phase for o in post) for phase in ['standing','grasp','lift','hold','source_tail']},'sim_seconds':a['sim_seconds'],'physics_steps_in_verifier':0,'training_ready':False}
    write(out/'validation.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','verify']);p.add_argument('--contract',required=True);args=p.parse_args();print(json.dumps(run(args.contract) if args.command=='run' else verify(args.contract),indent=2))
