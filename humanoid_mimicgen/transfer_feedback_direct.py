"""Paired 0/+2cm contact translation probe with a measured-start approach bridge.

All cases use an explicitly declared closed-loop original-token/mode0 prefix. State writes after initialization are
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

from .standing_handoff import StandingSonic,stable_sample,extended_measurement,body_future


def direct_grasp_due(stage,steps):
    return stage=='approach' and steps>=1800


def translate_contact(source,delta):
    delta=np.asarray(delta,float)
    if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>.02000001:raise ValueError('Only <=2cm translation approved')
    result={k:np.asarray(v).copy() for k,v in source.items()}
    result['right_pose'][:,:3]+=delta
    return result


def approach_pose(start,end,u):
    u=float(np.clip(u,0,1));alpha=10*u**3-15*u**4+6*u**5
    return np.r_[np.asarray(start)[:3]+alpha*(np.asarray(end)[:3]-np.asarray(start)[:3]),quat(Slerp([0,1],rot(np.stack([start[3:],end[3:]])))(alpha))]


def decode_prefix_token(sonic,d,token):
    sonic.history.append({'omega':d.qvel[3:6].copy(),'q':(d.qpos[sonic.qids]-sonic.default)[sonic.order],
        'dq':d.qvel[sonic.vids][sonic.order].copy(),'last':sonic.last.copy(),'gravity':rot(d.qpos[3:7]).inv().apply([0,0,-1])})
    observation=np.concatenate([np.asarray(token)]+[np.concatenate([h[key] for h in sonic.history]) for key in ('omega','q','dq','last','gravity')]).astype(np.float32)
    raw=sonic.decoder.run(None,{sonic.decoder.get_inputs()[0].name:observation[None]})[0][0]
    if not np.isfinite(raw).all():raise ValueError('Nonfinite prefix network output')
    sonic.last=raw.copy();sonic.tick+=1;target=sonic.default+sonic.scale*raw[sonic.reverse]
    sonic.records.append({'time':float(d.time),'mode':'original_token_prefix','token':np.asarray(token).tolist(),'target':target.tolist()})
    return target


def check_contract(path):
    path=Path(path);r=json.loads(path.read_text())
    if sha(path)!=(path.parent/'experiment.sha256').read_text().strip():raise ValueError('Not frozen')
    if r['record_status']!='frozen_ready' or not r['preflight']['passed']:raise ValueError('Not ready')
    if r['control']['case']!='anchored_legs' or r['control']['fallback_allowed'] or r['budget']['attempts_per_case']!=1:raise ValueError('Invalid case')
    if r['control'].get('prefix')!='live_original_tokens_then_mode0':raise ValueError('Wrong prefix interface')
    if r['physics']['acceptance_thresholds']['standing_gate'].get('enabled') is not False:raise ValueError('Direct-grasp bypass must be declared')
    hashes=r['preflight']['evidence_sha256']
    if str(Path(__file__).resolve()) not in hashes:raise ValueError('Unbound executor')
    for key in ['model','initial','controls','reference_states','reference_evidence','targets','task','source_controller','program','transfer_manifest']:
        if r['inputs'][key] not in hashes:raise ValueError('Unbound '+key)
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Evidence changed '+p)
    runtime=json.loads(Path(r['runtime']['actual_loaded_session_evidence']).read_text())
    if not runtime['passed'] or not runtime['sessions_loaded'] or len(runtime['actual_session_providers'])!=3:raise ValueError('Missing sessions')
    if any(x!=['CPUExecutionProvider'] for x in runtime['actual_session_providers'].values()):raise ValueError('Wrong providers')
    return r


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
    source=np.load(r['inputs']['reference_states'])['samples'];sonic.anchor=np.r_[source[1599,1:8],source[1599,1+b.qids]].copy();ctrl=np.load(r['inputs']['controls'])['ctrl'];targets=np.load(r['inputs']['targets'])
    old=json.loads(Path(r['inputs']['source_controller']).read_text());program=json.loads(Path(r['inputs']['program']).read_text())
    initialize_source(m,d,np.load(r['inputs']['initial']))
    np.savez_compressed(out/'actual_initial.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qacc_warmstart=d.qacc_warmstart.copy())
    states=[];obs=[];commands=[];tracking=[];events=[];failure=None;exception=None;steps=0;stage='prefix';si=1599;contact_started=None;settled=0.;bad=lost=handtime=0.;longest_stable=0.;anchor=None;last_target=None;approach_start=None;transfer=json.loads(Path(r['inputs']['transfer_manifest']).read_text());object_anchor=np.asarray(transfer['target_object_anchor_pose'])
    start=now();clock=time.perf_counter();limit=round(r['budget']['simulation_seconds_per_attempt']/.005)
    try:
        while steps<limit:
            if steps<1600:
                if steps%4==0:
                    if steps<1000:target=decode_prefix_token(sonic,d,old[steps//4]['token'])
                    else:
                        target=sonic.step(d,[],(None,None),np.zeros(2),.8,body_future(program,steps*.005),None)
                        if sonic.records[-1]['mode']!=0:raise ValueError('Prefix must use encoder0 at5..8s')
                    last_target=target.copy()
                fingers=targets['fingers'][steps]
                b.robot.apply_action(ActionCmd('wbc_torque',low_cmd=MotorCommand(q_target=target,kp=sonic.kp,kd=sonic.kd),use_sensor=False,left_hand_cmd=MotorCommand.hand_command(fingers[:7]),right_hand_cmd=MotorCommand.hand_command(fingers[7:])))
                source_index=steps
                commands.append({'time_s':float(d.time),'stage':'prefix','source_index':source_index,'mode':'original_token_prefix' if steps<1000 else 0,'body_target_rad':target.tolist(),'fingers_rad':fingers.tolist(),'ctrl_Nm':d.ctrl.tolist()})
            else:
                if steps==1600:
                    qerr=float(abs(d.qpos[:50]-source[1599,1:51]).max());verr=float(abs(d.qvel[:49]-source[1599,58:107]).max())
                    rooterr=float(np.linalg.norm(d.qpos[:3]-source[1599,1:4]))
                    if qerr>.05 or verr>.2 or rooterr>.01:raise ValueError('Robot prefix arrival mismatch')
                    anchor=source[1599,1:8].copy();stage='approach'
                    approach_start={side:np.r_[d.xpos[bid],d.xquat[bid]].copy() for side,bid in [('left',b.left),('right',b.right)]}
                    events.append({'event':'handoff','time_s':float(d.time),'root_arrival_error_m':rooterr,'robot_qpos_component_error_max':qerr,'robot_qvel_component_error_max':verr,'root_anchor':anchor.tolist(),'measured_wrist_starts':{k:v.tolist() for k,v in approach_start.items()},'fixed_leg_reference':sonic.anchor.tolist()})
                if stage in ('approach','standing'):source_index=1599;j=1599
                else:source_index=si;j=si-1
                if stage not in ('approach','standing'):
                    if si==1880:
                        ok=handtime>=.12 and obs[-1]['contacts']['hand_object_N']>.1;events.append({'event':'lift_gate','time_s':float(d.time),'passed':ok})
                        if not ok:failure='lift_contact_gate';break
                    if si==2434:
                        ok=obs[-1]['clearance_m']>=.03 and obs[-1]['contacts']['hand_object_N']>.1;events.append({'event':'hold_gate','time_s':float(d.time),'passed':ok})
                        if not ok:failure='hold_entry_gate';break
                    stage='grasp' if si<=1879 else ('lift' if si<=2433 else ('hold' if si<=2654 else 'source_tail'))
                if steps%4==0:
                    poses=[approach_pose(approach_start[side],targets[side+'_pose'][1599],(steps-1600)/200) if stage=='approach' else targets[side+'_pose'][j] for side in ('left','right')]
                    wrists=[(p[:3],p[3:]) for p in poses]
                    torso=targets['torso_pose'][j];rp=targets['root_pose'][j]
                    bf=None
                    target=sonic.step(d,wrists,(torso[:3],torso[3:]),np.zeros(2),float(rp[2]),bf,float(rot(rp[3:]).as_euler('xyz')[2]))
                    expected=1
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
            if stage=='prefix' and c['hand_object_N']>.1:failure='unexpected_prefix_object_contact';break
            if stage in ('approach','standing'):
                object_pose=np.asarray(ob['object_pose']);displacement=float(np.linalg.norm(object_pose[:3]-object_anchor[:3]));angle=float((rot(object_anchor[3:]).inv()*rot(object_pose[3:])).magnitude())
                ob.update(object_anchor_displacement_m=displacement,object_anchor_rotation_rad=angle)
                if displacement>.005 or angle>.1:failure='object_moved_before_grasp';break
            if direct_grasp_due(stage,steps):
                contact_started=float(d.time)
                events.append({'event':'direct_grasp_start','time_s':contact_started,'standing_gate_bypassed':True,'root_pose':d.qpos[:7].tolist()})
                stage='grasp';si=1600
            elif stage!='approach' and steps>1600:
                if si==2679:break
                si+=1
        if not failure and (contact_started is None or si<2679):failure='budget_exhausted'
    except Exception as e:failure='executor_exception';exception=repr(e)
    finally:
        np.savez_compressed(out/'states.npz',samples=np.asarray(states));write(out/'observations.json',obs);write(out/'contacts.json',[o['contacts'] for o in obs]);write(out/'commands.json',commands);write(out/'tracking.json',tracking);write(out/'events.json',events);write(out/'controller.json',sonic.records);write(out/'planner.json',sonic.planner_records)
        audit={'experiment_id':r['experiment_id'],'case':case,'object_translation_m':transfer['translation_m'],'standing_gate_bypassed':True,'approach_duration_s':1.,'fixed_leg_reference_is_common_source':True,'started_at':start,'ended_at':now(),'contract_sha256':sha(contract),'physics_steps':steps,'recorded_steps':len(states),'missing_measurement_steps':steps-len(states),'failure':failure,'exception':exception,'completed':failure is None and si==2679,'contact_started_at_s':contact_started,'longest_stable_s':longest_stable,'actual_encoder_modes':sorted(set(str(x['mode']) for x in sonic.records)),'actual_contact_modes':sorted(set(x['mode'] for x in tracking)),'prefix_interface':'live_original_tokens_then_mode0','planner_inference_calls':sonic.serial,'encoder_calls':sum(x['mode']!='original_token_prefix' for x in sonic.records),'decoder_calls':len(sonic.records),'online_ik_calls':0,'midrun_state_resets':0,'simulation_wall_s':time.perf_counter()-clock,'sim_seconds':float(d.time),'fallback':False}
        write(out/'execution_audit.json',audit);b.close()
    return audit


def verify(contract):
    from cap_sim.backends.humano import Backend
    r=check_contract(contract);root=Path(contract).parent;out=root/'execution';a=json.loads((out/'execution_audit.json').read_text());ob=json.loads((out/'observations.json').read_text());tr=json.loads((out/'tracking.json').read_text());s=np.load(out/'states.npz')['samples']
    b=Backend()
    try:
        with no_physics():b.describe(json.loads(Path(r['inputs']['task']).read_text()));native=b.verify(out)
    finally:b.close()
    if len(ob)<=1600:
        result={'experiment_id':r['experiment_id'],'case':r['control']['case'],'standing_pass':False,'success':False,'failure':a['failure'],'native':native,'stable_hold_s':0.,'hold_min_clearance_m':None,'max_clearance_m':max((o['clearance_m'] for o in ob),default=None),'max_hand_N':max((o['contacts']['hand_object_N'] for o in ob),default=0.),'max_table_N':max((o['contacts']['robot_table_N'] for o in ob),default=0.),'phase_coverage':{k:0 for k in ['approach','standing','grasp','lift','hold']},'post_prefix_metrics':None,'sim_seconds':a['sim_seconds'],'physics_steps_in_verifier':0,'training_ready':False}
        write(out/'validation.json',result);return result
    post=ob[1600:];anchor=np.asarray(ob[1599]['root_pose']);times=np.array([o['time_s'] for o in ob]);p=np.array([o['object_pose'] for o in ob]);wr=np.array([o['right_pose'] for o in ob]);rel=rot(wr[:,3:]).inv().apply(p[:,:3]-wr[:,:3]);rq=(rot(wr[:,3:]).inv()*rot(p[:,3:])).as_quat();velocity=np.gradient(p[:,2],times)
    indices=[i for i,o in enumerate(ob) if o['stage'] not in ('approach','standing') and 2433<=o['source_poststep_frame']<=2654];longest=runlength=0;holdmin=None
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
    result={'experiment_id':r['experiment_id'],'case':r['control']['case'],'standing_pass':a['contact_started_at_s'] is not None,'success':bool(a['completed'] and native['success'] and hold>=1 and max(o['contacts']['robot_table_N'] for o in ob)<=5),'failure':a['failure'],'native':native,'stable_hold_s':hold,'hold_min_clearance_m':holdmin,'max_clearance_m':max(o['clearance_m'] for o in post),'max_hand_N':max(o['contacts']['hand_object_N'] for o in post),'max_table_N':max(o['contacts']['robot_table_N'] for o in post),'table_contact_gt_0p1_s':sum(o['contacts']['robot_table_N']>.1 for o in post)*.005,'max_root_xy_drift_m':float(max(np.linalg.norm(np.asarray(o['root_pose'][:2])-anchor[:2]) for o in post)),'feet':foot,'first_leg_target_jump_rad':tr[0]['target_jump_leg_max_rad'],'max_leg_target_jump_rad':max(t['target_jump_leg_max_rad'] for t in tr),'leg_velocity_rad_s':stat([x for o in post for x in o['leg_velocity_rad_s']]),'wrist_tracking':{side:{'position_m':stat([t[side]['position_error_m'] for t in tr]),'angle_rad':stat([t[side]['orientation_error_rad'] for t in tr])} for side in ('left','right')},'phase_coverage':{phase:sum(o['stage']==phase for o in post) for phase in ['approach','standing','grasp','lift','hold','source_tail']},'sim_seconds':a['sim_seconds'],'physics_steps_in_verifier':0,'training_ready':False}
    write(out/'validation.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','verify']);p.add_argument('--contract',required=True);args=p.parse_args();print(json.dumps(run(args.contract) if args.command=='run' else verify(args.contract),indent=2))
