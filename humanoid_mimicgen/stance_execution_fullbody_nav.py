"""Frozen navigation-to-contact pilot: no original motion-token/mode0 prefix.

Navigation: live SONIC encoder0 fullbody tracking + locomotion planner velocity/facing. At arrival,
explicitly switch to source leg posture anchored at the achieved root, settle,
then replan arms from measured state. Never set physical state after reset.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from cap_sim.backends.native_sonic import Sonic,rot,quat
from .standing_handoff import StandingSonic,extended_measurement,stable_sample
from .navigation_reference import body_future_from_planner
from .stance_selection import navigation_command,yaw_of
from .planned_approach import plan_from_measured,planned_pose
from .action_replay import sha,write,now,initialize_source
from .privileged import no_physics


class NavigationSonic(StandingSonic):
    navigating=True
    def plan_legs(self,d,velocity,height,facing_yaw=None):
        if self.navigating:
            Sonic.plan_legs(self,d,velocity,height,facing_yaw)
            self.planner_records.append({'time':float(d.time),'type':'locomotion','actual_planner_mode':1 if np.linalg.norm(velocity)>1e-6 else 4,'velocity_world_m_s':np.asarray(velocity).tolist(),'facing_yaw_rad':float(facing_yaw),'reference':self.ref.tolist()})
        else:super().plan_legs(d,velocity,height,facing_yaw)


def check_contract(path):
    path=Path(path);r=json.loads(path.read_text())
    if sha(path)!=(path.parent/'experiment.sha256').read_text().strip():raise ValueError('Contract not frozen')
    if r['record_status']!='frozen_ready' or not r['preflight']['passed']:raise ValueError('Not ready')
    if r['control']['prefix']!='none_replanned_fullbody_navigation' or r['control']['fallback_allowed']:raise ValueError('Wrong interface')
    hashes=r['preflight']['evidence_sha256']
    for p in [str(Path(__file__).resolve()),*map(r['inputs'].get,['model','initial','targets','task','reference_states','reference_evidence','stance_selection'])]:
        if p not in hashes:raise ValueError('Missing binding '+str(p))
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Changed evidence '+p)
    runtime=json.loads(Path(r['runtime']['actual_loaded_session_evidence']).read_text())
    if not runtime['passed'] or not runtime['sessions_loaded'] or len(runtime['actual_session_providers'])!=3 or any(x!=['CPUExecutionProvider'] for x in runtime['actual_session_providers'].values()):raise ValueError('Runtime mismatch')
    if r['budget']['attempts_per_case']!=1:raise ValueError('Invalid budget')
    return r


def run(contract):
    import mujoco
    from cap_sim.backends.humano import Backend
    from humano.core.action import ActionCmd
    from humano.mt.base import MotorCommand
    r=check_contract(contract);root=Path(contract).parent;out=root/'execution';out.mkdir(exist_ok=False)
    selected=json.loads(Path(r['inputs']['stance_selection']).read_text())['selected']
    if selected is None:raise ValueError('No preflight stance')
    goal=np.asarray(selected['xyyaw']);task=json.loads(Path(r['inputs']['task']).read_text());b=Backend()
    with no_physics():b.describe(task)
    m,d=b.m,b.d
    for key,v in json.loads(Path(r['physics']['physics_parameters_manifest']).read_text())['arrays'].items():
        if not np.array_equal(getattr(m,key),v):raise ValueError('Physics changed '+key)
    sonic=NavigationSonic(m,b.robot,task['resources']['sonic_models'],b.initial,case='anchored_legs')
    providers={k:getattr(sonic,k).get_providers() for k in ('encoder','decoder','planner')};write(out/'actual_providers.json',providers)
    if any(v!=['CPUExecutionProvider'] for v in providers.values()):raise ValueError('Providers')
    targets={k:v.copy() for k,v in np.load(r['inputs']['targets']).items()};source=np.load(r['inputs']['reference_states'])['samples']
    initialize_source(m,d,np.load(r['inputs']['initial']));np.savez_compressed(out/'actual_initial.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qacc_warmstart=d.qacc_warmstart.copy())
    initialroot=d.qpos[:7].copy();initial_rot=rot(initialroot[3:]);rel={}
    for side,bid in [('left',b.left),('right',b.right),('torso',b.torso)]:rel[side]=(initial_rot.inv().apply(d.xpos[bid]-initialroot[:3]),initial_rot.inv()*rot(d.xquat[bid]))
    ob0=extended_measurement(b);object_anchor=np.asarray(ob0['object_pose']);source_index=-1;stage='navigation';steps=0;si=1600;phase_start=0;settled=0.;bad=lost=handtime=0.;failure=exception=None;plan_times=plan_poses=None;anchor=None;holdposes=None;contact_started=None;arrival=None;last_target=None;goalposes=None
    states=[];obs=[];commands=[];tracking=[];events=[];start=now();clock=time.perf_counter()
    try:
        while steps<round(r['budget']['simulation_seconds_per_attempt']/.005):
            velocity=np.zeros(2);face=float(goal[2]);source_index=-1
            if stage=='navigation':
                velocity,face,reached=navigation_command([*d.qpos[:2],yaw_of(d.qpos[3:7])],goal)
                if reached and steps%4==0:
                    arrival=float(d.time);anchor=d.qpos[:7].copy();sonic.navigating=False;sonic.anchor=np.r_[d.qpos[:7],source[1599,1+b.qids]].copy();sonic.ref=None
                    holdposes={s:np.r_[d.xpos[bid],d.xquat[bid]].copy() for s,bid in [('left',b.left),('right',b.right),('torso',b.torso)]}
                    stage='prestanding';phase_start=steps;velocity[:]=0
                    events.append({'event':'navigation_arrived','time_s':float(d.time),'goal_xyyaw':goal.tolist(),'actual_root':d.qpos[:7].tolist()})
                elif d.time>=16:failure='navigation_timeout';break
            if stage in ('grasp','lift','hold','source_tail'):
                source_index=si
                if si==1880 and not(handtime>=.12 and obs[-1]['contacts']['hand_object_N']>.1):failure='lift_contact_gate';break
                if si==2434 and not(obs[-1]['clearance_m']>=.01 and obs[-1]['contacts']['hand_object_N']>.1):failure='hold_entry_gate';break
                stage='grasp' if si<=1879 else ('lift' if si<=2433 else ('hold' if si<=2654 else 'source_tail'))
            if steps%4==0:
                if stage=='navigation':
                    heading=rot(d.qpos[3:7]);goalposes={s:np.r_[d.qpos[:3]+heading.apply(p),quat(heading*q)] for s,(p,q) in rel.items()}
                elif stage=='prestanding':goalposes=holdposes
                elif stage=='approach':goalposes={s:planned_pose(plan_times,plan_poses[s],(steps-phase_start)*.005) for s in ('left','right','torso')}
                else:
                    j=1599 if stage=='standing' else si-1;goalposes={s:targets[s+'_pose'][j] for s in ('left','right','torso')}
                wrists=[(goalposes[s][:3],goalposes[s][3:]) for s in ('left','right')];torso=goalposes['torso'];height=float(source[1599,3])
                future=body_future_from_planner(sonic,d,velocity,height,face) if stage=='navigation' else None
                target=sonic.step(d,wrists,(torso[:3],torso[3:]),velocity,height,future,face)
                expected_mode=0 if stage=='navigation' else 1
                if sonic.records[-1]['mode']!=expected_mode:raise ValueError('Encoder mode deviation')
                row={'time_s':float(d.time),'stage':stage,'mode':expected_mode,'source_index':source_index,'target_jump_leg_max_rad':0. if last_target is None else float(max(abs(target[:12]-last_target[:12]))),'velocity_world_m_s':velocity.tolist(),'facing_yaw_rad':face}
                for side,bid in [('left',b.left),('right',b.right)]:
                    p=goalposes[side];row[side]={'target_pose':p.tolist(),'actual_pose':np.r_[d.xpos[bid],d.xquat[bid]].tolist(),'position_error_m':float(np.linalg.norm(d.xpos[bid]-p[:3])),'orientation_error_rad':float((rot(p[3:]).inv()*rot(d.xquat[bid])).magnitude())}
                tracking.append(row);last_target=target.copy()
            fingers=targets['fingers'][1599 if source_index<0 else source_index]
            b.robot.apply_action(ActionCmd('wbc_torque',low_cmd=MotorCommand(q_target=target,kp=sonic.kp,kd=sonic.kd),use_sensor=False,left_hand_cmd=MotorCommand.hand_command(fingers[:7]),right_hand_cmd=MotorCommand.hand_command(fingers[7:])))
            commands.append({'time_s':float(d.time),'stage':stage,'source_index':source_index,'mode':expected_mode,'body_target_rad':target.tolist(),'fingers_rad':fingers.tolist(),'ctrl_Nm':d.ctrl.tolist()})
            mujoco.mj_step(m,d);steps+=1;mujoco.mj_forward(m,d)
            ob=extended_measurement(b);ob.update(time_s=float(d.time),stage=stage,source_poststep_frame=source_index);states.append(np.r_[d.time,d.qpos,d.qvel,d.ctrl]);obs.append(ob)
            c=ob['contacts'];bad=bad+.005 if c['robot_table_N']>5 else 0.;handtime=handtime+.005 if c['hand_object_N']>.1 else 0.;lost=lost+.005 if stage in ('lift','hold','source_tail') and c['hand_object_N']<=.1 else 0.
            tilt=float(np.arccos(np.clip(rot(d.qpos[3:7]).as_matrix()[2,2],-1,1)))
            if not np.isfinite(d.qpos).all():failure='nonfinite'
            elif d.qpos[2]<.3 or tilt>np.deg2rad(50):failure='fall'
            elif c['robot_table_N']>50 or bad>=.05-1e-10:failure='table_contact'
            elif lost>=.03-1e-10:failure='contact_lost'
            elif any(w.number for w in d.warning):failure='simulation_warning'
            if failure:break
            if stage in ('navigation','prestanding','approach','standing'):
                op=np.asarray(ob['object_pose']);err=float(np.linalg.norm(op[:3]-object_anchor[:3]));ang=float((rot(object_anchor[3:]).inv()*rot(op[3:])).magnitude())
                if err>.005 or ang>.1:failure='object_moved_before_grasp';break
                if c['hand_object_N']>.1:failure='premature_object_contact';break
            if stage in ('prestanding','standing'):
                desired=holdposes if stage=='prestanding' else {s:targets[s+'_pose'][1599] for s in ('left','right')}
                errs=[np.linalg.norm(d.xpos[bid]-desired[s][:3]) for s,bid in [('left',b.left),('right',b.right)]];angles=[(rot(desired[s][3:]).inv()*rot(d.xquat[bid])).magnitude() for s,bid in [('left',b.left),('right',b.right)]]
                stable=stable_sample(np.linalg.norm(d.qpos[:2]-anchor[:2]),np.linalg.norm(d.qvel[:2]),abs(d.qvel[5]),max(errs),max(angles),ob['foot_ground_N']);settled=settled+.005 if stable else 0.
                ob.update(stable_sample=stable,continuous_stable_s=settled)
                if steps-phase_start>=200 and steps%4==0 and settled>=.3-1e-10:
                    if stage=='prestanding':
                        # Inactive left wrist and torso stay at achieved posture. The
                        # right contact target remains object-relative world pose.
                        for s,bid in [('left',b.left),('torso',b.torso)]:targets[s+'_pose'][:]=np.r_[d.xpos[bid],d.xquat[bid]]
                        np.savez_compressed(out/'adapted_execution_targets.npz',**targets)
                        plan_times,plan_poses=plan_from_measured(m,d.qpos.copy(),targets,out);stage='approach'
                        events.append({'event':'measured_stance_plan','time_s':float(d.time),'actual_root':d.qpos[:7].tolist()})
                    else:stage='grasp';si=1600;contact_started=float(d.time);events.append({'event':'contact_started','time_s':contact_started})
                    phase_start=steps;settled=0.
                elif steps-phase_start>=600:failure=stage+'_timeout';break
            elif stage=='approach':
                if (steps-phase_start)*.005>=plan_times[-1] and steps%4==0:stage='standing';phase_start=steps;settled=0.
            elif stage in ('grasp','lift','hold','source_tail'):
                if si==2679:break
                si+=1
        if failure is None and (contact_started is None or si!=2679):failure='budget_exhausted'
    except Exception as e:failure='executor_exception';exception=repr(e)
    finally:
        np.savez_compressed(out/'states.npz',samples=np.asarray(states));write(out/'observations.json',obs);write(out/'commands.json',commands);write(out/'contacts.json',[x['contacts'] for x in obs]);write(out/'tracking.json',tracking);write(out/'events.json',events);write(out/'controller.json',sonic.records);write(out/'planner.json',sonic.planner_records)
        audit={'experiment_id':r['experiment_id'],'started_at':start,'ended_at':now(),'simulation_wall_s':time.perf_counter()-clock,'sim_seconds':float(d.time),'physics_steps':steps,'recorded_steps':len(states),'missing_measurement_steps':steps-len(states),'failure':failure,'exception':exception,'completed':failure is None and si==2679,'selected_stance':goal.tolist(),'navigation_arrived_s':arrival,'actual_final_root':d.qpos[:7].tolist(),'contact_started_at_s':contact_started,'actual_encoder_modes':sorted(set(x['mode'] for x in sonic.records)),'planner_inference_calls':sonic.serial,'encoder_calls':len(sonic.records),'decoder_calls':len(sonic.records),'original_token_prefix_count':0,'mode0_calls':sum(x['mode']==0 for x in sonic.records),'midrun_state_resets':0,'fallback':False,'contract_sha256':sha(contract)}
        write(out/'execution_audit.json',audit);b.close()
    return audit


def verify(contract):
    from cap_sim.backends.humano import Backend
    r=check_contract(contract);out=Path(contract).parent/'execution';a=json.loads((out/'execution_audit.json').read_text());obs=json.loads((out/'observations.json').read_text());tr=json.loads((out/'tracking.json').read_text());b=Backend()
    try:
        with no_physics():b.describe(json.loads(Path(r['inputs']['task']).read_text()));native=b.verify(out)
    finally:b.close()
    stages={}
    for stage in dict.fromkeys(o['stage'] for o in obs):
        rows=[o for o in obs if o['stage']==stage];tracking=[t for t in tr if t['stage']==stage]
        stages[stage]={'steps':len(rows),'start_poststep_s':rows[0]['time_s'],'end_s':rows[-1]['time_s'],'max_table_N':max(o['contacts']['robot_table_N'] for o in rows),'table_contact_s':sum(o['contacts']['robot_table_N']>.1 for o in rows)*.005,'max_clearance_m':max(o['clearance_m'] for o in rows),'right_tracking_rmse_m':float(np.sqrt(np.mean([t['right']['position_error_m']**2 for t in tracking]))) if tracking else None}
    held=[o for o in obs if o['stage'] in ('lift','hold','source_tail') and o['clearance_m']>=.01 and o['contacts']['hand_object_N']>.1 and o['contacts']['support_N']<.1]
    result={'experiment_id':r['experiment_id'],'native':native,'completed':a['completed'],'failure':a['failure'],'grasp_and_lift_observed':len(held)>=40,'lifted_contact_duration_s':len(held)*.005,'max_clearance_m':max((o['clearance_m'] for o in obs),default=None),'max_table_N':max((o['contacts']['robot_table_N'] for o in obs),default=None),'stages':stages,'physics_steps_in_verifier':0,'training_ready':False}
    write(out/'validation.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','verify']);p.add_argument('--contract',required=True);args=p.parse_args();print(json.dumps(run(args.contract) if args.command=='run' else verify(args.contract),indent=2))
