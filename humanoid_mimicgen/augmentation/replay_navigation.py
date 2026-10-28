"""Walk to the replay stance, blend from measured posture, replay, hold.

Task success uses the native predicate. Tracking/contact errors are measurements,
not vetoes. Physical state and decoder outputs are never overridden.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import mujoco

from . import direct_grasp as direct
from ..workflow.contracts import file_hash, now
from ..workflow.navigation.archive import verify_evidence
from ..workflow.navigation.whole_body import BodyReference, set_body_reference
from ..workflow.sonic_record import save_action_bundle


def navigation_command(xy, goal, speed):
    delta=np.asarray(goal[:2])-xy
    distance=float(np.linalg.norm(delta))
    return dict(velocity_world_xy_m_s=(delta*speed/distance).tolist() if distance else [0.,0.],
                facing_yaw_rad=float(goal[2]),distance_m=distance)


def compensate_pose(desired, measured, position_gain, orientation_gain):
    target=desired.copy()
    target[:3]+=position_gain*(desired[:3]-measured[:3])
    error=direct.rot(desired[3:])*direct.rot(measured[3:]).inv()
    correction=direct.Rotation.from_rotvec(orientation_gain*error.as_rotvec())
    target[3:]=np.roll((correction*direct.rot(desired[3:])).as_quat(),1)
    return target


def measured_support_poses(poses,reference,measured):
    targets={side:p.copy() for side,p in poses.items()}
    rotation=direct.rot(measured[3:7])*direct.rot(reference[3:7]).inv()
    for side in ('left','torso'):
        targets[side][:3]=measured[:3]+rotation.apply(poses[side][:3]-reference[:3])
        targets[side][3:]=np.roll((rotation*direct.rot(poses[side][3:])).as_quat(),1)
    return targets


def replay_body_enabled(navigation,stage):
    return not navigation.get('parking') or (navigation['parking'].get('leg_mode')=='source_replay' and stage in ('transition','operation','hold'))


class ReplayProgram:
    def __init__(self, model, initial, reference, fingers, source_rows, dt, transition_s, hold_s):
        self.model=model
        self.initial=initial.copy()
        self.reference=reference
        self.fingers=fingers
        self.rows=source_rows
        self.dt=dt
        self.transition_s=transition_s
        self.hold_s=hold_s
        self.terminal_sent_at=None
        self.tangent=np.zeros(model.nv)
        mujoco.mj_differentiatePos(model,self.tangent,1.,initial,reference[0])

    def sample(self, elapsed, *, policy_tick=True):
        if elapsed < self.transition_s:
            u=elapsed/self.transition_s
            q=self.initial.copy()
            weight=float(10*u**3-15*u**4+6*u**5)
            mujoco.mj_integratePos(self.model,q,self.tangent,weight)
            return dict(stage='transition',qpos=q,fingers=self.fingers[0],source_row=-1,completed=False,translation_weight=weight)
        replay_time=elapsed-self.transition_s
        index=min(len(self.reference)-1,int((replay_time+1e-8)/self.dt))
        holding=replay_time>=len(self.reference)*self.dt-1e-9
        if index==len(self.reference)-1 and policy_tick and self.terminal_sent_at is None:self.terminal_sent_at=elapsed
        return dict(stage='hold' if holding else 'operation',qpos=self.reference[index],
                    fingers=self.fingers[index],source_row=int(self.rows[index]),
                    completed=self.terminal_sent_at is not None and elapsed-self.terminal_sent_at>=self.hold_s-1e-9)


def run(case, prepared_adapter=None):
    case=Path(case)
    contract=json.loads((case/'experiment.json').read_text())
    assert file_hash(case/'experiment.json')==(case/'experiment.sha256').read_text().strip()
    verify_evidence(contract['preflight']['evidence_sha256'])
    direct.save(case/'execution/started.json',dict(started_at=now()))
    a,cfg=prepared_adapter if prepared_adapter is not None else direct.initialize(case)
    model,data=a.m,a.d
    nav=cfg['navigation']
    plan=json.loads((case/'offline_plan.json').read_text())
    route=np.asarray(plan['selected']['navigation_path'])
    goal=route[-1]
    parking=None
    if nav.get('parking'):
        from .replay_parking import ReplayParking
    initial_height=float(data.qpos[2])
    with np.load(case/'reference.npz') as z:
        reference=z['qpos'].copy();fingers=z['fingers'].copy();rows=z['source_rows'].copy()
    if nav.get('parking'):parking=ReplayParking(a,cfg,route,reference[0])
    fk=BodyReference(model,{**cfg['strategy']['bindings']['wrists'],'torso':cfg['strategy']['bindings']['torso']},[])
    poses=a.poses()
    desired={s:p.copy() for s,p in poses.items()}
    heading=direct.rot(data.qpos[3:7])
    relative={side:(heading.inv().apply(p[:3]-data.qpos[:3]),heading.inv()*direct.rot(p[3:])) for side,p in poses.items()}
    program=None;program_start=None;stage='navigation';native=False;completed=False;failure=None;navigation_done=False
    states=[];nexts=[];commands=[];ctrls=[];observations=[];events=[];navigation=[]
    velocity=np.zeros(2);face=float(goal[2]);latched_fingers=fingers[0].copy();source_row=-1
    retry=nav.get('retry',{'max_retries':0});retry_index=0;attempts=[];attempt_native=False
    recovery=None;recovery_start=None;normal_q=None;translation=np.zeros(3)
    object_anchor=np.asarray(a.observe()['object_pose']) if retry['max_retries'] else None
    source_desired={s:p.copy() for s,p in desired.items()}
    integral=None;integral_bias=np.zeros(3)
    if nav.get('wrist_integral'):
        from ..workflow.feedback import IntegralFeedback
        integral=IntegralFeedback(['right'],**nav['wrist_integral'])
    checkpoint=nav.get('checkpoint',{});branch=None;start_step=0
    if not checkpoint.get('enabled',True):checkpoint={}
    if checkpoint:
        from . import navigation_checkpoint as checkpoints
        if checkpoint.get('resume'):
            branch=checkpoints.restore(case,a,cfg,parking,integral)
            start_step=branch['step'];saved=branch['loop'];prefix=branch['prefix']
            initial_height,poses,desired,source_desired,program_start,stage,native,navigation_done,velocity,face,latched_fingers,source_row,retry_index,attempts,attempt_native,recovery_start,normal_q,translation,object_anchor,integral_bias,events=(saved[k] for k in checkpoints.LOOP_FIELDS)
            states,nexts,ctrls,commands,observations,navigation=(prefix[k] for k in ('states','nexts','ctrls','commands','observations','navigation'))
    started=now();clock=time.monotonic();out=case/'execution'
    try:
        for step in range(start_step,round(cfg['simulation_budget_s']/a.dt)):
            if checkpoint and step==0:checkpoints.capture(case,'navigation_start',a,locals())
            tick=step%a.control_stride==0
            if stage=='navigation' and tick:
                distance=float(np.linalg.norm(data.qpos[:2]-goal[:2]))
                cmd=parking.navigation_command(cfg['strategy']['policy']['control_dt_s']) if parking is not None else navigation_command(data.qpos[:2],goal,nav['route_limits']['speed_m_s'])
                arrived=cmd['arrived'] if parking is not None else distance<=nav['route_limits']['stop_radius_m']
                if arrived:
                    velocity[:]=0
                    if parking is not None:
                        parking.begin();stage='parking';poses=parking.hold_poses;desired={s:p.copy() for s,p in poses.items()}
                    else:
                        program=ReplayProgram(model,data.qpos.copy(),reference,fingers,rows,a.dt,nav['transition_s'],cfg['hold_s'])
                        program_start=float(data.time);stage='transition'
                        events.append(dict(event='navigation_arrived',time_s=float(data.time),distance_m=distance,root_pose=data.qpos[:7].tolist()))
                else:
                    velocity=np.asarray(cmd['velocity_world_xy_m_s']);face=cmd['facing_yaw_rad']
                    navigation.append(dict(cmd,time_s=float(data.time)))
                    orientation=direct.rot(data.qpos[3:7])
                    poses={s:np.r_[data.qpos[:3]+orientation.apply(p),np.roll((orientation*r).as_quat(),1)] for s,(p,r) in relative.items()}
                    desired={s:p.copy() for s,p in poses.items()}
            if stage=='parking':
                if tick and parking.update(cfg['strategy']['policy']['control_dt_s']):
                    if checkpoint:checkpoints.capture(case,'parked',a,locals())
                    program=ReplayProgram(model,data.qpos.copy(),reference,fingers,rows,a.dt,nav['transition_s'],cfg['hold_s'])
                    program_start=float(data.time);stage='transition'
                    events.append(dict(event='navigation_arrived',time_s=float(data.time),distance_m=parking.error(),root_pose=data.qpos[:7].tolist(),measured_after_parking=True))
                poses=parking.hold_poses;desired={s:p.copy() for s,p in poses.items()}
            if program is None and recovery is None:source_desired={s:p.copy() for s,p in desired.items()}
            if recovery is not None:
                sample=recovery.sample(float(data.time)-recovery_start,data.qpos)
                if sample['completed'] and tick:
                    from .replay_parking import retry_translation,retry_reachable
                    measured_object=np.asarray(a.observe()['object_pose']);translation=retry_translation(object_anchor,measured_object)
                    events.append(dict(event='recovery_complete',time_s=float(data.time),retry_index=retry_index,restored=recovery.restored,upper_joint_error_rad=recovery.error))
                    events.append(dict(event='retry_reanchored',time_s=float(data.time),retry_index=retry_index,object_pose=measured_object.tolist(),translation_world_m=translation.tolist()))
                    entry_poses,_=fk.poses(reference[0]);retry_pose=entry_poses['right'].copy();retry_pose[:3]+=translation
                    reachable,reach_audit=retry_reachable(model,data.qpos.copy(),retry_pose,cfg['strategy']['policy'])
                    events.append(dict(event='retry_reachability',time_s=float(data.time),retry_index=retry_index,**reach_audit))
                    if not reachable:failure='retry_target_unreachable';break
                    if integral is not None:integral.bias['right'][:]=0
                    recovery=None;attempt_native=False
                    program=ReplayProgram(model,data.qpos.copy(),reference,fingers,rows,a.dt,nav['transition_s'],cfg['hold_s'])
                    program_start=float(data.time);stage='transition'
            if program is not None:
                if normal_q is None:normal_q=data.qpos.copy();navigation_done=True
                sample=program.sample(float(data.time)-program_start,policy_tick=tick)
                if sample['completed']:
                    last=observations[-1]
                    from .four_task_navigation import retained as task_retained
                    retained=task_retained(cfg.get('task_family','apple'),[last])
                    attempts.append(dict(retry_index=retry_index,started_at_sim_s=program_start,ended_at_sim_s=float(data.time),native_success=attempt_native,retained_grasp=retained,completed=True,operation_started=True,terminal_stage=stage,failure=None))
                    if (attempt_native and retained) or retry_index>=retry['max_retries']:
                        completed=True;break
                    from .replay_parking import ReplayRecovery
                    retry_index+=1;program=None;recovery_start=float(data.time)
                    if parking is not None and nav['parking'].get('leg_mode')=='source_replay':
                        a.sonic.blend_to_standing(np.r_[data.qpos[:7],data.qpos[a.sonic.qids]],nav['parking']['response_s'],measured_start=np.r_[data.qpos[:7],data.qpos[a.sonic.qids]])
                    recovery=ReplayRecovery(model,data.qpos.copy(),normal_q,fingers[0],retry)
                    events.append(dict(event='retry_requested',time_s=float(data.time),retry_index=retry_index,reason='acquisition_or_retained_grasp_failed',restore_duration_s=recovery.duration))
                    sample=recovery.sample(0.,data.qpos)
            if program is not None or recovery is not None:
                if sample['stage']!=stage:
                    stage=sample['stage'];events.append(dict(event=stage+'_started',time_s=float(data.time)))
                latched_fingers=sample['fingers'].copy()
                if tick:
                    if replay_body_enabled(nav,stage):set_body_reference(a.sonic,sample['qpos'])
                    source_desired,_=fk.poses(sample['qpos']);desired={s:p.copy() for s,p in source_desired.items()}
                    if parking is not None and nav['parking'].get('support_frame')=='measured_root':
                        desired=measured_support_poses(desired,sample['qpos'],data.qpos)
                    if recovery is None:desired['right'][:3]+=translation*sample.get('translation_weight',1.)
                    desired['right'][2]+=cfg['wrist_z_offset_m']
                    poses={s:p.copy() for s,p in desired.items()}
                    feedback_sides=cfg['strategy']['bindings']['hands'] if cfg.get('task_family') else ['right']
                    for side in feedback_sides:poses[side]=compensate_pose(desired[side],a.poses()[side],cfg['feedback_gain'],nav.get('orientation_feedback_gain',0.))
                    integral_bias=np.zeros(3)
                    if integral is not None and program is not None:
                        integral_bias=integral.update('right',desired['right'][:3],a.poses()['right'][:3],cfg['strategy']['policy']['control_dt_s'])
                        poses['right'][:3]+=integral_bias
                    source_row=sample['source_row']
                    face=float(direct.rot(sample['qpos'][3:7]).as_euler('xyz')[2])
            a.command(poses,velocity,face,float(initial_height if program is None else sample['qpos'][2]),latched_fingers,policy_tick=tick)
            states.append(np.r_[data.time,data.qpos.copy(),data.qvel.copy()])
            ctrls.append(np.r_[data.time,a.last_target,data.ctrl.copy()])
            commands.append(dict(time_s=float(data.time),stage=stage,source_row=source_row,policy_tick=tick,
                                 body_target_rad=a.last_target.tolist(),ctrl_Nm=data.ctrl.tolist(),fingers=latched_fingers.tolist(),
                                 desired={s:p.tolist() for s,p in desired.items()},source_desired={s:p.tolist() for s,p in source_desired.items()},encoder_pose={s:p.tolist() for s,p in poses.items()},
                                 retry_index=retry_index,reference_translation_world_m=translation.tolist(),
                                 wrist_integral_bias_m=integral_bias.tolist(),
                                 hand_source_row=sample['source_row'] if program is not None else -1))
            a.step();nexts.append(np.r_[data.time,data.qpos.copy(),data.qvel.copy()])
            ob=a.observe()
            measured=a.poses()
            tracking={s:dict(position_m=float(np.linalg.norm(measured[s][:3]-desired[s][:3])),
                             angle_rad=float((direct.rot(desired[s][3:]).inv()*direct.rot(measured[s][3:])).magnitude())) for s in desired}
            ob.update(stage=stage,source_row=source_row,retry_index=retry_index,tracking=tracking,navigation_error_m=float(np.linalg.norm(data.qpos[:2]-goal[:2])),
                      feet={s:np.r_[data.xpos[model.body(name).id],data.xquat[model.body(name).id]].tolist() for s,name in cfg['strategy']['bindings']['feet'].items()})
            observations.append(ob)
            if (step+1)%a.control_stride==0:
                passed=a.native_success();native=passed or native
                if program is not None:attempt_native=passed or attempt_native
            if not np.isfinite(data.qpos).all():raise RuntimeError('nonfinite simulator state')
        if not completed:failure=failure or 'simulation_budget_exhausted'
    except Exception as exc:
        failure=type(exc).__name__+': '+str(exc)
    finally:
        try:
            elapsed=time.monotonic()-clock;ended=now()
            from .four_task_navigation import retained as task_retained
            terminal_retained=task_retained(cfg.get('task_family','apple'),observations)
            if nexts and (not attempts or attempts[-1]['retry_index']!=retry_index):
                attempts.append(dict(retry_index=retry_index,started_at_sim_s=program_start if program is not None else recovery_start if recovery is not None else float(states[0][0]),ended_at_sim_s=float(data.time),native_success=attempt_native if program is not None else False,retained_grasp=terminal_retained,completed=False,operation_started=program is not None,terminal_stage=stage,failure=failure))
            if parking is not None:
                events=sorted(events+parking.events,key=lambda x:x['time_s'])
                direct.save(out/'parking_plans.json',parking.plans)
            direct.save(out/'attempts.json',attempts)
            np.savez_compressed(out/'trajectory.npz',state=np.asarray(states),next_state=np.asarray(nexts),commands=np.asarray(ctrls))
            np.savez_compressed(out/'terminal.npz',qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy())
            for name,value in [('observations',observations),('commands',commands),('controller',a.sonic.records),('planner',getattr(a.sonic,'planner_records',[])),('events',events),('navigation',navigation),('actual_providers',a.providers),('actual_model_files',a.model_files)]:direct.save(out/(name+'.json'),value)
            try:
                record=save_action_bundle(out,a.sonic.records,[dict(c,fingers_rad=c['fingers']) for c in commands[:len(nexts)]],a.dt,.02,expected_physics_steps=len(nexts))
            except Exception as exc:
                record=dict(coverage_complete=False,error=str(exc))
            audit=dict(experiment_id=case.name,started_at=started,ended_at=ended,simulation_compute_wall_s=elapsed,
                       physics_steps=len(nexts),sim_seconds=len(nexts)*a.dt,completed=completed,failure=failure,stage=stage,
                       native_success=native,navigation_executed=True,navigation_arrived=navigation_done,simulator_warning_count=sum(int(w.number) for w in data.warning),
                       parking_enabled=parking is not None,parking_corrections=parking.corrections if parking is not None else 0,
                       retry_count=retry_index,attempts=attempts,
                       actual_encoder_modes=sorted({x['mode'] for x in a.sonic.records}),original_token_steps=0,mode0_calls=0,
                       midrun_state_resets=0,post_decoder_overrides=0,api_calls_during_physics=0,sonic_action_record=record,
                       contract_sha256=file_hash(case/'experiment.json'),training_ready=False)
            if checkpoint:
                audit.update(executed_physics_steps=len(nexts)-start_step,inherited_physics_steps=start_step,
                             executed_sim_seconds=(len(nexts)-start_step)*a.dt,checkpoint_restorations=int(branch is not None),
                             fully_new_rollout=branch is None,lineage=[] if branch is None else [branch['lineage']],
                             splice_state_exact=bool(not start_step or np.array_equal(states[start_step],nexts[start_step-1])))
            direct.save(out/'execution_audit.json',audit)
            result=dict(experiment_id=case.name,success=bool(native and completed),native_success=native,completed=completed,
                        failure=failure,stage=stage,navigation_executed=True,navigation_arrived=navigation_done,
                        physics_steps=len(nexts),offset_m=cfg['offset_m'],
                        retry_count=retry_index,first_attempt_native_success=attempts[0]['native_success'] if attempts else False,
                        retained_grasp=terminal_retained,
                        max_clearance_m=max((o['clearance_m'] for o in observations),default=0),
                        max_obstacle_N=max((o['contacts']['robot_obstacle_N'] for o in observations),default=0),
                        max_hand_object_N=max((o['contacts']['hand_object_N'] for o in observations),default=0),
                        obstacle_contact_gt_0p1_s=sum(o['contacts']['robot_obstacle_N']>.1 for o in observations)*a.dt,
                        final_clearance_m=observations[-1]['clearance_m'] if observations else None,audit=audit,training_ready=False)
            direct.save(case/'result.json',result)
            direct.save(case/'reflection.json',dict(observed=result,criterion='native task completion; contact/tracking are diagnostics'))
            if checkpoint:checkpoints.finalize(case)
            print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:
            a.close()
    return result


if __name__=='__main__':
    if sys.argv[1]=='run':
        run(Path(sys.argv[2]));sys.exit(0)
    root,plan,profile=map(Path,sys.argv[1:4])
    provenance=json.loads(Path(sys.argv[4]).read_text())
    navigation=json.loads(profile.read_text())
    case,a,cfg=direct.prepare(root,plan,'simple_navigation_replay',navigation.get('feedback_gain',0.),0.,provenance,keep_adapter=True,navigation=navigation)
    run(case,(a,cfg))
