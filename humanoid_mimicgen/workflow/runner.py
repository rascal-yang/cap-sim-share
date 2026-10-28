"""Live native stage execution. All task limits come from a frozen strategy."""
import json
import time
from pathlib import Path
import numpy as np
from .contracts import WorkflowError, canonical, now, phase_at
from .execution import GateMonitor, base_is_stable, finite_record
from .native import NativeAdapter, validate_physics, validate_model_identity
from .planning import measured_approach, measured_retreat, measured_posture_retreat, navigate_command, validate_source, measured_navigation_path
from .calibration import equilibrium_command
from .feedback import IntegralFeedback,can_recover


def execute_native(contract):
    from .service import check_contract
    from cap_sim.backends.native_sonic import rot,quat
    from ..planned_approach import planned_pose
    from ..standing_root_reference import standing_root_reference
    c=check_contract(contract);root=Path(contract).parent;out=root/'execution';out.mkdir(exist_ok=False)
    strategy=c['workflow']['strategy'];policy=strategy['policy'];bindings=strategy['bindings'];phases=strategy['phases']
    targets={k:v.copy() for k,v in np.load(c['inputs']['targets']).items()}
    advanced=policy['navigation'].get('profile')=='replay_navigation'
    if advanced:
        from .navigation.integration import load_navigation_source
        navigation_plan=json.loads(Path(policy['navigation']['plan_file']).read_text())
        source=load_navigation_source(c['inputs']['reference_states'],navigation_plan)
    else:
        source=np.load(c['inputs']['reference_states'])['samples']
        validate_source(targets,strategy,c['workflow']['alignment'])
    initial=np.load(c['inputs']['initial'])
    entry=phases[0]['start']+strategy['command_pose_offset'];last=phases[-1]['end'];frame=phases[0]['start']
    whole_body=advanced and policy['navigation'].get('navigation_only') is False
    operation=None;operation_command=None;operation_fingers=None;native_task_success=False;navigation_handoff_ready=False
    selected=c['workflow']['selection'];route=np.asarray(selected['navigation_path']);goal=route[-1];waypoint=1
    task=json.loads(Path(c['inputs']['task']).read_text());start=now();clock=time.monotonic();adapter=None
    states=[];observations=[];commands=[];tracking=[];events=[];failure=None;error_type=None;steps=0;stage='navigation';online_planning_s=0.
    stage_started=0.;settled=0.;gate=GateMonitor(policy['gates']);phase_id=None;completed=False;holdposes=None;plan_times=plan_poses=None
    attempted=[];retry_index=0;force_tick=False;retreat_goals=None
    recovery=policy.get('recovery',{'max_retries':0});base_targets={k:v.copy() for k,v in targets.items()}
    comp=policy['compensation'];feedback=IntegralFeedback(bindings['hands'],gain_per_s=comp['gain_per_s'],cap_m=comp['cap_m']) if comp['mode']=='bounded_integral' else None
    contact_started=None;feedback_log=[];goalposes=None
    from .motion import DistanceVelocityProfile, upright_root_reference
    nav=policy['navigation'];advanced=nav.get('profile')=='replay_navigation';smooth=nav.get('profile') in ('distance_velocity','replay_navigation')
    route_controller=None
    if advanced:
        from .navigation.control import RouteController
        from .navigation.integration import validate_bound_plan
        navigation_plan=json.loads(Path(nav['plan_file']).read_text())
        validate_bound_plan(navigation_plan,c['inputs'],strategy)
        route_controller=RouteController(route,nav['route_limits'])
    velocity_profile=DistanceVelocityProfile(speed=nav['speed_m_s'],acceleration=nav['acceleration_m_s2'],response_s=nav['response_s']) if smooth else None
    nav_velocity=np.zeros(2);nav_quiet_s=0.;normal_q=None;recovery_settled=0.;posture_audit=None
    from .parking import RecentWindow,predicted_stop,correction_decision,planted_stance_reference,corrected_reference
    parking=nav.get('parking');correction=nav.get('correction');nav_cycle=0;nav_previous_error=None
    nav_elapsed=0.;nav_cycle_started=0.;parking_finished=False;nav_settle_pending=True;nav_log=[];stop_prediction=None;stance_shift_until=0.;parking_reference_xy=goal[:2].copy()
    trigger=policy.get('grasp_trigger');judge=None;triggered_retries=[];last_query_elapsed=None;grasp_clock_started=0.;source_geometry=None
    if trigger:
        from .grasp_trigger import CodexGraspJudge,geometry,query_due,trigger_targets
        judge=CodexGraspJudge(trigger)
    def write(name,value):(out/name).write_bytes(canonical(finite_record(value)))
    def begin_recovery(reason,ob):
        nonlocal retry_index,stage,stage_started,holdposes,gate,phase_id,force_tick,failure
        if not can_recover(reason,ob,retry_index,recovery['max_retries'],policy['acceptance']['support_force_N']):return False
        record={'reason':reason,'retry_index':retry_index,'time_s':float(adapter.d.time),'observation':ob,
                'feedback_bias_m':{h:v.tolist() for h,v in feedback.bias.items()} if feedback else {},
                'next_action':'open supported object, retreat through checked path, reobserve and retarget, settle and retry',
                'hypothesis':'Measured wrist bias and object motion require reanchored approach; no unobserved gain changes'}
        write('retry_'+str(retry_index)+'_reflection.json',record)
        np.savez_compressed(out/('retry_'+str(retry_index)+'_checkpoint.npz'),qpos=adapter.d.qpos.copy(),qvel=adapter.d.qvel.copy(),ctrl=adapter.d.ctrl.copy(),qacc_warmstart=adapter.d.qacc_warmstart.copy())
        events.append(dict(record,event='retry_requested'));retry_index+=1
        holdposes=adapter.poses();stage='recovery_open';stage_started=float(adapter.d.time)
        phase_id=None;gate=GateMonitor(policy['gates']);force_tick=True;failure=None
        return True
    try:
        adapter=NativeAdapter(task,strategy,initial);m,d=adapter.m,adapter.d;dt=adapter.dt;stride=adapter.control_stride
        recent=RecentWindow(policy['settle']['window_s'],dt)
        if whole_body:
            from .navigation.integration import validate_operation_inputs
            validate_operation_inputs(navigation_plan,c['inputs'],strategy,dt)
        if not np.isclose(c['workflow']['alignment']['command_dt_s'],dt):raise WorkflowError('unsupported_capability','Source command resampling is not implicit')
        physics=json.loads(Path(c['physics']['physics_parameters_manifest']).read_text())
        validate_physics(m,physics,sonic=adapter.sonic,reward_stride=adapter.b.reward_stride)
        import mujoco
        validate_model_identity(mujoco.MjModel.from_binary_path(c['inputs']['model']),m)
        write('actual_model_files.json',adapter.model_files)
        write('actual_providers.json',adapter.providers)
        np.savez_compressed(out/'controller_initial.npz',decoder_last=adapter.sonic.last,**{k:np.asarray([h[k] for h in adapter.sonic.history]) for k in adapter.sonic.history[0]})
        np.savez_compressed(out/'actual_initial.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qacc_warmstart=d.qacc_warmstart.copy())
        initial_pose=adapter.poses();initial_rot=rot(d.qpos[3:7]);rel={s:(initial_rot.inv().apply(p[:3]-d.qpos[:3]),initial_rot.inv()*rot(p[3:])) for s,p in initial_pose.items()}
        object_anchor=np.asarray(adapter.observe()['object_pose']);write('initial_observation.json',adapter.observe())
        if trigger:
            source_geometry=[{'source_frame':i,'geometry':geometry(adapter,source[i,1:1+m.nq],targets['fingers'][phases[0]['end']-1])} for i in (phases[0]['start'],(phases[0]['start']+phases[0]['end'])//2,phases[0]['end']-1)]
        maxsteps=int(policy['budgets']['simulation_s']/dt)
        while steps<maxsteps:
            tick=steps%stride==0 or force_tick;force_tick=False;profile_updated=False;velocity=np.zeros(2);face=float(goal[2]);source_index=-1
            if stage=='navigation':
                nav_limits=dict(nav)
                if nav_cycle:nav_limits['stop_radius_m']=correction['tolerance_m']
                velocity,face,arrived,waypoint=navigate_command([*d.qpos[:2],rot(d.qpos[3:7]).as_euler('xyz')[2]],route,waypoint,nav_limits)
                terminal_waypoint=np.array_equal(route[waypoint,:2],route[-1,:2])
                stop_prediction=None
                if parking and terminal_waypoint:
                    stop_prediction=predicted_stop(d.qpos[:2],d.qvel[:2],parking['model'])
                    delta=goal[:2]-d.qpos[:2]
                    arrived=bool(np.linalg.norm(goal[:2]-stop_prediction)<=parking['trigger_radius_m'] or
                                 np.linalg.norm(delta)<=parking['trigger_radius_m'] or np.dot(goal[:2]-stop_prediction,delta)<=0.)
                correction_timeout=bool(nav_cycle and float(d.time)-nav_cycle_started>=correction['max_duration_s'])
                arrived=arrived or correction_timeout
                if advanced:
                    if tick:
                        route_tick=route_controller.step([*d.qpos[:2],rot(d.qpos[3:7]).as_euler('xyz')[2]],[*d.qvel[:2],d.qvel[5]],policy['control_dt_s'])
                        nav_velocity=np.asarray(route_tick['velocity_world_xy_m_s']);velocity_profile.velocity=nav_velocity.copy()
                        nav_log.append(dict(route_tick,time_s=float(d.time)))
                    velocity=nav_velocity.copy();face=route_tick['facing_yaw_rad'];arrived=route_tick['arrived'];waypoint=route_tick['waypoint']
                if smooth and not advanced:
                    if tick:
                        profile_updated=True
                        drive_delta=route[waypoint,:2]-d.qpos[:2]
                        if nav_cycle and np.linalg.norm(drive_delta)>0:
                            # Bounded motion at the established speed ceiling;
                            # avoid indefinite low-speed gait for tiny errors.
                            drive_delta=drive_delta/np.linalg.norm(drive_delta)*max(np.linalg.norm(drive_delta),nav['speed_m_s']*nav['response_s'])
                        nav_velocity=velocity_profile.step(drive_delta,policy['control_dt_s'],measured_velocity=None if nav_cycle else d.qvel[:2])
                        slow=np.linalg.norm(d.qvel[:2])<=policy['settle']['speed_m_s'] and abs(d.qvel[5])<=policy['settle']['yaw_speed_rad_s']
                        quiet=arrived and slow and np.linalg.norm(nav_velocity)<=policy['settle']['speed_m_s']
                        nav_quiet_s=nav_quiet_s+policy['control_dt_s'] if quiet else 0.
                    velocity=nav_velocity.copy()
                    # Enter blended standing on geometric arrival; walking at tiny
                    # speeds does not itself guarantee a stable idle gait.
                if tick:nav_log.append({'time_s':float(d.time),'navigation_cycle':nav_cycle,'actual_xy':d.qpos[:2].tolist(),
                    'velocity_xy_m_s':d.qvel[:2].tolist(),'command_xy_m_s':velocity.tolist(),'predicted_stop_xy':stop_prediction.tolist() if stop_prediction is not None else None,
                    'goal_xy':goal[:2].tolist(),'correction_timeout':correction_timeout})
                if arrived and tick:
                    prehandoff_velocity=velocity.copy()
                    adapter.sonic.navigating=False
                    adapter.sonic.anchor=np.r_[d.qpos[:7],(initial['qpos'][adapter.b.qids] if advanced else source[entry,1+adapter.b.qids])].copy()
                    adapter.sonic.anchor[:7]=standing_root_reference(d.qpos[:7],(initial['qpos'][:7] if advanced else source[entry,1:8]))
                    if nav.get('standing_orientation')=='upright':adapter.sonic.anchor[:7]=upright_root_reference(d.qpos[:7])
                    if smooth:adapter.sonic.blend_to_standing(adapter.sonic.anchor,nav['response_s'])
                    else:adapter.sonic.ref=None
                    holdposes=adapter.poses();stage='settle';stage_started=float(d.time);velocity[:]=0
                    events.append({'event':'navigation_stop_requested','time_s':float(d.time),'actual_root':d.qpos[:7].tolist(),'goal_xyyaw':goal.tolist(),'measured_velocity_xy_m_s':d.qvel[:2].tolist(),'command_velocity_xy_m_s':prehandoff_velocity.tolist(),'quiet_duration_s':nav_quiet_s,'navigation_cycle':nav_cycle,'predicted_stop_xy':stop_prediction.tolist() if stop_prediction is not None else None,'correction_timeout':correction_timeout})
                elif nav_elapsed>=policy['budgets']['navigation_s']:raise WorkflowError('navigation_timeout')
            if smooth and stage!='navigation':
                if tick and not profile_updated:nav_velocity=velocity_profile.step([0.,0.],policy['control_dt_s'])
                velocity=nav_velocity.copy()
            if stage=='whole_body_operation':
                from .navigation.whole_body import set_body_reference
                measured=adapter.observe();rows=recent.rows
                measured['recent_root_drift_m']=None if not rows or float(d.time)-rows[0]['time_s']+1e-12<policy['settle']['window_s']-dt else max(float(np.linalg.norm(np.array(x['root_pose'][:2])-rows[0]['root_pose'][:2])) for x in rows)
                operation_command=operation.command(dt,base_is_stable(measured,policy['settle']),policy_tick=tick)
                if operation_command['completed']:
                    completed=True;frame=last;break
                source_index=operation.applied_frame if operation.applied_frame is not None else -1
                if source_index>=0:frame=source_index
                if tick:set_body_reference(adapter.sonic,operation_command['qpos'])
            contact_stage=stage=='contact'
            phase=phase_at(phases,frame) if contact_stage else None
            if contact_stage:
                source_index=frame
                if phase_id!=phase['id']:
                    ob=adapter.observe()
                    if phase['requires_contact'] and (gate.contact_s<policy['gates']['contact_entry_s'] or ob['contacts']['hand_object_N']<=policy['gates']['contact_force_N']):
                        if begin_recovery('contact_entry_failed',ob):continue
                        raise WorkflowError('contact_entry_failed',phase['id'])
                    if phase['requires_clearance'] and ob['clearance_m']<policy['acceptance']['clearance_m']:
                        if begin_recovery('clearance_entry_failed',ob):continue
                        raise WorkflowError('clearance_entry_failed',phase['id'])
                    phase_id=phase['id'];events.append({'event':'phase_started','phase':phase_id,'time_s':float(d.time),'source_frame':frame})
            if stage=='whole_body_operation' and source_index>=0:phase=phase_at(phases,source_index)
            label=phase_id if contact_stage else stage
            if tick:
                if stage=='whole_body_operation':goalposes=operation_command['poses']
                elif stage=='navigation':
                    heading=rot(d.qpos[3:7]);goalposes={s:np.r_[d.qpos[:3]+heading.apply(pos),quat(heading*ori)] for s,(pos,ori) in rel.items()}
                elif stage in ('settle','recovery_open'):goalposes={s:p.copy() for s,p in holdposes.items()}
                elif stage=='grasp_probe':
                    probe_frame=min(phases[0]['end']-1,phases[0]['start']+int((float(d.time)-stage_started)/dt))
                    goalposes={s:targets[s+'_pose'][probe_frame+strategy['command_pose_offset']].copy() for s in ('left','right','torso')}
                elif stage in ('approach','recovery_retreat'):goalposes={s:planned_pose(plan_times,plan_poses[s],float(d.time)-stage_started) for s in ('left','right','torso')}
                else:goalposes={s:targets[s+'_pose'][frame+strategy['command_pose_offset']].copy() for s in ('left','right','torso')}
                desired={s:p.copy() for s,p in goalposes.items()}
                comp=policy['compensation']
                if stage in ('approach','grasp_probe','contact') and comp['mode']=='identified_equilibrium':
                    for hand in bindings['hands']:goalposes[hand][:3]=equilibrium_command(desired[hand][:3],comp['calibrations'][hand],comp['scope'])
                poses=adapter.poses()
                if feedback and stage in ('approach','grasp_probe','contact'):
                    for hand in bindings['hands']:
                        bias=feedback.update(hand,desired[hand][:3],poses[hand][:3],policy['control_dt_s'])
                        goalposes[hand][:3]+=bias
                        feedback_log.append({'time_s':float(d.time),'retry_index':retry_index,'hand':hand,'bias_m':bias.tolist(),'error_m':(desired[hand][:3]-poses[hand][:3]).tolist()})
                row={'time_s':float(d.time),'stage':label,'source_index':source_index,'retry_index':retry_index,'mode':1}
                for side in ('left','right'):
                    p=desired[side];row[side]={'target_pose':p.tolist(),'commanded_pose':goalposes[side].tolist(),'actual_pose':poses[side].tolist(),
                        'position_error_m':float(np.linalg.norm(poses[side][:3]-p[:3])),
                        'orientation_error_rad':float((rot(p[3:]).inv()*rot(poses[side][3:])).magnitude())}
                tracking.append(row)
            fingers=np.asarray(nav['fingers_rad']) if advanced else targets['fingers'][frame if contact_stage else entry]
            height=float(initial['qpos'][2] if advanced else source[entry,3])
            if stage=='whole_body_operation':
                if tick:operation_fingers=operation_command['fingers'].copy()
                fingers=operation_fingers;height=float(operation_command['qpos'][2])
            adapter.command(goalposes,velocity,face,height,fingers,policy_tick=tick)
            attempted.append({'time_s':float(d.time),'stage':label,'source_index':source_index,'requested_source_index':operation.frame if stage=='whole_body_operation' else source_index,'retry_index':retry_index,'mode':1,'velocity_world_xy_m_s':velocity.tolist(),'locomotion_planner_active':bool(adapter.sonic.navigating),'fingers_rad':fingers.tolist(),'body_target_rad':adapter.last_target.tolist(),'ctrl_Nm':d.ctrl.tolist()})
            adapter.step();steps+=1;ob=adapter.observe();ob.update(stage=label,source_poststep_frame=source_index,retry_index=retry_index)
            states.append(np.r_[d.time,d.qpos,d.qvel,d.ctrl]);observations.append(ob);recent.append(ob);commands.append(attempted[-1])
            if whole_body and steps%stride==0:
                native_task_success=adapter.native_success() or native_task_success
                ob['native_task_success']=native_task_success
            if label=='navigation':nav_elapsed+=dt
            limits=policy['gates'];contacts=ob['contacts']
            if whole_body and contacts['robot_object_N']>policy['whole_body_operation']['object_force_stop_N']:
                raise WorkflowError('operation_object_force_limit')
            failure=gate.update(dt,hand_force=contacts['robot_object_N'] if stage=='whole_body_operation' else contacts['hand_object_N'],obstacle_force=contacts['robot_obstacle_N'],requires_contact=bool(phase and phase['requires_contact']))
            tilt=float(np.arccos(np.clip(rot(d.qpos[3:7]).as_matrix()[2,2],-1,1)))
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise WorkflowError('nonfinite_state')
            if d.qpos[2]<limits['fall_height_m'] or tilt>limits['fall_tilt_rad']:raise WorkflowError('fall')
            if any(w.number for w in d.warning):raise WorkflowError('simulation_warning')
            if failure:
                if begin_recovery(failure,ob):continue
                break
            if not contact_stage and not stage.startswith('recovery_') and not (stage=='whole_body_operation' and source_index>=0):
                op=np.asarray(ob['object_pose'])
                if np.linalg.norm(op[:3]-object_anchor[:3])>limits['precontact_translation_m'] or (rot(object_anchor[3:]).inv()*rot(op[3:])).magnitude()>limits['precontact_rotation_rad']:
                    raise WorkflowError('object_moved_before_grasp')
                if contacts['robot_object_N']>limits['contact_force_N']:raise WorkflowError('premature_object_contact')
            if stage=='settle':
                elapsed=float(d.time)-stage_started;window=policy['settle']['window_s']
                rows=recent.rows
                drift=None if elapsed<window or not rows else max(float(np.linalg.norm(np.array(x['root_pose'][:2])-rows[0]['root_pose'][:2])) for x in rows)
                ob['recent_root_drift_m']=drift;stable=base_is_stable(ob,policy['settle']);settled=settled+dt if stable and float(d.time)>=stance_shift_until else 0.
                ob.update(stable_sample=stable,continuous_stable_s=settled)
                if elapsed>=policy['settle']['minimum_s'] and steps%stride==0 and settled+1e-12>=policy['settle']['continuous_s']:
                    actual=adapter.poses()
                    if nav_settle_pending:
                        error=float(np.linalg.norm(d.qpos[:2]-goal[:2]))
                        events.append({'event':'navigation_settled','time_s':float(d.time),'navigation_cycle':nav_cycle,'actual_root':d.qpos[:7].tolist(),'error_m':error})
                        nav_settle_pending=False
                    if correction and not parking_finished:
                        decision=correction_decision(error,nav_previous_error,nav_cycle,correction)
                        events.append({'event':'parking_decision','time_s':float(d.time),'navigation_cycle':nav_cycle,'error_m':error,'previous_error_m':nav_previous_error,'decision':decision})
                        if decision=='correct' and correction.get('method','walk')=='planted_feet':
                            reference_xy=corrected_reference(parking_reference_xy,goal[:2],d.qpos[:2]) if nav_cycle and correction.get('residual_feedback',False) else parking_reference_xy.copy()
                            pstart=time.monotonic();target_q,shift_audit=planted_stance_reference(m,d.qpos.copy(),np.r_[reference_xy,goal[2]],bindings,policy,ob['contacts']['events']);online_planning_s+=time.monotonic()-pstart
                            write('stance_correction_'+str(nav_cycle+1)+'.json',shift_audit)
                            if target_q is not None:
                                parking_reference_xy=reference_xy.copy();nav_previous_error=error;nav_cycle+=1;holdposes=actual
                                anchor=np.r_[target_q[:7],target_q[adapter.b.qids]]
                                adapter.sonic.blend_to_standing(anchor,shift_audit['duration_s'],measured_start=np.r_[d.qpos[:7],d.qpos[adapter.b.qids]])
                                stage_started=float(d.time);stance_shift_until=stage_started+shift_audit['duration_s'];settled=0.;nav_settle_pending=True;force_tick=True
                                events.append({'event':'correction_started','method':'planted_feet','time_s':float(d.time),'navigation_cycle':nav_cycle,'error_m':error,'duration_s':shift_audit['duration_s'],'reference_root_xy':parking_reference_xy.tolist(),'evaluation_goal_xy':goal[:2].tolist()})
                                continue
                            events.append({'event':'correction_skipped','time_s':float(d.time),'reason':'stationary_geometry_or_support_check_failed'})
                            parking_finished=True
                        if decision=='correct' and correction.get('method','walk')=='walk' and nav_elapsed<policy['budgets']['navigation_s']:
                            pstart=time.monotonic();new_route,route_audit=measured_navigation_path(m,d.qpos.copy(),goal,policy);online_planning_s+=time.monotonic()-pstart
                            write('correction_route_'+str(nav_cycle+1)+'.json',{'audit':route_audit,'path':new_route.tolist() if new_route is not None else None,'actual_root':d.qpos[:7].tolist(),'physics_steps':0})
                            if new_route is not None:
                                nav_previous_error=error;nav_cycle+=1;route=np.asarray(new_route);waypoint=1
                                heading=rot(d.qpos[3:7]);rel={h:(heading.inv().apply(p[:3]-d.qpos[:3]),heading.inv()*rot(p[3:])) for h,p in actual.items()}
                                adapter.sonic.navigating=True;adapter.sonic.ref=None;adapter.sonic.handoff_until_tick=-1
                                # Reset only command generator state at rest, never robot/controller history.
                                velocity_profile=DistanceVelocityProfile(speed=nav['speed_m_s'],acceleration=nav['acceleration_m_s2'],response_s=nav['response_s'])
                                nav_velocity=np.zeros(2);nav_quiet_s=0.;settled=0.;stage='navigation';nav_cycle_started=float(d.time);nav_settle_pending=True;force_tick=True
                                events.append({'event':'correction_started','time_s':float(d.time),'navigation_cycle':nav_cycle,'error_m':error})
                                continue
                            events.append({'event':'correction_skipped','time_s':float(d.time),'reason':'no_collision_checked_route'})
                        parking_finished=True
                    if advanced:
                        from .navigation.integration import measured_handoff
                        pstart=time.monotonic();handoff_qpos=d.qpos.copy();handoff_time=float(d.time)
                        handoff=measured_handoff(navigation_plan,handoff_qpos,output_dir=out/'measured_operation_reference')
                        online_planning_s+=time.monotonic()-pstart
                        if navigation_plan.get('reference_mode')=='apple_lateral_v1':
                            handoff['simulation_time_advanced_s']=float(d.time)-handoff_time
                            if not np.array_equal(d.qpos,handoff_qpos) or float(d.time)!=handoff_time:
                                raise WorkflowError('context_changed','Simulation advanced during bounded planning pause')
                        write('measured_navigation_handoff.json',handoff)
                        if not handoff['feasible']:raise WorkflowError('handoff_not_ready',handoff.get('reason'))
                        navigation_handoff_ready=True
                        events.append({'event':'navigation_handoff_ready','time_s':float(d.time),'operation_executed':False})
                        if not whole_body:
                            stage='handoff';completed=True;break
                        from .navigation.whole_body import WholeBodyOperation
                        pstart=time.monotonic()
                        operation=WholeBodyOperation(adapter,navigation_plan,strategy,source,targets['fingers'],
                            generated_reference=handoff.get('generated_reference'))
                        online_planning_s+=time.monotonic()-pstart
                        stage='whole_body_operation';force_tick=True
                        events.append({'event':'whole_body_operation_started','time_s':float(d.time),'physical_state_reset':False,'source_start':operation.start})
                        continue
                    if retreat_goals is None:
                        retreat_goals={h:p.copy() for h,p in actual.items()};normal_q=d.qpos.copy()
                        np.save(out/'normal_posture_qpos.npy',normal_q)
                    goals={s:(targets[s+'_pose'][entry] if s in bindings['hands'] else actual[s]) for s in ('left','right')}
                    pstart=time.monotonic();plan_times,plan_poses,pa=measured_approach(m,d.qpos.copy(),goals,policy);online_planning_s+=time.monotonic()-pstart
                    write('planning_audit_'+str(retry_index)+'.json',dict(pa,planning_wall_s=online_planning_s));np.savez_compressed(out/('planned_approach_'+str(retry_index)+'.npz'),time_s=plan_times,**{k+'_pose':v for k,v in plan_poses.items()})
                    for s in ('left','right'):
                        if s not in bindings['hands']:targets[s+'_pose'][:]=actual[s]
                    targets['torso_pose'][:]=plan_poses['torso'][-1];np.savez_compressed(out/'adapted_execution_targets.npz',**targets)
                    stage='approach';stage_started=float(d.time);grasp_clock_started=stage_started;last_query_elapsed=None;events.append({'event':'measured_stance_plan','time_s':float(d.time),'actual_root':d.qpos[:7].tolist()})
                elif elapsed>=policy['budgets']['settle_s']:raise WorkflowError('settling_timeout')
            elif stage in ('approach','grasp_probe'):
                elapsed=float(d.time)-stage_started
                window=policy['settle']['window_s']
                rows=recent.rows
                ob['recent_root_drift_m']=None if elapsed<window else max(float(np.linalg.norm(np.array(x['root_pose'][:2])-rows[0]['root_pose'][:2])) for x in rows)
                stable=base_is_stable(ob,policy['settle']);settled=settled+dt if stable and float(d.time)>=stance_shift_until else 0.
                ob.update(stable_sample=stable,continuous_stable_s=settled)
                if trigger and steps%stride==0:
                    judge_elapsed=float(d.time)-grasp_clock_started
                    if query_due(stage,judge_elapsed,float(plan_times[-1]),last_query_elapsed,judge.calls,trigger):
                        last_query_elapsed=judge_elapsed
                        evidence={'measured':ob,'geometry':geometry(adapter,d.qpos.copy(),targets['fingers'][phases[0]['end']-1]),
                                  'source_reference':source_geometry,'execution':{'stage':stage,'approach_elapsed_s':judge_elapsed,'planned_approach_s':float(plan_times[-1]),'query_index':judge.calls,'query_budget':trigger['max_calls'],'fingers_held_open':True,'source_access':True,'simulation_paused':True}}
                        decision=judge.decide(evidence,out/'grasp_decisions'/str(judge.calls))
                        events.append({'event':'grasp_judgment','time_s':float(d.time),'retry_index':retry_index,'query_index':judge.calls-1,'decision':decision})
                        if decision['action']=='abort':raise WorkflowError('gpt_grasp_aborted',decision['reason'])
                        if decision['action']=='close_now':
                            np.savez_compressed(out/('gpt_pretrigger_targets_'+str(retry_index)+'.npz'),**targets)
                            actual=adapter.poses();targets=trigger_targets(targets,actual,bindings['hands'],start=phases[0]['start'],lift=phases[0]['end'],end=last,offset=strategy['command_pose_offset'])
                            np.savez_compressed(out/('gpt_trigger_targets_'+str(retry_index)+'.npz'),**targets)
                            triggered_retries.append(retry_index);frame=phases[0]['start'];phase_id=None;stage='contact';contact_started=float(d.time);force_tick=True
                            events.append({'event':'contact_started','retry_index':retry_index,'time_s':contact_started,'actual_root':d.qpos[:7].tolist(),'base_stable':stable,'wrist_pose_gate':False,'trigger':'gpt_privileged','skipped_approach_remaining_s':max(0.,float(plan_times[-1])-judge_elapsed),'wrist_hold':{k:v.tolist() for k,v in actual.items()},'original_contact_wrist_trajectory_replaced':True})
                            continue
                        if judge.calls>=trigger['max_calls']:raise WorkflowError('grasp_decision_budget_exhausted')
                if stage=='approach' and elapsed>=plan_times[-1] and settled+1e-12>=policy['settle']['continuous_s'] and steps%stride==0:
                    if trigger:
                        stage='grasp_probe';stage_started=float(d.time);events.append({'event':'grasp_probe_started','time_s':float(d.time),'fingers':'open','retry_index':retry_index})
                    else:
                        stage='contact';contact_started=float(d.time);events.append({'event':'contact_started','retry_index':retry_index,'time_s':contact_started,'actual_root':d.qpos[:7].tolist(),'base_stable':True,'wrist_pose_gate':False})
                elif float(d.time)-grasp_clock_started>=policy['budgets']['approach_s']:raise WorkflowError('approach_timeout')
            elif stage=='recovery_open':
                elapsed=float(d.time)-stage_started
                if contacts['support_N']<=policy['acceptance']['support_force_N']:raise WorkflowError('recovery_support_lost')
                if elapsed>=recovery['release_min_s'] and contacts['hand_object_N']<=limits['contact_force_N'] and steps%stride==0:
                    pstart=time.monotonic()
                    if recovery.get('posture_restore',False):
                        plan_times,plan_poses,pa=measured_posture_retreat(m,d.qpos.copy(),normal_q,policy)
                    else:plan_times,plan_poses,pa=measured_retreat(m,d.qpos.copy(),retreat_goals,policy)
                    online_planning_s+=time.monotonic()-pstart;posture_audit=pa;recovery_settled=0.
                    np.savez_compressed(out/('planned_retreat_'+str(retry_index)+'.npz'),time_s=plan_times,**{k+'_pose':v for k,v in plan_poses.items()})
                    write('retreat_plan_'+str(retry_index)+'.json',pa)
                    stage='recovery_retreat';stage_started=float(d.time)
                    events.append({'event':'retreat_started','retry_index':retry_index,'time_s':float(d.time)})
                elif elapsed>=recovery['release_timeout_s']:raise WorkflowError('recovery_release_timeout')
            elif stage=='recovery_retreat':
                elapsed=float(d.time)-stage_started
                if contacts['support_N']<=policy['acceptance']['support_force_N']:raise WorkflowError('recovery_support_lost')
                restored=True
                if recovery.get('posture_restore',False):
                    actual=adapter.poses();addresses=np.asarray(posture_audit['upper_qpos_addresses'],int)
                    errors={'upper_joint_max_rad':float(np.max(np.abs(d.qpos[addresses]-normal_q[addresses]))),
                            'wrist_max_m':max(float(np.linalg.norm(actual[h][:3]-plan_poses[h][-1,:3])) for h in ('left','right')),
                            'pose_max_rad':max(float((rot(plan_poses[h][-1,3:]).inv()*rot(actual[h][3:])).magnitude()) for h in ('left','right','torso'))}
                    restored=errors['upper_joint_max_rad']<=recovery['joint_tolerance_rad'] and errors['wrist_max_m']<=recovery['position_tolerance_m'] and errors['pose_max_rad']<=recovery['angle_tolerance_rad']
                    recovery_settled=recovery_settled+dt if restored and elapsed>=plan_times[-1] else 0.
                    ob['recovery_posture_errors']=errors;ob['recovery_converged_s']=recovery_settled
                    restored=recovery_settled+1e-12>=policy['settle']['continuous_s']
                if elapsed>=plan_times[-1] and restored and steps%stride==0:
                    from ..skill_planning import adapt_poses
                    current_object=np.asarray(ob['object_pose'])
                    for hand in bindings['hands']:targets[hand+'_pose']=adapt_poses(base_targets[hand+'_pose'],object_anchor,current_object)
                    base_targets={k:v.copy() for k,v in targets.items()};object_anchor=current_object.copy()
                    frame=phases[0]['start'];phase_id=None;gate=GateMonitor(policy['gates'])
                    holdposes=adapter.poses();settled=0.;stage='settle';stage_started=float(d.time)
                    events.append({'event':'retry_reanchored','retry_index':retry_index,'time_s':float(d.time),'object_pose':current_object.tolist()})
                elif elapsed>=recovery['retreat_timeout_s']:raise WorkflowError('recovery_retreat_timeout')
            elif contact_stage:
                frame+=1
                if frame==last:completed=True;break
        if not completed and failure is None:failure='simulation_budget_exhausted'
    except Exception as exc:
        failure=exc.code if isinstance(exc,WorkflowError) else 'execution_error';error_type=type(exc).__name__
        write('error.json',{'reason_code':failure,'error_type':error_type,'detail':exc.detail if isinstance(exc,WorkflowError) else str(exc)})
    finally:
        np.savez_compressed(out/'states.npz',samples=np.asarray(states))
        for name,value in [('observations',observations),('commands',commands),('attempted_commands',attempted),('tracking',tracking),('feedback',feedback_log),('navigation',nav_log),('events',events),('contacts',[x['contacts'] for x in observations])]:write(name+'.json',value)
        if operation:
            write('whole_body_operation.json',{'commands':operation.audit,'gates':operation.clock.log})
        sonic_record=None
        if adapter:
            from .sonic_record import save_action_bundle
            try:sonic_record=save_action_bundle(out,adapter.sonic.records,commands,dt,policy['control_dt_s'],expected_physics_steps=steps)
            except WorkflowError as exc:
                sonic_record=dict(coverage_complete=False,reason=exc.code,detail=exc.detail);completed=False;failure=failure or exc.code
            write('controller.json',adapter.sonic.records);write('planner.json',adapter.sonic.planner_records)
        audit={'experiment_id':root.name,'sonic_action_record':sonic_record,'started_at':start,'ended_at':now(),'simulation_wall_s':time.monotonic()-clock,
               'online_planning_wall_s':online_planning_s,'sim_seconds':float(adapter.d.time) if adapter else 0.,
               'navigation_corrections':nav_cycle,'navigation_sim_s':nav_elapsed,'api_calls_during_execution':judge.calls if judge else 0,'model_judgment_wall_s':judge.wall_s if judge else 0.,'simulation_compute_wall_s':time.monotonic()-clock-(judge.wall_s if judge else 0.),'gpt_triggered_retries':triggered_retries,'recovery_count':retry_index,'completed':completed,'failure':failure,'error_type':error_type,'physics_steps':steps,'recorded_steps':len(states),'attempted_commands':len(attempted),'completed_commands':len(commands),
               'acceptance_scope':'whole_body_operation_native' if whole_body else 'navigation_handoff_only' if advanced else 'manipulation',
               'native_task_success':native_task_success if whole_body else None,'navigation_handoff_ready':navigation_handoff_ready,
               'actual_encoder_modes':sorted(set(x['mode'] for x in commands)),'original_token_prefix_count':0,'mode0_calls':0,
               'midrun_state_resets':0,'fallback':False,'contact_started_at_s':contact_started,'stage':stage,'source_end_exclusive':frame}
        write('execution_audit.json',audit)
        if adapter:
            np.savez_compressed(out/'terminal.npz',qpos=adapter.d.qpos.copy(),qvel=adapter.d.qvel.copy(),ctrl=adapter.d.ctrl.copy(),qacc_warmstart=adapter.d.qacc_warmstart.copy())
            adapter.close()
    return audit
