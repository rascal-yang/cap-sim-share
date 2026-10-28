"""Deterministic source alignment, candidate validation and measured-state planning."""
import numpy as np
from .contracts import WorkflowError, positive


def rank_candidates(rows):
    valid=[r for r in rows if r['feasible']]
    return min(valid,key=lambda r:(r['travel_m'],r['facing_error_rad'],r['id'])) if valid else None


def navigate_command(current,path,index,limits):
    current=np.asarray(current,float);path=np.asarray(path,float)
    if current.shape!=(3,) or path.ndim!=2 or path.shape[1]!=3 or len(path)<2 or not np.isfinite(path).all():
        raise WorkflowError('invalid_input','Navigation poses')
    for key in ('speed_m_s','waypoint_radius_m','stop_radius_m'):positive(limits[key],key)
    endpoint=len(path)-1
    while endpoint>1 and np.array_equal(path[endpoint,:2],path[endpoint-1,:2]):endpoint-=1
    index=min(max(1,index),endpoint)
    while index<endpoint and np.linalg.norm(current[:2]-path[index,:2])<=limits['waypoint_radius_m']:index+=1
    delta=path[index,:2]-current[:2];dist=float(np.linalg.norm(delta));stop=index==endpoint and dist<=limits['stop_radius_m']
    return (np.zeros(2) if stop else delta*limits['speed_m_s']/dist),float(path[-1,2]),stop,index


def validate_source(targets,strategy,alignment):
    n=strategy['source_frame_count']
    if alignment.get('finger_source')!='executed_commands' or alignment.get('pose_alignment')!='explicit_offset':
        raise WorkflowError('invalid_input','Measured fingers cannot substitute source commands')
    positive(alignment.get('command_dt_s'),'source command dt')
    for side in ('left','right','torso'):
        a=np.asarray(targets[side+'_pose'])
        if a.shape!=(n,7) or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a[:,3:],axis=1),1):
            raise WorkflowError('invalid_input','Invalid source poses')
    fingers=np.asarray(targets['fingers'])
    if fingers.shape!=(n,14) or not np.isfinite(fingers).all():raise WorkflowError('unsupported_capability','G1 Dex3 requires two 7-joint command streams')


def candidate_plan(model,initial,targets,entry,bindings,policy,candidate):
    from ..stationary_planning import StationaryPlanner
    from ..stance_selection import yaw_of,yaw_quat,wrap
    from ..reachability_navigation import plan_base_path
    planner=StationaryPlanner(model,penetration_tolerance_m=policy['planning']['penetration_tolerance_m'])
    q=initial.copy();pose=np.asarray(candidate['xyyaw'],float);q[:2]=pose[:2];q[3:7]=yaw_quat(pose[2])
    goals=planner.wrist_poses(q)
    for hand in bindings['hands']:goals[hand]=targets[hand+'_pose'][entry]
    opts=policy['planning'];solved=planner.solve(q,goals,**opts['ik'])
    row={'id':candidate['id'],'xyyaw':pose.tolist(),'feasible':False,'ik_attempts':solved['attempts'],'reason':'unreachable'}
    if not solved['success']:return row
    path,info=planner.connect(q,solved['qpos'],solved['joint_ids'],**opts['rrt'])
    row['manipulation_path_audit']=info
    if path is None:row['reason']='approach_path_failed';return row
    def valid(base):
        probe=initial.copy();probe[:2]=base[:2];probe[3:7]=yaw_quat(base[2]);return not planner.collisions(probe)
    start=np.r_[initial[:2],yaw_of(initial[3:7])]
    route,info=plan_base_path(start,pose,valid,**opts['base_path'])
    row['navigation_path_audit']=info
    if route is None:row['reason']='navigation_path_failed';return row
    objid=model.body(bindings['object']).id;planner._forward(initial);obj=planner.d.xpos[objid]
    row.update(feasible=True,reason=None,navigation_path=route.tolist(),stance_reference_qpos=q.tolist(),
               entry_qpos=solved['qpos'].tolist(),travel_m=float(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1).sum()),
               facing_error_rad=abs(wrap(np.arctan2(obj[1]-pose[1],obj[0]-pose[0])-pose[2])))
    return row


def measured_approach(model,q,goals,policy):
    from ..stationary_planning import StationaryPlanner
    from ..planned_approach import timed_joint_reference
    from cap_sim.backends.native_sonic import rot
    opts=policy['planning'];planner=StationaryPlanner(model,penetration_tolerance_m=opts['penetration_tolerance_m'])
    solved=planner.solve(q,goals,**opts['ik'])
    if not solved['success']:raise WorkflowError('unreachable','Measured stance IK failed')
    path,info=planner.connect(q,solved['qpos'],solved['joint_ids'],**opts['rrt'])
    if path is None:raise WorkflowError('planning_budget_exhausted','Measured stance connection failed')
    return reference_from_path(model,planner,path,policy,{'ik_group':solved['group'],'path':info},policy['budgets']['approach_s'])


def reference_from_path(model,planner,path,policy,audit,budget):
    from ..planned_approach import timed_joint_reference
    from cap_sim.backends.native_sonic import rot
    limits=policy['approach']
    times,points=timed_joint_reference(path,joint_speed=limits['joint_speed_rad_s'],minimum_duration=limits.get('minimum_duration_s',policy['control_dt_s']),dt=policy['control_dt_s'])
    poses={hand:[] for hand in ('left','right','torso')}
    torso=model.body('torso_link').id
    for point in points:
        if planner.collisions(point):raise WorkflowError('unreachable','Interpolated path collides')
        wrists=planner.wrist_poses(point)
        for hand in ('left','right'):poses[hand].append(wrists[hand])
        poses['torso'].append(np.r_[planner.d.xpos[torso],planner.d.xquat[torso]])
    poses={k:np.array(v) for k,v in poses.items()};factor=1.
    for side in ('left','right'):
        factor=max(factor,float(np.max(np.linalg.norm(np.diff(poses[side][:,:3],axis=0),axis=1)/np.diff(times)))/limits['wrist_speed_m_s'])
        angle=(rot(poses[side][1:,3:])*rot(poses[side][:-1,3:]).inv()).magnitude()
        factor=max(factor,float(np.max(angle/np.diff(times)))/limits['wrist_speed_rad_s'])
    times*=factor
    if times[-1]>budget:raise WorkflowError('planning_budget_exhausted','Approach duration')
    return times,poses,dict(audit,duration_s=float(times[-1]),physics_steps=0,reference_only=True)


def nominal_stance(source_root,source_object,target_object):
    from ..skill_planning import adapt_poses
    from ..stance_selection import yaw_of
    pose=adapt_poses(np.asarray(source_root),np.asarray(source_object),np.asarray(target_object))
    return np.r_[pose[:2],yaw_of(pose[3:])]


def measured_retreat(model,q,measured_start_poses,policy):
    """Plan physical retreat from current q; torso is an output, not an IK hand."""
    goals={h:measured_start_poses[h].copy() for h in ('left','right')}
    return measured_approach(model,q,goals,policy)


def measured_posture_retreat(model,q,normal,policy):
    """Restore known arm AND waist posture through a collision-checked path.

    Targets are FK in the current root frame, executed by SONIC mode1, never a
    direct joint write. Measured convergence remains a separate runtime check.
    """
    from ..stationary_planning import StationaryPlanner
    from .motion import recovery_configuration
    opts=policy['planning'];planner=StationaryPlanner(model,penetration_tolerance_m=opts['penetration_tolerance_m'])
    joints=planner.arms+planner.torso;addresses=model.jnt_qposadr[joints]
    target=recovery_configuration(q,normal,addresses)
    path,info=planner.connect(q,target,joints,**opts['rrt'])
    if path is None:raise WorkflowError('planning_budget_exhausted','Normal posture connection failed')
    audit={'path':info,'target_upper_joint_rad':target[addresses].tolist(),'upper_qpos_addresses':addresses.tolist(),
           'normal_source':'first measured stable stance','root_frame':'measured at recovery planning','ik_group':None}
    return reference_from_path(model,planner,path,policy,audit,policy['recovery']['retreat_timeout_s']-policy['settle']['continuous_s'])


def measured_navigation_path(model,q,goal,policy):
    """Check a new base route using measured upper body; never edit live data."""
    from ..stationary_planning import StationaryPlanner
    from ..stance_selection import yaw_of,yaw_quat
    from ..reachability_navigation import plan_base_path
    planner=StationaryPlanner(model,penetration_tolerance_m=policy['planning']['penetration_tolerance_m'])
    def valid(pose):
        probe=q.copy();probe[:2]=pose[:2];probe[3:7]=yaw_quat(pose[2])
        return not planner.collisions(probe)
    path,audit=plan_base_path(np.r_[q[:2],yaw_of(q[3:7])],goal,valid,**policy['planning']['base_path'])
    return path,audit
