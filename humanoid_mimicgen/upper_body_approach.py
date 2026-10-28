"""Measured-state arms, then arms+waist IK/RRT; root and legs fixed in planning only."""
import time
import numpy as np
from cap_sim.backends.native_sonic import rot
from .action_replay import write
from .privileged import no_physics
from .planned_approach import timed_joint_reference

def plan_from_measured(model,measured_qpos,targets,out):
    import mujoco
    from .stationary_planning import StationaryPlanner
    planner=StationaryPlanner(model);q=np.asarray(measured_qpos).copy()
    goals={side:targets[side+'_pose'][1599] for side in ('left','right')}
    start=time.perf_counter()
    with no_physics():
        solved=planner.solve(q,goals,seed=17)
        write(out/'measured_start_plan.json',{'measured_qpos':q.tolist(),'goals':{k:v.tolist() for k,v in goals.items()},'ik':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in solved.items()}})
        if not solved['success']:raise ValueError('No stationary IK solution from actual stance')
        path,info=planner.connect(q,solved['qpos'],solved['joint_ids'],seed=17,resolution=.02)
        if path is None:raise ValueError('No collision-free connection from actual stance: '+str(info))
        times,points=timed_joint_reference(path)
        poses={side:[] for side in ('left','right','torso')};torso_id=model.body('torso_link').id
        for point in points:
            if planner.collisions(point):raise ValueError('Timed reference has collision')
            wrist=planner.wrist_poses(point)
            for side in ('left','right'):poses[side].append(wrist[side])
            poses['torso'].append(np.r_[planner.d.xpos[torso_id],planner.d.xquat[torso_id]].copy())
        poses={k:np.asarray(v) for k,v in poses.items()}
        # Global slow-down preserves every waypoint and joint-space shape.
        factor=1.
        for side in ('left','right'):
            factor=max(factor,float(np.max(np.linalg.norm(np.diff(poses[side][:,:3],axis=0),axis=1)/np.diff(times))/.05))
            angles=(rot(poses[side][1:,3:])*rot(poses[side][:-1,3:]).inv()).magnitude()
            factor=max(factor,float(max(angles/np.diff(times))/.4))
        times*=factor
        if times[-1]>20:raise ValueError('Approach exceeds20s budget')
        np.savez_compressed(out/'planned_approach.npz',time_s=times,qpos=points,**{k+'_pose':v for k,v in poses.items()})
        write(out/'planning_audit.json',{'method':info,'group':solved['group'],'duration_s':float(times[-1]),'samples':len(times),'planning_wall_s':time.perf_counter()-start,'physics_steps':0,'joint_speed_limit_rad_s':.35,'wrist_speed_limit_m_s':.05,'wrist_angular_speed_limit_rad_s':.4,'qpos_is_reference_only':True,'root_legs_unchanged':bool(np.array_equal(points[:,:19],np.repeat(q[None,:19],len(points),axis=0)))})
    return times,poses

