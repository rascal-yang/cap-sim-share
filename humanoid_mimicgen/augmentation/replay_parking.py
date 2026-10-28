"""Measured parking and recovery references, without task-admission vetoes."""
import copy
import numpy as np

from ..workflow.motion import DistanceVelocityProfile, upright_root_reference
from ..workflow.parking import correction_decision, corrected_reference, planted_stance_reference


class ReplayParking:
    def __init__(self,adapter,cfg,route,entry):
        self.a=adapter;self.cfg=cfg;self.route=np.asarray(route);self.entry=entry
        self.o=cfg['navigation']['parking'];self.goal=self.route[-1];self.waypoint=1
        self.endpoint=len(self.route)-1
        while self.endpoint>1 and np.array_equal(self.route[self.endpoint,:2],self.route[self.endpoint-1,:2]):self.endpoint-=1
        self.profile=DistanceVelocityProfile(speed=cfg['navigation']['route_limits']['speed_m_s'],acceleration=self.o['acceleration_m_s2'],response_s=self.o['response_s'])
        self.policy=copy.deepcopy(cfg['strategy']['policy'])
        self.policy['navigation'].update(response_s=self.o['response_s'],standing_orientation='upright',correction=dict(tolerance_m=self.o['tolerance_m'],max_corrections=self.o['max_corrections'],minimum_progress_m=.003,max_duration_s=4.))
        self.policy['budgets']['settle_s']=self.o['settle_s']
        self.elapsed=0.;self.started=None;self.quiet=0.;self.corrections=0;self.previous_error=None
        self.reference_xy=self.goal[:2].copy();self.events=[];self.plans=[];self.done=False;self.hold_poses=None;self.normal_q=None

    def navigation_command(self,dt):
        self.elapsed+=dt;d=self.a.d
        while self.waypoint<self.endpoint and np.linalg.norm(d.qpos[:2]-self.route[self.waypoint,:2])<.05:self.waypoint+=1
        distance=float(np.linalg.norm(d.qpos[:2]-self.goal[:2]))
        reached=distance<=self.o['arrival_radius_m'] or self.elapsed>=self.o.get('navigation_s',16.)
        velocity=self.profile.step(self.route[self.waypoint,:2]-d.qpos[:2],dt,measured_velocity=d.qvel[:2])
        return dict(velocity_world_xy_m_s=velocity.tolist(),facing_yaw_rad=float(self.goal[2]),distance_m=distance,waypoint=self.waypoint,arrived=reached,
                    handoff_reason='arrival' if distance<=self.o['arrival_radius_m'] else 'finite_navigation_window' if reached else None)

    def begin(self):
        d=self.a.d;sonic=self.a.sonic
        anchor=np.r_[upright_root_reference(d.qpos[:7]),self.entry[sonic.qids]]
        sonic.blend_to_standing(anchor,self.o['response_s'])
        self.started=float(d.time);self.quiet=0.;self.hold_poses=self.a.poses()
        self.events.append(dict(event='navigation_stop_requested',time_s=float(d.time),root_pose=d.qpos[:7].tolist(),velocity_xy_m_s=d.qvel[:2].tolist()))

    def error(self):return float(np.linalg.norm(self.a.d.qpos[:2]-self.goal[:2]))

    def correct(self):
        error=self.error();decision=correction_decision(error,self.previous_error,self.corrections,self.policy['navigation']['correction'])
        self.events.append(dict(event='parking_decision',time_s=float(self.a.d.time),error_m=error,decision=decision,corrections=self.corrections))
        if decision!='correct':return False
        xy=corrected_reference(self.reference_xy,self.goal[:2],self.a.d.qpos[:2]) if self.corrections else self.reference_xy.copy()
        step=xy-self.a.d.qpos[:2];length=float(np.linalg.norm(step));maximum=self.o.get('max_translation_m',.03)
        if length>maximum:xy=self.a.d.qpos[:2]+step*(maximum/length)
        q,audit=planted_stance_reference(self.a.m,self.a.d.qpos.copy(),np.r_[xy,self.goal[2]],self.cfg['strategy']['bindings'],self.policy,self.a.observe()['contacts']['events'])
        self.plans.append(audit)
        if q is None:
            self.events.append(dict(event='correction_skipped',time_s=float(self.a.d.time),reason='reference_not_feasible',error_m=error));return False
        self.reference_xy=xy;self.previous_error=error;self.corrections+=1;d=self.a.d;sonic=self.a.sonic
        sonic.blend_to_standing(np.r_[q[:7],q[sonic.qids]],audit['duration_s'],measured_start=np.r_[d.qpos[:7],d.qpos[sonic.qids]])
        self.started=float(d.time);self.quiet=0.;self.hold_poses=self.a.poses()
        self.events.append(dict(event='correction_started',time_s=float(d.time),error_m=error,corrections=self.corrections,duration_s=audit['duration_s']))
        return True


    def update(self,dt):
        if self.done:return True
        d=self.a.d;ob=self.a.observe();s=self.policy['settle'];elapsed=float(d.time)-self.started
        stable=np.linalg.norm(d.qvel[:2])<=s['speed_m_s'] and abs(d.qvel[5])<=s['yaw_speed_rad_s'] and min(ob['foot_ground_N'])>=s['foot_force_N']
        blend_done=self.a.sonic.tick>=self.a.sonic.handoff_until_tick if hasattr(self.a.sonic,'handoff_until_tick') else elapsed>=self.o['response_s']
        self.quiet=self.quiet+dt if stable and blend_done else 0.
        settled=elapsed>=s['minimum_s'] and self.quiet>=s['continuous_s']-1e-9
        if not settled and elapsed<self.o['settle_s']:return False
        if self.correct():return False
        self.done=True;self.normal_q=d.qpos.copy()
        self.events.append(dict(event='parking_complete',time_s=float(d.time),error_m=self.error(),root_pose=d.qpos[:7].tolist(),velocity_xy_m_s=d.qvel[:2].tolist(),settled=bool(settled),corrections=self.corrections))
        return True


def retry_translation(initial_object,current_object):
    return np.asarray(current_object[:3])-np.asarray(initial_object[:3])


def retry_reachable(model,current,right_pose,policy):
    from ..stationary_planning import StationaryPlanner
    options=policy['planning'];planner=StationaryPlanner(model,penetration_tolerance_m=options['penetration_tolerance_m'])
    fit=planner.solve(current,{'right':right_pose},**options['ik'])
    return fit['success'],dict(reachable=fit['success'],attempts=fit['attempts'],physics_steps=0,reference_only=True)


class ReplayRecovery:
    def __init__(self,model,current,normal,open_fingers,options):
        import mujoco
        from ..workflow.motion import recovery_configuration
        self.model=model;self.initial=current.copy();self.o=options;self.fingers=open_fingers.copy()
        self.addresses=np.array([model.jnt_qposadr[j] for j in range(model.njnt) if any(part in (model.joint(j).name or '') for part in ('waist','shoulder','elbow','wrist'))],dtype=int)
        self.target=recovery_configuration(current,normal,self.addresses)
        self.tangent=np.zeros(model.nv);mujoco.mj_differentiatePos(model,self.tangent,1.,current,self.target)
        self.duration=max(options['restore_s'],1.875*float(np.max(abs(self.target[self.addresses]-current[self.addresses]),initial=0))/options['joint_speed_rad_s'])
        self.quiet=0.;self.restored=False;self.error=None

    def sample(self,elapsed,measured):
        import mujoco
        t=max(0.,elapsed-self.o['release_s']);u=min(1.,t/self.duration);q=self.initial.copy()
        mujoco.mj_integratePos(self.model,q,self.tangent,float(10*u**3-15*u**4+6*u**5))
        self.error=float(np.max(abs(measured[self.addresses]-self.target[self.addresses]),initial=0))
        self.quiet=self.quiet+float(self.model.opt.timestep) if u>=1 and self.error<=self.o['joint_tolerance_rad'] else 0.
        self.restored=self.quiet>=.3-1e-9
        return dict(stage='recovery_open' if elapsed<self.o['release_s'] else 'recovery_restore',qpos=q,fingers=self.fingers,source_row=-1,
                    completed=bool(self.restored or t>=self.duration+self.o['settle_s']))

