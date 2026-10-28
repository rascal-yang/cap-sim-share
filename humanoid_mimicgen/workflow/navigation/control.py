"""Deterministic measured route tracking with bounded translation and heading.

Commands are world XY m/s and absolute world yaw radians. Command slew limits
are not claims about actual braking. Unvalidated corners use a controlled slow
turn, not a shortcut outside the collision-checked polyline corridor.
"""
import numpy as np
from ..contracts import WorkflowError, positive
from .prior import finite, wrap


class RouteController:
    def __init__(self,route,limits):
        self.route=np.asarray(route,float)
        if self.route.ndim!=2 or self.route.shape[1]!=3 or len(self.route)<2 or not np.isfinite(self.route).all():
            raise WorkflowError('invalid_input','Navigation route')
        for key in ('speed_m_s','acceleration_m_s2','response_s','stop_radius_m','waypoint_radius_m',
                    'yaw_speed_rad_s','yaw_acceleration_rad_s2','yaw_tolerance_rad',
                    'settle_speed_m_s','settle_yaw_speed_rad_s','max_dt_s'):
            positive(limits[key],key)
        self.o=dict(limits);self.index=1;self.velocity=np.zeros(2);self.yaw_rate=0.;self.heading=None;self.elapsed=0.

    def step(self,xyyaw,measured_velocity,dt):
        pose=finite(xyyaw,(3,),'Measured navigation pose');measured=finite(measured_velocity,(3,),'Measured vx vy yaw-rate')
        positive(dt,'Navigation dt');o=self.o
        self.elapsed+=dt
        if self.heading is None:self.heading=float(pose[2])
        last=len(self.route)-1
        while self.index<last:
            target=self.route[self.index];incoming=target[:2]-self.route[self.index-1,:2]
            outgoing=self.route[self.index+1,:2]-target[:2]
            ni,no=np.linalg.norm(incoming),np.linalg.norm(outgoing)
            straight=ni>1e-9 and no>1e-9 and incoming@outgoing/(ni*no)>1-1e-6
            reached=np.linalg.norm(target[:2]-pose[:2])<=o['waypoint_radius_m']
            quiet=np.linalg.norm(measured[:2])<=o['settle_speed_m_s'] and np.linalg.norm(self.velocity)<=o['settle_speed_m_s']
            yaw_ok=abs(float(wrap(target[2]-pose[2])))<=o['yaw_tolerance_rad']
            if reached and yaw_ok and (straight or quiet):self.index+=1
            else:break
        a=self.route[self.index-1,:2];b=self.route[self.index,:2];ab=b-a
        u=np.clip((pose[:2]-a)@ab/max(float(ab@ab),1e-20),0,1)
        cross=float(np.linalg.norm(pose[:2]-(a+u*ab)))
        delta=b-pose[:2];distance=float(np.linalg.norm(delta));terminal=self.index==last
        radius=o['stop_radius_m'] if terminal else o['waypoint_radius_m']
        direction=delta/distance if distance else np.zeros(2)
        closing=max(0.,float(measured[:2]@direction))
        remaining=max(0.,distance-.5*radius-closing*o['response_s'])
        speed=min(o['speed_m_s'],np.sqrt(2*o['acceleration_m_s2']*remaining),remaining/o['response_s'])
        desired=direction*speed
        change=desired-self.velocity;norm=float(np.linalg.norm(change))
        if norm:self.velocity+=change*min(1.,o['acceleration_m_s2']*dt/norm)
        target_yaw=float(self.route[self.index,2]);error=float(wrap(target_yaw-self.heading))
        desired_rate=np.sign(error)*min(o['yaw_speed_rad_s'],np.sqrt((o['yaw_acceleration_rad_s2']*o['max_dt_s'])**2+2*o['yaw_acceleration_rad_s2']*abs(error))-o['yaw_acceleration_rad_s2']*o['max_dt_s'],abs(error)/dt)
        self.yaw_rate+=float(np.clip(desired_rate-self.yaw_rate,-o['yaw_acceleration_rad_s2']*dt,o['yaw_acceleration_rad_s2']*dt))
        self.heading=float(wrap(self.heading+self.yaw_rate*dt))
        arrived=bool(terminal and distance<=radius and abs(float(wrap(target_yaw-pose[2])))<=o['yaw_tolerance_rad']
                     and np.linalg.norm(measured[:2])<=o['settle_speed_m_s'] and abs(measured[2])<=o['settle_yaw_speed_rad_s']
                     and np.linalg.norm(self.velocity)<=o['settle_speed_m_s'])
        return {'velocity_world_xy_m_s':self.velocity.tolist(),'facing_yaw_rad':self.heading,
                'waypoint':self.index,'arrived':arrived,'state':'SETTLE' if arrived else 'BRAKE' if speed<o['speed_m_s'] else 'WALK',
                'distance_m':distance,'corridor_error_m':cross,'elapsed_s':self.elapsed,
                'predicted_stop_xy':(pose[:2]+measured[:2]*o['response_s']).tolist(),
                'prediction_kind':'configured_response_not_calibrated_physical_guarantee'}


def validate_route_calibration(artifact,scope,quality,limits):
    from ..parking import validate_calibration
    model=validate_calibration(artifact,scope,quality)
    if limits['speed_m_s']>model['observed_speed_max_m_s'] or limits['response_s']<model['tau_s']:
        raise WorkflowError('needs_calibration','Route exceeds measured speed/response range')
    if model['residual_max_m']>limits['stop_radius_m']:
        raise WorkflowError('needs_calibration','Stopping residual exceeds arrival radius')
    return model
