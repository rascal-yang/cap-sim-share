"""Reusable distance-driven velocity envelope and root-relative recovery targets.

Limits are explicit inputs, not learned robot capabilities. The velocity envelope
is replanned from measured position every control tick; acceleration bounds apply
to commands, not to the physical robot's response.
"""
import numpy as np
from .contracts import positive, WorkflowError


class DistanceVelocityProfile:
    def __init__(self, *, speed, acceleration, response_s):
        for name,value in [('speed',speed),('acceleration',acceleration),('response_s',response_s)]:positive(value,name)
        self.speed=float(speed);self.acceleration=float(acceleration);self.response_s=float(response_s)
        self.velocity=np.zeros(2)

    def step(self, delta, dt, *, measured_velocity=None, arrival_radius=0.):
        positive(dt,'velocity profile dt');delta=np.asarray(delta,float)
        if delta.shape!=(2,) or not np.isfinite(delta).all():raise WorkflowError('invalid_input','Navigation displacement')
        distance=float(np.linalg.norm(delta))
        # Linear terminal envelope approaches zero continuously; the braking
        # envelope also prevents a long route from retaining cruise speed nearby.
        positive(arrival_radius,'arrival radius',zero=True)
        measured=np.zeros(2) if measured_velocity is None else np.asarray(measured_velocity,float)
        if measured.shape!=(2,) or not np.isfinite(measured).all():raise WorkflowError('invalid_input','Measured navigation velocity')
        closing=max(0.,float(measured@delta/distance)) if distance else 0.
        remaining=max(0.,distance-arrival_radius-closing*self.response_s)
        speed=min(self.speed,np.sqrt(2*self.acceleration*remaining),remaining/self.response_s)
        desired=delta*(speed/distance) if distance else np.zeros(2)
        change=desired-self.velocity;norm=float(np.linalg.norm(change))
        self.velocity+=change*min(1.,self.acceleration*dt/norm) if norm else change
        return self.velocity.copy()


def recovery_configuration(current, normal, addresses):
    """Copy only arm/waist joints; preserve measured root, legs, fingers, objects."""
    current=np.asarray(current,float);normal=np.asarray(normal,float);addresses=np.asarray(addresses,int)
    if current.shape!=normal.shape or not np.isfinite(current).all() or not np.isfinite(normal).all():
        raise WorkflowError('invalid_input','Recovery configuration')
    target=current.copy();target[addresses]=normal[addresses]
    return target


def blend_stance_reference(walking, standing, times, duration):
    """C2 blend of existing future reference to stance; quaternion SLERP."""
    from scipy.spatial.transform import Rotation
    positive(duration,'handoff duration');walking=np.asarray(walking,float);standing=np.asarray(standing,float)
    u=np.clip(np.asarray(times,float)/duration,0,1);weight=u**3*(10-15*u+6*u*u)
    result=walking*(1-weight[:,None])+standing*weight[:,None]
    start=Rotation.from_quat(np.roll(walking[:,3:7],-1,axis=1));end=Rotation.from_quat(np.roll(standing[3:7],-1))
    rotation=start*Rotation.from_rotvec((start.inv()*end).as_rotvec()*weight[:,None])
    result[:,3:7]=np.roll(rotation.as_quat(),1,axis=1)
    return result


def upright_root_reference(measured_root):
    """World-up orientation with measured XYZ/yaw; a reference, never state."""
    from scipy.spatial.transform import Rotation
    q=np.array(measured_root,float,copy=True)
    if q.shape!=(7,) or not np.isfinite(q).all():raise WorkflowError('invalid_input','Measured root pose')
    yaw=Rotation.from_quat(np.roll(q[3:],-1)).as_euler('xyz')[2]
    q[3:]=np.roll(Rotation.from_euler('z',yaw).as_quat(),1)
    return q
