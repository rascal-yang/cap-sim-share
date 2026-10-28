"""Reusable measured-condition gates, independent of robot and source indices."""
from .contracts import WorkflowError, positive


class GateMonitor:
    def __init__(self, limits):
        self.limits=limits
        for key in ('contact_force_N','contact_loss_s','collision_force_N','collision_duration_s','collision_instant_N'):
            positive(limits[key],key)
        self.contact_s=self.lost_s=self.collision_s=0.
        self.stopped=None

    def update(self,dt,*,hand_force,obstacle_force,requires_contact):
        if self.stopped:raise WorkflowError('execution_stopped',self.stopped)
        positive(dt,'physics dt');positive(hand_force,'hand force',zero=True);positive(obstacle_force,'obstacle force',zero=True)
        p=self.limits
        self.contact_s=self.contact_s+dt if hand_force>p['contact_force_N'] else 0.
        self.lost_s=self.lost_s+dt if requires_contact and hand_force<=p['contact_force_N'] else 0.
        self.collision_s=self.collision_s+dt if obstacle_force>p['collision_force_N'] else 0.
        if obstacle_force>p['collision_instant_N'] or self.collision_s+1e-12>=p['collision_duration_s']:self.stopped='collision'
        elif self.lost_s+1e-12>=p['contact_loss_s']:self.stopped='contact_lost'
        return self.stopped


def base_is_stable(ob, limits):
    return (ob['recent_root_drift_m'] is not None and ob['recent_root_drift_m']<=limits['drift_m']
            and ob['root_speed_m_s']<=limits['speed_m_s'] and ob['yaw_speed_rad_s']<=limits['yaw_speed_rad_s']
            and min(ob['foot_ground_N'])>limits['foot_force_N'])


def entry_ready(actual,targets,hands,limits):
    import numpy as np
    from scipy.spatial.transform import Rotation
    def rot(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])
    return all(np.linalg.norm(actual[h][:3]-targets[h][:3])<=limits['position_m']
               and (rot(actual[h][3:]).inv()*rot(targets[h][3:])).magnitude()<=limits['orientation_rad'] for h in hands)


def finite_record(value):
    """Unknown nonfinite measurements stay null in audits, never become zero."""
    import math
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:finite_record(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_record(v) for v in value]
    return value
