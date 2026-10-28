"""Measured, bounded feedback and conservative physical recovery eligibility."""
import numpy as np
from .contracts import positive


class IntegralFeedback:
    def __init__(self,hands,*,gain_per_s,cap_m):
        self.gain=positive(gain_per_s,'feedback gain');self.cap=positive(cap_m,'feedback cap')
        self.bias={h:np.zeros(3) for h in hands}

    def update(self,hand,desired,measured,dt):
        positive(dt,'feedback dt')
        error=np.asarray(desired)-np.asarray(measured)
        if not np.isfinite(error).all():raise ValueError('Nonfinite feedback observation')
        bias=self.bias[hand]+self.gain*dt*error
        norm=float(np.linalg.norm(bias))
        self.bias[hand]=bias*min(1.,self.cap/norm) if norm else bias
        return self.bias[hand].copy()


def can_recover(reason,ob,retries,max_retries,support_force_N):
    return (reason in ('contact_lost','contact_entry_failed','clearance_entry_failed')
            and retries<max_retries and ob['contacts']['support_N']>support_force_N
            and ob['contacts']['robot_obstacle_N']==0)
