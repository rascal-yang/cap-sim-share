"""Bounded world-frame orientation integral bias; never writes motor/state values."""
import numpy as np
from cap_sim.backends.native_sonic import rot,quat
from scipy.spatial.transform import Rotation

class TorsoFeedback:
    def __init__(self,gain=.8,limit=.5):
        if not 0<gain<=2 or not 0<limit<=.5:raise ValueError('Invalid feedback limits')
        self.gain=gain;self.limit=limit;self.correction=np.zeros(3)
    def update(self,desired,measured,dt):
        if not 0<dt<=.02:raise ValueError('Invalid dt')
        error=(rot(desired)*rot(measured).inv()).as_rotvec()
        self.correction+=self.gain*dt*error
        n=np.linalg.norm(self.correction)
        if n>self.limit:self.correction*=self.limit/n
        return quat(Rotation.from_rotvec(self.correction)*rot(desired))
