"""Reference-only source roll/pitch with measured root translation and yaw."""
import numpy as np
from scipy.spatial.transform import Rotation
from cap_sim.backends.native_sonic import rot,quat

def standing_root_reference(measured,source):
    result=np.array(measured,copy=True);angles=rot(source[3:7]).as_euler('xyz')
    angles[2]=rot(measured[3:7]).as_euler('xyz')[2]
    result[3:7]=quat(Rotation.from_euler('xyz',angles))
    return result
