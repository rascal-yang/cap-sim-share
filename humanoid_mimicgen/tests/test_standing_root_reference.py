import numpy as np
from cap_sim.backends.native_sonic import rot,quat
from scipy.spatial.transform import Rotation
from humanoid_mimicgen.standing_root_reference import standing_root_reference

def test_reference_keeps_measured_position_yaw_and_source_roll_pitch():
    measured=np.r_[[1,2,.78],quat(Rotation.from_euler('xyz',[-.1,-.2,.4]))]
    source=np.r_[[4,5,.79],quat(Rotation.from_euler('xyz',[.01,.1,-.3]))]
    out=standing_root_reference(measured,source)
    assert np.array_equal(out[:3],measured[:3])
    assert np.allclose(rot(out[3:]).as_euler('xyz'),[.01,.1,.4])
    assert np.allclose(rot(measured[3:]).as_euler('xyz'),[-.1,-.2,.4])
