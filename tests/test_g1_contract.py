import pytest
from cap_sim.backends.g1 import Backend

@pytest.mark.parametrize('action',[
 {'stage':'lift','duration_s':float('nan')},
 {'stage':'contact','duration_s':1,'fingers':[1,2]},
 {'stage':'approach','duration_s':1,'wrist_quaternion_wxyz':[0,0,0,0]},
 {'stage':'approach','duration_s':1,'pelvis_height_m':.2},
])
def test_invalid_g1_targets_rejected_before_worker(action):
 with pytest.raises(ValueError):Backend().execute(action,30)


def test_g1_substep_rounding_cannot_overrun_budget():
 with pytest.raises(ValueError):Backend().execute({'stage':'recover','duration_s':.103},.103)
