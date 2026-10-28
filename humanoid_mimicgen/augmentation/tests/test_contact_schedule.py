import pytest
from humanoid_mimicgen.augmentation.suite_physics import token_prefix_steps, uses_original_token

def test_switch_has_no_duplicate_tick():
    assert token_prefix_steps('token_prefix_mode1',3,8)==3
    assert [uses_original_token('token_prefix_mode1',i,3) for i in range(8)]==[True]*3+[False]*5
    assert token_prefix_steps('original_token',-1,8)==8
    assert token_prefix_steps('cartesian_source_legs',-1,8)==0

@pytest.mark.parametrize('start',[-1,8,1.5])
def test_invalid_switch(start):
    with pytest.raises(ValueError): token_prefix_steps('token_prefix_mode1',start,8)

def test_rotation_sign_and_wrap():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from humanoid_mimicgen.augmentation.trajectory_diagnostics import rotation_residual
    def q(deg):return Rotation.from_euler('z',deg,degrees=True).as_quat()[[3,0,1,2]]
    np.testing.assert_allclose(rotation_residual(q(-179),q(179)),[0,0,np.deg2rad(2)],atol=1e-12)
