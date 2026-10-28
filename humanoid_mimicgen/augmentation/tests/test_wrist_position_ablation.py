import numpy as np
import pytest
from humanoid_mimicgen.augmentation.wrist_position_ablation import adjust_reference

def test_world_shift_once_and_only_right_position():
    pose=np.zeros((3,7));pose[:,3]=1;pose[1,:3]=[1,2,3]
    before=pose.copy();out,bias,log=adjust_reference(pose,[.02,-.03],[.9,1.9,3],np.zeros(3),True)
    np.testing.assert_allclose(log['desired_world_m'],[1.02,1.97,3])
    np.testing.assert_allclose(bias,[.0012,.0007,0])
    np.testing.assert_array_equal(pose,before)
    np.testing.assert_array_equal(out[[0,2]],pose[[0,2]])
    np.testing.assert_array_equal(out[:,3:],pose[:,3:])
    np.testing.assert_allclose(out[1,:3],pose[1,:3]+bias)

def test_old_integrator_cap_and_disabled_identity():
    pose=np.zeros((3,7));pose[:,3]=1
    out,bias,log=adjust_reference(pose,[0,0],[-100,0,0],np.zeros(3),True)
    np.testing.assert_allclose(bias,[.02,0,0])
    out,bias,log=adjust_reference(pose,[0,0],[-100,0,0],np.zeros(3),False)
    np.testing.assert_array_equal(out,pose);np.testing.assert_array_equal(bias,np.zeros(3))

def test_reject_nonfinite_feedback():
    with pytest.raises(ValueError):adjust_reference(np.zeros((3,7)),[0,0],[np.nan,0,0],np.zeros(3),True)

def test_checkpoint_version_is_rejected(tmp_path):
    import json
    from humanoid_mimicgen.augmentation.retry_state import load
    out=tmp_path/'execution';out.mkdir();(out/'checkpoint_manifest.json').write_text(json.dumps({'version':2}))
    with pytest.raises(ValueError,match='Unsupported checkpoint'):load(tmp_path,0)
