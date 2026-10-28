import numpy as np
import pytest


def test_translation_moves_only_right_xyz_without_mutating_source():
    from humanoid_mimicgen.contact_transfer import translate_contact
    pose=np.array([[1.,2.,3.,1.,0.,0.,0.],[2.,3.,4.,1.,0.,0.,0.]])
    src={'right_pose':pose.copy(),'left_pose':pose.copy(),'fingers':np.ones((2,14))}
    out=translate_contact(src,[0,.02,0])
    np.testing.assert_allclose(out['right_pose'][:,:3]-src['right_pose'][:,:3],[[0,.02,0],[0,.02,0]])
    np.testing.assert_array_equal(out['right_pose'][:,3:],src['right_pose'][:,3:])
    np.testing.assert_array_equal(out['left_pose'],src['left_pose'])
    np.testing.assert_array_equal(src['right_pose'],pose)
    np.testing.assert_array_equal(out['fingers'],src['fingers'])


def test_approach_has_exact_endpoints_and_shortest_rotation():
    from humanoid_mimicgen.contact_transfer import approach_pose
    a=np.array([0.,0.,0.,1.,0.,0.,0.]);b=np.array([0.,.02,0.,-1.,0.,0.,0.])
    np.testing.assert_allclose(approach_pose(a,b,0)[:3],a[:3])
    np.testing.assert_allclose(approach_pose(a,b,1)[:3],b[:3])
    np.testing.assert_allclose(approach_pose(a,b,.5)[:3],[0,.01,0])
    assert abs(approach_pose(a,b,.5)[3])==1


def test_transfer_rejects_unapproved_large_offset():
    from humanoid_mimicgen.contact_transfer import translate_contact
    with pytest.raises(ValueError):translate_contact({'right_pose':np.zeros((2,7))},[0,.2,0])
