import numpy as np
import pytest
from humanoid_mimicgen.boundaries import boundary_clip


def test_both_endpoints_are_preserved():
    e={'time_s':np.arange(5)*.1,'qpos':np.arange(10).reshape(5,2),'source_frame_index':np.arange(5)}
    clip,meta=boundary_clip(e,1,3,'open hand','grasp established')
    assert clip['source_frame_index'].tolist()==[1,2,3]
    assert meta['duration_s']==pytest.approx(.2)
    assert meta['end']['state']['qpos']==[6,7]
    assert meta['start']['state']['qpos']==[2,3]


def test_shared_boundary_is_identical():
    e={'time_s':np.arange(6)*.1,'qpos':np.arange(6)[:,None]}
    left,_=boundary_clip(e,0,3,'a','b');right,_=boundary_clip(e,3,5,'b','c')
    np.testing.assert_array_equal(left['qpos'][-1],right['qpos'][0])


@pytest.mark.parametrize('start,end',[(-1,2),(1,5),(3,1),(2,2),(True,3),(1.5,3)])
def test_invalid_boundaries_rejected(start,end):
    with pytest.raises(ValueError):boundary_clip({'time_s':np.arange(5)},start,end,'a','b')


def test_missing_semantics_rejected():
    with pytest.raises(ValueError):boundary_clip({'time_s':np.arange(5)},0,3,'','b')
