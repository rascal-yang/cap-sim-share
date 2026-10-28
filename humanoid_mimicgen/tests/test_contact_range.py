import numpy as np
import pytest
from humanoid_mimicgen.contact_range import shifted_targets, physical_classification

def test_translation_preserves_other_channels_and_source():
    src={'right_pose':np.array([[1.,2.,3.,1.,0.,0.,0.]]),'fingers':np.ones((1,14)),'left_pose':np.zeros((1,7))}
    dst=shifted_targets(src,[.08,-.04,0.])
    assert np.allclose(dst['right_pose'][0,:3],[1.08,1.96,3.])
    assert np.array_equal(dst['right_pose'][:,3:],src['right_pose'][:,3:])
    assert np.array_equal(dst['fingers'],src['fingers'])
    assert np.array_equal(src['right_pose'][0,:3],[1.,2.,3.])

def test_tabletop_scan_rejects_vertical_or_nonfinite_shift():
    for delta in ([0,0,.02],[float('nan'),0,0],[.2,0,0]):
        with pytest.raises(ValueError):shifted_targets({},delta)

def test_no_contact_is_not_a_grasp_failure_rate():
    assert physical_classification(False,False,0,0,0,0)=='no_contact_execution'

def test_contact_warning_is_separate_from_task_completion():
    assert physical_classification(True,True,1.,.01,0.,0.)=='clean_grasp_lift_hold'
    assert physical_classification(True,True,1.,.01,.09,5.5)=='grasp_lift_hold_with_table_contact'
    assert physical_classification(True,False,0.,.01,0.,0.)=='contact_execution_failed'
