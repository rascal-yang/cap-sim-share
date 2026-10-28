import numpy as np
import pytest


def test_original_finger_close_preserves_timing_and_endpoints():
    from humanoid_mimicgen.contact_controller import finger_schedule
    commands=finger_schedule(np.zeros((250,14)),[{'duration_s':1.5},{'duration_s':1.5},{'duration_s':1.5,'right_fingers':[1]*7}],1900)
    assert np.array_equal(commands[1599],np.zeros(14))
    assert np.allclose(commands[1749,7:],.5)
    assert np.array_equal(commands[1899,7:],np.ones(7))
    assert np.array_equal(commands[:,:7],np.zeros((1900,7)))


def test_controller_contract_rejects_unfrozen_input(tmp_path):
    from humanoid_mimicgen.contact_controller import check_contract
    with pytest.raises((ValueError,FileNotFoundError)):
        check_contract(tmp_path/'experiment.json')
