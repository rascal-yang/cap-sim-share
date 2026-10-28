import numpy as np
import pytest
from humanoid_mimicgen.workflow.calibration import identify_response, equilibrium_command
from humanoid_mimicgen.workflow.execution import GateMonitor
from humanoid_mimicgen.workflow.contracts import WorkflowError, phase_at


def test_identified_response_not_experiment_gains():
    rng=np.random.default_rng(3);u=rng.normal(size=(200,3));y=np.zeros_like(u)
    for i in range(len(u)-1):y[i+1]=.7*y[i]+.3*u[i]+[.003,-.006,.009]
    c=identify_response(u,y,scope={'robot':'x'},max_prediction_error_m=1e-8)
    np.testing.assert_allclose(equilibrium_command([0,0,0],c,{'robot':'x'}),[-.01,.02,-.03],atol=1e-10)
    with pytest.raises(WorkflowError,match='needs_calibration'):equilibrium_command([0,0,0],c,{'robot':'other'})
    with pytest.raises(WorkflowError,match='needs_calibration'):identify_response(np.zeros((10,3)),np.zeros((10,3)),scope={},max_prediction_error_m=.001)


def test_dynamic_phase_lookup_and_contact_stop():
    phases=[{'id':'close','start':10,'end':15},{'id':'raise','start':15,'end':25}]
    assert phase_at(phases,16)['id']=='raise'
    gates=GateMonitor({'contact_force_N':.2,'contact_loss_s':.04,'collision_force_N':4.,'collision_duration_s':.03,'collision_instant_N':40.})
    assert gates.update(.02,hand_force=0,obstacle_force=0,requires_contact=False) is None
    assert gates.update(.02,hand_force=0,obstacle_force=0,requires_contact=True) is None
    assert gates.update(.02,hand_force=0,obstacle_force=0,requires_contact=True)=='contact_lost'
    with pytest.raises(WorkflowError):gates.update(.02,hand_force=2,obstacle_force=0,requires_contact=False)


def test_collision_cannot_be_overridden_by_good_grasp():
    gates=GateMonitor({'contact_force_N':.2,'contact_loss_s':.04,'collision_force_N':4.,'collision_duration_s':.03,'collision_instant_N':40.})
    assert gates.update(.02,hand_force=20,obstacle_force=41,requires_contact=True)=='collision'
