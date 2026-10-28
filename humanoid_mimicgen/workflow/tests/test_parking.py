import numpy as np
import pytest
from humanoid_mimicgen.workflow.parking import fit_stop_response, predicted_stop, correction_decision, RecentWindow
from humanoid_mimicgen.workflow.contracts import WorkflowError


def test_stop_response_identification_is_rotation_equivariant():
    v=np.array([[.3,.1],[-.2,.4]])
    model=fit_stop_response(v,v*.4)
    assert model['tau_s']==pytest.approx(.4)
    np.testing.assert_allclose(predicted_stop([1,2],[.3,.1],model),[1.12,2.04])
    assert model['residual_max_m']<1e-12
    with pytest.raises(WorkflowError):fit_stop_response([[0,0]],[[.1,0]])
    with pytest.raises(WorkflowError):fit_stop_response(v,-v)


def test_corrections_are_bounded_and_do_not_turn_into_grasp_veto():
    opts={'tolerance_m':.03,'max_corrections':2,'minimum_progress_m':.003}
    assert correction_decision(.02,None,0,opts)=='within_tolerance'
    assert correction_decision(.06,None,0,opts)=='correct'
    assert correction_decision(.07,.06,1,opts)=='no_progress'
    assert correction_decision(.05,.08,2,opts)=='budget_exhausted'
    assert correction_decision(.05,.08,1,opts)=='correct'


def test_bounded_window_matches_old_history_query():
    w=RecentWindow(.3,.005);all_rows=[]
    for i in range(10000):
        row={'time_s':(i+1)*.005,'root_pose':[i*.001,0]};all_rows.append(row);w.append(row)
        expected=[x for x in all_rows if x['time_s']>=row['time_s']-.3-.005/2]
        assert list(w.rows)==expected
        assert len(w.rows)<=62


def test_duplicate_pairs_do_not_count_as_independent_calibration():
    m=fit_stop_response([[.3,0],[.3,0]],[[.1,0],[.1,0]])
    assert m['sample_count']==2 and m['unique_sample_count']==1


def test_calibration_binds_sonic_weights_and_requires_quality():
    from humanoid_mimicgen.workflow.parking import response_scope,validate_calibration
    s={'adapter':'g1','bindings':{},'phases':[{'start':1}],'command_pose_offset':0,'policy':{'navigation':{},'settle':{},'control_dt_s':.02}}
    scope=response_scope(s,'scene','source','physics',{'encoder.onnx':'old'})
    other=response_scope(s,'scene','source','physics',{'encoder.onnx':'new'})
    row={'velocity_xy_m_s':[.3,0],'displacement_xy_m':[.1,0]}
    a={'schema_version':2,'kind':'zero_intercept_velocity_response','scope':scope,'evidence_sha256':{},'samples':[row],'model':fit_stop_response([[.3,0]],[[.1,0]])}
    with pytest.raises(WorkflowError):validate_calibration(a,other,{'minimum_samples':1,'max_residual_m':.03})
    with pytest.raises(WorkflowError):validate_calibration(a,scope,{'minimum_samples':2,'max_residual_m':.03})


def test_stationary_correction_requires_com_inside_measured_support():
    from humanoid_mimicgen.workflow.parking import support_contains
    points=[[-.1,-.1],[-.1,.1],[.1,-.1],[.1,.1]]
    assert support_contains(points,[0,0])
    assert not support_contains(points,[.2,0])
    assert not support_contains([[0,0],[1,0]],[.5,0])


def test_reference_residual_update_does_not_change_evaluation_goal():
    from humanoid_mimicgen.workflow.parking import corrected_reference
    goal=np.array([1.,2.]);reference=goal.copy()
    shifted=corrected_reference(reference,goal,[1.03,1.98])
    np.testing.assert_allclose(shifted,[.97,2.02])
    np.testing.assert_array_equal(goal,[1,2]);np.testing.assert_array_equal(reference,goal)
