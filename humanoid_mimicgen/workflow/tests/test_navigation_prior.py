import numpy as np
import pytest
from humanoid_mimicgen.workflow.navigation.prior import extract_prior, transfer_stance
from humanoid_mimicgen.workflow.contracts import WorkflowError


def example():
    t=np.arange(101)*.02
    root=np.tile([1.,0.,.8,1.,0.,0.,0.],(101,1))
    obj=np.tile([2.,0.,.7,1.,0.,0.,0.],(101,1))
    return {'time_s':t,'root_pose':root,'object_pose':obj,'root_velocity':np.zeros((101,6)),
            'support_ok':np.ones(101,dtype=bool),'success':True,'success_criterion':'native',
            'entry_index':60,'end_index':100,'task':'test'}, {'object':'target','frame':{'kind':'world_heading','yaw_rad':0.,'evidence':'table edge'},'key_indices':[60,80,100]}, {'window_s':.2,'lookback_s':1.,'speed_m_s':.05,'yaw_speed_rad_s':.1,'drift_m':.02}


def test_extract_and_transfer_rotated_scene_keeps_ground_height():
    trace,binding,options=example(); prior=extract_prior(trace,binding,options)
    assert prior['status']=='completed'
    np.testing.assert_allclose(prior['root_in_task_xyyaw'],[-1,0,0],atol=1e-9)
    np.testing.assert_allclose(transfer_stance(prior,[4,3,2,np.pi/2]),[4,2,np.pi/2],atol=1e-9)
    assert prior['source_root_height_m']==.8
    assert prior['success_criterion']=='native'


def test_spherical_object_rotation_does_not_change_table_frame():
    trace,binding,options=example(); a=extract_prior(trace,binding,options)
    trace['object_pose'][:,3:]=[0,0,0,1]
    b=extract_prior(trace,binding,options)
    assert a['root_in_task_xyyaw']==b['root_in_task_xyyaw']


def test_missing_success_and_unstable_source_are_not_successful_priors():
    trace,binding,options=example();trace['success']=None
    assert extract_prior(trace,binding,options)['status']=='unverified_source'
    trace['success']=True;trace['root_velocity'][:,0]=.3
    assert extract_prior(trace,binding,options)['status']=='unsupported_entry_pattern'


def test_missing_phase_and_ambiguous_frame_rejected():
    trace,binding,options=example();trace.pop('entry_index')
    with pytest.raises(WorkflowError,match='phase'):extract_prior(trace,binding,options)
    trace,binding,options=example();binding['frame']={'kind':'object_axis','axis':[0,0,1]}
    with pytest.raises(WorkflowError):extract_prior(trace,binding,options)


def test_stable_window_must_be_recent_and_contiguous():
    trace,binding,options=example();trace['support_ok'][45:61]=False
    options['lookback_s']=.25
    assert extract_prior(trace,binding,options)['status']=='unsupported_entry_pattern'


def test_reject_bad_time_and_integer_success():
    trace,binding,options=example();trace['time_s'][5]=trace['time_s'][4]
    with pytest.raises(WorkflowError):extract_prior(trace,binding,options)
    trace,binding,options=example();trace['success']=1
    assert extract_prior(trace,binding,options)['status']=='unverified_source'


def test_stationary_before_later_walk_is_not_operation_stance():
    trace,binding,options=example()
    trace['root_velocity'][45:61,0]=.3
    trace['root_pose'][45:61,0]+=np.linspace(0,.3,16)
    assert extract_prior(trace,binding,options)['status']=='unsupported_entry_pattern'
