import copy
import numpy as np
import pytest
from humanoid_mimicgen.workflow.contracts import WorkflowError


def sample():
    n=51; t=np.arange(n)*.01
    root=np.tile([0.,0.,.8,1.,0.,0.,0.],(n,1));root[:,0]=.05*np.sin(t*4)
    feet={name:np.tile([0.,y,.03,1.,0.,0.,0.],(n,1)) for name,y in [('left',.1),('right',-.1)]}
    trace=dict(time_s=t,root_pose=root,qpos=root,root_velocity=np.zeros((n,6)),
               support_ok=np.ones(n,bool),entry_index=30,end_index=50,success=True,success_criterion='native',
               object_pose=np.tile([.3,0.,.7,1.,0.,0.,0.],(n,1)))
    bindings=dict(feet=['left','right'],root_body='pelvis',object='apple',frame=dict(kind='world_heading',yaw_rad=0.,evidence='table'),
                  operation_stages=dict(entry=30,closure=35,first_contact=38,lift=40,end=50))
    options=dict(window_s=.1,lookback_s=.3,foot_drift_m=.002,foot_angle_rad=.02,max_windows=3,
                 open_fingers_rad=[0.]*14,finger_tolerance_rad=1e-6)
    fingers=np.zeros((n,14));fingers[35:,8]=.7
    return trace,bindings,options,{**feet,'pelvis':root},fingers


def test_timestamp_alignment_does_not_assume_four_substeps():
    from humanoid_mimicgen.workflow.navigation.lateral_prior import align_hand_commands
    f,rows=align_hand_commands([1.,1.006,1.02,1.026],[1.,1.02],np.array([[0.]*14,[1.]*14]),.02)
    assert rows.tolist()==[0,0,1,1]
    assert f[:,0].tolist()==[0,0,1,1]


@pytest.mark.parametrize('times,control', [([.99,1.01],[1.,1.02]),([1.,1.041],[1.,1.02]),([1.,1.03],[1.,1.04])])
def test_missing_hand_coverage_is_not_extrapolated(times,control):
    from humanoid_mimicgen.workflow.navigation.lateral_prior import align_hand_commands
    with pytest.raises(WorkflowError):align_hand_commands(times,control,np.zeros((2,14)),.02)


def test_support_windows_allow_pelvis_motion_and_preserve_both_feet():
    from humanoid_mimicgen.workflow.navigation.lateral_prior import extract_support_prior
    result=extract_support_prior(*sample())
    assert result['status']=='completed'
    assert len(result['support_windows'])>1
    assert result['support_windows'][0]['feet']['left'][1]==pytest.approx(.1)
    assert result['key_indices']==[30,35,38,40,50]
    assert result['dynamic_execution_verified'] is False


@pytest.mark.parametrize('change,expected',[('feet','unsupported_entry_pattern'),('closed','invalid_operation_entry'),('success','unverified_source')])
def test_no_false_support_or_open_entry(change,expected):
    from humanoid_mimicgen.workflow.navigation.lateral_prior import extract_support_prior
    args=list(sample())
    if change=='feet':args[3]['left'][:,0]=np.arange(51)*.005
    if change=='closed':args[4][30,8]=.5
    if change=='success':args[0]['success']=False
    assert extract_support_prior(*args)['status']==expected
