import math
import pytest


def test_stage_diagnosis_distinguishes_command_actual_and_base_retreat():
    from cap_sim.diagnostics import stage_diagnosis
    before={'time':0.,'root_pose':[0.,.7,.78,1.,0.,0.,0.]}
    result={'completed':True,'failure':None,'observation':{'time':3.,'root_pose':[0.,.9,.683,1.,0.,0.,0.],
            'right_position':[0.,.48,.85],'right_quaternion_wxyz':[1.,0.,0.,0.]}}
    action={'stage':'approach','pelvis_height_m':.68,'right_position':[0.,.32,.80],
            'right_quaternion_wxyz':[0.,0.,0.,1.]}
    d=stage_diagnosis(action,before,result)
    m=d['measurements']
    assert m['pelvis_height_error_m']==pytest.approx(.003)
    assert m['base_displacement_xyz_m']==pytest.approx([0.,.2,-.097])
    assert m['right_wrist_position_error_m']==pytest.approx(math.sqrt(.16**2+.05**2))
    assert m['right_wrist_orientation_error_rad']==pytest.approx(math.pi)
    assert d['requested_mode']==1


def test_missing_data_is_not_zero_and_mode0_height_channel_is_not_effective():
    from cap_sim.diagnostics import stage_diagnosis
    d=stage_diagnosis({'body_keyframes':[[0,0,0,.6,1,0,0,0],[1,0,0,.6,1,0,0,0]],'pelvis_height_m':.4},{},
                      {'failure':None,'observation':{}})
    assert d['requested_mode']==0
    assert d['measurements']['pelvis_height_error_m'] is None
    assert d['measurements']['right_wrist_position_error_m'] is None
    assert d['targets']['pelvis_height_m'] is None
    assert d['targets']['body_reference_root_height_m']==.6


@pytest.mark.parametrize('action',[{'pelvis_height_m':'bad'},{'body_keyframes':[1,2]}])
def test_invalid_action_still_has_diagnostic_instead_of_crashing(action):
    from cap_sim.diagnostics import stage_diagnosis
    d=stage_diagnosis(action,{}, {'failure':'invalid_action','observation':{'root_pose':[0,0,.7,1,0,0,0]}})
    assert d['failure']=='invalid_action'
    assert d['measurements']['pelvis_height_error_m'] is None
