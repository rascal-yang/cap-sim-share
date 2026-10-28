"""Catch false hold labels, dropped objects, timestamp errors and contact-side mixing."""
import numpy as np
import pytest
from humanoid_mimicgen.privileged import segment_phases


def evidence():
    t=np.arange(251)*.02
    z=.7+np.clip(t-2,0,1)*.1
    hand=t>=1
    return dict(time_s=t,object_pose=np.column_stack([t*0,t*0,z,t*0+1,t*0,t*0,t*0]),
                clearance_m=z-.7,left_contact=np.zeros(len(t),bool),right_contact=hand,
                support_contact=t<=2,relative_position=np.tile([.02,0,0],(len(t),1)),
                relative_quaternion=np.tile([1.,0,0,0],(len(t),1)))


def test_extract_ordered_grasp_lift_hold():
    r=segment_phases(evidence())
    assert r['status']=='extracted'
    stages=r['segments']; assert [s['name'] for s in stages]==['grasp','lift','hold']
    assert stages[0]['start_frame']==50
    assert 100<=stages[1]['start_frame']<=110
    assert 150<=stages[2]['start_frame']<=155
    assert stages[-1]['end_frame']==251
    assert r['active_hand']=='right'


def test_airborne_without_hand_contact_is_not_grasp():
    e=evidence();e['right_contact'][:]=False
    assert segment_phases(e)['status']=='rejected'


def test_touching_supported_object_is_not_lift():
    e=evidence();e['support_contact'][:]=True
    assert segment_phases(e)['status']=='rejected'


def test_slip_cannot_be_called_hold():
    e=evidence();e['relative_position'][:,0]=e['time_s']*.2
    r=segment_phases(e)
    assert r['missing_stages']==['hold']
    assert all(s['name']!='hold' for s in r['segments'])


def test_hold_stops_before_release():
    e=evidence();e['right_contact'][200:]=False
    r=segment_phases(e)
    assert r['status']=='extracted'
    assert r['segments'][-1]['end_frame']<=200


def test_duplicate_time_rejected():
    e=evidence();e['time_s'][5]=e['time_s'][4]
    with pytest.raises(ValueError):segment_phases(e)


def test_short_touch_does_not_start_grasp():
    e=evidence();e['right_contact'][10]=True
    assert segment_phases(e)['segments'][0]['start_frame']==50


def test_truncated_lift_exports_partial_without_fabricating_hold():
    e={k:v[:160] for k,v in evidence().items()}
    r=segment_phases(e)
    assert r['status']=='partial'
    assert [s['name'] for s in r['segments']]==['grasp','lift']
    assert r['missing_stages']==['hold']
