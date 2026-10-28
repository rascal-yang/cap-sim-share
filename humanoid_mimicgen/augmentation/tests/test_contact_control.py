import numpy as np
import pytest
from humanoid_mimicgen.augmentation.contact_control import bounded_servo, blend_targets, validate_contact_control


def test_force_feedback_sign_rate_and_integrator_bounds():
    assert bounded_servo(0, 0, [20, 30], .01, .02, .1, .2) == pytest.approx(.002)
    assert bounded_servo(.2, 0, [20, 30], .01, .02, .1, .2) == .2
    assert bounded_servo(.1, 40, [20, 30], .01, .02, .1, .2) == pytest.approx(.098)
    assert bounded_servo(.1, 25, [20, 30], .01, .02, .1, .2) == .1


def test_reference_blend_preserves_prefix_and_unselected_joints():
    q=np.zeros(3);ref=np.ones(3)
    assert np.array_equal(blend_targets(q,ref,[1],1,0),q)
    assert np.array_equal(blend_targets(q,ref,[1],.5,1),[0,.5,0])


def valid():
    return dict(contact_bodies=['pelvis'],force_range_N=[10,30],ramp_s=.4,
        joints=[dict(name='knee',replay_blend=.5,offset_rad=.05,kp_scale=1.,kd_scale=1.,force_gain_rad_per_N_s=.001)],
        max_offset_rad=.2,max_rate_rad_s=.1,force_stop_N=400)


def test_contract_rejects_unknown_contact_and_joint():
    d=valid();validate_contact_control(d,['pelvis'],['knee'])
    with pytest.raises(ValueError):validate_contact_control(d,['torso'],['knee'])
    with pytest.raises(ValueError):validate_contact_control(d,['pelvis'],['hip'])

@pytest.mark.parametrize('field,value',[('force_range_N',[30,10]),('ramp_s',0),('force_stop_N',20),('max_offset_rad',float('nan'))])
def test_bad_contract_rejected(field,value):
    with pytest.raises(ValueError):validate_contact_control(valid()|{field:value},['pelvis'],['knee'])


def test_reviewed_success_cannot_silently_replace_native():
    from humanoid_mimicgen.augmentation.contact_review import quality
    result={'success':False,'independent_success':False,'reviewed_success':True,'reviewed_independent_success':True,'coverage':1.,'failure':None}
    assert not quality(result,{'assessment':'native_valid'})
    assert quality(result,{'assessment':'propose_contact_revision'})
    assert not quality(result|{'failure':'guard'}, {'assessment':'propose_contact_revision'})


def test_criteria_audit_rejects_missing_evidence_fields_and_nan():
    from humanoid_mimicgen.augmentation.contact_review import validate_audit
    a={'assessment':'native_valid','rationale':'measured contact absent','contact_bodies':['pelvis'],'min_contact_force_N':1}
    assert validate_audit(a,['pelvis'])==a
    with pytest.raises(ValueError):validate_audit(a|{'min_contact_force_N':float('nan')},['pelvis'])
    with pytest.raises(ValueError):validate_audit(a,['foot'])
