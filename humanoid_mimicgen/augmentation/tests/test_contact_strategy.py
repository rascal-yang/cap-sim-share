import pytest
from humanoid_mimicgen.augmentation.contact_strategy import force_hold_summary,validate_strategy

def test_hold_does_not_bridge_interrupted_force_or_geometry():
 rows=[{'force':21,'distance':0}]*30+[{'force':0,'distance':0}]+[{'force':21,'distance':0}]*30
 assert force_hold_summary(rows,20,.02,.001)['longest_s']==.6
 rows=[{'force':21,'distance':.002}]*60
 assert not force_hold_summary(rows,20,.02,.001)['meets_one_second']
 rows=[{'force':21,'distance':0}]*50
 assert force_hold_summary(rows,20,.02,.001)['meets_one_second']

def test_strategy_rejects_unknown_contact_body_and_false_native_claim():
 s={'contact_bodies':['palm'],'acceptance_assessment':'contact_omission','acceptance_reason':'native filter excludes palm body','change_kind':'reuse','expected_response':'hold','falsify':'no force'}
 assert validate_strategy(s,['palm'])==s
 with pytest.raises(ValueError):validate_strategy(s,['finger'])
 s['acceptance_assessment']='force_equals_native_success'
 with pytest.raises(ValueError):validate_strategy(s,['palm'])

def test_transfer_rejects_postdecoder_or_fixed_boundary_change(monkeypatch):
 from humanoid_mimicgen.augmentation import contact_strategy_transfer as t
 monkeypatch.setattr(t,'validate_decision',lambda *a:None)
 fixed={k:None for k in t.PROPS};fixed.update(action='run',episode=0,contact_start=10,contact_end=20,tail_s=0,contact_control={'joints':[]})
 import copy
 d=copy.deepcopy(fixed);d['strategy']={'contact_bodies':['finger'],'acceptance_assessment':'native_valid','acceptance_reason':'source passes','change_kind':'reuse','expected_response':'lift','falsify':'no lift'}
 assert t.validate(d,fixed,['finger'],30)==d
 d['contact_start']=11
 with pytest.raises(ValueError):t.validate(d,fixed,['finger'],30)
 d['contact_start']=10;d['contact_control']['joints']=['elbow']
 with pytest.raises(ValueError):t.validate(d,fixed,['finger'],30)

def test_contact_force_aggregation_distinguishes_points_from_body_total():
 from humanoid_mimicgen.augmentation.contact_strategy_diagnostics import aggregate_normal_forces
 rows=[{'body':'ankle','normal_N':93.98},{'body':'ankle','normal_N':80.0},{'body':'ankle','normal_N':68.99},{'body':'hand','normal_N':2.0}]
 total,single=aggregate_normal_forces(rows)
 assert total['ankle']==pytest.approx(242.97)
 assert single['ankle']==pytest.approx(93.98)
 assert total['hand']==2
