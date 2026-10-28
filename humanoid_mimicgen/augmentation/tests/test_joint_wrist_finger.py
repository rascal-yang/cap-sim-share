import copy
import pytest
from humanoid_mimicgen.augmentation import finger_study as study
from humanoid_mimicgen.augmentation.tests.test_finger_press import params,NAMES

def fixture(monkeypatch):
 monkeypatch.setattr(study,'validate_decision',lambda *a:None)
 fixed={k:None for k in study.SCHEMA if k!='finger_control'}
 fixed.update(episode=0,action='run',preencoder={'reference_blend':0.,'translation_m':[0.,0.,-.01],'rotation_vector_rad':[0.,0.,0.],'ramp_s':.4,'force_band_N':[25.,35.],'force_gain_m_per_N_s':[0.,0.,-.0004],'max_feedback_m':.02,'feedback_rate_m_s':.01})
 d=copy.deepcopy(fixed);d['finger_control']=params();d['finger_control']['joints'].append({'name':NAMES[4],'offset_rad':.1,'force_gain_rad_per_N_s':0.})
 return fixed,d

def test_joint_mode_unlocks_wrist_only_and_requires_both_fingers(monkeypatch):
 fixed,d=fixture(monkeypatch);d['preencoder']['translation_m'][2]=-.02
 assert study.validate(d,fixed,NAMES,100,joint_wrist=True)==d
 with pytest.raises(ValueError):study.validate(d,fixed,NAMES,100)
 d['tail_s']=1
 with pytest.raises(ValueError):study.validate(d,fixed,NAMES,100,joint_wrist=True)

def test_joint_mode_rejects_out_of_bounds_wrist_and_single_finger(monkeypatch):
 fixed,d=fixture(monkeypatch);d['preencoder']['translation_m'][2]=-.2
 with pytest.raises(ValueError):study.validate(d,fixed,NAMES,100,joint_wrist=True)
 d['preencoder']=fixed['preencoder'];d['finger_control']['joints'].pop()
 with pytest.raises(ValueError):study.validate(d,fixed,NAMES,100,joint_wrist=True)
