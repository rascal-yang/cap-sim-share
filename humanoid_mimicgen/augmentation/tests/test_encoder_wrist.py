import numpy as np
import pytest
from humanoid_mimicgen.augmentation.encoder_wrist import transform_right,integrate_force,validate_parameters

def params():
 return {'reference_blend':0.,'translation_m':[0.,0.,-.02],'rotation_vector_rad':[0.,0.,0.], 'ramp_s':.5,'force_band_N':[20.,35.],'force_gain_m_per_N_s':[0.,0.,-.001],'max_feedback_m':.03,'feedback_rate_m_s':.02}

def test_only_right_wrist_changes_and_input_is_immutable():
 p=np.zeros((3,7));p[:,3]=1.;original=p.copy();ref=p[1].copy();ref[0]=.1
 result=transform_right(p,ref,params(),1.,np.array([0.,0.,-.01]))
 np.testing.assert_array_equal(p,original);np.testing.assert_array_equal(result[[0,2]],p[[0,2]])
 np.testing.assert_allclose(result[1,:3],[0,0,-.03]);np.testing.assert_array_equal(result[1,3:],p[1,3:])

def test_zero_weight_and_quaternion_convention():
 p=np.zeros((3,7));p[:,3]=1.;c=params();c['reference_blend']=1.;c['rotation_vector_rad']=[0.,0.,.2]
 r=p[1].copy();r[0]=.1
 np.testing.assert_array_equal(transform_right(p,r,c,0.,np.zeros(3)),p)
 z=transform_right(p,r,c,1.,np.zeros(3));np.testing.assert_allclose(z[1,3:],[np.cos(.1),0,0,np.sin(.1)],atol=1e-12)

def test_force_integral_direction_rate_and_bound():
 c=params();x=integrate_force(np.zeros(3),0.,c)
 np.testing.assert_allclose(x,[0,0,-.0004]);assert np.linalg.norm(integrate_force(np.array([0.,0.,-.03]),0.,c))<=.03+1e-12
 np.testing.assert_array_equal(integrate_force(x,25.,c),x)

def test_invalid_parameters_rejected():
 c=params();assert validate_parameters(c)==c
 for key,value in [('translation_m',[1.,0,0]),('max_feedback_m',float('nan')),('force_band_N',[35,20])]:
  bad=c|{key:value}
  with pytest.raises(ValueError):validate_parameters(bad)

def test_encoder_plan_rejects_postdecoder_or_unfixed_controls():
 import json,copy
 from pathlib import Path
 from humanoid_mimicgen.augmentation.encoder_wrist_study import validate_plan
 from humanoid_mimicgen.augmentation.contact_review import PROPS
 fixed={'action':'run','rationale':'test','hypothesis':'test','episode':0,'tail_s':2,'wrist_delta_m':[0,0,0],'contact_start':238,'contact_end':370,'safety_guard':None,'reference_case':'reference','criteria_decision':'audit','baseline_case':'baseline','resume':{'kind':'restart','case':'','frame':0,'rationale':'test'},'contact_control':{'joints':[]}}
 d=copy.deepcopy(fixed)|{'preencoder':params()}
 assert validate_plan(d,fixed)==d
 for key,value in [('wrist_delta_m',[.01,0,0]),('tail_s',3),('baseline_case','different'),('contact_control',{'joints':[{'name':'right_elbow_joint'}]})]:
  with pytest.raises(ValueError):validate_plan(d|{key:value},fixed)
