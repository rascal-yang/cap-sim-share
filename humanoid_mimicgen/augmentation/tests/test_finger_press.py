import numpy as np
import pytest
from humanoid_mimicgen.augmentation.finger_press import FingerTargets,validate_fingers,validate_saved_hands
NAMES=['right_hand_thumb_0_joint','right_hand_thumb_1_joint','right_hand_thumb_2_joint','right_hand_index_0_joint','right_hand_index_1_joint','right_hand_middle_0_joint','right_hand_middle_1_joint']

def params():return {'start_frame':10,'ramp_s':.2,'max_offset_rad':.8,'max_rate_rad_s':1.,'force_band_N':[25.,35.],'joints':[{'name':NAMES[3],'offset_rad':.2,'force_gain_rad_per_N_s':.01}]}

def test_preserves_prefix_and_all_unselected_commands():
 c=FingerTargets(params(),NAMES,np.tile([0.,1.7],(7,1)));source=np.zeros(7,dtype=np.float32)
 for i in range(11):np.testing.assert_array_equal(c.apply(source,i,0.),source)
 x=c.apply(source,11,0.);assert 0<x[3]<=.020001;np.testing.assert_array_equal(x[[0,1,2,4,5,6]],source[[0,1,2,4,5,6]]);np.testing.assert_array_equal(source,np.zeros(7))

def test_saturates_joint_offset_and_rate():
 c=FingerTargets(params(),NAMES,np.tile([0.,.1],(7,1)));prev=np.zeros(7)
 for i in range(100):
  x=c.apply(np.zeros(7),i,0.);assert x[3]<=.100001;assert abs(x[3]-prev[3])<=.020001;prev=x

def test_rejects_other_fingers_and_invalid_saved_commands():
 c=params();c['joints'][0]['name']=NAMES[0]
 with pytest.raises(ValueError):validate_fingers(c,NAMES,100)
 with pytest.raises(ValueError):validate_saved_hands(np.zeros((2,7)),np.zeros((3,7)),3)
 with pytest.raises(ValueError):validate_saved_hands(np.full((3,7),np.nan),np.zeros((3,7)),3)
 l,r=validate_saved_hands(np.zeros((3,7)),np.ones((3,7)),3);assert l.dtype==r.dtype==np.float32
