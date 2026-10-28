import numpy as np
import pytest
from humanoid_mimicgen.augmentation.leg_fast_retry import join_hands,require_boundary,arrays_exact

def test_merge_preserves_prefix_and_new_suffix():
 parent=np.arange(24).reshape(8,3);suffix=np.ones((3,3))*99
 out=join_hands(parent,suffix,5,8)
 assert np.array_equal(out[:5],parent[:5]) and np.array_equal(out[5:],suffix)
 with pytest.raises(ValueError):join_hands(parent,suffix,4,8)

def test_later_checkpoint_requires_integral_state_not_assumed_zero():
 require_boundary({'kind':'checkpoint','frame':352,'case':'/case'},352,'/case')
 with pytest.raises(ValueError):require_boundary({'kind':'checkpoint','frame':390,'case':'/case'},352,'/case')

def test_exact_gate_catches_suffix_divergence():
 x={'state':np.zeros((4,2)),'commands':np.zeros((4,2))};y={k:v.copy() for k,v in x.items()}
 assert arrays_exact(x,y)
 y['state'][-1,0]=1e-9
 assert not arrays_exact(x,y)

def test_reused_response_must_match_frozen_parent():
 from humanoid_mimicgen.augmentation.leg_fast_retry import require_same_model
 model={'inverse':[[1.,0.,0.]],'source_cases':['case']}
 require_same_model(model,model)
 with pytest.raises(ValueError):require_same_model({'inverse':[[2.,0.,0.]],'source_cases':['case']},model)
