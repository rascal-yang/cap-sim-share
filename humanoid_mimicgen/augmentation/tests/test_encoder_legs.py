import numpy as np
import pytest
from humanoid_mimicgen.augmentation.encoder_legs import transform, JOINTS

def fixture():
 q=np.zeros((120,12));v=np.ones_like(q)*.03;ref=q.copy();ref[:,[0,3,4,6,9,10]]=.1
 return q,v,ref

def test_zero_is_exact_identity():
 q,v,ref=fixture();a,b=transform(q,v,ref,[0,3,4,6,9,10],[0]*6,[0]*6,20,.4)
 assert np.array_equal(a,q) and np.array_equal(b,v)

def test_selected_reference_and_velocity_consistent():
 q,v,ref=fixture();ids=[0,3,4,6,9,10];a,b=transform(q,v,ref,ids,[.8]*6,[.02]*6,20,.4)
 assert np.allclose(a[-1,ids],.1)
 assert np.array_equal(a[:20],q[:20]) and np.array_equal(b[:20],v[:20])
 other=[1,2,5,7,8,11];assert np.array_equal(a[:,other],q[:,other]);assert np.array_equal(b[:,other],v[:,other])
 assert np.allclose(b[21:,ids]-v[21:,ids],np.gradient((a-q)[:,ids],.02,axis=0)[21:])
 assert np.max(np.abs(b-v))<2

@pytest.mark.parametrize('blend,offset,ramp', [([2]*6,[0]*6,.4),([0]*6,[float('nan')]*6,.4),([0]*6,[.31]*6,.4),([0]*6,[0]*6,0)])
def test_invalid_parameters_rejected(blend,offset,ramp):
 q,v,ref=fixture()
 with pytest.raises(ValueError):transform(q,v,ref,[0,3,4,6,9,10],blend,offset,20,ramp)

def test_excessive_reference_correction_is_rejected():
 q,v,ref=fixture();ref[:]=10
 with pytest.raises(ValueError):transform(q,v,ref,[0,3,4,6,9,10],[1]*6,[0]*6,20,.4)

def test_unexecuted_tracking_is_unavailable():
 from humanoid_mimicgen.augmentation.encoder_legs import tracking_rmse
 assert tracking_rmse(np.zeros((5,6)),np.zeros((5,6)),5) is None
 assert tracking_rmse(np.zeros((0,6)),np.zeros((0,6)),0) is None
 assert tracking_rmse(np.ones((5,6)),np.zeros((5,6)),2)==[1.]*6

def test_execution_metadata_describes_observer_only():
 from humanoid_mimicgen.augmentation.encoder_legs import execution_metadata
 r={'control_variant':'mode1_plus_explicit_joint_contact_residual','token_alone_replays_execution':False}
 execution_metadata(r)
 assert r['post_decoder_residual_calls']==0 and r['contact_observer_only']
 assert r['token_alone_replays_execution']=='pending_independent_replay'

def test_observer_contract_rejects_actual_joint_corrections():
 from humanoid_mimicgen.augmentation.encoder_legs import observer_contract
 with pytest.raises(ValueError):observer_contract({'contact_control':{'joints':[{'name':'knee'}]}})
 c=observer_contract({'interface':'residual','contact_control':{'joints':[]},'token_alone_replays_execution':False})
 assert c['postdecoder_corrections'] is False
 assert c['token_alone_replays_execution']=='pending_independent_replay'
