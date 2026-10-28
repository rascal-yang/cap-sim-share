import numpy as np
import pytest
from humanoid_mimicgen.augmentation.leg_response import identify,advance,predict,paired

def test_identification_and_coupled_feedback():
 j=np.array([[1.,.2,0],[.1,.8,0],[0,.1,.5],[1,.2,0],[.1,.8,0],[0,.1,.5]])
 base=np.arange(6)*.1;eps=.05
 plus=np.array([base+j[:,i]*eps for i in range(3)]);minus=np.array([base-j[:,i]*eps for i in range(3)])
 model=identify(plus,minus,eps);assert np.allclose(model['jacobian'],j)
 want=np.array([.1,-.1,.05]);u,rate=advance(np.zeros(3),j@want,model['inverse'],.5,.3,1.,.02)
 assert np.allclose(u,want*.02/.5)

def test_singular_response_does_not_invent_control():
 m=identify(np.zeros((3,6)),np.zeros((3,6)),.05)
 assert m['rank']==0
 u,v=advance(np.zeros(3),np.ones(6),m['inverse'],.5,.3,1.,.02)
 assert np.array_equal(u,np.zeros(3))

def test_bounds_rate_and_forecast_derivative():
 u,rate=advance(np.array([.299,0,0]),np.ones(6)*10,np.ones((3,6)),.2,.3,.1,.02)
 assert np.all(u<=.3) and np.max(abs(rate))<=.1+1e-10
 q,v=predict(u,rate,np.array([0.,.1,1.]),.3)
 assert np.all(abs(q)<=.3)
 assert v[-1,0]==0
 assert np.array_equal(paired([1,2,3]),[1,2,3,1,2,3])

def test_nonfinite_model_rejected():
 with pytest.raises(ValueError):identify(np.full((3,6),np.nan),np.zeros((3,6)),.05)

def test_terminal_preview_uses_actual_clamped_times_and_zero_velocity():
 from humanoid_mimicgen.augmentation.leg_response import forecast
 ids,q,v=forecast(np.zeros(3),np.ones(3)*.1,97,100,.3)
 assert ids[1]==99 and np.allclose(q[1],.004)
 assert np.all(v[1:]==0)
 ids,q,v=forecast(np.array([.1,0,0]),np.ones(3)*.1,99,100,.3)
 assert np.allclose(q,np.tile([.1,0,0],(10,1))) and np.all(v==0)
