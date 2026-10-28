from collections import deque
import pytest
np=pytest.importorskip('numpy')
pytest.importorskip('scipy')
from cap_sim.backends.native_sonic import Sonic


def test_full_body_encoder_uses_scalar_mode_and_isaac_order():
 class Session:
  def __init__(self,n):self.n=n;self.input=None
  def get_inputs(self):return [type('Input',(),{'name':'x'})()]
  def run(self,_,feed):self.input=feed['x'];return [np.zeros((1,self.n),np.float32)]
 s=Sonic.__new__(Sonic);s.encoder=Session(64);s.decoder=Session(29);s.qids=np.arange(7,36);s.vids=np.arange(6,35);s.order=np.arange(28,-1,-1);s.reverse=np.arange(29);s.default=np.zeros(29);s.scale=np.ones(29);s.last=np.zeros(29);s.tick=0;s.records=[]
 h={'omega':np.zeros(3),'q':np.zeros(29),'dq':np.zeros(29),'last':np.zeros(29),'gravity':np.array([0,0,-1])};s.history=deque([h]*10,maxlen=10)
 d=type('Data',(),{'qpos':np.r_[0,0,.8,1,0,0,0,np.zeros(29)],'qvel':np.zeros(35),'time':0})()
 ref=np.tile(np.r_[0,0,.8,1,0,0,0,np.arange(29)/100],(10,1))
 s.step(d,[],None,[0,0],.74,(ref,ref))
 e=s.encoder.input[0]
 assert e[:4].tolist()==[0,0,0,0]
 np.testing.assert_allclose(e[4:33],np.arange(28,-1,-1)/100)
 assert np.all(e[294:584]==0)
 assert s.decoder.input.shape==(1,994)


def test_native_reward_cadence_preserves_one_second_hold():
 from cap_sim.backends.humano import reward_stride
 stride=reward_stride(.02,.005)
 assert stride==4
 ticks=[i for i in range(1,201) if i%stride==0]
 assert len(ticks)==50 and ticks[-1]*.005==1.
 with pytest.raises(ValueError):reward_stride(.013,.005)
