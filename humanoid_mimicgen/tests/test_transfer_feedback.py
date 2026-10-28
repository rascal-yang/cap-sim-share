import numpy as np
from types import SimpleNamespace
from collections import deque


def test_original_token_decoder_receives_current_state_history():
    from humanoid_mimicgen.transfer_feedback import decode_prefix_token
    class Decoder:
        def get_inputs(self):return [SimpleNamespace(name='obs')]
        def run(self,_,feed):
            self.actual=feed['obs'].copy();return [np.full((1,29),.25,np.float32)]
    state=SimpleNamespace(qpos=np.r_[0.,0.,.8,1.,0.,0.,0.,np.ones(29)*.1],qvel=np.arange(35)*.01,time=0.)
    history=deque([dict(omega=np.zeros(3),q=np.zeros(29),dq=np.zeros(29),last=np.zeros(29),gravity=np.array([0,0,-1])) for _ in range(10)],maxlen=10)
    sonic=SimpleNamespace(history=history,qids=np.arange(7,36),vids=np.arange(6,35),default=np.zeros(29),order=np.arange(29),reverse=np.arange(29),scale=np.ones(29)*2,last=np.ones(29),decoder=Decoder(),tick=0,records=[])
    target=decode_prefix_token(sonic,state,np.zeros(64))
    np.testing.assert_allclose(target,.5)
    np.testing.assert_allclose(sonic.history[-1]['q'],.1)
    np.testing.assert_allclose(sonic.history[-1]['last'],1.)
    np.testing.assert_allclose(sonic.last,.25)
    assert sonic.tick==1 and sonic.records[-1]['mode']=='original_token_prefix'
