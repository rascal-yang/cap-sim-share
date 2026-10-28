import numpy as np
from humanoid_mimicgen.navigation_reference import body_future_from_planner


def test_live_planner_reference_is_sampled_at_correct_future_rate():
    class Fake:
        ref=None;tick=10;ref_tick=0
        def plan_legs(self,d,v,h,y):self.called=(d,v,h,y);self.ref=np.zeros((100,36));self.ref_tick=self.tick
        def sample(self,f):return np.repeat(np.asarray(f)[:,None],36,axis=1)
    s=Fake();ref,future=body_future_from_planner(s,'state',np.array([.1,0]),.78,.2)
    assert s.called[0]=='state'
    np.testing.assert_allclose(ref[:,0],np.arange(10)*3)
    np.testing.assert_allclose(future[:,0]-ref[:,0],.3)
