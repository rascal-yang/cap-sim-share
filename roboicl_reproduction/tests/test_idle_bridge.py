from types import SimpleNamespace
import numpy as np
import pytest
from cap_sim.backends.native_sonic import Sonic
from repro.idle_bridge import StandingIdlePlanner

class SessionBoundary:
    def run(self, outputs, feed):
        self.feed = {k:v.copy() for k,v in feed.items()}
        self.original = feed
        return np.zeros((1,4,36)), np.array([4])

def execute(velocity, height, enabled=True):
    sonic = Sonic.__new__(Sonic)
    sonic.ref=None; sonic.tick=0; sonic.serial=0; sonic.qids=np.arange(7,36)
    boundary=SessionBoundary()
    sonic.planner=StandingIdlePlanner(boundary, enabled=enabled)
    d=SimpleNamespace(qpos=np.r_[0,0,.78,1,0,0,0,np.zeros(29)])
    sonic.plan_legs(d,velocity,height,.1)
    return boundary.feed, sonic.planner

def test_standing_zero_velocity_reaches_idle_not_squat():
    feed, adapter=execute([0,0],.78)
    assert feed['mode'].tolist()==[0]
    assert feed['movement_direction'].tolist()==[[0,0,0]]
    assert feed['height'].tolist()==[-1]
    assert feed['target_vel'].tolist()==[-1]

@pytest.mark.parametrize('velocity,height', [([.25,0],.78),([.01,0],.78),([0,0],.6)])
def test_walking_and_requested_squat_preserve_complete_native_feed(velocity,height):
    expected,_=execute(velocity,height,False)
    actual,_=execute(velocity,height,True)
    for key in expected:
        np.testing.assert_array_equal(actual[key],expected[key])
        assert actual[key].dtype==expected[key].dtype

def test_correction_does_not_mutate_caller_and_records_actual_input():
    feed,adapter=execute([0,0],.78,False)
    original={k:v.copy() for k,v in feed.items()}
    adapter.enabled=True
    adapter.run(None,feed)
    for key in original: np.testing.assert_array_equal(feed[key],original[key])
    assert adapter.records[-1]['before']['mode']==[4]
    assert adapter.records[-1]['after']['mode']==[0]
    assert adapter.records[-1]['transformed'] is True
