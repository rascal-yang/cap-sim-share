import numpy as np
from humanoid_mimicgen.stance_selection import rank_feasible, navigation_command


def test_reachability_precedes_facing_and_cost():
    rows=[{'id':'bad','feasible':False,'travel_m':0.,'turn_rad':0.,'facing_error_rad':0.}, {'id':'good','feasible':True,'travel_m':.5,'turn_rad':.2,'facing_error_rad':.4}]
    assert rank_feasible(rows)[0]['id']=='good'


def test_facing_is_only_preference_among_feasible():
    rows=[dict(id='side',feasible=True,travel_m=.5,turn_rad=.2,facing_error_rad=1.),dict(id='front',feasible=True,travel_m=.5,turn_rad=.2,facing_error_rad=.1)]
    assert rank_feasible(rows)[0]['id']=='front'


def test_navigation_speed_bounded_and_stops_at_target():
    v,yaw,reached=navigation_command([0,0,0],[1,1,.3])
    assert np.linalg.norm(v)<=.150000001 and not reached
    v,yaw,reached=navigation_command([1,1,.3],[1,1,.3])
    assert reached and np.array_equal(v,[0.,0.])
