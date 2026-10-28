import numpy as np
from humanoid_mimicgen.reachability_navigation import plan_base_path, choose_plan, path_command

def test_detour_around_obstacle_checks_edges():
    def valid(p): return not (.3 <= p[0] <= .7 and abs(p[1]) < .16)
    path, audit=plan_base_path([0,0,0],[1,0,0],valid,resolution=.1,padding=.4)
    assert path is not None and audit['method']=='astar'
    assert np.max(abs(path[:,1])) >= .16
    for a,b in zip(path,path[1:]):
        assert all(valid(a+u*(b-a)) for u in np.linspace(0,1,51))

def test_blocked_path_is_rejected():
    path,_=plan_base_path([0,0,0],[1,0,0],lambda p: p[0]<.4,resolution=.1,padding=.2)
    assert path is None

def test_current_stationary_plan_has_priority_over_facing():
    rows=[{'id':'current','feasible':True,'requires_navigation':False,'score':10}, {'id':'other','feasible':True,'requires_navigation':True,'score':0}]
    assert choose_plan(rows)['id']=='current'

def test_infeasible_current_does_not_skip_navigation():
    rows=[{'id':'current','feasible':False,'requires_navigation':False,'score':0}, {'id':'other','feasible':True,'requires_navigation':True,'score':1}]
    assert choose_plan(rows)['id']=='other'

def test_waypoint_navigation_does_not_cut_across_detour():
    path=np.array([[0,0,0],[0,.4,0],[1,.4,0],[1,0,0]])
    v,yaw,stop,index=path_command([0,0,0],path,1)
    assert np.allclose(v,[0,.5]) and not stop and index==1

def test_endpoint_requests_stop_without_claiming_reachability():
    v,_,stop,_=path_command([.95,0,0],np.array([[0,0,0],[1,0,0]]),1)
    assert stop and np.allclose(v,0)
