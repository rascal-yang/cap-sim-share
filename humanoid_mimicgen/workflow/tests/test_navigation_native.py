import mujoco
import numpy as np
from humanoid_mimicgen.workflow.navigation.native import NativeEvaluator


def fixture(ground_z=0):
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="ground" type="plane" size="4 4 .1"/>
      <body name="robot" pos="0 0 .1"><freejoint/><geom name="base" type="sphere" size=".1" mass="1"/></body>
      <body name="object" pos="2 0 .2"><geom name="obj" type="sphere" size=".1"/></body>
      <geom name="wall" type="box" pos="1 0 .2" size=".1 .3 .2"/>
      </worldbody></mujoco>'''.replace('name="ground"',f'name="ground" pos="0 0 {ground_z}"'))
    q=np.tile(m.qpos0,(3,1));q[:,0]=[0,.5,1.]
    prior={'source_stance_xyyaw':[0,0,0],'source_task_frame_xyzyaw':[2,0,.2,0],
           'operation_interval':[0,2],'key_indices':[0,2]}
    bindings={'root_body':'robot','object':'object','feet':['robot'],'ground_geoms':['ground'],
              'operation_bodies':['robot'],'contact_policy':[]}
    options={'path':{'resolution':.1,'padding':.3,'max_expansions':50},'collision_tolerance_m':.001,
             'corridor_margin_m':0.,'sample_translation_m':.05,'sample_angle_rad':.05,
             'max_checks':1000,'planning_timeout_s':2.,'support_distance_m':.005,'relation_tolerance_m':.03,'relation_tolerance_rad':.05}
    return m,q,prior,bindings,options


def test_later_operation_collision_rejects_reachable_entry():
    m,q,p,b,o=fixture();e=NativeEvaluator(m,m.qpos0,q,p,[2,0,.2,0],b,o)
    result=e({'id':'current','xyyaw':[0,0,0]})
    assert result['feasible'] is False
    assert result['reason']=='operation_collision'
    assert result['failed_source_index']>0


def test_budget_exhaustion_is_not_feasible():
    m,q,p,b,o=fixture();o['max_checks']=1
    r=NativeEvaluator(m,m.qpos0,q,p,[2,0,.2,0],b,o)({'id':'current','xyyaw':[0,0,0]})
    assert r['reason']=='planning_budget_exhausted'
    assert not r['feasible']


def test_environment_height_not_silently_applied_to_robot():
    m,q,p,b,o=fixture();q[:]=q[0]
    r=NativeEvaluator(m,m.qpos0,q,p,[2,0,1.,0],b,o)({'id':'current','xyyaw':[0,0,0]})
    assert r['reason']=='operation_relation_changed'


def test_clear_operation_produces_route_and_full_coverage():
    m,q,p,b,o=fixture();q[:]=q[0]
    r=NativeEvaluator(m,m.qpos0,q,p,[2,0,.2,0],b,o)({'id':'current','xyyaw':[0,0,0]})
    assert r['feasible']
    assert r['operation_checked']
    assert r['checked_source_indices']==[0,1,2]


def test_missing_ground_support_is_rejected():
    m,q,p,b,o=fixture(ground_z=-10);q[:]=q[0]
    r=NativeEvaluator(m,m.qpos0,q,p,[2,0,.2,0],b,o)({'id':'current','xyyaw':[0,0,0]})
    assert not r['feasible']


def test_deeply_buried_foot_is_not_valid_support():
    m,q,p,b,o=fixture(ground_z=.08);q[:]=q[0]
    assert not NativeEvaluator(m,m.qpos0,q,p,[2,0,.2,0],b,o)({'id':'current','xyyaw':[0,0,0]})['feasible']


def test_manipulated_object_cannot_cross_wall():
    xml='''<mujoco><worldbody><geom name="ground" type="plane" size="4 4 .1"/>
    <body name="robot" pos="0 0 .1"><freejoint/><geom name="base" type="sphere" size=".1" mass="1"/></body>
    <body name="object" pos="2 0 .3"><freejoint/><geom name="obj" type="sphere" size=".1" mass="1"/></body>
    <geom name="wall" type="box" pos="1 0 .3" size=".1 .3 .2"/></worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);_,_,p,b,o=fixture();q=np.tile(m.qpos0,(3,1));q[:,7]=[2,1,2]
    p['source_task_frame_xyzyaw']=[2,0,.3,0]
    r=NativeEvaluator(m,m.qpos0,q,p,[2,0,.3,0],b,o)({'id':'current','xyyaw':[0,0,0]})
    assert not r['feasible']
    assert r['reason']=='operation_collision'


def test_rejected_candidate_still_reports_actual_geometry_work():
    m,q,p,b,o=fixture();e=NativeEvaluator(m,m.qpos0,q,p,[2,0,.2,0],b,o)
    r=e({'id':'current','xyyaw':[0,0,0]})
    assert not r['feasible']
    assert r['geometry_checks']>0
