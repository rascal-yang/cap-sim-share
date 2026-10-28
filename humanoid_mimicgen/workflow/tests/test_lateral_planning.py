import copy
import json
import numpy as np
import mujoco
import pytest
from humanoid_mimicgen.workflow.contracts import WorkflowError


def mechanism(table_collision=False):
    def limb(name,pos):
        size=.0201 if name in ('lf','rf') else .02
        return f'<body name="{name}" pos="{pos}"><joint name="{name}_x" type="slide" axis="1 0 0" range="-1 1"/><joint name="{name}_y" type="slide" axis="0 1 0" range="-1 1"/><joint name="{name}_z" type="slide" axis="0 0 1" range="-1 1"/><geom type="sphere" size="{size}" mass=".1"/></body>'
    xml='<mujoco><option timestep=".005"/><worldbody><geom name="ground" type="plane" size="2 2 .1"/>'
    xml+=('<geom name="table" type="box" pos=".6 0 .6" size=".4 .4 .05"/>' if table_collision else
          '<geom name="table" type="box" pos=".6 0 .3" size=".2 .4 .05"/>')
    xml+='<body name="pelvis" pos="0 0 .8"><freejoint/><geom size=".05" mass="1"/>'
    xml+=limb('lf','0 .1 -.78')+limb('rf','0 -.1 -.78')+limb('lw','.2 .2 -.2')+limb('rw','.35 -.1 -.2')+'</body>'
    xml+='<body name="apple" pos=".55 0 .4"><freejoint/><geom name="apple_geom" size=".05" mass=".1"/></body></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);q=np.tile(m.qpos0,(5,1));q[:,0]=np.linspace(0,.01,5)
    for foot in ['lf','rf']:q[:,m.jnt_qposadr[m.joint(foot+'_x').id]]=-q[:,0]
    d.qpos[:]=q[0];mujoco.mj_forward(m,d)
    feet={name:np.r_[d.xpos[m.body(name).id],d.xquat[m.body(name).id]].tolist() for name in ['lf','rf']}
    bindings=dict(root_body='pelvis',feet=['lf','rf'],wrists=dict(left='lw',right='rw'),object='apple',
                  ground_geoms=['ground'],support_geoms=['table'],operation_bodies=['rw'],
                  contact_policy=[dict(start=0,end=4,allowed_geom_pairs=[['apple_geom','table']])],
                  navigation_allowed_geom_pairs=[['apple_geom','table']])
    prior=dict(operation_interval=[0,4],source_stance_xyyaw=[0,0,0],source_task_frame_xyzyaw=[.55,0,.4,0],
               source_initial_object_pose=[.55,0,.4,1,0,0,0],key_indices=[0,2,4],
               surface=dict(normal=[1,0,0],tangent=[0,1,0],geom='table'),
               support_windows=[dict(feet=feet,root_pose=q[0,:7].tolist(),representative_row=0,qpos=q[0].tolist())])
    native=dict(path=dict(resolution=.04,padding=.5,max_expansions=1000),collision_tolerance_m=.001,support_distance_m=.003,
                corridor_margin_m=0.,sample_translation_m=.02,sample_angle_rad=.04,max_checks=50000,planning_timeout_s=20.,
                relation_tolerance_m=.003,relation_tolerance_rad=.03)
    options=dict(native=native,ik=dict(max_nfev=50,key_stride=2,position_m=.003,angle_rad=.03,foot_position_m=.002,foot_angle_rad=.02,
        joint_speed_rad_s=10.,root_speed_m_s=1.,timeout_s=15.,posture_weight=.001,bridge_duration_s=2.,bridge_samples=11),
        search=dict(step_m=.025,max_candidates=21,planning_timeout_s=60.))
    candidate=dict(id='source',feet=feet,xyyaw=[0,0,0],root_pose=q[0,:7].tolist(),seed_qpos=q[0].tolist(),requires_navigation=False)
    return m,q,prior,bindings,options,candidate


def test_retarget_changes_wrist_without_moving_feet_or_freezing_pelvis():
    from humanoid_mimicgen.workflow.navigation.lateral_ik import LateralSolver
    m,q,p,b,o,c=mechanism();solver=LateralSolver(m,q,np.arange(5)*.1,p,b,o)
    result=solver.solve(c,np.array([0,.04,0]),q[0])
    assert result['feasible'],result
    d=mujoco.MjData(m)
    for i,new in enumerate(result['qpos']):
        d.qpos[:]=new;mujoco.mj_forward(m,d)
        assert d.xpos[m.body('lf').id][:2]==pytest.approx([0,.1],abs=.002)
        assert d.xpos[m.body('rw').id][1]==pytest.approx(-.06,abs=.003)
    assert np.ptp(result['qpos'][:,0])>.005


def test_lateral_candidates_keep_depth_height_and_limit_count():
    from humanoid_mimicgen.workflow.navigation.lateral import lateral_candidates
    m,q,p,b,o,c=mechanism()
    rows,truncated=lateral_candidates(p,c['feet'],q[0],.1,dict(step_m=.025,max_candidates=4))
    assert len(rows)==4 and truncated
    assert rows[0]['id']=='current'
    assert all(r['feet']['lf'][0]==0 and r['feet']['lf'][2]==pytest.approx(.02) for r in rows)


def test_depth_shift_and_excess_speed_rejected():
    from humanoid_mimicgen.workflow.navigation.lateral_ik import LateralSolver
    m,q,p,b,o,c=mechanism()
    with pytest.raises(WorkflowError):LateralSolver(m,q,np.arange(5)*.1,p,b,o).solve(c,np.array([.1,0,0]),q[0])
    o['ik']['root_speed_m_s']=.001
    result=LateralSolver(m,q,np.arange(5)*.1,p,b,o).solve(c,np.zeros(3),q[0])
    assert not result['feasible'] and result['reason']=='reference_speed_limit'


def test_reference_bundle_rejects_mutation_and_duplicate_rows(tmp_path):
    from humanoid_mimicgen.workflow.navigation.reference_bundle import save_reference_bundle,load_reference_bundle
    arrays=dict(qpos=np.tile([0,0,.8,1,0,0,0],(3,1)),time_s=np.array([1.,1.1,1.2]),source_rows=np.array([10,11,12]),
                fingers=np.zeros((3,14)),bridge_qpos=np.tile([0,0,.8,1,0,0,0],(2,1)),bridge_time_s=np.array([0.,1.]))
    ref=save_reference_bundle(tmp_path/'bundle',arrays,dict(operation_interval=[10,12],source_evidence_sha256={}))
    assert load_reference_bundle(ref)['source_rows'].tolist()==[10,11,12]
    with open(ref['arrays_path'],'ab') as f:f.write(b'mutation')
    with pytest.raises(WorkflowError):load_reference_bundle(ref)
    arrays['source_rows'][1]=10
    with pytest.raises(WorkflowError):save_reference_bundle(tmp_path/'bad',arrays,dict(operation_interval=[10,12]))


def test_off_table_offset_does_not_create_a_scene():
    from humanoid_mimicgen.workflow.navigation.lateral import shifted_initial,surface_frame
    m,q,p,b,o,c=mechanism();b['lateral_surface']=dict(geom='table',normal_local=[1,0,0]);p['surface']=surface_frame(m,q[0],b)
    moved,delta=shifted_initial(m,q[0],p,b,.1)
    assert delta.tolist()==[0,.1,0]
    assert moved[-7:][:3]==pytest.approx([.55,.1,.4])
    with pytest.raises(WorkflowError,match='footprint'):shifted_initial(m,q[0],p,b,.5)


def test_ik_budget_and_idle_hand_collision_are_rejected():
    from humanoid_mimicgen.workflow.navigation.lateral_ik import LateralSolver
    m,q,p,b,o,c=mechanism();o['ik']['timeout_s']=1e-12
    assert LateralSolver(m,q,np.arange(5)*.1,p,b,o).solve(c,np.zeros(3),q[0])['reason']=='planning_budget_exhausted'
    m,q,p,b,o,c=mechanism(table_collision=True)
    r=LateralSolver(m,q,np.arange(5)*.1,p,b,o).solve(c,np.zeros(3),q[0])
    assert not r['feasible'] and r['reason']=='operation_collision'


def test_actual_apple_mesh_support_not_oversized_bounding_sphere():
    from pathlib import Path
    from humanoid_mimicgen.workflow.navigation.lateral import shifted_initial,surface_frame
    manifest=json.loads((Path(__file__).parent/'fixtures/navigation/apple_lateral.json').read_text())
    if not Path(manifest['files']['model']).exists():pytest.skip('Local frozen source unavailable')
    m=mujoco.MjModel.from_binary_path(manifest['files']['model']);q=np.load(manifest['files']['initial'])['qpos']
    b=manifest['bindings'];d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);obj=m.body(b['object']).id
    p=dict(surface=surface_frame(m,q,b),source_initial_object_pose=np.r_[d.xpos[obj],d.xquat[obj]].tolist())
    unchanged,delta=shifted_initial(m,q,p,b,0.)
    np.testing.assert_array_equal(unchanged,q)


def test_joint_limit_and_optimizer_exhaustion_cannot_certify_infeasibility():
    from humanoid_mimicgen.workflow.navigation.lateral_ik import LateralSolver
    m,q,p,b,o,c=mechanism();q[:,m.jnt_qposadr[m.joint('rw_x').id]]=1.1
    r=LateralSolver(m,q,np.arange(5)*.1,p,b,o).solve(c,np.zeros(3),q[0])
    assert not r['feasible']
    m,q,p,b,o,c=mechanism();o['ik']['max_nfev']=1
    r=LateralSolver(m,q,np.arange(5)*.1,p,b,o).solve(c,np.array([0,.04,0]),q[0])
    assert r['reason']=='planning_budget_exhausted'


def test_object_aligned_candidate_is_not_delayed_by_intermediate_grid():
    from humanoid_mimicgen.workflow.navigation.lateral import lateral_candidates
    m,q,p,b,o,c=mechanism()
    rows,_=lateral_candidates(p,c['feet'],q[0],.2,dict(step_m=.025,max_candidates=21))
    assert rows[1]['feet']['lf'][1]==pytest.approx(.3) # current is identical to source template


def test_pose_jacobian_matches_independent_finite_difference():
    from humanoid_mimicgen.workflow.navigation.lateral_ik import pose_residual_jacobian
    m,q,p,b,o,c=mechanism();d=mujoco.MjData(m);d.qpos[:]=q[0];mujoco.mj_forward(m,d)
    names=['lf','rf','lw','rw'];ids={n:m.body(n).id for n in names}
    goals={n:np.r_[d.xpos[i]+[0,.02,0],d.xquat[i]] for n,i in ids.items()};dofs=np.arange(m.nv-6)
    error,jac=pose_residual_jacobian(m,d,goals,ids,dofs)
    numerical=np.zeros_like(jac)
    for k,j in enumerate(dofs):
        values=[]
        for sign in [-1,1]:
            d.qpos[:]=q[0];v=np.zeros(m.nv);v[j]=sign*1e-6;mujoco.mj_integratePos(m,d.qpos,v,1.);mujoco.mj_forward(m,d)
            values.append(pose_residual_jacobian(m,d,goals,ids,dofs)[0])
        numerical[:,k]=(values[1]-values[0])/2e-6
    np.testing.assert_allclose(jac,numerical,atol=2e-5)


def test_single_supported_foot_cannot_certify_planted_operation():
    from humanoid_mimicgen.workflow.navigation.lateral_ik import LateralSolver
    m,q,p,b,o,c=mechanism();q[1:,m.jnt_qposadr[m.joint('lf_z').id]]+=.0015
    r=LateralSolver(m,q,np.arange(5)*.1,p,b,o).solve(c,np.zeros(3),q[0])
    assert not r['feasible'] and r['reason']=='operation_support_lost'


def test_replay_first_nominal_translates_source_and_limits_nearby_fallbacks():
    from humanoid_mimicgen.workflow.navigation.lateral import replay_candidates
    m,q,p,b,o,c=mechanism()
    rows=list(replay_candidates(p,c['feet'],q[0],.1,dict(step_m=.025,max_candidates=3)))
    assert len(rows)==3
    assert [r['lateral_shift_m'] for r in rows]==pytest.approx([.1,.075,.125])
    assert rows[0]['feet']['lf'][1]==pytest.approx(.2)
    assert all(r['root_pose'][0]==q[0,0] and r['root_pose'][2]==q[0,2] for r in rows)


def test_reuse_preserves_nonkeyframe_source_motion_exactly():
    from humanoid_mimicgen.workflow.navigation.lateral_ik import LateralSolver
    m,q,p,b,o,c=mechanism();o['ik']['reuse_source_first']=True
    q[1,0]+=.0003
    for foot in ['lf','rf']:q[1,m.jnt_qposadr[m.joint(foot+'_x').id]]=-q[1,0]
    solver=LateralSolver(m,q,np.arange(5)*.1,p,b,o)
    r=solver.solve(c,np.zeros(3),q[0])
    assert r['feasible'],r
    assert r['reference_reused_without_ik']
    np.testing.assert_array_equal(r['qpos'],q)
    assert r['reference_modification']['max_body_joint_correction_rad']==0


def test_small_change_limit_rejects_large_joint_correction():
    from humanoid_mimicgen.workflow.navigation.lateral_ik import LateralSolver
    m,q,p,b,o,c=mechanism();o['ik']['max_body_joint_correction_rad']=.005
    r=LateralSolver(m,q,np.arange(5)*.1,p,b,o).solve(c,np.array([0,.04,0]),q[0])
    assert not r['feasible'] and r['reason']=='reference_correction_limit'
