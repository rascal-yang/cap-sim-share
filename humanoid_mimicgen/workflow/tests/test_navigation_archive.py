import json
import numpy as np
import mujoco
import pytest
from humanoid_mimicgen.workflow.navigation.archive import load_archive, verify_evidence
from humanoid_mimicgen.workflow.contracts import WorkflowError


def archive(tmp_path):
    xml='''<mujoco><worldbody><geom name="ground" type="plane" size="2 2 .1"/>
    <body name="pelvis" pos="0 0 .1"><freejoint/><geom name="foot" type="sphere" size=".1" mass="1"/></body>
    <body name="object" pos="1 0 .1"><geom type="sphere" size=".1"/></body></worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);mujoco.mj_saveModel(m,str(tmp_path/'scene.mjb'),None)
    q=np.tile(m.qpos0,(30,1));state=np.c_[np.arange(30)*.02,q,np.zeros((30,m.nv))]
    np.savez(tmp_path/'trajectory.npz',state=state)
    (tmp_path/'result.json').write_text(json.dumps({'native_success':True,'task':'example'}))
    manifest={'files':{'model':'scene.mjb','trajectory':'trajectory.npz','result':'result.json'},
        'bindings':{'object':'object','root_body':'pelvis','feet':['pelvis'],'ground_geoms':['ground'],
                    'frame':{'kind':'world_heading','yaw_rad':0.,'evidence':'test table'}},
        'entry_index':20,'end_index':29,'state_key':'state','support_distance_m':.002,
        'success_field':'native_success','success_criterion':'native'}
    p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest));return p


def test_load_real_native_state_layout_and_readonly_hashes(tmp_path):
    path=archive(tmp_path);trace,bindings,evidence=load_archive(path)
    assert trace['root_pose'].shape==(30,7)
    assert trace['support_ok'].all()
    assert trace['support_evidence_kind']=='geometric_contact_not_force'
    assert trace['success'] is True
    verify_evidence(evidence)
    (tmp_path/'result.json').write_text('{}')
    with pytest.raises(WorkflowError,match='changed'):verify_evidence(evidence)


def test_out_of_range_phase_rejected_before_fk(tmp_path):
    p=archive(tmp_path);d=json.loads(p.read_text());d['entry_index']=31;p.write_text(json.dumps(d))
    with pytest.raises(WorkflowError):load_archive(p)
