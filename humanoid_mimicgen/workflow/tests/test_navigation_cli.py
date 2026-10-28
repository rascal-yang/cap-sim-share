import json
from pathlib import Path
import pytest
from humanoid_mimicgen.workflow.navigation.__main__ import main
from humanoid_mimicgen.workflow.navigation.integration import validate_bound_plan
from humanoid_mimicgen.workflow.contracts import WorkflowError
from humanoid_mimicgen.workflow.tests.test_navigation_archive import archive


def test_extract_cli_no_overwrite_and_missing_input(tmp_path,capsys):
    p=archive(tmp_path);opt=tmp_path/'options.json'
    opt.write_text(json.dumps(dict(window_s=.1,lookback_s=.3,speed_m_s=.05,yaw_speed_rad_s=.1,drift_m=.02)))
    out=tmp_path/'prior.json';args=['extract','--manifest',str(p),'--options',str(opt),'--out',str(out)]
    assert main(args)==0
    result=json.loads(out.read_text());assert result['status']=='completed'
    assert result['support_evidence_kind']=='geometric_contact_not_force'
    assert main(args)!=0
    assert main(['extract','--manifest',str(tmp_path/'missing'),'--options',str(opt),'--out',str(out)])!=0


def test_unbound_plan_cannot_execute(tmp_path):
    with pytest.raises(WorkflowError):validate_bound_plan({'status':'completed'}, {}, {})


def test_execute_refuses_unfrozen_contract(tmp_path):
    plan=tmp_path/'plan.json';plan.write_text('{}')
    assert main(['execute','--plan',str(plan),'--experiment',str(tmp_path)])!=0


def test_bound_plan_checks_scene_source_phase_and_corridor(tmp_path):
    from humanoid_mimicgen.workflow.contracts import file_hash
    files={}
    for name in ('model','initial','reference_states'):
        p=tmp_path/name;p.write_text(name);files[name]=str(p)
    plan={'kind':'replay_navigation_plan','status':'completed','selected':{'operation_checked':True},
          'evidence_sha256':{v:file_hash(v) for v in files.values()},
          'scene_input_sha256':{k:file_hash(v) for k,v in files.items()},
          'source_trajectory_sha256':file_hash(files['reference_states']),
          'prior':{'operation_interval':[20,29],'bindings':{'object':'object'}},
          'native_options':{'corridor_margin_m':.1}}
    strategy={'phases':[{'start':20}],'command_pose_offset':0,'bindings':{'object':'object'},
              'policy':{'navigation':{'profile':'replay_navigation','navigation_only':True,'route_limits':{'corridor_radius_m':.08}}}}
    assert validate_bound_plan(plan,files,strategy)==plan['selected']
    strategy['policy']['navigation']['route_limits']['corridor_radius_m']=.2
    with pytest.raises(WorkflowError,match='corridor'):validate_bound_plan(plan,files,strategy)


def test_extract_to_native_plan_end_to_end_without_physics(tmp_path):
    import numpy as np
    p=archive(tmp_path);manifest=json.loads(p.read_text())
    manifest['bindings'].update(operation_bodies=['pelvis'],contact_policy=[]);p.write_text(json.dumps(manifest))
    options=tmp_path/'extract_options.json';options.write_text(json.dumps(dict(window_s=.1,lookback_s=.3,speed_m_s=.05,yaw_speed_rad_s=.1,drift_m=.02)))
    prior=tmp_path/'prior.json';assert main(['extract','--manifest',str(p),'--options',str(options),'--out',str(prior)])==0
    with np.load(tmp_path/'trajectory.npz') as z:np.savez(tmp_path/'initial.npz',qpos=z['state'][0,1:8])
    scene={'files':{'model':'scene.mjb','initial':'initial.npz'},'frame':manifest['bindings']['frame'],
      'search':{'max_candidates':2,'offsets_xyyaw':[[0,0,0]],'planning_timeout_s':5.},
      'native_options':{'path':{'resolution':.1,'padding':.3,'max_expansions':50},'collision_tolerance_m':.001,
          'corridor_margin_m':.08,'sample_translation_m':.05,'sample_angle_rad':.05,'max_checks':1000,
          'planning_timeout_s':5.,'support_distance_m':.005,'relation_tolerance_m':.03,'relation_tolerance_rad':.05}}
    sp=tmp_path/'scene.json';sp.write_text(json.dumps(scene));out=tmp_path/'plan.json'
    assert main(['plan','--prior',str(prior),'--scene',str(sp),'--out',str(out)])==0
    result=json.loads(out.read_text());assert result['selected']['id']=='current'
    assert result['physics_steps']==0
    (tmp_path/'initial.npz').write_bytes(b'changed')
    from humanoid_mimicgen.workflow.navigation.archive import verify_evidence
    with pytest.raises(WorkflowError):verify_evidence(result['evidence_sha256'])


def test_navigation_source_uses_declared_state_key(tmp_path):
    import numpy as np
    from humanoid_mimicgen.workflow.navigation.integration import load_navigation_source
    p=archive(tmp_path);plan={'source_manifest':str(p)}
    source=load_navigation_source(tmp_path/'trajectory.npz',plan)
    assert source.shape==(30,14)
