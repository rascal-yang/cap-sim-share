import json
from pathlib import Path
import numpy as np
import mujoco
import pytest
from humanoid_mimicgen.workflow.navigation.__main__ import main
from humanoid_mimicgen.workflow.tests.test_lateral_planning import mechanism


def source_fixture(path):
    m,short,p,b,o,c=mechanism();mujoco.mj_saveModel(m,str(path/'model.mjb'),None)
    q=np.tile(short[0],(51,1));q[:,0]=np.linspace(0,.01,51)
    for name in ['lf','rf']:q[:,m.jnt_qposadr[m.joint(name+'_x').id]]=-q[:,0]
    np.savez(path/'source.npz',state=np.c_[np.arange(51)*.01,q,np.zeros((51,m.nv))])
    np.savez(path/'initial.npz',qpos=q[0],qvel=np.zeros(m.nv))
    hands=np.zeros((26,14));hands[18:,8]=.2
    np.savez(path/'hands.npz',left=hands[:,:7],right=hands[:,7:]);np.savez(path/'controller.npz',records=np.c_[np.arange(26)*.02,np.ones(26)])
    (path/'result.json').write_text(json.dumps(dict(native_success=True,original_token_steps=0,encoder_modes=[1],control_variant='synthetic_test')))
    b.update(frame=dict(kind='world_heading',yaw_rad=0.,evidence='toy box'),capture_bodies=['pelvis','lf','rf','lw','rw'],
        lateral_surface=dict(geom='table',normal_local=[1,0,0]),operation_stages=dict(entry=30,closure=36,first_contact=38,lift=40,end=50))
    b['contact_policy'][0].update(start=30,end=50)
    manifest=dict(files={k:str(path/v) for k,v in dict(model='model.mjb',trajectory='source.npz',initial='initial.npz',
        result='result.json',hands='hands.npz',controller='controller.npz').items()},bindings=b,entry_index=30,end_index=50,
        state_key='state',support_distance_m=.002,success_field='native_success',success_criterion='native',
        hand_alignment=dict(kind='controller_timestamps',control_dt_s=.02))
    o['extraction']=dict(window_s=.1,lookback_s=.3,foot_drift_m=.002,foot_angle_rad=.02,max_windows=2,open_fingers_rad=[0.]*14,finger_tolerance_rad=1e-6)
    o['handoff_timeout_s']=30.
    (path/'manifest.json').write_text(json.dumps(manifest));(path/'options.json').write_text(json.dumps(o))
    return path/'manifest.json',path/'options.json'


def test_lateral_cli_extract_and_plan_readonly_no_overwrite(tmp_path):
    manifest,options=source_fixture(tmp_path);prior=tmp_path/'prior.json';out=tmp_path/'plan'
    args=['lateral-extract','--manifest',str(manifest),'--options',str(options),'--out',str(prior)]
    assert main(args)==0
    assert json.loads(prior.read_text())['support_windows']
    assert main(args)!=0
    args=['lateral-plan','--manifest',str(manifest),'--options',str(options),'--initial',str(tmp_path/'initial.npz'),'--offset-m','.04','--out-dir',str(out)]
    assert main(args)==0
    plan=json.loads((out/'plan.json').read_text());assert plan['selected']['id']=='current' and plan['physics_steps']==0
    assert plan['generated_reference']
    assert main(args)!=0


def test_lateral_bad_scene_keeps_failure_evidence(tmp_path):
    manifest,options=source_fixture(tmp_path);out=tmp_path/'bad'
    assert main(['lateral-plan','--manifest',str(manifest),'--options',str(options),'--initial',str(tmp_path/'initial.npz'),
                 '--offset-m','.6','--out-dir',str(out)])!=0
    assert json.loads((out/'plan.json').read_text())['reason']=='invalid_lateral_scene'
    assert str(manifest.resolve()) in json.loads((out/'plan.json').read_text())['evidence_sha256']


def test_measured_handoff_regenerates_for_actual_feet(tmp_path):
    from humanoid_mimicgen.workflow.navigation.lateral import plan_lateral
    from humanoid_mimicgen.workflow.navigation.lateral_execution import regenerate_handoff
    from humanoid_mimicgen.workflow.navigation.reference_bundle import load_reference_bundle
    manifest,options=source_fixture(tmp_path)
    plan=plan_lateral(manifest,tmp_path/'initial.npz',json.loads(options.read_text()),tmp_path/'planned',.04)
    assert plan['status']=='completed',plan
    q=np.load(tmp_path/'planned/initial.npz')['qpos'];q[1]+=.01
    result=regenerate_handoff(plan,q,tmp_path/'handoff')
    assert result['feasible'],result
    bundle=load_reference_bundle(result['generated_reference'])
    np.testing.assert_array_equal(bundle['bridge_qpos'][0],q)
    assert bundle['metadata']['candidate']['feet']['lf'][1]==pytest.approx(.11)


def test_astar_budget_is_inconclusive_and_audited(tmp_path,monkeypatch):
    from humanoid_mimicgen.workflow.navigation import lateral
    manifest,options=source_fixture(tmp_path);opts=json.loads(options.read_text());opts['native']['path']['max_expansions']=1
    monkeypatch.setattr(lateral,'plan_base_path',lambda *a,**k:(None,dict(method='astar_exhausted',expansions=1,checks=3)))
    r=lateral.plan_lateral(manifest,tmp_path/'initial.npz',opts,tmp_path/'budget',.04)
    assert r['status']=='planning_budget_exhausted' and r['search_incomplete']
    assert r['candidates'][0]['navigation_path_audit']['expansions']==1


def test_route_timeout_retains_already_valid_candidate(tmp_path,monkeypatch):
    from humanoid_mimicgen.workflow.navigation import lateral
    from humanoid_mimicgen.workflow.contracts import WorkflowError
    manifest,options=source_fixture(tmp_path);original=lateral.lateral_candidates;calls=[]
    def candidates(*a,**k):
        rows,truncated=original(*a,**k);rows[0]['requires_navigation']=True;return rows,truncated
    def route(start,goal,*a,**k):
        calls.append(1)
        if len(calls)>1:raise WorkflowError('planning_budget_exhausted','Route deadline')
        return np.array([start,goal]),dict(method='direct',checks=2,expansions=0)
    monkeypatch.setattr(lateral,'lateral_candidates',candidates);monkeypatch.setattr(lateral,'plan_base_path',route)
    r=lateral.plan_lateral(manifest,tmp_path/'initial.npz',json.loads(options.read_text()),tmp_path/'budget',.04)
    assert r['status']=='completed' and r['selected']['id']=='current' and r['search_incomplete']


def test_replay_first_stops_after_first_feasible_stance(tmp_path,monkeypatch):
    from humanoid_mimicgen.workflow.navigation import lateral
    manifest,options=source_fixture(tmp_path);o=json.loads(options.read_text())
    o['search'].update(mode='replay_first',max_candidates=3)
    original=lateral.LateralSolver.solve;calls=[]
    def solve(self,candidate,*args):
        calls.append(candidate['id'])
        return original(self,candidate,*args)
    monkeypatch.setattr(lateral.LateralSolver,'solve',solve)
    r=lateral.plan_lateral(manifest,tmp_path/'initial.npz',o,tmp_path/'first',.04)
    assert r['status']=='completed',r
    assert len(calls)==len(r['candidates'])==1
    assert r['selection_policy']=='first_feasible_replay_prior'


def test_replay_first_rejected_screen_falls_back_once_and_keeps_reason(tmp_path,monkeypatch):
    from humanoid_mimicgen.workflow.navigation import lateral
    manifest,options=source_fixture(tmp_path);o=json.loads(options.read_text());o['search'].update(mode='replay_first',max_candidates=3)
    original=lateral.LateralSolver.screen;calls=[]
    def screen(self,candidate,*args):
        calls.append(candidate['id'])
        if len(calls)==1:return dict(feasible=False,operation_checked=False,reason='operation_collision',physics_steps=0)
        return original(self,candidate,*args)
    monkeypatch.setattr(lateral.LateralSolver,'screen',screen)
    r=lateral.plan_lateral(manifest,tmp_path/'initial.npz',o,tmp_path/'fallback',.04)
    assert r['status']=='completed' and len(calls)==2
    assert r['candidates'][0]['screening']['reason']=='operation_collision'
    assert r['selected']['proposal']=='local_fallback'
