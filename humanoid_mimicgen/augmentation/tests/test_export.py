import json
from pathlib import Path
import numpy as np
import pytest
from humanoid_mimicgen.augmentation.export import export_case, export_job
from humanoid_mimicgen.augmentation.jobs import JobStore
from humanoid_mimicgen.workflow.contracts import canonical, file_hash, WorkflowError


def trace(tmp_path, *, success=True):
    root = tmp_path/'trace'; out = root/'execution'; out.mkdir(parents=True)
    def save(path, obj): path.write_bytes(canonical(obj))
    native = root/'native_preflight.json'
    save(native, {'model_nq': 7, 'model_nv': 6, 'model_nu': 2, 'physics_dt_s': .01})
    np.savez(root/'targets.npz', fingers=np.zeros((2,14)))
    strategy = {'phases': [{'id':'hold','start':0,'end':2,'requires_clearance':True,'requires_contact':True}],
                'policy': {'acceptance': {'clearance_m': .01, 'hold_duration_s': .02, 'root_drift_m': .03, 'support_force_N': .1},
                           'gates': {'contact_force_N': .1}, 'recovery': {'max_retries': 0}}}
    contract = {'experiment_id': 'synthetic_fixture', 'record_status':'frozen_ready',
                'inputs': {'targets': str(root/'targets.npz')},
                'workflow': {'strategy': strategy},
                'preflight': {'passed': True, 'evidence_sha256': {str(native):file_hash(native), str(root/'targets.npz'):file_hash(root/'targets.npz')}}}
    save(root/'experiment.json', contract)
    (root/'experiment.sha256').write_text(file_hash(root/'experiment.json'))
    q0 = np.array([0.,0,1,1,0,0,0]); q1=q0.copy(); q1[0]=.001; q2=q0.copy(); q2[0]=.002
    np.savez(out/'actual_initial.npz', qpos=q0, qvel=np.zeros(6), ctrl=np.zeros(2))
    np.savez(out/'states.npz', samples=np.array([np.r_[.01,q1,np.zeros(6),[1,2]],np.r_[.02,q2,np.zeros(6),[3,4]]]))
    commands=[{'time_s':i*.01,'mode':1,'source_index':i,'retry_index':0,'stage':'hold',
               'ctrl_Nm':v, 'fingers_rad':[0]*14,'body_target_rad':[.1,.2]} for i,v in enumerate(([1,2],[3,4]))]
    save(out/'commands.json',commands); save(out/'attempted_commands.json',commands)
    obs=[{'time_s':(i+1)*.01,'stage':'hold','retry_index':0,'root_pose':q.tolist(),
          'clearance_m':.02,'contacts':{'hand_object_N':1.,'support_N':0.,'robot_obstacle_N':0.}} for i,q in enumerate((q1,q2))]
    save(out/'observations.json',obs); save(out/'events.json',[])
    save(out/'execution_audit.json',{'physics_steps':2,'recorded_steps':2,'completed_commands':2,
         'original_token_prefix_count':0,'mode0_calls':0,'midrun_state_resets':0,'fallback':False,
         'actual_encoder_modes':[1], 'recovery_count':0,'completed':success,'failure':None if success else 'contact_lost'})
    save(root/'result.json',{'physical_task_completed':success,'training_ready':False,'verification_errors':[],
                            'status':'completed' if success else 'failed'})
    return root


def test_measured_state_action_alignment(tmp_path):
    root=trace(tmp_path); result=export_case(root,tmp_path/'export')
    arrays=np.load(Path(result['path'])/'trajectory.npz')
    np.testing.assert_allclose(arrays['qpos'][:,0],[0,.001])
    np.testing.assert_allclose(arrays['next_qpos'][:,0],[.001,.002])
    np.testing.assert_array_equal(arrays['action_ctrl'],[[1,2],[3,4]])
    assert result['physical_success'] is True and result['transitions'] == 2


def test_attempted_commands_are_separate(tmp_path):
    root=trace(tmp_path,success=False); p=root/'execution/attempted_commands.json'
    attempted=json.loads(p.read_text()); attempted.append(dict(attempted[-1],time_s=.02)); p.write_text(json.dumps(attempted))
    result=export_case(root,tmp_path/'export')
    assert result['transitions']==2 and result['attempted_commands']==3 and not result['physical_success']


def test_corrupt_or_truncated_trace_rejected(tmp_path):
    root=trace(tmp_path); p=root/'execution/observations.json'; p.write_text('[]')
    with pytest.raises(WorkflowError,match='alignment'): export_case(root,tmp_path/'export')
    assert not (tmp_path/'export').exists()


def test_mode1_is_not_token_and_visual_status_preserved(tmp_path):
    result=export_case(trace(tmp_path),tmp_path/'export')
    assert result['action_representation']=='sonic_mode1_commands_and_applied_ctrl'
    assert result['sonic_tokens_available'] is False and result['training_ready'] is False


def test_reexport_is_idempotent_and_detects_changed_artifact(tmp_path):
    root=trace(tmp_path); a=export_case(root,tmp_path/'export'); b=export_case(root,tmp_path/'export')
    assert a==b
    (tmp_path/'export/trajectory.npz').write_bytes(b'corrupt')
    with pytest.raises(WorkflowError,match='integrity'): export_case(root,tmp_path/'export')


def test_claimed_success_requires_measured_hold(tmp_path):
    root=trace(tmp_path); p=root/'execution/observations.json'
    rows=json.loads(p.read_text()); rows[-1]['contacts']['hand_object_N']=0.; p.write_text(json.dumps(rows))
    with pytest.raises(WorkflowError,match='success_evidence'): export_case(root,tmp_path/'export')


def test_failed_cases_retained_in_denominator(tmp_path):
    root=trace(tmp_path); store=JobStore(tmp_path/'store')
    ident=store.submit({'request_id':'job','output_root':str(tmp_path),'conditions':{'mode':'explicit','cases':[{'case_id':'a'},{'case_id':'b'}]}}, {})['job_id']
    store.record_case(ident,'a','verified',{'experiment_path':str(root)})
    store.record_case(ident,'b','failed',{'reason_code':'initial_collision'})
    result=export_job(store,ident)
    assert result['case_count']==2 and result['successes']==1 and result['success_rate']==.5
    index=json.loads(Path(result['index_path']).read_text())
    assert index['cases'][1]['reason_code']=='initial_collision'


def test_export_rejects_unfinished_batch_without_freezing_index(tmp_path):
    store=JobStore(tmp_path/'store')
    ident=store.submit({'request_id':'job','conditions':{'mode':'explicit','cases':[{'case_id':'a'}]}}, {})['job_id']
    with pytest.raises(WorkflowError,match='incomplete_workflow'): export_job(store,ident)
    assert not (store.root/'datasets'/ident/'index.json').exists()


def test_historical_export_uses_hash_matching_source_snapshot(tmp_path):
    root=trace(tmp_path)
    live=tmp_path/'repo/humanoid_mimicgen/workflow/batch.py'; live.parent.mkdir(parents=True); live.write_text('old')
    expected=file_hash(live)
    snap=root/'source_snapshot/workflow/batch.py'; snap.parent.mkdir(parents=True); snap.write_text('old')
    c=json.loads((root/'experiment.json').read_text()); c['preflight']['evidence_sha256'][str(live)]=expected
    (root/'experiment.json').write_bytes(canonical(c)); (root/'experiment.sha256').write_text(file_hash(root/'experiment.json'))
    live.write_text('new')
    assert export_case(root,tmp_path/'export')['transitions']==2
    snap.write_text('tampered')
    with pytest.raises(WorkflowError,match='integrity_error'): export_case(root,tmp_path/'other')


def test_prefix_truncated_trace_cannot_use_initial_as_prestate(tmp_path):
    root=trace(tmp_path,success=False); out=root/'execution'
    for name in ('commands.json','attempted_commands.json','observations.json'):
        p=out/name; p.write_bytes(canonical(json.loads(p.read_text())[1:]))
    with np.load(out/'states.npz') as a: samples=a['samples'][1:].copy()
    np.savez(out/'states.npz',samples=samples)
    p=out/'execution_audit.json'; a=json.loads(p.read_text())
    for key in ('physics_steps','recorded_steps','completed_commands'): a[key]=1
    p.write_bytes(canonical(a))
    with pytest.raises(WorkflowError,match='alignment'): export_case(root,tmp_path/'export')


def test_optional_video_is_copied_hashed_and_not_visually_approved(tmp_path):
    root=trace(tmp_path); video=root/'video'; video.mkdir()
    # Deliberately not a decodable video: export records evidence, never approves visual quality.
    (video/'external.mp4').write_bytes(b'unreviewed-video-placeholder')
    result=export_case(root,tmp_path/'export')
    item=result['videos']['external']
    assert file_hash(tmp_path/'export'/item['path'])==file_hash(video/'external.mp4')==item['sha256']
    assert item['visual_acceptance']=='unknown' and not result['training_ready']
