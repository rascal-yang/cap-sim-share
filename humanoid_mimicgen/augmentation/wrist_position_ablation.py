"""Fresh-start apple A/B: unchanged reference versus historical preencoder integral."""
import copy,json,time,shutil
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from ..wrist_feedback import position_bias
from ..workflow.contracts import now,file_hash
from .agent import save
from . import suite_physics as ph
from . import rollback_benchmark as rb
from . import generalization_runtime as runtime
from .suite_evidence import install
from .leg_fast_retry import exact_rollout
from .encoder_wrist_study import replay_audit


def adjust_reference(frame,shift,measured,bias,enabled):
    frame=np.asarray(frame,float);shift=np.asarray(shift,float);measured=np.asarray(measured,float);bias=np.asarray(bias,float)
    if frame.shape!=(3,7) or shift.shape!=(2,) or measured.shape!=(3,) or bias.shape!=(3,) or not all(np.isfinite(x).all() for x in (frame,shift,measured,bias)):raise ValueError('Invalid wrist observation')
    desired=frame[1,:3]+np.r_[shift,0.]
    updated=position_bias(bias,desired,measured,.02) if enabled else np.zeros(3)
    out=frame.copy();out[1,:3]+=updated
    return out,updated,{'desired_world_m':desired.tolist(),'measured_world_m':measured.tolist(),'error_world_m':(desired-measured).tolist(),'bias_world_m':updated.tolist(),'commanded_world_m':(desired+updated).tolist()}


@contextmanager
def feedback_scope(config,enabled,replay=False):
    old_encode,old_write,old_checkpoint=ph.encode_reference,ph.write_new,runtime.write_checkpoint_bundle
    state={'bias':np.zeros(3),'logs':[]};legacy=Path(__file__).parents[1]/'wrist_feedback.py'
    def encode(s,d,q,v,poses,index,shift,pose_frame=None):
        if replay:raise ValueError('Saved-token replay unexpectedly called encoder')
        frame=poses[index] if pose_frame is None else pose_frame
        actual=d.xpos[s.m.body('right_wrist_yaw_link').id].copy()
        adjusted,state['bias'],row=adjust_reference(frame,shift,actual,state['bias'],enabled)
        row.update(frame=int(index),time_s=float(d.time));state['logs'].append(row)
        return old_encode(s,d,q,v,poses,index,shift,adjusted)
    def write(path,value):
        if Path(path).name=='experiment.json':
            value=copy.deepcopy(value)
            value['purpose'].update(hypothesis='Does restoring historical right wrist position integral before encoding improve fixed-policy apple position transfer?',planned_changes=[{'right_wrist_feedback_enabled':enabled and not replay,'saved_token_replay':replay,'fixed_baseline_candidate':0}],known_confounders=['Feedback starts at mode1 handoff only; original approach prefix remains unchanged.','Feedback tracks unchanged baseline wrist reference INCLUDING fixed API offset; this does not restore early navigation/replanning/early-finger logic.'])
            value['pipeline'].update(entrypoint='wrist_position_ablation.run',external_model_calls='Zero new API calls; reuse authenticated historical candidate0 plus user-authorized deterministic A/B feedback',feedback_and_reflection_rules='One full rollout per arm per scene; no retuning, no retry, no rollback',allowed_retry_changes=[],optimization_enabled=False)
            value['control']['right_wrist_position_feedback']={'enabled':enabled and not replay,'gain_per_s':.5,'cap_m':.02,'dt_s':.02,'frame':'world','error':'baseline wrist reference including fixed API offset and scene translation minus measured wrist-body origin','target_change':'right xyz only before encoder; nominal source, orientation, left, torso, legs and hands unchanged','activation':'mode1 frames only; no accumulation during original-token prefix','reset':'zero at every fresh episode','scope':'not full historical navigation/approach policy'}
            value['control']['checkpoint_reuse_supported']=not enabled
            value['control']['body_postdecoder_identity']=True
            value['budget'].update(attempts_per_case=1,recovery_attempts=0,external_api_calls=0,comparison=config['ablation'])
            for p in [legacy,Path(config['config_path']),Path(config['historical_feedback_contract']),Path(config['protocol_path'])]:
                value['preflight']['evidence_paths'].append(str(p));value['preflight']['evidence_sha256'][str(p)]=file_hash(p)
            shutil.copy2(legacy,Path(path).parent/'source_snapshot'/'historical_wrist_feedback.py')
        if Path(path).name in ('result.json','execution_audit.json'):
            value.update(right_wrist_position_feedback_enabled=enabled and not replay,new_api_calls=0,control_variant='saved_token_hands_identity_PD' if replay else 'mode1_right_wrist_integral' if enabled else 'matched_mode1_baseline',checkpoint_reuse_supported=not enabled)
        return old_write(path,value)
    def checkpoints(case,states,metadata):
        old_checkpoint(case,states,metadata)
        if enabled:
            # Deliberately reject legacy rollback loaders: this A/B uses fresh starts,
            # and the external integral is not in their controller extension schema.
            p=case/'execution/checkpoint_manifest.json';m=json.loads(p.read_text());m.update(version=2,rollback_supported=False,reason='Position-integral A/B: fresh starts only; use saved tokens+hands for replay');save(p,m)
    ph.encode_reference=encode;ph.write_new=write;runtime.write_checkpoint_bundle=checkpoints
    try:yield state
    finally:ph.encode_reference=old_encode;ph.write_new=old_write;runtime.write_checkpoint_bundle=old_checkpoint


def execute(config,item,arm,folder,replay_source=None):
    spec=json.loads(Path(item['spec_path']).read_text())
    with feedback_scope(config,arm=='feedback',replay_source is not None) as state:
        row=rb.execute(config,spec,item['spec_path'],item['batch_path'],0,folder,rb.restart(),replay_source)
    case=Path(row['result']['experiment_path']);save(case/'execution/wrist_position_feedback.json',state['logs'])
    metrics=json.loads((case/'execution/metrics.json').read_text());heights=[m['object_xyz']['target'][2] for m in metrics]
    row['diagnostics']={'max_lift_m':max(heights)-heights[0],'final_lift_m':heights[-1]-heights[0],'final_hand_object_N':metrics[-1]['hand_object_N'],'feedback_ticks':len(state['logs']),'max_bias_m':max([np.linalg.norm(x['bias_world_m']) for x in state['logs']],default=0),'feedback_log':str(case/'execution/wrist_position_feedback.json')}
    if state['logs']:row['diagnostics']['mode1_wrist_rmse_m']=float(np.sqrt(np.mean([np.dot(x['error_world_m'],x['error_world_m']) for x in state['logs']])))
    save(folder/'trial.json',row);return row


def compare_arms(a,b,prefix):
    ac=Path(a['result']['experiment_path']);bc=Path(b['result']['experiment_path'])
    with np.load(ac/'execution/trajectory.npz') as x,np.load(bc/'execution/trajectory.npz') as y:
        prefix_exact=all(np.array_equal(x[k][:prefix*4],y[k][:prefix*4]) for k in ('state','next_state','commands'))
    with np.load(ac/'execution/hand_commands.npz') as x,np.load(bc/'execution/hand_commands.npz') as y:
        n=min(len(x['left']),len(y['left']));hands_exact=all(np.array_equal(x[k][:n],y[k][:n]) for k in ('left','right'))
    x=json.loads((ac/'execution/wrist_position_feedback.json').read_text());y=json.loads((bc/'execution/wrist_position_feedback.json').read_text());n=min(len(x),len(y))
    nominal_exact=all(x[i]['frame']==y[i]['frame'] and x[i]['desired_world_m']==y[i]['desired_world_m'] for i in range(n))
    rmse=lambda rows:float(np.sqrt(np.mean([np.dot(r['error_world_m'],r['error_world_m']) for r in rows[:n]]))) if n else None
    return {'prefix_exact':prefix_exact,'source_hands_exact_common_frames':hands_exact,'nominal_world_targets_exact':nominal_exact,'common_mode1_frames':n,'common_mode1_rmse_m':{'baseline':rmse(x),'feedback':rmse(y)},'full_coverage_both':a['result']['coverage']==b['result']['coverage']==1}


def run(config):
    if any(config['ablation'].get(k)!=v for k,v in {'gain_per_s':.5,'cap_m':.02,'dt_s':.02}.items()):raise ValueError('Historical integral settings are fixed in this ablation')
    install();root=Path(config['output_root']);started=now();tick=time.monotonic();results=[];old_runtime=rb.runtime;rb.runtime=runtime
    save(root/'run_started.json',{'started_at':started,'config_sha256':file_hash(config['config_path'])})
    try:
        for i,item in enumerate(config['conditions']):
            folder=root/item['condition_id'];folder.mkdir();rec={'condition_id':item['condition_id'],'shift':item['shift'],'started_at':now(),'arms':{},'replays':{},'status':'running'};results.append(rec)
            try:
                for arm in (['baseline','feedback'] if i%2==0 else ['feedback','baseline']):
                    row=execute(config,item,arm,folder/arm);rec['arms'][arm]=row
                    print(json.dumps({'condition':item['condition_id'],'arm':arm,'success':row['success'],'failure':row['result']['failure'],'max_lift_m':row['diagnostics']['max_lift_m']}),flush=True)
                a,b=rec['arms']['baseline'],rec['arms']['feedback'];spec=json.loads(Path(item['spec_path']).read_text());rec['comparison']=compare_arms(a,b,spec['base_plan']['contact_start'])
                rec['historical_baseline_match']=exact_rollout(item['historical_seed_case'],a['result']['experiment_path'])
                if not rec['historical_baseline_match']['exact'] or not all(rec['comparison'][k] for k in ('prefix_exact','source_hands_exact_common_frames','nominal_world_targets_exact')):raise ValueError('A/B invariant failed')
                for arm,row in rec['arms'].items():
                    if not row['success']:continue
                    replay=execute(config,item,arm,folder/(arm+'_token_replay'),Path(row['result']['experiment_path']));proof=replay_audit(row['result'],replay['result'])
                    with np.load(Path(row['result']['experiment_path'])/'execution/hand_commands.npz') as x,np.load(Path(replay['result']['experiment_path'])/'execution/hand_commands.npz') as y:proof['hands_exact']=all(np.array_equal(x[k],y[k]) for k in ('left','right'))
                    passed=replay['success'] and proof['exact'] and proof['hands_exact'];rec['replays'][arm]={'trial':replay,'proof':proof,'passed':bool(passed)}
                    if not passed:raise ValueError('Successful rollout failed token/hands replay')
                rec['status']='complete'
            except Exception as e:
                import traceback
                rec.update(status='error',error=str(e),traceback=traceback.format_exc())
            rec['ended_at']=now();save(folder/'summary.json',rec);save(root/'progress.json',results)
            if rec['status']=='error':raise RuntimeError('A/B implementation/verification failed; stop before remaining scenes: '+rec['error'])
        out={'experiment_id':root.name,'started_at':started,'ended_at':now(),'numeric_wall_s':time.monotonic()-tick,'new_api_calls':0,'new_api_tokens':0,'results':results};save(root/'summary.json',out);return out
    finally:rb.runtime=old_runtime
