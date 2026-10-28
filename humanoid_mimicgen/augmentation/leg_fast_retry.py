"""Isolated checkpoint-based retries reusing measured leg response, with exact proofs."""
import copy,json,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from . import suite_physics as ph
from .leg_response import scope
from .leg_response_study import SCHEMA,FIELDS,inspect_result
from .contact_review import PROPS,tools
from .quality_suite import api_decide,verify_origin,PROPERTIES,validate_decision
from .suite_provider import configured_provider
from .contact_batch import BudgetProvider
from .suite_evidence import install
from .retry_state import load,offer
from .saved_token_replay import verify_source_contract
from .encoder_wrist_study import replay_audit
from .suite_inspect import inspect_case
from .contact_strategy import shallow_video
from .agent import save
from ..workflow.contracts import now,file_hash

SKILL=Path(__file__).parent/'skills/leg_fast_retry/SKILL.md'

def join_hands(parent,suffix,frame,total):
    parent=np.asarray(parent);suffix=np.asarray(suffix)
    if parent.ndim!=2 or suffix.ndim!=2 or parent.shape[1]!=suffix.shape[1] or frame>len(parent) or frame<0 or frame+len(suffix)!=total:raise ValueError('Incomplete hand prefix/suffix')
    return np.concatenate([parent[:frame],suffix],axis=0)

def require_boundary(resume,start,parent):
    if resume['kind']!='checkpoint' or resume['frame']!=start or resume['case']!=str(parent):raise ValueError('Only verified mode1 handoff checkpoint is supported; later frames require integral restoration')

def require_same_model(model,frozen):
    if model!=frozen:raise ValueError('Response model differs from verified parent frozen model')

def arrays_exact(a,b):return set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a)

@contextmanager
def branch_support(config,plan):
    original=ph.write_new;resume=plan['resume'];branch=resume['kind']=='checkpoint';parent=Path(resume['case']) if branch else None;written=False
    if branch:require_boundary(resume,plan['contact_start'],config['rollback_case']);load(parent,resume['frame'])
    def write(path,value):
        nonlocal written
        path=Path(path)
        if path.name=='experiment.json':
            value=copy.deepcopy(value);bound=[Path(config['response_model_path'])]
            if branch:bound.append(parent/'execution/hand_commands.npz')
            for p in bound:value['preflight']['evidence_paths'].append(str(p));value['preflight']['evidence_sha256'][str(p)]=file_hash(p)
            value['pipeline'].update(entrypoint='leg_fast_retry.run',feedback_and_reflection_rules='reuse calibrated response; API chooses bounded parameters; exact checkpoint validation first; local feedback; render only selected results at end',allowed_retry_changes=['feedback parameters and tail2..3s; checkpoint fixed at pre-mode1 boundary'])
            value['control']['checkpoint_local_feedback_state']={'frame':resume['frame'],'integral_offset':'zero at first mode1 tick; no previous reference feedback executed','hands':'inherited prefix concatenated with newly executed suffix'} if branch else {'restart':True}
            value['purpose']['known_confounders'].append('Branch prefix is inherited, not newly simulated; fresh full restart and token replay required before final acceptance') if branch else None
        if branch and path.name in ('execution_audit.json','result.json'):
            case=Path(value['experiment_path']);out=case/'execution/hand_commands.npz';total=len(np.load(case/'execution/controller.npz')['records'])
            if not written:
                with np.load(parent/'execution/hand_commands.npz') as p,np.load(out) as s:joined={k:join_hands(p[k],s[k],resume['frame'],total) for k in ('left','right')}
                np.savez_compressed(out,**joined);written=True
            value.update(hand_commands_include_inherited_prefix=True,hand_command_frames=total,rollback_local_integral_restored='zero at mode1 boundary')
        return original(path,value)
    ph.write_new=write
    try:yield
    finally:ph.write_new=original

def validate(d,fixed,config,parameters=None,tail=None,restart=False):
    if set(d)!=set(SCHEMA):raise ValueError('Unexpected fields')
    validate_decision({k:d[k] for k in PROPERTIES},[fixed['episode']])
    for k,v in fixed.items():
        if k not in ('action','rationale','hypothesis','leg_response','tail_s','resume') and d[k]!=v:raise ValueError('Fixed field changed: '+k)
    if d['action'] not in ('run','needs_tool','reject') or d['contact_control']['joints']:raise ValueError('Unsupported action or residual')
    if not 2<=d['tail_s']<=3 or abs(d['tail_s']/.02-round(d['tail_s']/.02))>1e-6:raise ValueError('Tail must be20ms aligned in2..3s')
    if restart:
        if d['resume']['kind']!='restart' or d['resume']['case'] or d['resume']['frame']!=0:raise ValueError('Full restart required')
    else:require_boundary(d['resume'],d['contact_start'],config['rollback_case'])
    p=d['leg_response']
    if set(p)!=set(FIELDS):raise ValueError('Invalid feedback fields')
    for k,(lo,hi) in FIELDS.items():
        if type(p[k]) not in (int,float) or not np.isfinite(p[k]) or not lo<=p[k]<=hi:raise ValueError('Invalid '+k)
    model=json.loads(Path(config['response_model_path']).read_text())
    if any(p[k]!=model['calibration_parameters'][k] for k in ('calibration_amplitude_rad','ramp_s')):raise ValueError('Calibration parameters fixed')
    if parameters is not None and p!=parameters:raise ValueError('Verification must preserve parameters exactly')
    if tail is not None and d['tail_s']!=tail:raise ValueError('Verification must preserve tail exactly')
    return d

def exact_rollout(a,b):
    a=Path(a);b=Path(b);ta=np.load(a/'execution/trajectory.npz');tb=np.load(b/'execution/trajectory.npz');rawa=np.load(a/'execution/controller.npz')['records'];rawb=np.load(b/'execution/controller.npz')['records']
    fields=['state','next_state','commands'];result={k:bool(np.array_equal(ta[k],tb[k])) for k in fields};result['controller_exact']=bool(np.array_equal(rawa,rawb))
    with np.load(a/'execution/hand_commands.npz') as x,np.load(b/'execution/hand_commands.npz') as y:result['hands_exact']=arrays_exact({k:x[k] for k in x.files},{k:y[k] for k in y.files})
    result['exact']=all(result.values());return result

def compact_audit(case):
    p=Path(case);a=json.loads((p/'response_audit.json').read_text());rows=json.loads((p/'execution/leg_response.json').read_text())
    a['last_feedback_rate_rad_s']=rows[-1]['offset_rate_rad_s'];a['last_goal_error_rad']=rows[-1]['error_rad'];a['last_offsets_rad']=rows[-1]['offset_paired_rad'];return a

def run(config,provider_config):
    install();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True);start=now();tick=time.monotonic();spec=config['task'];src=spec['source'];parent=Path(config['rollback_case']);verify_source_contract(parent);pr=json.loads((parent/'result.json').read_text());old=verify_origin(pr['decision_path'])['decision'];fixed={k:copy.deepcopy(old[k]) for k in PROPS};fixed['leg_response']=old['leg_response'];model=json.loads(Path(config['response_model_path']).read_text());require_same_model(model,json.loads((parent/'response_variant.json').read_text())['model']);parent_modes=np.load(parent/'execution/controller.npz')['records'][:,1];assert pr['branch_start_frame']==0 and np.all(parent_modes[:fixed['contact_start']]==-1) and parent_modes[fixed['contact_start']]==1;history=[];status='budget_exhausted';error=None;equivalence=None;proof=None
    provider=BudgetProvider(configured_provider(provider_config,root/'.api_rotation.json'),time.monotonic()+1800)
    def decide(stage,label,parameters=None,tail=None,restart=False):
        folder=root/label;folder.mkdir();suggested=copy.deepcopy(fixed);suggested['resume']={'kind':'restart' if restart else 'checkpoint','case':'' if restart else str(parent),'frame':0 if restart else fixed['contact_start'],'rationale':'Full restart verification' if restart else 'Restore verified pre-mode1 boundary with zero feedback integral'}
        if parameters is not None:suggested['leg_response']=parameters
        if tail is not None:suggested['tail_s']=tail
        context={'stage':stage,'fixed_plan':{k:v for k,v in suggested.items() if k not in ('action','rationale','hypothesis')},'checkpoint_offer':offer(parent),'response_model':model,'previous_full_result':compact_audit(parent),'previous_new':[{'label':h['label'],'parameters':h['parameters'],'tail_s':h['tail_s'],'audit':h['audit']} for h in history],'remaining_candidates':3-sum(h['label'].startswith('candidate') for h in history),'native_criteria':pr['native_acceptance']['parameters'],'reference_semantics':'successful original-token measured q/dq; not PD targets','test_shift':src['object_delta_xy_m'],'rendering':'deferred until final selection; no new images needed to diagnose measured angular errors','required_exact_parameters':parameters,'required_exact_tail_s':tail}
        d,dp=api_decide(provider,folder,context,[src['episode']],tools=tools(SCHEMA),validator=lambda d,eps:validate(d,fixed,config,parameters,tail,restart),skill_path=SKILL);save(folder/'plan.json',d);return d,dp
    def execute(label,d,dp,replay_source=None):
        ar=root/label;ar.mkdir();verify_origin(dp)
        with branch_support(config,d):
            with scope(config,d,{'kind':'feedback'},model,replay=replay_source is not None):result=ph.physical_case(spec['dataset'],src['episode'],ar,'saved_token_replay' if replay_source else 'token_prefix_mode1',src['object_delta_xy_m'],decision_path=dp,contact_control=replay_source is None,saved_token_case=replay_source)
        c=Path(result['experiment_path']);audit=inspect_result(c)
        if replay_source is None:audit=compact_audit(c)
        row={'label':label,'parameters':d['leg_response'],'tail_s':d['tail_s'],'result':result,'audit':audit,'decision_path':str(dp),'video':None};history.append(row);save(ar/'numeric_result.json',row);save(root/'progress.json',history);print(json.dumps({'label':label,'success':result['success'],'hold':audit['max_hold_ticks'],'z':audit['final_pelvis_z_m'],'new_steps':result.get('executed_physics_steps',result['physics_steps']),'simwall':result['simulation_wall_s']}),flush=True);return row
    try:
        d,dp=decide('rollback_equivalence','equivalence_plan',old['leg_response'],old['tail_s'])
        if d['action']!='run':status=d['action']
        else:
            eq=execute('equivalence',d,dp);equivalence=exact_rollout(parent,eq['result']['experiment_path']);save(root/'rollback_equivalence.json',equivalence)
            if not equivalence['exact']:raise ValueError('Rollback differs from full source rollout; stop')
            for i in range(3):
                d,dp=decide('repair',f'candidate_plan_{i:02d}')
                if d['action']!='run':status=d['action'];break
                row=execute(f'candidate_{i:02d}',d,dp)
                if not row['result']['success']:continue
                fd,fp=decide('full_restart_verification','restart_plan',d['leg_response'],d['tail_s'],restart=True)
                if fd['action']!='run':status='restart_not_authorized_by_api';break
                full=execute('full_restart',fd,fp);match=exact_rollout(row['result']['experiment_path'],full['result']['experiment_path']);save(root/'branch_vs_full.json',match)
                if not full['result']['success'] or not match['exact']:status='branch_success_full_restart_failed';break
                rep=execute('token_replay',fd,fp,Path(full['result']['experiment_path']));proof=replay_audit(full['result'],rep['result'])
                with np.load(Path(full['result']['experiment_path'])/'execution/hand_commands.npz') as x,np.load(Path(rep['result']['experiment_path'])/'execution/hand_commands.npz') as y:proof['hands_exact']=arrays_exact({k:x[k] for k in x.files},{k:y[k] for k in y.files})
                if rep['result']['success'] and proof['exact'] and proof['hands_exact']:status='success_exact_full_restart_and_token_replay'
                else:status='token_replay_failed'
                break
    except Exception as exc:
        import traceback
        error={'error':str(exc),'traceback':traceback.format_exc()};save(root/'ERROR.json',error);status='error'
    physics_api_done=now();numeric_wall=time.monotonic()-tick
    # Render only verification baseline + success chain or best unsuccessful candidate.
    selected=[h for h in history if h['label'] in ('equivalence','full_restart','token_replay') or h['result']['success']]
    candidates=[h for h in history if h['label'].startswith('candidate')]
    if candidates and not any(h['result']['success'] for h in candidates):
        best=min(candidates,key=lambda h:np.linalg.norm(h['audit'].get('final_mean_error_rad',[1e9])));selected.append(best)
    for row in selected:
        try:
            rt=time.monotonic();c=Path(row['result']['experiment_path']);ins=inspect_case(c,c/'inspection');row.update(rendering_wall_s=time.monotonic()-rt,inspection=ins,video=shallow_video(root,c,ins,{'equivalence':'回溯一致性验证','full_restart':'从头成功验证','token_replay':'独立token回放'}.get(row['label'],'闭环尝试_'+row['label'].split('_')[-1])));save(root/'progress.json',history)
        except Exception as exc:row['render_error']=str(exc)
    video_index=[]
    for row in history:
        if row.get('video'):
            v=Path(row['video']);srcvideo=Path(row['inspection']['video']);assert file_hash(v)==file_hash(srcvideo);video_index.append({'video':str(v),'source':str(srcvideo),'sha256':file_hash(v)})
    save(root/'videos.json',video_index)
    calls=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')];summary={'experiment_id':root.name,'started_at':start,'numeric_ended_at':physics_api_done,'ended_at':now(),'numeric_wall_s':numeric_wall,'wall_s':time.monotonic()-tick,'status':status,'error':error,'rollback_equivalence':equivalence,'token_proof':proof,'attempts':history,'api_requests':len(calls),'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'unknown_usage_calls':sum(c.get('usage') is None for c in calls),'api_wall_s':sum(c.get('wall_s') or 0 for c in calls),'training_ready':False};save(root/'summary.json',summary);return summary
