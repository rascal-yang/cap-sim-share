"""Equal wall-clock paired rollback benchmark (batched API planning)."""
import math


def budget_metrics(rows,budget_s,shared_cost_s):
    if not math.isfinite(budget_s) or budget_s<=0 or not math.isfinite(shared_cost_s) or shared_cost_s<0:raise ValueError('Invalid budget')
    valid=[r for r in rows if r['elapsed_s']+shared_cost_s<=budget_s]
    successes=[r['elapsed_s']+shared_cost_s for r in valid if r['success']]
    return {'attempts':len(valid),'success':bool(successes),'first_success_s':min(successes) if successes else None,'overrun_attempts':len(rows)-len(valid)}

import copy,json,time
from pathlib import Path
from contextlib import contextmanager,nullcontext
import numpy as np
from .agent import save
from . import suite_physics as ph
from . import rollback_runtime as runtime
from .rollback_plan import compile_parameters,patch_from_plan,write_compiled,verify_compiled
from .quality_suite import api_decide
from .contact_review import tools
from .suite_provider import configured_provider
from .contact_batch import BudgetProvider
from .suite_evidence import install
from .leg_fast_retry import exact_rollout
from .rollback_legs import scope as leg_scope
from .encoder_legs import observer_contract
from .encoder_wrist_study import replay_audit
from .suite_inspect import inspect_case
from .contact_strategy import shallow_video
from ..workflow.contracts import now,file_hash

SKILL=Path(__file__).parent/'skills/rollback_benchmark/SKILL.md'


def candidate_schema(profile,count=8):
    number={'type':'number'};vec={'type':'array','items':number,'minItems':3,'maxItems':3}
    fields={'tail_s':number}
    if profile=='chair':fields.update({k:number for k in ('feedback_timescale_s','max_offset_rad','max_rate_rad_s')})
    elif profile=='button':fields.update(translation_m=vec,rotation_vector_rad=vec,feedback_gain_z=number,index_offset_rad=number)
    else:fields['wrist_delta_m']=vec
    candidate={'type':'object','properties':{'parameters':{'type':'object','properties':fields,'required':list(fields),'additionalProperties':False},'rationale':{'type':'string'}},'required':['parameters','rationale'],'additionalProperties':False}
    return {'candidates':{'type':'array','items':candidate,'minItems':count,'maxItems':count}}


def validate_candidates(d,spec):
    count=spec.get('candidate_count',8)
    if set(d)!={'candidates'} or len(d['candidates'])!=count:raise ValueError(f'Exactly{count} candidates required')
    errors=[];seen={}
    for index,c in enumerate(d['candidates']):
        try:
            if set(c)!={'parameters','rationale'} or not isinstance(c['rationale'],str) or not c['rationale'].strip():raise ValueError('Invalid candidate fields')
            plan=compile_parameters(spec['profile'],spec['base_plan'],c['parameters'])
            if plan['contact_end']>=spec['source_frames']+round(plan['tail_s']/.02):raise ValueError('horizon truncates annotated contact_end')
        except ValueError as exc:errors.append(f'candidate[{index}]: {exc}')
        key=json.dumps(c.get('parameters'),sort_keys=True)
        if key in seen:errors.append(f'candidate[{index}] duplicates candidate[{seen[key]}]; supply distinct plausible parameters')
        else:seen[key]=index
    if d['candidates'][0]['parameters']!=patch_from_plan(spec['profile'],spec['base_plan']):errors.append('Candidate0 must preserve baseline exactly')
    if not errors and compile_parameters(spec['profile'],spec['base_plan'],d['candidates'][0]['parameters'])!=spec['base_plan']:errors.append('Compact baseline patch loses fixed controls')
    for index,expected in enumerate(spec.get('preserved_candidate_parameters',[])):
        if d['candidates'][index]['parameters']!=expected:errors.append(f'candidate[{index}] changed preserved prior parameters')
    if errors:raise ValueError('; '.join(errors))
    return d


@contextmanager
def evidence_scope(config,spec_path,decision_path):
    old=ph.write_new
    def write(path,value):
        name=Path(path).name
        if name=='contact_contract.json':value=observer_contract(value)
        if name=='experiment.json':
            value=copy.deepcopy(value);o=verify_compiled(decision_path)['origin']
            bound=[Path(config['config_path']),Path(spec_path),Path(o['api_batch_path'])]
            for p,h in config.get('planning_carry_evidence',{}).items():
                if file_hash(p)!=h:raise ValueError('Prior planning time evidence changed')
                bound.append(Path(p))
            for p in bound:value['preflight']['evidence_paths'].append(str(p));value['preflight']['evidence_sha256'][str(p)]=file_hash(p)
            value['pipeline'].update(entrypoint='rollback_benchmark.run',allowed_retry_changes=['authenticated compact API patch compiled locally; checkpoint chosen before earliest mutable control'],feedback_and_reflection_rules='shared precommitted candidate queue; complete controller-state restore; actual completion deadline; no online API',decision_compilation=o)
            value['control']['body_postdecoder_identity']=True
            if 'joint_residual_interface' in value['control']['stage_interfaces'][-1]:value['control']['stage_interfaces'][-1]['joint_residual_interface']=observer_contract(value['control']['stage_interfaces'][-1]['joint_residual_interface'])
            value['budget']['equal_wall_comparison']=config['benchmark']
        if name in ('result.json','execution_audit.json'):
            value.update(post_decoder_residual_calls=0,contact_observer_only=True,token_alone_replays_execution='pending_independent_replay')
            if value.get('mode')!='saved_token_replay':value['control_variant']='checkpoint_complete_mode1_preencoder_identity_body_PD'
        return old(path,value)
    ph.write_new=write
    try:yield
    finally:ph.write_new=old


def audit_result(result,profile):
    c=Path(result['experiment_path']);t=np.load(c/'execution/trajectory.npz');records=np.load(c/'execution/controller.npz')['records']
    identity=bool(np.array_equal(t['commands'][:,1:30],np.repeat(records[:,-29:],4,axis=0)[:len(t['commands'])]))
    if not identity:raise ValueError('Decoder body target modified')
    with np.load(c/'execution/hand_commands.npz') as h:
        if any(len(h[k])!=len(records) for k in ('left','right')):raise ValueError('Incomplete hands')
    out={'native_success':result['success'],'success':result['success'],'body_identity':identity,'coverage':result['coverage'],'failure':result['failure']}
    if profile=='button':
        from .rollback_button_acceptance import evaluate,PROFILE
        a=evaluate(c,c/'execution/benchmark_button_acceptance.json');out.update(success=a['revised_success'],acceptance_profile=PROFILE,button=a)
    return out


def execute(config,spec,spec_path,batch_path,index,folder,resume,replay_source=None):
    tick=time.monotonic();folder.mkdir(parents=True,exist_ok=False);dp=folder/'compiled_plan.json';d=write_compiled(dp,batch_path,index,spec_path,resume);state={'offset':np.zeros(3)} if spec['profile']=='chair' and replay_source is None else None
    model=json.loads(Path(spec['response_model_path']).read_text()) if spec['profile']=='chair' else None
    legconfig={'config_path':config['config_path'],'measured_replay_case':spec.get('measured_replay_case'),'task':{'source':{'object_delta_xy_m':spec['shift']}}}
    with evidence_scope(config,spec_path,dp):
        with leg_scope(legconfig,d,{'kind':'feedback'},model,replay=replay_source is not None,state=state) if spec['profile']=='chair' else nullcontext():
            result=runtime.physical_case(spec['dataset'],d['episode'],folder,'saved_token_replay' if replay_source else 'token_prefix_mode1',spec['shift'],dp,profile=spec['profile'],leg_state=state,saved_token_case=replay_source)
    audit=audit_result(result,spec['profile']);row={'candidate_index':index,'result':result,'audit':audit,'success':audit['success'],'attempt_wall_s':time.monotonic()-tick,'decision_path':str(dp)};save(folder/'trial.json',row);return row


def restart():return {'kind':'restart','case':'','frame':0,'rationale':'Matched fresh initial-state execution'}

def checkpoint(case,frame):return {'kind':'checkpoint','case':str(case),'frame':int(frame),'rationale':'Restore complete controller and physics state before any changed policy input'}


def run(config,provider_config):
    install();root=Path(config['output_root']);start=now();total_tick=time.monotonic();results=[]
    save(root/'run_started.json',{'started_at':start,'config_sha256':file_hash(config['config_path'])})
    for task_index,spec_path in enumerate(config['spec_paths']):
        spec=json.loads(Path(spec_path).read_text());tr=root/spec['profile'];tr.mkdir();task={'profile':spec['profile'],'status':'running','conditions':[],'verification':None};results.append(task);save(root/'progress.json',results)
        try:
            pr=tr/'planning';pr.mkdir();provider=BudgetProvider(configured_provider(provider_config,root/'.api_rotation.json'),time.monotonic()+900);pt=time.monotonic()
            context={'task':spec['task'],'profile':spec['profile'],'baseline_patch':patch_from_plan(spec['profile'],spec['base_plan']),'known_experience':spec['experience'],'fixed_constraints':spec['constraints'],'benchmark':config['benchmark'],'candidate_count':spec.get('candidate_count',8),'preserved_candidate_parameters':spec.get('preserved_candidate_parameters',[]),'objective':'Return requested count of plausible shared parameter patches; preserve prior prefix when supplied; candidate0 exact baseline.'}
            batch,bp=api_decide(provider,pr,context,[spec['base_plan']['episode']],tools=tools(candidate_schema(spec['profile'],spec.get('candidate_count',8))),validator=lambda d,eps:validate_candidates(d,spec),skill_path=SKILL);current_planning=time.monotonic()-pt;carry=config.get('planning_carry_s',{}).get(spec['profile'],0);planning=current_planning+carry
            save(tr/'planning_cost.json',{'wall_s':planning,'current_wall_s':current_planning,'prior_failed_request_span_s':carry,'decision_path':str(bp)})
            seed=execute(config,spec,spec_path,bp,0,tr/'seed',restart());seedcase=Path(seed['result']['experiment_path']);safe_frame=min(spec['base_plan']['contact_start'],spec['base_plan'].get('finger_control',{}).get('start_frame',10**9));task.update(planning_wall_s=planning,seed=seed,safe_checkpoint_frame=safe_frame)
            # Both early and genuinely active-controller state restoration are exact gates.
            proofs=[]
            for label,frame in [('boundary',safe_frame),('active',min(spec['base_plan']['contact_start']+40,seed['result']['physics_steps']//4-1))]:
                q=execute(config,spec,spec_path,bp,0,tr/('equivalence_'+label),checkpoint(seedcase,frame));proof=exact_rollout(seedcase,q['result']['experiment_path']);proofs.append({'frame':frame,'proof':proof,'trial':q});save(tr/'equivalence.json',proofs)
                if not proof['exact']:raise ValueError('Complete-state rollback equivalence failed at '+str(frame))
            task['equivalence']=proofs;shared=planning/config['benchmark']['order_repeats']+seed['attempt_wall_s'];task['shared_cost_s']=shared
            print(json.dumps({'task':spec['profile'],'stage':'equivalence_passed','shared_s':shared}),flush=True)
            for repeat in range(config['benchmark']['order_repeats']):
                count=spec.get('candidate_count',8);order=config.get('candidate_orders',{}).get(spec['profile'],{}).get(str(repeat))
                if order is None:order=np.random.default_rng(config['benchmark']['order_seeds'][repeat]).permutation(np.arange(1,count)).tolist()
                if sorted(order)!=list(range(1,count)):raise ValueError('Candidate order must contain each nonseed candidate exactly once')
                condition={'repeat':repeat,'candidate_order':order,'arms':{}};task['conditions'].append(condition)
                arms=['restart','rollback'] if (task_index+repeat)%2==0 else ['rollback','restart']
                for arm in arms:
                    folder=tr/f'condition_{repeat:02d}'/arm;folder.mkdir(parents=True);clock=time.monotonic();rows=[dict(seed,elapsed_s=0.,shared_seed=True)]
                    for candidate in order:
                        if time.monotonic()-clock+shared>=config['benchmark']['wall_budget_s']:break
                        row=execute(config,spec,spec_path,bp,candidate,folder/f'candidate_{candidate:02d}',restart() if arm=='restart' else checkpoint(seedcase,safe_frame));row.update(elapsed_s=time.monotonic()-clock,shared_seed=False);rows.append(row)
                        condition['arms'][arm]={'rows':rows,'metrics':budget_metrics(rows,config['benchmark']['wall_budget_s'],shared),'arm_wall_s':time.monotonic()-clock};save(root/'progress.json',results)
                        print(json.dumps({'task':spec['profile'],'repeat':repeat,'arm':arm,'candidate':candidate,'success':row['success'],'elapsed_including_shared_s':row['elapsed_s']+shared}),flush=True)
                    metrics=budget_metrics(rows,config['benchmark']['wall_budget_s'],shared);condition['arms'][arm]={'rows':rows,'metrics':metrics,'arm_wall_s':time.monotonic()-clock,'queue_exhausted':len(rows)==count,'full_planning_charge_metrics':budget_metrics(rows,config['benchmark']['wall_budget_s'],planning+seed['attempt_wall_s']),'new_in_budget_attempts':sum(not r.get('shared_seed') and r['elapsed_s']+shared<=config['benchmark']['wall_budget_s'] for r in rows),'provisional_new_successes':sum(not r.get('shared_seed') and r['elapsed_s']+shared<=config['benchmark']['wall_budget_s'] and r['success'] for r in rows),'curves':{str(b):budget_metrics(rows,b,shared) for b in config['benchmark']['report_budgets_s']}};save(root/'progress.json',results)
            winners=[row for co in task['conditions'] for arm,a in co['arms'].items() if arm=='rollback' for row in a['rows'] if row['success'] and row['elapsed_s']+shared<=config['benchmark']['wall_budget_s'] and not row.get('shared_seed')]
            if winners:
                winner=winners[0];index=winner['candidate_index'];full=execute(config,spec,spec_path,bp,index,tr/'verify_full',restart());match=exact_rollout(winner['result']['experiment_path'],full['result']['experiment_path']);rep=execute(config,spec,spec_path,bp,index,tr/'verify_token',restart(),Path(full['result']['experiment_path']));proof=replay_audit(full['result'],rep['result'])
                with np.load(Path(full['result']['experiment_path'])/'execution/hand_commands.npz') as a,np.load(Path(rep['result']['experiment_path'])/'execution/hand_commands.npz') as b:proof['hands_exact']=all(np.array_equal(a[k],b[k]) for k in ('left','right'))
                task['verification']={'winner':winner,'full':full,'branch_full_exact':match,'token_replay':rep,'token_proof':proof,'passed':bool(full['success'] and rep['success'] and match['exact'] and proof['exact'] and proof['hands_exact'])}
                if not task['verification']['passed']:raise ValueError('Winner full/token verification failed')
            task['status']='complete'
        except Exception as exc:
            import traceback
            task.update(status='error',error=str(exc),traceback=traceback.format_exc());save(tr/'ERROR.json',task);print(json.dumps({'task':spec['profile'],'error':str(exc)}),flush=True)
        save(tr/'summary.json',task);save(root/'progress.json',results)
    numeric_end=now();numeric_wall=time.monotonic()-total_tick
    videos=[]
    for task in results:
        selected=[]
        if task.get('seed'):selected.append(('初始策略',task['seed']))
        if task.get('verification'):selected.extend([('成功从头验证',task['verification']['full']),('独立token回放',task['verification']['token_replay'])])
        for label,row in selected:
            try:
                c=Path(row['result']['experiment_path']);tick=time.monotonic();ins=inspect_case(c,c/'inspection');v=shallow_video(root,c,ins,task['profile']+'_'+label);videos.append({'profile':task['profile'],'label':label,'video':v,'sha256':file_hash(v),'inspection':ins,'rendering_wall_s':time.monotonic()-tick})
            except Exception as exc:videos.append({'profile':task['profile'],'label':label,'render_error':str(exc)})
            save(root/'videos.json',videos)
    calls=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')]
    out={'experiment_id':root.name,'started_at':start,'numeric_ended_at':numeric_end,'ended_at':now(),'numeric_wall_s':numeric_wall,'total_wall_s':time.monotonic()-total_tick,'results':results,'videos':videos,'api_requests':len(calls),'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'unknown_usage_calls':sum(c.get('usage') is None for c in calls),'training_ready':False};save(root/'summary.json',out);return out
