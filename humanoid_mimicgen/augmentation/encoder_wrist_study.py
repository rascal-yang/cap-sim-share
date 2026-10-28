"""Bounded API-led encoder-only button pilot followed by saved-token regression."""
import copy,json,time
from pathlib import Path
import numpy as np
from .agent import save
from .contact_review import PROPS,tools
from .quality_suite import api_decide,validate_decision,PROPERTIES
from .encoder_wrist import validate_parameters
from .contact_control import validate_contact_control
from .retry_state import validate_resume
from .suite_provider import configured_provider
from .contact_batch import BudgetProvider
from .suite_physics import physical_case,check_frozen
from .suite_evidence import install
from .suite_inspect import inspect_case
from .direct_augmentation import compact_case
from .saved_token_replay import extract_tokens,verify_source_contract
from ..workflow.contracts import now,file_hash

SKILL=Path(__file__).parent/'skills/encoder_wrist/SKILL.md'
NUM={'type':'number'}
VECTOR={'type':'array','items':NUM,'minItems':3,'maxItems':3}
PRE={'reference_blend':NUM,'translation_m':VECTOR,'rotation_vector_rad':VECTOR,'ramp_s':NUM,'force_band_N':{'type':'array','items':NUM,'minItems':2,'maxItems':2},'force_gain_m_per_N_s':VECTOR,'max_feedback_m':NUM,'feedback_rate_m_s':NUM}
SCHEMA=copy.deepcopy(PROPS)|{'baseline_case':{'type':'string'},'preencoder':{'type':'object','properties':PRE,'required':list(PRE),'additionalProperties':False}}

def validate_plan(d,fixed):
    if set(d)!=set(SCHEMA):raise ValueError('Unexpected plan fields')
    validate_decision({k:d[k] for k in PROPERTIES},[fixed['episode']]);validate_parameters(d['preencoder']);validate_resume(d['resume'],[])
    for k in ('episode','tail_s','contact_start','contact_end','wrist_delta_m','safety_guard','reference_case','criteria_decision','baseline_case','contact_control'):
        if d[k]!=fixed[k]:raise ValueError('Fixed field changed: '+k)
    if d['action'] not in ('run','needs_tool','reject'):raise ValueError('Invalid plan action')
    return d

def evidence(case):
    case=Path(case);e=compact_case(case);metrics=json.loads((case/'execution/metrics.json').read_text())
    picks=sorted(set(range(0,len(metrics),20))|{len(metrics)-1})
    e['force_and_hold_samples']=[{'frame':i,'force_N':metrics[i].get('downward_force_N'),'hold':metrics[i]['hold_counts']} for i in picks]
    ip=case/'inspection/inspection.json'
    if ip.exists():
        ins=json.loads(ip.read_text());e['contact_pairs']=ins['contact_pairs'];e['inspection']={'review_images':ins['review_images'][-1:]}
    p=case/'execution/preencoder.json'
    if p.exists():
        rows=json.loads(p.read_text());e['preencoder_samples']=rows[::20]+rows[-1:]
    return e

def replay_audit(source,replay):
    src=Path(source['experiment_path']);dst=Path(replay['experiment_path']);check_frozen(dst)
    a=np.load(src/'execution/controller.npz')['records'];b=np.load(dst/'execution/controller.npz')['records']
    t=np.load(src/'execution/trajectory.npz');u=np.load(dst/'execution/trajectory.npz')
    proof={'tokens_exact':bool(np.array_equal(extract_tokens(a,len(a)),extract_tokens(b,len(b)))),'state_exact':bool(np.array_equal(t['state'],u['state'])),'next_state_exact':bool(np.array_equal(t['next_state'],u['next_state'])),'commands_exact':bool(np.array_equal(t['commands'],u['commands'])),'source_success':source['success'],'replay_success':replay['success'],'source':str(src),'replay':str(dst)}
    proof['exact']=all(proof[k] for k in ('tokens_exact','state_exact','next_state_exact','commands_exact'))
    save(dst/'token_replay_audit.json',proof);return proof

def run(config,provider_config):
    install();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True);start=now();tick=time.monotonic()
    source=Path(config['success_case']);failed=Path(config['token_failed_case']);sr=json.loads((source/'result.json').read_text());verify_source_contract(source)
    fixed=copy.deepcopy(json.loads(Path(sr['decision_path']).read_text())['decision']);fixed['contact_control']['joints']=[];fixed['baseline_case']=str(source)
    provider=BudgetProvider(configured_provider(provider_config,root/'.api_rotation.json'),time.monotonic()+1800)
    base={'stage':'encoder_only_plan','task':sr['task'],'objective':'native success and exact independent token replay with no post-decoder changes','object_shift_xy_m':sr['object_delta_xy_m'],'fixed_plan':fixed,'control_skill':str(SKILL),'source_reference':evidence(fixed['reference_case']),'successful_postdecoder_case':evidence(source),'failed_saved_token_case':evidence(failed),'max_physical_trials':4}
    history=[];status='budget_exhausted'
    for i in range(4):
        ar=root/f'attempt_{i:02d}';ar.mkdir();context=base|{'stage':'encoder_only_plan' if i==0 else 'encoder_only_retry','previous_attempts':history,'remaining_attempts':4-i}
        d,dp=api_decide(provider,ar,context,[sr['episode']],tools=tools(SCHEMA),validator=lambda d,eps:validate_plan(d,fixed),skill_path=SKILL)
        save(ar/'plan.json',d);print(json.dumps({'event':'api_plan','attempt':i,'parameters':d['preencoder'],'action':d['action'],'rationale':d['rationale']},ensure_ascii=False),flush=True)
        if d['action']!='run':status=d['action'];break
        r=physical_case(config['dataset'],d['episode'],ar,'token_prefix_mode1',sr['object_delta_xy_m'],decision_path=dp,preencoder=True);save(ar/'physics.result.json',r)
        case=Path(r['experiment_path']);raw=np.load(case/'execution/controller.npz')['records'];actual=np.load(case/'execution/trajectory.npz')['commands']
        assert np.array_equal(np.repeat(raw[:,-29:],4,axis=0)[:len(actual)],actual[:,1:30]),'Postdecoder target was changed'
        ins=inspect_case(case,case/'inspection');current=evidence(case)
        row={'attempt':i,'parameters':d['preencoder'],'result':current,'case':str(case),'video':ins['video'],'generation_wall_s':r['total_wall_s'],'generation_simulation_wall_s':r['simulation_wall_s'],'rendering_wall_s':ins['rendering_and_inspection_wall_s'],'replay':None}
        if r['coverage']==1:
            rr=physical_case(config['dataset'],d['episode'],ar,'saved_token_replay',sr['object_delta_xy_m'],decision_path=dp,saved_token_case=case);proof=replay_audit(r,rr)
            row['replay']={'result':rr,'proof':proof}
            if r['success'] and rr['success'] and proof['exact']:
                ri=inspect_case(Path(rr['experiment_path']),Path(rr['experiment_path'])/'inspection');row['replay']['video']=ri['video'];row['rendering_wall_s']+=ri['rendering_and_inspection_wall_s'];status='success'
        history.append(row);save(root/'progress.json',history);print(json.dumps({'event':'attempt_end','attempt':i,'success':r['success'],'failure':r['failure'],'hold_s':r['evidence_summary']['longest_press_s'],'replay_exact':row['replay']['proof']['exact'] if row['replay'] else None}),flush=True)
        if status=='success':break
    calls=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')]
    summary={'experiment_id':root.name,'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'status':status,'attempts':history,'api_requests':len(calls),'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'unknown_usage_calls':sum(c.get('usage') is None for c in calls),'api_wall_s':sum(c.get('wall_s') or 0 for c in calls),'training_ready':False}
    save(root/'summary.json',summary);return summary
