"""API-controlled isolated chair preencoder-leg pilot with a fresh zero baseline."""
import copy,json,time
from pathlib import Path
import numpy as np
from . import suite_physics as ph
from .encoder_legs import JOINTS,validate as validate_legs,reference_scope,tracking_rmse
from .contact_review import PROPS,tools,obj,bounded
from .contact_strategy_transfer import compact
from .contact_strategy import shallow_video
from .quality_suite import api_decide,verify_origin,validate_decision,PROPERTIES
from .suite_provider import configured_provider
from .contact_batch import BudgetProvider
from .suite_evidence import install
from .suite_inspect import inspect_case
from .encoder_wrist_study import replay_audit
from .agent import save
from ..workflow.contracts import now,file_hash

SKILL=Path(__file__).parent/'skills/encoder_legs/SKILL.md'
SCHEMA=copy.deepcopy(PROPS)|{'leg_reference':obj({'blend':{'type':'array','items':bounded(0,1),'minItems':6,'maxItems':6},'offset_rad':{'type':'array','items':bounded(-.3,.3),'minItems':6,'maxItems':6},'ramp_s':bounded(.2,2)})}

def validate_plan(d,fixed,baseline=False):
    if set(d)!=set(SCHEMA):raise ValueError('Unexpected plan fields')
    validate_decision({k:d[k] for k in PROPERTIES},[fixed['episode']]);validate_legs(d['leg_reference'])
    for k,v in fixed.items():
        if k not in ('action','rationale','hypothesis','leg_reference') and d[k]!=v:raise ValueError('Fixed field changed: '+k)
    if d['action'] not in ('run','needs_tool','reject') or d['contact_control']['joints']:raise ValueError('Invalid action or forbidden postdecoder residual')
    if baseline and (any(d['leg_reference']['blend']) or any(d['leg_reference']['offset_rad'])):raise ValueError('Baseline must have zero reference correction')
    return d

def audit(case):
    import mujoco
    case=Path(case);r=json.loads((case/'result.json').read_text());m=mujoco.MjModel.from_binary_path(str(case/'scene.mjb'));t=np.load(case/'execution/trajectory.npz');raw=np.load(case/'execution/controller.npz')['records'];identity=np.array_equal(t['commands'][:,1:30],np.repeat(raw[:,-29:],4,axis=0)[:len(t['commands'])]);assert identity
    out={'case':str(case),'postdecoder_body_identity':identity,'native_success':r['success'],'coverage':r['coverage'],'failure':r['failure'],'original_token_steps':r['original_token_steps'],'encoder_modes':r['encoder_modes']}
    p=case/'leg_reference.npz'
    if p.exists() and len(t['next_state'])>=4:
        a=np.load(p);ids=a['indices'];names=a['names'].tolist();qids=[int(m.joint(n).qposadr[0]) for n in names];actual=t['next_state'][3::4,1:1+m.nq][:,qids];n=len(actual);start=r['planned_prefix_steps'];ref=a['corrected_q'][:n];delta=ref-a['source_q'][:n]
        out.update(joint_order=list(JOINTS),reference_correction_max_rad=np.max(np.abs(delta[:,ids]),axis=0).tolist(),tracking_rmse_rad=tracking_rmse(actual[:,ids],ref[:,ids],start),final_reference_rad=ref[-1,ids].tolist(),final_measured_rad=actual[-1,ids].tolist(),final_source_rad=a['source_q'][n-1,ids].tolist(),final_success_reference_rad=a['reference_q'][n-1,ids].tolist(),final_pelvis_z_m=float(t['next_state'][-1,3]))
        # Saved-state contact reconstruction, no physics integration; all foot contacts kept by pair.
        d=mujoco.MjData(m);samples=[]
        for frame in sorted(set([min(start,n-1),n-1])):
            st=t['next_state'][frame*4+3];d.qpos[:]=st[1:1+m.nq];d.qvel[:]=st[1+m.nq:];d.ctrl[:]=t['commands'][frame*4+3,-m.nu:];mujoco.mj_forward(m,d);pairs={}
            for ci,c in enumerate(d.contact):
                bs=[m.body(int(m.geom_bodyid[int(g)])).name for g in (c.geom1,c.geom2)]
                if not any('ankle' in b for b in bs):continue
                f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);key=' / '.join(bs);pairs[key]=pairs.get(key,0)+max(0,float(f[0]))
            samples.append({'frame':frame,'foot_pair_normal_N':pairs})
        out['foot_contact_samples']=samples
        enc=raw[:,-1791:-29]
        for i in range(start,n):
            future=np.minimum(i+np.arange(10)*5,len(a['corrected_q'])-1)
            assert np.array_equal(enc[i,661:781],a['corrected_q'][future,:12].astype(np.float32).reshape(-1))
            assert np.array_equal(enc[i,781:901],a['corrected_dq'][future,:12].astype(np.float32).reshape(-1))
        out['encoder_leg_inputs_exact']=True
    save(case/'leg_audit.json',out);return out

def run(config,provider_config):
    install();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True);start=now();tick=time.monotonic();spec=config['task'];src=spec['source'];fixed=copy.deepcopy(json.loads(Path(src['decision_path']).read_text())['decision']);fixed.pop('strategy',None);fixed['contact_control']['joints']=[]
    fixed['leg_reference']={'blend':[0]*6,'offset_rad':[0]*6,'ramp_s':.4}
    provider=BudgetProvider(configured_provider(provider_config,root/'.api_rotation.json'),time.monotonic()+1800);history=[];status='budget_exhausted';error=None
    try:
        for i in range(4):
            ar=root/f'attempt_{i:02d}';ar.mkdir();baseline=i==0
            context={'stage':'baseline' if baseline else 'repair','task':spec['task'],'fixed_plan':{k:v for k,v in fixed.items() if k not in ('rationale','hypothesis','action')},'joint_order':list(JOINTS),'source':compact(src),'successful_reference':{'case':config['leg_reference_case'],'native_success':True,'historical_interface':'six-joint postdecoder0.8 reference blend; NOT current executable control','token_alone_replays_execution':False},'previous':[{'parameters':x['parameters'],'result':compact(x['result']) if x.get('result') else None,'audit':x.get('audit'),'preflight_error':x.get('preflight_error')} for x in history],'remaining_repair_decisions':3 if baseline else 4-i,'objective':'Fresh unchanged native sitting success with independent saved token+hands reproduction; baseline must be zero correction; body decoder and PD unchanged'}
            d,dp=api_decide(provider,ar,context,[src['episode']],tools=tools(SCHEMA),validator=lambda d,eps:validate_plan(d,fixed,baseline),skill_path=SKILL);save(ar/'plan.json',d)
            if d['action']!='run':status=d['action'];break
            verify_origin(dp)
            try:
                with reference_scope(config,d):result=ph.physical_case(spec['dataset'],src['episode'],ar,'token_prefix_mode1',src['object_delta_xy_m'],decision_path=dp,contact_control=True)
            except ValueError as exc:
                # Only pre-execution failures may be fed back as preflight errors.
                cases=list((ar/'experiments').glob('*'))
                if any((c/'experiment.json').exists() for c in cases):raise
                row={'parameters':d,'preflight_error':str(exc),'result':None};history.append(row);save(ar/'preflight_error.json',row);save(root/'progress.json',history)
                if baseline:raise
                continue
            case=Path(result['experiment_path']);a=audit(case);render_tick=time.monotonic();ins=inspect_case(case,case/'inspection');render_s=time.monotonic()-render_tick;row={'baseline':baseline,'parameters':d,'result':result,'audit':a,'inspection':ins,'rendering_wall_s':render_s,'video':shallow_video(root,case,ins,'零修正对照' if baseline else '腿部参考修正'),'replay':None};history.append(row);save(root/'progress.json',history)
            if baseline:
                with np.load(case/'leg_reference.npz') as ref:assert np.array_equal(ref['source_q'],ref['corrected_q']) and np.array_equal(ref['source_dq'],ref['corrected_dq'])
            if result['success']:
                with reference_scope(config,d,enabled=False):rr=ph.physical_case(spec['dataset'],src['episode'],ar,'saved_token_replay',src['object_delta_xy_m'],decision_path=dp,saved_token_case=case)
                rc=Path(rr['experiment_path']);proof=replay_audit(result,rr)
                with np.load(case/'execution/hand_commands.npz') as x,np.load(rc/'execution/hand_commands.npz') as y:proof['hands_exact']=all(np.array_equal(x[k],y[k]) for k in ['left','right'])
                ra=audit(rc);render_tick=time.monotonic();ri=inspect_case(rc,rc/'inspection');row['replay']={'result':rr,'proof':proof,'audit':ra,'inspection':ri,'rendering_wall_s':time.monotonic()-render_tick,'video':shallow_video(root,rc,ri,'独立回放')}
                if rr['success'] and proof['exact'] and proof['hands_exact']:status='baseline_success_exact_replay' if baseline else 'native_success_exact_replay'
            save(ar/'result.json',row);save(root/'progress.json',history);print(json.dumps({'attempt':i,'baseline':baseline,'success':result['success'],'status':status}),flush=True)
            if status.endswith('success_exact_replay'):break
    except Exception as exc:
        import traceback
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};save(root/'ERROR.json',error);status='error'
    calls=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')];summary={'experiment_id':root.name,'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'status':status,'error':error,'attempts':history,'api_requests':len(calls),'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'unknown_usage_calls':sum(c.get('usage') is None for c in calls),'api_wall_s':sum(c.get('wall_s') or 0 for c in calls),'training_ready':False};save(root/'summary.json',summary);return summary
