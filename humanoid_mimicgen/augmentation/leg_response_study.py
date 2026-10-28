"""Serial API-configured local calibration, measured-replay feedback, token proof."""
import copy,json,time
from pathlib import Path
import numpy as np
from . import suite_physics as ph
from .leg_response import scope,identify
from .contact_review import PROPS,obj,bounded,tools
from .quality_suite import api_decide,verify_origin,PROPERTIES,validate_decision
from .suite_provider import configured_provider
from .contact_batch import BudgetProvider
from .suite_evidence import install
from .suite_inspect import inspect_case
from .contact_strategy import shallow_video
from .contact_strategy_transfer import compact
from .encoder_wrist_study import replay_audit
from .agent import save
from ..workflow.contracts import now

SKILL=Path(__file__).parent/'skills/leg_response/SKILL.md'
FIELDS={'calibration_amplitude_rad':(.02,.08),'ramp_s':(.2,1.5),'feedback_timescale_s':(.4,2.),'max_offset_rad':(.05,.3),'max_rate_rad_s':(.05,.4)}
SCHEMA=copy.deepcopy(PROPS)|{'leg_response':obj({k:bounded(*v) for k,v in FIELDS.items()})}

def validate(d,fixed,calibration=None):
    if set(d)!=set(SCHEMA):raise ValueError('Unexpected fields')
    validate_decision({k:d[k] for k in PROPERTIES},[fixed['episode']])
    for k,v in fixed.items():
        if k not in ('action','rationale','hypothesis','leg_response') and d[k]!=v:raise ValueError('Fixed field changed: '+k)
    if d['action'] not in ('run','needs_tool','reject') or d['contact_control']['joints']:raise ValueError('Unsupported action or residual')
    p=d['leg_response']
    if set(p)!=set(FIELDS):raise ValueError('Invalid response fields')
    for k,(lo,hi) in FIELDS.items():
        if type(p[k]) not in (int,float) or not np.isfinite(p[k]) or not lo<=p[k]<=hi:raise ValueError('Invalid '+k)
    if calibration and any(p[k]!=calibration[k] for k in ('calibration_amplitude_rad','ramp_s')):raise ValueError('Keep fitted calibration amplitude/ramp fixed')
    return d

def inspect_result(case):
    import mujoco
    c=Path(case);r=json.loads((c/'result.json').read_text());m=mujoco.MjModel.from_binary_path(str(c/'scene.mjb'));t=np.load(c/'execution/trajectory.npz');raw=np.load(c/'execution/controller.npz')['records'];assert np.array_equal(t['commands'][:,1:30],np.repeat(raw[:,-29:],4,axis=0)[:len(t['commands'])]);names=[s+'_'+j+'_joint' for s in ('left','right') for j in ('hip_pitch','knee','ankle_pitch')];qids=[int(m.joint(n).qposadr[0]) for n in names];actual=t['next_state'][3::4,1:1+m.nq][:,qids];met=json.loads((c/'execution/metrics.json').read_text());out={'case':str(c),'postdecoder_body_identity':True,'success':r['success'],'failure':r['failure'],'coverage':r['coverage'],'final_mean_q':np.mean(actual[-10:],axis=0).tolist(),'final_q':actual[-1].tolist(),'final_pelvis_z_m':float(t['next_state'][-1,3]),'native_final':met[-1]['_sit_metrics'],'max_hold_ticks':max(x['hold_counts']['_sit_hold_count'] for x in met),'hip_contact_force_final_N':met[-1].get('contact_control',{}).get('selected_force_N')}
    if (c/'measured_goal.npz').exists():
        goal=np.load(c/'measured_goal.npz')['q'];out['goal_final_mean_q']=np.mean(goal[-10:],axis=0).tolist();out['final_mean_error_rad']=(np.mean(goal[-10:],axis=0)-np.mean(actual[-10:],axis=0)).tolist();logs=json.loads((c/'execution/leg_response.json').read_text());enc=raw[:,-1791:-29];ids=[0,3,4,6,9,10]
        for row in logs:
            i=row['frame'];assert np.array_equal(enc[i,661:781].reshape(10,12)[:,ids],np.array(row['encoder_q']));assert np.array_equal(enc[i,781:901].reshape(10,12)[:,ids],np.array(row['encoder_dq']))
        out.update(encoder_inputs_exact=True,saturation_ticks=sum(x['saturated'] for x in logs),joint_limit_clipped_ticks=sum(x['joint_limit_clipped'] for x in logs),final_offset_paired_rad=logs[-1]['offset_paired_rad'] if logs else None)
    save(c/'response_audit.json',out);return out

def run(config,provider_config):
    install();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True);start=now();tick=time.monotonic();spec=config['task'];src=spec['source'];old=json.loads(Path(src['decision_path']).read_text())['decision'];fixed={k:copy.deepcopy(old[k]) for k in PROPS};fixed['contact_control']['joints']=[];history=[];status='budget_exhausted';error=None;model=None
    provider=BudgetProvider(configured_provider(provider_config,root/'.api_rotation.json'),time.monotonic()+2400)
    def decide(stage,folder,calibration=None):
        folder.mkdir();context={'stage':stage,'task':spec['task'],'fixed_plan':{k:v for k,v in fixed.items() if k not in ('action','rationale','hypothesis')},'measured_replay_case':config['measured_replay_case'],'reference_semantics':'actual q/dq from native-success original-token replay; source shift0, test shift+5mm','source':compact(src),'model':model,'previous':[{'variant':h['variant'],'audit':h['audit']} for h in history],'calibration_parameters':calibration,'remaining_feedback_candidates':2 if stage=='calibration' else 2-sum(h['variant']['kind']=='feedback' for h in history)}
        d,dp=api_decide(provider,folder,context,[src['episode']],tools=tools(SCHEMA),validator=lambda d,eps:validate(d,fixed,calibration),skill_path=SKILL);save(folder/'plan.json',d);return d,dp
    def execute(label,d,dp,variant):
        ar=root/label;ar.mkdir();verify_origin(dp)
        with scope(config,d,variant,model):result=ph.physical_case(spec['dataset'],src['episode'],ar,'token_prefix_mode1',src['object_delta_xy_m'],decision_path=dp,contact_control=True)
        c=Path(result['experiment_path']);audit=inspect_result(c)
        if history:
            b=Path(history[0]['result']['experiment_path']);bt=np.load(b/'execution/trajectory.npz');ct=np.load(c/'execution/trajectory.npz');n=src['planned_prefix_steps']*4;assert np.array_equal(bt['state'][:n],ct['state'][:n]) and np.array_equal(bt['commands'][:n],ct['commands'][:n])
        rt=time.monotonic();ins=inspect_case(c,c/'inspection');h={'variant':variant,'parameters':d['leg_response'],'result':result,'audit':audit,'inspection':ins,'rendering_wall_s':time.monotonic()-rt,'video':shallow_video(root,c,ins,label),'replay':None};history.append(h);save(ar/'result.json',h);save(root/'progress.json',history);print(json.dumps({'label':label,'success':result['success'],'pelvis':audit['final_pelvis_z_m'],'hold':audit['max_hold_ticks']}),flush=True)
        if result['success']:
            with scope(config,d,variant,model,replay=True):rr=ph.physical_case(spec['dataset'],src['episode'],ar,'saved_token_replay',src['object_delta_xy_m'],decision_path=dp,saved_token_case=c)
            rc=Path(rr['experiment_path']);proof=replay_audit(result,rr)
            with np.load(c/'execution/hand_commands.npz') as x,np.load(rc/'execution/hand_commands.npz') as y:proof['hands_exact']=all(np.array_equal(x[k],y[k]) for k in ('left','right'))
            ra=inspect_result(rc);rt=time.monotonic();ri=inspect_case(rc,rc/'inspection');h['replay']={'result':rr,'proof':proof,'audit':ra,'inspection':ri,'rendering_wall_s':time.monotonic()-rt,'video':shallow_video(root,rc,ri,'独立token回放')};save(ar/'result.json',h);save(root/'progress.json',history)
        return h
    try:
        d,dp=decide('calibration',root/'calibration_plan')
        if d['action']!='run':status=d['action']
        else:
            variants=[('零扰动标定',[0.,0.,0.])]
            eps=d['leg_response']['calibration_amplitude_rad']
            for k,name in enumerate(['髋','膝','踝']):
                for sign in [1,-1]:v=np.zeros(3);v[k]=sign*eps;variants.append((name+('正扰动' if sign>0 else '负扰动'),v.tolist()))
            for i,(name,vector) in enumerate(variants):
                h=execute(f'calibration_{i:02d}_{name}',d,dp,{'kind':'probe','vector':vector})
                if h['replay'] and h['replay']['result']['success'] and h['replay']['proof']['exact'] and h['replay']['proof']['hands_exact']:status='calibration_success_exact_replay';break
                if h['result']['coverage']!=1 or h['result']['failure']:raise ValueError('Calibration stopped/partial; cannot fit response')
            if status!='calibration_success_exact_replay':
                plus=[history[1+2*k]['audit']['final_mean_q'] for k in range(3)];minus=[history[2+2*k]['audit']['final_mean_q'] for k in range(3)];model=identify(plus,minus,eps);model.update(source_cases=[h['result']['experiment_path'] for h in history],baseline_q=history[0]['audit']['final_mean_q'],goal_q=history[0]['audit']['goal_final_mean_q'],calibration_parameters=d['leg_response']);save(root/'response_model.json',model)
                if model['rank']==0:status='no_observable_response'
                else:
                    for i in range(2):
                        fd,fp=decide('feedback',root/f'feedback_plan_{i:02d}',d['leg_response'])
                        if fd['action']!='run':status=fd['action'];break
                        h=execute(f'feedback_{i:02d}',fd,fp,{'kind':'feedback'})
                        if h['replay'] and h['replay']['result']['success'] and h['replay']['proof']['exact'] and h['replay']['proof']['hands_exact']:status='feedback_success_exact_replay';break
    except Exception as exc:
        import traceback
        error={'error':str(exc),'traceback':traceback.format_exc()};save(root/'ERROR.json',error);status='error'
    calls=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')];s={'experiment_id':root.name,'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'status':status,'error':error,'attempts':history,'model':model,'api_requests':len(calls),'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'unknown_usage_calls':sum(c.get('usage') is None for c in calls),'api_wall_s':sum(c.get('wall_s') or 0 for c in calls),'training_ready':False};save(root/'summary.json',s);return s
