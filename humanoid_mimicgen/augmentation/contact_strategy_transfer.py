"""Cross-task API forward test of contact-strategy reasoning with existing mode1 capabilities."""
import copy,json,time,shutil
from pathlib import Path
import numpy as np
from .contact_strategy import SKILL,STRATEGY,validate_strategy,shallow_video
from .contact_review import PROPS,tools
from .quality_suite import api_decide,validate_decision,PROPERTIES,verify_origin
from .suite_provider import configured_provider
from .contact_batch import BudgetProvider
from .suite_evidence import install
from . import suite_physics as ph
from .suite_inspect import inspect_case
from .encoder_wrist_study import replay_audit
from .agent import save
from ..workflow.contracts import now,file_hash
SCHEMA=copy.deepcopy(PROPS)|{'strategy':{'type':'object','properties':STRATEGY,'required':list(STRATEGY),'additionalProperties':False}}

def compact(r):
 e=r.get('evidence_summary',{});samples=e.get('samples',[])
 return {'case':r['experiment_path'],'success':r['success'],'failure':r['failure'],'coverage':r['coverage'],'summary':{k:v for k,v in e.items() if k!='samples'},'samples':[samples[i] for i in sorted(set([0,len(samples)//2,len(samples)-1]))] if samples else [],'native_acceptance':{k:v for k,v in (r.get('native_acceptance') or {}).items() if k!='reviewed_shadow'}}

def validate(d,fixed,bodies,frames):
 if set(d)!=set(SCHEMA):raise ValueError('Unexpected fields')
 validate_decision({k:d[k] for k in PROPERTIES},[fixed['episode']]);validate_strategy(d['strategy'],bodies)
 for k,v in fixed.items():
  if k not in ['action','rationale','hypothesis','tail_s','wrist_delta_m','contact_end'] and d[k]!=v:raise ValueError('Fixed field changed: '+k)
 if d['contact_control']['joints']:raise ValueError('Postdecoder corrections forbidden')
 if not d['contact_start']<=d['contact_end']<frames+round(d['tail_s']/.02):raise ValueError('Invalid phase end')
 if d['action'] not in ['run','needs_tool','reject']:raise ValueError('Invalid action')
 return d

def run(config,provider_config):
 import mujoco
 install();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True);start=now();tick=time.monotonic();results=[];writer=ph.write_new;active={}
 gate=json.loads(Path(config['button_gate']).read_text());assert gate['status']=='physical_success_exact_replay';assert any(x['audit']['physical_gate'] and x['replay']['proof']['exact'] and x['replay']['proof']['hands_exact'] for x in gate['attempts'])
 def write(path,value):
  if Path(path).name=='experiment.json':
   value=copy.deepcopy(value);value['pipeline'].update(entrypoint='contact_strategy_transfer.run',feedback_and_reflection_rules='API contact-strategy reasoning; no body residual or gain changes; contact telemetry and unchanged safety',allowed_retry_changes=['API-selected common bilateral wrist translation<=5cm and tail<=3s']);value['purpose']['planned_changes'].append({'strategy_skill_transfer':True,'body_postdecoder_identity':True});value['control']['stage_interfaces'][-1]['joint_residual_interface']={'interface':'identity: empty joint list; contact observer/safety only'} if 'joint_residual_interface' in value['control']['stage_interfaces'][-1] else {'interface':'saved token with fixed base PD'}
   p=Path(config['config_path']);value['preflight']['evidence_paths'].append(str(p));value['preflight']['evidence_sha256'][str(p)]=file_hash(p);shutil.copy2(p,Path(path).parent/'source_snapshot'/'transfer_config.json')
  return writer(path,value)
 ph.write_new=write
 try:
  for spec in config['tasks']:
   tr=root/spec['task'];tr.mkdir();history=[];status='budget_exhausted';error=None;src=spec['source'];fixed=copy.deepcopy(json.loads(Path(src['decision_path']).read_text())['decision']);fixed['contact_control']['joints']=[];m=mujoco.MjModel.from_binary_path(str(Path(src['experiment_path'])/'scene.mjb'));bodies=[m.body(i).name for i in range(m.nbody) if m.body(i).name];provider=BudgetProvider(configured_provider(provider_config,root/'.api_rotation.json'),time.monotonic()+900)
   try:
    for i in range(2):
     ar=tr/f'attempt_{i:02d}';ar.mkdir();context={'task':spec['task'],'capability_profile':'NONBUTTON existing mode1 paired Cartesian reference only. Button-specific preencoder/finger schema does not apply. Editable physical fields: wrist_delta_m applied to BOTH wrists uniformly during contact_start..contact_end, contact_end, tail_s0..3. Other fixed fields unchanged. No postdecoder correction. Source hand commands fixed. No elbow/independent-finger/rotation control in this cross-task profile. Do not claim unavailable controls.','fixed_plan':{k:v for k,v in fixed.items() if k not in ['action','rationale','hypothesis']},'available_bodies':bodies,'source':compact(src),'previous':[{'parameters':h['parameters'],'result':compact(h['result'])} for h in history],'remaining_trials':2-i,'objective':'Fresh native task success and exact saved-token+hands replay; do NOT use button20N gate. If source already succeeds, reuse is permitted but fresh execution required. Diagnose task-specific failure, choose supported repair or explain capability blocker. Native thresholds fixed.'}
     d,dp=api_decide(provider,ar,context,[src['episode']],tools=tools(SCHEMA),validator=lambda d,eps:validate(d,fixed,bodies,src['source_frames']),skill_path=SKILL);save(ar/'plan.json',d)
     if d['action']!='run':status=d['action'];save(ar/'not_executed.json',d);break
     r=ph.physical_case(spec['dataset'],src['episode'],ar,'token_prefix_mode1',src['object_delta_xy_m'],decision_path=dp,contact_control=True);save(ar/'physics.result.json',r);case=Path(r['experiment_path']);raw=np.load(case/'execution/controller.npz')['records'];t=np.load(case/'execution/trajectory.npz');assert np.array_equal(t['commands'][:,1:30],np.repeat(raw[:,-29:],4,axis=0)[:len(t['commands'])]);ins=inspect_case(case,case/'inspection');row={'parameters':d,'result':r,'inspection':ins,'video':shallow_video(root,case,ins,'生成'),'replay':None};history.append(row);save(tr/'progress.json',history)
     if r['success']:
      rr=ph.physical_case(spec['dataset'],src['episode'],ar,'saved_token_replay',src['object_delta_xy_m'],decision_path=dp,saved_token_case=case);rc=Path(rr['experiment_path']);proof=replay_audit(r,rr)
      with np.load(case/'execution/hand_commands.npz') as a,np.load(rc/'execution/hand_commands.npz') as b:proof['hands_exact']=all(np.array_equal(a[k],b[k]) for k in ['left','right'])
      ri=inspect_case(rc,rc/'inspection');row['replay']={'result':rr,'proof':proof,'inspection':ri,'video':shallow_video(root,rc,ri,'回放')}
      if rr['success'] and proof['exact'] and proof['hands_exact']:status='native_success_exact_replay'
     save(ar/'result.json',row);save(tr/'progress.json',history);print(json.dumps({'task':spec['task'],'attempt':i,'success':r['success'],'status':status}),flush=True)
     if status=='native_success_exact_replay':break
   except Exception as exc:
    import traceback
    status='error';error={'error':str(exc),'traceback':traceback.format_exc()};save(tr/'ERROR.json',error)
   result={'task':spec['task'],'status':status,'error':error,'attempts':history};results.append(result);save(tr/'summary.json',result);save(root/'progress.json',results)
 finally:ph.write_new=writer
 calls=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')];summary={'experiment_id':root.name,'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'results':results,'api_requests':len(calls),'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'unknown_usage_calls':sum(c.get('usage') is None for c in calls),'api_wall_s':sum(c.get('wall_s') or 0 for c in calls),'training_ready':False};save(root/'summary.json',summary);return summary
