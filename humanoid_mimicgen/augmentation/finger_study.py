"""API-led finger-only button trials with compact diagnostics and exact action replay."""
import copy,json,time
from pathlib import Path
import numpy as np
from .agent import save
from .contact_review import tools
from .encoder_wrist_study import SCHEMA as WRIST_SCHEMA,replay_audit
from .quality_suite import api_decide,validate_decision,PROPERTIES
from .finger_press import validate_fingers
from .suite_provider import configured_provider
from .contact_batch import BudgetProvider
from .suite_physics import physical_case,check_frozen
from .suite_evidence import install
from .suite_inspect import inspect_case
from .saved_token_replay import verify_source_contract
from ..workflow.contracts import now

SKILL=Path(__file__).parent/'skills/finger_press/SKILL.md'
NUM={'type':'number'};STR={'type':'string'}
JP={'name':STR,'offset_rad':NUM,'force_gain_rad_per_N_s':NUM}
FP={'start_frame':{'type':'integer'},'ramp_s':NUM,'max_offset_rad':NUM,'max_rate_rad_s':NUM,'force_band_N':{'type':'array','items':NUM,'minItems':2,'maxItems':2},'joints':{'type':'array','minItems':1,'maxItems':2,'items':{'type':'object','properties':JP,'required':list(JP),'additionalProperties':False}}}
SCHEMA=copy.deepcopy(WRIST_SCHEMA)|{'finger_control':{'type':'object','properties':FP,'required':list(FP),'additionalProperties':False}}

def brief(case):
 case=Path(case);r=json.loads((case/'result.json').read_text());m=json.loads((case/'execution/metrics.json').read_text());start=r['planned_prefix_steps'];picks=sorted(set([start,min(start+20,len(m)-1),min(start+60,len(m)-1),len(m)-1]))
 e={'case':str(case),'success':r['success'],'independent_success':r['independent_success'],'coverage':r['coverage'],'failure':r['failure'],'max_downward_force_N':r['max_downward_force_N'],'longest_press_s':r['evidence_summary']['longest_press_s'],'max_hold':r['evidence_summary']['max_hold_counts'],'samples':[{'frame':i,'downward_force_N':m[i].get('downward_force_N'),'hold':m[i]['hold_counts'],'robot_table_N':m[i]['robot_table_N']} for i in picks]}
 ip=case/'inspection/inspection.json'
 if ip.exists():
  ins=json.loads(ip.read_text());e['contact_pairs']=ins['contact_pairs']
 if (case/'execution/finger_control.json').exists():
  f=json.loads((case/'execution/finger_control.json').read_text());e['finger_samples']=[f[i] for i in picks if i<len(f)]
 return e

def validate(d,fixed,names,frames,joint_wrist=False):
 if set(d)!=set(SCHEMA):raise ValueError('Unexpected plan fields')
 validate_decision({k:d[k] for k in PROPERTIES},[fixed['episode']]);validate_fingers(d['finger_control'],names,frames)
 if joint_wrist:
  from .encoder_wrist import validate_parameters
  validate_parameters(d['preencoder'])
  if len(d['finger_control']['joints'])!=2:raise ValueError('Joint study requires both index joints')
 mutable=('action','rationale','hypothesis')+(('preencoder',) if joint_wrist else ())
 for k in fixed:
  if k not in mutable and d[k]!=fixed[k]:raise ValueError('Fixed field changed: '+k)
 if d['action'] not in ('run','reject','needs_tool'):raise ValueError('Invalid action')
 return d

def run(config,provider_config):
 install();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True);start=now();tick=time.monotonic();history=[];status='budget_exhausted';error=None
 joint_wrist=config.get('joint_wrist',False);skill=SKILL.parent.parent/'wrist_finger_joint/SKILL.md' if joint_wrist else SKILL
 baseline=Path(config['baseline_case']);verify_source_contract(baseline);br=json.loads((baseline/'result.json').read_text());fixed=copy.deepcopy(json.loads(Path(br['decision_path']).read_text())['decision']);fixed['baseline_case']=str(baseline)
 fixed['contact_control']['contact_bodies']=['right_hand_index_0_link','right_hand_index_1_link']
 geometry=json.loads(Path(config['geometry']).read_text());provider=BudgetProvider(configured_provider(provider_config,root/'.api_rotation.json'),time.monotonic()+1500)
 fixed_context={k:v for k,v in fixed.items() if k not in ('action','rationale','hypothesis')}
 try:
  for i in range(3):
   ar=root/f'attempt_{i:02d}';ar.mkdir();context={'stage':'finger_plan' if i==0 else 'finger_retry','task':br['task'],'fixed_plan':fixed_context,'native_acceptance':{'downward_force_N_min':20,'hold_ticks':50,'tick_s':.02,'contact_distance_max_m':.001},'baseline':brief(baseline),'geometry':geometry,'prior_attempts':[{'attempt':h['attempt'],'parameters':h['parameters'],'result':h['evidence']} for h in history],'remaining_trials':3-i}
   # Only the relevant latest image, not repeated historical full-state or acceptance-code blobs.
   image_case=Path(history[-1]['case']) if history else baseline;ins=json.loads((image_case/'inspection/inspection.json').read_text());context['evidence']={'inspection':{'review_images':ins['review_images'][-1:]}}
   context['joint_wrist_enabled']=joint_wrist
   if joint_wrist:context['editable_fields']=['preencoder','finger_control'];context['elbow_interface']='mode1 has no explicit elbow target; do not claim elbow optimization'
   if config.get('prior_evidence'):context['prior_finger_only_trials']=json.loads(Path(config['prior_evidence']).read_text())
   d,dp=api_decide(provider,ar,context,[br['episode']],tools=tools(SCHEMA),validator=lambda d,ep:validate(d,fixed,geometry['hand_command_order'],br['source_frames']+br['tail_policy_ticks'],joint_wrist=joint_wrist),skill_path=skill)
   save(ar/'plan.json',d);print(json.dumps({'event':'plan','attempt':i,'parameters':({'preencoder':d['preencoder'],'finger_control':d['finger_control']} if joint_wrist else d['finger_control']),'rationale':d['rationale'],'action':d['action']},ensure_ascii=False),flush=True)
   if d['action']!='run':status=d['action'];break
   r=physical_case(config['dataset'],d['episode'],ar,'token_prefix_mode1',br['object_delta_xy_m'],decision_path=dp,preencoder=True,finger_control=True);save(ar/'physics.result.json',r);case=Path(r['experiment_path'])
   raw=np.load(case/'execution/controller.npz')['records'];trace=np.load(case/'execution/trajectory.npz');assert np.array_equal(np.repeat(raw[:,-29:],4,axis=0)[:len(trace['commands'])],trace['commands'][:,1:30])
   ins=inspect_case(case,case/'inspection');row={'attempt':i,'parameters':({'preencoder':d['preencoder'],'finger_control':d['finger_control']} if joint_wrist else d['finger_control']),'case':str(case),'result':r,'evidence':brief(case),'video':ins['video'],'rendering_wall_s':ins['rendering_and_inspection_wall_s'],'replay':None}
   # Persist physical outcome before any replay/reporting can fail.
   history.append(row);save(root/'progress.json',history)
   if r['coverage']==1:
    rr=physical_case(config['dataset'],d['episode'],ar,'saved_token_replay',br['object_delta_xy_m'],decision_path=dp,saved_token_case=case);proof=replay_audit(r,rr)
    a=np.load(case/'execution/hand_commands.npz');b=np.load(Path(rr['experiment_path'])/'execution/hand_commands.npz');proof['hands_exact']=all(np.array_equal(a[k],b[k]) for k in ('left','right'));assert proof['exact'] and proof['hands_exact']
    save(Path(rr['experiment_path'])/'token_replay_audit.json',proof);row['replay']={'result':rr,'proof':proof}
    if r['success'] and r['independent_success'] and rr['success']:
     ri=inspect_case(Path(rr['experiment_path']),Path(rr['experiment_path'])/'inspection');row['replay']['video']=ri['video'];row['rendering_wall_s']+=ri['rendering_and_inspection_wall_s'];status='success'
   save(ar/'result.json',row);save(root/'progress.json',history);print(json.dumps({'event':'attempt_end','attempt':i,'success':r['success'],'hold_s':r['evidence_summary']['longest_press_s'],'failure':r['failure'],'replay_exact':row['replay']['proof']['exact'] if row['replay'] else None}),flush=True)
   if status=='success':break
 except Exception as exc:
  import traceback
  status='error';error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};save(root/'ERROR.json',error);print(json.dumps(error),flush=True)
 calls=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')]
 s={'experiment_id':root.name,'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'status':status,'error':error,'attempts':history,'api_requests':len(calls),'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'unknown_usage_calls':sum(c.get('usage') is None for c in calls),'api_wall_s':sum(c.get('wall_s') or 0 for c in calls),'training_ready':False};save(root/'summary.json',s);return s
