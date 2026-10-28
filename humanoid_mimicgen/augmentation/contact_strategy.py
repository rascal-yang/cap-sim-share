"""Isolated API-led contact strategy study; native and physical evidence stay separate."""
import copy,json,time,shutil
from pathlib import Path
import numpy as np
from .agent import save
from .finger_study import SCHEMA as FINGER_SCHEMA,brief
from .encoder_wrist import validate_parameters
from .finger_press import validate_fingers
from .quality_suite import api_decide,validate_decision,PROPERTIES,verify_origin
from .contact_review import tools
from .suite_provider import configured_provider
from .contact_batch import BudgetProvider
from .suite_physics import physical_case,check_frozen
from .suite_evidence import install
from .suite_inspect import inspect_case
from .encoder_wrist_study import replay_audit
from .saved_token_replay import verify_source_contract
from ..workflow.contracts import now,file_hash

SKILL=Path(__file__).parent/'skills/contact_strategy/SKILL.md'
STR={'type':'string'}
STRATEGY={'contact_bodies':{'type':'array','items':STR,'minItems':1},'acceptance_assessment':{'type':'string','enum':['native_valid','contact_omission','uncertain']},'acceptance_reason':STR,'change_kind':{'type':'string','enum':['contact','hold','reuse']},'expected_response':STR,'falsify':STR}
SCHEMA=copy.deepcopy(FINGER_SCHEMA)|{'strategy':{'type':'object','properties':STRATEGY,'required':list(STRATEGY),'additionalProperties':False}}

def force_hold_summary(rows,threshold,dt,max_distance):
 hold=longest=0
 for row in rows:
  hold=hold+1 if row['force']>=threshold and row['distance']<=max_distance else 0
  longest=max(longest,hold)
 return {'longest_s':longest*dt,'meets_one_second':longest*dt>=1,'threshold_N':threshold,'distance_max_m':max_distance}

def validate_strategy(s,bodies):
 if set(s)!=set(STRATEGY) or not s['contact_bodies'] or not set(s['contact_bodies'])<=set(bodies):raise ValueError('Unknown strategy body/fields')
 if s['acceptance_assessment'] not in STRATEGY['acceptance_assessment']['enum'] or s['change_kind'] not in STRATEGY['change_kind']['enum']:raise ValueError('Invalid strategy assessment')
 if any(not isinstance(s[k],str) or not s[k].strip() for k in ['acceptance_reason','expected_response','falsify']):raise ValueError('Missing falsifiable strategy')
 return s

def contact_audit(case):
 """Saved-state only. Sum downward force on button, including native-omitted bodies."""
 import mujoco
 case=Path(case);m=mujoco.MjModel.from_binary_path(str(case/'scene.mjb'));d=mujoco.MjData(m);trace=np.load(case/'execution/trajectory.npz');result=json.loads((case/'result.json').read_text());rows=[];peaks={};counts={};native_keys=('palm','thumb','index','middle','ring','little','pinky')
 buttons={g for g in range(m.ngeom) if any(k in (m.geom(g).name or '') for k in ['E_button_01_13','E_button_02_6'])}
 if not buttons:raise ValueError('Button contact audit cannot assess other task objects')
 for i in range(3,len(trace['next_state']),4):
  st=trace['next_state'][i];d.qpos[:]=st[1:1+m.nq];d.qvel[:]=st[1+m.nq:];d.ctrl[:]=trace['commands'][i,-m.nu:];mujoco.mj_forward(m,d);by={};distance=float('inf')
  for ci,c in enumerate(d.contact):
   a,b=int(c.geom1),int(c.geom2);other=b if a in buttons else a if b in buttons else None
   if other is None:continue
   name=m.body(int(m.geom_bodyid[other])).name or ''
   if not name.startswith('right_') or not any(k in name for k in ['hand','wrist']):continue
   f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);world=np.asarray(c.frame).reshape(3,3).T@f[:3];down=max(0.,float(world[2] if a in buttons else -world[2]));by[name]=by.get(name,0)+down;distance=min(distance,float(c.dist))
  for name,f in by.items():peaks[name]=max(peaks.get(name,0),f);counts[name]=counts.get(name,0)+int(f>.1)
  rows.append({'frame':i//4,'force':sum(by.values()),'distance':distance,'by_body':by})
 summary=force_hold_summary(rows,20,.02,.001);summary.update(native_success=result['success'],coverage=result['coverage'],failure=result['failure'],final_force_N=rows[-1]['force'],peak_by_body_N=peaks,contact_ticks=counts,omitted_active_bodies=[n for n,v in peaks.items() if v>.1 and not any(k in n.lower() for k in native_keys)],physics_steps=0,physical_gate=bool(summary['meets_one_second'] and result['coverage']==1 and not result['failure']),native_checker_unchanged=True)
 from .button_acceptance import evaluate,PROFILE
 reviewed=evaluate(case,case/'execution/button_acceptance.json')
 summary.update(all_right_diagnostic_gate=summary['physical_gate'],physical_gate=reviewed['revised_success'],revised_success=reviewed['revised_success'],acceptance_profile=PROFILE,revised_hold_s=reviewed['revised_hold_s'])
 save(case/'contact_strategy_audit.json',summary);return summary

def validate_plan(d,fixed,names,frames,bodies):
 if set(d)!=set(SCHEMA):raise ValueError('Unexpected plan fields')
 basic={k:copy.deepcopy(d[k]) for k in PROPERTIES};tail=d['tail_s']
 if type(tail) not in (int,float) or not np.isfinite(tail) or not 0<=tail<=5 or abs(tail/.02-round(tail/.02))>1e-6:raise ValueError('Tail must be20ms aligned in0..5s')
 basic['tail_s']=min(tail,3);validate_decision(basic,[fixed['episode']]);validate_parameters(d['preencoder']);validate_fingers(d['finger_control'],names,frames+round(tail/.02));validate_strategy(d['strategy'],bodies)
 for k in fixed:
  if k not in ('action','rationale','hypothesis','preencoder','finger_control','tail_s','strategy') and d[k]!=fixed[k]:raise ValueError('Fixed field changed: '+k)
 if d['action'] not in ('run','reject','needs_tool'):raise ValueError('Select run/reject/needs_tool')
 return d

def shallow_video(root,case,inspection,suffix):
 name=Path(case).name+'_'+suffix+'.mp4';dest=root/name;src=Path(inspection['video']);shutil.copy2(src,dest);assert file_hash(src)==file_hash(dest);return str(dest)

def run(config,provider_config):
 from . import suite_physics as ph
 install();root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True);start=now();tick=time.monotonic();history=[];status='budget_exhausted';error=None
 baseline=Path(config['baseline_case']);verify_source_contract(baseline);br=json.loads((baseline/'result.json').read_text());fixed=copy.deepcopy(json.loads(Path(br['decision_path']).read_text())['decision']);fixed.pop('strategy',None);fixed['baseline_case']=str(baseline)
 geometry=json.loads(Path(config['geometry']).read_text());bodies=config['safety_bodies'];fixed['contact_control']['contact_bodies']=bodies;fixed['contact_control']['joints']=[]
 experience=json.loads(Path(config['experience']).read_text());provider=BudgetProvider(configured_provider(provider_config,root/'.api_rotation.json'),time.monotonic()+1500)
 writer=ph.write_new
 def write(path,value):
  if Path(path).name=='experiment.json':
   value=copy.deepcopy(value);plan=json.loads(Path(value['pipeline']['planner_identity_and_prompt_hash']['decision_path']).read_text())['decision'];value['purpose']['hypothesis']='API selects evidenced contact strategy; native and expanded physical button gate reported separately';value['purpose']['planned_changes'].append({'api_strategy':plan['strategy'],'tail_s':plan['tail_s']});value['purpose']['invariants']=[x for x in value['purpose']['invariants'] if not x.startswith('fixed wrist strategy')];value['pipeline']['allowed_retry_changes']=['API preencoder/finger parameters and tail0..5s within frozen per-case plan'];value['pipeline']['entrypoint']='humanoid_mimicgen.augmentation.contact_strategy.run';value['metrics']['expanded_contact_gate']='Approved button_palm_mesh_v1: native contact geometry plus exact right_hand_palm_link mesh on right_wrist_yaw_link; same20N/1mm/50ticks, full-safe execution. Native separately retained. All-right force remains diagnostic only.';value['pipeline']['experience_reuse']=plan['strategy']['change_kind']
   for p in [Path(config['experience']),Path(config['geometry']),Path(config['config_path'])]:
    value['preflight']['evidence_paths'].append(str(p));value['preflight']['evidence_sha256'][str(p)]=file_hash(p);shutil.copy2(p,Path(path).parent/'source_snapshot'/('strategy_'+p.name))
  return writer(path,value)
 ph.write_new=write
 try:
  for i in range(3):
   ar=root/f'attempt_{i:02d}';ar.mkdir();context={'task':br['task'],'objective':'Pass approved button_palm_mesh_v1 acceptance and exact token+hands replay; preserve original native result separately','fixed_plan':{k:v for k,v in fixed.items() if k not in ['action','rationale','hypothesis']},'editable':['preencoder','finger_control','tail_s','strategy'],'geometry':geometry,'experience':experience,'baseline':brief(baseline),'previous':[{'parameters':h['parameters'],'diagnostic':h['audit']} for h in history],'remaining_trials':3-i,'tail_budget_s':5,'api_controls_physics':True}
   image_case=Path(history[-1]['case']) if history else baseline;ip=image_case/'inspection/inspection.json'
   if ip.exists():context['evidence']={'inspection':{'review_images':json.loads(ip.read_text())['review_images'][-1:]}}
   d,dp=api_decide(provider,ar,context,[br['episode']],tools=tools(SCHEMA),validator=lambda d,eps:validate_plan(d,fixed,config['hand_command_order'],br['source_frames'],bodies),skill_path=SKILL);save(ar/'plan.json',d)
   if d['action']!='run':status=d['action'];break
   verify_origin(dp);result=physical_case(config['dataset'],br['episode'],ar,'token_prefix_mode1',br['object_delta_xy_m'],decision_path=dp,preencoder=True,finger_control=True);case=Path(result['experiment_path']);save(ar/'physics.result.json',result);audit=contact_audit(case);ins=inspect_case(case,case/'inspection')
   raw=np.load(case/'execution/controller.npz')['records'];trace=np.load(case/'execution/trajectory.npz');assert np.array_equal(np.repeat(raw[:,-29:],4,axis=0)[:len(trace['commands'])],trace['commands'][:,1:30])
   row={'attempt':i,'case':str(case),'parameters':d,'result':result,'audit':audit,'inspection':ins,'video':shallow_video(root,case,ins,'生成'),'replay':None};history.append(row);save(root/'progress.json',history)
   if audit['physical_gate']:
    rr=physical_case(config['dataset'],br['episode'],ar,'saved_token_replay',br['object_delta_xy_m'],decision_path=dp,saved_token_case=case);rc=Path(rr['experiment_path']);proof=replay_audit(result,rr)
    with np.load(case/'execution/hand_commands.npz') as a,np.load(rc/'execution/hand_commands.npz') as b:proof['hands_exact']=all(np.array_equal(a[k],b[k]) for k in ('left','right'))
    ra=contact_audit(rc);ri=inspect_case(rc,rc/'inspection');row['replay']={'result':rr,'proof':proof,'audit':ra,'inspection':ri,'video':shallow_video(root,rc,ri,'独立回放')};save(rc/'token_replay_audit.json',proof)
    if proof['exact'] and proof['hands_exact'] and ra['physical_gate']:status='physical_success_exact_replay'
   save(ar/'result.json',row);save(root/'progress.json',history);print(json.dumps({'attempt':i,'status':status,'audit':audit}),flush=True)
   if status=='physical_success_exact_replay':break
 except Exception as exc:
  import traceback
  error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};save(root/'ERROR.json',error);status='error'
 finally:ph.write_new=writer
 calls=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')];summary={'experiment_id':root.name,'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'status':status,'error':error,'attempts':history,'api_requests':len(calls),'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'unknown_usage_calls':sum(c.get('usage') is None for c in calls),'api_wall_s':sum(c.get('wall_s') or 0 for c in calls),'training_ready':False};save(root/'summary.json',summary);return summary
