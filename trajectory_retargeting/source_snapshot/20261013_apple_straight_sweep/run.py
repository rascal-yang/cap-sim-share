"""Single-factor sweep of the verified straight-approach reference policy.

No change to cap-sim, HUMANO, robot physics, source route, timing or fingers.
GPT-authored deterministic retarget stages; GPT supplies failure reflections.
"""
import copy
import hashlib
import runpy
import sys
import time
from pathlib import Path
from cap_sim.core import run_task
from cap_sim.io import read, write, digest
from cap_sim.planners import FilePlanner
from cap_sim.diagnostics import validate_reflection
from cap_sim.task_skills import EvolvingSkillPlanner

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
OLD=ROOT.parent/'20261013_apple_ordered_retry'
base=runpy.run_path(str(ROOT.parent/'20261013_apple_variants/run.py'))
Backend=base['VariantBackend']
CASES=[
 ('baseline_y_minus_10cm',{'translation':[0,-.10,0]}),
 ('y_plus_10cm',{'translation':[0,.10,0]}),
 ('y_minus_15cm',{'translation':[0,-.15,0]}),
 ('y_plus_15cm',{'translation':[0,.15,0]}),
 ('y_minus_20cm',{'translation':[0,-.20,0]}),
 ('y_plus_20cm',{'translation':[0,.20,0]}),
 ('table_plus_7p5cm',{'translation':[0,0,0],'table_delta':.075}),
 ('table_minus_7p5cm',{'translation':[0,0,0],'table_delta':-.075}),
 ('table_plus_10cm',{'translation':[0,0,0],'table_delta':.10}),
 ('table_minus_10cm',{'translation':[0,0,0],'table_delta':-.10}),
]

class Planner:
 def __init__(self,case,ordinal,prior):
  self.case=case;self.ordinal=ordinal;self.prior=prior
 def plan(self,q):
  start=time.time();p=ROOT/self.case;k=f"a{q['attempt']:03d}_f{q.get('feedback_index',0):02d}"
  q=copy.deepcopy(q)
  if self.prior:q['previous_case_reflection']=self.prior
  write(p/'mailbox'/f'{k}_request.json',q)
  c=read(p/'config.json');c['model_path']=str(p/'run/attempts'/f"{q['attempt']:03d}"/'scene.mjb')
  result=runpy.run_path(str(ROOT/'policy.py'))['decide'](q,c,ROOT)
  a=result['analysis'];a['strategy']={'strategy_id':'reference_local','applicability':'Verified small variants; larger requested points remain candidates','approach':'Unchanged original250-token straight approach','stance':'Original root/legs/waist/left-arm reference, except measured transition anchor','grasp':'Reference hand commands and contact-derived right-arm retarget','next_observation_gate':'Per-stage feedback; measured sustained hand plus current thumb/index contacts before lift','abort_or_recovery':'Original contact/fall/table gates; record failure, no added walking/turning'}
  a['expected_effect']='Same route and reference contact pattern complete grasp at the specified single-factor point.'
  a['reject_if']='Hard stop, missing opposing contacts, failed load transfer or fixed budget.'
  if self.prior:a['previous_case_reflection_sha256']=digest(self.prior)
  result['provenance']={'identity':'Current GPT authored/selected deterministic CaP reference retarget policy; not a per-stage external model call','live_external_api':False,'policy_sha256':hashlib.sha256((ROOT/'policy.py').read_bytes()).hexdigest(),'case_ordinal':self.ordinal}
  write(p/'mailbox'/f'{k}_response.json',result)
  write(p/'mailbox'/f'{k}_timing.json',{'decision_wall_s':time.time()-start})
  print('STAGE',self.case,k,result['actions'][0]['name'],flush=True)
  return result
 def reflect(self,q):
  p=ROOT/self.case;k=f"a{q['attempt']:03d}_reflection"
  q={**q,'case_id':self.case,'case_ordinal':self.ordinal,'previous_case_reflection':self.prior}
  write(p/'mailbox'/f'{k}_request.json',q);print('WAIT_REFLECTION',self.case,flush=True)
  response=p/'mailbox'/f'{k}_response.json';start=time.time()
  while not response.exists():time.sleep(.2)
  write(p/'mailbox'/f'{k}_timing.json',{'decision_wall_s':time.time()-start})
  return read(response)

def recover():
 source=read(OLD/'conversion_run/success/attempt_001/trajectory.json')
 task=copy.deepcopy(source['task']);task['references']=[];task['skills']=[];task['scene']={'translation':[0,0,0]}
 task['generation_kind']='reference_recovery';task['budget'].update(attempts=1,optimization_attempts=0)
 write(ROOT/'recovery_program.json',[{'analysis':{'evidence':['Validated source0 converted successful actions; execute anew in original scene'],'hypothesis':'Current unchanged reference interface still completes original task before sweep'},'actions':source['actions']}])
 b=Backend()
 try:return run_task(task,FilePlanner(ROOT/'recovery_program.json'),b,ROOT/'reference_recovery')
 finally:b.close()

def main():
 assert not (ROOT/'batch_summary.json').exists(),'Existing batch requires inspection; no silent rerun'
 start=time.time();recovery=recover();write(ROOT/'recovery_gate.json',recovery)
 assert recovery['success'],'Stop: original reference recovery failed'
 ref=ROOT/'reference_recovery/success/attempt_001/trajectory.json';results=[];previous=None
 write(ROOT/'batch_contract.json',{'cases':CASES,'max_generation_attempts':10,'attempts_per_point':1,'reference_recovery_attempts':1,'optimization_attempts':0,'control':'Original250-token prefix + mode0 full-body reference with right-arm IK retarget; no mode1 exclusivity','route':'Original straight approach; no new walk/turn stages','vertical':'Table and apple translated together; apple remains supported','single_factor':True,'no_git_commit':True})
 for ordinal,(case,scene) in enumerate(CASES,1):
  assert len(results)<10
  if previous:validate_reflection(previous['reflection'],previous['result'],previous['diagnostics'],Backend.interface)
  task=read(OLD/'conversion_run/task.json');task['references']=[str(ref)];task['skills']=[]
  task['scene']=scene;task['generation_kind']='augmented';task['budget'].update(attempts=1,optimization_attempts=0,feedback_calls=12)
  task['contact_prior']={'object_body':'apple_001','wrist_body':'right_wrist_yaw_link','object_key':'target'}
  task['policy_checks']={'require_reflection':True}
  task['code_version']='Straight source-route position/height sweep; contact-derived right-arm retarget, frozen physics/timing/fingers'
  p=ROOT/case
  write(p/'config.json',{'diagnosis':f'Point{ordinal}/10: {scene}. Keep verified straight route, timing, hand commands and physics; single-factor candidate. No new turning or walking.','offset':[0.,0.,0.]})
  planner=EvolvingSkillPlanner(Planner(case,ordinal,previous['reflection'] if previous else None),[WORKSPACE/'cap-sim/task_skills/g1_dex3_apple_pick'],p/'skill_versions')
  b=Backend()
  try:summary=run_task(task,planner,b,p/'run')
  finally:b.close()
  result=read(p/'run/attempts/001/result.json')
  results.append({'case':case,'scene':scene,'summary':summary,'result':result})
  if not result['acceptance']['success']:
   previous={'reflection':read(p/'run/attempts/001/reflection.json'),'result':result,'diagnostics':read(p/'run/attempts/001/diagnostics.json')}
  write(ROOT/'batch_progress.json',{'completed_points':len(results),'successful_points':sum(x['summary']['success'] for x in results),'results':results})
  print('RESULT',case,summary['success'],result['failure'],flush=True)
 write(ROOT/'batch_summary.json',{'completed_points':len(results),'successful_points':sum(x['summary']['success'] for x in results),'generation_attempts':len(results),'reference_recovery':recovery,'wall_s_including_preparation_and_reflection':time.time()-start,'results':results})
 print('BATCH_COMPLETE',flush=True)

if __name__=='__main__':main()
