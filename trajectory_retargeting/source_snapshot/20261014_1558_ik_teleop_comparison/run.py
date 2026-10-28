"""Matched target comparison, native physics unchanged; no automatic policy search."""
import copy, hashlib, json, runpy, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import mujoco
from cap_sim.core import run_task
from cap_sim.io import read,write,digest
from cap_sim.contact_prior import map_prior
from targets import interpolate,errors,rotation
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
OLD=ROOT.parent/'20261013_apple_straight_sweep'
variant=runpy.run_path(str(ROOT.parent/'20261013_apple_variants/run.py'))
Base=variant['VariantBackend']
original=runpy.run_path(str(OLD/'policy.py'))['decide']
CASES=runpy.run_path(str(OLD/'run.py'))['CASES']

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

class Backend(Base):
 def __init__(self,mode):self.mode=mode;self.measurements=[];self.stage_number=0;self.nominal=None
 def reset(self,t,out):
  o=super().reset(t,out)
  self.providers={k:getattr(self.sonic,k).get_providers() for k in ('encoder','decoder','planner')}
  assert all(x==['CPUExecutionProvider'] for x in self.providers.values())
  write(Path(out)/'actual_providers.json',self.providers)
  return o
 def execute(self,a,remaining):
  self.stage_number+=1
  if a['stage']=='native_prefix':return super().execute(a,remaining)
  nominal=copy.deepcopy(self.nominal);start=float(self.d.time);original_step=self.sonic.step
  data=mujoco.MjData(self.m);data.qpos[:]=self.d.qpos
  def goal(t):
   pose=interpolate(nominal['frames'],t)
   data.qpos[:7]=pose[:7];data.qpos[self.qids]=pose[7:]
   mujoco.mj_forward(self.m,data)
   wrists=[(data.xpos[b].copy(),data.xquat[b].copy()) for b in (self.left,self.right)]
   wrists[1]=(wrists[1][0]+nominal['shift'],wrists[1][1])
   return pose,wrists,(data.xpos[self.torso].copy(),data.xquat[self.torso].copy())
  def record(t,kind):
   pose,wrists,torso=goal(t);p,r=errors(self.d.xpos[self.right],self.d.xquat[self.right],*wrists[1])
   self.measurements.append({'time':float(self.d.time),'stage_number':self.stage_number,'stage':a['stage'],'name':a['name'],'stage_time':t,'kind':kind,'position_error_m':p,'orientation_error_rad':r,'actual_right_p':self.d.xpos[self.right].tolist(),'actual_right_q':self.d.xquat[self.right].tolist(),'target_right_p':wrists[1][0].tolist(),'target_right_q':wrists[1][1].tolist(),'root_error_m':float(np.linalg.norm(self.d.qpos[:3]-pose[:3])),'root_actual':self.d.qpos[:3].tolist(),'root_reference':pose[:3].tolist()})
   return pose,wrists,torso
  def step(d,wrists,torso,velocity,height,body_future=None,facing_yaw=None):
   pose,desired,desired_torso=record(float(d.time)-start,'control_sample')
   if self.mode=='teleop':
    answer=original_step(d,desired,desired_torso,np.zeros(2),float(pose[2]),None,float(rotation(pose[3:7]).as_euler('xyz')[2]))
   else:answer=original_step(d,wrists,torso,velocity,height,body_future,facing_yaw)
   assert self.sonic.records[-1]['mode']==(1 if self.mode=='teleop' else 0)
   return answer
  self.sonic.step=step
  try:result=super().execute(a,remaining)
  finally:self.sonic.step=original_step
  record(float(self.d.time)-start,'stage_end')
  write(self.out/'tracking.json',self.measurements)
  return result

class Planner:
 def __init__(self,b,mode):self.b=b;self.mode=mode
 def plan(self,q):
  actions=q['references'][0]['actions'];cursor=-1
  for a in q.get('executed_actions',[]):
   if a.get('name','').startswith('converted_'):cursor=int(a['name'].split('_')[1])
  i=min(cursor+1,len(actions)-1);a=copy.deepcopy(actions[i]);a['name']=f'converted_{i:03d}';a['replan_after']=True
  if a['stage']!='native_prefix' and a['stage']=='lift' and q['observation']['contact_s']<.12:
   i=3;a=copy.deepcopy(actions[i]);a['name']=f'converted_{i:03d}';a['replan_after']=True
  if a['stage']!='native_prefix':
   prior=q['contact_privileged']['reference_priors'][0];mapped=q['contact_privileged']['initial_targets'][0]
   source=map_prior(prior,{'objects':{prior['object_key']:prior['reference_initial_object_pose']}},prior['reference_scale'])
   shift=np.array(mapped['suggested_wrist_position_world_m'])-np.array(source['suggested_wrist_position_world_m'])
   frames=copy.deepcopy(a['body_keyframes'])
   self.b.nominal={'frames':frames,'shift':shift}
   if self.mode=='teleop':
    del a['body_keyframes']
    # Actual Cartesian stream is computed by FK from the shared nominal targets,
    # never from IK outputs. No physical qpos writes occur in goal construction.
   else:
    c={'model_path':str(self.b.out/'scene.mjb'),'diagnosis':'Matched control-interface comparison','offset':[0,0,0]}
    a=original(q,c,ROOT)['actions'][0]
  return {'analysis':{'evidence':[{'case':q['task']['scene'],'source_action':i,'interface':self.mode}], 'hypothesis':'Shared nominal wrist goals allow matched physical tracking comparison; retain native prefix and original safety/acceptance gates.'},'actions':[a]}
 def reflect(self,q):
  d=q['diagnostics']['last_stage']
  return {'source_attempt':q['attempt'],'result_sha256':q['result_sha256'],'failed_stage':d.get('stage','unknown'),'observed_facts':[{'path':'result.failure','value':q['result']['failure']}],'hypothesis':'Fixed controller comparison did not pass this scene; separate tracking error from contact failure using saved measurements.','alternative_causes':['Whole-body reference versus planner leg coordination','Contact retention or collision along the path'],'next_change':{'description':'No parameter changes within this frozen comparison. Continue the next predeclared arm/case.','controls':[]},'expected_effect':'Measure differences without adaptive tuning.','reject_if':'Control mode, physics, or target contract mismatch.','hypothesis_outcome':'inconclusive'}

def main():
 case,mode=sys.argv[1:3];assert mode in ('ik','teleop')
 scene=({'translation':[0,0,0]} if case=='reference' else dict(CASES)[case]);out=ROOT/case/mode
 assert not out.exists(),'Never overwrite or silently resume a physical attempt'
 task=read(OLD/'y_minus_20cm/run/task.json');task['scene']=scene
 task['code_version']='20261014 matched IK versus teleop; same nominal reference FK targets'
 task['planner_identity']='Frozen GPT-authored comparison policy; no external model calls'
 task['budget'].update(attempts=1,optimization_attempts=0)
 b=Backend(mode);started=now();wall=time.perf_counter()
 try:
  summary=run_task(task,Planner(b,mode),b,out)
  write(out/'execution_timing.json',{'started_at':started,'ended_at':now(),'wall_s':time.perf_counter()-wall})
  write(out/'comparison_result.json',{'mode':mode,'scene':scene,'summary':summary,'control_samples':len(b.measurements)})
  print('COMPLETE',case,mode,summary['success'],flush=True)
 finally:b.close()
if __name__=='__main__':main()
