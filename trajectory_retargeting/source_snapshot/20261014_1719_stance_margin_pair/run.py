"""Common measured safe stance, then frozen IK/mode0 versus mode1 comparison."""
import sys,copy,runpy,time,json
from pathlib import Path
import numpy as np
from cap_sim.core import run_task
from cap_sim.io import read,write
from clearance import hip_clearance
from standing_planner import StandingPlanner
from prefix_backend import ShortPrefix,Base
ROOT=Path(__file__).resolve().parent;PREV=ROOT.parent/'20261014_1558_ik_teleop_comparison'
from previous_runtime import previous
now=previous['now']
class Backend(previous['Backend'],ShortPrefix):
 def __init__(self,mode,tokens):
  super().__init__(mode);self.tokens=tokens;self.stance_ok=False;self.stance_delta=None;self.clearance_records=[];self.settle_elapsed=0.;self.last_stance_gate=None
 def reset(self,t,out):
  o=super().reset(t,out)
  self.planner_commands=[]
  self.sonic.planner=StandingPlanner(self.sonic.planner,self.planner_commands,lambda:float(self.d.time))
  return o
 def execute(self,a,remaining):
  if a['name']=='stance_not_ready':return {'completed':False,'reusable':False,'failure':'stance_clearance_or_velocity_gate','samples':[],'observation':self.observe()}
  if a['name'].startswith('stance_settle'):
   self.stage_number+=1
   result=ShortPrefix.execute(self,a,remaining)
   self.settle_elapsed+=a['duration_s']
   c=hip_clearance(self.m,self.d);speed=float(np.linalg.norm(self.d.qvel[:3]));self.stance_ok=c['minimum_m']>=.10 and speed<.08 and not result['failure']
   write(self.out/'stance_gate.json',{'passed':self.stance_ok,'clearance':c,'base_speed_m_s':speed,'required_clearance_m':.10,'max_speed_m_s':.08,'time':float(self.d.time),'root_pose':self.d.qpos[:7].tolist(),'result_failure':result['failure']})
   write(self.out/'planner_commands.json',self.planner_commands)
   self.last_stance_gate=read(self.out/'stance_gate.json');write(self.out/f'stance_gate_{self.stage_number:03d}.json',self.last_stance_gate)
   return result
  original=self.sonic.step
  def monitored(*args,**kwargs):
   c=hip_clearance(self.m,self.d);self.clearance_records.append({'time':float(self.d.time),'minimum_m':c['minimum_m']})
   return original(*args,**kwargs)
  self.sonic.step=monitored
  try:return super().execute(a,remaining)
  finally:
   self.sonic.step=original
   write(self.out/'clearance_samples.json',self.clearance_records)
   write(self.out/'planner_commands.json',self.planner_commands)
class Planner:
 def __init__(self,b,mode):self.b=b;self.delegate=previous['Planner'](b,mode)
 def plan(self,q):
  done=q.get('executed_actions',[])
  if done and not any(a.get('name','').startswith('stance_settle') for a in done):
   o=q['observation'];a={'name':'stance_settle','stage':'recover','duration_s':2.,'velocity_xy':[0,0],'pelvis_height_m':o['root_pose'][2],'replan_after':True}
   return {'analysis':{'evidence':[{'root':o['root_pose']}],'hypothesis':'Common zero-velocity SONIC mode1 settle before comparing interfaces; qualify measured hip margin and base speed'},'actions':[a]}
  if done and not self.b.stance_ok:
   gate=self.b.last_stance_gate
   if gate and gate['clearance']['minimum_m']>=.10 and self.b.settle_elapsed<4. and not gate['result_failure']:
    o=q['observation']
    return {'analysis':{'evidence':[gate],'hypothesis':'Enough geometric clearance but not yet settled; observe after another bounded half-second without changing the gate'},'actions':[{'name':f'stance_settle_{self.b.settle_elapsed:.1f}','stage':'recover','duration_s':.5,'velocity_xy':[0,0],'pelvis_height_m':o['root_pose'][2],'replan_after':True}]}
   return {'analysis':{'evidence':['stance_gate.json failed'],'hypothesis':'Do not approach from an unqualified stance'},'actions':[{'name':'stance_not_ready','stage':'hold','duration_s':.1}]}
  adjusted=copy.deepcopy(q)
  if done:
   source=adjusted['references'][0]['actions'][1]['body_keyframes'][0]
   if self.b.stance_delta is None:
    self.b.stance_delta=np.r_[np.array(q['observation']['root_pose'][:2])-np.array(source[1:3]),0.]
    write(self.b.out/'stance_anchor.json',{'root_translation_m':self.b.stance_delta.tolist(),'right_wrist_goals':'unchanged original world targets; left wrist and torso follow shifted stance','physical_state_write':False})
   delta=self.b.stance_delta
   for a in adjusted['references'][0]['actions']:
    for frame in a.get('body_keyframes',[]):frame[1:4]=(np.array(frame[1:4])+delta).tolist()
   for target in adjusted['contact_privileged']['initial_targets']:
    target['suggested_wrist_position_world_m']=(np.array(target['suggested_wrist_position_world_m'])-delta).tolist()
  decision=self.delegate.plan(adjusted)
  for a in decision['actions']:
   if a['stage']=='native_prefix':a['tokens']=self.b.tokens
  return decision
 def reflect(self,q):return self.delegate.reflect(q)
def main():
 case,mode,tokens,tag=sys.argv[1:5];tokens=int(tokens);assert mode in ('ik','teleop')
 out=ROOT/tag/case/mode;assert not out.exists()
 task=read(PREV/'reference/ik/task.json');task['scene']=({'translation':[0,0,0]} if case=='reference' else dict(previous['CASES'])[case]);task['code_version']='Safe-stance matched-interface comparison';task['budget']['feedback_calls']=17
 b=Backend(mode,tokens);started=now();wall=time.perf_counter()
 try:
  summary=run_task(task,Planner(b,mode),b,out)
  write(out/'execution_timing.json',{'started_at':started,'ended_at':now(),'wall_s':time.perf_counter()-wall})
  write(out/'comparison_result.json',{'mode':mode,'prefix_tokens':tokens,'scene':task['scene'],'stance_qualified':b.stance_ok,'summary':summary})
  print('COMPLETE',tag,case,mode,b.stance_ok,summary['success'],flush=True)
 finally:b.close()
if __name__=='__main__':main()
