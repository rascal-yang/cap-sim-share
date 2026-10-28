"""GPT-authored bounded CaP feedback policy, not a live model call per checkpoint.
Only current measured states and this run's own attempts; never opens expert data.
"""
import json,time,subprocess,hashlib,math
from pathlib import Path
r=Path(__file__).resolve().parent
NAMES=['G1WholebodyTableApplePickTeleop','G1WholebodyTableButtonPressTeleop','G1WholebodyGroundBottlePickTeleop','G1WholebodyGroundBoxPickTeleop','G1WholebodyTablePlatePickTeleop']
sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def write(p,analysis,actions):
 out=p.with_name(p.name.replace('_request','_response'));assert not out.exists()
 out.write_text(json.dumps({'analysis':analysis,'actions':actions,'planner_provenance':{'kind':'GPT-authored CaP feedback program','policy_sha256':sha,'live_model_call':False}},ensure_ascii=False,indent=2))
def hold(o,duration=.2,stage='hold',fingers=False,replan=False):
 pose=o['root_pose']+[max(j['range'][0],min(j['range'][1],v)) for j,v in zip(o['joint_interface']['body'],o['body_joints'])]
 a={'stage':stage,'duration_s':duration,'body_keyframes':[[0]+pose,[duration]+pose],'replan_after':replan}
 if fingers:
  a['right_fingers']=[.1,-.7,-1.1,1.0,1.0,1.0,1.0]
  a['left_fingers']=[-.1,.7,1.1,-1.0,-1.0,-1.0,-1.0]
 return a
def respond(p):
 q=json.loads(p.read_text());o=q['observation'];n=q['attempt'];f=q.get('feedback_index',0);name=q['task']['name'];ground='Ground' in name;dual=any(x in name for x in ['Box','Plate']);button='Button' in name
 obj=o['objects'].get('target',o['objects'].get('articulated'));hist=q['history'];last=hist[-1] if hist else {};remaining=q.get('remaining_sim_seconds',30)
 ev=[{'measured_time':o['time'],'object':obj[:3],'root':o['root_pose'][:3],'hand_contact_N':o['contacts']['hand_object_N'],'contact_s':o['contact_s'],'previous_failure':last.get('failure')}]
 analysis={'evidence':ev,'hypothesis':'Reuse staged approach; adjust measured errors within fixed permissions and stop on failed contact or budget.'}
 if q['phase']=='optimize':
  actions=q['successful_trajectory']['actions'];scale=1.12
  for a in actions:
   a['duration_s']=round(a['duration_s']*scale/.005)*.005
   if 'body_keyframes' in a:
    for row in a['body_keyframes']:row[0]*=scale
  if sum(a['duration_s'] for a in actions)>30:actions=q['successful_trajectory']['actions']
  analysis['hypothesis']='Preserve accepted baseline; slow its actions by 12% where budget permits, then retain only if original acceptance and smoothness gates pass.';write(p,analysis,actions);return
 if f==0:
  # Each reset is counted. Contact failures change wrist roll and stance; table collisions reduce advance.
  delta=[0,-.1,.1,-.05,.05][(n-1)%5];duration=(1.65 if ground else 2.0)+delta
  if last.get('failure')=='unexpected_table_contact':duration-=.1
  analysis['hypothesis']=f"Retry diagnosis before reset: {last.get('failure','initial case')}; use {duration:.2f}s approach, then actively brake. Contact geometry changes by attempt, never physics or acceptance."
  write(p,analysis,[{'stage':'walk','duration_s':round(duration,3),'velocity_xy':[.23,0],'pelvis_height_m':.75,'replan_after':True}]);return
 if remaining<1 or f>=11:
  analysis['hypothesis']='No validated completion within remaining phase budget; hold safely and terminate this attempt.';write(p,analysis,[hold(o)]);return
 if ground and 'Bottle' in name and obj[2]<.10:
  analysis['hypothesis']='Bottle toppled before a grasp; stop this attempt and change approach/stance on reset rather than chase it.';write(p,analysis,[hold(o)]);return
 if f==1:
  cfg={'stage':'recover','duration_s':2,'move_s':1.5}
 else:
  x,y,z=obj[:3];roll=[0,math.pi/2,-math.pi/2][(n-1)%3]
  cfg={'root_z':(.47 if ground else .65),'waist_pitch':.45,'duration_s':3,'move_s':2.5,'right_fingers':[0,-.2,-.25,.3,.3,.3,.3]}
  if ground:cfg['stance_extra_y']=.08+.025*((n-1)%4)
  if button:
   # Finger-first downward approach. Teleop orientation was empirically more responsive than full-body wrists.
   zgoal=.91-max(0,f-2)*.015
   a={'stage':'press','duration_s':3,'pelvis_height_m':.67,'left_position':[-.69,.23,.85],'right_position':[x-.02+(n%3-1)*.015,y+.02,zgoal],'right_quaternion_wxyz':[2**-.5,0,2**-.5,0],'right_fingers':[0,-.2,-.2,.35,.45,.35,.45],'replan_after':True}
   analysis['hypothesis']='Use verified Cartesian wrist orientation response; descend 1.5cm per checkpoint, vary lateral alignment across retries; measured native 20N hold criterion remains authoritative.';write(p,analysis,[a]);return
  contact=o['contact_s']>=.12
  events=o['contacts']['events'];left=any(any('left_hand' in b for b in e['bodies']) for e in events);right=any(any('right_hand' in b for b in e['bodies']) for e in events)
  contact=contact and (not dual or (left and right))
  prior=q.get('executed_actions',[])
  if contact and prior and prior[-1]['stage'] not in ('contact','lift'):
   analysis['hypothesis']='Required hand contacts are present; close fingers while holding current body reference before any lift.';write(p,analysis,[hold(o,1.5,'contact',True,True)]);return
  lift=contact and prior and prior[-1]['stage'] in ('contact','lift')
  if lift:
   cfg['stage']='lift';cfg['root_z']=min(.76,o['root_pose'][2]+.04)
   for side in (['left','right'] if dual else ['right']):cfg[side+'_position']=[*o[side+'_position'][:2],o[side+'_position'][2]+.035]
   cfg['right_fingers']=[.1,-.7,-1.1,1.,1.,1.,1.]
   if dual:cfg['left_fingers']=[-.1,.7,1.1,-1.,-1.,-1.,-1.]
   analysis['hypothesis']='Contact gate passed after closure; lift only 3.5cm, stop automatically on lost contact, then re-observe.'
  else:
   h=(.43 if ground else z+.15) if f==2 else (max(z+.15,.31) if ground else z+.055)
   offset=.15 if dual else .11
   desired={'right':[x-offset,y-(.18 if dual else 0),h]}
   if dual:desired['left']=[x-offset,y+.18,h]
   if f>=4:
    # Bounded error compensation; preserve observation-dependent goals, avoid uncontrolled escalation.
    for side,pos in desired.items():
     for j in range(3):pos[j]+=max(-.035,min(.035,(pos[j]-o[side+'_position'][j])*.5))
   cfg.update({k+'_position':v for k,v in desired.items()})
   cfg['right_quaternion_wxyz']=[math.cos(roll/2),math.sin(roll/2),0,0]
   if not dual:cfg['left_position']=[o['root_pose'][0]-.02,.23,max(.68,o['root_pose'][2]+.18)]
   analysis['hypothesis']='No sufficient opposing contact yet: approach above object first, descend slowly, then compensate bounded measured wrist error; vary wrist roll and stance after diagnosed failure.'
 cfg.update({'evidence':ev,'hypothesis':analysis['hypothesis']+' Generated by frozen GPT-authored feedback code; no live model call at this checkpoint.'})
 subprocess.run(['/home/kinetra-4090-01/HUMANO/.venv/bin/python',str(r/'kinematic_plan.py'),str(p)],input=json.dumps(cfg),text=True,check=True,stdout=subprocess.DEVNULL)
 out=p.with_name(p.name.replace('_request','_response'));d=json.loads(out.read_text());d['planner_provenance']={'kind':'GPT-authored CaP feedback program','policy_sha256':sha,'live_model_call':False};out.write_text(json.dumps(d,ensure_ascii=False,indent=2))
if __name__=='__main__':
 start=time.time()
 while time.time()-start<10800:
  complete=0
  for name in NAMES:
   case=r/'runs/no_reference'/name
   if (case/'summary.json').exists():complete+=1;continue
   for p in sorted((r/'mailbox/no_reference'/name).glob('*_request.json')):
    if p.with_name(p.name.replace('_request','_response')).exists():continue
    num=int(p.name[1:4]);attempt=case/'attempts'/f'{num:03d}'
    if not attempt.exists() or (attempt/'result.json').exists():continue
    try:respond(p);print('ANSWERED',name,p.name,flush=True)
    except Exception as e:print('ERROR',name,p.name,str(e),flush=True);raise
  if complete==len(NAMES):break
  time.sleep(.5)
