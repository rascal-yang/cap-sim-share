import json,time,copy,hashlib
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent
class Planner:
 def __init__(self,out):self.out=Path(out);self.ref=json.loads((r/'reference.json').read_text());self.config=None
 def plan(self,q):
  phase=self.out.name;n=q['attempt'];f=q.get('feedback_index',0);key=f'{phase}_a{n:03d}_f{f:02d}';mail=r/'mailbox';mail.mkdir(exist_ok=True);(mail/(key+'_request.json')).write_text(json.dumps(q));start=time.time()
  cfg=r/'configs'/f'{phase}_{n:03d}.json'
  print('DECISION',key,flush=True)
  while not cfg.exists():time.sleep(.2)
  c=json.loads(cfg.read_text());s=self.ref['samples'];ts=np.array([v['time'] for v in s]);rate=c.get('rate',1.)
  def interp(field,t):
   a=np.array([v[field] for v in s]);flat=a.reshape(len(s),-1);return np.array([np.interp(t,ts,flat[:,i]) for i in range(flat.shape[1])]).reshape(a.shape[1:])
  o=q['observation'];executed=q.get('executed_actions',[]);cursor=0
  for a in executed:
   if a.get('name','').startswith('source_'):cursor=float(a['name'].split('_')[1])
  target=min(ts[-1],cursor+2.*rate);duration=max(.1,round((target-cursor)/rate/.005)*.005)
  wants_lift=interp('object',target)[2]>s[0]['object'][2]+.015
  gate=o['contact_s']>=.12
  if wants_lift and not gate:target=min(target,7.5);duration=1.5
  pose=interp('pose',target);lo=np.array([j['range'][0] for j in o['joint_interface']['body']]);hi=np.array([j['range'][1] for j in o['joint_interface']['body']])
  action={'name':f'source_{target:.5f}','stage':'lift' if wants_lift and gate else 'contact','duration_s':duration,'replan_after':True}
  for side,i in [('left',0),('right',1)]:
   limits=np.array([j['range'] for j in o['joint_interface'][side+'_hand']]);action[side+'_fingers']=np.clip(interp('finger_commands',target)[i],limits[:,0],limits[:,1]).tolist()
  if c['mode']=='body':
   query=np.arange(0,duration+1.001,.1);frames=[]
   for t in query:
    v=interp('pose',min(ts[-1],cursor+t*rate));v[3:7]/=np.linalg.norm(v[3:7]);v[7:]=np.clip(v[7:],lo,hi);frames.append([round(float(t),5)]+v.tolist())
   action['body_keyframes']=frames
  else:
   shift=np.array(q['task']['scene'].get('translation',[0,0,0]));root=pose[:3]+shift;v=(root[:2]-o['root_pose'][:2])/max(duration,.1);v*=min(1.,.3/max(np.linalg.norm(v),1e-9));action.update(velocity_xy=v.tolist(),pelvis_height_m=float(np.clip(root[2],.4,.85)))
   for side in ['left','right']:
    action[side+'_position']=(interp(side+'_position',target)+shift+np.array(c.get('wrist_offset',[0,0,0]))).tolist();w=interp(side+'_quaternion_wxyz',target);w/=np.linalg.norm(w);action[side+'_quaternion_wxyz']=w.tolist()
  if cursor>=ts[-1] and gate:
   action.pop('body_keyframes',None);action.update(name='measured_lift',stage='lift',duration_s=1.5,velocity_xy=[0,0],pelvis_height_m=.78,right_position=(np.array(o['right_position'])+[0,0,.035]).tolist(),right_quaternion_wxyz=o['right_quaternion_wxyz'])
  if f>=12:action={'name':'budget_end','stage':'hold','duration_s':.1,'replan_after':False}
  result={'analysis':{'evidence':[{'time':o['time'],'root':o['root_pose'][:3],'wrist':o['right_position'],'target':o['objects']['target'][:3],'contact_s':o['contact_s'],'hand_N':o['contacts']['hand_object_N'],'source_cursor':cursor}], 'hypothesis':c['diagnosis'],'change':c},'actions':[action],'planner_provenance':{'kind':'GPT session authored stage policy','live_model_call':False,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
  (mail/(key+'_response.json')).write_text(json.dumps(result));(mail/(key+'_timing.json')).write_text(json.dumps({'decision_wall_s':time.time()-start}));return result
