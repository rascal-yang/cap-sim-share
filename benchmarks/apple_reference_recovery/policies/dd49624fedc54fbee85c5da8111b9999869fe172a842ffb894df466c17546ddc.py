import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def decide(q,c,r):
 ref=json.loads((r/'reference.json').read_text());s=ref['samples'];ts=np.array([v['time'] for v in s]);o=q['observation'];f=q.get('feedback_index',0);done=q.get('executed_actions',[])
 def interp(field,t):
  a=np.array([v[field] for v in s]);flat=a.reshape(len(s),-1);return np.array([np.interp(t,ts,flat[:,i]) for i in range(flat.shape[1])]).reshape(a.shape[1:])
 obj=np.array(o['objects']['target'][:3]);shift=obj-np.array(s[0]['object'][:3]);rootgoal=interp('pose',6.7)[:3]+shift;rootgoal+=np.array(c.get('stance_offset',[0,0,0]));root=np.array(o['root_pose'][:3]);err=rootgoal[:2]-root[:2]
 def wrist(t):
  w=interp('right_quaternion_wxyz',t);w/=np.linalg.norm(w);return interp('right_position',t)+shift+np.array(c.get('offset',[0,0,0])),w
 def fingers(t):
  a=interp('finger_commands',t)[1];lim=np.array([j['range'] for j in o['joint_interface']['right_hand']]);return np.clip(a,lim[:,0],lim[:,1]).tolist()
 def arm(name,t,dur,closed=False):
  p,w=wrist(t);return {'name':name,'stage':'contact' if closed else 'approach','duration_s':dur,'velocity_xy':[0,0],'pelvis_height_m':c.get('height',.78),'right_position':p.tolist(),'right_quaternion_wxyz':w.tolist(),'right_fingers':fingers(7.5) if closed else [0]*7,'replan_after':True}
 names=[a['name'] for a in done];gate=o['contact_s']>=.12
 reason=''
 if not any(n.startswith('pregrasp') for n in names) and np.linalg.norm(err)>c.get('stop_distance',.045) and f<c.get('walk_feedback_limit',5):
  v=err*.7;v*=min(1.,c.get('walk_speed',.22)/max(np.linalg.norm(v),1e-9));
  if c.get('constant_walk'):v=err/max(np.linalg.norm(err),1e-9)*c['walk_speed']
  action={'name':'walk_to_reference_stance','stage':'walk','duration_s':c.get('walk_duration',1.5),'velocity_xy':v.tolist(),'facing_yaw':0.,'pelvis_height_m':.78,'replan_after':True};reason='Reach measured reference base position with existing lower-body planner; unspecified wrists follow root.'
 elif c.get('brake') and 'brake' not in names:
  action={'name':'brake','stage':'walk','duration_s':c.get('brake_duration',1.),'velocity_xy':[0,0],'pelvis_height_m':c.get('height',.78),'replan_after':True};reason='Brake walking before lowering the hand; observe root drift.'
 elif 'pregrasp' not in names:
  action=arm('pregrasp',5.8,2.);reason='Reference approach wrist pose with open fingers, after measured stance approach.'
 elif 'contact_pose' not in names:
  action=arm('contact_pose',6.7,2.);reason='Move open hand to the reference preclosure pose; preserve orientation.'
 elif 'close' not in names:
  action=arm('close',6.7,1.5,True);reason='Execute recorded finger commands separately from reference time; contact need not exist before closure.'
 elif gate:
  action=arm('lift',7.5,1.5,True);action.update(stage='lift',right_position=(np.array(o['right_position'])+[0,0,.035]).tolist(),right_quaternion_wxyz=o['right_quaternion_wxyz']);reason='Measured contact sustained; advance lift 3.5cm and reobserve.'
 else:
  p,w=wrist(6.7);error=p-np.array(o['right_position']);correction=np.clip(error,-.08,.08)*c.get('correction_gain',.6)
  action=arm('correct_contact',6.7,1.5,True);action['right_position']=(p+correction).tolist();reason='No sustained target contact: correct measured wrist tracking residual, keeping reference closure and orientation. Bounded correction enters actual Cartesian command.'
 if f>=12:action={'name':'budget_end','stage':'hold','duration_s':.1};reason='Feedback budget reached; stop.'
 return {'analysis':{'evidence':[{'root_error':err.tolist(),'right_wrist':o['right_position'],'target':obj.tolist(),'contact_s':o['contact_s'],'hand_force_N':o['contacts']['hand_object_N'],'previous_failure':q['history'][-1]['failure'] if q['history'] else None}],'hypothesis':reason,'change':c['diagnosis']},'actions':[action]}
