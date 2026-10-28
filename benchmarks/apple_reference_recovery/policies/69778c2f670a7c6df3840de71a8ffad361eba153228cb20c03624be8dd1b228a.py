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
  p,w=wrist(t);return {'name':name,'stage':'contact' if closed else 'approach','duration_s':dur,'velocity_xy':c.get('arm_velocity',[0,0]),'pelvis_height_m':c.get('height',.78),'right_position':p.tolist(),'right_quaternion_wxyz':w.tolist(),'right_fingers':fingers(7.5) if closed else [0]*7,'replan_after':True}
 names=[a['name'] for a in done];gate=o['contact_s']>=.12
 reason=''
 if not any(n.startswith('pregrasp') for n in names) and np.linalg.norm(err)>c.get('stop_distance',.045) and f<c.get('walk_feedback_limit',5):
  v=err*.7;v*=min(1.,c.get('walk_speed',.22)/max(np.linalg.norm(v),1e-9));
  if c.get('constant_walk'):v=err/max(np.linalg.norm(err),1e-9)*c['walk_speed']
  action={'name':'walk_to_reference_stance','stage':'walk','duration_s':c.get('walk_duration',1.5),'velocity_xy':v.tolist(),'facing_yaw':0.,'pelvis_height_m':.78,'replan_after':True};reason='Reach measured reference base position with existing lower-body planner; unspecified wrists follow root.'
 elif c.get('brake') and 'brake' not in names:
  action={'name':'brake','stage':'walk','duration_s':c.get('brake_duration',1.),'velocity_xy':c.get('brake_velocity',[0,0]),'pelvis_height_m':c.get('height',.78),'replan_after':True};reason='Brake walking before lowering the hand; observe root drift.'
 elif c.get('settle') and 'settle' not in names:
  action={'name':'settle','stage':'walk','duration_s':c['settle'],'velocity_xy':[0,0],'pelvis_height_m':c.get('height',.78),'replan_after':True};reason='Reobserve after active braking before arm approach.'
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
 if c.get('body_stance') and action['name'] not in ('walk_to_reference_stance','budget_end'):
  import mujoco
  from scipy.optimize import least_squares
  phase='reference_recovery' if q['task'].get('generation_kind')=='reference_recovery' else 'generation'
  model=r/'run'/phase/'attempts'/f"{q['attempt']:03d}"/'scene.mjb';m=mujoco.MjModel.from_binary_path(str(model));d=mujoco.MjData(m)
  ids=np.array([m.joint(x).qposadr[0] for x in o['body_joint_names']]);lim=np.array([j['range'] for j in o['joint_interface']['body']]);pose=interp('pose',7.5);pose[:7]=o['root_pose'];pose[7:]=np.clip(pose[7:],lim[:,0],lim[:,1]);d.qpos[:7]=pose[:7];d.qpos[ids]=pose[7:]
  if action['name'] in ('brake','settle'):
   pose[22:]=np.asarray(o['body_joints'])[15:];d.qpos[ids]=pose[7:]
  else:
   sl=slice(22,29);jids=ids[sl];seed=pose[7:][sl].copy();goal=np.asarray(action['right_position']);grot=R.from_quat(np.roll(action['right_quaternion_wxyz'],-1));bid=m.body('right_wrist_yaw_link').id
   def residual(x):
    d.qpos[jids]=x;mujoco.mj_forward(m,d);actual=R.from_quat(np.roll(d.xquat[bid],-1));return np.r_[(d.xpos[bid]-goal)*12,(grot.inv()*actual).as_rotvec()*2,(x-seed)*.03]
   sol=least_squares(residual,seed,bounds=(lim[sl,0],lim[sl,1]),max_nfev=80);pose[29:36]=sol.x
   reason+=' Offline arm IK position residual='+str(float(np.linalg.norm(residual(sol.x)[:3])/12))
  initial=np.r_[o['root_pose'],o['body_joints']];dur=action['duration_s'];action['body_keyframes']=[[0.]+initial.tolist(),[dur]+pose.tolist(),[dur+1.]+pose.tolist()]
  reason+=' Use reference standing legs through existing full-body SONIC channel; no state injection.'
 if f>=12:action={'name':'budget_end','stage':'hold','duration_s':.1};reason='Feedback budget reached; stop.'
 return {'analysis':{'evidence':[{'root_error':err.tolist(),'right_wrist':o['right_position'],'target':obj.tolist(),'contact_s':o['contact_s'],'hand_force_N':o['contacts']['hand_object_N'],'previous_failure':q['history'][-1]['failure'] if q['history'] else None}],'hypothesis':reason,'change':c['diagnosis']},'actions':[action]}
