"""Reference-conditioned CaP: new lateral alignment, IK retargeting, contact gates.
No expert control/state injection; outputs the same bounded SONIC action API.
"""
import json,hashlib,copy,math
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R,Slerp
from serve_remaining import hold
r=Path(__file__).resolve().parent
sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def respond(p):
 q=json.loads(p.read_text());o=q['observation'];n=q['attempt'];f=q.get('feedback_index',0);ref=q['references'][0];samples=ref['samples'];name=q['task']['name'];pick='Pick' in name
 previous=q['history'][-1] if q['history'] else {};shift=np.asarray(q['task']['scene']['translation']);executed=q.get('executed_actions',[])
 initial_request=p.with_name(f'a{n:03d}_f00_request.json')
 if initial_request.exists():
  initial_ob=json.loads(initial_request.read_text())['observation'];target0=initial_ob['objects'].get('target',initial_ob['objects'].get('articulated'));shift=np.asarray(target0[:3])-np.asarray(samples[0]['object'][:3])
 analysis={'evidence':[{'reference_sha256':ref['source_sha256'],'reference_saved_state_success':ref['native_success_from_saved_states'],'previous_failure':previous.get('failure'),'measured_time':o['time'],'hand_force_N':o['contacts']['hand_object_N'],'contact_s':o['contact_s'],'scene_translation':shift.tolist()}],'hypothesis':'Retarget reference geometry under unchanged controller, contact gates and native acceptance.'}
 def finish(actions):
  dest=p.with_name(p.name.replace('_request','_response'));assert not dest.exists()
  dest.write_text(json.dumps({'analysis':analysis,'actions':actions,'planner_provenance':{'kind':'GPT-authored reference-conditioned CaP','policy_sha256':sha,'live_model_call':False}},indent=2))
 if q['phase']=='optimize':
  actions=copy.deepcopy(q['successful_trajectory']['actions']);scale=min(1.12,29.9/sum(a['duration_s'] for a in actions))
  for a in actions:
   a['duration_s']=round(a['duration_s']*scale/.005)*.005
   if 'body_keyframes' in a:
    for row in a['body_keyframes']:row[0]*=scale
  analysis['hypothesis']='Accepted success is preserved. Slower reference candidate is retained only if same acceptance and measured smoothness improve.';finish(actions);return
 if f==0:
  speed=.1+[0,.015,-.015,.025,-.025][(n-1)%5]
  analysis['hypothesis']=f"Failure diagnosis before reset: {previous.get('failure','first reference attempt')}. Insert a new lateral approach at {speed:.3f} m/s for the 10cm object shift, then retarget measured reference contacts."
  finish([{'name':'new_lateral_alignment','stage':'walk','duration_s':1.,'velocity_xy':[0,speed],'pelvis_height_m':.74,'replan_after':True}]);return
 remaining=q.get('remaining_sim_seconds',30)
 if f>=12 or remaining<.5:
  analysis['hypothesis']='Feedback/time budget reached without verified completion; hold and finish.';finish([hold(o)]);return
 endref=float(samples[-1]['time']);cursor=0.
 for a in executed:
  if a.get('name','').startswith('reference_to_'):cursor=float(a['name'].split('reference_to_')[1])
 if cursor>=endref-1e-5:
  analysis['hypothesis']='Reference motion exhausted without native success; hold current actual posture to test required dwell, do not fabricate a pass.';finish([hold(o,min(2.,remaining),replan=False)]);return
 rate=max(1.,endref/25.)*[1.,.9,1.1,.95,1.05][(n-1)%5]
 duration=min(2.5,remaining);target=min(endref,cursor+duration*rate)
 contact_t=ref.get('first_hand_contact_s')
 if pick and contact_t is not None and cursor<contact_t<target:target=contact_t
 duration=max(.1,round(((target-cursor)/rate)/.005)*.005)
 ts=np.array([x['time'] for x in samples]);poses=np.array([x['pose'] for x in samples]);finger=np.array([x['fingers'] for x in samples]);obj=np.array([x['object'][:3] for x in samples]);query=np.linspace(cursor,target,max(2,int((target-cursor)/.1)+1))
 def interp(arr,t):return np.array([np.interp(t,ts,arr[:,i]) for i in range(arr.shape[1])])
 wants_lift=pick and interp(obj,target)[2]>obj[0,2]+.015
 events=o['contacts']['events'];bilateral=any(s in name for s in ['Box','Plate','Basket'])
 sides=[any(any(side+'_hand' in b for b in e['bodies']) for e in events) for side in ['left','right']]
 gated=o['contact_s']>=.12 and (not bilateral or all(sides))
 if wants_lift and not gated:
  target=cursor;query=np.array([cursor,cursor]);duration=min(2.,remaining)
  analysis['hypothesis']='Reference is about to lift but actual required hand contacts are absent. Hold reference contact posture and apply bounded wrist alignment correction; do not advance lift time.'
 model=r/'runs/reference_augmented'/name/'attempts'/f'{n:03d}'/'scene.mjb';m=mujoco.MjModel.from_binary_path(str(model));d=mujoco.MjData(m)
 qids=np.array([m.joint(x).qposadr[0] for x in o['body_joint_names']]);jids=np.array([m.joint(x).id for x in o['body_joint_names']]);limits=m.jnt_range[jids]
 frames=[];residuals=[];actualroot=np.asarray(o['root_pose'][:3]);source_root=interp(poses[:,:3],cursor)
 # Offset relative to measured current root, not a world-frame state overwrite.
 correction=shift-(actualroot-source_root);correction[2]=0.;correction=np.clip(correction,-.15,.15)
 for t in query:
  pose=interp(poses,t);pose[3:7]/=np.linalg.norm(pose[3:7]);pose[7:]=np.clip(pose[7:],limits[:,0],limits[:,1]);d.qpos[:7]=pose[:7];d.qpos[qids]=pose[7:]
  for side,sl in [('left',slice(15,22)),('right',slice(22,29))]:
   bid=m.body(side+'_wrist_yaw_link').id;mujoco.mj_forward(m,d);goal=d.xpos[bid].copy()+correction
   # Small retry offsets derive from contact failures, never alter scene/physics.
   if pick and wants_lift and not gated:goal[2]-=min(.025,.005*((n-1)%5))
   ids=qids[sl];lo=limits[sl,0];hi=limits[sl,1];seed=d.qpos[ids].copy()
   def residual(x):
    d.qpos[ids]=x;mujoco.mj_forward(m,d);return np.r_[(d.xpos[bid]-goal)*12,(x-seed)*.04]
   sol=least_squares(residual,seed,bounds=(lo,hi),max_nfev=35);d.qpos[ids]=sol.x;pose[7+sl.start:7+sl.stop]=sol.x;residuals.append(float(np.linalg.norm(residual(sol.x)[:3])/12))
  frames.append(pose.tolist())
 # Blend from the measured posture to the next sparse reference chunk.
 initial=np.r_[o['root_pose'],np.clip(o['body_joints'],limits[:,0],limits[:,1])].tolist()
 times=np.linspace(min(.15,duration/3),duration,len(frames));body=[[0.]+initial]+[[float(t)]+v for t,v in zip(times,frames)]
 ff=np.stack([interp(finger[:,i,:],target) for i in range(2)])
 for i,side in enumerate(['left','right']):
  lim=np.array([j['range'] for j in o['joint_interface'][side+'_hand']]);ff[i]=np.clip(ff[i],lim[:,0],lim[:,1])
 stage=('lift' if wants_lift and gated else 'contact') if pick else ('kick' if 'Ball' in name else 'sit' if 'Sit' in name else 'press' if 'Button' in name else 'push')
 action={'name':f'reference_to_{target:.6f}','stage':stage,'duration_s':duration,'body_keyframes':body,'left_fingers':ff[0].tolist(),'right_fingers':ff[1].tolist(),'replan_after':True}
 analysis['evidence'].append({'source_cursor_s':cursor,'source_target_s':target,'root_relative_retarget_m':correction.tolist(),'max_kinematic_residual_m':max(residuals),'bilateral_contact_required':bilateral,'contact_gate':gated})
 finish([action])
