import copy
from cap_sim.contact_prior import map_prior
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

def decide(q,c,r):
 o=q['observation'];f=q.get('feedback_index',0);done=q.get('executed_actions',[])
 assert q['phase']=='solve', 'Optimization disabled by user'
 baseline=q['references'][0];actions=baseline['actions'];cursor=-1
 for a in done:
  if a.get('name','').startswith('converted_'):cursor=int(a['name'].split('_')[1])
 i=min(cursor+1,len(actions)-1);a=copy.deepcopy(actions[i]);a['name']=f'converted_{i:03d}';a['replan_after']=True;residuals=[]
 if a['stage']!='native_prefix':
  if a['stage']=='lift' and o['contact_s']<.12:
   i=3;a=copy.deepcopy(actions[i]);a['name']=f'converted_{i:03d}';a['replan_after']=True
  model=c['model_path'];m=mujoco.MjModel.from_binary_path(str(model));d=mujoco.MjData(m);ids=np.array([m.joint(j).qposadr[0] for j in o['body_joint_names']]);lim=np.array([j['range'] for j in o['joint_interface']['body']]);bid=m.body('right_wrist_yaw_link').id;prior=q['contact_privileged']['reference_priors'][0];mapped=q['contact_privileged']['initial_targets'][0];source=map_prior(prior,{'objects':{prior['object_key']:prior['reference_initial_object_pose']}},prior['reference_scale']);shift=np.array(mapped['suggested_wrist_position_world_m'])-np.array(source['suggested_wrist_position_world_m'])+np.array(c.get('offset',[0,0,0]));frames=[]
  for row in a['body_keyframes']:
   pose=np.array(row[1:]);d.qpos[:7]=pose[:7];d.qpos[ids]=pose[7:];mujoco.mj_forward(m,d);goal=d.xpos[bid].copy()+shift;rotation=R.from_quat(np.roll(d.xquat[bid].copy(),-1));seed=pose[29:36].copy()
   def residual(x):
    d.qpos[ids[22:29]]=x;mujoco.mj_forward(m,d);return np.r_[(d.xpos[bid]-goal)*12,(rotation.inv()*R.from_quat(np.roll(d.xquat[bid],-1))).as_rotvec()*2,(x-seed)*.03]
   sol=least_squares(residual,seed,bounds=(lim[22:29,0],lim[22:29,1]),max_nfev=70);pose[29:36]=sol.x;residuals.append(float(np.linalg.norm(residual(sol.x)[:3])/12));frames.append([row[0]]+pose.tolist())
  frames[0]=[0.]+o['root_pose']+o['body_joints'];a['body_keyframes']=frames
 if f>=12:a={'name':'budget_end','stage':'hold','duration_s':.1,'body_keyframes':[[0.]+o['root_pose']+o['body_joints'],[.1]+o['root_pose']+o['body_joints']]}
 return {'analysis':{'evidence':[{'baseline_acceptance':baseline['acceptance'],'baseline_action_index':i,'contact_s':o['contact_s'],'root':o['root_pose'][:3],'wrist':o['right_position'],'max_IK_residual':max(residuals,default=0),'mapped_contact_feedback':q['contact_privileged']['current_targets'],'contact_derived_wrist_shift_m':shift.tolist() if residuals else None}],'hypothesis':'Replay native validated approach from initial state, retarget right-arm frames using wrist translation derived from measured object-local contact centroids and wrist-local contact offsets; preserve tracking compensation from the successful reference. Keep fingers/legs/physics/controller/acceptance fixed; gate lift on actual contacts.','change':c['diagnosis']},'actions':[a]}
