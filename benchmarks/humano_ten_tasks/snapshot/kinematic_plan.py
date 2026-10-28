"""GPT-requested kinematic target construction. Never steps or writes execution state.
Inputs are current measured observation and robot geometry, not source actions.
"""
import sys,json
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
r=Path(__file__).resolve().parent
request=Path(sys.argv[1]);cfg=json.load(sys.stdin);dct=json.loads(request.read_text());o=dct['observation'];n=dct['attempt'];name=dct['task']['name'];group=request.parent.parent.name
model_path=r/'runs'/group/name/'attempts'/f'{n:03d}'/'scene.mjb'
if not model_path.exists():model_path=sorted((r/'runs'/group/name/'attempts').glob('*/scene.mjb'))[-1]
m=mujoco.MjModel.from_binary_path(str(model_path));d=mujoco.MjData(m)
init=json.loads((r/'mailbox'/group/name/'a001_f00_request.json').read_text())['observation']
d.qpos[:7]=o['root_pose'];jids=np.array([m.joint(k).id for k in o['body_joint_names']]);qi=m.jnt_qposadr[jids];start=np.r_[o['root_pose'],o['body_joints']]
d.qpos[qi]=o['body_joints'];d.qpos[qi[:15]]=init['body_joints'][:15];d.qpos[2]=cfg.get('root_z',init['root_pose'][2]);d.qpos[3:7]=init['root_pose'][3:7]
if cfg.get('preserve_yaw'):
 yaw=R.from_quat(np.roll(o['root_pose'][3:7],-1)).as_euler('xyz')[2];d.qpos[3:7]=np.roll((R.from_euler('z',yaw)*R.from_quat(np.roll(init['root_pose'][3:7],-1))).as_quat(),1)
for side in ['left','right']:
 for a in o['joint_interface'][side+'_hand']:d.qpos[m.jnt_qposadr[m.joint(a['name']).id]]=a['qpos'][0]
d.qpos[qi[14]]=cfg.get('waist_pitch',init['body_joints'][14])
residuals={}
# Solve a double-support crouch from the reset stance at the current root XY.
if 'root_z' in cfg:
 desired_z=d.qpos[2];d.qpos[2]=init['root_pose'][2];mujoco.mj_forward(m,d)
 feet={side:(d.xpos[m.body(side+'_ankle_roll_link').id].copy(),R.from_matrix(d.xmat[m.body(side+'_ankle_roll_link').id].reshape(3,3))) for side in ['left','right']}
 if cfg.get('stance_extra_y'):
  for side in feet:feet[side][0][1]+=cfg['stance_extra_y']*(1 if side=='left' else -1)
 d.qpos[2]=desired_z
 for side,ix in [('left',slice(0,6)),('right',slice(6,12))]:
  ids=jids[ix];addr=qi[ix];bid=m.body(side+'_ankle_roll_link').id;pos,ori=feet[side]
  def foot_res(x):
   d.qpos[addr]=x;mujoco.mj_forward(m,d)
   return np.r_[(d.xpos[bid]-pos)*10,(ori.inv()*R.from_matrix(d.xmat[bid].reshape(3,3))).as_rotvec()]
  sol=least_squares(foot_res,np.clip(d.qpos[addr],m.jnt_range[ids,0]+.001,m.jnt_range[ids,1]-.001),bounds=(m.jnt_range[ids,0]+.001,m.jnt_range[ids,1]-.001),max_nfev=160)
  d.qpos[addr]=sol.x;residuals[side+'_foot']=float(np.linalg.norm(foot_res(sol.x)))
for side,ix in [('left',slice(15,22)),('right',slice(22,29))]:
 if side+'_position' not in cfg:continue
 ids=jids[ix];addr=qi[ix];base=d.qpos[addr].copy();bid=m.body(side+'_wrist_yaw_link').id;target=np.array(cfg[side+'_position']);q=cfg.get(side+'_quaternion_wxyz',[1,0,0,0]);rotation=R.from_quat(np.roll(q,-1));seed=base.copy()
 def residual(x):
  d.qpos[addr]=x;mujoco.mj_forward(m,d)
  return np.r_[(d.xpos[bid]-target)*12,(rotation.inv()*R.from_matrix(d.xmat[bid].reshape(3,3))).as_rotvec()*.6,(x-seed)*.03]
 sol=least_squares(residual,np.clip(base,m.jnt_range[ids,0]+.001,m.jnt_range[ids,1]-.001),bounds=(m.jnt_range[ids,0]+.001,m.jnt_range[ids,1]-.001),max_nfev=160)
 d.qpos[addr]=sol.x;mujoco.mj_forward(m,d);residuals[side]={'position_error_m':float(np.linalg.norm(d.xpos[bid]-target)),'predicted_wrist':d.xpos[bid].tolist()}
end=np.r_[d.qpos[:7],d.qpos[qi]];middle=start.copy();middle[2:7]=end[2:7];middle[7:22]=end[7:22]
duration=cfg.get('duration_s',4.0);move=cfg.get('move_s',2.8);frames=[np.r_[0,start].tolist(),np.r_[move,end].tolist(),np.r_[duration,end].tolist()]
a={'stage':cfg.get('stage','approach'),'duration_s':duration,'body_keyframes':frames,'replan_after':cfg.get('replan_after',True)}
for side in ['left','right']:
 if side+'_fingers' in cfg:a[side+'_fingers']=cfg[side+'_fingers']
response={'analysis':{'evidence':cfg['evidence']+[{'kinematic_prediction_only':residuals}],'hypothesis':cfg['hypothesis']},'actions':[a]}
p=request.with_name(request.name.replace('_request','_response'));assert not p.exists();p.write_text(json.dumps(response,ensure_ascii=False,indent=2));print(residuals)
