"""Offline measured-state reference audit; prohibited before no-reference freeze."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,time,hashlib,argparse,shutil
from pathlib import Path
import numpy as np,pandas as pd,mujoco
from cap_sim.backends.humano import Backend
from cap_sim.io import write,read
from humano.core.action import ActionCmd
from humano.mt.base import MotorCommand
from humano.mt.sonic import policy_parameters as pp
p=argparse.ArgumentParser();p.add_argument('--episode',type=int,default=0);args=p.parse_args();episode=args.episode
r=Path(__file__).resolve().parent
freeze=read(r/'no_reference_frozen.json');assert freeze['cases']==10
for f,h in freeze['sha256'].items():assert hashlib.sha256((r/f).read_bytes()).hexdigest()==h
out=r/'references';out.mkdir(exist_ok=True)
for case in sorted((r/'runs/no_reference').iterdir()):
 if not (case/'summary.json').exists():continue
 name=case.name;dest=out/(name+'.json')
 if dest.exists():
  existing=read(dest)
  if episode==0 or existing['native_success_from_saved_states'] or existing['episode']>=episode:continue
  archive=out/'candidates'/name;archive.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,archive/'episode_000000.json')
 started=time.time();b=Backend();task=read(case/'task.json')
 if episode:
  prepared=out/'source_prepared'/name;prepared.mkdir(parents=True,exist_ok=True)
  meta=read(r/'dataset'/name/'sonic'/f'episode_{episode:06d}_replay.json');write(prepared/'environment_config.json',json.loads(meta['environment_config']))
  task['resources']['prepared']=str(prepared);task['resources']['initial_state']=str(r/'dataset'/name/'sonic'/f'episode_{episode:06d}_start.npz')
 b.describe(task)
 try:
  source=r/'dataset'/name/f'data/chunk-000/episode_{episode:06d}.parquet';df=pd.read_parquet(source)
  print(name,len(df),'columns',list(df.columns),flush=True)
  if 'sim.qpos' not in df:raise ValueError('Exact physical states unavailable')
  records=[];first_pass=None;first_contact=None;peakhand=0.;next_reward=0.;passed=False
  def scalar(v):return float(np.asarray(v).reshape(-1)[0])
  times=np.array([scalar(v) for v in df['sim.time']]);times-=times[0]
  for i,(_,row) in enumerate(df.iterrows()):
   b.d.time=float(times[i]);b.d.qpos[:]=np.asarray(row['sim.qpos'])[:b.m.nq];b.d.qvel[:]=np.asarray(row['sim.qvel'])[:b.m.nv];mujoco.mj_forward(b.m,b.d)
   if 'sim.ctrl' in df:b.d.ctrl[:]=np.asarray(row['sim.ctrl'])[:b.m.nu]
   else:
    action=np.asarray(row['action.wbc']);b.robot.apply_action(ActionCmd('wbc_torque',low_cmd=MotorCommand(q_target=action[:29],kp=pp.kps,kd=pp.kds),use_sensor=False,left_hand_cmd=MotorCommand.hand_command(action[29:36]),right_hand_cmd=MotorCommand.hand_command(action[36:43])))
   mujoco.mj_forward(b.m,b.d)
   if times[i]+1e-6>=next_reward:
    info=b.e._get_info();reward=float(b.e.task.compute_reward(info,mujoco_env=b.e.mujoco));passed=bool(b.e.task.check_success(info,mujoco_env=b.e.mujoco));next_reward=times[i]+.02
   c=b.contacts();peakhand=max(peakhand,c['hand_object_N'])
   if c['hand_object_N']>.1 and first_contact is None:first_contact=float(times[i])
   if passed and first_pass is None:first_pass=float(times[i])
   if i%5==0 or passed or i==len(df)-1:
    ob=b.observe();records.append({'time':float(times[i]),'pose':np.r_[b.d.qpos[:7],b.d.qpos[b.qids]].tolist(),'left_position':ob['left_position'],'right_position':ob['right_position'],'left_quaternion_wxyz':ob['left_quaternion_wxyz'],'right_quaternion_wxyz':ob['right_quaternion_wxyz'],'fingers':[b.d.qpos[b.m.jnt_qposadr[getattr(b.robot,s+'_hand_actuated_index')]].tolist() for s in ['left','right']],'object':ob['objects'].get('target',ob['objects'].get('articulated')),'contacts':c})
   if first_pass is not None and times[i]>=first_pass+.2:break
  report={'kind':'native_measured_reference','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'episode':episode,'extracted_unix_s':time.time(),'no_reference_freeze_unix_s':freeze['frozen_unix_s'],'native_success_from_saved_states':first_pass is not None,'first_success_s':first_pass,'first_hand_contact_s':first_contact,'peak_hand_force_N':peakhand,'simulated_reexecution':False,'control_evidence':'recorded sim.ctrl' if 'sim.ctrl' in df else 'PD reconstructed from recorded action.wbc and sampled qpos/qvel; contact forces are estimates, not exact recorded forces','task':name,'samples':records,'wall_s':time.time()-started,'skill':{'approach':'Track measured body motion through SONIC, retarget arm endpoints to changed object; never set physical states during execution.','contact':'Use measured hand-object contacts before lifting; corrective motions are bounded and re-observed.','retry':'Diagnose saved failure before reset; alter motion timing/alignment only.','optimization':'Only after success; keep accepted baseline if candidate fails.'}}
  write(dest,report);print('EXTRACTED',name,first_pass,first_contact,len(records),flush=True)
 finally:b.close()
