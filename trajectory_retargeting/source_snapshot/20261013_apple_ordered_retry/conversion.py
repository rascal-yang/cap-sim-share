"""Native validated prefix + local CaP reference conversion, ORIGINAL scene first."""
import json,time,hashlib,runpy
from pathlib import Path
import numpy as np,pandas as pd
from cap_sim.backends.humano import Backend
from cap_sim.backends.native_sonic import rot
from cap_sim.core import run_task
from cap_sim.io import read,write
from humano.mt.base import MotorCommand
from humano.core.action import ActionCmd
r=Path(__file__).resolve().parent
runpy.run_path(str(r/'pipeline.cap.py'))['gate_native'](r)
old=r.parent/'20261013_cap_ten_tasks';name='G1WholebodyTableApplePickTeleop';df=pd.read_parquet(old/'dataset'/name/'data/chunk-000/episode_000000.parquet')
class HybridBackend(Backend):
 interface={**Backend.interface,'native_prefix':'Validated episode0 token prefix, replay from original initial state; no checkpoint restoration'}
 def reset(self,t,out):
  o=super().reset(t,out);self.d.qacc_warmstart[:]=self.initial['qacc_warmstart'];return o
 def execute(self,a,remaining):
  if a.get('stage')!='native_prefix':
   result=super().execute(a,remaining)
  else:
   assert self.ticks==0 and a['tokens']==250 and remaining>=5
   rows=[];failure=None;bad=0.
   for k in range(a['tokens']):
    row=df.iloc[k];token=np.asarray(row['action.motion_token']);f=np.asarray(row['action.wbc']);s=self.sonic
    s.history.append({'omega':self.d.qvel[3:6].copy(),'q':(self.d.qpos[self.qids]-s.default)[s.order],'dq':self.d.qvel[self.vids][s.order].copy(),'last':s.last.copy(),'gravity':rot(self.d.qpos[3:7]).inv().apply([0,0,-1])})
    obs=np.concatenate([token]+[np.concatenate([h[z] for h in s.history]) for z in ['omega','q','dq','last','gravity']]).astype(np.float32)
    raw=s.decoder.run(None,{s.decoder.get_inputs()[0].name:obs[None]})[0][0];s.last=raw.copy();s.tick+=1;target=s.default+s.scale*raw[s.reverse];self.fingers=[f[29:36],f[36:43]]
    s.records.append({'time':float(self.d.time),'mode':'original_token_prefix','token':token.tolist(),'target':target.tolist()})
    for sub in range(4):
     self.robot.apply_action(ActionCmd('wbc_torque',low_cmd=MotorCommand(q_target=target,kp=s.kp,kd=s.kd),use_sensor=False,left_hand_cmd=MotorCommand.hand_command(self.fingers[0]),right_hand_cmd=MotorCommand.hand_command(self.fingers[1])))
     self.mj.mj_step(self.m,self.d);self.mj.mj_forward(self.m,self.d);self.ticks+=1
     if self.ticks%self.reward_stride==0:
      info=self.e._get_info();self.native_reward=float(self.e.task.compute_reward(info,mujoco_env=self.e.mujoco));self.native_passed=bool(self.e.task.check_success(info,mujoco_env=self.e.mujoco))
     c=self.contacts();self.hand_contact_s=self.hand_contact_s+.005 if c['hand_object_N']>.1 else 0.;stable=bool(self.d.qpos[2]>.3 and rot(self.d.qpos[3:7]).as_matrix()[2,2]>np.cos(np.deg2rad(50)))
     sample={'time':float(self.d.time),'dt':.005,'native_success':self.native_passed,'stable':stable,'native_reward':self.native_reward};self.samples.append(sample);rows.append(sample);self.states.append(np.r_[self.d.time,self.d.qpos,self.d.qvel,self.d.ctrl]);self.contact_records.append(c)
     bad=bad+.005 if c['robot_table_N']>5 else 0.
     if bad>=.05:failure='unexpected_table_contact';break
     if not stable:failure='robot_fall_or_simulation_warning';break
    if failure:break
   result={'completed':failure is None,'reusable':failure is None,'failure':failure,'samples':rows,'observation':self.observe()}
  np.savez_compressed(self.out/'states.npz',samples=np.asarray(self.states));write(self.out/'contacts.json',self.contact_records);return result
class Planner:
 def plan(self,q):
  n=q['attempt'];f=q.get('feedback_index',0);mail=r/'conversion_mailbox';mail.mkdir(exist_ok=True);key=f'a{n:03d}_f{f:02d}';write(mail/(key+'_request.json'),q);print('DECISION',key,flush=True);started=time.time();cfg=r/'conversion_configs'/f'{n:03d}.json'
  while not cfg.exists():time.sleep(.2)
  c=read(cfg);src=r/'convert_policy.py';raw=src.read_bytes();h=hashlib.sha256(raw).hexdigest();d=r/'conversion_policies';d.mkdir(exist_ok=True);(d/(h+'.py')).write_bytes(raw)
  result=runpy.run_path(str(src))['decide'](q,c,r);result['provenance']={'sha256':h,'live_model_call':False};write(mail/(key+'_response.json'),result);write(mail/(key+'_timing.json'),{'decision_wait_wall_s':time.time()-started});return result
if __name__=='__main__':
 task=read(old/'runs/no_reference'/name/'task.json');task['references']=[str((r.parent/'20261013_apple_recovery/reference.json').resolve())];task['code_version']='eda87f8 plus isolated token-prefix adapter';task['budget'].update(attempts=10,optimization_attempts=0,feedback_calls=12);task['generation_kind']='same_scene_conversion';b=HybridBackend()
 try:
  summary=run_task(task,Planner(),b,r/'conversion_run',resume=(r/'conversion_run').exists());out=r/'conversion';out.mkdir(exist_ok=True);write(out/'result.json',summary);print(summary,flush=True)
 finally:b.close()
