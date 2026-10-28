import sys,time,copy,hashlib,runpy,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'20261013_apple_ordered_retry'
sys.path.insert(0,str(OLD))
from conversion import HybridBackend
from cap_sim.core import run_task
from cap_sim.io import read,write
runpy.run_path(str(OLD/'pipeline.cap.py'))['gate_conversion'](OLD)
CASES={'x_plus_10cm':{'translation':[.1,0,0]},'x_minus_10cm':{'translation':[-.1,0,0]},'y_minus_10cm':{'translation':[0,-.1,0]},'table_plus_5cm':{'translation':[0,0,0],'table_delta':.05},'table_minus_5cm':{'translation':[0,0,0],'table_delta':-.05},'apple_scale_110':{'translation':[0,0,0],'apple_scale':1.1}}
class VariantBackend(HybridBackend):
 def _prepare(self,task):
  if hasattr(self,'task') and self.task==task:return
  super()._prepare(task)
  s=task['scene'].get('apple_scale',1.);dz=task['scene'].get('table_delta',0.)
  bid=self.object_ids['target'];tab=self.m.body('table').id;jid=self.m.body_jntadr[bid];adr=self.m.jnt_qposadr[jid]
  before={'mass':float(self.m.body_mass[bid]),'inertia':self.m.body_inertia[bid].tolist(),'apple_position':self.initial_q[adr:adr+3].tolist(),'table_position':self.m.body_pos[tab].tolist()}
  gids=np.flatnonzero(self.m.geom_bodyid==bid)
  def bottom():
   points=[]
   for g in gids:
    mid=self.m.geom_dataid[g];a=self.m.mesh_vertadr[mid];n=self.m.mesh_vertnum[mid];v=self.m.mesh_vert[a:a+n]
    points.append(v@self.d.geom_xmat[g].reshape(3,3).T+self.d.geom_xpos[g])
   p=np.concatenate(points);return float(p[:,2].min()),np.ptp(p,axis=0).tolist()
  low,extent=bottom();before.update(bottom=low,extent=extent)
  tg=self.m.geom('table_geom').id;center=self.initial_q[adr:adr+2];tablexy=self.d.geom_xpos[tg,:2];half=self.m.geom_size[tg,:2]
  if np.any(np.abs(center-tablexy)+np.array(extent[:2])/2>half):raise ValueError('Invalid tabletop variant: apple footprint leaves table support')
  if s!=1 or dz:
   spec=self.e.mjSpec
   if dz:spec.body(self.m.body(tab).name).pos[2]+=dz
   if s!=1:
    for mid in set(int(self.m.geom_dataid[g]) for g in gids):spec.mesh(self.m.mesh(mid).name).scale*=s
    for g in gids:spec.geom(self.m.geom(g).name).pos*=s
   self.m,self.d=spec.recompile(self.m,self.d)
   self.e.mjModel=self.e.mujoco.mjModel=self.m;self.e.mjData=self.e.mujoco.mjData=self.d
   self.robot.mjModel=self.m;self.robot.mjData=self.d
   self.e.mujoco._refresh_mj_object_handles()
   self.e.mujoco.joints,self.e.mujoco.actuators=self.robot.setup_control(self.d,self.m,mjSpec=spec)
   self.m.body_mass[bid]=before['mass'];self.m.body_inertia[bid]=np.array(before['inertia'])*s*s
   self.mj.mj_setConst(self.m,self.d)
   self.d.qpos[:]=self.initial_q;self.mj.mj_forward(self.m,self.d)
   newlow,_=bottom();self.initial_q[adr+2]+=low-newlow+dz
   self.model_xml=spec.to_xml();self._restore()
  low,extent=bottom()
  self.geometry={'before':before,'after':{'mass':float(self.m.body_mass[bid]),'inertia':self.m.body_inertia[bid].tolist(),'apple_position':self.initial_q[adr:adr+3].tolist(),'table_position':self.m.body_pos[tab].tolist(),'bottom':low,'extent':extent},'scale':s,'table_delta':dz}
  assert abs(self.m.body_mass[bid]-before['mass'])<1e-10
  assert np.allclose(np.array(extent),np.array(before['extent'])*s,atol=1e-6)
  assert abs(low-before['bottom']-dz)<1e-6
 def reset(self,t,out):
  o=super().reset(t,out);write(Path(out)/'geometry.json',self.geometry);return o
class Planner:
 def __init__(self,case):self.case=case
 def plan(self,q):
  n=q['attempt'];f=q.get('feedback_index',0);p=ROOT/self.case;mail=p/'mailbox';mail.mkdir(parents=True,exist_ok=True);key=f'a{n:03d}_f{f:02d}'
  write(mail/(key+'_request.json'),q);print('DECISION',self.case,key,flush=True)
  cfg=p/'configs'/f'{n:03d}.json';start=time.time()
  while not cfg.exists():time.sleep(.2)
  c=read(cfg);c['model_path']=str(p/'run/attempts'/f'{n:03d}'/'scene.mjb')
  raw=(ROOT/'policy.py').read_bytes();sha=hashlib.sha256(raw).hexdigest();snap=p/'policies';snap.mkdir(exist_ok=True);(snap/(sha+'.py')).write_bytes(raw)
  result=runpy.run_path(str(ROOT/'policy.py'))['decide'](q,c,ROOT)
  result['provenance']={'sha256':sha,'live_model_call':False,'identity':'Current GPT session authored policy; no external model API'}
  write(mail/(key+'_response.json'),result);write(mail/(key+'_timing.json'),{'wall_s':time.time()-start});return result
if __name__=='__main__':
 case=sys.argv[1];task=read(OLD/'conversion_run/task.json');task['references']=[str(OLD/'conversion_run/success/attempt_001/trajectory.json')];task['scene']=CASES[case];task['generation_kind']='augmented';task['budget']['optimization_attempts']=0
 task['code_version']='isolated variants adapter: native prefix + local right arm retarget';b=VariantBackend();out=ROOT/case/'run'
 try:print(run_task(task,Planner(case),b,out,resume=out.exists()),flush=True)
 finally:b.close()
