"""Read-only generic contact geometry/force/tracking diagnostics for API needs_tool."""
import json
from pathlib import Path
import numpy as np
from .agent import save

def aggregate_normal_forces(pairs):
 total={};single={}
 for p in pairs:
  name=p['body'];force=p['normal_N'];total[name]=total.get(name,0)+force;single[name]=max(single.get(name,0),force)
 return total,single

def diagnose(spec,out):
 import mujoco
 from scipy.spatial.transform import Rotation as R
 from .suite_physics import open_source,source_poses
 src=spec['source'];case=Path(src['experiment_path']);out=Path(out);out.mkdir(parents=True,exist_ok=True);b,df,files,task=open_source(spec['dataset'],src['episode'],out,src['object_delta_xy_m'])
 try:
  m=b.m;d=b.d;tr=np.load(case/'execution/trajectory.npz');ctrl=np.load(case/'execution/controller.npz')['records'];q,v,poses=source_poses(b,df);target=b.object_ids.get('target',b.object_ids.get('articulated'))
  if target is None:raise ValueError('No registered target')
  def desc(body,parent):
   while body>0:
    if body==parent:return True
    body=int(m.body_parentid[body])
   return False
  tg={g for g in range(m.ngeom) if desc(int(m.geom_bodyid[g]),target)};robot_root=m.body('pelvis').id;robot={i for i in range(m.nbody) if desc(i,robot_root)}
  plan=json.loads(Path(src['decision_path']).read_text())['decision'];names=plan['contact_control']['contact_bodies'];peaks={};single_peaks={};impacts=[];samples=[];errors=[];point_offsets=[np.array([.18,-.025,0]),np.array([.18,.025,0])]
  for i,st in enumerate(tr['next_state']):
   d.qpos[:]=st[1:1+m.nq];d.qvel[:]=st[1+m.nq:];d.ctrl[:]=tr['commands'][i,-m.nu:];mujoco.mj_forward(m,d);pairs=[];total=np.zeros(3)
   for ci,c in enumerate(d.contact):
    a,z=int(c.geom1),int(c.geom2);other=z if a in tg else a if z in tg else None
    if other is None or int(m.geom_bodyid[other]) not in robot:continue
    f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);world=np.asarray(c.frame).reshape(3,3).T@f[:3];on_target=-world if a in tg else world;name=m.body(int(m.geom_bodyid[other])).name;normal=max(0.,float(f[0]));total+=on_target;pairs.append({'body':name,'normal_N':normal,'force_on_target_world_N':on_target.tolist(),'distance_m':float(c.dist),'point_world':c.pos.tolist()})
   aggregate,single=aggregate_normal_forces(pairs)
   for name,f in aggregate.items():peaks[name]=max(peaks.get(name,0),f)
   for name,f in single.items():single_peaks[name]=max(single_peaks.get(name,0),f)
   row={'frame':i//4,'substep':i,'time_s':float(st[0]),'target_xyz':d.xpos[target].tolist(),'root_xyz':d.qpos[:3].tolist(),'force_on_target_world_N':total.tolist(),'pairs':pairs}
   if pairs:impacts.append(row)
   if i%4!=3:continue
   frame=i//4
   if frame<len(ctrl) and ctrl[frame,1]==1:
    # Encoder local VR targets refer to pre-control root and source root orientation.
    e=ctrl[frame,-1791:-29];pre=tr['state'][frame*4];anchor=R.from_quat(np.roll(q[min(frame,len(q)-1),3:7],-1));targets=e[901:910].reshape(3,3);wrists=[]
    for j,body in enumerate([b.left,b.right]):
     desired=pre[1:4]+anchor.apply(targets[j]);actual=d.xpos[body]+d.xmat[body].reshape(3,3)@point_offsets[j];err=actual-desired;errors.append(np.linalg.norm(err));wrists.append({'side':['left','right'][j],'commanded_vr_point_world':desired.tolist(),'actual_vr_point_world':actual.tolist(),'error_m':err.tolist(),'actual_wrist_origin_world':d.xpos[body].tolist(),'wrist_to_target_origin_world':(d.xpos[target]-d.xpos[body]).tolist()})
    row['wrist_tracking']=wrists
   if frame%20==0 or frame in [src['planned_prefix_steps'],len(ctrl)-1]:
    nearest=[]
    for name in names:
     body=m.body(name).id;best=None
     for g in range(m.ngeom):
      if int(m.geom_bodyid[g])!=body or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
      for z in tg:
       points=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,g,z,.5,points)
       if abs(abs(dist)-np.linalg.norm(points[3:]-points[:3]))>.002:continue
       if best is None or dist<best['distance_m']:best={'body':name,'target_geom':m.geom(z).name,'distance_m':float(dist),'from_point':points[:3].tolist(),'to_point':points[3:].tolist(),'toward_surface_world':(points[3:]-points[:3]).tolist()}
     if best:nearest.append(best)
    row['nearest_target_surfaces']=nearest;samples.append(row)
  strongest=sorted(impacts,key=lambda x:sum(p['normal_N'] for p in x['pairs']),reverse=True)[:8]
  result={'source_case':str(case),'physics_steps':0,'meaning':'reconstructed saved-state contacts and post-step actual VR-point error vs pre-control command; geometry/force diagnosis not dynamics prediction','target_body':m.body(target).name,'peak_normal_force_by_body_N':peaks,'peak_single_contact_normal_force_by_body_N':single_peaks,'force_aggregation':'peak per-substep sum over all contacts on each body; single-contact peak reported separately','wrist_vr_point_error_rmse_m':float(np.sqrt(np.mean(np.square(errors)))) if errors else None,'strongest_contact_substeps':strongest,'phase_samples':samples,'capability_limit':'paired wrist translations can affect decoder body outputs indirectly; this read-only tool does not predict pelvis/foot motion or prove a repair feasible; no direct leg/elbow channel in transfer profile'};save(out/'diagnostic.json',result);return result
 finally:b.close()
