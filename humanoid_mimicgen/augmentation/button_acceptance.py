"""Versioned button acceptance override: exact palm collision mesh, unchanged native gates."""
import json
from pathlib import Path
import numpy as np
from .agent import save
from ..workflow.contracts import file_hash,now
PROFILE='button_palm_mesh_v1'
NATIVE_KEYS=('palm','thumb','index','middle','ring','little','pinky')

def accepts_contact(body,mesh,revised=False):
 return any(k in body.lower() for k in NATIVE_KEYS) or bool(revised and body=='right_wrist_yaw_link' and mesh=='right_hand_palm_link')

def hold_result(rows):
 count=longest=0;counts=[]
 for r in rows:
  good=r['contacts']>=1 and r['distance_m']<=.001 and r['force_N']>=20.
  count=count+1 if good else 0;longest=max(longest,count);counts.append(count)
 return {'success':longest>=50,'longest_ticks':longest,'longest_s':longest*.02,'hold_counts':counts}

def evaluate(case,output=None):
 import mujoco
 from .saved_token_replay import verify_source_contract
 case=Path(case);verify_source_contract(case);r=json.loads((case/'result.json').read_text())
 if r['task']!='G1WholebodyTableButtonPressTeleop' or r.get('branch_start_frame',0)!=0:raise ValueError('Profile requires full-start button rollout')
 acceptance=r['native_acceptance'];parameters=acceptance['parameters']
 for key,value in {'min_downward_force_n':20.,'touch_distance_m':.001,'min_contact_points':1,'hold_steps':50}.items():
  if parameters.get(key)!=value:raise ValueError('Native thresholds differ: '+key)
 m=mujoco.MjModel.from_binary_path(str(case/'scene.mjb'));d=mujoco.MjData(m)
 if not np.isclose(m.opt.timestep*4,.02):raise ValueError('Unsupported control cadence')
 trace=np.load(case/'execution/trajectory.npz');buttons={g for g in range(m.ngeom) if any(k in (m.geom(g).name or '') for k in ['E_button_01_13','E_button_02_6'])};added=[];mesh_names={}
 for g in range(m.ngeom):
  mesh=m.mesh(m.geom_dataid[g]).name if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH else '';mesh_names[g]=mesh or ''
  body=m.body(int(m.geom_bodyid[g])).name or ''
  if accepts_contact(body,mesh_names[g],True) and not accepts_contact(body,mesh_names[g],False) and (m.geom_contype[g] or m.geom_conaffinity[g]):added.append({'geom_id':g,'body':body,'mesh':mesh_names[g],'local_position_m':m.geom_pos[g].tolist()})
 if not buttons or len(added)!=1:raise ValueError('Exact reviewed right palm mesh missing/ambiguous')
 native=[];revised=[];palm_force=[]
 for i in range(3,len(trace['next_state']),4):
  st=trace['next_state'][i];d.qpos[:]=st[1:1+m.nq];d.qvel[:]=st[1+m.nq:];d.ctrl[:]=trace['commands'][i,-m.nu:];mujoco.mj_forward(m,d)
  rows=[{'force_N':0.,'distance_m':1e9,'contacts':0} for _ in range(2)];extra=0.
  for ci,c in enumerate(d.contact):
   a,b=int(c.geom1),int(c.geom2);other=b if a in buttons else a if b in buttons else None
   if other is None:continue
   body=m.body(int(m.geom_bodyid[other])).name or '';orig=accepts_contact(body,mesh_names[other],False);allow=accepts_contact(body,mesh_names[other],True)
   if not allow:continue
   f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);world=np.asarray(c.frame).reshape(3,3).T@f[:3];down=max(0.,float(world[2] if a in buttons else -world[2]))
   for j,ok in enumerate([orig,allow]):
    if ok:rows[j]['force_N']+=down;rows[j]['contacts']+=1;rows[j]['distance_m']=min(rows[j]['distance_m'],float(c.dist))
   if allow and not orig:extra+=down
  native.append(rows[0]);revised.append(rows[1]);palm_force.append(extra)
 n=hold_result(native);v=hold_result(revised);old=json.loads((case/'execution/metrics.json').read_text());old_counts=[x['hold_counts']['_button_hold_count'] for x in old]
 if n['hold_counts']!=old_counts or n['success']!=r['native_success']:raise ValueError('Reconstructed native checker disagrees with recorded execution')
 safe=bool(r['coverage']==1 and not r['failure']);result={'profile':PROFILE,'reviewed_at':now(),'case':str(case),'physics_steps':0,'native_success':r['success'],'revised_success':bool(safe and v['success']),'native_hold_s':n['longest_s'],'revised_hold_s':v['longest_s'],'native_counts_exact':True,'final_native_force_N':native[-1]['force_N'],'final_revised_force_N':revised[-1]['force_N'],'final_added_palm_force_N':palm_force[-1],'safe_full_execution':safe,'added_contact_geometry':added,'thresholds':{'downward_force_N':20,'contact_distance_m':.001,'contact_count':1,'consecutive_ticks':50,'tick_s':.02},'native_rule_unchanged':True,'only_rule_change':'add exact right_hand_palm_link collision mesh on right_wrist_yaw_link; exclude wrist shell/other wrist links','source_hashes':{name:file_hash(case/name) for name in ['experiment.json','result.json','scene.mjb','execution/trajectory.npz','execution/metrics.json']},'validator_sha256':file_hash(Path(__file__))}
 if output:save(Path(output),result)
 return result
