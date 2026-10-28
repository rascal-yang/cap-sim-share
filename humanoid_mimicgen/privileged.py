"""Extract contact phases from saved states. Never runs an action or integrates physics.

Raw HUMANO contacts are geometric evidence, not measured forces. Clips retain
source indices, clock, actions and measured states; they are not fresh rollouts.
"""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def runs(mask):
    changes=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(changes==1), np.flatnonzero(changes==-1)))


def segment_phases(e):
    """Conservative ordered phases; reject or retain partial evidence, never pad hold."""
    t=np.asarray(e['time_s'],float)
    if len(t)<3 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Need finite strictly increasing sample times')
    keys=('object_pose','clearance_m','left_contact','right_contact','support_contact',
          'relative_position','relative_quaternion')
    for k in keys:
        if len(e[k])!=len(t) or not np.isfinite(e[k]).all():raise ValueError('Invalid evidence: '+k)
    dt=float(np.median(np.diff(t)))
    if np.max(np.diff(t))>dt*1.5:raise ValueError('Missing frames: resampling is not implicit')
    hand=max(('left','right'),key=lambda s:np.sum(e[s+'_contact']))
    c=np.asarray(e[hand+'_contact'],bool);support=np.asarray(e['support_contact'],bool)
    z=np.asarray(e['object_pose'])[:,2];rise=z-np.median(z[:min(10,len(z))])
    clearance=np.asarray(e['clearance_m'])
    result={'status':'rejected','active_hand':hand,'segments':[], 'missing_stages':['grasp','lift','hold'],
            'thresholds':{'contact_min_s':.12,'lift_rise_m':.01,'lift_clearance_m':.002,
                          'hold_clearance_m':.03,'hold_min_s':.5,'hold_vertical_speed_m_s':.04,
                          'hold_position_range_m':.02,'hold_rotation_range_rad':.25},
            'reason':'No sustained hand contact'}
    contacts=[(a,b) for a,b in runs(c) if t[b-1]-t[a]+dt>=.12-1e-8]
    if not contacts:return result
    # Select a contact bout which contains lift; earlier accidental taps stay outside.
    liftmask=c & ~support & (rise>=.01) & (clearance>=.002)
    liftbouts=[(a,b) for a,b in runs(liftmask) if t[b-1]-t[a]+dt>=.08-1e-8]
    bout=next(((a,b,l) for a,b in contacts for l,h in liftbouts if a<l<b),None)
    if bout is None:
        result['reason']='No sustained supported-to-airborne transition';return result
    a,b,l=bout
    # The threshold confirms lift; backtrack its onset to last supported sample.
    prior=np.flatnonzero(support[a:l+1]);l=max(a+1,a+int(prior[-1])+1) if len(prior) else l
    vel=np.gradient(z,t)
    stable=c & ~support & (clearance>=.03) & (np.abs(vel)<=.04)
    rel=np.asarray(e['relative_position']);quat=np.asarray(e['relative_quaternion'])
    if np.any(np.abs(np.linalg.norm(quat,axis=1)-1)>1e-5):raise ValueError('Nonunit relative quaternion')
    hold=None
    for h in range(l+1,b):
        end=int(np.searchsorted(t,t[h]+.5-1e-8,side='left'))+1
        if end>b or not np.all(stable[h:end]):continue
        if np.max(np.linalg.norm(rel[h:end]-rel[h],axis=1))>.02:continue
        dots=np.clip(np.abs(quat[h:end]@quat[h]),0,1)
        if np.max(2*np.arccos(dots))>.25:continue
        k=end
        while k<b and stable[k] and np.linalg.norm(rel[k]-rel[h])<=.02 and 2*np.arccos(np.clip(abs(quat[k]@quat[h]),0,1))<=.25:k+=1
        hold=(h,k);break
    spans=[('grasp',a,l),('lift',l,hold[0] if hold else b)]
    if hold:spans.append(('hold',*hold))
    for name,start,end in spans:
        if end<=start:continue
        result['segments'].append({'name':name,'start_frame':int(start),'end_frame':int(end),
            'start_s':float(t[start]),'end_exclusive_s':float(t[end]) if end<len(t) else float(t[-1]+dt),
            'duration_s':float(t[end-1]-t[start]+dt),'contact_fraction':float(np.mean(c[start:end])),
            'min_clearance_m':float(np.min(clearance[start:end])), 'max_clearance_m':float(np.max(clearance[start:end])),
            'max_abs_vertical_speed_m_s':float(np.max(np.abs(vel[start:end])))})
    result.update(status='extracted' if hold else 'partial',missing_stages=[] if hold else ['hold'],
                  reason='All three phases supported by saved states' if hold else 'No >=0.5 s stable elevated hold in recorded data')
    return result


@contextmanager
def no_physics():
    """Forward-only initialization, with integration tripwires including split steps."""
    import mujoco
    from humano.engines.mujoco import MujocoSimulator
    saved={k:getattr(mujoco,k) for k in ('mj_step','mj_step1','mj_step2')}
    old=MujocoSimulator.step
    def forbidden(*a,**kw):raise RuntimeError('Integration forbidden in saved-state extraction')
    def initialize(self,*a,**kw):
        mujoco.mj_forward(self.mjModel,self.mjData)
    try:
        for k in saved:setattr(mujoco,k,forbidden)
        MujocoSimulator.step=initialize
        yield
    finally:
        MujocoSimulator.step=old
        for k,v in saved.items():setattr(mujoco,k,v)


def extract(model, qpos, qvel, times, actions, out, provenance, measured_contacts=None):
    import mujoco
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    data=mujoco.MjData(model);n=len(times)
    if qpos.shape!=(n,model.nq) or qvel.shape!=(n,model.nv):raise ValueError('Source model/state mismatch')
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():raise ValueError('Invalid source state')
    obj=model.body('apple_001').id
    def beneath(body,root):
        while body and body!=root:body=int(model.body_parentid[body])
        return body==root
    obj_geoms=[g for g in range(model.ngeom) if beneath(int(model.geom_bodyid[g]),obj) and (model.geom_contype[g] or model.geom_conaffinity[g])]
    table=model.geom('table_geom').id
    if model.geom_type[table]!=mujoco.mjtGeom.mjGEOM_BOX:raise ValueError('Only box tabletop supported')
    poses={key:np.empty((n,7)) for key in ('object_pose','left_pose','right_pose')}
    e={'time_s':np.asarray(times),'clearance_m':np.empty(n), 'left_contact':np.zeros(n,bool),
       'right_contact':np.zeros(n,bool),'support_contact':np.zeros(n,bool),'robot_table_contact':np.zeros(n,bool)}
    events=[]
    for i in range(n):
        data.qpos[:]=qpos[i];data.qvel[:]=qvel[i];data.time=float(times[i])
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_collision(model,data)
        if not np.allclose(data.geom_xmat[table].reshape(3,3),np.eye(3),atol=1e-6):raise ValueError('Table must be horizontal')
        top=data.geom_xpos[table,2]+model.geom_size[table,2];bottom=np.inf
        for g in obj_geoms:
            if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Expected mesh apple collider')
            mid=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mid]);num=int(model.mesh_vertnum[mid])
            vertices=model.mesh_vert[start:start+num]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
            bottom=min(bottom,float(vertices[:,2].min()))
        e['clearance_m'][i]=bottom-top
        for key,body in [('object_pose',obj),('left_pose',model.body('left_wrist_yaw_link').id),('right_pose',model.body('right_wrist_yaw_link').id)]:
            poses[key][i]=np.r_[data.xpos[body],data.xquat[body]]
        rows=[]
        for c in data.contact:
            gs=[int(c.geom1),int(c.geom2)];bs=[int(model.geom_bodyid[g]) for g in gs];names=[model.body(b).name for b in bs]
            if float(c.dist)>0:continue
            isobj=any(g in obj_geoms for g in gs)
            if isobj:
                for hand in ('left','right'):
                    if any(name.startswith(hand+'_hand') for name in names):e[hand+'_contact'][i]=True
                if table in gs:e['support_contact'][i]=True
                rows.append({'bodies':names,'distance_m':float(c.dist)})
            elif table in gs and any('link' in name for name in names):e['robot_table_contact'][i]=True
        events.append(rows)
    e.update(poses)
    hand=max(('left','right'),key=lambda s:np.sum(e[s+'_contact']))
    hp=poses[hand+'_pose'];op=poses['object_pose']
    hr=Rotation.from_quat(hp[:,[4,5,6,3]]);orr=Rotation.from_quat(op[:,[4,5,6,3]])
    e['relative_position']=hr.inv().apply(op[:,:3]-hp[:,:3])
    e['relative_quaternion']=(hr.inv()*orr).as_quat()[:,[3,0,1,2]]
    result=segment_phases(e)
    arrays={**e,'qpos':qpos,'qvel':qvel,'source_frame_index':np.arange(n),**actions}
    if measured_contacts is not None:
        for key in ('hand_object_N','support_N','robot_table_N'):
            arrays['measured_'+key]=np.asarray([r[key] for r in measured_contacts])
    np.savez_compressed(out/'evidence.npz',**arrays)
    mujoco.mj_saveModel(model,str(out/'scene.mjb'),None)
    write(out/'contact_geometry.json',events)
    for seg in result['segments']:
        a,b=seg['start_frame'],seg['end_frame']
        np.savez_compressed(out/(seg['name']+'.npz'),**{k:v[a:b] for k,v in arrays.items()})
    result.update(provenance=provenance,physics_steps=0,training_ready=False,
                  evidence_sha256=sha(out/'evidence.npz'),scene_sha256=sha(out/'scene.mjb'),
                  contact_evidence='saved-state collision geometry; no force closure claim',
                  measured_contact_forces_available=measured_contacts is not None,
                  robot_table_contact_frames=int(e['robot_table_contact'].sum()),
                  source_frame_count=n,source_time_range_s=[float(times[0]),float(times[-1])])
    write(out/'segments.json',result)
    return result


def raw_episode(dataset, episode, out):
    import pandas as pd
    from cap_sim.backends.humano import Backend
    root=Path(dataset).resolve();stem=f'episode_{episode:06d}'
    matches=list(root.glob('data/**/'+stem+'.parquet'))
    if len(matches)!=1:raise ValueError('Need one source parquet')
    parquet=matches[0];replay=root/'sonic'/(stem+'_replay.json');initial=root/'sonic'/(stem+'_start.npz')
    prepared=Path(out).parent/(stem+'_prepared');prepared.mkdir(exist_ok=False)
    config=json.loads(json.loads(replay.read_text())['environment_config']);write(prepared/'environment_config.json',config)
    task={'name':'G1WholebodyTableApplePickTeleop','resources':{'prepared':str(prepared),'initial_state':str(initial)},'scene':{}}
    df=pd.read_parquet(parquet);backend=Backend()
    try:
        with no_physics():
            backend.describe(task);m=backend.m
            if not np.all(df['sim.nq']==m.nq) or not np.all(df['sim.nv']==m.nv):raise ValueError('Dimension mismatch')
            q=np.stack(df['sim.qpos'])[:,:m.nq];v=np.stack(df['sim.qvel'])[:,:m.nv]
            actions={key.replace('.','_'):np.stack(df[key]) for key in ('action.wbc','action.motion_token','teleop.left_hand_joints','teleop.right_hand_joints')}
            return extract(m,q,v,df['sim.time'].to_numpy(),actions,out,{
                'source_type':'original_HUMANO_dataset','files_sha256':{str(p):sha(p) for p in (parquet,replay,initial)},
                'clock':'original sim.time, not video timestamp; source_frame_index maps directly to parquet row',
                'actions':'original columns copied; no inferred state/action causal offset',
                'model_initialization':'MujocoSimulator.step explicitly replaced with mj_forward; integration blocked',
                'sonic_inference_executed':False})
    finally:backend.close()


def native_attempt(source,out):
    import mujoco
    from .dataset import import_native
    source=Path(source).resolve()
    # Existing importer verifies the matching successful export, hashes and timestamps.
    checked=Path(out).parent/'native_verified.npz';import_native(source,checked)
    m=mujoco.MjModel.from_binary_path(str(source/'scene.mjb'))
    s=np.load(source/'states.npz',allow_pickle=False)['samples']
    contacts=json.loads((source/'contacts.json').read_text())
    with no_physics():
        return extract(m,s[:,1:1+m.nq],s[:,1+m.nq:1+m.nq+m.nv],s[:,0],
                       {'recorded_ctrl':s[:,1+m.nq+m.nv:]},out,{
                           'source_type':'historical_cap_sim_success_not_original_dataset',
                           'files_sha256':{str(source/p):sha(source/p) for p in ('scene.mjb','states.npz','contacts.json','controller.json')},
                           'actions':'recorded actuator ctrl; original motion tokens not reconstructed',
                           'clock':'original sim.time, source_frame_index maps to saved state row'},contacts)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--episodes',nargs='+',type=int,default=[0,1,2]);p.add_argument('--native');p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=now();tick=time.perf_counter();results=[]
    for ep in args.episodes:
        r=raw_episode(args.dataset,ep,out/f'episode_{ep:06d}');results.append({'id':f'episode_{ep:06d}',**r});print(ep,r['status'],r['missing_stages'],flush=True)
    if args.native:
        r=native_attempt(args.native,out/'historical_native');results.append({'id':'historical_native',**r});print('native',r['status'],r['missing_stages'],flush=True)
    write(out/'summary.json',{'started_at':start,'ended_at':now(),'state_replay_wall_s':time.perf_counter()-tick,'physics_steps':0,'simulation_s':0,'results':results})


if __name__=='__main__':main()
