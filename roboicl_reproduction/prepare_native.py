"""Prepare TRAIN observation memory from saved source states; zero physics steps."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import mujoco
from repro.bridge import scene

ROOT=Path(__file__).resolve().parents[2]
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',required=True,type=Path);a=parser.parse_args();out=a.experiment.resolve()
    source=ROOT/'experiments/20261013_apple_straight_sweep/reference_recovery/success/attempt_001'
    validation=json.loads((source/'independent_validation.json').read_text())
    if validation.get('success') is not True:raise ValueError('Source not independently accepted')
    trajectory=json.loads((source/'trajectory.json').read_text());task=trajectory['task']
    model=mujoco.MjModel.from_binary_path(str(source/'scene.mjb'));data=mujoco.MjData(model)
    states=np.load(source/'states.npz')['samples'];initial=np.load(task['resources']['initial_state'])
    rows=np.vstack([np.r_[0,initial['qpos'],initial['qvel'],initial['ctrl']],states])
    contacts=json.loads((source/'contacts.json').read_text())
    if len(contacts)!=len(states):raise ValueError('Source contacts/state lengths differ')
    indices=np.unique(np.linspace(0,len(rows)-1,13).astype(int));selected_images=set(indices[[0,6,10,12]])
    observations=[]
    from humano.robots.hands import DEX3_HAND
    hand_ids={side:[i for i in range(model.njnt) if model.joint(i).name in getattr(DEX3_HAND,side+'_actuated_joints')] for side in ('left','right')}
    if any(len(ids)!=7 for ids in hand_ids.values()):raise ValueError('Source Dex3 actuated order mismatch')
    hand_qids={side:model.jnt_qposadr[ids] for side,ids in hand_ids.items()}
    for index in indices:
        row=rows[index];data.time=row[0];data.qpos[:]=row[1:1+model.nq];data.qvel[:]=row[1+model.nq:1+model.nq+model.nv];data.ctrl[:]=row[1+model.nq+model.nv:]
        mujoco.mj_forward(model,data)
        ob={'source_frame':int(index),'time':float(data.time),'root_pose':data.qpos[:7].tolist(),'root_velocity':data.qvel[:6].tolist(),
            'measured_fingers':{side:data.qpos[ids].tolist() for side,ids in hand_qids.items()},
            'fingers_semantics':'measured seven joint angles; NOT previous command',
            'contacts':{k:v for k,v in (contacts[index-1] if index else {}).items() if k!='events'}}
        for side in ('left','right'):
            b=model.body(side+'_wrist_yaw_link').id;ob[side+'_position']=data.xpos[b].tolist();ob[side+'_quaternion_wxyz']=data.xquat[b].tolist()
        b=model.body('torso_link').id;ob['torso_quaternion_wxyz']=data.xquat[b].tolist()
        if index in selected_images:ob['images']=scene.render_views(model,data,out/'train_images',f'frame_{index:05d}')
        observations.append(ob)
    demo={'kind':'TRAIN_observation_memory_cross_control_v2','source_control':'native motion prefix then mode0 body targets',
          'live_control':'fresh world-delta wrists + Dex3 joint targets, SONIC mode1; no source replay',
          'source_sha256':{f:sha(source/f) for f in ('states.npz','contacts.json','scene.mjb','trajectory.json','independent_validation.json')},
          'source_accepted':validation,'observations':observations,
          'interpretation':'Selected true measured source observations, with explicit temporal gaps. Learn stance, approach, orientation, finger configuration and contact conditions. These are NOT action-command labels and not original same-interface one-shot RoboICL.',
          'physics_steps':0}
    write(out/'train_memory.json',demo)
    write(out/'source_evidence.json',{'source':str(source),'physics':scene.physics_signature(model),'source_sha256':demo['source_sha256'],'validation':validation,'physics_steps':0,
        'selected_frames':indices.tolist(),'finger_order_qpos_indices':{side:ids.tolist() for side,ids in hand_qids.items()}})
    task=json.loads((ROOT/'experiments/20261014_1957_roboicl_lateral/tasks/y_minus_10cm.json').read_text())
    task['scene']={'translation':[0,0,0]};task['budget'].update(attempts=3,feedback_calls=80)
    task['generation_kind']='roboicl_train_observation_live_closed_loop';task['code_version']='roboicl_reproduction/native_run.py'
    write(out/'task.json',task)
    print(json.dumps({'experiment':str(out),'selected_frames':len(indices),'physics_steps':0}))
if __name__=='__main__':main()
