"""Run zero-physics HMG component checks on the actual apple scene.

This is a planning report, never a generated physical training episode. No
backend reset, SONIC inference, or mj_step is used. Scene variants are private
kinematic model copies, not changes to any historical scene.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import mujoco
from .skill_planning import skill_batches,adapt_poses,rollout_action
from .stationary_planning import StationaryPlanner
from .privileged import no_physics


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def dump(path,obj):
    def conv(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,default=conv,allow_nan=False)


def run(workspace,out):
    workspace=Path(workspace).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    start=now();tick=time.perf_counter();source=workspace/'experiments/20261013_apple_straight_sweep/reference_recovery/attempts/001'
    targetfile=workspace/'experiments/20261016_1619_apple_contact_controller/targets.npz'
    casesfile=Path(__file__).with_name('ten_conditions.json')
    cases=[{'id':'origin','translation':[0,0,0],'table_delta':0},{'id':'y_plus_2cm','translation':[0,.02,0],'table_delta':0}]+json.loads(casesfile.read_text())
    sources=[source/'scene.mjb',source/'states.npz',targetfile,casesfile,Path(__file__),Path(__file__).with_name('skill_planning.py'),Path(__file__).with_name('stationary_planning.py')]
    dump(out/'planning_record.json',{'experiment_id':out.name,'started_at':start,'kind':'offline_kinematic_planning_only','physics_steps':0,'source_hashes':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'runtime_interface':'MuJoCo mj_forward + SciPy least_squares; no SONIC sessions or actuation','position_tolerance_m':.003,'angle_tolerance_rad':.03,'collision_penetration_tolerance_m':1e-5,'joint_edge_resolution_rad':.04,'max_seeds_per_group':3,'max_ik_nfev_per_seed':100,'max_rrt_iterations':600,'seed':17,'case_count':len(cases),'robot_state':'source measured state at t=8s; original prefix is not regenerated','original_sequences_accessible':True,'method':'trajectory_retargeting; stationary CPU planning subset','new_physics_experiment':False})
    data=np.load(source/'states.npz')['samples'];targets=np.load(targetfile);results=[]
    with no_physics():
        for case in cases:
            ct=time.perf_counter();m=mujoco.MjModel.from_binary_path(str(source/'scene.mjb'));q=data[1599,1:1+m.nq].copy();delta=np.asarray(case['translation']);objadr=int(m.jnt_qposadr[m.joint('apple_001_joint').id]);anchor=q[objadr:objadr+7].copy();newanchor=anchor.copy();newanchor[:3]+=delta;q[objadr:objadr+7]=newanchor
            tablebody=int(m.geom_bodyid[m.geom('table_geom').id]);m.body_pos[tablebody,2]+=case['table_delta']
            p=StationaryPlanner(m);goals={'left':targets['left_pose'][1599].copy(),'right':adapt_poses(targets['right_pose'][1599],anchor,newanchor)}
            solve=p.solve(q,goals,seed=17);path=None;connection={'reason':'no_ik_solution'}
            if solve['success']:path,connection=p.connect(q,solve['qpos'],solve['joint_ids'],seed=17)
            cdir=out/case['id'];cdir.mkdir()
            adapted=adapt_poses(targets['right_pose'][1599:2680],anchor,newanchor)
            np.savez_compressed(cdir/'reference.npz',right_pose=adapted,left_pose=targets['left_pose'][1599:2680],fingers=targets['fingers'][1599:2680],source_frame=np.arange(1599,2680))
            if path is not None:np.savez_compressed(cdir/'stationary_path.npz',qpos=path)
            entry={'case':case,'ik':solve,'connection':connection,'path_samples':0 if path is None else len(path),'planning_success':path is not None,'planning_wall_s':time.perf_counter()-ct,'physics_steps':0,'physical_success':None,'training_ready':False,'object_states':'fixed initial pose for approach collision checking; contact reference is not executed','held_object_collision_validated':False,'full_contact_path_ik_validated':False}
            dump(cdir/'planning.json',entry);results.append(entry)
            print(case['id'],solve['success'],solve.get('group'),connection,flush=True)
    # A continuous grasp/lift/hold contact skill keeps one anchor; semantic phases
    # are retained without making independent re-anchored skills.
    schedule={'apple_contact_group':skill_batches({'grasp_lift_hold':'right'}),'apple_semantic_closed_ranges':{'grasp':[1599,1879],'lift':[1879,2433],'hold':[2433,2654]},'bimanual_synthetic_test':skill_batches({'pick_l':'left','pick_r':'right','place_l':'left','place_r':'right'},[('pick_l','place_l'),('pick_r','place_r')],[('pick_l','pick_r'),('place_l','place_r')]),'bimanual_physical_execution':False}
    dump(out/'skill_schedule.json',schedule)
    noise=rollout_action(np.zeros(29),.005,seed=17);np.savez_compressed(out/'noise_semantics_only.npz',**noise)
    summary={'experiment_id':out.name,'started_at':start,'ended_at':now(),'planning_wall_s':time.perf_counter()-tick,'simulation_wall_s':None,'rendering_wall_s':None,'physics_steps':0,'case_count':len(results),'stationary_entry_plans':sum(r['planning_success'] for r in results),'results':[{'case':r['case']['id'],'planning_success':r['planning_success'],'group':r['ik'].get('group'),'connection':r['connection'],'planning_wall_s':r['planning_wall_s']} for r in results],'training_ready':False,'not_reproduced':['Homie hybrid control','cuRobo full-body batch IK','locomotion and footstep execution','attached-object collision checking','dense contact joint-path validation','noisy physical rollouts','policy training/evaluation'],'paper':'https://arxiv.org/html/2605.27724v1'}
    dump(out/'summary.json',summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',required=True);parser.add_argument('--out',required=True);args=parser.parse_args();print(json.dumps(run(args.workspace,args.out),indent=2))

if __name__=='__main__':main()
