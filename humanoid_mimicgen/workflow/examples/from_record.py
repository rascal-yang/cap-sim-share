"""Explicit historical evidence migration, not defaults for arbitrary new tasks.

Outputs a portable request beside copied manifests; supplied record remains readonly.
Calibration is intentionally disabled, never copied from an empirical old controller.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import numpy as np


def build(record,out,*,hands,object_body,support_geom,speed,stop_radius,pose_offset):
    from humanoid_mimicgen.stationary_planning import StationaryPlanner
    from humanoid_mimicgen.reachability_navigation import plan_base_path
    record=Path(record).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    c=json.loads(record.read_text());inp=c['inputs'];g=c['physics']['safety_gates'];old=c['physics']['acceptance_thresholds'];standing=old['standing_gate']
    boundaries=inp['boundaries']
    if set(boundaries)!={'grasp','lift','hold'}:raise ValueError('This migration tool expects historical grasp/lift/hold labels; generic workflow accepts explicit phase metadata')
    keys=sorted(boundaries,key=boundaries.get);n=inp['steps'];phases=[]
    for i,key in enumerate(keys):phases.append({'id':key,'start':boundaries[key]+1,'end':boundaries[keys[i+1]]+1 if i+1<len(keys) else n,
                                             'requires_contact':key!='grasp','requires_clearance':key=='hold'})
    targets=dict(np.load(inp['targets']));states=np.load(inp['reference_states'])['samples'];dt=c['physics']['physics_dt_s']
    # Source movement sets approach speed scales; never insert a case-specific displacement.
    from cap_sim.backends.native_sonic import rot
    interval=slice(phases[0]['start'],phases[-1]['end']);maxlinear=max(float(np.max(np.linalg.norm(np.diff(targets[h+'_pose'][interval,:3],axis=0),axis=1)/dt)) for h in hands)
    maxangular=max(float(np.max((rot(targets[h+'_pose'][interval][1:,3:])*rot(targets[h+'_pose'][interval][:-1,3:]).inv()).magnitude()/dt)) for h in hands)
    # Algorithmic solver defaults are recorded, not hidden in task-specific code.
    def defaults(func,names):return {k:inspect.signature(func).parameters[k].default for k in names}
    ik=defaults(StationaryPlanner.solve,['max_seeds','max_nfev','seed','position_tolerance','angle_tolerance'])
    rrt=defaults(StationaryPlanner.connect,['seed','max_iterations','resolution'])
    bp=defaults(plan_base_path,['resolution','padding','max_expansions'])
    import mujoco
    m=mujoco.MjModel.from_binary_path(inp['model']);joint_ids=[j for j in range(m.njnt) if any(x in m.joint(j).name for x in ('shoulder','elbow','wrist','waist'))]
    adr=m.jnt_qposadr[joint_ids];maxjoint=float(np.max(abs(np.diff(states[interval,1+adr],axis=0)))/dt)
    policy={'control_dt_s':c['physics']['control_dt_s'],
        'navigation':{'speed_m_s':speed,'waypoint_radius_m':ik['position_tolerance'],'stop_radius_m':stop_radius},
        'settle':{'window_s':standing['continuous_s'],'drift_m':standing['position_xy_m'],'speed_m_s':standing['speed_xy_m_s'],
                  'yaw_speed_rad_s':standing['root_local_z_angular_velocity_rad_s'],'foot_force_N':standing['both_foot_ground_min_N'],
                  'minimum_s':standing['minimum_dwell_s'],'continuous_s':standing['continuous_s']},
        'approach':{'joint_speed_rad_s':maxjoint,'wrist_speed_m_s':maxlinear,'wrist_speed_rad_s':maxangular},
        'planning':{'penetration_tolerance_m':inspect.signature(StationaryPlanner).parameters['penetration_tolerance_m'].default,'ik':ik,'rrt':rrt,'base_path':bp},
        'gates':{'contact_force_N':old['hand_contact_N'],'contact_entry_s':standing['continuous_s'],
                 'contact_loss_s':g['lift_hold_contact_loss_s'],'collision_force_N':g['robot_table_force_N'],
                 'collision_duration_s':g['robot_table_force_duration_s'],'collision_instant_N':g['instant_robot_table_N'],
                 'fall_height_m':g['fall_root_z_m'],'fall_tilt_rad':float(np.deg2rad(g['fall_tilt_deg'])),
                 'precontact_translation_m':g['object_precontact_translation_m'],'precontact_rotation_rad':g['object_precontact_rotation_rad']},
        'acceptance':{'clearance_m':g['hold_entry_clearance_m'],'hold_duration_s':old['independent_hold_s'],
                      'root_drift_m':standing['position_xy_m'],'support_force_N':old['support_force_max_N']},
        'budgets':{'navigation_s':g['navigation_timeout_s'],'settle_s':standing['timeout_s'],
                   'approach_s':c['budget']['simulation_seconds_per_attempt'],'simulation_s':c['budget']['simulation_seconds_per_attempt']},
        'compensation':{'mode':'none'},'origins':{}}
    for name in policy:
        if name=='origins':continue
        kind='numerical_setting' if name=='planning' else 'explicit_disabled' if name=='compensation' else 'task_requirement'
        evidence='Explicit CLI speed/stop-radius; no fixed stance offset' if name=='navigation' else 'Source contact measured speed extrema' if name=='approach' else 'Historical evidence migration; review supplied record for applicability'
        policy['origins'][name]={'kind':kind,'evidence':evidence}
    alignment={'finger_source':'executed_commands','command_dt_s':dt,'pose_alignment':'explicit_offset','origin_record':os.path.relpath(record,out)}
    (out/'alignment.json').write_text(json.dumps(alignment,indent=2))
    bindings={'hands':hands,'object':object_body,'root_body':'pelvis','wrists':{'left':'left_wrist_yaw_link','right':'right_wrist_yaw_link'},
              'torso':'torso_link','feet':{'left':'left_ankle_roll_link','right':'right_ankle_roll_link'},'ground_geoms':['ground'],'support_geoms':[support_geom]}
    s={'schema_version':1,'parent_id':None,'kind':'trajectory_retargeting','adapter':'g1_sonic_native_v1','context':{},'bindings':bindings,
       'phases':phases,'command_pose_offset':pose_offset,'source_frame_count':n,'policy':policy,'proposal':{},'evidence':{'source_alignment':'alignment.json'},
       'provenance':{'original_sequences_accessible':True,'settings_origin':'explicit historical evidence migration; not universal defaults'}}
    files={k:os.path.relpath(Path(inp[k]).resolve(),out) for k in ('model','initial','targets','reference_states','task')};files.update(record=os.path.relpath(record,out),alignment='alignment.json')
    request={'strategy':s,'files':files,'strategy_root':'strategies','experiment_root':os.path.relpath(record.parents[1],out),'decision':{'mode':'deterministic'}}
    (out/'request.json').write_text(json.dumps(request,indent=2));return out/'request.json'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--out',required=True);p.add_argument('--active-hand',choices=['left','right'],action='append',required=True)
    p.add_argument('--object',required=True);p.add_argument('--support',required=True);p.add_argument('--speed',type=float,required=True);p.add_argument('--stop-radius',type=float,required=True);p.add_argument('--pose-offset',type=int,choices=[-1,0],required=True)
    a=p.parse_args();print(build(a.record,a.out,hands=a.active_hand,object_body=a.object,support_geom=a.support,speed=a.speed,stop_radius=a.stop_radius,pose_offset=a.pose_offset))
