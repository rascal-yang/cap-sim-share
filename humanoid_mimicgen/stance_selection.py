"""Privileged, deterministic reachability-first stance selection for G1.

IK/collision feasibility are hard filters. Facing is only a cost among feasible
candidates. Kinematic navigation screening does not prove dynamic feasibility.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .stationary_planning import StationaryPlanner


def wrap(a):return float(np.arctan2(np.sin(a),np.cos(a)))
def yaw_of(q):return float(Rotation.from_quat(np.roll(q,-1)).as_euler('xyz')[2])
def yaw_quat(y):return np.roll(Rotation.from_euler('z',y).as_quat(),1)


def rank_feasible(rows):
    valid=[]
    for row in rows:
        if row['feasible']:
            item=dict(row);item['score']=item['travel_m']+.10*item['turn_rad']+.12*item['facing_error_rad'];valid.append(item)
    return sorted(valid,key=lambda r:(r['score'],r['id']))


def navigation_command(current_xyyaw,goal_xyyaw,max_speed=.15):
    current=np.asarray(current_xyyaw,float);goal=np.asarray(goal_xyyaw,float)
    if current.shape!=(3,) or goal.shape!=(3,) or not np.isfinite(np.r_[current,goal]).all():raise ValueError('Invalid base poses')
    delta=goal[:2]-current[:2];distance=float(np.linalg.norm(delta));turn=abs(wrap(goal[2]-current[2]));reached=distance<=.025 and turn<=.12
    velocity=np.zeros(2) if reached else delta*min(.8,max_speed/max(distance,1e-10))
    return velocity,float(goal[2]),reached


def select_stance(model,initial,source_stance,targets,*,seed=17):
    p=StationaryPlanner(model);initial=np.asarray(initial).copy();source=np.asarray(source_stance).copy()
    obj=initial[50:53];startyaw=yaw_of(initial[3:7]);sourceyaw=yaw_of(source[3:7]);rows=[]
    # Object-relative nominal stance, then nearby offsets. Initial stance also
    # checked first: if stationary manipulation works, navigation is omitted.
    current=initial.copy();source[50:57]=initial[50:57]
    source_object=targets['source_object_anchor'];delta=obj-source_object[:3]
    nominal=source[:2]+delta[:2]
    candidates=[('current',initial[:2].copy(),startyaw,initial.copy())]
    for ix,dx in enumerate([0.,-.06]):
        for iy,dy in enumerate([0.,-.08,.08]):
            xy=nominal+np.array([dx,dy]);face=np.arctan2(obj[1]-xy[1],obj[0]-xy[0])
            for iz,yaw in enumerate([sourceyaw,sourceyaw+.5*wrap(face-sourceyaw),face]):
                q=source.copy();q[:2]=xy;q[3:7]=yaw_quat(yaw);candidates.append((f'candidate_{ix}_{iy}_{iz}',xy,yaw,q))
    for name,xy,yaw,q in candidates:
        # Inactive left hand keeps its body-relative source pose; right target is
        # fixed by object-relative contact adaptation, not rotated with the base.
        poses=p.wrist_poses(q);goals={'left':poses['left'],'right':targets['right_pose'][1599]}
        solved=p.solve(q,goals,max_seeds=1,max_nfev=60,seed=seed)
        row={'id':name,'xyyaw':[*xy,float(yaw)],'feasible':False,'travel_m':float(np.linalg.norm(xy-initial[:2])),'turn_rad':abs(wrap(yaw-startyaw)),'facing_error_rad':abs(wrap(np.arctan2(obj[1]-xy[1],obj[0]-xy[0])-yaw)),'ik_group':solved.get('group'),'ik_attempts':solved['attempts'],'rejection':None}
        if not solved['success'] or solved['group']!='arms':row['rejection']='no_arms_only_ik'
        else:
            # Straight base corridor, torso/arms initially held in body frame.
            blocked=[]
            for u in np.linspace(0,1,31):
                corridor=initial.copy();corridor[:2]=initial[:2]+u*(xy-initial[:2]);corridor[3:7]=yaw_quat(startyaw+u*wrap(yaw-startyaw))
                contacts=p.collisions(corridor)
                if contacts:blocked=contacts;break
            if blocked:row.update(rejection='static_base_corridor_collision',corridor_collisions=blocked)
            else:row.update(feasible=True,entry_qpos=solved['qpos'].tolist(),stance_reference_qpos=q.tolist())
        rows.append(row)
        if name=='current' and row['feasible']:break
    ranked=rank_feasible(rows)
    return {'selected':ranked[0] if ranked else None,'candidates':rows,'seed':seed,'score':'travel_m +0.10*turn_rad +0.12*facing_error_rad among feasible only','physics_steps':0,'dynamic_navigation_validated':False,'original_sequences_accessible':True}
