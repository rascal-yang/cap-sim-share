"""HMG-inspired stationary-first planning and bounded collision-aware navigation.

CPU IK/RRT and planar A* substitute for the paper's GPU whole-body planner.
All poses are planning copies. Dynamic stability is checked by the executor.
"""
import heapq
import time
import numpy as np
from .stance_selection import yaw_of, yaw_quat, wrap
from .stationary_planning import StationaryPlanner


def choose_plan(rows):
    valid=[r for r in rows if r['feasible']]
    return min(valid,key=lambda r:(r['requires_navigation'],r['score'],r['id'])) if valid else None


def plan_base_path(start,goal,valid,*,resolution=.06,padding=.45,max_expansions=2500):
    start=np.asarray(start,float);goal=np.asarray(goal,float)
    if start.shape!=(3,) or goal.shape!=(3,) or not np.isfinite([*start,*goal]).all() or resolution<=0 or padding<=0 or max_expansions<1:raise ValueError('Invalid path inputs')
    checks=0
    def edge(a,b):
        nonlocal checks
        # Fixed heading translation then checked terminal rotation. Every edge
        # is sampled at <=2cm / <=0.05rad, including diagonal edges.
        delta=b-a;delta[2]=wrap(delta[2]);n=max(1,int(np.ceil(np.linalg.norm(delta[:2])/.02)),int(np.ceil(abs(delta[2])/.05)))
        for u in np.linspace(0,1,n+1):
            checks+=1
            if not valid(a+u*delta):return False
        return True
    approach=goal.copy();approach[2]=start[2]
    if not edge(approach,goal) or not edge(start,start):return None,{'method':'rejected_endpoint','checks':checks}
    if edge(start,approach):
        return np.stack([start,approach,goal]),{'method':'direct','checks':checks,'expansions':0}
    lo=np.minimum(start[:2],goal[:2])-padding;hi=np.maximum(start[:2],goal[:2])+padding
    origin=start[:2];target=tuple(np.rint((goal[:2]-origin)/resolution).astype(int));root=(0,0)
    def pose(k):return np.r_[origin+resolution*np.asarray(k),start[2]]
    queue=[(float(np.linalg.norm(goal[:2]-start[:2])),0.,root)];cost={root:0.};parent={};expanded=0;end=None
    while queue and expanded<max_expansions:
        _,g,k=heapq.heappop(queue)
        if g>cost[k]+1e-10:continue
        expanded+=1;a=pose(k)
        if np.linalg.norm(a[:2]-goal[:2])<=resolution*1.5 and edge(a,approach):end=k;break
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nxt=(k[0]+dx,k[1]+dy);b=pose(nxt)
            if np.any(b[:2]<lo) or np.any(b[:2]>hi):continue
            ng=g+resolution*np.hypot(dx,dy)
            if ng>=cost.get(nxt,float('inf')) or not edge(a,b):continue
            cost[nxt]=ng;parent[nxt]=k;heapq.heappush(queue,(ng+np.linalg.norm(b[:2]-goal[:2]),ng,nxt))
    if end is None:return None,{'method':'astar_exhausted','checks':checks,'expansions':expanded}
    keys=[end]
    while keys[-1]!=root:keys.append(parent[keys[-1]])
    raw=[pose(k) for k in reversed(keys)]+[approach];path=[raw[0]];i=0
    while i<len(raw)-1:
        j=len(raw)-1
        while j>i+1 and not edge(raw[i],raw[j]):j-=1
        path.append(raw[j]);i=j
    path.append(goal)
    return np.asarray(path),{'method':'astar','checks':checks,'expansions':expanded,'resolution_m':resolution,'collision_sample_m':.02}


def path_command(current,path,index,*,speed=.5,waypoint_radius=.035,stop_radius=.10):
    current=np.asarray(current,float);path=np.asarray(path,float)
    if len(path)<2:raise ValueError('Path needs at least two poses')
    # Last pose may differ only in yaw. Heading is a soft target, and the
    # measured-state manipulation check decides feasibility after stopping.
    endpoint=len(path)-1
    while endpoint>1 and np.linalg.norm(path[endpoint,:2]-path[endpoint-1,:2])<1e-8:endpoint-=1
    index=min(max(1,index),endpoint)
    while index<endpoint and np.linalg.norm(current[:2]-path[index,:2])<=waypoint_radius:index+=1
    delta=path[index,:2]-current[:2];dist=float(np.linalg.norm(delta));stop=index==endpoint and dist<=stop_radius
    heading=float(path[0,2] if index<endpoint else path[-1,2])
    return (np.zeros(2) if stop else delta*speed/max(dist,1e-10)),heading,stop,index


def select_navigation(model,initial,source_stance,targets,*,seed=17):
    planner=StationaryPlanner(model);initial=np.asarray(initial,float).copy();source=np.asarray(source_stance,float).copy();start=np.r_[initial[:2],yaw_of(initial[3:7])];obj=initial[50:53];rows=[];begin=time.perf_counter()
    def candidate(name,q):
        xy=q[:2];yaw=yaw_of(q[3:7]);current=name=='current';poses=planner.wrist_poses(q)
        solved=planner.solve(q,{'left':poses['left'],'right':targets['right_pose'][1599]},max_seeds=1,max_nfev=80,seed=seed)
        row={'id':name,'xyyaw':[*xy,yaw],'feasible':False,'requires_navigation':not current,'ik_group':solved.get('group'),'ik_attempts':solved['attempts'],'rejection':None,'score':1e9}
        if not solved['success']:row['rejection']='stationary_ik_failed';return row
        connection,ca=planner.connect(q,solved['qpos'],solved['joint_ids'],seed=seed,max_iterations=300,resolution=.03)
        row['manipulation_path_audit']=ca
        if connection is None:row['rejection']='stationary_connection_failed';return row
        def valid(pose):
            probe=initial.copy();probe[:2]=pose[:2];probe[3:7]=yaw_quat(pose[2]);return not planner.collisions(probe)
        path,pa=plan_base_path(start,row['xyyaw'],valid)
        row['navigation_path_audit']=pa
        if path is None:row['rejection']='base_path_failed';return row
        length=float(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum());facing=abs(wrap(np.arctan2(obj[1]-xy[1],obj[0]-xy[0])-yaw));turn=abs(wrap(yaw-start[2]))
        row.update(feasible=True,entry_qpos=solved['qpos'].tolist(),stance_reference_qpos=q.tolist(),navigation_path=path.tolist(),travel_m=length,turn_rad=turn,facing_error_rad=facing,score=length+.1*turn+.12*facing)
        return row
    # Hierarchical arms -> arms+waist checks at actual current pose come first.
    row=candidate('current',initial);rows.append(row)
    if row['feasible']:return {'selected':row,'candidates':rows,'physics_steps':0,'planning_wall_s':time.perf_counter()-begin,'dynamic_navigation_validated':False,'method':'stationary_first_arms_then_waist; IK/RRT; planar_Astar','original_sequences_accessible':True}
    nominal=source[:2]+obj[:2]-targets['source_object_anchor'][:2];sourceyaw=yaw_of(source[3:7])
    for ix,dx in enumerate([0.,-.06]):
        for iy,dy in enumerate([0.,-.08,.08]):
            xy=nominal+np.array([dx,dy]);face=np.arctan2(obj[1]-xy[1],obj[0]-xy[0])
            for iz,yaw in enumerate([sourceyaw,sourceyaw+.5*wrap(face-sourceyaw),face]):
                # Switch configuration keeps the current upper-body pose.
                q=initial.copy();q[:2]=xy;q[3:7]=yaw_quat(yaw)
                rows.append(candidate(f'candidate_{ix}_{iy}_{iz}',q))
    return {'selected':choose_plan(rows),'candidates':rows,'physics_steps':0,'planning_wall_s':time.perf_counter()-begin,'dynamic_navigation_validated':False,'method':'stationary_first_arms_then_waist; IK/RRT; planar_Astar','original_sequences_accessible':True}
