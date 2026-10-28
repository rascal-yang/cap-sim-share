"""Paper-derived HMG planning primitives, independent of controller/physics.

Skill ordering follows Sec. 4.2; pose adaptation follows Sec. 3; noisy execution
with clean labels follows Sec. 4.4. CPU joint-space RRT is an explicit substitute
for the paper's cuRobo implementation, not an official implementation.
"""
import numpy as np
from .retarget import _rotation_matrix, _quat_multiply


def skill_batches(skills, precedence=(), concurrent=()):
    """Return greedy batches after collapsing simultaneous-start components.

    A skill maps to left/right/both. Ambiguous same-arm ordering and cycles are
    rejected, never silently serialized. Batches are completion barriers.
    """
    skills=dict(skills)
    if not skills or any(v not in ('left','right','both') for v in skills.values()):
        raise ValueError('Need named skills and valid hands')
    parent={s:s for s in skills}
    def root(s):
        while parent[s]!=s:s=parent[s]
        return s
    def pair(edge):
        if len(edge)!=2 or any(s not in skills for s in edge) or edge[0]==edge[1]:
            raise ValueError('Invalid constraint')
        return edge
    for edge in concurrent:
        a,b=pair(edge);parent[root(b)]=root(a)
    groups={}
    for s in skills:groups.setdefault(root(s),[]).append(s)
    pred={g:set() for g in groups}
    for edge in precedence:
        a,b=pair(edge);a,b=root(a),root(b)
        if a==b:raise ValueError('Precedence conflicts with simultaneous start')
        pred[b].add(a)
    result=[];remaining=set(groups)
    while remaining:
        ready=sorted(g for g in remaining if not(pred[g]&remaining))
        if not ready:raise ValueError('Cyclic precedence')
        batch=sorted(s for g in ready for s in groups[g]);used=set()
        for s in batch:
            hands={'left','right'} if skills[s]=='both' else {skills[s]}
            if used&hands:raise ValueError('Same-arm skills need a total order')
            used|=hands
        result.append(batch);remaining.difference_update(ready)
    return result


def _pose(p):
    p=np.asarray(p,float)
    if p.shape[-1:]!=(7,) or not np.isfinite(p).all() or not np.allclose(np.linalg.norm(p[...,3:],axis=-1),1,atol=1e-6):
        raise ValueError('Expected finite xyz + unit wxyz poses')
    return p


def adapt_poses(poses,source_anchor,target_anchor):
    """Apply a single fixed SE(3) anchor delta; never follow moving object twice."""
    poses=_pose(poses);source=_pose(source_anchor);target=_pose(target_anchor)
    if source.shape!=(7,) or target.shape!=(7,):raise ValueError('Anchors must be single poses')
    inv=source[3:]*np.array([1,-1,-1,-1]);delta=_quat_multiply(target[3:],inv)
    result=np.empty_like(poses);result[...,:3]=(poses[...,:3]-source[:3])@_rotation_matrix(delta).T+target[:3]
    result[...,3:]=_quat_multiply(delta,poses[...,3:]);return result


def rollout_action(action,sigma,*,seed):
    """One explicitly seeded draw; callers must use distinct per-tick seeds.

    Does not execute physics or manufacture off-policy observations. The label
    must be paired with the measured pre-action observation during real rollout.
    """
    action=np.asarray(action,float);sigma=np.asarray(sigma,float)
    if not np.isfinite(action).all() or not np.isfinite(sigma).all() or np.any(sigma<0):
        raise ValueError('Finite action and nonnegative noise required')
    noise=np.random.default_rng(seed).normal(size=action.shape)*sigma
    if noise.shape!=action.shape:raise ValueError('Noise shape must match action')
    return {'label_action':action.copy(),'executed_action':action+noise,'noise':noise,'seed':seed}


def plan_joint_path(start,goal,bounds,is_valid,*,seed=0,max_iterations=1000,resolution=.04,extension=.2):
    """Bounded seeded RRT, with every returned edge discretely checked.

    Coordinates must be bounded scalar joints, not free-joint quaternions.
    This is sampled geometric validation, not continuous swept collision proof.
    Callback must include the relevant robot/scene/held-object constraints.
    """
    start=np.asarray(start,float);goal=np.asarray(goal,float);bounds=np.asarray(bounds,float)
    if (start.ndim!=1 or not len(start) or goal.shape!=start.shape or bounds.shape!=(len(start),2)
        or not all(np.isfinite(x).all() for x in (start,goal,bounds)) or np.any(bounds[:,0]>=bounds[:,1])
        or not np.isfinite(resolution) or not np.isfinite(extension) or resolution<=0 or extension<=0
        or type(max_iterations) is not int or max_iterations<0):raise ValueError('Invalid joint planning inputs')
    checks=0
    def valid(q):
        nonlocal checks
        checks+=1
        return bool(np.all(q>=bounds[:,0]) and np.all(q<=bounds[:,1]) and is_valid(q))
    def samples(a,b):
        n=max(1,int(np.ceil(np.max(abs(b-a))/resolution)))
        return np.linspace(a,b,n+1)
    def edge(a,b):return all(valid(q) for q in samples(a,b))
    for name,q in [('start',start),('goal',goal)]:
        if not valid(q):return None,{'reason':'invalid_'+name,'checks':checks}
    if edge(start,goal):return samples(start,goal),{'method':'direct','checks':checks,'iterations':0}
    rng=np.random.default_rng(seed);nodes=[start.copy()];parents=[-1]
    for i in range(max_iterations):
        sample=goal if rng.random()<.2 else rng.uniform(bounds[:,0],bounds[:,1])
        k=int(np.argmin(np.linalg.norm(np.asarray(nodes)-sample,axis=1)));d=sample-nodes[k]
        target=nodes[k]+d*min(1,extension/max(np.linalg.norm(d),1e-15))
        if not edge(nodes[k],target):continue
        nodes.append(target);parents.append(k)
        if edge(target,goal):
            chain=[goal];j=len(nodes)-1
            while j>=0:chain.append(nodes[j]);j=parents[j]
            chain.reverse();path=[chain[0][None]]
            for a,b in zip(chain,chain[1:]):path.append(samples(a,b)[1:])
            return np.concatenate(path),{'method':'rrt','checks':checks,'iterations':i+1}
    return None,{'reason':'budget_exhausted','checks':checks,'iterations':max_iterations}
