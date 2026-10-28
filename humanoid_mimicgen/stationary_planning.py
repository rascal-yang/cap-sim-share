"""CPU stationary subset of HMG whole-body planning on native MuJoCo geometry.

Uses arms, then arms+torso. Root/legs/fingers/object are fixed. A failure requests
further locomotion planning; this module never teleports or executes a robot.
Uses measured wrist body frames, not teleop hand-tip offsets. Collision checks
use enabled model contacts, not cuRobo's inflated sphere approximation.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .skill_planning import plan_joint_path, _pose


def rotation(q):return Rotation.from_quat(np.roll(q,-1,axis=-1))


class StationaryPlanner:
    def __init__(self,model,*,allowed_geom_pairs=(),penetration_tolerance_m=1e-5):
        self.m=model;self.d=mujoco.MjData(model)
        self.wrists={s:model.body(s+'_wrist_yaw_link').id for s in ('left','right')}
        self.arms=[j for j in range(model.njnt) if any(part in model.joint(j).name for part in ('shoulder','elbow','wrist'))]
        self.torso=[j for j in range(model.njnt) if 'waist_' in model.joint(j).name]
        self.allowed={frozenset(p) for p in allowed_geom_pairs};self.tolerance=float(penetration_tolerance_m)
        if not np.isfinite(self.tolerance) or self.tolerance<0:raise ValueError('Invalid collision tolerance')
        for j in self.arms+self.torso:
            if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[j]:raise ValueError('Expected bounded scalar joints')

    def _forward(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.m.nq,) or not np.isfinite(q).all():raise ValueError('Invalid qpos')
        self.d.qpos[:]=q;self.d.qvel[:]=0;self.d.ctrl[:]=0
        mujoco.mj_forward(self.m,self.d)

    def wrist_poses(self,q):
        self._forward(q)
        return {s:np.r_[self.d.xpos[b],self.d.xquat[b]].copy() for s,b in self.wrists.items()}

    def collisions(self,q):
        self._forward(q);result=[]
        # Existing scene-ground and object-support contacts are irrelevant to robot planning.
        # Only foot-ground contacts are excluded for the robot; tables stay obstacles.
        for c in self.d.contact:
            if c.dist>=-self.tolerance:continue
            gs=[int(c.geom1),int(c.geom2)];names=[self.m.geom(g).name or str(g) for g in gs]
            bodies=[self.m.body(int(self.m.geom_bodyid[g])).name or '' for g in gs]
            robot=[('link' in b or 'hand' in b or 'torso' in b or 'pelvis' in b) for b in bodies]
            if not any(robot):continue
            if frozenset(names) in self.allowed:continue
            ground=[('floor' in n or 'ground' in n) for n in names+bodies]
            if any(ground) and any(('ankle' in b or 'foot' in b) for b in bodies):continue
            result.append({'geoms':names,'bodies':bodies,'penetration_m':float(-c.dist)})
        return result

    def solve(self,current,targets,*,max_seeds=3,max_nfev=100,seed=0,position_tolerance=.003,angle_tolerance=.03):
        current=np.array(current,float,copy=True);self._forward(current)
        if not targets or not set(targets)<=set(self.wrists):raise ValueError('Invalid target hands')
        targets={s:_pose(p).copy() for s,p in targets.items()}
        if any(p.shape!=(7,) for p in targets.values()):raise ValueError('Target must be one pose per hand')
        if type(max_seeds) is not int or max_seeds<1 or max_nfev<1:raise ValueError('Invalid IK budget')
        rng=np.random.default_rng(seed);attempts=[]
        for name,joints in [('arms',self.arms),('arms_torso',self.arms+self.torso)]:
            adr=self.m.jnt_qposadr[joints];bounds=self.m.jnt_range[joints];q=current.copy()
            def residual(x):
                q[adr]=x;poses=self.wrist_poses(q);errors=[]
                for s,target in targets.items():
                    errors.extend((poses[s][:3]-target[:3])*10)
                    errors.extend((rotation(target[3:])*rotation(poses[s][3:]).inv()).as_rotvec())
                return np.r_[errors,.001*(x-current[adr])]
            for k in range(max_seeds):
                initial=np.clip(current[adr]+(rng.normal(0,.15,len(adr)) if k else 0),bounds[:,0]+1e-8,bounds[:,1]-1e-8)
                fit=least_squares(residual,initial,bounds=(bounds[:,0],bounds[:,1]),max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                q[adr]=fit.x;poses=self.wrist_poses(q);errors={s:{'position_m':float(np.linalg.norm(poses[s][:3]-p[:3])), 'angle_rad':float((rotation(p[3:])*rotation(poses[s][3:]).inv()).magnitude())} for s,p in targets.items()}
                collisions=self.collisions(q)
                ok=all(e['position_m']<=position_tolerance and e['angle_rad']<=angle_tolerance for e in errors.values()) and not collisions
                attempts.append({'group':name,'seed_index':k,'errors':errors,'collisions':collisions,'nfev':fit.nfev,'accepted':ok})
                if ok:return {'success':True,'group':name,'qpos':q.copy(),'joint_ids':joints,'attempts':attempts,'physics_steps':0}
        return {'success':False,'reason':'no_stationary_solution','attempts':attempts,'physics_steps':0}

    def connect(self,start,goal,joint_ids,*,seed=0,max_iterations=600,resolution=.04):
        start=np.asarray(start,float);goal=np.asarray(goal,float);adr=self.m.jnt_qposadr[joint_ids]
        fixed=np.ones(self.m.nq,bool);fixed[adr]=False
        if not np.array_equal(start[fixed],goal[fixed]):raise ValueError('Stationary connector cannot change frozen coordinates')
        def valid(x):
            q=start.copy();q[adr]=x
            return not self.collisions(q)
        path,info=plan_joint_path(start[adr],goal[adr],self.m.jnt_range[joint_ids],valid,seed=seed,max_iterations=max_iterations,resolution=resolution)
        if path is None:return None,info
        full=np.repeat(start[None],len(path),axis=0);full[:,adr]=path
        return full,info
