"""Bounded support-constrained retargeting, with full-path verification."""
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from ..contracts import WorkflowError, positive
from ..native import descendants
from ...privileged import no_physics
from .native import NativeEvaluator
from .prior import finite, rotation2, yaw


def rotation(q):return Rotation.from_quat(np.roll(q,-1))


def skew(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def right_jacobian(v):
    theta=np.linalg.norm(v);a=skew(v)
    if theta<1e-5:return np.eye(3)-.5*a+a@a/6
    return np.eye(3)-(1-np.cos(theta))/theta**2*a+(theta-np.sin(theta))/theta**3*(a@a)


def pose_residual_jacobian(model,data,goals,ids,dofs):
    """Residual derivative in MuJoCo's velocity tangent coordinates."""
    parts=[];blocks=[]
    for name,g in goals.items():
        bid=ids[name];jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jacBody(model,data,jp,jr,bid)
        e=(rotation(g[3:]).inv()*rotation(data.xquat[bid])).as_rotvec();theta=np.linalg.norm(e);a=skew(e)
        inverse=np.eye(3)-.5*a+(1/12 if theta<1e-5 else (1-.5*theta/np.tan(.5*theta))/theta**2)*(a@a)
        parts.extend(10*(data.xpos[bid]-g[:3]));parts.extend(e)
        blocks.extend([10*jp[:,dofs],inverse@rotation(g[3:]).as_matrix().T@jr[:,dofs]])
    return np.asarray(parts),np.vstack(blocks)


class LateralSolver:
    def __init__(self,model,source_qpos,source_times,prior,bindings,options,deadline=None):
        self.m=model;self.d=mujoco.MjData(model);self.ref=mujoco.MjData(model)
        self.q=np.asarray(source_qpos,float);self.t=np.asarray(source_times,float)
        self.prior=prior;self.b=bindings;self.options=options;self.o=options['ik'];self.entry,self.end=prior['operation_interval']
        self.global_deadline=float('inf') if deadline is None else deadline;self.deadline=self.global_deadline
        if self.q.shape!=(len(self.t),model.nq) or not np.isfinite(self.q).all() or not np.isfinite(self.t).all() or np.any(np.diff(self.t)<=0):
            raise WorkflowError('invalid_input','Reference state/time shape')
        for k in ('max_nfev','key_stride','bridge_samples'):
            if type(self.o[k]) is not int or self.o[k]<1:raise WorkflowError('invalid_input',k)
        if self.o['bridge_samples']<2:raise WorkflowError('invalid_input','Bridge requires endpoints')
        for k in ('position_m','angle_rad','foot_position_m','foot_angle_rad','joint_speed_rad_s','root_speed_m_s','timeout_s','posture_weight','bridge_duration_s'):
            positive(self.o[k],k)
        robot=descendants(model,model.body(bindings['root_body']).id)
        self.limited=[j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in robot and model.jnt_limited[j]]
        self.joints=[j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in robot and
            model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE) and
            not any(x in (model.joint(j).name or '') for x in ('hand','finger','thumb','index','middle'))]
        self.dofs=np.r_[np.arange(6),model.jnt_dofadr[self.joints]].astype(int)
        self.names=[*bindings['feet'],*bindings['wrists'].values()]
        self.ids={n:model.body(n).id for n in self.names}
        self.source={n:[] for n in self.names}
        for row in self.q:
            self.budget()
            self.ref.qpos[:]=row;mujoco.mj_forward(model,self.ref)
            for n,b in self.ids.items():self.source[n].append(np.r_[self.ref.xpos[b],self.ref.xquat[b]])
        self.source={k:np.asarray(v) for k,v in self.source.items()}

    def check_joints(self,q,index=None):
        for j in self.limited:
            name=self.m.joint(j).name or '';value=float(q[self.m.jnt_qposadr[j]])
            tolerance=self.o.get('measured_hand_limit_tolerance_rad',1e-7) if 'hand' in name else 1e-7
            lo,hi=self.m.jnt_range[j]
            if value<lo-tolerance or value>hi+tolerance:
                raise WorkflowError('joint_limit_violation',str(dict(joint=name,source_row=index,value=value,limits=[float(lo),float(hi)],tolerance=tolerance)))

    def budget(self):
        if time.monotonic()>self.deadline:raise WorkflowError('planning_budget_exhausted','Candidate IK/verification timeout')

    def check_geometry(self,evaluator,q,index):
        if evaluator.collisions(q,index):raise WorkflowError('operation_collision',str(evaluator.last_collision_pairs))
        for group in evaluator.foot_groups:
            supported=any(abs(c.dist)<=self.options['native']['support_distance_m'] and
                ((int(c.geom1) in evaluator.ground and int(self.m.geom_bodyid[c.geom2]) in group) or
                 (int(c.geom2) in evaluator.ground and int(self.m.geom_bodyid[c.geom1]) in group)) for c in evaluator.d.contact)
            if not supported:raise WorkflowError('operation_support_lost','Both feet must remain supported at source row '+str(index))

    def goals(self,index,candidate,delta):
        goals={k:np.array(v,float) for k,v in candidate['feet'].items()}
        shift=np.asarray(candidate['xyyaw'][:2])-np.asarray(self.prior['source_stance_xyyaw'][:2])
        for side,n in self.b['wrists'].items():
            goals[n]=self.source[n][index].copy();goals[n][:3]+=delta if side=='right' else np.r_[shift,0.]
        return goals

    def preferred(self,index,candidate,delta):
        q=self.q[index].copy();q[:2]+=np.asarray(candidate['xyyaw'][:2])-self.prior['source_stance_xyyaw'][:2]
        obj=self.m.body(self.b['object']).id;j=int(self.m.body_jntadr[obj]);a=int(self.m.jnt_qposadr[j])
        if j<0 or self.m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:raise WorkflowError('unsupported_capability','Free apple required')
        q[a:a+3]+=delta
        return q

    def errors(self,q,goals):
        self.d.qpos[:]=q;mujoco.mj_forward(self.m,self.d);out={}
        for n,g in goals.items():
            b=self.ids[n];out[n]=dict(position_m=float(np.linalg.norm(self.d.xpos[b]-g[:3])),
                angle_rad=float((rotation(g[3:]).inv()*rotation(self.d.xquat[b])).magnitude()))
        return out

    def poses_valid(self,q,goals):
        errors=self.errors(q,goals)
        return all(v['position_m']<=self.o['foot_position_m' if n in self.b['feet'] else 'position_m'] and
                   v['angle_rad']<=self.o['foot_angle_rad' if n in self.b['feet'] else 'angle_rad'] for n,v in errors.items()),errors

    def fit(self,preferred,goals,seed=None):
        self.budget();base=preferred.copy();start=np.zeros(len(self.dofs))
        if seed is not None:
            vel=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,vel,1.,base,seed);start=vel[self.dofs]
        low=np.full(len(start),-.5);high=-low;low[:3]=-.08;high[:3]=.08
        for i,j in enumerate(self.joints,6):
            if self.m.jnt_limited[j]:low[i],high[i]=self.m.jnt_range[j]-base[self.m.jnt_qposadr[j]]
        def pose(x):
            q=base.copy();v=np.zeros(self.m.nv);v[self.dofs]=x;mujoco.mj_integratePos(self.m,q,v,1.);return q
        cached=[None,None,None]
        weights=np.r_[np.full(6,.2),np.full(len(start)-6,self.o['posture_weight'])]
        def objective(x):
            self.budget()
            if cached[0] is None or not np.array_equal(cached[0],x):
                q=pose(x);self.d.qpos[:]=q;mujoco.mj_forward(self.m,self.d)
                error,jac=pose_residual_jacobian(self.m,self.d,goals,self.ids,self.dofs)
                mapping=np.eye(len(x));mapping[3:6,3:6]=right_jacobian(x[3:6])
                cached[:]=[x.copy(),np.r_[error,weights*x],np.vstack([jac@mapping,np.diag(weights)])]
            return cached[1],cached[2]
        ok,errors=self.poses_valid(base,goals)
        if ok:
            self.check_joints(base)
            return base,errors
        result=least_squares(lambda x:objective(x)[0],np.clip(start,low+1e-8,high-1e-8),jac=lambda x:objective(x)[1],bounds=(low,high),
                             max_nfev=self.o['max_nfev'],ftol=1e-6,xtol=1e-6,gtol=1e-6)
        self.solver_audit.append(dict(status=int(result.status),nfev=result.nfev,optimality=float(result.optimality)))
        if result.status==0:raise WorkflowError('planning_budget_exhausted','IK max_nfev reached: '+str(result.nfev))
        q=pose(result.x);ok,errors=self.poses_valid(q,goals)
        if not ok:raise WorkflowError('operation_ik_residual',str(errors))
        self.check_joints(q)
        return q,errors

    def verify_segment(self,a,b,dt,goals_a,goals_b,evaluator,index):
        v=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,v,1.,a,b)
        if np.linalg.norm(v[:3])/dt>self.o['root_speed_m_s'] or np.max(abs(v[3:6]))/dt>self.o['joint_speed_rad_s'] or np.max(abs(v[self.m.jnt_dofadr[self.joints]]),initial=0)/dt>self.o['joint_speed_rad_s']:
            raise WorkflowError('reference_speed_limit','Generated velocity exceeds declared bound')
        n=max(1,int(np.ceil(np.linalg.norm(v[:3])/self.options['native']['sample_translation_m'])),int(np.ceil(np.max(abs(v[3:]),initial=0)/self.options['native']['sample_angle_rad'])))
        if n>self.options['native']['max_checks']:raise WorkflowError('planning_budget_exhausted','Interpolation budget')
        summary=dict(source_row=index,samples=n+1,root_speed_m_s=float(np.linalg.norm(v[:3])/dt),
                     joint_speed_rad_s=float(np.max(abs(v[self.m.jnt_dofadr[self.joints]]),initial=0)/dt),peak_position_m=0.,peak_angle_rad=0.)
        for u in np.linspace(0,1,n+1):
            self.budget();q=a.copy();mujoco.mj_integratePos(self.m,q,v,float(u))
            self.check_joints(q,index)
            self.check_geometry(evaluator,q,index)
            # Foot poses are fixed; wrist goals interpolate on SE(3), never interpolate quaternion components.
            goals={}
            for name,g in goals_a.items():
                h=goals_b[name];r=rotation(g[3:]);rr=r*Rotation.from_rotvec((r.inv()*rotation(h[3:])).as_rotvec()*u)
                goals[name]=np.r_[g[:3]*(1-u)+h[:3]*u,np.roll(rr.as_quat(),1)]
            ok,errors=self.poses_valid(q,goals)
            if not ok:raise WorkflowError('interpolated_pose_residual',str(errors))
            summary['peak_position_m']=max(summary['peak_position_m'],max(e['position_m'] for e in errors.values()))
            summary['peak_angle_rad']=max(summary['peak_angle_rad'],max(e['angle_rad'] for e in errors.values()))
        return summary

    def correction(self,q,preferred):
        v=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,v,1.,preferred,q)
        metrics=dict(max_body_joint_correction_rad=float(np.max(abs(v[self.m.jnt_dofadr[self.joints]]),initial=0)),
                     max_root_correction_m=float(np.linalg.norm(v[:3])),max_root_correction_rad=float(np.linalg.norm(v[3:6])))
        for key,value in metrics.items():
            if key in self.o:
                positive(self.o[key],key,zero=True)
                if value>self.o[key]+1e-9:raise WorkflowError('reference_correction_limit',str(dict(metric=key,value=value,limit=self.o[key])))
        return metrics

    def screen(self,candidate,delta,current):
        """Cheap necessary entry/contact/lift checks, never a trajectory certificate."""
        self.deadline=min(self.global_deadline,time.monotonic()+self.o['timeout_s']);self.solver_audit=[]
        result=dict(feasible=False,operation_checked=False,physics_steps=0,checking_kind='sparse_screen_only')
        frame=np.asarray(self.prior['source_task_frame_xyzyaw']).copy();frame[:3]+=delta
        try:
            with no_physics():
                evaluator=NativeEvaluator(self.m,current,self.q,self.prior,frame,self.b,self.options['native'])
                stages=self.b.get('operation_stages',{})
                rows=sorted({self.entry,stages.get('closure',self.entry),stages.get('lift',self.end)})
                for index in rows:
                    preferred=self.preferred(index,candidate,delta)
                    q,_=self.fit(preferred,self.goals(index,candidate,delta))
                    self.correction(q,preferred);self.check_geometry(evaluator,q,index)
                result.update(feasible=True,checked_source_rows=rows)
        except WorkflowError as exc:result.update(reason=exc.code,detail=exc.detail)
        return result

    def solve(self,candidate,object_delta,current):
        delta=finite(object_delta,(3,),'Object delta');normal=np.asarray(self.prior['surface']['normal'])
        if abs(delta[2])>1e-9 or abs(np.dot(delta,normal))>1e-9:raise WorkflowError('invalid_lateral_scene','Depth/height change')
        self.deadline=min(self.global_deadline,time.monotonic()+self.o['timeout_s']);started=time.monotonic();self.solver_audit=[]
        row=dict(feasible=False,operation_checked=False,reason=None,physics_steps=0)
        try:
            with no_physics():
                preferred_rows=np.asarray([self.preferred(i,candidate,delta) for i in range(self.entry,self.end+1)])
                reuse=bool(self.o.get('reuse_source_first',False))
                if reuse:
                    for j,i in enumerate(range(self.entry,self.end+1)):
                        self.budget()
                        if not self.poses_valid(preferred_rows[j],self.goals(i,candidate,delta))[0]:reuse=False;break
                peak=0.
                if reuse:generated=preferred_rows.copy()
                else:
                    keys=sorted(set(range(self.entry,self.end+1,self.o['key_stride']))|set(self.prior['key_indices'])|{self.end})
                    solved={};seed=None;peak=0.
                    for i in keys:
                        seed,errors=self.fit(self.preferred(i,candidate,delta),self.goals(i,candidate,delta),seed)
                        solved[i]=seed.copy();peak=max(peak,max(v['position_m'] for v in errors.values()))
                    output=[]
                    for i in range(self.entry,self.end+1):
                        if i in solved:q=solved[i].copy()
                        else:
                            a=max(k for k in keys if k<i);b=min(k for k in keys if k>i)
                            v=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,v,1.,solved[a],solved[b]);q=solved[a].copy()
                            mujoco.mj_integratePos(self.m,q,v,float((self.t[i]-self.t[a])/(self.t[b]-self.t[a])))
                        output.append(q)
                    generated=np.asarray(output);frame=np.asarray(self.prior['source_task_frame_xyzyaw']).copy();frame[:3]+=delta
                frame=np.asarray(self.prior['source_task_frame_xyzyaw']).copy();frame[:3]+=delta
                modification=dict(max_body_joint_correction_rad=0.,max_root_correction_m=0.,max_root_correction_rad=0.)
                ev=NativeEvaluator(self.m,current,self.q,self.prior,frame,self.b,self.options['native'])
                verification=[];margin=float('inf')
                for j,i in enumerate(range(self.entry,self.end+1)):
                    self.budget();ok,errors=self.poses_valid(generated[j],self.goals(i,candidate,delta))
                    self.check_joints(generated[j],i)
                    for key,value in self.correction(generated[j],preferred_rows[j]).items():modification[key]=max(modification[key],value)
                    if not ok:raise WorkflowError('interpolated_pose_residual',str(errors))
                    margin=min(margin,min(self.o['foot_position_m' if n in self.b['feet'] else 'position_m']-v['position_m'] for n,v in errors.items()))
                    verification.append(dict(source_row=i,pose_errors=errors))
                    self.check_geometry(ev,generated[j],i)
                    if j:verification[-1]['previous_segment']=self.verify_segment(generated[j-1],generated[j],self.t[i]-self.t[i-1],self.goals(i-1,candidate,delta),self.goals(i,candidate,delta),ev,i-1)
                floor=self.o.get('minimum_position_margin_m',0.);positive(floor,'Minimum pose margin',zero=True)
                if margin<floor:raise WorkflowError('operation_margin_too_small','Full-state minimum positional residual margin')
                # A bridge cannot move the planted feet. Reject it if direct interpolation violates support.
                start=finite(current,(self.m.nq,),'Bridge start');bridge=[];bt=np.linspace(0,self.o['bridge_duration_s'],self.o['bridge_samples'])
                v=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,v,1.,start,generated[0])
                start_goals={}
                self.d.qpos[:]=start;mujoco.mj_forward(self.m,self.d)
                for n,b in self.ids.items():start_goals[n]=np.r_[self.d.xpos[b],self.d.xquat[b]].copy()
                for n in self.b['feet']:
                    if np.linalg.norm(start_goals[n][:3]-np.asarray(candidate['feet'][n])[:3])>self.o['foot_position_m']:
                        raise WorkflowError('landing_support_mismatch','Bridge cannot step '+n)
                for u in bt/bt[-1]:
                    q=start.copy();mujoco.mj_integratePos(self.m,q,v,float(10*u**3-15*u**4+6*u**5));bridge.append(q)
                bridge_verification=[]
                for j in range(1,len(bridge)):
                    ga={};gb={}
                    for values,q in ((ga,bridge[j-1]),(gb,bridge[j])):
                        self.d.qpos[:]=q;mujoco.mj_forward(self.m,self.d)
                        for n,b in self.ids.items():values[n]=np.asarray(candidate['feet'][n]) if n in self.b['feet'] else np.r_[self.d.xpos[b],self.d.xquat[b]]
                    bridge_verification.append(self.verify_segment(bridge[j-1],bridge[j],bt[j]-bt[j-1],ga,gb,ev,None))
                row.update(feasible=True,operation_checked=True,reference_reused_without_ik=reuse,reference_modification=modification,qpos=generated,source_rows=np.arange(self.entry,self.end+1),
                    time_s=self.t[self.entry:self.end+1].copy(),bridge_qpos=np.asarray(bridge),bridge_time_s=bt,
                    geometry_checks=ev.checks,peak_keyframe_position_error_m=peak,
                    minimum_position_margin_m=margin,verification=verification,bridge_verification=bridge_verification,
                    entry_qpos=generated[0].tolist(),checking_kind='whole_reference_kinematics_not_dynamics')
        except WorkflowError as exc:row.update(reason=exc.code,detail=exc.detail)
        row['planning_wall_s']=time.monotonic()-started;row['solver_audit']=self.solver_audit
        return row
