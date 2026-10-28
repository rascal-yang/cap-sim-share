"""Whole-reference kinematic screening on planning copies, never live state.

Candidate reference retargeting preserves articulation and root height. If a new
surface height requires a new manipulation trajectory, reject rather than claim
that planar navigation solves it. Arbitrary body targets support seated tasks.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import WorkflowError, positive
from ..native import descendants
from .prior import finite, rotation2, yaw, wrap


class NativeEvaluator:
    def __init__(self, model, current, qpos, prior, target_frame, bindings, options):
        import mujoco
        self.m=model;self.d=mujoco.MjData(model);self.ref=mujoco.MjData(model)
        self.current=finite(current,(model.nq,),'Current qpos').copy()
        self.qpos=np.asarray(qpos,float)
        if self.qpos.ndim!=2 or self.qpos.shape[1]!=model.nq or not np.isfinite(self.qpos).all():
            raise WorkflowError('invalid_input','Reference qpos')
        self.prior=prior;self.frame=finite(target_frame,(4,),'Target frame')
        self.b=bindings;self.o=options;self.robot=descendants(model,model.body(bindings['root_body']).id)
        root=model.body(bindings['root_body']).id;j=int(model.body_jntadr[root])
        if j<0 or model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[j]!=0:
            raise WorkflowError('unsupported_capability','First free robot root required')
        self.foot_groups=[descendants(model,model.body(name).id) for name in bindings['feet']]
        self.feet=set().union(*self.foot_groups)
        self.object_bodies=descendants(model,model.body(bindings['object']).id)
        self.ground={model.geom(name).id for name in bindings['ground_geoms']}
        self.body_targets=[model.body(name).id for name in bindings['operation_bodies']]
        if not self.body_targets:raise WorkflowError('needs_annotation','Operation bodies required')
        for k in ('sample_translation_m','sample_angle_rad','planning_timeout_s','relation_tolerance_m','relation_tolerance_rad','support_distance_m'):
            positive(options[k],k)
        for k in ('collision_tolerance_m','corridor_margin_m'):positive(options[k],k,zero=True)
        if type(options['max_checks']) is not int or options['max_checks']<1:
            raise WorkflowError('invalid_input','Geometry budget')
        self.contact=[]
        for phase in bindings['contact_policy']:
            if not 0<=phase['start']<=phase['end']<len(qpos):raise WorkflowError('invalid_input','Contact interval')
            pairs={frozenset((model.geom(a).id,model.geom(b).id)) for a,b in phase['allowed_geom_pairs']}
            self.contact.append((phase['start'],phase['end'],pairs))
        self.entry,self.end=prior['operation_interval']
        if not 0<=self.entry<=self.end<len(qpos):raise WorkflowError('invalid_input','Operation interval')
        self.started=time.monotonic();self.checks=0

    def budget(self):
        self.checks+=1
        if self.checks>self.o['max_checks'] or time.monotonic()-self.started>self.o['planning_timeout_s']:
            raise WorkflowError('planning_budget_exhausted','Geometry checks/time')

    def collisions(self,q,index=None):
        import mujoco
        self.budget();self.d.qpos[:]=q;mujoco.mj_forward(self.m,self.d)
        allowed={frozenset((self.m.geom(a).id,self.m.geom(b).id)) for a,b in self.b.get('navigation_allowed_geom_pairs',[])} if index is None else set()
        if index is not None:
            for first,last,pairs in self.contact:
                if first<=index<=last:allowed.update(pairs)
        found=[]
        for c in self.d.contact:
            if c.dist>=-self.o['collision_tolerance_m']:continue
            a,b=int(c.geom1),int(c.geom2);ba,bb=int(self.m.geom_bodyid[a]),int(self.m.geom_bodyid[b])
            if ba not in self.robot|self.object_bodies and bb not in self.robot|self.object_bodies:continue
            if ((a in self.ground and bb in self.feet) or (b in self.ground and ba in self.feet)) and c.dist>=-self.o['support_distance_m']:continue
            if frozenset((a,b)) in allowed and c.dist>=-self.o['support_distance_m']:continue
            found.append([self.m.geom(a).name,self.m.geom(b).name])
        # Navigation requires both feet on declared ground; contact operation may
        # transfer support to explicitly allowed robot/environment contacts.
        grounded=[]
        for group in self.foot_groups:
            grounded.append(any(abs(c.dist)<=self.o['support_distance_m'] and
                ((int(c.geom1) in self.ground and int(self.m.geom_bodyid[c.geom2]) in group) or
                 (int(c.geom2) in self.ground and int(self.m.geom_bodyid[c.geom1]) in group)) for c in self.d.contact))
        support=all(grounded) if index is None else any(grounded)
        if index is not None and not support:
            support=any(abs(c.dist)<=self.o['support_distance_m'] and frozenset((int(c.geom1),int(c.geom2))) in allowed and
                        ((int(self.m.geom_bodyid[c.geom1]) in self.robot)!=(int(self.m.geom_bodyid[c.geom2]) in self.robot)) for c in self.d.contact)
        if not support:found.append(['unsupported_robot','declared_support'])
        self.last_collision_pairs=found
        return found

    def segment(self,a,b,index):
        import mujoco
        velocity=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,velocity,1.,a,b)
        n=max(1,int(np.ceil(np.linalg.norm(velocity[:3])/self.o['sample_translation_m'])),
              int(np.ceil(np.max(abs(velocity[3:]))/self.o['sample_angle_rad'])))
        if n>self.o['max_checks']-self.checks:raise WorkflowError('planning_budget_exhausted','Segment subdivision budget')
        for t in np.linspace(0.,1.,n+1):
            q=a.copy();mujoco.mj_integratePos(self.m,q,velocity,float(t))
            if self.collisions(q,index):return False
        return True

    def transformed(self,index,candidate):
        """Reanchor robot to candidate; move object replay relative to new object frame."""
        import mujoco
        source=self.qpos[index];q=self.current.copy()
        old=np.asarray(self.prior['source_stance_xyyaw']);new=np.asarray(candidate['xyyaw'])
        delta=float(wrap(new[2]-old[2]));r=rotation2(delta)
        for j in range(self.m.njnt):
            body=int(self.m.jnt_bodyid[j]);adr=int(self.m.jnt_qposadr[j]);typ=self.m.jnt_type[j]
            width=7 if typ==mujoco.mjtJoint.mjJNT_FREE else 4 if typ==mujoco.mjtJoint.mjJNT_BALL else 1
            if body in self.robot:q[adr:adr+width]=source[adr:adr+width]
        q[:2]=new[:2]+r@(source[:2]-old[:2])
        orientation=Rotation.from_euler('z',delta)*Rotation.from_quat(np.roll(source[3:7],-1))
        q[3:7]=np.roll(orientation.as_quat(),1)
        # Articulated object motion and free object displacement follow replay;
        # unrelated environment joints retain new-scene initial values.
        object_bodies=descendants(self.m,self.m.body(self.b['object']).id)
        sf=np.asarray(self.prior['source_task_frame_xyzyaw']);turn=self.frame[3]-sf[3]
        for j in range(self.m.njnt):
            if int(self.m.jnt_bodyid[j]) not in object_bodies:continue
            adr=int(self.m.jnt_qposadr[j]);typ=self.m.jnt_type[j]
            if typ==mujoco.mjtJoint.mjJNT_FREE:
                q[adr:adr+2]=self.frame[:2]+rotation2(turn)@(source[adr:adr+2]-sf[:2])
                q[adr+2]=source[adr+2]+self.frame[2]-sf[2]
                rot=Rotation.from_euler('z',turn)*Rotation.from_quat(np.roll(source[adr+3:adr+7],-1))
                q[adr+3:adr+7]=np.roll(rot.as_quat(),1)
            elif typ in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):
                q[adr]=self.current[adr]+source[adr]-self.qpos[self.entry,adr]
            else:raise WorkflowError('unsupported_capability','Object ball joint retargeting')
        return q

    def relations_valid(self,source,q):
        import mujoco
        self.budget();self.ref.qpos[:]=source;mujoco.mj_forward(self.m,self.ref)
        self.d.qpos[:]=q;mujoco.mj_forward(self.m,self.d)
        sf=np.asarray(self.prior['source_task_frame_xyzyaw']);r=Rotation.from_euler('z',self.frame[3]-sf[3])
        for body in self.body_targets:
            desired=self.frame[:3]+r.apply(self.ref.xpos[body]-sf[:3])
            angle=(r*Rotation.from_quat(np.roll(self.ref.xquat[body],-1))*Rotation.from_quat(np.roll(self.d.xquat[body],-1)).inv()).magnitude()
            if np.linalg.norm(self.d.xpos[body]-desired)>self.o['relation_tolerance_m'] or angle>self.o['relation_tolerance_rad']:return False
        return True

    def __call__(self,candidate):
        from ...reachability_navigation import plan_base_path
        from ...privileged import no_physics
        row={'feasible':False,'operation_checked':False,'geometry_checks':0}
        def rejected(reason,**detail):
            row.update(reason=reason,geometry_checks=self.checks,**detail)
            if 'collision' in reason or reason=='navigation_path_failed':
                row['last_collision_pairs']=getattr(self,'last_collision_pairs',[])
            return row
        try:
            with no_physics():
                goal=finite(candidate['xyyaw'],(3,),'Candidate pose');start=np.r_[self.current[:2],yaw(self.current[:7])]
                def valid(base):
                    q=self.current.copy();q[:2]=base[:2]
                    q[3:7]=np.roll((Rotation.from_euler('z',base[2]-start[2])*Rotation.from_quat(np.roll(self.current[3:7],-1))).as_quat(),1)
                    radius=self.o['corridor_margin_m']
                    offsets=[(0.,0.)]+[(radius*np.cos(a),radius*np.sin(a)) for a in np.arange(8)*np.pi/4] if radius else [(0.,0.)]
                    for delta in offsets:
                        probe=q.copy();probe[:2]+=delta
                        if self.collisions(probe):return False
                    return True
                route,audit=plan_base_path(start,goal,valid,**self.o['path']);row['navigation_path_audit']=audit
                if route is None:return rejected('navigation_path_failed')
                previous=self.current.copy();previous[:2]=goal[:2];previous[3:7]=np.roll((Rotation.from_euler('z',goal[2]-start[2])*Rotation.from_quat(np.roll(self.current[3:7],-1))).as_quat(),1)
                entry_q=self.transformed(self.entry,candidate)
                if not self.segment(previous,entry_q,None):return rejected('approach_collision')
                checked=[];previous=entry_q
                for index in range(self.entry,self.end+1):
                    q=self.transformed(index,candidate)
                    if not self.relations_valid(self.qpos[index],q):
                        return rejected('operation_relation_changed',failed_source_index=index)
                    # At a phase boundary the previous phase must also allow the transition.
                    if index>self.entry and not self.segment(previous,q,index-1):
                        return rejected('operation_collision',failed_source_index=index)
                    if self.collisions(q,index):return rejected('operation_collision',failed_source_index=index)
                    checked.append(index);previous=q
                row.update(feasible=True,reason=None,operation_checked=True,checked_source_indices=checked,
                           navigation_path=route.tolist(),entry_qpos=entry_q.tolist(),
                           travel_m=float(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1).sum()),
                           checking_kind='whole_reference_kinematics_not_dynamics')
        except WorkflowError as exc:
            row.update(reason=exc.code,detail=exc.detail)
        finally:row['geometry_checks']=self.checks
        return row
