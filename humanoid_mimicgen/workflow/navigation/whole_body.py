"""Continuous whole-body reference execution after measured navigation handoff.

All qpos writes are to FK/planning copies. The live robot receives SONIC mode1
references and recorded finger commands; no teleport, token prefix or reset.
"""
import numpy as np
from ..contracts import WorkflowError
from .operation import PhaseClock, poses_aligned, closure_frames
from .prior import yaw


class BodyReference:
    def __init__(self,model,pose_bodies,alignment_bodies):
        import mujoco
        self.model=model;self.data=mujoco.MjData(model)
        self.pose_ids={k:model.body(v).id for k,v in pose_bodies.items()}
        self.align_ids={name:model.body(name).id for name in alignment_bodies}

    def poses(self,q):
        import mujoco
        self.data.qpos[:]=q;mujoco.mj_forward(self.model,self.data)
        def pose(b):return np.r_[self.data.xpos[b],self.data.xquat[b]].copy()
        return ({k:pose(v) for k,v in self.pose_ids.items()},
                {k:pose(v) for k,v in self.align_ids.items()})


def set_body_reference(sonic,q):
    """Hold complete reference at the current phase; never preview past a gate."""
    sonic.navigating=False
    sonic.ref=np.repeat(np.r_[q[:7],q[sonic.qids]][None],100,axis=0)
    sonic.ref_tick=sonic.tick
    sonic.handoff_until_tick=sonic.tick+100


class WholeBodyOperation:
    def __init__(self,adapter,plan,strategy,source,fingers,generated_reference=None):
        import mujoco
        from .archive import load_archive
        from .native import NativeEvaluator
        self.a=adapter;self.config=dict(strategy['policy']['whole_body_operation']);self.offset=strategy['command_pose_offset']
        self.start=strategy['phases'][0]['start'];self.end=strategy['phases'][-1]['end']
        self.fingers=np.asarray(fingers,float)
        if self.fingers.ndim!=2 or self.fingers.shape[1]!=14 or len(self.fingers)<self.end or not np.isfinite(self.fingers).all():
            raise WorkflowError('invalid_input','Source finger alignment')
        if not np.allclose(self.fingers[self.start],self.config['open_fingers_rad'],atol=1e-7,rtol=0):
            raise WorkflowError('invalid_input','Operation must begin before source finger closure')
        self.config['geometry_gates']=sorted(set(self.config.get('geometry_gates',[]))|set(closure_frames(self.fingers,self.config['open_fingers_rad'],self.start,self.end)))
        self.clock=PhaseClock(self.start,self.end,self.config)
        self.generated_bridge=None
        if plan.get('reference_mode')=='apple_lateral_v1':
            from .reference_bundle import load_reference_bundle
            from .lateral_execution import validate_generated_binding,validate_derived_reference
            if generated_reference is None:raise WorkflowError('handoff_not_ready','Measured generated reference required')
            validate_generated_binding(plan,strategy)
            generated=validate_derived_reference(plan,generated_reference,source,adapter.d.qpos);bindings=plan['prior']['bindings']
            if (generated['metadata']['operation_interval']!=[self.start,self.end-1] or
                not np.array_equal(generated['fingers'],self.fingers[self.start:self.end]) or
                not np.allclose(generated['bridge_qpos'][0],adapter.d.qpos,atol=1e-9,rtol=0) or
                not np.isclose(generated['bridge_time_s'][-1],self.config['transition_s'],atol=1e-9,rtol=0)):
                raise WorkflowError('context_changed','Generated handoff state, fingers or time changed')
            self.references={int(i):q.copy() for i,q in zip(generated['source_rows'],generated['qpos'])}
            self.generated_bridge=(generated['bridge_time_s'],generated['bridge_qpos'])
        else:
            _,bindings,_=load_archive(plan['source_manifest'])
            evaluator=NativeEvaluator(adapter.m,adapter.d.qpos.copy(),source[:,1:1+adapter.m.nq],plan['prior'],
                plan['scene']['task_frame_xyzyaw'],bindings,plan['native_options'])
            candidate={'xyyaw':[*adapter.d.qpos[:2],float(yaw(adapter.d.qpos[:7]))]}
            self.references={i:evaluator.transformed(i+self.offset,candidate) for i in range(self.start,self.end)}
        self.initial=adapter.d.qpos.copy();self.entry=self.references[self.start]
        # All intermediate articulation checked against the actual measured scene.
        if self.generated_bridge is None and not evaluator.segment(self.initial,self.entry,None):
            raise WorkflowError('operation_transition_collision',str(evaluator.last_collision_pairs))
        self.tangent=np.zeros(adapter.m.nv)
        mujoco.mj_differentiatePos(adapter.m,self.tangent,1.,self.initial,self.entry)
        # Quintic interpolation peaks at 1.875: enforce declared physical reference speeds.
        approach=strategy['policy']['approach']
        duration=self.config['transition_s']
        if self.generated_bridge is None and np.max(abs(self.tangent[3:]))*1.875/duration>approach['joint_speed_rad_s']:
            raise WorkflowError('operation_transition_too_fast','Increase declared transition_s')
        self.config['alignment_bodies']=list(dict.fromkeys([*self.config['alignment_bodies'],bindings['object']]))
        self.fk=BodyReference(adapter.m,{**strategy['bindings']['wrists'],'torso':strategy['bindings']['torso']},self.config['alignment_bodies'])
        p0,_=self.fk.poses(self.initial);p1,_=self.fk.poses(self.entry)
        from scipy.spatial.transform import Rotation
        for side in ('left','right'):
            distance=np.linalg.norm(p1[side][:3]-p0[side][:3]);angle=(Rotation.from_quat(np.roll(p0[side][3:],-1)).inv()*Rotation.from_quat(np.roll(p1[side][3:],-1))).magnitude()
            if self.generated_bridge is None and (distance*1.875/duration>approach['wrist_speed_m_s'] or angle*1.875/duration>approach['wrist_speed_rad_s']):
                raise WorkflowError('operation_transition_too_fast','Wrist transition speed')
        if self.generated_bridge is not None:
            times,qs=self.generated_bridge
            for j in range(1,len(qs)):
                dt=times[j]-times[j-1];v=np.zeros(adapter.m.nv);mujoco.mj_differentiatePos(adapter.m,v,dt,qs[j-1],qs[j])
                if np.max(abs(v[3:]))>approach['joint_speed_rad_s']:raise WorkflowError('operation_transition_too_fast','Generated joint bridge')
                p0,_=self.fk.poses(qs[j-1]);p1,_=self.fk.poses(qs[j])
                for side in ('left','right'):
                    angle=(Rotation.from_quat(np.roll(p0[side][3:],-1)).inv()*Rotation.from_quat(np.roll(p1[side][3:],-1))).magnitude()
                    if np.linalg.norm(p1[side][:3]-p0[side][:3])/dt>approach['wrist_speed_m_s'] or angle/dt>approach['wrist_speed_rad_s']:
                        raise WorkflowError('operation_transition_too_fast','Generated wrist bridge')
        self.elapsed=0.;self.last_q=self.entry;self.frame=None;self.audit=[]
        self.control_dt=strategy['policy']['control_dt_s'];self.terminal_sent_at=None;self.applied_frame=None
        self.applied_fingers=np.asarray(self.config['open_fingers_rad'])
        self.object_bodies=[adapter.m.body(b).name for b in adapter.object_bodies]

    def command(self,dt,stable,*,policy_tick=True):
        import mujoco
        self.elapsed+=dt
        if self.elapsed<self.config['transition_s']:
            if self.generated_bridge is None:
                u=min(1.,self.elapsed/self.config['transition_s']);blend=10*u**3-15*u**4+6*u**5
                q=self.initial.copy();mujoco.mj_integratePos(self.a.m,q,self.tangent,blend)
            else:
                times,qs=self.generated_bridge;j=min(len(times)-1,int(np.searchsorted(times,self.elapsed,side='right')))
                q=qs[j-1].copy();v=np.zeros(self.a.m.nv);mujoco.mj_differentiatePos(self.a.m,v,1.,q,qs[j])
                mujoco.mj_integratePos(self.a.m,q,v,float((self.elapsed-times[j-1])/(times[j]-times[j-1])))
            state={'frame':None,'waiting':True,'completed':False,'transition':True}
        else:
            _,expected=self.fk.poses(self.last_q)
            actual={name:np.r_[self.a.d.xpos[self.a.m.body(name).id],self.a.d.xquat[self.a.m.body(name).id]] for name in expected}
            ob=self.a.observe()
            next_frame=self.clock.next
            boundary=next_frame in {self.start,self.end-1,*self.clock.gates,*self.config['geometry_gates']}
            if next_frame<self.end:
                boundary=boundary or not np.array_equal(self.fingers[next_frame],self.applied_fingers)
            complete=self.terminal_sent_at is not None and self.elapsed-self.terminal_sent_at+1e-12>=self.control_dt
            state=self.clock.step(dt,dict(can_advance=bool(policy_tick or not boundary),completion_ready=complete,aligned=poses_aligned(actual,expected,self.config),stable=bool(stable),
                contacts=ob['contacts']['events'],object_bodies=self.object_bodies,object_force_N=ob['contacts']['robot_object_N']))
            q=self.references[state['frame']] if state['frame'] is not None else self.entry
        self.last_q=q.copy();self.frame=state['frame'];poses,_=self.fk.poses(q)
        fingers=self.config['open_fingers_rad'] if self.frame is None else self.fingers[self.frame]
        if policy_tick:
            self.applied_frame=self.frame;self.applied_fingers=np.asarray(fingers).copy()
            if self.frame==self.end-1 and self.terminal_sent_at is None:self.terminal_sent_at=self.elapsed
        self.audit.append(dict(time_s=float(self.a.d.time),applied_source_index=self.applied_frame,**state))
        return dict(qpos=q,poses=poses,fingers=np.asarray(fingers),**state)


def validate_operation_model(model,strategy):
    """Static names checked before launch, including contact body ownership."""
    from ..native import descendants
    c=strategy['policy']['whole_body_operation'];b=strategy['bindings']
    robot=descendants(model,model.body(b['root_body']).id)
    try:
        for name in c['alignment_bodies']:
            if model.body(name).id not in robot:raise WorkflowError('invalid_input','Alignment body outside robot: '+name)
        for gate in c['contact_gates']:
            for group in gate['groups']:
                for name in group:
                    if model.body(name).id not in robot:raise WorkflowError('invalid_input','Contact group outside robot: '+name)
    except KeyError as exc:
        raise WorkflowError('invalid_input','Unknown operation body: '+str(exc)) from None
