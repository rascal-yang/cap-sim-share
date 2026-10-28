"""G1/Dex3 + SONIC adapter. Object/support bindings are explicit model names.

Other robot families are unsupported, never interpreted as G1. No physical state
writes after initial reset. All subsequent qpos writes belong to planning copies.
"""
from pathlib import Path
import numpy as np
from .contracts import WorkflowError


def descendants(model,root):
    result=set()
    for bid in range(model.nbody):
        b=bid
        while b and b!=root:b=int(model.body_parentid[b])
        if b==root:result.add(bid)
    return result


def geom_z_bounds(m,d,g):
    import mujoco
    typ=m.geom_type[g];center=d.geom_xpos[g];rotation=d.geom_xmat[g].reshape(3,3);size=m.geom_size[g]
    if typ==mujoco.mjtGeom.mjGEOM_PLANE:return float(center[2]),float(center[2])
    if typ==mujoco.mjtGeom.mjGEOM_MESH:
        mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
        z=(m.mesh_vert[start:start+count]@rotation.T+center)[:,2];return float(z.min()),float(z.max())
    if typ==mujoco.mjtGeom.mjGEOM_BOX:extent=float(abs(rotation[2])@size)
    elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:extent=float(size[0])
    elif typ in (mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_CYLINDER):
        axis=abs(rotation[2,2]);extent=float(size[1]*axis+size[0]*(1 if typ==mujoco.mjtGeom.mjGEOM_CAPSULE else np.sqrt(max(0,1-axis*axis))))
    else:raise WorkflowError('unsupported_capability','Object/support geometry type')
    return float(center[2]-extent),float(center[2]+extent)


class NativeAdapter:
    def __init__(self,task,strategy,initial):
        from cap_sim.backends.humano import Backend
        from ..privileged import no_physics
        from ..action_replay import initialize_source
        from ..standing_handoff import StandingSonic
        from cap_sim.backends.native_sonic import Sonic
        if strategy['adapter']!='g1_sonic_native_v1':raise WorkflowError('unsupported_capability','Robot adapter')
        self.b=Backend()
        try:
            with no_physics():self.b.describe(task)
            self.m,self.d=self.b.m,self.b.d;self.strategy=strategy;self.bindings=strategy['bindings']
            if self.m.na or self.m.nmocap:raise WorkflowError('unsupported_capability','Stateful actuator/mocap model')
            self.root=self.m.body(self.bindings['root_body']).id
            rootjoint=int(self.m.body_jntadr[self.root])
            if rootjoint<0 or int(self.m.jnt_qposadr[rootjoint])!=0 or int(self.m.jnt_type[rootjoint])!=0:
                raise WorkflowError('unsupported_capability','G1 root must be first free joint')
            self.object=self.m.body(self.bindings['object']).id;self.robot_bodies=descendants(self.m,self.root)
            self.object_bodies=descendants(self.m,self.object)
            self.wrists={side:self.m.body(self.bindings['wrists'][side]).id for side in ('left','right')}
            self.hands={side:descendants(self.m,bid) for side,bid in self.wrists.items()}
            self.feet={side:descendants(self.m,self.m.body(name).id) for side,name in self.bindings['feet'].items()}
            self.support={self.m.geom(name).id for name in self.bindings['support_geoms']}
            self.ground={self.m.geom(name).id for name in self.bindings['ground_geoms']}
            self.torso=self.m.body(self.bindings['torso']).id
            if self.wrists!={'left':self.b.left,'right':self.b.right} or self.torso!=self.b.torso:
                raise WorkflowError('unsupported_capability','G1 wrist/torso binding mismatch')
            if not self.support or not self.ground or len(self.feet)!=2:
                raise WorkflowError('invalid_input','Support, ground and bilateral feet required')
            class Mode1Sonic(StandingSonic):
                navigating=True
                handoff_until_tick=-1
                def plan_legs(inner,data,velocity,height,facing_yaw=None):
                    if inner.navigating:
                        Sonic.plan_legs(inner,data,velocity,height,facing_yaw)
                        speed=float(np.linalg.norm(velocity))
                        inner.planner_records.append({'time':float(data.time),'type':'live_locomotion','planner_mode':1 if speed>1e-6 else 4,'encoder_mode':1,'velocity_world_xy_m_s':np.asarray(velocity).tolist(),'target_vel':speed if speed>1e-6 else -1,'reference':inner.ref.tolist()})
                    elif inner.tick>=inner.handoff_until_tick:super().plan_legs(data,velocity,height,facing_yaw)
                def blend_to_standing(inner,anchor,duration,measured_start=None):
                    from .motion import blend_stance_reference
                    # SONIC reference frames are 30 Hz, independent of physics.
                    count=max(100,int(np.ceil(duration*30))+31)
                    if measured_start is not None:
                        future=np.repeat(np.asarray(measured_start)[None],count,axis=0)
                    elif inner.ref is None:
                        future=np.repeat(np.r_[self.d.qpos[:7],self.d.qpos[inner.qids]][None],count,axis=0)
                    else:future=inner.sample((inner.tick-inner.ref_tick)*.6+np.arange(count))
                    inner.anchor=anchor.copy();inner.navigating=False
                    inner.ref=blend_stance_reference(future,anchor,np.arange(count)/30.,duration)
                    inner.ref_tick=inner.tick;inner.handoff_until_tick=inner.tick+int(np.ceil(duration/.02))
                    inner.planner_records.append({'time':float(self.d.time),'type':'blended_standing_reference','duration_s':duration,'reference':inner.ref.tolist()})
            self.model_files=sonic_files(task)
            self.sonic=Mode1Sonic(self.m,self.b.robot,task['resources']['sonic_models'],self.b.initial,case='anchored_legs')
            self.providers={k:getattr(self.sonic,k).get_providers() for k in ('encoder','decoder','planner')}
            if any(v!=['CPUExecutionProvider'] for v in self.providers.values()):raise WorkflowError('preflight_failed','Provider mismatch')
            initialize_source(self.m,self.d,initial)
            self.dt=float(self.m.opt.timestep);ratio=strategy['policy']['control_dt_s']/self.dt
            if ratio<1 or not np.isclose(ratio,round(ratio)):raise WorkflowError('invalid_input','Control/physics period mismatch')
            self.control_stride=round(ratio);self.last_target=None
        except Exception:
            self.b.close();raise

    def poses(self):
        return {s:np.r_[self.d.xpos[bid],self.d.xquat[bid]].copy() for s,bid in {**self.wrists,'torso':self.torso}.items()}

    def observe(self):
        import mujoco
        m,d=self.m,self.d;hand=robot_object=support=obstacle=0.;foot={s:0. for s in self.feet};events=[]
        active=set().union(*(self.hands[s] for s in self.bindings['hands']))
        for i,c in enumerate(d.contact):
            gs=[int(c.geom1),int(c.geom2)];bs=[int(m.geom_bodyid[g]) for g in gs]
            force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);f=max(0.,float(force[0]))
            has_object=any(b in self.object_bodies for b in bs);has_robot=any(b in self.robot_bodies for b in bs)
            if has_object and any(b in active for b in bs):hand+=f
            if has_object and has_robot:robot_object+=f
            if has_object and any(g in self.support or g in self.ground for g in gs):support+=f
            ground_foot=any(g in self.ground for g in gs) and any(any(b in bodies for b in bs) for bodies in self.feet.values())
            if has_robot and not all(b in self.robot_bodies for b in bs) and not has_object and not ground_foot:obstacle+=f
            for side,bodies in self.feet.items():
                if any(g in self.ground for g in gs) and any(b in bodies for b in bs):foot[side]+=f
            if f>0:events.append({'bodies':[m.body(b).name for b in bs],'geoms':[m.geom(g).name or '' for g in gs],'force_N':f,'position':c.pos.tolist()})
        objgeom=[g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.object_bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]
        if not objgeom:raise WorkflowError('unsupported_capability','No object collision geometry')
        bottom=min(geom_z_bounds(m,d,g)[0] for g in objgeom);top=max(geom_z_bounds(m,d,g)[1] for g in self.support)
        poses=self.poses()
        return {'time_s':float(d.time),'root_pose':d.qpos[:7].tolist(),'root_velocity':d.qvel[:6].tolist(),
                'right_pose':poses['right'].tolist(),'left_pose':poses['left'].tolist(),
                'object_pose':np.r_[d.xpos[self.object],d.xquat[self.object]].tolist(),'clearance_m':bottom-top,
                'foot_ground_N':list(foot.values()),'root_speed_m_s':float(np.linalg.norm(d.qvel[:2])),
                'yaw_speed_rad_s':float(abs(d.qvel[5])),
                'contacts':{'hand_object_N':hand,'robot_object_N':robot_object,'robot_table_N':obstacle,
                            'robot_obstacle_N':obstacle,'support_N':support,'events':events}}

    def native_success(self):
        task=self.b.e.task;info=self.b.e._get_info()
        task.compute_reward(info,mujoco_env=self.b.e.mujoco)
        return bool(task.check_success(info,mujoco_env=self.b.e.mujoco))

    def command(self,poses,velocity,facing,height,fingers,*,policy_tick):
        from humano.core.action import ActionCmd
        from humano.mt.base import MotorCommand
        if policy_tick:
            self.last_target=self.sonic.step(self.d,[(poses[s][:3],poses[s][3:]) for s in ('left','right')],
                    (poses['torso'][:3],poses['torso'][3:]),velocity,height,None,facing)
            if self.sonic.records[-1]['mode']!=1:raise WorkflowError('execution_error','SONIC mode deviation')
        if self.last_target is None:raise WorkflowError('execution_error','First step must be a policy tick')
        self.b.robot.apply_action(ActionCmd('wbc_torque',low_cmd=MotorCommand(q_target=self.last_target,kp=self.sonic.kp,kd=self.sonic.kd),use_sensor=False,
            left_hand_cmd=MotorCommand.hand_command(fingers[:7]),right_hand_cmd=MotorCommand.hand_command(fingers[7:])))

    def step(self):
        import mujoco
        mujoco.mj_step(self.m,self.d);mujoco.mj_forward(self.m,self.d)

    def close(self):self.b.close()


def validate_physics(model,manifest,*,sonic=None,reward_stride=None):
    if 'arrays' in manifest:
        arrays=manifest['arrays']
    else:
        required={'physics_dt_s','kp','kd','reward_stride'}
        if not required<=manifest.keys() or sonic is None or reward_stride is None:
            raise WorkflowError('preflight_failed','Legacy physics requires actual PD and reward cadence')
        if (float(model.opt.timestep)!=manifest['physics_dt_s'] or reward_stride!=manifest['reward_stride']
            or not np.array_equal(sonic.kp,manifest['kp']) or not np.array_equal(sonic.kd,manifest['kd'])):
            raise WorkflowError('preflight_failed','Legacy timestep/PD/reward cadence changed')
        arrays={key:value for key,value in manifest.items() if key not in required}
    for key,value in arrays.items():
        if not hasattr(model,key) or not np.array_equal(getattr(model,key),np.asarray(value)):
            raise WorkflowError('preflight_failed','Physics array changed '+key)
    if 'arrays' in manifest:
        options={'dt':'timestep','gravity':'gravity','integrator':'integrator','solver':'solver','iterations':'iterations',
                 'tolerance':'tolerance','cone':'cone','disableflags':'disableflags','enableflags':'enableflags'}
        for key,attribute in options.items():
            if not np.array_equal(getattr(model.opt,attribute),manifest[key]):raise WorkflowError('preflight_failed','Physics option changed '+key)


def model_digest(model):
    """Compiled model bytes bind geometry, names, topology, dynamics and options."""
    import mujoco,hashlib
    buffer=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8)
    mujoco.mj_saveModel(model,None,buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def validate_model_identity(planned,live):
    if model_digest(planned)!=model_digest(live):
        raise WorkflowError('preflight_failed','Planning/execution model mismatch')


def sonic_files(task):
    import json
    from .contracts import file_hash
    profile=json.loads((Path(__file__).resolve().parents[2]/'runtime/humano-cpu.json').read_text())
    result={}
    for relative,expected in profile['sonic']['models'].items():
        path=(Path(task['resources']['sonic_models'])/relative).resolve()
        if not path.is_file() or file_hash(path)!=expected:
            raise WorkflowError('preflight_failed','Executed SONIC model mismatch: '+relative)
        result[str(path)]=expected
    return result
