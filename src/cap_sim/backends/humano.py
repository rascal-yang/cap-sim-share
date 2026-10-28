"""Native HUMANO tasks, original assets/reward checks, SONIC torque execution.
Run with the HUMANO Python environment; no modification to its checkout is needed.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import copy
import hashlib
import inspect
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Slerp
from ..io import read,write,digest
from .native_sonic import Sonic,rot,quat


def reward_stride(reward_dt, physics_dt):
    ratio=float(reward_dt)/float(physics_dt)
    if not np.isfinite(ratio) or ratio<1 or abs(ratio-round(ratio))>1e-7:
        raise ValueError('Reward cadence must be an integer physics interval')
    return round(ratio)


class Backend:
    interface={'backend':'humano','stages':['approach','contact','lift','hold','walk','kick','sit','push','press','recover'],
      'action':{'duration_s':'0.1..8, 5ms aligned','velocity_xy':'world m/s, norm <= .5','facing_yaw':'optional world yaw radians','pelvis_height_m':'.4..85','left_position':'world wrist xyz','right_position':'world wrist xyz','left_quaternion_wxyz':'unit quaternion','right_quaternion_wxyz':'unit quaternion','left_fingers':'7 joint radians','right_fingers':'7 joint radians','torso_quaternion_wxyz':'unit quaternion','body_keyframes':'optional rows [time_s, root_xyz, root_wxyz, 29 body joints]; SONIC reference, not physical state writes'},
      'gates':'lift requires sustained measured hand contact; stop on fall or unexpected table collision; native task acceptance frozen'}

    def _prepare(self,task):
        if hasattr(self,'task') and self.task==task:return
        import mujoco
        from humano.cli.render_sonic_isaac import _make_env,_reset_episode
        if hasattr(self,'env'):self.env.close()
        self.mj=mujoco;self.task=copy.deepcopy(task);resources=task['resources'];self.prepared=Path(resources['prepared'])
        self.env=_make_env('humano/'+task['name']+'-v0','mujoco',True,200,None,'dex3');self.e=self.env.unwrapped
        config=read(self.prepared/'environment_config.json');_reset_episode(self.env,config);self.m=self.e.mjModel;self.d=self.e.mjData;self.robot=self.e.task.robot
        self.reward_stride=reward_stride(config['metadata']['reward_dt'],self.m.opt.timestep)
        if abs(self.m.opt.timestep-.005)>1e-9:raise ValueError('Backend requires 5ms physics')
        self.object_ids={k:int(v.id) for k,v in self.e.mujoco.mj_objects.items()}
        root=getattr(self.e.mujoco,'articulated_object_root_body_name',None)
        if root:self.object_ids['articulated']=int(self.m.body(root).id)
        self.initial=np.load(resources['initial_state']);self.initial_q=self.initial['qpos'].copy();self.initial_v=self.initial['qvel'].copy()
        if self.m.nq!=len(self.initial_q):raise ValueError('Dataset/model nq mismatch')
        shift=np.asarray(task['scene'].get('translation',[0,0,0]),float)
        if shift.shape!=(3,) or not np.isfinite(shift).all() or np.linalg.norm(shift)>.2:raise ValueError('Unsupported variant translation')
        # Move task object at reset, not during execution. Articulated button has fixed root.
        if np.linalg.norm(shift):
            bid=self.object_ids.get('target',self.object_ids.get('articulated'))
            if bid is None:raise ValueError('Task object missing')
            jid=int(self.m.body_jntadr[bid]);typ=self.m.jnt_type[jid] if jid>=0 else -1
            if typ==mujoco.mjtJoint.mjJNT_FREE:self.initial_q[int(self.m.jnt_qposadr[jid]):int(self.m.jnt_qposadr[jid])+3]+=shift
            else:self.m.body_pos[bid]+=shift
        self.qids=self.m.jnt_qposadr[self.robot.body_joint_index];self.vids=self.m.jnt_dofadr[self.robot.body_joint_index]
        self.left=self.m.body('left_wrist_yaw_link').id;self.right=self.m.body('right_wrist_yaw_link').id;self.torso=self.m.body('torso_link').id
        self._restore()
        self.model_xml=self.e.mjSpec.to_xml()
        self.sonic=None

    def _restore(self):
        self.mj.mj_resetData(self.m,self.d);self.d.qpos[:]=self.initial_q;self.d.qvel[:]=self.initial_v;self.d.ctrl[:]=self.initial['ctrl'];self.d.time=0
        self.mj.mj_forward(self.m,self.d)
        native=self.e.task
        for key in vars(native):
            if key.startswith('_init_'):setattr(native,key,None)
            if key.endswith('_hold_count'):setattr(native,key,0)
        # Anchor native metrics to actual reset geometry, before the first action.
        info=self.e._get_info();native.compute_reward(info,mujoco_env=self.e.mujoco)
        for key in vars(native):
            if key.endswith('_hold_count'):setattr(native,key,0)
        self.states=[];self.samples=[];self.contact_records=[];self.hand_contact_s=0.;self.success=False;self.ticks=0;self.native_reward=0.;self.native_passed=False
        ob=self.robot.prepare_obs();self.fingers=[np.asarray(ob['left_hand_q']).copy(),np.asarray(ob['right_hand_q']).copy()]
        self.initial_object={k:self.d.xpos[v].copy() for k,v in self.object_ids.items()}
        self.target=self.d.qpos[self.qids].copy()

    def describe(self,task):
        self._prepare(task);self._restore();return self.observe()

    def reset(self,task,out):
        self._prepare(task);self._restore();self.out=Path(out)
        self.sonic=Sonic(self.m,self.robot,task['resources']['sonic_models'],self.initial)
        # Store compiled model, including fixed-body variant placement.
        tree=ET.fromstring(self.model_xml)
        for element in tree.findall('.//body'):
            if element.get('name'):
                element.set('pos',' '.join(map(str,self.m.body(element.get('name')).pos)))
        ET.ElementTree(tree).write(self.out/'scene_source.xml')
        self.mj.mj_saveModel(self.m,str(self.out/'scene.mjb'),None)
        source=Path(inspect.getfile(type(self.e.task)))
        if hashlib.sha256(source.read_bytes()).hexdigest()!=task['acceptance']['validator_id']:raise RuntimeError('Native validator changed')
        frozen={'task_source':hashlib.sha256(source.read_bytes()).hexdigest(),'config':digest(read(self.prepared/'environment_config.json')),'initial':hashlib.sha256(Path(task['resources']['initial_state']).read_bytes()).hexdigest(),'model':hashlib.sha256((self.out/'scene.mjb').read_bytes()).hexdigest()}
        for p in Path(task['resources']['sonic_models']).glob('policy/release/*.onnx'):frozen[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        p=Path(task['resources']['sonic_models'])/'planner/target_vel/V2/planner_sonic.onnx';frozen[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        source_root=source.parents[2]
        for p in sorted(source_root.rglob('*.py')):frozen[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).resolve().parents[1].rglob('*.py'):frozen[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        frozen['reward_stride']=self.reward_stride
        for e in tree.findall('.//mesh'):
            if e.get('file'):
                p=Path(e.get('file'))
                if p.exists():frozen[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        path=self.out.parents[1]/'native_resources.json'
        if path.exists() and read(path)!=frozen:raise RuntimeError('Native resources changed across attempts')
        write(path,frozen);return self.observe()

    def contacts(self):
        totals={'hand_object_N':0.,'robot_table_N':0.,'robot_object_N':0.,'support_N':0.,'max_penetration_m':0.};events=[]
        objids={v for k,v in self.object_ids.items() if k!='table'}
        def ancestor(b):
            while b>0:
                if b in objids:return True
                b=int(self.m.body_parentid[b])
            return False
        for i,c in enumerate(self.d.contact):
            bodies=[int(self.m.geom_bodyid[c.geom1]),int(self.m.geom_bodyid[c.geom2])];names=[self.m.body(b).name for b in bodies];gn=[self.m.geom(int(g)).name or '' for g in (c.geom1,c.geom2)]
            f=np.zeros(6);self.mj.mj_contactForce(self.m,self.d,i,f);force=max(0.,float(f[0]));obj=any(ancestor(b) for b in bodies);hand=any('hand' in n for n in names);robot=any(('link' in n or 'hand' in n) and not ancestor(b) for n,b in zip(names,bodies));support=any('floor' in n or 'ground' in n or 'table' in n for n in names+gn)
            if obj and hand:totals['hand_object_N']+=force
            if obj and robot:totals['robot_object_N']+=force
            if obj and support:totals['support_N']+=force
            if not obj and robot and any('table' in n for n in names+gn):totals['robot_table_N']+=force
            totals['max_penetration_m']=max(totals['max_penetration_m'],-float(c.dist))
            if force>.05 and (obj or (robot and support)):events.append({'bodies':names,'geoms':gn,'force_N':force,'position':c.pos.tolist()})
        return {**totals,'events':events}

    def joint_info(self, ids):
        return [{'name':self.m.joint(int(j)).name,'range':[float(x) if np.isfinite(x) else None for x in self.m.jnt_range[j]],'limited':bool(self.m.jnt_limited[j]),
                 'qpos':self.d.qpos[int(self.m.jnt_qposadr[j]):int(self.m.jnt_qposadr[j])+(7 if int(self.m.jnt_type[j])==0 else 1)].tolist()} for j in ids]

    def object_details(self):
        result={}
        for key,root in self.object_ids.items():
            children=[]
            for bid in range(1,self.m.nbody):
                b=bid
                while b and b!=root:b=int(self.m.body_parentid[b])
                if b==root:children.append(bid)
            js=[j for j in range(self.m.njnt) if self.m.jnt_bodyid[j] in children]
            result[key]={'root_body':self.m.body(root).name,'bodies':{self.m.body(b).name:np.r_[self.d.xpos[b],self.d.xquat[b]].tolist() for b in children},'joints':self.joint_info(js)}
        return result

    def observe(self):
        self.mj.mj_forward(self.m,self.d)
        return {'time':float(self.d.time),'root_pose':self.d.qpos[:7].tolist(),'root_velocity':self.d.qvel[:6].tolist(),'body_joints':self.d.qpos[self.qids].tolist(),'body_joint_names':[self.m.joint(int(j)).name for j in self.robot.body_joint_index],
          'left_position':self.d.xpos[self.left].tolist(),'right_position':self.d.xpos[self.right].tolist(),'wrist_position':self.d.xpos[self.right].tolist(),
          'left_quaternion_wxyz':self.d.xquat[self.left].tolist(),'right_quaternion_wxyz':self.d.xquat[self.right].tolist(),'torso_position':self.d.xpos[self.torso].tolist(),'torso_quaternion_wxyz':self.d.xquat[self.torso].tolist(),
          'fingers':[x.tolist() for x in self.fingers],'joint_interface':{'body':self.joint_info(self.robot.body_joint_index),'left_hand':self.joint_info(self.robot.left_hand_actuated_index),'right_hand':self.joint_info(self.robot.right_hand_actuated_index)},'objects':{k:np.r_[self.d.xpos[v],self.d.xquat[v]].tolist() for k,v in self.object_ids.items()},'object_details':self.object_details(),'contacts':self.contacts(),'contact_s':self.hand_contact_s}

    def execute(self,a,remaining):
        from humano.mt.base import MotorCommand
        from humano.core.action import ActionCmd
        allowed={'stage','duration_s','velocity_xy','facing_yaw','pelvis_height_m','left_position','right_position','left_quaternion_wxyz','right_quaternion_wxyz','left_fingers','right_fingers','torso_quaternion_wxyz','body_keyframes','name'}
        if set(a)-allowed:raise ValueError('Unknown native action channel')
        duration=float(a['duration_s']);stage=a['stage']
        if stage not in self.interface['stages'] or not .1<=duration<=8 or abs(duration/.005-round(duration/.005))>1e-7:raise ValueError('Invalid stage/duration')
        if stage=='lift' and self.hand_contact_s<.12:return {'completed':False,'reusable':False,'failure':'hand_contact_gate','samples':[],'observation':self.observe()}
        velocity=np.asarray(a.get('velocity_xy',[0,0]),float);height=float(a.get('pelvis_height_m',.74))
        if velocity.shape!=(2,) or not np.isfinite(velocity).all() or np.linalg.norm(velocity)>.5 or not .4<=height<=.85:raise ValueError('Invalid locomotion target')
        p0=[self.d.xpos[b].copy() for b in (self.left,self.right)];q0=[self.d.xquat[b].copy() for b in (self.left,self.right)];f0=[f.copy() for f in self.fingers]
        p1=[];q1=[];f1=[]
        for i,side in enumerate(('left','right')):
            p=np.asarray(a.get(side+'_position',p0[i]),float);q=np.asarray(a.get(side+'_quaternion_wxyz',q0[i]),float);f=np.asarray(a.get(side+'_fingers',f0[i]),float)
            if p.shape!=(3,) or q.shape!=(4,) or f.shape!=(7,) or not np.isfinite(np.r_[p,q,f]).all() or abs(np.linalg.norm(q)-1)>.001:raise ValueError('Invalid hand target')
            if np.linalg.norm(p-self.d.qpos[:3])>1.5:raise ValueError('Wrist target outside 1.5 m workspace')
            js=getattr(self.robot,side+'_hand_actuated_index');lo=self.m.jnt_range[js,0];hi=self.m.jnt_range[js,1]
            if np.any(f<lo-.001) or np.any(f>hi+.001):raise ValueError('Hand joint target outside limits')
            p1.append(p);q1.append(q);f1.append(f)
        tq=np.asarray(a.get('torso_quaternion_wxyz',self.d.xquat[self.torso]),float)
        if tq.shape!=(4,) or not np.isfinite(tq).all() or abs(np.linalg.norm(tq)-1)>.001:raise ValueError('Bad torso quaternion')
        sl=[Slerp([0,1],rot(np.array([q0[i],q1[i]]))) for i in range(2)]
        body=None
        if 'body_keyframes' in a:
            body=np.asarray(a['body_keyframes'],float)
            if body.ndim!=2 or body.shape[1]!=37 or not 2<=len(body)<=200 or not np.isfinite(body).all() or body[0,0]!=0 or np.any(np.diff(body[:,0])<=0):raise ValueError('Bad body keyframes')
            if np.max(np.abs(np.linalg.norm(body[:,4:8],axis=1)-1))>.001:raise ValueError('Bad keyframe orientation')
            if np.any(body[:,8:]<self.m.jnt_range[self.robot.body_joint_index,0]-.001) or np.any(body[:,8:]>self.m.jnt_range[self.robot.body_joint_index,1]+.001):raise ValueError('Body reference outside joint limits')
            sb=Slerp(body[:,0],rot(body[:,4:8]))
            def at(ts):
                ts=np.clip(ts,body[0,0],body[-1,0]);v=np.stack([np.interp(ts,body[:,0],body[:,j]) for j in range(1,37)],axis=-1);v[:,3:7]=quat(sb(ts));return v
        root_start=self.d.qpos[:3].copy()
        if self.task.get('policy_checks',{}).get('sonic_mode')==1 and body is not None:
            raise ValueError('mode1 policy rejects fullbody reference')
        if a.get('facing_yaw') is not None and not np.isfinite(a['facing_yaw']):raise ValueError('Invalid facing yaw')
        rows=[];failure=None;n=round(duration/.005);steps=min(n,int((remaining+1e-9)/.005));bad=0.;lost=0.
        record_start=len(self.sonic.records);contact_stats={};peak_table=peak_hand=0.;opposed=opposed_max=0.
        target_body=self.m.body(self.object_ids['target']).name if 'target' in self.object_ids else None
        for i in range(steps):
            u=(i+1)/n;alpha=u*u*(3-2*u);wrists=[(p0[j]+(p1[j]-p0[j])*alpha,quat(sl[j](alpha))) for j in range(2)]
            if stage in ('walk','recover'):
                for j,side in enumerate(('left','right')):
                    if side+'_position' not in a:wrists[j]=(wrists[j][0]+self.d.qpos[:3]-root_start,wrists[j][1])
            self.fingers=[f0[j]+(f1[j]-f0[j])*alpha for j in range(2)]
            if self.ticks%4==0:
                future=None if body is None else (at(i*.005+np.arange(10)*.1),at(i*.005+np.arange(10)*.1+.01))
                self.target=self.sonic.step(self.d,wrists,(self.d.xpos[self.torso],tq),velocity,height,future,a.get('facing_yaw'))
            command=MotorCommand(q_target=self.target,kp=self.sonic.kp,kd=self.sonic.kd)
            self.robot.apply_action(ActionCmd('wbc_torque',low_cmd=command,use_sensor=False,left_hand_cmd=MotorCommand.hand_command(self.fingers[0]),right_hand_cmd=MotorCommand.hand_command(self.fingers[1])))
            self.mj.mj_step(self.m,self.d);self.mj.mj_forward(self.m,self.d);self.ticks+=1
            if self.ticks%self.reward_stride==0:
                info=self.e._get_info();self.native_reward=float(self.e.task.compute_reward(info,mujoco_env=self.e.mujoco));self.native_passed=bool(self.e.task.check_success(info,mujoco_env=self.e.mujoco))
            reward=self.native_reward;passed=self.native_passed
            c=self.contacts();self.hand_contact_s=self.hand_contact_s+.005 if c['hand_object_N']>.1 else 0.;lost=0 if c['hand_object_N']>.1 else lost+.005
            peak_table=max(peak_table,c['robot_table_N']);peak_hand=max(peak_hand,c['hand_object_N'])
            link_forces={}
            for event in c['events']:
                if target_body and target_body in event['bodies']:
                    for name in event['bodies']:
                        if 'hand' in name:link_forces[name]=link_forces.get(name,0.)+event['force_N']
            for name in set(contact_stats)|set(link_forces):
                item=contact_stats.setdefault(name,{'peak_force_N':0.,'continuous_s':0.,'longest_s':0.})
                force=link_forces.get(name,0.);item['peak_force_N']=max(item['peak_force_N'],force)
                item['continuous_s']=item['continuous_s']+.005 if force>.1 else 0.
                item['longest_s']=max(item['longest_s'],item['continuous_s'])
            both=all(link_forces.get(name,0.)>.1 for name in ('right_hand_thumb_2_link','right_hand_index_1_link'))
            opposed=opposed+.005 if both else 0.;opposed_max=max(opposed_max,opposed)
            tilt=np.arccos(np.clip(rot(self.d.qpos[3:7]).as_matrix()[2,2],-1,1));stable=bool(np.isfinite(self.d.qpos).all() and self.d.qpos[2]>.30 and tilt<np.deg2rad(50))
            row={'time':float(self.d.time),'dt':.005,'native_success':passed,'stable':stable,'native_reward':reward}
            rows.append(row);self.samples.append(row);self.states.append(np.r_[self.d.time,self.d.qpos,self.d.qvel,self.d.ctrl]);self.contact_records.append(c)
            self.success=passed and stable
            bad=bad+.005 if c['robot_table_N']>5 else 0.
            if self.success:break
            if not stable or any(w.number for w in self.d.warning):failure='robot_fall_or_simulation_warning';break
            if bad>=.05:failure='unexpected_table_contact';break
            if stage=='lift' and lost>=.03:failure='contact_lost';break
        completed=len(rows)==n
        if not completed and not failure and not self.success:failure='execution_deadline'
        records=self.sonic.records[record_start:]
        controller={'actual_modes':sorted({r['mode'] for r in records}),
                    'control_steps':len(records),'last_planner_reference_root_height_m':records[-1]['planner_reference_root_height_m'] if records else None,
                    'final_joint_target_error_rad':(self.target-self.d.qpos[self.qids]).tolist(),
                    'peak_robot_table_N':peak_table,'peak_hand_object_N':peak_hand,
                    'per_link_target_contacts':contact_stats,'longest_thumb_index_contact_s':opposed_max,
                    'contact_scope':'Target-root-body contacts; stage-local durations, not force closure.'}
        return {'completed':completed,'reusable':completed and not failure,'failure':failure,'samples':rows,'observation':self.observe(),'controller_diagnostics':controller}

    def export(self,out):
        np.savez_compressed(Path(out)/'states.npz',samples=np.asarray(self.states));write(Path(out)/'contacts.json',self.contact_records);write(Path(out)/'controller.json',self.sonic.records)

    def metrics(self):
        if len(self.states)<2:return {'roughness':0.,'root_omega_rms':0.,'peak_table_force':0.}
        s=np.asarray(self.states);v=s[:,1+self.m.nq:1+self.m.nq+self.m.nv];acc=np.diff(v[:,self.vids],axis=0)/.005
        return {'roughness':float(np.sqrt(np.mean(acc*acc))),'root_omega_rms':float(np.sqrt(np.mean(np.sum(v[:,3:6]**2,axis=1)))),'peak_table_force':max((c['robot_table_N'] for c in self.contact_records),default=0.)}

    def close(self):
        if hasattr(self,'env'):self.env.close()

    def verify(self,out):
        """Independent re-evaluation of saved measured states, without stepping physics."""
        states=np.load(Path(out)/'states.npz')['samples'];native=self.e.task
        self.d.qpos[:]=self.initial_q;self.d.qvel[:]=self.initial_v;self.d.ctrl[:]=self.initial['ctrl'];self.mj.mj_forward(self.m,self.d)
        for key in vars(native):
            if key.startswith('_init_'):setattr(native,key,None)
            if key.endswith('_hold_count'):setattr(native,key,0)
        native.compute_reward(self.e._get_info(),mujoco_env=self.e.mujoco)
        for key in vars(native):
            if key.endswith('_hold_count'):setattr(native,key,0)
        successes=[];passed=False
        for tick,row in enumerate(states,1):
            self.d.time=float(row[0]);self.d.qpos[:]=row[1:1+self.m.nq];self.d.qvel[:]=row[1+self.m.nq:1+self.m.nq+self.m.nv];self.d.ctrl[:]=row[1+self.m.nq+self.m.nv:]
            self.mj.mj_forward(self.m,self.d)
            if tick%self.reward_stride==0:
                info=self.e._get_info();native.compute_reward(info,mujoco_env=self.e.mujoco);passed=bool(native.check_success(info,mujoco_env=self.e.mujoco))
            stable=self.d.qpos[2]>.30 and rot(self.d.qpos[3:7]).as_matrix()[2,2]>np.cos(np.deg2rad(50))
            if passed and stable:successes.append(float(row[0]))
        return {'success':bool(successes),'first_success_s':successes[0] if successes else None,'states_checked':len(states),'physics_steps':0,'method':'Native checker on restored measured qpos/qvel/ctrl and recomputed contacts'}
