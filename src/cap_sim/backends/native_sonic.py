"""CPU SONIC adapter. Same published encoder/decoder and torque gains in both groups.
Encoder field layout is from gear_sonic_deploy policy/release/observation_config.yaml.
Mode is a scalar followed by three zeros (not a one-hot vector).
"""
from collections import deque
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def rot(q):return R.from_quat(np.roll(q,-1,axis=-1))
def quat(r):return np.roll(r.as_quat(),1,axis=-1)


class Sonic:
    def __init__(self,model,robot,models,initial):
        import onnxruntime as ort
        from humano.mt.sonic import policy_parameters as p
        self.m=model;self.robot=robot;self.qids=model.jnt_qposadr[robot.body_joint_index];self.vids=model.jnt_dofadr[robot.body_joint_index]
        self.default=p.default_angles;self.scale=p.g1_action_scale;self.order=p.mujoco_to_isaaclab;self.reverse=p.isaaclab_to_mujoco;self.kp=p.kps;self.kd=p.kds
        opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
        def session(name):return ort.InferenceSession(str(Path(models)/name),sess_options=opt,providers=['CPUExecutionProvider'])
        self.encoder=session('policy/release/model_encoder.onnx');self.decoder=session('policy/release/model_decoder.onnx');self.planner=session('planner/target_vel/V2/planner_sonic.onnx')
        self.history=deque(maxlen=10);self.last=initial['last_action'].copy()
        for i in range(int(initial['history_size'])):
            self.history.append({'omega':initial['history_base_ang_vel'][i], 'q':initial['history_body_q'][i], 'dq':initial['history_body_dq'][i], 'last':initial['history_last_action'][i], 'gravity':rot(initial['history_base_quat'][i]).inv().apply([0,0,-1])})
        self.tick=0;self.ref=None;self.ref_tick=0;self.serial=0;self.records=[]

    def sample(self,frames):
        f=np.clip(frames,0,len(self.ref)-1);v=np.stack([np.interp(f,np.arange(len(self.ref)),self.ref[:,j]) for j in range(36)],axis=-1)
        v[:,3:7]=quat(Slerp(np.arange(len(self.ref)),rot(self.ref[:,3:7]))(f));return v

    def plan_legs(self,d,velocity,height,facing_yaw=None):
        if self.ref is None:context=np.repeat(np.r_[d.qpos[:7],d.qpos[self.qids]][None],4,axis=0)
        else:
            context=self.sample((self.tick-self.ref_tick)*.6+np.arange(4));context[:,:2]+=d.qpos[:2]-context[0,:2]
        speed=float(np.linalg.norm(velocity));direction=[*(np.asarray(velocity)/speed if speed>1e-6 else [1,0]),0]
        yaw=rot(d.qpos[3:7]).as_euler('xyz')[2] if facing_yaw is None else facing_yaw;face=[np.cos(yaw),np.sin(yaw),0]
        feed={'context_mujoco_qpos':context[None].astype(np.float32),'target_vel':np.array([speed if speed>1e-6 else -1],np.float32),'mode':np.array([1 if speed>1e-6 else 4],np.int64),'movement_direction':np.array([direction],np.float32),'facing_direction':np.array([face],np.float32),'height':np.array([height],np.float32),'random_seed':np.array([17+self.serial],np.int64),'has_specific_target':np.zeros((1,1),np.int64),'specific_target_positions':np.zeros((1,4,3),np.float32),'specific_target_headings':np.zeros((1,4),np.float32),'allowed_pred_num_tokens':np.array([[0]*10+[1]],np.int64)}
        raw,n=self.planner.run(None,feed);self.ref=raw[0,:int(n.item())].copy();self.ref_tick=self.tick;self.serial+=1

    def step(self,d,wrists,torso,velocity,height,body_future=None,facing_yaw=None):
        e=np.zeros(1762,np.float32)
        actual_root=rot(d.qpos[3:7])
        if body_future is not None:
            ref,future=body_future;e[0]=0
            e[4:294]=ref[:,7:36][:,self.order].reshape(-1)
            e[294:584]=((future[:,7:36]-ref[:,7:36])/.01)[:,self.order].reshape(-1)
            e[584:594]=ref[:,2];e[594]=ref[0,2]
            e[595:601]=(actual_root.inv()*rot(ref[0,3:7])).as_matrix()[:,:2].reshape(-1)
            e[601:661]=(actual_root.inv()*rot(ref[:,3:7])).as_matrix()[:,:,:2].reshape(-1)
        else:
            if self.ref is None or self.tick%25==0:self.plan_legs(d,velocity,height,facing_yaw)
            ref=self.sample((self.tick-self.ref_tick)*.6+np.arange(10)*3);future=self.sample((self.tick-self.ref_tick)*.6+np.arange(10)*3+.3)
            anchor=rot(ref[0,3:7]);e[0]=1;e[595:601]=(actual_root.inv()*anchor).as_matrix()[:,:2].reshape(-1)
            e[661:781]=ref[:,7:19].reshape(-1);e[781:901]=((future[:,7:19]-ref[:,7:19])/.01).reshape(-1)
            ps=[];qs=[]
            for (p,q),offset in zip([*wrists,torso],[[.18,-.025,0],[.18,.025,0],[0,0,.35]]):
                rr=rot(q);ps.extend(anchor.inv().apply(np.asarray(p)+rr.apply(offset)-d.qpos[:3]));qs.extend(quat(anchor.inv()*rr))
            e[901:910]=ps;e[910:922]=qs
        token=self.encoder.run(None,{self.encoder.get_inputs()[0].name:e[None]})[0][0]
        self.history.append({'omega':d.qvel[3:6].copy(),'q':(d.qpos[self.qids]-self.default)[self.order],'dq':d.qvel[self.vids][self.order].copy(),'last':self.last.copy(),'gravity':actual_root.inv().apply([0,0,-1])})
        if len(self.history)!=10:raise ValueError('SONIC needs ten history frames')
        obs=np.concatenate([token]+[np.concatenate([h[k] for h in self.history]) for k in ['omega','q','dq','last','gravity']]).astype(np.float32)
        raw=self.decoder.run(None,{self.decoder.get_inputs()[0].name:obs[None]})[0][0];self.last=raw.copy();self.tick+=1
        target=self.default+self.scale*raw[self.reverse]
        if not np.isfinite(target).all():raise RuntimeError('Nonfinite SONIC target')
        self.records.append({'time':float(d.time),'mode':int(e[0]),
                             'planner_reference_root_height_m':float(ref[0,2]),
                             'requested_pelvis_height_m':float(height) if body_future is None else None,
                             'measured_root_height_m':float(d.qpos[2]),
                             'token':token.tolist(),'encoder':e.tolist(),'target':target.tolist()})
        return target
