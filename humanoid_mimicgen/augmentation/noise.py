"""Explicit post-SONIC arm target perturbations, confined to augmentation jobs."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import numpy as np
from .contracts import require
from ..workflow.contracts import canonical, file_hash


def validate_noise(config):
    require(isinstance(config,dict) and set(config)=={'seed','sigma_rad','interface','resample','joint_group'},'Motion noise fields')
    require(type(config['seed']) is int and config['seed']>=0,'Noise seed')
    require(type(config['sigma_rad']) in (int,float) and np.isfinite(config['sigma_rad']) and config['sigma_rad']>0,'Noise standard deviation')
    require(config['interface']=='post_sonic_joint_target' and config['resample']=='policy_tick' and config['joint_group']=='both_arms','Supported noise interface')
    return config


class JointNoise:
    def __init__(self,indices,sigma,seed,lower,upper):
        self.indices=np.asarray(indices,int);self.sigma=sigma
        self.lower=np.asarray(lower);self.upper=np.asarray(upper)
        self.rng=np.random.Generator(np.random.PCG64(seed));self.noise=None
    def apply(self,target,policy_tick):
        if policy_tick:self.noise=self.rng.normal(0,self.sigma,len(self.indices))
        require(self.noise is not None,'Noise requires initial policy tick')
        result=np.array(target,copy=True)
        result[self.indices]=np.clip(result[self.indices]+self.noise,self.lower,self.upper)
        return result


@contextmanager
def freeze_noise(config):
    """Add the declared interface before the existing service writes/freezes its contract."""
    from ..workflow import service
    original=service.write_new
    def write(path,value):
        if Path(path).name=='experiment.json':
            value=copy.deepcopy(value)
            value['control']['motion_noise']={**config,'stages':'all including navigation and recovery','finger_and_leg_noise':False,'joint_range_clipping':True}
            value['pipeline']['entrypoint']='python -m humanoid_mimicgen.augmentation execute-case --experiment '+str(Path(path).parent)
        return original(path,value)
    service.write_new=write
    try:yield
    finally:service.write_new=original


def execute_case(experiment):
    from ..workflow import service,runner
    root=Path(experiment);c=service.check_contract(root/'experiment.json')
    declared=c['control'].get('motion_noise')
    if not declared:return service.execute(root)
    config={k:declared[k] for k in ('seed','sigma_rad','interface','resample','joint_group')};validate_noise(config)
    original=runner.NativeAdapter
    class NoisyAdapter(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            ids=list(map(int,self.b.robot.body_joint_index));names=[self.m.joint(i).name for i in ids]
            indices=[i for i,n in enumerate(names) if n.startswith(('left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist'))]
            require(len(indices)==14,'Expected 14 explicit arm joints')
            jointids=np.array(ids)[indices];limits=self.m.jnt_range[jointids]
            require(self.m.jnt_limited[jointids].all(),'Arm joint limits required')
            self.disturbance=JointNoise(indices,config['sigma_rad'],config['seed'],limits[:,0],limits[:,1])
            self.clean_target=None;self.noise_rows=[];self.noise_names=[names[i] for i in indices]
        def command(self,poses,velocity,facing,height,fingers,*,policy_tick):
            from humano.core.action import ActionCmd
            from humano.mt.base import MotorCommand
            if self.clean_target is not None:self.last_target=self.clean_target.copy()
            super().command(poses,velocity,facing,height,fingers,policy_tick=policy_tick)
            self.clean_target=self.last_target.copy();nominal_ctrl=self.d.ctrl.copy()
            self.last_target=self.disturbance.apply(self.clean_target,policy_tick)
            self.b.robot.apply_action(ActionCmd('wbc_torque',low_cmd=MotorCommand(q_target=self.last_target,kp=self.sonic.kp,kd=self.sonic.kd),use_sensor=False,
                left_hand_cmd=MotorCommand.hand_command(fingers[:7]),right_hand_cmd=MotorCommand.hand_command(fingers[7:])))
            self.noise_rows.append((float(self.d.time),self.clean_target.copy(),self.last_target.copy(),nominal_ctrl,self.d.ctrl.copy(),bool(policy_tick)))
        def close(self):
            try:
                if hasattr(self,'noise_rows'):
                    rows=self.noise_rows;out=root/'execution'
                    np.savez_compressed(out/'motion_noise.npz',time_s=np.array([x[0] for x in rows]),
                        nominal_body_target_rad=np.array([x[1] for x in rows]),executed_body_target_rad=np.array([x[2] for x in rows]),
                        nominal_ctrl_Nm=np.array([x[3] for x in rows]),executed_ctrl_Nm=np.array([x[4] for x in rows]),policy_tick=np.array([x[5] for x in rows],bool))
                    (out/'motion_noise.json').write_bytes(canonical({'config':config,'arm_joint_names':self.noise_names,'arm_target_indices':self.disturbance.indices.tolist(),'seed_algorithm':'PCG64',
                        'attempted_commands':len(rows),'resampling':'new independent Gaussian per policy tick; held between ticks; joint range clipping',
                        'labels':'nominal targets at observed disturbed state; actual control separately recorded','physics_state_resets':0,
                        'npz_sha256':file_hash(out/'motion_noise.npz')}))
            finally:super().close()
    runner.NativeAdapter=NoisyAdapter
    try:return service.execute(root)
    finally:runner.NativeAdapter=original
