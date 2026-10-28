"""Right wrist reference correction BEFORE SONIC encoding; never changes decoder outputs."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

FIELDS={'reference_blend','translation_m','rotation_vector_rad','ramp_s','force_band_N','force_gain_m_per_N_s','max_feedback_m','feedback_rate_m_s'}

def validate_parameters(c):
    if not isinstance(c,dict) or set(c)!=FIELDS:raise ValueError('Unexpected encoder-wrist fields')
    for key,limit in [('translation_m',.05),('rotation_vector_rad',.35),('force_gain_m_per_N_s',.005)]:
        a=np.asarray(c[key],float)
        if a.shape!=(3,) or not np.isfinite(a).all() or np.linalg.norm(a)>limit+1e-12:raise ValueError('Invalid '+key)
    for key,lo,hi in [('reference_blend',0,1),('ramp_s',.1,3),('max_feedback_m',0,.05),('feedback_rate_m_s',.001,.05)]:
        if type(c[key]) not in (int,float) or not np.isfinite(c[key]) or not lo<=c[key]<=hi:raise ValueError('Invalid '+key)
    f=np.asarray(c['force_band_N'],float)
    if f.shape!=(2,) or not np.isfinite(f).all() or not 0<=f[0]<f[1]<=50:raise ValueError('Invalid force_band_N')
    return c

def clip_norm(a,limit):
    a=np.asarray(a,float);n=np.linalg.norm(a)
    return a*(min(1.,limit/n) if n else 1.)

def integrate_force(value,force,c):
    lo,hi=c['force_band_N'];error=lo-force if force<lo else hi-force if force>hi else 0.
    rate=clip_norm(np.asarray(c['force_gain_m_per_N_s'])*error,c['feedback_rate_m_s'])
    return clip_norm(np.asarray(value)+rate*.02,c['max_feedback_m'])

def transform_right(poses,commanded,c,weight,feedback):
    out=np.array(poses,copy=True)
    if weight==0:return out
    blend=weight*c['reference_blend'];out[1,:3]+=(np.asarray(commanded[:3])-out[1,:3])*blend+weight*(np.asarray(c['translation_m'])+feedback)
    current=R.from_quat(np.roll(out[1,3:],-1));goal=R.from_quat(np.roll(commanded[3:],-1))
    mixed=R.from_rotvec((goal*current.inv()).as_rotvec()*blend)*current
    mixed=R.from_rotvec(np.asarray(c['rotation_vector_rad'])*weight)*mixed
    out[1,3:]=np.roll(mixed.as_quat(),1)
    return out

class EncoderWrist:
    def __init__(self,b,s,params):
        import mujoco
        from .saved_token_replay import verify_source_contract
        self.b=b;self.c=validate_parameters(params['preencoder']);self.start=params['contact_start'];self.feedback=np.zeros(3);self.logs=[]
        if not hasattr(b.e.task,'_button_metrics'):raise ValueError('This force sensor adapter requires native button metrics')
        self.source=Path(params['reference_case']);verify_source_contract(self.source)
        result=json.loads((self.source/'result.json').read_text())
        if result['mode']!='original_token' or not result['success'] or result['episode']!=params['episode']:raise ValueError('Successful original-token reference required')
        trace=np.load(self.source/'execution/trajectory.npz');d=mujoco.MjData(b.m)
        names=[b.m.joint(int(j)).name for j in b.robot.body_joint_index];indices=[i for i,n in enumerate(names) if n.startswith('right_') and any(x in n for x in ('shoulder','elbow','wrist'))]
        if len(indices)!=7:raise ValueError('Expected seven right-arm joints')
        ref=[]
        for i in range(0,len(trace['state']),4):
            d.qpos[:]=trace['state'][i,1:1+b.m.nq];d.qpos[s.qids[indices]]=trace['commands'][i,1:30][indices];mujoco.mj_kinematics(b.m,d)
            ref.append(np.r_[d.xpos[b.right],d.xquat[b.right]])
        self.commanded=np.asarray(ref)
        self.evidence=[self.source/n for n in ['result.json','experiment.json','experiment.sha256','execution/trajectory.npz']]
    def apply(self,poses,index):
        force=float(self.b.e.task._button_metrics(mujoco_env=self.b.e.mujoco)['button_downward_force_n'])
        weight=float(np.clip((index-self.start)*.02/self.c['ramp_s'],0,1))
        if index>=self.start:self.feedback=integrate_force(self.feedback,force,self.c)
        ref=self.commanded[min(index,len(self.commanded)-1)]
        out=transform_right(poses,ref,self.c,weight,self.feedback)
        self.logs.append({'frame':index,'weight':weight,'downward_force_N':force,'feedback_world_m':self.feedback.tolist(),'right_before':poses[1].tolist(),'commanded_reference':ref.tolist(),'right_after':out[1].tolist(),'left_unchanged':bool(np.array_equal(out[0],poses[0])),'torso_unchanged':bool(np.array_equal(out[2],poses[2]))})
        return out
