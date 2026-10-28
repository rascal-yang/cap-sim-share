"""Explicit post-SONIC joint residuals and bounded contact-force admittance.

No state writes, gain mutation, model calls or automatic controller fallback.
Joint reference replacement is declared and its commands are exported separately
from the SONIC token. Tokens alone cannot reproduce these executions.
"""
import math
from pathlib import Path
import numpy as np
from .agent import save
from ..workflow.contracts import file_hash

CONTROL_FIELDS={'contact_bodies','force_range_N','ramp_s','joints','max_offset_rad','max_rate_rad_s','force_stop_N'}
JOINT_FIELDS={'name','replay_blend','offset_rad','kp_scale','kd_scale','force_gain_rad_per_N_s'}


def finite(x):
    return type(x) in (int,float) and math.isfinite(x)


def validate_contact_control(c,bodies,joints):
    if set(c)!=CONTROL_FIELDS:raise ValueError('Invalid contact control fields')
    if not c['contact_bodies'] or len(set(c['contact_bodies']))!=len(c['contact_bodies']) or not set(c['contact_bodies'])<=set(bodies):raise ValueError('Unknown/duplicate contact body')
    f=c['force_range_N']
    if len(f)!=2 or not all(finite(x) for x in f) or not 0<=f[0]<f[1]<=1000:raise ValueError('Invalid force interval')
    for key,lo,hi in [('ramp_s',.1,3),('max_offset_rad',.001,.35),('max_rate_rad_s',.001,.5),('force_stop_N',f[1],1500)]:
        if not finite(c[key]) or not lo<=c[key]<=hi:raise ValueError(f'Invalid {key}: expected finite number in [{lo}, {hi}]')
    seen=set()
    for j in c['joints']:
        if set(j)!=JOINT_FIELDS or j['name'] not in joints or j['name'] in seen:raise ValueError('Unknown/duplicate joint')
        seen.add(j['name'])
        for key,lo,hi in [('replay_blend',0,1),('offset_rad',-c['max_offset_rad'],c['max_offset_rad']),('kp_scale',.5,1.5),('kd_scale',.5,2),('force_gain_rad_per_N_s',-.02,.02)]:
            if not finite(j[key]) or not lo<=j[key]<=hi:raise ValueError('Invalid joint '+key)
    return c


def bounded_servo(value,force,band,gain,dt,rate,limit):
    error=band[0]-force if force<band[0] else band[1]-force if force>band[1] else 0.
    return float(np.clip(value+np.clip(gain*error,-rate,rate)*dt,-limit,limit))


def blend_targets(target,reference,indices,blend,weight):
    out=np.array(target,copy=True);out[indices]+=weight*blend*(np.asarray(reference)[indices]-out[indices]);return out


def contact_force(model,data,body_names,target_body):
    import mujoco
    selected={model.body(n).id for n in body_names}
    def child(body,parent):
        while body>0:
            if body==parent:return True
            body=int(model.body_parentid[body])
        return False
    force=0.;pairs=[]
    for i,c in enumerate(data.contact):
        a,z=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
        if not ((a in selected and child(z,target_body)) or (z in selected and child(a,target_body))):continue
        f=np.zeros(6);mujoco.mj_contactForce(model,data,i,f)
        normal=max(0.,float(f[0]));force+=normal
        pairs.append({'body':model.body(a if a in selected else z).name,'normal_force_N':normal,'distance_m':float(c.dist)})
    return force,pairs


class ContactController:
    def __init__(self,b,s,params,source_frames):
        import json
        from .quality_suite import verify_origin
        self.b=b;self.s=s;self.c=params['contact_control'];self.start=params['contact_start'];self.logs=[];self.servo=np.zeros(len(s.qids));self.previous=None
        self.names=[b.m.joint(int(j)).name for j in b.robot.body_joint_index]
        validate_contact_control(self.c,[b.m.body(i).name for i in range(b.m.nbody)],self.names)
        self.specs=[(self.names.index(j['name']),j) for j in self.c['joints']]
        self.target=b.object_ids.get('target',b.object_ids.get('articulated'))
        if self.target is None:raise ValueError('No registered task target')
        self.shadow_supported=hasattr(b.e.task,'_is_sitting_now')
        baseline=Path(params['reference_case']);result=json.loads((baseline/'result.json').read_text())
        if not result['success'] or result['mode']!='original_token':raise ValueError('Successful original replay reference required')
        self.reference=np.load(baseline/'execution/controller.npz')['records'][:,-len(s.qids):]
        if len(self.reference)<source_frames:raise ValueError('Reference shorter than source')
        self.audit=verify_origin(params['criteria_decision'])['decision']
        if self.audit['assessment']=='needs_tool':raise ValueError('Unresolved acceptance audit')
        self.hold=0;self.shadow_ever=False
        self.evidence=[baseline/'execution/controller.npz',baseline/'result.json',Path(params['criteria_decision'])]
        origin=json.loads(Path(params['criteria_decision']).read_text())['origin']
        self.evidence.extend([Path(origin['request']),Path(origin['response'])])

    def contract(self):
        return {'interface':'SONIC mode1 decoder plus explicit joint-reference blend, offset, PD gains and force admittance','token_alone_replays_execution':False,
            'force_measurement':'sum positive normal forces for exact selected robot bodies against task object; N; pre-control 50Hz',
            'contact_control':self.c,'criteria_review':self.audit,'reference_hashes':{str(p):file_hash(p) for p in self.evidence},
            'prefix':'no residual or gain changes before contact_start','reference_source':'successful original-token executed joint targets',
            'bounds':'selected joint target slew and joint limits; force integrator bounded; global force stop',
            'shadow_acceptance':'chair only: native position/height/speed/chair stability gates and hold count, reviewed exact-body force contact predicate; native untouched'}

    def apply(self,target,index):
        q=np.array(target,copy=True);kp=self.s.kp.copy();kd=self.s.kd.copy()
        force,pairs=contact_force(self.b.m,self.b.d,self.c['contact_bodies'],self.target)
        active=index>=self.start
        weight=min(1.,max(0.,(index-self.start)*.02/self.c['ramp_s']))
        if active:
            for k,j in self.specs:
                self.servo[k]=bounded_servo(self.servo[k],force,self.c['force_range_N'],j['force_gain_rad_per_N_s'],.02,self.c['max_rate_rad_s'],self.c['max_offset_rad'])
                ref=self.reference[min(index,len(self.reference)-1),k]
                residual=float(np.clip(j['offset_rad']+self.servo[k],-self.c['max_offset_rad'],self.c['max_offset_rad']))
                q[k]+=weight*(j['replay_blend']*(ref-q[k])+residual)
                if self.previous is not None:q[k]=np.clip(q[k],self.previous[k]-self.c['max_rate_rad_s']*.02,self.previous[k]+self.c['max_rate_rad_s']*.02)
                joint=int(self.b.robot.body_joint_index[k])
                if self.b.m.jnt_limited[joint]:q[k]=np.clip(q[k],*self.b.m.jnt_range[joint])
                kp[k]*=1+weight*(j['kp_scale']-1);kd[k]*=1+weight*(j['kd_scale']-1)
        self.previous=q.copy()
        self.logs.append({'frame':index,'active':active,'weight':weight,'contact_force_N':force,'pairs':pairs,'sonic_target_rad':np.asarray(target).tolist(),'executed_target_rad':q.tolist(),'kp':kp.tolist(),'kd':kd.tolist(),'force_offset_rad':self.servo.tolist()})
        return q,kp,kd

    def measured(self):
        force,pairs=contact_force(self.b.m,self.b.d,self.c['contact_bodies'],self.target)
        auditforce,_=contact_force(self.b.m,self.b.d,self.audit['contact_bodies'],self.target)
        t=self.b.e.task
        if hasattr(t,'_is_sitting_now'):
            m=t._sit_metrics(self.b.e._get_info(),mujoco_env=self.b.e.mujoco)
            m['has_chair_contact']=bool(auditforce>0 and auditforce>=self.audit['min_contact_force_N'])
            self.hold=self.hold+1 if t._is_sitting_now(m) else 0
            self.shadow_ever|=self.hold>=t.hold_steps
        return {'selected_force_N':force,'reviewed_contact_force_N':auditforce,'shadow_hold_count':self.hold,'shadow_success':self.shadow_ever if self.shadow_supported else None,'pairs':pairs}

    def force_stop(self):
        force,_=contact_force(self.b.m,self.b.d,self.c['contact_bodies'],self.target)
        return force>self.c['force_stop_N']

    def save(self,out):
        save(out/'contact_control.json',self.logs)
