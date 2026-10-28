"""Independent right-index targets, explicitly recorded outside SONIC body tokens."""
import numpy as np

FIELDS={'start_frame','ramp_s','max_offset_rad','max_rate_rad_s','force_band_N','joints'}
JOINTS={'right_hand_index_0_joint','right_hand_index_1_joint'}

def validate_fingers(c,names,frames):
    if not isinstance(c,dict) or set(c)!=FIELDS:raise ValueError('Unexpected finger fields')
    if type(c['start_frame']) is not int or not 0<=c['start_frame']<frames:raise ValueError('Invalid finger start')
    for k,lo,hi in [('ramp_s',.1,3.),('max_offset_rad',.001,.8),('max_rate_rad_s',.01,1.5)]:
        if type(c[k]) not in (int,float) or not np.isfinite(c[k]) or not lo<=c[k]<=hi:raise ValueError('Invalid '+k)
    f=np.asarray(c['force_band_N'],float)
    if f.shape!=(2,) or not np.isfinite(f).all() or not 0<=f[0]<f[1]<=50:raise ValueError('Invalid finger force band')
    seen=set()
    if not 1<=len(c['joints'])<=2:raise ValueError('One or two index joints required')
    for j in c['joints']:
        if set(j)!={'name','offset_rad','force_gain_rad_per_N_s'} or j['name'] not in JOINTS or j['name'] not in names or j['name'] in seen:raise ValueError('Unknown/duplicate/non-index joint')
        seen.add(j['name'])
        for k,lim in [('offset_rad',c['max_offset_rad']),('force_gain_rad_per_N_s',.04)]:
            if type(j[k]) not in (int,float) or not np.isfinite(j[k]) or abs(j[k])>lim:raise ValueError('Invalid '+k)
    return c

def validate_saved_hands(left,right,frames):
    arrays=[np.asarray(x,np.float32) for x in (left,right)]
    if any(a.shape!=(frames,7) or not np.isfinite(a).all() for a in arrays):raise ValueError('Invalid saved hand command shape/values')
    return tuple(arrays)

class FingerTargets:
    def __init__(self,c,names,ranges,frames=10**9):
        self.c=validate_fingers(c,names,frames);self.names=names;self.ranges=np.asarray(ranges);self.integral=np.zeros(7);self.previous=None;self.logs=[]
    def apply(self,source,index,force):
        q=np.asarray(source,np.float32).copy()
        if q.shape!=(7,) or not np.isfinite(q).all() or not np.isfinite(force):raise ValueError('Invalid source hands/force')
        c=self.c;w=float(np.clip((index-c['start_frame'])*.02/c['ramp_s'],0,1));lo,hi=c['force_band_N'];err=lo-force if force<lo else hi-force if force>hi else 0.
        if w>0:
            for j in c['joints']:
                k=self.names.index(j['name']);self.integral[k]=np.clip(self.integral[k]+np.clip(j['force_gain_rad_per_N_s']*err,-c['max_rate_rad_s'],c['max_rate_rad_s'])*.02,-c['max_offset_rad'],c['max_offset_rad'])
                desired=float(q[k])+w*np.clip(j['offset_rad']+self.integral[k],-c['max_offset_rad'],c['max_offset_rad'])
                if self.previous is not None:desired=np.clip(desired,self.previous[k]-c['max_rate_rad_s']*.02,self.previous[k]+c['max_rate_rad_s']*.02)
                q[k]=np.clip(desired,*self.ranges[k])
        self.previous=q.copy();self.logs.append({'frame':index,'weight':w,'downward_force_N':float(force),'source_right_rad':np.asarray(source).tolist(),'executed_right_rad':q.tolist(),'force_integral_rad':self.integral.tolist()})
        return q
