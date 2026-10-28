"""Measured phase progression; reference time cannot outrun contact readiness."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import WorkflowError, positive
from .prior import finite


def validate_operation(config, start=None, end=None):
    try:
        for key in ('transition_s','position_m','orientation_rad','continuous_s','timeout_s','object_force_stop_N'):
            positive(config[key],key)
        if config['timeout_s']<config['continuous_s']:
            raise WorkflowError('invalid_input','Impossible operation gate budget')
        finite(config['open_fingers_rad'],(14,),'Open hand command')
        bodies=config['alignment_bodies']
        if not isinstance(bodies,list) or not bodies or any(not isinstance(b,str) or not b for b in bodies) or len(set(bodies))!=len(bodies):
            raise WorkflowError('invalid_input','Alignment body names')
        gates=config['contact_gates'];frames=[]
        if not isinstance(gates,list):raise WorkflowError('invalid_input','Contact gates')
        for gate in gates:
            frame=gate['frame'];frames.append(frame)
            if type(frame) is not int or (start is not None and not start<frame<end):
                raise WorkflowError('invalid_input','Contact gate outside operation')
            positive(gate['minimum_force_N'],'Contact force')
            if gate['minimum_force_N']>=config['object_force_stop_N']:
                raise WorkflowError('invalid_input','Contact minimum exceeds stopping threshold')
            groups=gate['groups']
            if not isinstance(groups,list) or not groups or any(not isinstance(g,list) or not g for g in groups):
                raise WorkflowError('invalid_input','Contact groups')
            names=[b for group in groups for b in group]
            if any(not isinstance(b,str) or not b for b in names) or len(names)!=len(set(names)):
                raise WorkflowError('invalid_input','Contact groups must be disjoint named bodies')
        geometry=config.get('geometry_gates',[])
        if not isinstance(geometry,list) or any(type(i) is not int or (start is not None and not start<i<end) for i in geometry):
            raise WorkflowError('invalid_input','Geometry gate outside operation')
        if frames!=sorted(set(frames)):raise WorkflowError('invalid_input','Ordered unique contact gates')
    except (KeyError,TypeError) as exc:
        raise WorkflowError('invalid_input','Incomplete whole_body_operation: '+str(exc)) from None
    return config


def poses_aligned(actual, expected, config):
    def rot(q):return Rotation.from_quat(np.roll(q,-1))
    for body in config['alignment_bodies']:
        a=finite(actual[body],(7,),'Measured body pose');b=finite(expected[body],(7,),'Target body pose')
        if not np.isclose(np.linalg.norm(a[3:]),1) or not np.isclose(np.linalg.norm(b[3:]),1):
            raise WorkflowError('invalid_input','Body quaternion')
        if np.linalg.norm(a[:3]-b[:3])>config['position_m'] or (rot(a[3:]).inv()*rot(b[3:])).magnitude()>config['orientation_rad']:
            return False
    return True


class PhaseClock:
    """One step per source physics interval, holding the previous frame at gates."""
    def __init__(self,start,end,config):
        if type(start) is not int or type(end) is not int or not 0<=start<end:
            raise WorkflowError('invalid_input','Operation interval')
        self.config=validate_operation(config,start,end);self.start=start;self.end=end
        self.next=start;self.frame=None;self.wait_s=0.;self.ready_s=0.;self.completed=False
        self.gates={g['frame']:g for g in config['contact_gates']};self.log=[]

    def step(self,dt,observation):
        positive(dt,'Operation dt')
        positive(observation['object_force_N'],'Measured object force',zero=True)
        if observation['object_force_N']>self.config['object_force_stop_N']:
            raise WorkflowError('operation_object_force_limit')
        if self.completed:return dict(frame=self.frame,completed=True,waiting=False)
        if self.next==self.end:
            if not observation.get('completion_ready',True):
                return dict(frame=self.frame,completed=False,waiting=True)
            self.completed=True;return dict(frame=self.frame,completed=True,waiting=False)
        gate=self.gates.get(self.next);waiting=self.next==self.start or gate is not None or self.next in self.config.get('geometry_gates',[])
        if waiting:
            ready=observation['aligned'] is True and observation['stable'] is True
            if gate:
                events=observation['contacts'];objects=set(observation['object_bodies'])
                for e in events:positive(e['force_N'],'Measured contact force',zero=True)
                for group in gate['groups']:
                    force=sum(e['force_N'] for e in events if set(e['bodies'])&set(group) and set(e['bodies'])&objects)
                    ready=ready and force>=gate['minimum_force_N']
            self.wait_s+=dt;self.ready_s=self.ready_s+dt if ready else 0.
            if self.ready_s+1e-12<self.config['continuous_s']:
                if self.wait_s+1e-12>=self.config['timeout_s']:
                    raise WorkflowError('operation_gate_timeout',str(self.next))
                return dict(frame=self.frame,completed=False,waiting=True,waiting_for_frame=self.next)
            if not observation.get('can_advance',True):
                if self.wait_s+1e-12>=self.config['timeout_s']:raise WorkflowError('operation_gate_timeout',str(self.next))
                return dict(frame=self.frame,completed=False,waiting=True,waiting_for_frame=self.next)
            self.log.append(dict(frame=self.next,wait_s=self.wait_s,continuous_s=self.ready_s))
            self.wait_s=self.ready_s=0.
        if not observation.get('can_advance',True):
            return dict(frame=self.frame,completed=False,waiting=True,waiting_for_frame=self.next)
        self.frame=self.next;self.next+=1
        return dict(frame=self.frame,completed=False,waiting=False)


def closure_frames(fingers,open_fingers,start,end):
    """Every per-hand departure from the declared open command needs geometry."""
    h=np.asarray(fingers,float);opened=finite(open_fingers,(14,),'Open finger command')
    if h.ndim!=2 or h.shape[1]!=14 or not np.isfinite(h).all() or not 0<=start<end<=len(h):
        raise WorkflowError('invalid_input','Finger reference')
    result=set()
    for side in (slice(0,7),slice(7,14)):
        is_open=np.all(np.isclose(h[:,side],opened[side],atol=1e-7,rtol=0),axis=1)
        result.update(i for i in range(start+1,end) if is_open[i-1] and not is_open[i])
    return sorted(result)
