"""Measured-result bridge; original cap_sim executor remains unchanged."""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import json
import time
import numpy as np
from cap_sim.backends.humano import Backend
from cap_sim.backends.native_sonic import rot
from .session import StepResult

# Reuse audited read-only camera/physics helpers; never the previous pipeline.
_helper = Path(__file__).resolve().parents[2]/'roboicl_adapter/scene.py'
_spec=importlib.util.spec_from_file_location('roboicl_readonly_scene',_helper)
scene=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(scene)
STATE_KEYS=('time','root_pose','root_velocity','left_position','right_position',
            'left_quaternion_wxyz','right_quaternion_wxyz','torso_quaternion_wxyz',
            'fingers','measured_fingers')
CAMERAS=('head','external')


class ContinuousContactGate:
    def __init__(self):
        self.time=0.;self.duration=0.
    def update(self, now, force):
        if now<=self.time+1e-10:return self.duration>=.05-1e-9
        dt=now-self.time;self.time=now
        self.duration=self.duration+dt if force>5 else 0.
        return self.duration>=.05-1e-9


def classify_result(result, success):
    failure=result.get('failure')
    if success:return True,'native_success'
    if failure=='hand_contact_gate' and not result['samples']:return False,failure
    if failure:return True,failure
    return False,None


class PhysicalStop(Exception):
    def __init__(self, contacts):self.contacts=contacts


class NativeBridge(Backend):
    def __init__(self,directory,expected_physics):
        self.directory=Path(directory);self.expected_physics=expected_physics
        self.guard=ContinuousContactGate();self.guard_active=False;self.observation_index=0
        self.simulation_wall_s=0.;self.render_wall_s=0.;self.results=[];self.cached=None

    def _prepare(self,task):
        super()._prepare(task)
        if scene.physics_signature(self.m)!=self.expected_physics:
            raise RuntimeError('Compiled model differs from frozen physical baseline')

    def reset(self,task,out):
        super().reset(task,out)
        self.d.qacc_warmstart[:]=self.initial['qacc_warmstart']
        providers={k:getattr(self.sonic,k).get_providers() for k in ('encoder','decoder','planner')}
        if any(p!=['CPUExecutionProvider'] for p in providers.values()):raise RuntimeError('Unexpected actual provider')
        (Path(out)/'actual_providers.json').write_text(json.dumps(providers,indent=2))
        (Path(out)/'actual_physics.json').write_text(json.dumps(scene.physics_signature(self.m),indent=2))
        self.limits={s:self.m.jnt_range[getattr(self.robot,s+'_hand_actuated_index')].tolist() for s in ('left','right')}
        self.guard=ContinuousContactGate()
        return self.capture()

    def contacts(self):
        contacts=super().contacts()
        if self.guard_active and self.guard.update(float(self.d.time),contacts['robot_table_N']):
            raise PhysicalStop(contacts)
        return contacts

    def capture(self):
        raw=super().observe()
        state={k:raw[k] for k in STATE_KEYS if k!='measured_fingers'}
        state['measured_fingers']=[[v['qpos'][0] for v in raw['joint_interface'][s+'_hand']] for s in ('left','right')]
        started=time.perf_counter()
        paths=scene.render_views(self.m,self.d,self.directory/'images',f'obs_{self.observation_index:05d}')
        self.render_wall_s+=time.perf_counter()-started;self.observation_index+=1
        self.cached={'state':state,'images':dict(zip(CAMERAS,paths))}
        return deepcopy(self.cached)

    def step(self,command,remaining):
        if 'body_keyframes' in command:raise ValueError('No mode0 or replay channels')
        before=float(self.d.time);nrecords=len(self.sonic.records);started=time.perf_counter()
        self.guard_active=True
        try:
            result=super().execute(command,remaining)
        except PhysicalStop as stop:
            # Native executor checks contacts after mj_step/tick/reward, before
            # appending this tick. Preserve that real terminal tick exactly once.
            row={'time':float(self.d.time),'dt':.005,'native_success':self.native_passed,
                 'stable':bool(np.isfinite(self.d.qpos).all() and self.d.qpos[2]>.3 and
                    np.arccos(np.clip(rot(self.d.qpos[3:7]).as_matrix()[2,2],-1,1))<np.deg2rad(50)),
                 'native_reward':self.native_reward}
            self.hand_contact_s=self.hand_contact_s+.005 if stop.contacts['hand_object_N']>.1 else 0.
            if not self.samples or self.samples[-1]['time']<self.d.time-1e-9:
                self.samples.append(row);self.states.append(np.r_[self.d.time,self.d.qpos,self.d.qvel,self.d.ctrl]);self.contact_records.append(stop.contacts)
            result={'failure':'unexpected_table_contact','completed':False,
                    'samples':[s for s in self.samples if s['time']>before+1e-9]}
            self.success=False
        except ValueError as error:
            if abs(float(self.d.time)-before)>1e-9:raise
            self.results.append({'command':command,'error':str(error),'executed':False})
            return StepResult(deepcopy(self.cached),False,0,interrupt='action_rejected: '+str(error))
        finally:
            self.guard_active=False;self.simulation_wall_s+=time.perf_counter()-started
        elapsed=float(self.d.time)-before
        records=self.sonic.records[nrecords:]
        modes=sorted({r['mode'] for r in records})
        if elapsed>0 and (modes!=[1] or not records):raise RuntimeError('Actual SONIC mode evidence mismatch')
        terminal,event=classify_result(result,self.success)
        if elapsed<=1e-9:
            ob=deepcopy(self.cached)
        else:
            ob=self.capture()
        tracking={s:{'position_error_m':float(np.linalg.norm(np.array(ob['state'][s+'_position'])-command[s+'_position'])),
                     'orientation_error_rad':float(2*np.arccos(np.clip(abs(np.dot(ob['state'][s+'_quaternion_wxyz'],command[s+'_quaternion_wxyz'])),0,1)))} for s in ('left','right')}
        if not terminal and elapsed>0 and getattr(self,'tracking_guard',None) is not None and self.tracking_guard.observe(tracking):
            event='chunk_interrupted_tracking_error'
        feedback={'tracking':tracking,'actual_modes':modes,'hand_contact_s':self.hand_contact_s,
                  'contacts':{k:v for k,v in super().contacts().items() if k!='events'},
                  'continuous_table_contact_s':self.guard.duration}
        self.results.append({'command':deepcopy(command),'elapsed_s':elapsed,'result':result,'feedback':feedback})
        return StepResult(ob,elapsed>1e-9,elapsed,terminal=terminal,interrupt=event,feedback=feedback)
