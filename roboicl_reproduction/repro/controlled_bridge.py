"""Opt-in standing-idle admission and feedback from the saved physical prefix.

No trajectory targets, physics parameters, encoder modes, or native acceptance
are changed here. Admission checks occur before each bounded native command;
velocity evidence is accumulated at every saved 5 ms tick between commands.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .bridge import NativeBridge
from .idle_bridge import StandingIdlePlanner
from .session import StepResult

PHYSICS_DT = .005
PLANAR_SPEED_LIMIT = .03
LOCAL_Z_ANGULAR_SPEED_LIMIT = .1
STABILITY_DURATION = .5
FINGER_DELTA_TOLERANCE = 1e-6


class MeasuredStabilityGate:
    def __init__(self, initial_time=0.):
        self.time = float(initial_time)
        self.duration = 0.
        self.planar_speed = None
        self.local_z_angular_speed = None
        self.last_sample_valid = False
        self.standing_idle_active = False

    def update(self, now, root_velocity, *, standing_idle_active=True):
        now = float(now)
        if not np.isfinite(now) or now < self.time-1e-9:
            raise ValueError('Saved sample time must be finite and monotonic')
        if now <= self.time+1e-9:
            return
        dt = now-self.time
        self.time = now
        velocity = np.asarray(root_velocity, dtype=float)
        valid = velocity.shape == (6,) and np.isfinite(velocity).all()
        self.planar_speed = float(np.linalg.norm(velocity[:2])) if valid else None
        self.local_z_angular_speed = float(velocity[5]) if valid else None
        self.last_sample_valid = bool(valid)
        self.standing_idle_active = bool(standing_idle_active)
        within = (valid and self.standing_idle_active and self.planar_speed <= PLANAR_SPEED_LIMIT+1e-12
                  and abs(self.local_z_angular_speed) <= LOCAL_Z_ANGULAR_SPEED_LIMIT+1e-12)
        self.duration = self.duration+dt if within and abs(dt-PHYSICS_DT) <= 1e-8 else 0.

    def feedback(self):
        return {'ready': bool(self.last_sample_valid and self.duration >= STABILITY_DURATION-1e-9),
                'continuous_s': self.duration, 'last_sample_time_s': self.time,
                'planar_speed_m_s': self.planar_speed,
                'root_local_z_angular_speed_rad_s': self.local_z_angular_speed,
                'planar_speed_limit_m_s': PLANAR_SPEED_LIMIT,
                'root_local_z_angular_speed_limit_rad_s': LOCAL_Z_ANGULAR_SPEED_LIMIT,
                'required_continuous_s': STABILITY_DURATION, 'sample_dt_s': PHYSICS_DT,
                'standing_idle_active_at_last_sample': self.standing_idle_active,
                'angular_definition': 'abs(root free-joint local qvel[5]); not Euler yaw derivative',
                'evidence': 'saved physical states under actual standing-idle feed and requested idle; '
                            'duplicate reads add no duration; gaps/movement/non-idle reset'}


def dex3_closure_profile(hand_spec, actual_joint_names):
    """Name-checked directional proxy from the installed hand's open/close poses."""
    initial = np.asarray(hand_spec.init_qpos, dtype=float)
    if initial.shape != (7,) or not np.isfinite(initial).all():
        raise ValueError('Dex3 initial profile must contain seven finite joints')
    profile = {'joint_names': {}, 'source_joint_names': {}, 'open_qpos': {}, 'close_qpos': {}, 'closing_signs': {},
               'delta_tolerance_rad': FINGER_DELTA_TOLERANCE,
               'interpretation': 'Per-joint direction from installed Dex3 open/close profile; '
                                 'not contact, enclosure, or force-closure evidence. '
                                 'Unknown-direction changes conservatively require stability.'}
    for side in ('left', 'right'):
        expected = list(getattr(hand_spec, side+'_actuated_joints'))
        actual = actual_joint_names[side]
        if len(actual)!=7 or len(set(actual))!=7 or set(actual)!=set(expected):
            raise ValueError('Actual Dex3 joint order/identities differ from closure profile: '+side)
        closed = np.asarray(getattr(hand_spec, 'close_qpos_'+side), dtype=float)
        if closed.shape != (7,) or not np.isfinite(closed).all():
            raise ValueError('Dex3 closure profile must contain seven finite joints')
        order = [expected.index(name) for name in actual]
        delta = (closed-initial)[order]
        profile['joint_names'][side] = list(actual)
        profile['source_joint_names'][side] = expected
        profile['open_qpos'][side] = initial[order].tolist()
        profile['close_qpos'][side] = closed[order].tolist()
        profile['closing_signs'][side] = np.where(abs(delta)>FINGER_DELTA_TOLERANCE,
                                                 np.sign(delta), 0).astype(int).tolist()
    return profile


def new_closure(targets, previous_commands, profile):
    targets = np.asarray(targets, dtype=float)
    previous = np.asarray(previous_commands, dtype=float)
    if (targets.shape != (2, 7) or previous.shape != (2, 7)
            or not np.isfinite(targets).all() or not np.isfinite(previous).all()):
        raise ValueError('Closure admission requires finite bilateral seven-joint targets')
    closing, unknown = [], []
    for index, side in enumerate(('left', 'right')):
        for joint, sign, delta in zip(profile['joint_names'][side], profile['closing_signs'][side],
                                     targets[index]-previous[index]):
            if sign and delta*sign > FINGER_DELTA_TOLERANCE:
                closing.append(joint)
            elif not sign and abs(delta) > FINGER_DELTA_TOLERANCE:
                unknown.append(joint)
    return {'new_closure': bool(closing), 'closing_joints': closing,
            'unknown_direction_changes': unknown, 'requires_stability': bool(closing or unknown),
            'reference': 'previous commanded fingers, named installed Dex3 closure directions'}


def _contact_summary(states, contacts):
    summary = {'sample_count': len(states), 'sample_dt_s': PHYSICS_DT,
               'peak_robot_table_N': 0., 'peak_hand_object_N': 0.,
               'robot_table_intervals': [], 'hand_object_intervals': [],
               'interval_semantics': 'Observed sample occupancy within this command only; '
                                     'not contact normals or a force-closure assertion',
               'table_interval_threshold_N': 0., 'hand_interval_threshold_N': .1,
               'contact_pairs': []}
    for key, label, threshold in (('robot_table_N','robot_table',0.), ('hand_object_N','hand_object',.1)):
        interval = None
        for state, contact in zip(states, contacts):
            time = float(state[0]); force = float(contact[key])
            if not np.isfinite(force):
                raise ValueError('Saved contact force must be finite')
            summary['peak_'+key] = max(summary['peak_'+key], force)
            if force > threshold:
                if interval is None or abs(time-interval['last_sample_s']-PHYSICS_DT)>1e-8:
                    interval = {'first_sample_s': time, 'last_sample_s': time,
                                'observed_duration_s': 0., 'sample_count': 0, 'peak_force_N': 0.}
                    summary[label+'_intervals'].append(interval)
                interval['last_sample_s'] = time
                interval['sample_count'] += 1
                interval['observed_duration_s'] = interval['sample_count']*PHYSICS_DT
                interval['peak_force_N'] = max(interval['peak_force_N'], force)
            else:
                interval = None
    pairs = {}
    for state, contact in zip(states, contacts):
        sample_pairs = {}
        for event in contact.get('events', []):
            bodies, geoms = event['bodies'], event['geoms']
            table = any('table' in name for name in bodies+geoms)
            robot = any('link' in name or 'hand' in name for name in bodies)
            hand = any('hand' in name for name in bodies)
            other_support = any('floor' in name or 'ground' in name for name in bodies+geoms)
            if not ((table and robot) or (hand and not table and not other_support)):
                continue
            key = tuple(sorted(zip(bodies,geoms)))
            sample_pairs[key] = sample_pairs.get(key,0.)+float(event['force_N'])
        for key, force in sample_pairs.items():
            item = pairs.setdefault(key, {'bodies':[p[0] for p in key], 'geoms':[p[1] for p in key],
                'peak_force_N':0., 'first_sample_s':float(state[0]), 'last_sample_s':float(state[0]), 'sample_count':0})
            item['last_sample_s'] = float(state[0]); item['sample_count'] += 1
            item['peak_force_N'] = max(item['peak_force_N'],force)
    summary['contact_pairs'] = list(pairs.values())
    return summary


def _array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


class ControlAwareBridge(NativeBridge):
    def __init__(self, directory, expected_physics, *, standing_idle=False):
        super().__init__(directory, expected_physics)
        if type(standing_idle) is not bool:
            raise ValueError('standing_idle must be an explicit bool')
        self.standing_idle = standing_idle
        self.idle_adapter = None
        self.closure_profile = None
        self.stability = MeasuredStabilityGate()
        self.execution_terminal = False

    def reset(self, task, out):
        # The inherited env reset performs a pre-restore bootstrap step. Count
        # it without suppressing it or changing its inputs, separately from LIVE.
        import mujoco
        original_step = mujoco.mj_step
        proof = {'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                 'calls':[], 'restored_initial_state':None,
                 'interpretation':'Inherited environment bootstrap before restoring frozen initial state; '
                                  'discarded from LIVE, counted separately from saved action physics.'}
        def counted_step(model, data, *args, **kwargs):
            call = {'requested_nstep':int(kwargs.get('nstep',args[0] if args else 1)),
                    'timestep_s':float(model.opt.timestep),'before_time_s':float(data.time),
                    'control_before':np.asarray(data.ctrl).tolist(),
                    'control_before_sha256':_array_sha(data.ctrl),
                    'sonic_constructed':getattr(self,'sonic',None) is not None}
            try:
                return original_step(model,data,*args,**kwargs)
            finally:
                call['after_time_s'] = float(data.time)
                call['control_after_sha256'] = _array_sha(data.ctrl)
                ticks = (call['after_time_s']-call['before_time_s'])/call['timestep_s']
                call['measured_physics_steps'] = round(ticks) if ticks>=0 and abs(ticks-round(ticks))<1e-7 else None
                proof['calls'].append(call)
        mujoco.mj_step = counted_step
        try:
            observation = super().reset(task, out)
            expected = {'qpos':self.initial_q,'qvel':self.initial_v,
                        'ctrl':self.initial['ctrl'],'qacc_warmstart':self.initial['qacc_warmstart']}
            hashes = {key:{'actual_sha256':_array_sha(getattr(self.d,key)),
                           'expected_sha256':_array_sha(value)} for key,value in expected.items()}
            matched = all(item['actual_sha256']==item['expected_sha256'] for item in hashes.values())
            proof['restored_initial_state'] = {'time_s':float(self.d.time),'hashes':hashes,'all_hashes_match':matched}
            if not matched or abs(float(self.d.time))>1e-12:
                raise RuntimeError('Reset failed to restore frozen qpos/qvel/ctrl/warmstart at time zero')
        finally:
            mujoco.mj_step = original_step
            counts = [call['measured_physics_steps'] for call in proof['calls']]
            proof['actual_bootstrap_physics_steps'] = sum(counts) if all(n is not None for n in counts) else None
            proof['ended_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
            (Path(out)/'reset_physics.json').write_text(json.dumps(proof,indent=2,allow_nan=False))
        from humano.robots import hands
        names = {side: [self.m.joint(int(j)).name for j in getattr(self.robot, side+'_hand_actuated_index')]
                 for side in ('left', 'right')}
        self.closure_profile = dex3_closure_profile(hands.DEX3_HAND, names)
        source = Path(hands.__file__).resolve()
        self.closure_profile['source'] = {'path': str(source),
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'profile': 'DEX3_HAND'}
        self.idle_adapter = StandingIdlePlanner(self.sonic.planner, enabled=self.standing_idle,
                                               clock=lambda: float(self.d.time))
        self.sonic.planner = self.idle_adapter
        self.stability = MeasuredStabilityGate(initial_time=float(self.d.time))
        self.execution_terminal = False
        return observation

    def _actual_idle(self, sample_start_time=None):
        if (self.idle_adapter is None or not self.idle_adapter.enabled
                or getattr(getattr(self, 'sonic', None), 'planner', None) is not self.idle_adapter
                or not self.idle_adapter.records):
            return False
        if sample_start_time is None:
            record = self.idle_adapter.records[-1]
        else:
            record = next((r for r in reversed(self.idle_adapter.records)
                           if r.get('time') is not None and r['time']<=sample_start_time+1e-9), None)
            if record is None:
                return False
        actual = record['after']
        return bool(record['transformed'] and actual['mode']==[0]
                    and actual['target_vel']==[-1] and actual['height']==[-1]
                    and actual['movement_direction']==[[0,0,0]])

    def control_status(self):
        return {'standing_idle_enabled': self.standing_idle,
                'actual_standing_idle_active': self._actual_idle(),
                'stability': self.stability.feedback(), 'closure_profile': deepcopy(self.closure_profile),
                'actual_planner_feed': deepcopy(self.idle_adapter.records[-1])
                    if self.idle_adapter is not None and self.idle_adapter.records else None}

    def step(self, command, remaining):
        if self.execution_terminal:
            raise RuntimeError('terminal bridge; reset before any further physical motion')
        if self.closure_profile is None:
            raise RuntimeError('Closure profile not established by reset')
        closure = new_closure([command[s+'_fingers'] for s in ('left','right')],
                              self.fingers, self.closure_profile)
        requires = command['stage'] in ('contact','lift') or closure['requires_stability']
        requested_idle = (float(np.linalg.norm(command.get('velocity_xy',[0,0])))<=1e-6
                          and float(command.get('pelvis_height_m',.74))>=.72)
        actual_idle = self._actual_idle()
        ready = self.stability.feedback()['ready']
        admission = {'allowed': bool(not requires or (requested_idle and actual_idle and ready)),
                     'requires_stability': requires, 'requested_idle': requested_idle,
                     'actual_standing_idle_active': actual_idle, 'closure': closure,
                     'reason': 'Contact/lift or new closure requires actual standing-idle and measured stable dwell; '
                               'explicit hold/opening/recover with no new closure can gather new evidence.'}
        if not admission['allowed']:
            feedback = {'admission': admission, 'stability': self.stability.feedback(),
                        'command_contacts': _contact_summary([],[]),
                        'actual_planner_feed': self.control_status()['actual_planner_feed']}
            result = StepResult(deepcopy(self.cached), False, 0., interrupt='standing_stability_gate', feedback=feedback)
            self.results.append({'command':deepcopy(command), 'elapsed_s':0.,
                                 'result':{'failure':'standing_stability_gate','samples':[]}, 'feedback':feedback})
            return result
        start = len(self.states)
        before = float(self.d.time)
        try:
            result = super().step(command, remaining)
            states = self.states[start:]
            contacts = self.contact_records[start:]
            if (len(self.states)!=len(self.contact_records) or len(states)!=len(contacts)
                    or bool(states)!=result.executed
                    or abs(float(self.d.time)-before-result.elapsed_s)>1e-8
                    or (states and abs(float(states[-1][0])-float(self.d.time))>1e-8)):
                raise ValueError('Execution must match actual saved state/contact prefix')
            for state in states:
                eligible = requested_idle and self._actual_idle(float(state[0])-PHYSICS_DT)
                self.stability.update(state[0],state[1+self.m.nq:1+self.m.nq+6],
                                      standing_idle_active=eligible)
            feedback = deepcopy(result.feedback or {})
            feedback.update(admission=admission, stability=self.stability.feedback(),
                            command_contacts=_contact_summary(states,contacts),
                            actual_planner_feed=self.control_status()['actual_planner_feed'])
            self.execution_terminal = result.terminal
            if self.results:
                self.results[-1]['feedback'] = deepcopy(feedback)
            return replace(result, feedback=feedback)
        except Exception:
            self.execution_terminal = True
            raise

    def export(self, out):
        super().export(out)
        values = {'planner_inputs.json':self.idle_adapter.records if self.idle_adapter is not None else [],
                  'closure_profile.json':self.closure_profile, 'stability_gate.json':self.stability.feedback()}
        for name, value in values.items():
            (Path(out)/name).write_text(json.dumps(value,indent=2,allow_nan=False))
