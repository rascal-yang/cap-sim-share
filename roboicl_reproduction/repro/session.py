"""Explicit HUMANO protocol adaptation, using unchanged upstream LIVE memory.

A step here is one observed command transition, NOT a 25Hz RoboDojo step.
The caller supplies a frozen-profile converter and execution adapter. No physics
backend is created here. Exceptions during execution poison the session because
elapsed time cannot prove what executed. No joint/mode0 safety fallback.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from .offline import VENDOR  # adds only the pinned owned source to import path
from roboicl.policy.trajectory_memory import LiveTrajectoryMemory


class CorrectionExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class StepResult:
    observation: dict
    executed: bool
    elapsed_s: float
    terminal: bool = False
    interrupt: str | None = None
    feedback: dict | None = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Session:
    def __init__(self, initial, *, convert, state_keys, cameras, horizon=2,
                 maximum=300, anchors=12, corrections=2, train_messages=()):
        if any(type(x) is not int or x < 1 for x in (horizon, maximum, anchors)) or anchors < 2:
            raise ValueError('positive integer budgets, anchors >= 2 required')
        if type(corrections) is not int or corrections < 0:
            raise ValueError('invalid correction budget')
        self.convert, self.state_keys, self.cameras = convert, set(state_keys), set(cameras)
        self.horizon, self.maximum, self.corrections = horizon, maximum, corrections
        self.memory = LiveTrajectoryMemory(anchors, retain_latest=True)
        self.memory.configure(maximum)
        self.archive = {0: self._observation(initial)}
        self.step = 0
        self.terminal = False
        self.seen = {m['call_id'] for m in train_messages if m.get('type') == 'function_call'}
        self.tail, self.ledger, self.receipts = [], [], []
        self.train = deepcopy(list(train_messages))
        self.rejections = 0

    def _observation(self, value):
        if (not isinstance(value, dict) or set(value) != {'state', 'images'}
                or not isinstance(value['state'], dict) or not isinstance(value['images'], dict)
                or set(value['state']) != self.state_keys or set(value['images']) != self.cameras):
            raise ValueError('observation whitelist differs from frozen state/camera profile')
        json.dumps(value, allow_nan=False)
        if any(not isinstance(image, str) or not image for image in value['images'].values()):
            raise ValueError('encoded camera strings required')
        return deepcopy(value)

    def _message(self, step):
        entry = self.archive[step]
        return {'role': 'user', 'content': [
            {'type': 'input_text', 'text': json.dumps({'source': 'LIVE', 'observation_step': step, 'robot_state': entry['state']})},
            *[{'type': 'input_image', 'image_url': entry['images'][cam], 'detail': 'original'} for cam in sorted(self.cameras)]]}

    def context(self):
        return deepcopy(self.train) + self.memory.render(self._message(0), self.step, self._message, tail=self.tail)

    def _correction(self):
        self.rejections += 1
        if self.rejections > self.corrections:
            self.terminal = True
            raise CorrectionExhausted('HUMANO fail-closed: correction budget exhausted; no implicit hold/mode switch')

    def submit(self, call, execute):
        if self.terminal:
            raise RuntimeError('terminal session; create a new episode explicitly')
        if (not isinstance(call, dict) or call.get('type') != 'function_call'
                or call.get('name') != 'act' or not isinstance(call.get('call_id'), str)
                or not call['call_id'].strip() or call['call_id'] in self.seen):
            raise ValueError('one act with a fresh call_id required')
        cid = call['call_id']; self.seen.add(cid)
        self.tail.append(deepcopy(call))
        try:
            args = json.loads(call['arguments'])
            if set(args) != {'actions', 'execution_note'} or not isinstance(args['execution_note'], str) or len(args['execution_note']) > 400:
                raise ValueError('invalid act arguments')
            rows = args['actions']
            if not isinstance(rows, list) or not 1 <= len(rows) <= self.horizon:
                raise ValueError('invalid horizon')
            commands = self.convert(deepcopy(rows), deepcopy(self.archive[self.step]['state']))
            if not isinstance(commands, list) or len(commands) != len(rows):
                raise ValueError('converter must preserve all prediction rows')
        except (ValueError, KeyError, TypeError) as error:
            self.tail.append({'type': 'function_call_output', 'call_id': cid, 'output': json.dumps({
                'error': 'invalid_action', 'detail': str(error), 'executed_steps': 0,
                'observation_step': self.step, 'recovery': 'Correct from SAME observation; no motion executed.'})})
            self._correction()
            return
        start = self.step; records = []; events = []
        for index, command in enumerate(commands):
            if self.step >= self.maximum:
                self.terminal = True
                break
            try:
                result = execute(deepcopy(command))
                if (not isinstance(result, StepResult) or type(result.executed) is not bool
                        or type(result.terminal) is not bool or type(result.elapsed_s) not in (int, float)
                        or not math.isfinite(result.elapsed_s) or result.elapsed_s < 0
                        or (result.interrupt is not None and (not isinstance(result.interrupt, str) or not result.interrupt))):
                    raise ValueError('invalid execution evidence')
                observed = self._observation(result.observation)
                if result.executed:
                    if result.elapsed_s <= 0:
                        raise ValueError('executed command requires measured positive duration')
                    self.step += 1
                    self.archive[self.step] = observed
                    records.append({'step': self.step, 'command': deepcopy(command), 'elapsed_s': result.elapsed_s,
                                    'feedback': deepcopy(result.feedback)})
                elif result.elapsed_s != 0 or observed != self.archive[self.step] or not result.interrupt:
                    raise ValueError('zero-step rejection must preserve observation and elapsed time')
                if result.interrupt:
                    events.append({'type': result.interrupt, 'action_index': index, 'executed': result.executed})
                self.terminal = result.terminal or self.step >= self.maximum
            except Exception as error:
                self.terminal = True
                # Preserve evidence already received, never invent an unknown result.
                self.ledger.extend(records)
                unknown = {'error': 'execution_outcome_unknown', 'detail': str(error),
                           'terminal': True, 'confirmed_executed_steps': len(records),
                           'executed_steps': None, 'unexecuted_steps': None,
                           'uncertain_action_index': index,
                           'observation_status': 'last confirmed observation, before unknown execution'}
                self.tail.append({'type': 'function_call_output', 'call_id': cid,
                                  'output': json.dumps(unknown)})
                if records:
                    self.memory.capture(start, self.step, self.tail, summary={
                        'confirmed_executed_steps': len(records),
                        'execution_outcome_unknown': True, 'terminal': True})
                    self.tail = []
                raise
            if result.interrupt or self.terminal or not result.executed:
                break
        self.ledger.extend(records)
        receipt = {'source': 'LIVE', 'executed_step_interval': [start, self.step],
                   'predicted_steps': len(commands), 'executed_steps': len(records),
                   'unexecuted_steps': len(commands)-len(records), 'terminal': self.terminal,
                   'actual_elapsed_s': sum(r['elapsed_s'] for r in records),
                   'executed_command_ledger_sha256': digest(records), 'controller_events': events,
                   'last_submitted_command': records[-1]['command'] if records else None,
                   'latest_feedback': deepcopy(result.feedback) if commands else None}
        self.receipts.append(deepcopy(receipt))
        self.tail.append({'type': 'function_call_output', 'call_id': cid, 'output': json.dumps(receipt)})
        if records:
            self.memory.capture(start, self.step, self.tail, summary={
                'executed_steps': len(records), 'discarded_prediction_steps': len(commands)-len(records),
                'execution_note': args['execution_note'], 'controller_interrupted': bool(events)})
            self.tail = []; self.rejections = 0
        elif not self.terminal:
            self._correction()
