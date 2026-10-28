"""Independent RoboICL-inspired planner. No imports of historical policies."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path


PROMPT = '''You are the sole online planner for a simulated HUMANO G1 with Dex3 hands.
Generate NEW short actions from the supplied demonstration observations, current
state/images and actual interaction receipts. Demonstration transitions are
MEASURED changes, not recovered control commands. Do not replay a source timeline,
token prefix, joint trajectory, or a stored policy. Do not infer execution from
intent. Source omissions are gaps. The target is the CURRENT apple, not its old
position in the demonstration. You control SONIC Cartesian teleop mode1 only:
world wrist positions (meters), wxyz orientations, world velocity_xy, facing_yaw,
pelvis height, torso orientation and each hand's seven joint position commands.
The supplied interface defines bounds and units. The learned controller handles
balance and torque execution; there is no IK or reference-mode fallback.
Use task instruction, current geometry, measured contact and tracking feedback.
This pilot explicitly exposes privileged object poses/contact measurements and
head/external RGB; it is not the original RGB-only dual-arm RoboICL benchmark.
Plan from the original initial state, including any approach. No preprogrammed
walk or grasp precedes you. Preserve physical criteria, controller and budgets.
Return JSON: {"analysis":{"evidence":[...],"hypothesis":"testable prediction",
"change":"what you propose and why"},"actions":[ONE interface action]}.
Each action must have replan_after:true, duration_s between .1 and 3 (multiple
of .005), stage among walk,approach,contact,lift,hold,recover. Include explicit
wrist orientation when setting wrist position. No body_keyframes, native_prefix,
tokens, reuse_prefix, joint-body controls, code or file paths. Missing hand fields
hold previous commands. Walk/recover with omitted wrist positions follows base
translation. Other stages hold world wrists. Observe after every short segment.
Lift only after measured sustained hand contact >=.12 seconds. Collisions and
fall limits are enforced externally. Reanchor after locomotion. Failed tracking
does not prove bad grasp geometry. Do not blindly repeat ineffective commands.
You may use tools ONLY to read the given mailbox request and image paths and to
write its corresponding response. Never inspect other files, conversation,
source trajectories or repository code. Do not run physics or generate policies.
'''

ACTION_FIELDS = {'stage', 'name', 'duration_s', 'velocity_xy', 'facing_yaw',
                 'pelvis_height_m', 'left_position', 'right_position',
                 'left_quaternion_wxyz', 'right_quaternion_wxyz',
                 'left_fingers', 'right_fingers', 'torso_quaternion_wxyz', 'replan_after'}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def verify_frozen(directory):
    directory = Path(directory)
    path = directory/'experiment.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != (directory/'contract.sha256').read_text().strip():
        raise ValueError('Frozen contract hash mismatch')
    contract = json.loads(path.read_text())
    if contract.get('record_status') != 'frozen' or not contract.get('preflight', {}).get('passed'):
        raise ValueError('Experiment has not passed preflight')
    for path, digest in contract['pipeline']['source_snapshot_and_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen input/source changed: '+path)
    return contract


def validate_decision(decision):
    if not isinstance(decision, dict) or set(decision) - {'analysis', 'actions'}:
        raise ValueError('Only analysis/actions accepted; no replay channels')
    analysis = decision.get('analysis', {})
    if not analysis.get('evidence') or not analysis.get('hypothesis'):
        raise ValueError('Measured evidence and falsifiable prediction required')
    actions = decision.get('actions')
    if not isinstance(actions, list) or len(actions) != 1:
        raise ValueError('Exactly one short action per observed turn')
    action = actions[0]
    if not isinstance(action, dict) or set(action) - ACTION_FIELDS:
        raise ValueError('Unknown or forbidden action channel')
    if action.get('stage') not in ('walk', 'approach', 'contact', 'lift', 'hold', 'recover'):
        raise ValueError('Forbidden stage')
    dt = action.get('duration_s', 0)
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or not .1 <= dt <= 3 or abs(dt/.005-round(dt/.005)) > 1e-7:
        raise ValueError('Invalid short duration')
    if action.get('replan_after') is not True:
        raise ValueError('Each action must request observed feedback')
    json.dumps(decision, allow_nan=False)
    for side in ('left', 'right'):
        if side+'_position' in action and side+'_quaternion_wxyz' not in action:
            raise ValueError('Explicit wrist orientation required')
    return copy.deepcopy(decision)


class InteractionMemory:
    def __init__(self, limit=12):
        self.limit = limit
        self.archive = []
        self.previous = None

    def start(self, observation):
        self.previous = copy.deepcopy(observation)

    def commit(self, executed_actions, observation, receipt):
        if self.previous is None:
            raise ValueError('Missing starting observation')
        self.archive.append({'observation_before': self.previous,
                             'executed_actions': copy.deepcopy(executed_actions),
                             'receipt': copy.deepcopy(receipt),
                             'observation_after': copy.deepcopy(observation)})
        self.previous = copy.deepcopy(observation)

    def blocks(self):
        n = len(self.archive)
        indices = list(range(n)) if n <= self.limit else [0] + list(range(n-self.limit+1, n))
        result = []
        previous = -1
        for i in indices:
            block = copy.deepcopy(self.archive[i])
            block['gap_before'] = i > previous+1
            block['interaction_index'] = i
            result.append(block)
            previous = i
        return result


def build_context(request, demonstration, memory):
    interface = copy.deepcopy(request['interface'])
    interface.get('action', {}).pop('body_keyframes', None)
    interface['stages'] = ['walk', 'approach', 'contact', 'lift', 'hold', 'recover']
    return {'instruction': request['task']['instruction'], 'interface': interface,
            'budget': request['task']['budget'], 'observation': request['observation'],
            'demonstration': demonstration, 'live_interactions': memory,
            'attempt': request['attempt'], 'feedback_index': request.get('feedback_index', 0),
            'remaining_sim_seconds': request.get('remaining_sim_seconds', request['task']['budget']['sim_seconds']),
            'observation_scope': 'privileged state/contact plus head and external RGB',
            'previous_reflection': request.get('previous_reflection')}


class MailboxPlanner:
    def __init__(self, directory, demonstration, timeout_s=1200):
        self.directory = Path(directory)
        self.demo = demonstration
        self.timeout = timeout_s
        self.memory = InteractionMemory()
        self.seen_actions = 0
        self.attempt = None
        self.wait_s = 0

    def exchange(self, key, payload, prompt=PROMPT):
        request = self.directory / (key+'_request.json')
        response = self.directory / (key+'_response.json')
        if request.exists() or response.exists():
            raise FileExistsError('Refuse stale mailbox entry: '+key)
        write(request, {'system': prompt, 'input': payload})
        started = time.monotonic()
        print('WAITING_PLANNER', request, flush=True)
        while not response.exists():
            if time.monotonic()-started > self.timeout:
                raise TimeoutError('Isolated planner did not respond: '+key)
            time.sleep(.2)
        elapsed = time.monotonic()-started
        self.wait_s += elapsed
        result = json.loads(response.read_text())
        write(self.directory/(key+'_provenance.json'), {
            'identity': 'Codex isolated subagent, fork_turns=none; session model inherited',
            'external_api_call': False, 'surrounding_conversation_supplied': False,
            'os_file_access_sandbox': False, 'wall_s': elapsed,
            'request_sha256': hashlib.sha256(request.read_bytes()).hexdigest(),
            'response_sha256': hashlib.sha256(response.read_bytes()).hexdigest()})
        return result

    def plan(self, request):
        if self.attempt != request['attempt']:
            self.memory = InteractionMemory()
            self.memory.start(request['observation'])
            self.attempt = request['attempt']
            self.seen_actions = 0
        executed = request.get('executed_actions', [])
        if len(executed) > self.seen_actions:
            self.memory.commit(executed[self.seen_actions:], request['observation'],
                               request['diagnostics']['last_stage'])
            self.seen_actions = len(executed)
        payload = build_context(request, self.demo, self.memory.blocks())
        key = f"a{request['attempt']:03d}_f{request.get('feedback_index', 0):02d}"
        return validate_decision(self.exchange(key, payload))

    def reflect(self, request):
        prompt = ('Return only reflection JSON: source_attempt, result_sha256, failed_stage, '
                  'observed_facts:[{path,value}], hypothesis, alternative_causes:[], '
                  'next_change:{description,controls:[]}, expected_effect, reject_if, '
                  'hypothesis_outcome:supported|not_supported|inconclusive. Copy attempt '
                  'and result hash exactly. Facts must reference exact result.* or diagnostics.* '
                  'paths in input. No actions. This trial has no retry. Diagnose measured '
                  'tracking/contact before interpreting spatial generalization. Do not '
                  'read any file except this request and image paths explicitly supplied.')
        payload = {key: request[key] for key in ('attempt', 'result_sha256', 'result', 'diagnostics')}
        return self.exchange(f"a{request['attempt']:03d}_reflection", payload, prompt)
