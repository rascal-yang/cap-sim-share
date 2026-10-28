"""Short numerical API loop: cached geometric plan, one physical cycle, reflection."""
import json
import math
from pathlib import Path
import time

from .agent import save
from .contracts import require
from .lateral_tools import LocalDriver, PARAMETERS, TEXT, NUM, experience_summary, validate_raise
from ..workflow.contracts import digest, file_hash, now
from ..workflow.navigation.archive import verify_evidence

PARAMS = {
    'run_grasp': {'candidate_id': TEXT, 'wrist_raise_m': NUM, 'rationale': TEXT},
    'reflect_grasp': PARAMETERS['reflect_grasp'],
}
INSTRUCTIONS = '''Choose one registered tool for a MuJoCo apple-grasp simulation.
Use only supplied privileged numerical evidence; no images are needed. The host
has verified a cached geometric plan and task skill. Call run_grasp to choose a
bounded wrist correction and execute one frozen experiment. Then read its measured
result and call reflect_grasp. Native lift and clean collision quality are separate;
accept requires both. A retry at the attempt limit records a proposal for a future
batch, not another execution. Keep rationale concise. No final summary call is
needed. SONIC mode1 executes a retargeted original trajectory from an initialized
operation stance; this does not establish walking or independent skill generation.'''


NAV_PARAMS = dict(PARAMS, run_grasp=dict(PARAMS['run_grasp'],
    execution={'type': 'string', 'enum': ['restart', 'navigation_start', 'parked']},
    parent_prepared_id={'type': 'string'}, feedback_gain={'type': 'number', 'minimum': 0, 'maximum': 3}))
NAV_INSTRUCTIONS = """Choose registered tools for numeric apple navigation and grasp.
No images. run_grasp starts from the original far state with precision parking,
then retargets the original replay using SONIC mode1. Choose feedback_gain in [0,3]
and wrist_raise_m in [0,.03]. First execution must be restart with empty parent ID.
Read measured results and reflect_grasp; retry allows another run in this session.
For a later run choose restart (empty parent), or an available checkpoint and its
parent_prepared_id from observed results. parked retains navigation, so wrist_raise_m
must match parent; feedback_gain may change. navigation_start has no inherited steps.
No force/contact abortion gates are added: contacts are measured diagnostics. Native
success and clean contact quality are separate; accept requires both, otherwise use
retry or stop. At budget end retry is only a proposal. All body tokens and separate
hands are saved. This is trajectory retargeting, not independent skill generation.
No extra final response is needed after terminal reflection."""


def validate_branch(args, rows, proof):
    require(args['execution'] in ('restart', 'navigation_start', 'parked'), 'Unknown execution branch')
    gain = args['feedback_gain']
    require(isinstance(gain, (float, int)) and not isinstance(gain, bool) and math.isfinite(gain)
            and 0 <= gain <= 3, 'feedback_gain must be finite within [0,3]')
    if args['execution'] == 'restart':
        require(args['parent_prepared_id'] == '', 'Restart has no parent')
        return None
    ident = args['parent_prepared_id']
    require(ident in rows and ident in proof.get('observed_result_ids', []), 'Checkpoint parent must be an observed session result')
    parent = rows[ident]
    require(parent.get('reflection', {}).get('action') == 'retry', 'Reflect retry before branching')
    require(args['execution'] in parent.get('result', {}).get('available_checkpoints', []), 'Checkpoint unavailable')
    if args['execution'] == 'parked':
        require(args['wrist_raise_m'] == parent['decision']['wrist_raise_m'], 'Changed wrist offset requires restart or navigation_start')
    return dict(case=parent['case'], name=args['execution'])


def validate_cached_plan(config):
    p = json.loads(Path(config['cached_plan']).read_text())
    require(p.get('status') == 'completed' and p.get('selected', {}).get('feasible'), 'Cached plan not feasible')
    require(p['offset_m'] == config['offset_m'], 'Cached plan offset differs', 'integrity_error')
    require(Path(p['source_manifest']).resolve() == Path(config['manifest']).resolve(), 'Cached source differs', 'integrity_error')
    require(p['lateral_options'] == json.loads(Path(config['options']).read_text()), 'Cached options differ', 'integrity_error')
    source_initial = str(Path(config['initial']).resolve())
    require(p['evidence_sha256'].get(source_initial) == file_hash(source_initial), 'Cached source initial differs', 'integrity_error')
    require(p['scene_input_sha256']['initial'] == file_hash(p['scene_files']['initial']), 'Cached shifted initial differs', 'integrity_error')
    verify_evidence(p['evidence_sha256'])
    from ..workflow.navigation.reference_bundle import load_reference_bundle
    load_reference_bundle(p['generated_reference'])
    return p


class PrivilegedDriver(LocalDriver):
    def navigation_cycle(self, plan, value, proof, gain, resume):
        ident = digest(proof)[:16]
        request = self.root / ('navigation_request_' + ident + '.json')
        save(request, dict(root=str(self.root), plan=str(plan), wrist_raise_m=value,
                           feedback_gain=gain, resume=resume, provenance=proof))
        self.invoke('humanoid_mimicgen.augmentation.navigation_api', [request],
                    self.root / ('navigation_' + ident + '.log'))
        response = json.loads(request.with_suffix('.result.json').read_text())
        return Path(response['case']), response

    def cycle(self, plan, value, proof):
        ident = digest(proof)[:16]
        provenance = self.root / ('provenance_' + ident + '.json')
        save(provenance, proof)
        tick = time.monotonic()
        self.invoke('humanoid_mimicgen.augmentation.direct_grasp',
                    ['cycle', self.root, plan, 'apple_privileged_grasp', 0, value, provenance],
                    self.root / ('cycle_' + ident + '.log'))
        cycle_s = time.monotonic() - tick
        case = Path((self.root / 'latest_case.txt').read_text())
        tick = time.monotonic()
        self.invoke('humanoid_mimicgen.augmentation.lateral_delivery', ['audit_numeric', case], case / 'audit.log')
        result = json.loads((case / 'result.json').read_text())
        audit = json.loads((case / 'independent_audit.json').read_text())
        require(audit['recorded_native_matches'] and audit['decoder_targets_unmodified'] and audit['contract_sha_valid'],
                'Independent physical evidence mismatch', 'integrity_error')
        # Contact attribution uses saved numbers, without creating a renderer.
        pairs = {}
        for observation in json.loads((case / 'execution/observations.json').read_text()):
            for event in observation['contacts']['events']:
                if 'table' in event['bodies'] and any('hand' in b or 'wrist' in b for b in event['bodies']):
                    pair = ' / '.join(event['bodies'])
                    row = pairs.setdefault(pair, dict(samples=0, peak_N=0., first_time_s=observation['time_s']))
                    row['samples'] += 1
                    row['peak_N'] = max(row['peak_N'], event['force_N'])
        for row in pairs.values():
            row['duration_s'] = row['samples'] * .005
        save(case / 'contact_diagnosis.json', pairs)
        timing = dict(cycle_wall_s=cycle_s, audit_wall_s=time.monotonic()-tick, render_wall_s=0.,
                      rendered=False, verifier_sonic_sessions_loaded=0, perception_mode='privileged_only', ended_at=now())
        save(case / 'delivery_timing.json', timing)
        summary = {k: result[k] for k in ('physics_steps', 'failure', 'completed', 'max_clearance_m',
                   'max_obstacle_N', 'max_hand_object_N', 'max_root_xy_drift_m', 'wrist_tracking')}
        summary.update(case=str(case), native_success=result['native_success'], clean_success=result['success'],
                       table_contact_s=result['obstacle_contact_gt_0p1_s'], contact_diagnosis=pairs,
                       terminal_half_second=audit['final_half_second'], feet=audit['feet'], timing=timing,
                       result_sha256=file_hash(case/'result.json'), audit_sha256=file_hash(case/'independent_audit.json'))
        return case, summary


class PrivilegedSession:
    allowed_parameters = PARAMS
    instructions = INSTRUCTIONS
    finish_after_reflection = True

    def __init__(self, root, config, driver=None):
        self.root, self.config = Path(root).resolve(), config
        verify_evidence(config.get('input_sha256', {}))
        p = validate_cached_plan(config)
        self.plan = Path(config['cached_plan']).resolve()
        self.plan_hash = file_hash(self.plan)
        self.navigation = config.get('navigation_mode', 'precision') == 'precision'
        require(config.get('navigation_mode', 'precision') in ('precision', 'direct_stance'), 'Unknown navigation mode')
        self.allowed_parameters = NAV_PARAMS if self.navigation else PARAMS
        self.instructions = NAV_INSTRUCTIONS if self.navigation else INSTRUCTIONS
        self.driver = driver or PrivilegedDriver(self.root)
        self.path = self.root/'tool_state.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(
            config_sha256=digest(config), prepared={}, selected_id=p['selected']['id'])
        require(self.state['config_sha256'] == digest(config), 'Session config changed', 'integrity_error')
        experience = []
        for path in config.get('experience_results', []):
            value = experience_summary(path)
            full = json.loads(Path(path).read_text())
            value.update(failure=full.get('failure'), max_hand_object_N=full.get('max_hand_object_N'))
            experience.append(value)
        self.model_context = dict(perception_mode='privileged_only', cached_plan=dict(
            selected_id=self.state['selected_id'], offset_m=p['offset_m'], search_incomplete=p['search_incomplete'],
            scene_change=p['scene_change'], planning='host cache reused after hash/source/scene/options checks; not a new API planning call'),
            experience=experience, thresholds=dict(native_lift_m=.09, clean_obstacle_peak_N=5,
            force_stop_N=200, obstacle_stop_N=50), next_hypothesis=config.get('next_hypothesis'),
            skill_summary='Pick apple from table; require bilateral support before operation, measured wrist alignment before closure, stop on excessive force/fall, then hold for 1s.')

        if self.navigation:
            self.model_context['initialization'] = 'original far state and source controller history; precision navigation executed'
            self.model_context['navigation'] = dict(initialization='original_far_state', mode='precision',
                checkpoints=['navigation_start', 'parked'], policy='caller-selected complete-state rollback; no automatic collision reset')
            self.model_context['thresholds'] = dict(native_lift_m=.09, clean_obstacle_peak_N=5)
            self.model_context['skill_summary'] = 'Precision parking then replay operation; native success and retained grasp, contacts reported separately; no force abortion gates.'

    def tool_definitions(self):
        descriptions = dict(run_grasp='Choose checked stance and wrist raise; freeze, numerically preflight and execute once; return measured privileged results. No image or render.',
                            reflect_grasp='Reflect on a previously returned result. Accept only native and clean success; retry may propose the next batch.')
        return [dict(type='function', name=n, description=descriptions[n], strict=True,
                     parameters=dict(type='object', properties=p, required=list(p), additionalProperties=False)) for n, p in self.allowed_parameters.items()]

    def persist(self):
        save(self.path, self.state)

    def next_step(self):
        rows = list(self.state['prepared'].items())
        if rows and not rows[-1][1].get('reflection'):
            return 'Call reflect_grasp for prepared_id '+rows[-1][0]+' using the measured result.'
        return 'Call run_grasp with candidate_id '+self.state['selected_id']+'; select wrist_raise_m and concise rationale from numerical evidence.'

    def complete(self):
        rows = list(self.state['prepared'].values())
        return bool(rows and all(r.get('result', {}).get('physics_steps', 0) > 0 and r.get('reflection') for r in rows)
                    and (rows[-1]['reflection']['action'] in ('accept', 'stop') or len(rows) >= self.config['max_attempts']))

    def dispatch(self, name, args, provenance=None):
        verify_evidence(self.config.get('input_sha256', {}))
        require(file_hash(self.plan) == self.plan_hash, 'Cached plan changed', 'integrity_error')
        require(name in self.allowed_parameters and isinstance(args, dict) and set(args) == set(self.allowed_parameters[name]), 'Tool/argument fields')
        for key, value in args.items():
            if key not in ('wrist_raise_m', 'feedback_gain', 'parent_prepared_id'):
                require(isinstance(value, str) and bool(value.strip()), 'Nonempty text required')
        proof = provenance or {}
        rows = self.state['prepared']
        if name == 'run_grasp':
            value = validate_raise(args['wrist_raise_m'])
            require(args['candidate_id'] == self.state['selected_id'], 'Candidate differs from checked plan')
            require(bool(proof.get('call_id')), 'Missing decision identity')
            ident = digest(dict(call_id=proof['call_id']))[:24]
            signature = digest(args)
            if ident in rows:
                row = rows[ident]
                require(row['signature'] == signature, 'Changed arguments for same decision', 'integrity_error')
                require(bool(row.get('result')), 'Interrupted execution: inspect saved case; never implicitly repeat')
                self.verify_result(row)
                return row['result']
            require(len(rows) < self.config['max_attempts'], 'Physical attempt budget exhausted')
            require(not rows or list(rows.values())[-1].get('reflection', {}).get('action') == 'retry', 'Reflect before another attempt')
            resume = validate_branch(args, rows, proof) if self.navigation else None
            if resume:
                self.verify_result(rows[args['parent_prepared_id']])
            rows[ident] = dict(signature=signature, started_at=now(), decision=args)
            self.persist()  # Durable launch intent precedes any simulator process.
            proof = dict(proof, decision=args, perception_mode='privileged_only',
                         cached_plan_sha256=self.plan_hash, evidence_sha256=dict(proof.get('evidence_sha256', {}), **{str(self.plan): self.plan_hash}))
            if self.navigation:
                case, result = self.driver.navigation_cycle(self.plan, value, proof, args['feedback_gain'], resume)
            else:
                case, result = self.driver.cycle(self.plan, value, proof)
            row = rows[ident]
            row.update(case=str(case), contract_sha256=file_hash(case/'experiment.json'), result=dict(result, prepared_id=ident))
            self.persist()
            return row['result']
        require(args['prepared_id'] in rows, 'Case outside this session')
        row = rows[args['prepared_id']]
        require(bool(row.get('result')), 'Execute before reflection')
        self.verify_result(row)
        require(args['prepared_id'] in proof.get('observed_result_ids', []), 'Measured result must be in an earlier request')
        require(args['action'] in ('accept', 'retry', 'stop'), 'Unknown reflection action')
        require(args['action'] != 'accept' or row['result']['native_success'] and row['result']['clean_success'], 'Acceptance contradicts physical evidence')
        row['reflection'] = dict(args, recorded_at=now(), provenance=proof)
        save(Path(row['case'])/'api_reflection.json', row['reflection'])
        self.persist()
        return dict(recorded=True, workflow_ready_to_finish=self.complete())

    @staticmethod
    def verify_result(row):
        case = Path(row['case'])
        require(file_hash(case/'experiment.json') == row['contract_sha256'], 'Contract changed', 'integrity_error')
        for filename, key in [('result.json', 'result_sha256'), ('independent_audit.json', 'audit_sha256')]:
            if key in row['result']:
                require(file_hash(case/filename) == row['result'][key], 'Measured evidence changed', 'integrity_error')
