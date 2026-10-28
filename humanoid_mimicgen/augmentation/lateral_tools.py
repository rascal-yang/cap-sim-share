"""Bounded, persisted tools for API-directed lateral grasp experiments."""
import json
import math
import os
from pathlib import Path
import subprocess

from .agent import save
from .contracts import require
from ..workflow.contracts import canonical, digest, file_hash, now

SKILL = Path(__file__).parent / 'skills/lateral_grasp/SKILL.md'
REPO = Path(__file__).resolve().parents[2]
NUM = {'type': 'number', 'minimum': 0, 'maximum': 0.03}
TEXT = {'type': 'string', 'minLength': 1}
PARAMETERS = {
    'read_skill': {}, 'inspect_scene': {}, 'plan_lateral': {},
    'prepare_grasp': {'candidate_id': TEXT, 'wrist_raise_m': NUM, 'rationale': TEXT},
    'execute_grasp': {'prepared_id': TEXT, 'initial_review': TEXT},
    'reflect_grasp': {'prepared_id': TEXT, 'action': {'type': 'string', 'enum': ['accept', 'retry', 'stop']},
                      'rationale': TEXT, 'next_change': TEXT},
}
DESCRIPTIONS = {
    'read_skill': 'Read the actual lateral grasp task skill and capability limits.',
    'inspect_scene': 'Measure the fixed scene and successful replay support windows; read prior physical evidence. No physics.',
    'plan_lateral': 'Generate and verify a fresh lateral stance/operation plan at the supplied fixed apple offset. No physics.',
    'prepare_grasp': 'Select the planned candidate and right wrist height correction; freeze a new experiment and return its initial image. No physics.',
    'execute_grasp': 'After viewing the initial image, explicitly execute this prepared physical case once and return independent measured evidence and review images.',
    'reflect_grasp': 'Record an evidence-based accept/retry/stop decision; acceptance requires both native success and collision quality.',
}


def tool_definitions():
    return [dict(type='function', name=n, description=DESCRIPTIONS[n], strict=True,
                 parameters=dict(type='object', properties=p, required=list(p), additionalProperties=False))
            for n, p in PARAMETERS.items()]


def validate_raise(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 0.03,
            'Wrist raise must be finite metres in [0, .03]')
    return float(value)


def review_images(case):
    folder = Path(case).resolve() / 'inspection'
    manifest = folder / 'inspection.json'
    paths = json.loads(manifest.read_text())['review_images'] if manifest.exists() else [
        str(folder / name) for name in ('contact_sheet.jpg', 'contact_focus.jpg') if (folder / name).is_file()]
    require(bool(paths), 'No review image evidence', 'integrity_error')
    for path in paths:
        path = Path(path).resolve()
        require(path.parent == folder and path.is_file(), 'Invalid/missing review image', 'integrity_error')
    return list(paths)


def experience_summary(path):
    path = Path(path)
    result = json.loads(path.read_text())
    summary = {k: result[k] for k in ('experiment_id', 'offset_m', 'native_success', 'success',
               'max_clearance_m', 'max_obstacle_N', 'obstacle_contact_gt_0p1_s')}
    summary['wrist_raise_m'] = json.loads((path.parent / 'config.json').read_text()).get('wrist_z_offset_m')
    if (path.parent / 'contact_diagnosis.json').exists():
        summary['contact_diagnosis'] = json.loads((path.parent / 'contact_diagnosis.json').read_text())
    return summary


class LocalDriver:
    """Each simulator adapter has its own process, including independent verification."""
    def __init__(self, root):
        self.root = root

    def invoke(self, module, args, log):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
        env['CAP_SIM_REPOSITORY'] = str(REPO)
        with Path(log).open('xb') as handle:
            subprocess.run([str(REPO / 'scripts/humano_python.sh'), '-m', module, *map(str, args)],
                           cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=180)

    def prepare(self, plan, raise_m, provenance):
        ident = digest(provenance)[:16]
        path = self.root / ('provenance_' + ident + '.json')
        save(path, provenance)
        self.invoke('humanoid_mimicgen.augmentation.direct_grasp',
                    ['prepare', self.root, plan, 'apple_api_grasp', 0, raise_m, path],
                    self.root / ('prepare_' + ident + '.log'))
        return Path((self.root / 'latest_case.txt').read_text())

    def execute(self, case):
        self.invoke('humanoid_mimicgen.augmentation.direct_grasp', ['run', case], case / 'physics.log')
        self.invoke('humanoid_mimicgen.augmentation.lateral_delivery', ['audit', case], case / 'audit.log')
        self.invoke('humanoid_mimicgen.augmentation.lateral_delivery', ['render', case], case / 'render.log')
        result = json.loads((case / 'result.json').read_text())
        audit = json.loads((case / 'independent_audit.json').read_text())
        require(audit['recorded_native_matches'] and audit['decoder_targets_unmodified'], 'Physical evidence mismatch', 'integrity_error')
        return dict(case=str(case), native_success=result['native_success'], clean_success=result['success'],
                    physics_steps=result['physics_steps'], failure=result['failure'],
                    max_clearance_m=result['max_clearance_m'], max_obstacle_N=result['max_obstacle_N'],
                    table_contact_s=result['obstacle_contact_gt_0p1_s'],
                    contact_diagnosis=json.loads((case / 'contact_diagnosis.json').read_text()),
                    terminal_half_second=audit['final_half_second'],
                    wrist_tracking=result['wrist_tracking'], feet=audit['feet'],
                    video=json.loads((case / 'video.json').read_text()),
                    review_images=review_images(case),
                    result_sha256=file_hash(case / 'result.json'), audit_sha256=file_hash(case / 'independent_audit.json'))


class ToolSession:
    def __init__(self, root, config, driver=None):
        self.root = Path(root).resolve()
        self.config = config
        self.driver = driver or LocalDriver(self.root)
        self.path = self.root / 'tool_state.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(
            config_sha256=digest(config), skill_read=False, inspected=False, prepared={})
        require(self.state['config_sha256'] == digest(config), 'Session config changed', 'integrity_error')

    def persist(self):
        save(self.path, self.state)

    def expose_image(self, ident):
        row = self.state['prepared'][ident]
        require(file_hash(Path(row['case']) / 'initial.png') == row['initial_sha256'], 'Initial image changed', 'integrity_error')
        row['image_exposed'] = True
        self.persist()

    def complete(self):
        rows = list(self.state['prepared'].values())
        return bool(rows and all(r.get('result', {}).get('physics_steps', 0) > 0 and r.get('reflection') for r in rows)
                    and rows[-1]['reflection']['action'] in ('accept', 'stop'))

    def dispatch(self, name, args, provenance=None):
        from ..workflow.navigation.archive import verify_evidence
        verify_evidence(self.config.get('input_sha256', {}))
        require(name in PARAMETERS and isinstance(args, dict) and set(args) == set(PARAMETERS[name]), 'Tool/argument fields')
        for key, value in args.items():
            if key != 'wrist_raise_m':
                require(isinstance(value, str) and bool(value.strip()), 'Nonempty text required')
        if name == 'read_skill':
            self.state['skill_read'] = True
            self.persist()
            return dict(skill=SKILL.read_text(), sha256=file_hash(SKILL))
        require(self.state['skill_read'], 'Read skill before acting')
        if name == 'inspect_scene':
            from ..privileged import no_physics
            from ..workflow.navigation.lateral_prior import load_lateral_source
            from ..workflow.navigation.lateral import surface_frame
            import mujoco
            options = json.loads(Path(self.config['options']).read_text())
            with no_physics():
                trace, prior, _, _ = load_lateral_source(self.config['manifest'], options)
                model = mujoco.MjModel.from_binary_path(trace['files']['model'])
                surface = surface_frame(model, trace['qpos'][0], prior['bindings'])
            experience = []
            for path in self.config.get('experience_results', []):
                experience.append(experience_summary(path))
            result = dict(offset_m=self.config['offset_m'], source_control=prior['source_control'],
                          operation_interval=prior['operation_interval'], support_windows=[
                              {k: v for k, v in w.items() if k != 'qpos'} for w in prior['support_windows']],
                          surface=surface, experience=experience, original_actions_visible=True,
                          operation_kind='trajectory_retargeting', initialization='generated operation stance; no walking')
            save(self.root / 'scene_evidence.json', result)
            self.state['inspected'] = True
            self.persist()
            return result
        require(self.state['inspected'], 'Inspect scene and replay before planning')
        if name == 'plan_lateral':
            if not self.state.get('plan_file'):
                from ..workflow.navigation.lateral import plan_lateral
                plan = plan_lateral(self.config['manifest'], self.config['initial'],
                                    json.loads(Path(self.config['options']).read_text()), self.root / 'planning',
                                    self.config['offset_m'])
                require(plan['status'] == 'completed' and plan['selected']['feasible'], 'No checked stance/operation plan')
                path = self.root / 'planning/plan.json'
                self.state.update(plan_file=str(path), plan_sha256=file_hash(path), selected_id=plan['selected']['id'])
                self.persist()
            plan = json.loads(Path(self.state['plan_file']).read_text())
            return dict(plan_file=self.state['plan_file'], selected_id=self.state['selected_id'],
                        travel_m=plan['selected']['travel_m'], search_incomplete=plan['search_incomplete'],
                        right_wrist_max_error_m=max(v['pose_errors']['right_wrist_yaw_link']['position_m']
                                                   for v in plan['selected']['verification']),
                        initialization='This experiment initializes at the selected operation stance; navigation is unexecuted')
        require(self.state.get('plan_file'), 'Plan before preparing or executing')
        require(file_hash(self.state['plan_file']) == self.state['plan_sha256'], 'Plan changed', 'integrity_error')
        rows = self.state['prepared']
        if name == 'prepare_grasp':
            value = validate_raise(args['wrist_raise_m'])
            require(args['candidate_id'] == self.state['selected_id'], 'Candidate differs from checked plan')
            require(len(rows) < self.config['max_attempts'], 'Physical attempt budget exhausted')
            require(not rows or list(rows.values())[-1].get('reflection', {}).get('action') == 'retry',
                    'Previous attempt must be reflected with retry before another preparation')
            proof = dict(provenance or {}, decision=args)
            case = self.driver.prepare(self.state['plan_file'], value, proof)
            ident = digest({'case': str(case), 'contract': file_hash(case / 'experiment.json')})[:24]
            rows[ident] = dict(case=str(case), contract_sha256=file_hash(case / 'experiment.json'),
                               initial_sha256=file_hash(case / 'initial.png'), image_exposed=False, started=False)
            self.persist()
            return dict(prepared_id=ident, case=str(case), initial_image=str(case / 'initial.png'),
                        contract_sha256=rows[ident]['contract_sha256'], wrist_raise_m=value, physics_steps=0)
        require(args['prepared_id'] in rows, 'Prepared case outside this session')
        row = rows[args['prepared_id']]
        case = Path(row['case'])
        require(file_hash(case / 'experiment.json') == row['contract_sha256'], 'Prepared contract changed', 'integrity_error')
        if name == 'execute_grasp':
            if row.get('result'):
                result = row['result']
                for path, key in ((case / 'result.json', 'result_sha256'), (case / 'independent_audit.json', 'audit_sha256')):
                    if key in result:
                        require(file_hash(path) == result[key], 'Executed evidence changed', 'integrity_error')
                return result
            require(row['image_exposed'], 'Initial image must be exposed to the model before launch')
            require(args['prepared_id'] in (provenance or {}).get('observed_prepared_ids', []),
                    'Initial preparation/image must be in an earlier API request')
            require(file_hash(case / 'initial.png') == row['initial_sha256'], 'Initial image changed', 'integrity_error')
            require(not row['started'], 'Interrupted execution: explicit inspection required, no implicit rerun')
            save(case / 'initial_visual_review.json', dict(source='model API', review=args['initial_review'],
                 prepared_id=args['prepared_id'], image_sha256=row['initial_sha256'], at=now()))
            save(case / 'launch_authorization.json', dict(provenance=provenance or {}, arguments=args,
                 contract_sha256=row['contract_sha256'], initial_sha256=row['initial_sha256'], at=now()))
            save(case / 'launch_authorization_sha256.json', dict(sha256=file_hash(case / 'launch_authorization.json')))
            row['started'] = True
            self.persist()
            row['result'] = self.driver.execute(case)
            self.persist()
            return row['result']
        require(name == 'reflect_grasp' and row.get('result'), 'Execute before reflection')
        require(args['prepared_id'] in (provenance or {}).get('observed_result_ids', []),
                'Measured result must be in an earlier API request before reflection')
        require(args['action'] in ('accept', 'retry', 'stop'), 'Unknown reflection action')
        require(args['action'] != 'accept' or (row['result']['native_success'] and row['result']['clean_success']),
                'Acceptance contradicts physical native/collision evidence')
        row['reflection'] = dict(args, recorded_at=now(), provenance=provenance or {})
        save(case / 'api_reflection.json', row['reflection'])
        self.persist()
        return dict(recorded=True, action=args['action'], workflow_ready_to_finish=self.complete(),
                    remaining_preparations=self.config['max_attempts'] - len(rows))
