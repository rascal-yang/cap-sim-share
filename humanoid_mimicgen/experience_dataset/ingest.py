"""Read-only, task-agnostic adapters for persisted attempts and partial evidence."""
import json
import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from .schema import digest, new_record, now
from .store import atomic, file_hash

PRUNE = {'__pycache__', '.git', '.venv', 'venv', 'conda', 'site-packages', 'node_modules',
         'dependencies', 'runtime_dependencies', 'source_snapshot', 'source_snapshots',
         'assets', 'meshes', 'dataset', 'datasets', 'videos', 'video', 'renders',
         'render', 'success', 'successes', '_index', 'checks', 'models', 'weights', 'extscache'}
MARKERS = {'result.json', 'experiment.json', 'preflight_failure.json', 'execution_audit.json',
           'decision.json', 'failure.json', 'request.json'}
SMALL_FILES = ('result.json', 'experiment.json', 'preflight_failure.json', 'failure.json', 'summary.json',
               'task.json', 'request.json', 'decision.json', 'compiled_plan.json', 'plan.json',
               'token_replay_audit.json', 'strategy.json', 'reflection.json',
               'execution/execution_audit.json', 'execution/metrics.json',
               'execution/actual_providers.json', 'physics.json', 'runtime.json',
               'independent_validation.json', 'execution/contact_control.json')
PRIVATE = re.compile(r'api.?key|password|authorization|credential|secret|access.?token|headers|environment', re.I)
ADAPTER_VERSION = 6
SUPPORT_FILES = ('execution/observations.json', 'execution/commands.json', 'execution/states.npz',
                 'execution/trajectory.npz', 'execution/controller.npz', 'execution/hand_commands.npz',
                 'states.npz', 'contacts.json', 'samples.json')


def compact(value, depth=0):
    """Bound indexed facts; dense source remains available by evidence reference."""
    if depth > 7: return {'omitted': 'depth limit; see evidence'}
    if isinstance(value, dict):
        return {k: compact(v, depth+1) for k, v in value.items() if k not in ('source_sha256', 'source_snapshot_and_hashes')}
    if isinstance(value, list):
        if len(value) > 12:
            return {'count': len(value), 'first': compact(value[:2], depth+1), 'last': compact(value[-2:], depth+1), 'truncated': True}
        return [compact(v, depth+1) for v in value]
    if isinstance(value, str) and len(value) > 3000: return value[:3000]+' [truncated; see evidence]'
    return value


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if not PRIVATE.search(k)}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'(?i)Bearer\s+\S+|sk-[A-Za-z0-9_-]+|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9_]+', '[REDACTED]', value)
        value = re.sub(r'(https?://)[^/\s@]+@', r'\1[REDACTED]@', value)
    return value


class Resolver:
    def __init__(self, roots):
        self.roots = [Path(r).resolve() for r in roots]
        self.moves = {}
        self.observed = {}
        for root in self.roots:
            latest = root/'_index/latest-migration.json'
            if latest.is_file():
                target = Path(json.loads(latest.read_text())['path_map'])
                if self.allowed(target) and target.is_file(): self.moves.update(json.loads(target.read_text()))

    def allowed(self, path):
        p = Path(path).resolve()
        return any(p.is_relative_to(root) for root in self.roots)

    def resolve(self, path, base=None, *, existing=True):
        if not isinstance(path, (str, Path)): return None
        p = Path(path)
        if not p.is_absolute(): p = Path(base or self.roots[0])/p
        if not p.exists():
            for old, new in sorted(self.moves.items(), key=lambda x: -len(x[0])):
                if str(p) == old or str(p).startswith(old+'/'):
                    p = Path(new)/str(p)[len(old):].lstrip('/'); break
        return p.resolve() if self.allowed(p) and (not existing or p.is_file()) else None

    def identity(self, case):
        case = Path(case).resolve()
        # Stable through the recorded migration, independent of current mount prefix.
        for old, new in sorted(self.moves.items(), key=lambda x: -len(x[1])):
            if str(case) == new or str(case).startswith(new+'/'):
                return 'experiments/'+Path(old).name+str(case)[len(new):]
        for root in self.roots:
            if case.is_relative_to(root): return root.name+'/'+case.relative_to(root).as_posix()
        raise ValueError('Source outside allowed roots')


def inventory(roots):
    cases = set(); summaries = set(); owners = set()
    for root in roots:
        root = Path(root).resolve()
        for folder, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in PRUNE and not d.startswith('.') and not (Path(folder)/d).is_symlink())
            names = set(files)
            if 'summary.json' in names: summaries.add(Path(folder))
            if re.match(r'^20\d{6}(?:_|$)', Path(folder).name): owners.add(Path(folder))
            if names & MARKERS:
                path = Path(folder)
                if path.name == 'execution': path = path.parent
                cases.add(path)
    for path in sorted(summaries | owners, key=lambda p: -len(p.parts)):
        if not any(child == path or child.is_relative_to(path) or path.is_relative_to(child) for child in cases):
            cases.add(path)
    return sorted(cases)


def read(path, resolver, *, limit=8*1024*1024):
    path = resolver.resolve(path)
    if path is None: return None
    if path.stat().st_size > limit: return None
    raw = path.read_bytes()
    resolver.observed[str(path)] = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    return clean(value)


def fields(case, resolver):
    files = [case/name for name in SMALL_FILES]
    files += sorted(case.glob('feedback_*_decision.json'))
    files += sorted(case.glob('feedback_*_request.json'))
    files += sorted(case.glob('diagnosis_*.json'))
    files += sorted(case.glob('stage_*.json'))
    files += sorted((case/'execution').glob('retry_*_reflection.json'))
    return [p for p in files if resolver.resolve(p)]


def tri(value):
    return value if type(value) is bool else None


def obj(value):
    return value if isinstance(value, dict) else {}


def adapt(case, resolver):
    case = Path(case).resolve()
    resolver.observed = {}
    documents = {p.relative_to(case).as_posix(): read(p, resolver) for p in fields(case, resolver)}
    result = documents.get('result.json') or {}
    contract = documents.get('experiment.json') or {}
    audit = documents.get('execution/execution_audit.json') or {}
    if not all(isinstance(x, dict) for x in (result, contract, audit)):
        raise ValueError('Expected object-shaped result/contract/audit')
    task = documents.get('task.json') or {}; task_path = case/'task.json'
    if not task:
        for parent in list(case.parents)[:4]:
            if not resolver.allowed(parent): break
            task = read(parent/'task.json', resolver) or {}
            if task:
                task_path = parent/'task.json'; break
    task_id = result.get('task') or task.get('name') or task.get('id')
    if isinstance(task_id, dict): task_id = task_id.get('name')
    if not task_id:
        task_id = next((part for part in case.parts if part.startswith('G1Wholebody')), None)
    if not task_id:
        matches = re.findall(r'G1Wholebody[A-Za-z0-9]+', json.dumps(contract.get('inputs', {})))
        task_id = matches[0] if matches else None
    producer = ('suite' if 'mode' in result and 'physics_steps' in result else
                'workflow' if 'workflow' in contract or 'physical_task_completed' in result else
                'core' if 'attempt' in result and 'acceptance' in result else 'unknown')
    r = new_record(resolver.identity(case), task=task_id, producer=producer)
    if not result and not contract and documents.get('summary.json') is not None:
        r['record_kind'] = 'report'
        r['notes'].append('Summary-only source; not an independent physical attempt')
    r['experiment_id'] = result.get('experiment_id') or contract.get('experiment_id')
    ctx = r['context']; out = r['outcome']
    robot = contract.get('robot') or task.get('robot', {})
    ctx['robot'] = robot.get('name') if isinstance(robot, dict) else robot
    ctx['hand'] = robot.get('hand') if isinstance(robot, dict) else None
    robot_description = obj(contract.get('physics')).get('robot_and_hand')
    if isinstance(robot_description, str):
        if re.search(r'\bG1\b', robot_description): ctx['robot'] = 'G1'
        if re.search(r'\bDex3\b', robot_description, re.I): ctx['hand'] = 'Dex3'
        ctx['robot_and_hand_source'] = robot_description
    if not ctx['robot'] and isinstance(task_id, str) and task_id.startswith('G1Wholebody'):
        ctx['robot'] = 'G1'
        ctx['robot_source'] = 'native task identifier G1Wholebody'
    ctx['interface'] = result.get('mode') or task.get('controller') or task.get('backend')
    if producer == 'workflow': ctx['interface'] = 'workflow_mode1'
    ctx['actual_modes'] = result.get('encoder_modes', result.get('actual_modes'))
    if ctx['actual_modes'] is not None and result.get('original_token_steps', 0):
        ctx['actual_modes'] = ['original_token'] + list(ctx['actual_modes'])
    pipeline = obj(contract.get('pipeline'))
    implementation = pipeline.get('git_commit_and_dirty_patch')
    sources = pipeline.get('source_snapshot_and_hashes')
    if isinstance(implementation, dict): sources = sources or implementation.get('source_sha256')
    models = obj(contract.get('runtime')).get('model_hashes') or obj(documents.get('runtime.json')).get('model_sha256')
    ctx['controller_version'] = digest({'implementation': implementation, 'sources': sources, 'models': models}) if sources else None
    ctx['git_commit'] = implementation.get('commit') if isinstance(implementation, dict) else implementation
    ctx['stage'] = result.get('stage') or result.get('phase') or audit.get('stage')
    ctx['scene'] = {'object_delta_xy_m': result.get('object_delta_xy_m'), 'scene': task.get('scene')}
    if producer in ('suite', 'workflow'):
        ctx.update(method='trajectory_retargeting', original_actions_accessible=True)
    elif task.get('generation_kind'):
        ctx['method'] = task['generation_kind']
        ctx['original_actions_accessible'] = bool(task.get('references')) if 'references' in task else None
    acceptance = result.get('acceptance', {})
    out['task_success'] = tri(result.get('success', result.get('physical_task_completed', acceptance.get('success'))))
    out['native_success'] = tri(result.get('native_success', (result.get('native') or {}).get('success')))
    out['approved_alternative_success'] = tri(result.get('reviewed_success'))
    out['training_ready'] = tri(result.get('training_ready'))
    out['execution_complete'] = tri(audit.get('completed', result.get('completed')))
    if 'coverage' in result: out['execution_complete'] = result['coverage'] == 1
    failure = result.get('failure') or audit.get('failure') or result.get('reason_code')
    if documents.get('preflight_failure.json') is not None:
        r['status'] = 'preflight_rejected'
    elif result.get('status') in ('error', 'failed') and (result.get('error_type') or result.get('error')):
        r['status'] = 'infrastructure_error'
    elif result.get('status') == 'interrupted': r['status'] = 'interrupted'
    elif failure or out['task_success'] is False: r['status'] = 'physical_failure' if producer != 'unknown' else 'unknown'
    elif out['task_success'] is True or result.get('status') == 'completed': r['status'] = 'completed'
    elif not result and contract: r['status'] = 'unknown'  # No process liveness assumptions.
    r['metrics'] = compact({'failure': failure, 'result': {k: result[k] for k in
        ('physics_steps', 'coverage', 'hold_continuous_s', 'hold_min_clearance_m', 'max_obstacle_force_N', 'wrist_error_rmse_m', 'evidence_summary') if k in result},
        'measured': result.get('metrics') or documents.get('execution/metrics.json')})
    r['criteria'] = {'native': result.get('native_acceptance'), 'acceptance': acceptance or None,
                     'contract': contract.get('acceptance') or obj(contract.get('physics')).get('acceptance_thresholds')}
    r['timing'].update(started_at=result.get('started_at') or audit.get('started_at'),
                       ended_at=result.get('ended_at') or audit.get('ended_at'),
                       planning_s=result.get('initial_planning_wall_s'), simulation_s=result.get('simulation_wall_s', audit.get('simulation_wall_s')),
                       rendering_s=result.get('rendering_wall_s'), total_s=result.get('total_wall_s', result.get('wall_s')))
    # Evidence is hashed, but unrestricted contents are never sent to retrieval clients.
    def bind(path, role):
        path = resolver.resolve(path, case, existing=False)
        if path is None:
            r['missing'].append('evidence:'+role); return
        if not path.is_file():
            r['missing'].append('evidence:'+role)
            r['evidence'].append({'path':str(path), 'sha256':None, 'role':role})
            return
        sha = resolver.observed.get(str(path)) or file_hash(path)
        if not any(e['path'] == str(path) for e in r['evidence']):
            r['evidence'].append({'path': str(path), 'sha256': sha, 'role': role})
    for path in fields(case, resolver):
        bind(path, path.relative_to(case).as_posix())
    if task and resolver.resolve(task_path) and not any(e['path'] == str(task_path) for e in r['evidence']):
        bind(task_path, 'inherited_task')
    for name, source in (result.get('evidence') or {}).items():
        if isinstance(source, str): bind(source, 'referenced:'+name)
    for name in SUPPORT_FILES:
        if resolver.resolve(case/name): bind(case/name, 'measured:'+name)
    declared = obj(contract.get('preflight')).get('evidence_sha256') or {}
    for evidence in r['evidence']:
        expected = declared.get(evidence['path'])
        if expected:
            evidence['declared_sha256'] = expected
            if expected != evidence['sha256']:
                r['notes'].append('Frozen declared evidence hash mismatch: '+evidence['role'])
    decision_path = resolver.resolve(result.get('decision_path'), case)
    decision_doc = read(decision_path, resolver) if decision_path else documents.get('compiled_plan.json') or documents.get('decision.json') or documents.get('plan.json')
    if not decision_doc and producer == 'workflow':
        strategy = obj(obj(contract.get('workflow')).get('strategy'))
        decision_doc = {'decision': {'policy': strategy.get('policy'), 'phases': strategy.get('phases'),
                                    'selected': obj(contract.get('workflow')).get('selection')},
                        'origin': {'kind': 'recorded_workflow_strategy'}}
    if decision_path and all(e['path'] != str(decision_path) for e in r['evidence']):
        bind(decision_path, 'decision')
    decisions = [('initial', decision_doc)] + [(name, val) for name, val in documents.items() if 'feedback_' in name and 'decision' in name]
    for name, doc in decisions:
        if not isinstance(doc, dict): continue
        instruction = doc.get('decision', doc)
        if not isinstance(instruction, dict): continue
        if not result and instruction.get('action') in ('reject', 'needs_tool'):
            r['status'] = 'planning_rejected'; r['record_kind'] = 'proposal'
        elif not result and not contract:
            r['status'] = 'proposal'; r['record_kind'] = 'proposal'
        ident = r['attempt_id']+':'+name
        analysis = instruction.get('analysis', {})
        request_name = name.replace('_decision.json','_request.json') if name != 'initial' else 'request.json'
        r['orchestration'].append({'instruction_id': ident, 'decision_origin': doc.get('origin', {}).get('kind', 'unknown'),
                                  'observation_before': compact(obj(documents.get(request_name)).get('observation')), 'diagnosis': compact(analysis) or None,
                                  'instruction': compact(instruction), 'rationale': instruction.get('rationale'),
                                  'expected_response': instruction.get('hypothesis'), 'stage': name})
    if producer != 'unknown':
        r['control'].append({'instruction_id': r['orchestration'][0]['instruction_id'] if r['orchestration'] else None,
            'interface': ctx['interface'], 'actual_modes': result.get('encoder_modes', result.get('actual_modes')),
            'original_token_steps': result.get('original_token_steps'), 'providers': result.get('actual_session_providers', documents.get('execution/actual_providers.json')),
            'stage_interfaces': compact(obj(contract.get('control')).get('stage_interfaces') or contract.get('control')),
            'response_class': 'unknown', 'response_reason': 'No aligned desired/measured delta with explicit frame, window and tolerance extracted',
            'measurements': r['metrics'], 'parameter_source': str(decision_path) if decision_path else None})
    for name, diagnostic in sorted(documents.items()):
        if not name.startswith('diagnosis_') or not isinstance(diagnostic, dict): continue
        index = int(name.split('_')[1].split('.')[0])
        instruction_id = r['orchestration'][0]['instruction_id'] if r['orchestration'] else None
        for event in r['orchestration'][1:]:
            request_name = event['stage'].replace('_decision.json','_request.json')
            request = obj(documents.get(request_name))
            if isinstance(request.get('executed_actions'), list) and len(request['executed_actions']) < index:
                instruction_id = event['instruction_id']
        r['control'].append({'instruction_id': instruction_id, 'stage': diagnostic.get('stage'),
            'desired': compact(diagnostic.get('targets')), 'measurements': compact(diagnostic.get('measurements')),
            'requested_mode': diagnostic.get('requested_mode'), 'controller': compact(diagnostic.get('controller')),
            'start_s': diagnostic.get('start_s'), 'end_s': diagnostic.get('end_s'),
            'response_class':'unknown', 'response_reason':'No declared response tolerance; preserve measured targets separately'})
    # Related verification artifacts cannot become independent successful strategies.
    replay = result.get('saved_token_source')
    checkpoint = result.get('rollback', {}) or {}
    r['relations'].update(replay_of=replay, checkpoint_source=checkpoint.get('case'))
    if replay or result.get('mode') == 'saved_token_replay': r['record_kind'] = 'verification'
    replay_audit = documents.get('token_replay_audit.json')
    if isinstance(replay_audit, dict): out['replay_verified'] = tri(replay_audit.get('exact'))
    source = replay or checkpoint.get('case')
    if source:
        p = resolver.resolve(Path(source)/'result.json', case)
        if p: r['relations']['parent'] = digest({'source_identity': resolver.identity(p.parent)})
    # Conservative family: all cases from one owning experiment share a family.
    owner = next((p for p in reversed([case, *case.parents]) if re.match(r'^20\d{6}(?:_|$)', p.name)), case)
    r['relations']['family'] = digest({'owner': resolver.identity(owner)})
    if decision_doc and isinstance(decision_doc, dict): r['relations']['shared_plan'] = decision_doc.get('origin', {}).get('api_batch_sha256')
    if task_id and type(result.get('episode')) is int:
        r['relations']['source_demonstration'] = str(task_id)+':episode:'+str(result['episode'])
    r['missing'] += [k for k in ('robot', 'hand', 'interface', 'controller_version') if not ctx.get(k)]
    if not task_id: r['missing'].append('task_id')
    if not r['orchestration']: r['missing'].append('orchestration')
    r['missing'].append('aligned_control_response')
    r['evidence_level'] = 'recorded_observation' if producer != 'unknown' else 'reported'
    if producer == 'unknown': r['notes'].append('Unmapped producer: preserve provenance, no inferred physical result')
    for evidence in r['evidence']:
        if evidence['sha256'] is not None and file_hash(evidence['path']) != evidence['sha256']:
            raise ValueError('Source changed during ingestion')
    return clean(r)


def ingest(dataset, roots):
    resolver = Resolver(roots)
    ledger_path = dataset.root/'ingestion'/'sources.json'
    old = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    ledger = dict(old); counts = Counter(); entries = []; tasks = Counter()
    for case in inventory(roots):
        key = resolver.identity(case)
        try:
            paths = fields(case, resolver)
            paths += [case/name for name in SUPPORT_FILES if resolver.resolve(case/name)]
            paths += [Path(p) for p in old.get(key, {}).get('dependencies', []) if resolver.resolve(p, existing=False)]
            for parent in list(case.parents)[:4]:
                task_path = resolver.resolve(parent/'task.json')
                if task_path:
                    paths.append(task_path); break
            result_path = resolver.resolve(case/'result.json')
            if result_path:
                result_data = read(result_path, resolver)
                if isinstance(result_data, dict):
                    dependency = resolver.resolve(result_data.get('decision_path'), case)
                    if dependency: paths.append(dependency)
            def fingerprint_for(paths):
                return digest({'adapter_version': ADAPTER_VERSION, 'files': {str(p): [p.stat().st_size, p.stat().st_mtime_ns] if p.is_file() else None for p in paths}})
            fingerprint = fingerprint_for(paths)
            prior = old.get(key, {})
            if prior.get('fingerprint') == fingerprint and prior.get('status') != 'error':
                counts['unchanged'] += 1
                entries.append(prior); tasks[prior.get('task_id') or 'unknown'] += 1
                continue
            r = adapt(case, resolver)
            expected = dict(prior.get('terminal_evidence', {}))
            # Canonical records survive a crash before ledger publication. Never let
            # loss of this disposable progress cache reset a terminal evidence baseline.
            folder = dataset.root/'records'/r['attempt_id']
            revisions = sorted(int(p.name) for p in folder.iterdir() if p.name.isdigit() and (p/'record.json').exists()) if folder.exists() else []
            for revision in revisions:
                previous = dataset.get(r['attempt_id'], revision)
                if previous['status'] not in ('completed', 'physical_failure'): continue
                for e in previous['evidence']:
                    if e['sha256'] is not None:
                        expected.setdefault(e['path'], {'path':e['path'], 'role':e['role'],
                                                       'sha256':e.get('expected_sha256',e['sha256'])})
            for p, original in expected.items():
                current = next((e for e in r['evidence'] if e['path'] == p), None)
                if current is None:
                    r['evidence'].append(dict(original, availability='missing_since_terminal_import'))
                    r['missing'].append('evidence:terminal:'+original['role'])
                else:
                    current['expected_sha256'] = original['sha256']
                    if current['sha256'] != original['sha256']:
                        r['notes'].append('Evidence changed since first terminal import: '+original['role'])
            if r['status'] in ('completed', 'physical_failure'):
                for e in r['evidence']:
                    if e['sha256'] is not None: expected.setdefault(e['path'], {k:e[k] for k in ('path','sha256','role')})
            identity = dataset.put(r)
            paths += [Path(e['path']) for e in r['evidence']]
            fingerprint = fingerprint_for(paths)
            status = 'partial' if r['producer'] == 'unknown' else 'imported'
            entry = dict(source=key, source_path=str(case), fingerprint=fingerprint, dependencies=sorted({str(p) for p in paths}), terminal_evidence=expected, status=status, task_id=r['task_id'], **identity)
            tasks[r['task_id'] or 'unknown'] += 1
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            entry = dict(old.get(key, {}), source=key, source_path=str(case), status='error', error_type=type(exc).__name__)
            status = 'error'
        ledger[key] = entry; entries.append(entry); counts[status] += 1
    atomic(ledger_path, ledger)
    report = {'created_at': now(), 'roots': [str(p) for p in resolver.roots], 'discovered': len(entries),
              'counts': dict(counts), 'tasks': dict(tasks), 'entries': entries,
              'coverage_definition': 'recognized attempts plus leaf summaries and unmapped dated experiment directories; partial/report records are not validated physical attempts',
              'physics_steps': 0, 'api_calls': 0}
    atomic(dataset.root/'ingestion'/'latest.json', report)
    return report
