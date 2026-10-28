"""Model outputs never determine success; only timestamped physical evidence does."""
import copy
import math
import shutil
import time
from pathlib import Path
from .io import digest, read, write
from .contact_prior import extract_prior, enrich_request
from .diagnostics import stage_diagnosis, validate_reflection, check_mode


def validate_task(task):
    for key in ('name', 'backend', 'scene', 'acceptance', 'budget'):
        if key not in task:
            raise ValueError('Missing task field: ' + key)
    kind=task['acceptance'].get('kind','lift')
    if kind not in ('lift','native'):raise ValueError('Unknown acceptance kind')
    if kind=='native' and not task['acceptance'].get('validator_id'):raise ValueError('Native validator identity required')
    for key in (('hold_s',) if kind=='native' else ('lift_m', 'hold_s')):
        value = task['acceptance'][key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Acceptance must be positive and finite')
    b = task['budget']
    if not isinstance(b.get('feedback_calls', 8), int) or not 0 <= b.get('feedback_calls', 8) <= 100:
        raise ValueError('Invalid feedback call budget')
    if not isinstance(b['attempts'], int) or not 1 <= b['attempts'] <= 1000:
        raise ValueError('Invalid attempt budget')
    if not math.isfinite(b['sim_seconds']) or b['sim_seconds'] <= 0:
        raise ValueError('Invalid execution deadline')
    if not isinstance(b.get('optimization_attempts', 0), int) or b.get('optimization_attempts', 0) < 0:
        raise ValueError('Invalid optimization budget')
    checks=task.get('policy_checks',{})
    if not isinstance(checks,dict) or set(checks)-{'require_reflection','sonic_mode'}:
        raise ValueError('Invalid policy_checks')
    if 'require_reflection' in checks and not isinstance(checks['require_reflection'],bool):
        raise ValueError('require_reflection must be boolean')
    if 'sonic_mode' in checks and checks['sonic_mode'] != 1:
        raise ValueError('Only explicit mode1 enforcement is supported')


def evaluate(samples, criteria):
    """One evidence row represents its preceding physics interval, in seconds."""
    native=criteria.get('kind')=='native'
    elapsed = longest = peak = 0.
    previous = 0.
    for row in samples:
        keys=('time','dt') if native else ('time', 'dt', 'lift_m', 'support_force', 'hand_force')
        if any(k not in row for k in keys):raise ValueError('Missing physical evidence')
        values = [row[k] for k in keys]
        if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in values):
            raise ValueError('Non-finite or missing physical evidence')
        if row['dt'] <= 0 or row['time'] <= previous or abs(row['time'] - previous - row['dt']) > 1e-6:
            raise ValueError('Non-contiguous physics timestamps')
        if not isinstance(row.get('stable'), bool):
            raise ValueError('stable must be a measured boolean')
        previous = row['time']
        if native and not isinstance(row.get('native_success'),bool):raise ValueError('Measured native_success required')
        good = row['native_success'] and row['stable'] if native else (row['lift_m'] >= criteria['lift_m'] and row['support_force'] < .1
                and row['hand_force'] > .05 and row['stable'])
        elapsed = elapsed + row['dt'] if good else 0.
        longest = max(longest, elapsed)
        peak = max(peak, row.get('lift_m',0.))
    return {'success': longest + 1e-9 >= criteria['hold_s'], 'max_lift_m': peak,
            'max_hold_s': longest, 'sim_seconds': previous, 'criteria': copy.deepcopy(criteria)}


def better(metrics, baseline):
    return (metrics['roughness'] <= baseline['roughness'] * .9
            and metrics['root_omega_rms'] <= baseline['root_omega_rms'] * 1.05 + 1e-12
            and metrics['peak_table_force'] <= max(5., baseline['peak_table_force']))


def run_task(task, planner, backend, directory, resume=False):
    validate_task(task)
    task = copy.deepcopy(task)
    checks=task.get('policy_checks',{})
    strict_reflection=checks.get('require_reflection',False)
    directory = Path(directory)
    fingerprint = digest(task)
    if directory.exists():
        if not resume:
            raise FileExistsError(directory)
        if read(directory / 'task.json') != task:
            raise ValueError('Resume cannot change task, scene, acceptance or budget')
    else:
        directory.mkdir(parents=True)
        write(directory / 'task.json', task)
    prefixes, history, successes = [], [], []
    best = None
    optimized = 0
    for p in sorted((directory / 'attempts').glob('*')):
        if not (p / 'result.json').exists():
            raise RuntimeError('Interrupted attempt requires inspection; refusing a silent rerun: ' + str(p))
        result = read(p / 'result.json')
        if strict_reflection and not result['acceptance']['success']:
            if not (p/'reflection.json').exists():
                raise ValueError('Required reflection missing on resume; inspect before retry')
            validate_reflection(read(p/'reflection.json'),result,read(p/'diagnostics.json'),backend.interface)
        history.append(result)
        if (p / 'prefix.json').exists():
            prefixes.append(read(p / 'prefix.json'))
        if result['acceptance']['success']:
            artifact = directory / 'success' / f"attempt_{result['attempt']:03d}" / 'trajectory.json'
            if not artifact.exists():
                raise RuntimeError('Missing successful trajectory: ' + str(artifact))
            trajectory = read(artifact)
            evidence = read(p / 'samples.json')
            if (digest(evidence) != trajectory['samples_sha256'] or
                    trajectory['task'] != task or
                    evaluate(evidence, task['acceptance']) != result['acceptance'] or
                    trajectory['acceptance'] != result['acceptance']):
                raise RuntimeError('Successful trajectory evidence does not match')
            successes.append(result['attempt'])
        if result.get('selected'):
            best = result
        optimized += result['phase'] == 'optimize'
    references = [read(p) for p in task.get('references', [])]
    skills = [read(p) for p in task.get('skills', [])]
    inputs = {'references': references, 'skills': skills}
    contact_priors = []
    if task.get('contact_prior') and task.get('generation_kind') != 'reference_recovery':
        if not task.get('references'):
            raise ValueError('Contact prior requires a physically validated reference')
        contact_priors = [extract_prior(p, **task['contact_prior']) for p in task['references']]
        inputs['contact_priors'] = contact_priors
    frozen_inputs = directory / 'experience.json'
    if frozen_inputs.exists() and read(frozen_inputs) != inputs:
        raise RuntimeError('Reference or frozen skill changed during this run')
    write(frozen_inputs, inputs)
    if contact_priors:write(directory/'contact_priors.json', contact_priors)
    started = time.time()
    for number in range(len(history) + 1, task['budget']['attempts'] + 1):
        phase = 'optimize' if best else 'solve'
        if best and optimized >= task['budget'].get('optimization_attempts', 0):
            break
        previous_reflection=None
        if strict_reflection and history and not history[-1]['acceptance']['success']:
            prior=directory/'attempts'/f"{history[-1]['attempt']:03d}"
            previous_reflection=validate_reflection(read(prior/'reflection.json'),history[-1],read(prior/'diagnostics.json'),backend.interface)
        request = {'task': copy.deepcopy(task), 'interface': backend.interface,
                   'attempt': number, 'phase': phase, 'references': copy.deepcopy(references), 'skills': copy.deepcopy(skills),
                   'history': copy.deepcopy(history), 'prefixes': copy.deepcopy(prefixes),
                   'successful_baseline': copy.deepcopy(best),
                   'successful_trajectory': read(directory / 'success' / f"attempt_{best['attempt']:03d}" / 'trajectory.json') if best else None,
                   'observation': backend.describe(copy.deepcopy(task)) if hasattr(backend, 'describe') else {}}
        if previous_reflection is not None:
            request['previous_reflection']=copy.deepcopy(previous_reflection)
            request['previous_reflection_sha256']=digest(previous_reflection)
        enrich_request(request, contact_priors)
        out = directory / 'attempts' / f'{number:03d}'
        out.mkdir(parents=True)
        write(out / 'request.json', request)
        attempt_started = time.time()
        decision = planner.plan(request)
        planning_wall_s = time.time() - attempt_started
        analysis = decision.get('analysis', {})
        if not analysis.get('evidence') or not analysis.get('hypothesis'):
            raise ValueError('Record evidence and diagnosis BEFORE retry/reset')
        if previous_reflection is not None:
            if analysis.get('reflection_sha256') != digest(previous_reflection) or not analysis.get('change'):
                raise ValueError('Retry must acknowledge exact previous reflection and specify change')
        actions = decision.get('actions')
        if not isinstance(actions, list) or not actions or len(actions) > 200:
            raise ValueError('A bounded action program is required')
        reused = decision.get('reuse_prefix')
        prefix_actions = []
        if reused:
            selected = next((p for p in prefixes if p['id'] == reused), None)
            if selected is None or selected['task_digest'] != fingerprint:
                raise ValueError('Prefix must belong to this exact task and scene')
            prefix_actions = copy.deepcopy(selected['actions'])
        actions = prefix_actions + actions
        check_mode(actions,checks)
        if len(actions) > 200:
            raise ValueError('Prefix plus program exceeds action budget')
        decision['recorded_unix_s'] = time.time()
        write(out / 'decision.json', decision)
        write(out / 'program.json', actions)
        reset_time = time.time()
        observation = backend.reset(copy.deepcopy(task), out)
        write(out / 'reset.json', {'time': reset_time, 'observation': observation})
        samples, executed, valid_prefix, diagnoses = [], [], [], []
        failure = None
        acceptance = evaluate([], task['acceptance'])
        queue = copy.deepcopy(actions)
        feedback_index = 0
        while queue:
            if len(executed) >= 200:
                failure = 'action_budget'
                break
            action = queue.pop(0)
            replan = action.pop('replan_after', False) if isinstance(action, dict) else False
            remaining = task['budget']['sim_seconds'] - acceptance['sim_seconds']
            if remaining <= 1e-9:
                failure = 'execution_deadline'
                break
            before=copy.deepcopy(observation)
            try:
                if not isinstance(action, dict):raise ValueError('Action must be an object')
                if 'wrist_offset' in action:
                    import numpy as np
                    offset = np.asarray(action.pop('wrist_offset'), dtype=float)
                    if offset.shape != (3,) or not np.isfinite(offset).all():
                        raise ValueError('wrist_offset must be finite xyz')
                    target_key = {'mujoco_gripper':'position','humano':'right_position'}.get(task['backend'],'wrist_position')
                    if target_key in action:
                        raise ValueError('Use either absolute wrist target or offset')
                    action[target_key] = (np.asarray(observation['wrist_position']) + offset).tolist()
                result = backend.execute(copy.deepcopy(action), remaining)
            except (ValueError, KeyError, TypeError) as exc:
                result = {'completed': False, 'reusable': False,
                          'failure': 'invalid_action: ' + str(exc),
                          'samples': [], 'observation': observation}
            diagnosis=stage_diagnosis(action if isinstance(action,dict) else {},before,result)
            diagnoses.append(diagnosis)
            write(out/f'diagnosis_{len(diagnoses):03d}.json',diagnosis)
            modes=(result.get('controller_diagnostics') or {}).get('actual_modes',[])
            if checks.get('sonic_mode')==1 and modes and modes!=[1]:
                backend.export(out)
                raise RuntimeError('Measured controller mode violates mode1 policy')
            samples.extend(result['samples'])
            executed.append(action)
            write(out / f'stage_{len(executed):03d}.json', {k: v for k, v in result.items() if k != 'samples'})
            acceptance = evaluate(samples, task['acceptance'])
            if acceptance['sim_seconds'] > task['budget']['sim_seconds'] + 1e-6:
                raise ValueError('Backend exceeded fixed execution deadline')
            if result['completed'] and result['reusable'] and not result['failure']:
                valid_prefix = copy.deepcopy(executed)
            observation = result['observation']
            failure = result['failure']
            if acceptance['success'] or failure:
                break
            if replan:
                if feedback_index >= task['budget'].get('feedback_calls', 8):
                    failure = 'feedback_budget'
                    break
                feedback_index += 1
                feedback_request = {**request, 'feedback_index': feedback_index,
                                    'observation': copy.deepcopy(observation),
                                    'executed_actions': copy.deepcopy(executed),
                                    'remaining_actions': copy.deepcopy(queue),
                                    'remaining_sim_seconds': task['budget']['sim_seconds'] - acceptance['sim_seconds']}
                feedback_request['diagnostics']={'last_stage':copy.deepcopy(diagnosis),'stages':copy.deepcopy(diagnoses)}
                enrich_request(feedback_request, contact_priors)
                write(out / f'feedback_{feedback_index:03d}_request.json', feedback_request)
                update = planner.plan(feedback_request)
                write(out / f'feedback_{feedback_index:03d}_decision.json', update)
                analysis = update.get('analysis', {})
                new_actions = update.get('actions')
                if (not analysis.get('evidence') or not analysis.get('hypothesis') or
                        update.get('reuse_prefix') or not isinstance(new_actions, list) or
                        not new_actions or len(new_actions) + len(executed) > 200):
                    failure = 'invalid_feedback_program'
                    break
                queue = copy.deepcopy(new_actions)
                check_mode(queue,checks)
        backend.export(out)
        if hasattr(backend,'verify'):
            independent=backend.verify(out)
            write(out/'independent_validation.json',independent)
            if independent['success'] != acceptance['success']:
                raise RuntimeError('Offline physical validation disagrees with online success')
        metrics = backend.metrics()
        if not all(math.isfinite(metrics[k]) and metrics[k] >= 0
                   for k in ('roughness', 'root_omega_rms', 'peak_table_force')):
            raise ValueError('Invalid measured optimization metrics')
        chosen = acceptance['success'] and (best is None or better(metrics, best['metrics']))
        result = {'attempt': number, 'phase': phase, 'acceptance': acceptance,
                  'failure': failure or (None if acceptance['success'] else 'task_not_completed'),
                  'observation': observation, 'metrics': metrics, 'selected': chosen,
                  'wall_s': time.time() - attempt_started, 'initial_planning_wall_s': planning_wall_s, 'feedback_calls': feedback_index, 'reused_prefix': reused}
        write(out / 'samples.json', samples)
        if valid_prefix:
            prefix = {'id': f'attempt_{number:03d}', 'task_digest': fingerprint,
                      'actions': valid_prefix, 'source': str(out), 'task_success': acceptance['success']}
            prefixes.append(prefix)
            write(out / 'prefix.json', prefix)
        if acceptance['success']:
            successes.append(number)
            target = directory / 'success' / f'attempt_{number:03d}'
            target.mkdir(parents=True, exist_ok=True)
            for filename in ('samples.json', 'states.npz', 'contacts.json', 'scene.xml', 'scene.mjb', 'scene_source.xml', 'independent_validation.json'):
                if (out / filename).exists():shutil.copy2(out / filename, target / filename)
            write(target / 'trajectory.json', {'schema_version': 1, 'task': task,
                  'actions': executed, 'acceptance': acceptance, 'metrics': metrics,
                  'source_attempt': str(out.resolve()), 'samples_sha256': digest(samples),
                  'generation_kind': task.get('generation_kind', 'generated'),
                  'reference_sources': task.get('references', []), 'experience_sha256': digest(inputs)})
            if chosen:
                best = result
        write(out / 'result.json', result)  # Commit marker, after all evidence and success artifacts.
        diagnostic_packet={'attempt':number,'last_stage':diagnoses[-1] if diagnoses else None,'stages':diagnoses}
        write(out/'diagnostics.json',diagnostic_packet)
        if strict_reflection and not acceptance['success']:
            reflect_request={'request_kind':'reflection','attempt':number,'task':copy.deepcopy(task),
                             'interface':backend.interface,'result':copy.deepcopy(result),'result_sha256':digest(result),
                             'diagnostics':diagnostic_packet,'prior_reflection':previous_reflection,
                             'evidence_directory':str(out.resolve()),'actions':copy.deepcopy(executed)}
            write(out/'reflection_request.json',reflect_request)
            if not hasattr(planner,'reflect'):raise ValueError('Required reflection planner unavailable')
            reflect_started=time.time()
            reflected=planner.reflect(reflect_request)
            write(out/'reflection_candidate.json',reflected)
            validate_reflection(reflected,result,diagnostic_packet,backend.interface)
            write(out/'reflection.json',reflected)
            write(out/'reflection_timing.json',{'wall_s':time.time()-reflect_started})
        optimized += phase == 'optimize'
        history.append(result)
        write(directory / 'progress.json', {'attempts': len(history), 'successes': successes,
                                           'accepted_attempt': best['attempt'] if best else None})
    summary = {'success': best is not None, 'accepted_attempt': best['attempt'] if best else None,
               'successes': successes, 'attempts': len(history), 'optimization_attempts': optimized,
               'wall_s_this_invocation': time.time() - started,
               'status': 'success' if best else 'budget_exhausted', 'task_digest': fingerprint}
    write(directory / 'summary.json', summary)
    return summary
