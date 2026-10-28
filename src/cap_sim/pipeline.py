"""Public generation workflows: physically recover references before retargeting."""
import copy
import time
from pathlib import Path
from .core import run_task, validate_task
from .io import digest, read, write


def _recovery_task(task, reference_task):
    if reference_task is None:
        raise ValueError('References require reference_task with the original scene and initial state')
    validate_task(reference_task)
    # Scene and initial state may differ; controller, assets, permissions and
    # acceptance must not. Unknown task fields are compared conservatively.
    variable = {'name', 'scene', 'budget', 'references', 'skills', 'generation_kind', 'resources', 'contact_prior'}
    for key in set(task) | set(reference_task):
        if key not in variable and task.get(key) != reference_task.get(key):
            raise ValueError('Recovery and generation must use the same ' + key)
    resources = lambda t: {k:v for k,v in t.get('resources', {}).items() if k != 'initial_state'}
    if resources(task) != resources(reference_task):
        raise ValueError('Recovery and generation must use the same controller/assets resources')
    recovery = copy.deepcopy(reference_task)
    recovery['references'] = list(task.get('references', []))
    recovery['skills'] = list(task.get('skills', []))
    recovery['generation_kind'] = 'reference_recovery'
    recovery['budget']['optimization_attempts'] = 0
    return recovery


def _prepare(out, config, resume):
    out = Path(out)
    if out.exists():
        if not resume:
            raise FileExistsError(out)
        if read(out/'pipeline.json') != config:
            raise ValueError('Resume cannot change pipeline tasks, variants or reference contents')
    else:
        out.mkdir(parents=True)
        write(out/'pipeline.json', config)
    return out


def _config(task, recovery, variants=None):
    return {'task':task, 'recovery_task':recovery, 'variants':variants,
            'input_hashes':{str(p):digest(read(p)) for p in task.get('references', []) + task.get('skills', [])}}


def _recover(task, planner_factory, backend_factory, out, resume):
    target = out/'reference_recovery'
    summary = run_task(task, planner_factory(target), backend_factory(task), target,
                       resume=resume and target.exists())
    if not summary['success']:
        return summary, None
    # run_task also revalidates evidence hashes on resume; imported success flags
    # can never substitute for this physical execution and saved baseline.
    artifact = target/'success'/f"attempt_{summary['accepted_attempt']:03d}"/'trajectory.json'
    return summary, str(artifact.resolve())


def run_pipeline(task, planner_factory, backend_factory, directory, *, reference_task=None, resume=False):
    """Reference inputs require original-scene recovery; no-reference behavior is unchanged."""
    validate_task(task)
    if not task.get('references'):
        return run_task(task, planner_factory(directory), backend_factory(task), directory, resume=resume)
    recovery = _recovery_task(task, reference_task)
    out = _prepare(directory, _config(task, recovery), resume)
    started = time.time()
    restored, artifact = _recover(recovery, planner_factory, backend_factory, out, resume)
    result = {'success':False, 'status':'reference_recovery_failed', 'reference_recovery':restored,
              'generation':None, 'recovered_reference':artifact,
              'budget_scope':'Recovery and generation each have their explicit task attempt budget.'}
    if artifact:
        candidate = copy.deepcopy(task)
        candidate['references'] = [artifact]
        target = out/'generation'
        generated = run_task(candidate, planner_factory(target), backend_factory(candidate), target,
                             resume=resume and target.exists())
        result.update(success=generated['success'], status=generated['status'], generation=generated)
    result['wall_s_this_invocation'] = time.time()-started
    write(out/'summary.json', result)
    return result


def augment_pipeline(task, variants, planner_factory, backend_factory, directory, *, reference_task=None, resume=False):
    """Recover once, preserve its success, then run each explicit scene variant."""
    validate_task(task)
    if not isinstance(variants, list) or not 1 <= len(variants) <= 100 or not all(isinstance(v, dict) for v in variants):
        raise ValueError('Require 1..100 explicit scene variants')
    recovery = _recovery_task(task, reference_task if task.get('references') else task)
    out = _prepare(directory, _config(task, recovery, variants), resume)
    started = time.time()
    restored, artifact = _recover(recovery, planner_factory, backend_factory, out, resume)
    result = {'success':False, 'status':'reference_recovery_failed', 'reference_recovery':restored,
              'recovered_reference':artifact, 'variants':[], 'success_count':0,
              'budget_scope':'One original-scene recovery budget plus the explicit task budget per variant.'}
    if artifact:
        for i, variant in enumerate(variants):
            candidate = copy.deepcopy(task)
            candidate['scene'].update(variant)
            candidate['references'] = [artifact]
            candidate['generation_kind'] = 'augmented'
            candidate['name'] += f'_variant_{i:03d}'
            target = out/f'variant_{i:03d}'
            result['variants'].append(run_task(candidate, planner_factory(target), backend_factory(candidate), target,
                                               resume=resume and target.exists()))
        result['success_count'] = sum(x['success'] for x in result['variants'])
        result['success'] = result['success_count'] == len(variants)
        result['status'] = 'success' if result['success'] else 'variants_incomplete'
    result['wall_s_this_invocation'] = time.time()-started
    write(out/'summary.json', result)
    return result
