"""Offline pipeline orchestration; physical execution is a separately gated step."""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .cli import read, write_new


def generate_batch(demo, annotation, cases, out):
    from .dataset import save_demo
    from .retarget import retarget_demo, geometry_metrics
    from .annotation import validate_annotation

    validate_annotation(annotation, demo)
    if not isinstance(cases, list) or not cases:
        raise ValueError('A nonempty case list is required')
    names = [c['id'] for c in cases]
    if len(set(names)) != len(names) or any(not re.fullmatch(r'[a-z0-9_]+', n) for n in names):
        raise ValueError('Case IDs must be unique safe path components')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result, status, failed_case, error_type = [], 'error', None, None
    try:
        for case in cases:
            failed_case = case['id']
            generated = retarget_demo(demo, annotation, translation=case['translation'],
                                      rotation_wxyz=case.get('rotation_wxyz', [1, 0, 0, 0]),
                                      preserve_prefix_s=5.)
            generated['metadata']['case'] = case
            generated['metadata']['table_height_status'] = (
                'declared scene constraint only; no variant scene or physics constructed')
            metrics = geometry_metrics(demo, generated, annotation)
            if not metrics['invariants_passed']:
                raise ValueError('Generated trajectory failed geometric invariants')
            path = out / case['id'] / 'trajectory.npz'
            save_demo(generated, path)
            write_new(path.parent / 'geometry.json', metrics)
            result.append({'case': case, 'trajectory': str(path.resolve()), 'geometry': metrics,
                           'physics_executed': False, 'task_success': None})
        status, failed_case = 'completed', None
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        summary = {'kind': 'offline_geometry_only', 'status': status, 'case_count': len(cases),
                   'completed_case_count': len(result), 'failed_case': failed_case,
                   'error_type': error_type, 'results': result, 'wall_s': time.perf_counter() - started,
                   'physics_steps': 0, 'training_ready': False}
        write_new(out / 'summary.json', summary)
    return summary


def run_offline(source, out, model='gpt-6-astra', provider='codex', cases_path=None):
    from .dataset import import_native
    from .annotation import build_request, validate_annotation, annotate_openai
    from .model import annotate_codex

    if provider not in ('codex', 'openai'):
        raise ValueError('run requires a real GPT-6 provider: codex or openai')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    tick = time.perf_counter()
    timing, status, failed_stage, error_type = {}, 'error', 'import', None
    try:
        stage = time.perf_counter()
        demo = import_native(source, out / 'demo.npz')
        timing['import_wall_s'] = time.perf_counter() - stage
        failed_stage = 'annotation'
        stage = time.perf_counter()
        request = build_request(demo)
        annotate = annotate_codex if provider == 'codex' else annotate_openai
        annotation = annotate(request, model, out / 'model_call')
        validate_annotation(annotation, demo)
        write_new(out / 'annotation.json', annotation)
        timing['annotation_wall_s'] = time.perf_counter() - stage
        failed_stage = 'generation'
        summary = generate_batch(demo, annotation,
            read(cases_path or Path(__file__).with_name('ten_conditions.json')), out / 'variants')
        timing['generation_wall_s'] = summary['wall_s']
        status, failed_stage = 'completed', None
        return summary
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        write_new(out / 'timing.json', {'status': status, 'started_at': started_at,
            'failed_stage': failed_stage, 'error_type': error_type,
            'ended_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            **timing, 'total_wall_s': time.perf_counter() - tick,
            'simulation_wall_s': None, 'simulation_elapsed_s': None, 'rendering_wall_s': None})
