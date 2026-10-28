"""Independent entrypoint. Calls unchanged run_pipeline without raw references."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
import traceback
from zoneinfo import ZoneInfo

from cap_sim.backends.humano import Backend
from cap_sim.pipeline import run_pipeline
from adapter import MailboxPlanner, verify_frozen, write
from scene import geometry, physics_signature, render_views


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


class ObservedBackend(Backend):
    """Adds read-only images/audits. The original physics executor is inherited."""
    interface = copy.deepcopy(Backend.interface)
    interface['action'].pop('body_keyframes')

    def __init__(self, directory, expected_physics):
        self.directory = Path(directory)
        self.expected_physics = expected_physics
        self.image_index = 0
        self.simulation_wall_s = 0.
        self.render_wall_s = 0.

    def _prepare(self, task):
        super()._prepare(task)
        actual = physics_signature(self.m)
        if actual != self.expected_physics:
            write(self.directory/'physics_mismatch.json', {'expected': self.expected_physics, 'actual': actual})
            raise RuntimeError('Model physics differs from historical source; no control dispatched')

    def observe(self):
        observation = super().observe()
        observation['geometry'] = geometry(self.m, self.d)
        observation['fingers_semantics'] = 'fingers are last commanded positions; joint_interface contains measured positions'
        started = time.perf_counter()
        observation['images'] = render_views(self.m, self.d, self.directory/'images', f'obs_{self.image_index:04d}')
        self.render_wall_s += time.perf_counter()-started
        self.image_index += 1
        return observation

    def reset(self, task, out):
        observation = super().reset(task, out)
        # Historical HybridBackend restored the recorded initial warmstart too.
        self.d.qacc_warmstart[:] = self.initial['qacc_warmstart']
        providers = {name: getattr(self.sonic, name).get_providers() for name in ('encoder', 'decoder', 'planner')}
        if any(value != ['CPUExecutionProvider'] for value in providers.values()):
            raise RuntimeError('Unexpected SONIC provider')
        write(Path(out)/'actual_providers.json', providers)
        write(Path(out)/'actual_physics.json', physics_signature(self.m))
        return observation

    def execute(self, action, remaining):
        if 'body_keyframes' in action or action.get('stage') == 'native_prefix':
            raise ValueError('Source replay is forbidden')
        started = time.perf_counter()
        previous_render = self.render_wall_s
        result = super().execute(action, remaining)
        self.simulation_wall_s += time.perf_counter()-started-(self.render_wall_s-previous_render)
        modes = result.get('controller_diagnostics', {}).get('actual_modes', [])
        if modes and modes != [1]:
            self.export(self.out)
            raise RuntimeError('Actual control mode mismatch')
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--case', required=True, choices=['y_minus_10cm', 'y_plus_10cm', 'y_minus_20cm', 'y_plus_20cm'])
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    verify_frozen(experiment)
    directory = experiment/args.case
    directory.mkdir(exist_ok=False)
    task = json.loads((experiment/'tasks'/f'{args.case}.json').read_text())
    source = json.loads((experiment/'source_evidence.json').read_text())
    demo = json.loads((experiment/'demonstration.json').read_text())
    planner = MailboxPlanner(directory/'mailbox', demo)
    backend = ObservedBackend(directory, source['physics'])
    started_at = now()
    started = time.perf_counter()
    write(directory/'started.json', {'started_at': started_at, 'experiment_id': experiment.name})
    try:
        result = run_pipeline(task, lambda out: planner, lambda task: backend, directory/'run')
        write(directory/'result.json', result)
    except Exception as exc:
        write(directory/'error.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        if hasattr(backend, 'out') and hasattr(backend, 'states'):
            backend.export(backend.out)
        raise
    finally:
        write(directory/'timing.json', {'experiment_id': experiment.name, 'started_at': started_at,
              'ended_at': now(), 'total_wall_s': time.perf_counter()-started,
              'planning_wait_wall_s': planner.wait_s, 'simulation_compute_wall_s': backend.simulation_wall_s,
              'observation_render_wall_s': backend.render_wall_s,
              'simulation_elapsed_s': float(backend.d.time) if hasattr(backend, 'd') else 0.})
        write(directory/'done.json', {'ended_at': now(), 'status': 'error' if (directory/'error.json').exists() else 'complete'})
        backend.close()


if __name__ == '__main__':
    main()
