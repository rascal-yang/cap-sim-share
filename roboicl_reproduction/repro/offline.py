"""Run the unchanged upstream dialogue with synthetic, deterministic responses.

This is an offline protocol test, not an API, task solver or physics simulator.
Construction temporarily isolates ASTRA_* settings; use in a single process/thread.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import numpy as np

VENDOR = Path(__file__).resolve().parents[1] / 'vendor'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from roboicl.policy.dialogue_policy import Model


@contextmanager
def isolated_settings(settings):
    saved = {k: v for k, v in os.environ.items() if k.startswith('ASTRA_')}
    try:
        for k in saved:
            del os.environ[k]
        os.environ.update(settings)
        yield
    finally:
        for k in list(os.environ):
            if k.startswith('ASTRA_'):
                del os.environ[k]
        os.environ.update(saved)


class ScriptedPolicy(Model):
    def _request(self, payload):
        self.requests.append(deepcopy(payload))
        if not self.script:
            raise RuntimeError('Synthetic response script exhausted; no network fallback')
        return deepcopy(self.script.pop(0))


def create_policy(directory, responses):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads((VENDOR / 'configs/shared_harness.json').read_text())
    # Test scale: preserve algorithms, reduce tensor horizon and RGB cost only.
    config.update(predict_horizon=2, execute_horizon=2, live_anchor_count=3,
                  include_head_calibration=False, max_turns=20,
                  image_profile='thumbnail_jpeg', image_detail='low', image_layout='separate',
                  max_context_images=50)
    config_path = directory / 'synthetic_harness.json'
    config_path.write_text(json.dumps(config, indent=2))
    with isolated_settings({'ASTRA_API_KEY': 'offline-never-sent',
                            'ASTRA_HARNESS_CONFIG': str(config_path),
                            'ASTRA_LOG_DIR': str(directory / 'trace')}):
        policy = ScriptedPolicy({'model': 'synthetic-offline-fixture'})
    policy.requests = []
    policy.script = deepcopy(responses)
    return policy


def response(call_id, rows=None):
    if rows is None:
        rows = [[.001, 0, 0, 0, 0, 0, .5, .001, 0, 0, 0, 0, 0, .5] for _ in range(2)]
    return {'output': [{'type': 'function_call', 'name': 'act', 'call_id': call_id,
                        'arguments': json.dumps({'actions': rows, 'execution_note': 'SYNTHETIC protocol test'})}]}


def observation(step, command=None, event=None):
    state = {}
    for side, y in [('left', .2), ('right', -.2)]:
        state.update({f'{side}_ee_pose': [step*.001, y, .5, 1, 0, 0, 0],
                      f'{side}_arm_joint_state': [0.]*6, f'{side}_ee_joint_state': [.5]})
    meta = {'step': step, 'max_episode_steps': 30}
    if command is not None:
        meta['executed_action'] = command
    if event is not None:
        meta['controller_event'] = event
    return {'state': state, 'instruction': 'SYNTHETIC fixture, no physical task outcome',
            'vision': {cam: {'color': np.full((48, 64, 3), 120+step, dtype=np.uint8)}
                       for cam in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
            'astra_meta': meta}
