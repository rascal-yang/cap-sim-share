"""Extract measured demonstration information. No simulator environment/reset/step."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np

from adapter import PROMPT, write
from scene import measured_observation, physics_signature, render_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    baseline = root/'experiments/20261013_apple_straight_sweep'
    source = baseline/'reference_recovery/success/attempt_001'
    experiment = args.experiment.resolve()
    if (experiment/'demonstration.json').exists():
        raise FileExistsError('Do not overwrite prepared demonstration')
    trajectory = json.loads((source/'trajectory.json').read_text())
    validation = json.loads((source/'independent_validation.json').read_text())
    if not trajectory['acceptance']['success'] or not validation['success']:
        raise ValueError('Source is not independently accepted')
    model = mujoco.MjModel.from_binary_path(str(source/'scene.mjb'))
    data = mujoco.MjData(model)
    states = np.load(source/'states.npz')['samples']
    initial = np.load(trajectory['task']['resources']['initial_state'])
    rows = np.vstack([np.r_[0., initial['qpos'], initial['qvel'], initial['ctrl']], states])
    indices = np.unique(np.linspace(0, len(rows)-1, 13).astype(int))
    observations = []
    for index in indices:
        row = rows[index]
        data.time = row[0]
        data.qpos[:] = row[1:1+model.nq]
        data.qvel[:] = row[1+model.nq:1+model.nq+model.nv]
        mujoco.mj_forward(model, data)
        observation = measured_observation(model, data)
        # Keep image budget small: start, middle, contact approach and final.
        if index in (indices[0], indices[len(indices)//2], indices[-3], indices[-1]):
            observation['images'] = render_views(model, data, experiment/'demo_images', f'frame_{index:05d}')
        observations.append(observation)
    blocks = []
    for before, after in zip(observations, observations[1:]):
        delta = {side+'_wrist_delta_world_m':
                 (np.array(after[side+'_position'])-before[side+'_position']).tolist()
                 for side in ('left', 'right')}
        delta['root_delta_world_m'] = (np.array(after['root_pose'][:3])-before['root_pose'][:3]).tolist()
        blocks.append({'observation_before': before,
                       'measured_transition_not_command': delta,
                       'elapsed_s': after['time']-before['time'],
                       'observation_after': after})
    demo = {'kind': 'measured_demonstration_transitions', 'blocks': blocks,
            'source_control': 'original motion token prefix + retargeted mode0 body reference',
            'live_control': 'newly planned mode1 Cartesian teleop',
            'limitations': ['Measured deltas are not executable source commands.',
                           'Source controller differs from LIVE; no action-equivalence claim.',
                           'Source accepted native height, not a stable one-second grasp hold.',
                           'Intermediate frames omitted; no raw tokens or body trajectory supplied.']}
    write(experiment/'demonstration.json', demo)
    write(experiment/'source_evidence.json', {
        'source': str(source), 'validation': validation, 'acceptance': trajectory['acceptance'],
        'hashes': {str(source/f): hashlib.sha256((source/f).read_bytes()).hexdigest()
                   for f in ('trajectory.json', 'states.npz', 'scene.mjb', 'independent_validation.json')},
        'physics': physics_signature(model), 'physics_steps': 0})
    for name, offset in [('y_minus_10cm', -.1), ('y_plus_10cm', .1),
                         ('y_minus_20cm', -.2), ('y_plus_20cm', .2)]:
        task = json.loads((baseline/'y_minus_20cm/run/task.json').read_text())
        task['scene'] = {'translation': [0, offset, 0]}
        task['references'] = []
        task['skills'] = []
        task.pop('contact_prior', None)
        task['policy_checks'] = {'sonic_mode': 1, 'require_reflection': True}
        task['generation_kind'] = 'roboicl_inspired_fresh_actions'
        task['planner_identity'] = 'Isolated Codex subagent, fork_turns=none, no external API'
        task['code_version'] = 'Independent roboicl_adapter; original cap-sim unchanged'
        write(experiment/'tasks'/f'{name}.json', task)
    (experiment/'planner_prompt.txt').write_text(PROMPT)
    print('Prepared', experiment, 'physics_steps=0')


if __name__ == '__main__':
    main()
