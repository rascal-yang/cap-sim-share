"""Evidence-gated original-scene HUMANO smoke; no IK or controller fallback.

All helpers above ``prepare_manifest`` are simulator-free. Preparing a manifest
constructs and forwards a model but never steps it. Only ``run_smoke`` advances
physics, after checking the immutable contract and all bound input hashes.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def _now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _digest(value):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False, default=convert).encode()).hexdigest()


def input_hashes(task, demo, generated, annotation, parquet):
    """Hashes of the *actual* in-memory launch inputs, including finger arrays."""
    return {**{k: _digest(v) for k, v in {'task': task, 'demo': demo,
            'generated': generated, 'annotation': annotation}.items()}, 'parquet': _sha(parquet)}


def write_success_package(attempt, task, result):
    """Package existing accepted evidence without simulating or inventing actions."""
    from cap_sim.io import digest
    attempt = Path(attempt).resolve()
    if (result.get('success') is not True or result.get('failure') is not None
            or result.get('acceptance', {}).get('success') is not True):
        raise ValueError('Only an accepted attempt may produce a successful trajectory package')
    if attempt.parent.name != 'attempts' or not attempt.name.isdigit():
        raise ValueError('Successful evidence must identify an attempts/<number> directory')
    if not isinstance(result.get('observation', {}).get('object_details', {}).get('target', {}).get('root_body'), str):
        raise ValueError('Accepted observation must identify the native target root body')
    required = ('states.npz', 'scene.mjb', 'contacts.json', 'samples.json', 'independent_validation.json', 'commands.json')
    for name in required:
        if not (attempt/name).is_file():
            raise ValueError('Accepted attempt lacks evidence: ' + name)
    independent = json.loads((attempt/'independent_validation.json').read_text())
    if independent.get('success') is not True or independent.get('physics_steps') != 0:
        raise ValueError('Accepted attempt lacks successful independent saved-state validation')
    samples = json.loads((attempt/'samples.json').read_text())
    target = attempt.parent.parent/'success'/('attempt_' + attempt.name)
    if target.exists():
        raise FileExistsError(target)
    staging = attempt.parent.parent/('.success_attempt_' + attempt.name + '_pending')
    staging.mkdir(exist_ok=False)
    names = [*required, *[name for name in ('executed_reference.npz', 'scene_source.xml',
               'controller.json', 'actual_modes.json', 'handoff.json') if (attempt/name).is_file()]]
    for name in names:
        shutil.copy2(attempt/name, staging/name)
    def provenance(name):
        source = attempt/name
        if not source.exists():
            return {'available': False, 'reason': 'No dense phase was executed'}
        return {'source_path': str(source), 'package_path': name, 'sha256': _sha(source)}
    trajectory = {'schema_version': 1, 'task': task, 'acceptance': result['acceptance'],
                  'metrics': result.get('metrics', {}), 'source_attempt': str(attempt),
                  'samples_sha256': digest(samples), 'generation_kind': 'contact_trajectory_retargeting',
                  'action_representation': 'dense measured-reference tracking with original motion-token prefix',
                  'dense_reference': provenance('executed_reference.npz'),
                  'commands': provenance('commands.json'), 'original_actions_visible': True,
                  'source_evidence_sha256': {name: _sha(attempt/name) for name in names}}
    _write(staging/'trajectory.json', trajectory)
    target.parent.mkdir(exist_ok=True)
    staging.rename(target)
    return target/'trajectory.json'


def _resolve(contract, filename):
    path = Path(filename)
    return path if path.is_absolute() else Path(contract).parent / path


def validate_contract(path):
    """Reject a template, absent evidence, stale seal, or unloaded CPU sessions."""
    path = Path(path)
    record = json.loads(path.read_text())
    if record.get('record_status') != 'frozen':
        raise ValueError('experiment.json must be frozen before launch')
    seal = path.with_suffix('.sha256')
    if not seal.is_file() or seal.read_text().split()[0] != _sha(path):
        raise ValueError('Experiment seal missing or mismatched')
    template = json.loads((Path(__file__).parents[1] / 'docs/experiment-record.template.json').read_text())
    def required(expected, actual, prefix=''):
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                raise ValueError('Incomplete contract field: ' + prefix)
            for key, item in expected.items():
                if key not in actual:
                    raise ValueError('Missing contract field: ' + prefix + key)
                required(item, actual[key], prefix + key + '.')
        elif actual is None or actual == '':
            raise ValueError('Incomplete contract field: ' + prefix.rstrip('.'))
    required(template, record)
    if record['timezone'] != 'Asia/Shanghai' or not re.fullmatch(r'\d{8}_\d{4}_.+', record['experiment_id']):
        raise ValueError('Experiment needs minute-resolution ID and Asia/Shanghai timezone')
    for field in ('cases',):
        if not record[field]:
            raise ValueError('Missing contract ' + field)
    if record['control']['fallback_allowed'] is not False:
        raise ValueError('Controller fallback is forbidden')
    interfaces = record['control']['stage_interfaces']
    if not isinstance(interfaces, list) or not interfaces:
        raise ValueError('Missing stage interfaces')
    for item in interfaces:
        for name in template['control']['stage_entry_required_fields']:
            if name not in item or item[name] is None or item[name] == '':
                raise ValueError('Missing stage interface field ' + name)
    preflight = record['preflight']
    if preflight['passed'] is not True:
        raise ValueError('Preflight must pass')
    hashes = preflight.get('evidence_sha256', {})
    evidence = preflight['evidence_paths']
    if not isinstance(hashes, dict) or not hashes or not evidence:
        raise ValueError('Preflight needs evidence paths and SHA256 hashes')
    for filename in evidence:
        if filename not in hashes:
            raise ValueError('Evidence missing hash: ' + filename)
    for filename, expected in hashes.items():
        source = _resolve(path, filename)
        if not source.is_file() or _sha(source) != expected:
            raise ValueError('Evidence hash mismatch: ' + filename)
    for filename in (record['runtime']['actual_loaded_session_evidence'],
                     record['physics']['asset_and_initial_state_manifest'],
                     record['physics']['physics_parameters_manifest'],
                     preflight['comparison_diff_path']):
        if not isinstance(filename, str) or filename not in hashes:
            raise ValueError('Required evidence path lacks hash: ' + str(filename))
    runtime = json.loads(_resolve(path, record['runtime']['actual_loaded_session_evidence']).read_text())
    if runtime.get('passed') is not True or runtime.get('sessions_loaded') is not True:
        raise ValueError('Actual loaded-session runtime evidence required')
    providers = runtime.get('actual_session_providers', {})
    if len(providers) != 3 or any(v != ['CPUExecutionProvider'] for v in providers.values()):
        raise ValueError('All three actual SONIC sessions must use CPUExecutionProvider')
    versions = runtime.get('versions', {})
    if versions.get('tensorrt') != '10.13.0.35' or versions.get('onnxruntime') != '1.23.2':
        raise ValueError('Runtime must be TensorRT 10.13.0.35 + ONNX Runtime 1.23.2')
    if runtime.get('sonic_version') != '1.0' or runtime.get('tensorrt_used_for_sonic_inference') is not False:
        raise ValueError('SONIC must be original release 1.0 with CPU inference')
    if len(runtime.get('model_sha256', {})) != 3:
        raise ValueError('Loaded model hashes missing')
    return record


def validate_scene(task):
    scene = task.get('scene', {})
    translation = np.asarray(scene.get('translation', [0, 0, 0]), float)
    if (translation.shape != (3,) or not np.isfinite(translation).all()
            or np.any(translation != 0) or scene.get('table_delta', 0) != 0
            or scene.get('apple_scale', 1) != 1
            or set(scene) - {'translation', 'table_delta', 'apple_scale'}):
        raise ValueError('This smoke executor only supports the original scene')


def reanchor_handoff(generated, annotation, observation, handoff_time=5.):
    """Replace only the remaining transit with a measured-pose quintic bridge."""
    from scipy.spatial.transform import Rotation, Slerp
    times = np.asarray(generated['time_s'])
    first = int(np.searchsorted(times, handoff_time + 1e-9))
    stages = annotation['stages']
    current = next((s for s in stages if s['start_frame'] <= first < s['end_frame']), None)
    if current is None or current['kind'] != 'transit':
        raise ValueError('Prefix handoff requires an explicit transit before contact')
    contact = next((s for s in stages if s['kind'] == 'contact' and s['start_frame'] >= first), None)
    if contact is None or times[contact['start_frame']] <= handoff_time:
        raise ValueError('Missing positive-duration transit to first contact')
    end = contact['start_frame']
    result = {k: v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v) for k, v in generated.items()}
    u = np.clip((times[first:end] - handoff_time) / (times[end] - handoff_time), 0., 1.)
    alpha = u**3 * (10 - 15*u + 6*u*u)
    for side in ('left', 'right'):
        key = side + '_pose'
        actual = np.r_[observation[side + '_position'], observation[side + '_quaternion_wxyz']]
        target = result[key][end].copy()
        result[key][first:end, :3] = actual[:3] + alpha[:, None] * (target[:3] - actual[:3])
        rotations = Rotation.from_quat(np.roll(np.array([actual[3:], target[3:]]), -1, axis=1))
        result[key][first:end, 3:] = np.roll(Slerp([0, 1], rotations)(alpha).as_quat(), 1, axis=1)
    return result, {'time_s': float(handoff_time), 'first_modified_frame': first,
                    'end_modified_frame_exclusive': end, 'contact_modified': False,
                    'interpolation': 'quintic position and shortest-arc quaternion slerp',
                    'measured_observation': observation}


class SafetyMonitor:
    """Original hand, table and instability thresholds at 200 Hz."""
    def __init__(self):
        self.contact_s = self.table_s = self.lost_s = 0.
        self.is_lift = False

    def enter(self, name, contact_s, lift_stage_names=()):
        self.is_lift = name in lift_stage_names
        self.contact_s = contact_s
        self.lost_s = 0.
        if self.is_lift and contact_s + 1e-9 < .12:
            return 'hand_contact_gate'
        return None

    def update(self, *, hand_force, table_force, stable, warning):
        self.contact_s = self.contact_s + .005 if hand_force > .1 else 0.
        self.lost_s = 0. if hand_force > .1 else self.lost_s + .005
        self.table_s = self.table_s + .005 if table_force > 5. else 0.
        if not stable or warning:
            return 'robot_fall_or_simulation_warning'
        if self.table_s + 1e-9 >= .05:
            return 'unexpected_table_contact'
        if self.is_lift and self.lost_s + 1e-9 >= .03:
            return 'contact_lost'
        return None


def _physics(backend):
    from humano.mt.sonic import policy_parameters
    m = backend.m
    names = ('body_mass', 'body_inertia', 'body_pos', 'geom_friction', 'geom_solref',
             'geom_solimp', 'dof_damping', 'dof_armature', 'dof_frictionloss',
             'jnt_range', 'jnt_limited', 'actuator_gainprm', 'actuator_biasprm',
             'actuator_ctrlrange', 'actuator_forcerange', 'body_quat', 'geom_size',
             'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity', 'geom_group',
             'geom_bodyid', 'jnt_pos', 'jnt_axis')
    return {'physics_steps': 0, 'physics_dt_s': float(m.opt.timestep), 'control_dt_s': .02,
            'dimensions': {'nq': m.nq, 'nv': m.nv, 'nu': m.nu},
            'gravity': m.opt.gravity.tolist(), 'integrator': int(m.opt.integrator),
            'solver': int(m.opt.solver), 'iterations': int(m.opt.iterations),
            'tolerance': float(m.opt.tolerance), 'reward_stride': backend.reward_stride,
            'parameters': {name: getattr(m, name).tolist() for name in names},
            'compiled_mesh_sha256': {name: hashlib.sha256(np.ascontiguousarray(getattr(m, name)).tobytes()).hexdigest()
                                     for name in ('mesh_vert', 'mesh_face', 'mesh_normal', 'tex_data')},
            'model_xml_sha256': hashlib.sha256(backend.model_xml.encode()).hexdigest(),
            'pd_kp': np.asarray(policy_parameters.kps).tolist(),
            'pd_kd': np.asarray(policy_parameters.kds).tolist(),
            'initial_state_sha256': _digest({'qpos': backend.initial_q, 'qvel': backend.initial_v,
                                           'ctrl': backend.initial['ctrl'],
                                           'qacc_warmstart': backend.initial['qacc_warmstart']}),
            'action_permissions': 'SONIC torque body targets and 7-radian-joint commands per hand; no runtime state writes',
            'acceptance': backend.task['acceptance']}


def prepare_manifest(task, out):
    """Construct the original model, hash resources, save evidence; zero mj_step."""
    validate_scene(task)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    from cap_sim.backends.humano import Backend
    backend = Backend()
    try:
        backend._prepare(copy.deepcopy(task))
        backend.mj.mj_saveModel(backend.m, str(out/'scene.mjb'), None)
        (out/'scene_source.xml').write_text(backend.model_xml)
        np.savez_compressed(out/'initial_state.npz', **{k: backend.initial[k] for k in backend.initial.files})
        physics = _physics(backend)
        _write(out/'physics_manifest.json', physics)
        paths = [Path(task['resources']['initial_state']),
                 Path(task['resources']['prepared'])/'environment_config.json',
                 Path(inspect.getfile(type(backend.e.task))), out/'scene.mjb',
                 out/'scene_source.xml', out/'initial_state.npz']
        from humano.utils import resolve_data_path
        robot_xml = Path(resolve_data_path(backend.robot.mjcf_path, auto_download=False))
        compiler = ET.parse(robot_xml).getroot().find('compiler')
        paths.append(robot_xml)
        for element in ET.fromstring(backend.model_xml).iter():
            if element.get('file'):
                source = Path(element.get('file'))
                if not source.is_absolute():
                    directory = compiler.get('meshdir' if element.tag == 'mesh' else 'texturedir', '') if compiler is not None else ''
                    source = robot_xml.parent / directory / source
                    if not source.is_file():
                        raise ValueError('Unable to resolve compiled asset source: ' + str(source))
                paths.append(source)
        manifest = {'physics_steps': 0, 'created_at': _now(), 'task_sha256': _digest(task),
                    'files': {str(p.resolve()): _sha(p) for p in sorted(set(paths))},
                    'acceptance_source': str(Path(inspect.getfile(type(backend.e.task))))}
        _write(out/'assets_manifest.json', manifest)
        return {'asset_and_initial_state_manifest': str((out/'assets_manifest.json').resolve()),
                'physics_parameters_manifest': str((out/'physics_manifest.json').resolve()),
                'physics_dt_s': .005, 'control_dt_s': .02,
                'acceptance_source_and_hash': {'path': manifest['acceptance_source'],
                    'sha256': _sha(manifest['acceptance_source'])},
                'evidence_sha256': {str(p.resolve()): _sha(p) for p in
                    (out/'assets_manifest.json', out/'physics_manifest.json')}, 'physics_steps': 0}
    finally:
        backend.close()


def _check_launch_inputs(record, contract, task, demo, generated, annotation, parquet):
    from .dataset import validate_demo
    from .annotation import validate_annotation
    validate_scene(task)
    validate_demo(demo)
    validate_demo(generated)
    validate_annotation(annotation, demo)
    actual_hashes = input_hashes(task, demo, generated, annotation, parquet)
    if record['preflight'].get('input_sha256') != actual_hashes:
        raise ValueError('Frozen input hashes do not match launch inputs')
    source = Path(__file__).resolve()
    if not any(_resolve(contract, name).resolve() == source
               for name in record['preflight']['evidence_sha256']):
        raise ValueError('Actual native executor source must be frozen in preflight evidence')
    if record['budget']['attempts_per_case'] != 1 or record['pipeline']['optimization_enabled'] is not False:
        raise ValueError('Smoke permits exactly one attempt and no optimization')
    if record['budget']['recovery_attempts'] != 0 or record['budget']['optimization_attempts'] != 0:
        raise ValueError('No recovery or optimization attempts are permitted')
    if task['name'] != 'G1WholebodyTableApplePickTeleop' or task['backend'] != 'humano':
        raise ValueError('Only the original HUMANO apple task is supported')
    if task['acceptance'].get('kind') != 'native' or task['acceptance'].get('hold_s') != .005:
        raise ValueError('Original native acceptance and 5ms hold required')
    budget = record['budget']['simulation_seconds_per_attempt']
    if not isinstance(budget, (int, float)) or not np.isfinite(budget) or budget < 5.:
        raise ValueError('Simulation budget must cover the declared 5-second prefix')
    times = np.asarray(generated['time_s'])
    if (not np.array_equal(times, demo['time_s']) or len(times) < 1001
            or abs(times[0] - .005) > 1e-8 or not np.allclose(np.diff(times), .005, atol=1e-8)):
        raise ValueError('Dense native references require original aligned 200Hz samples starting at .005s')
    for key in ('left_fingers', 'right_fingers'):
        if not np.array_equal(generated[key], demo[key]):
            raise ValueError('Generated fingers differ from declared measured reference')
    prefix = times <= 5. + 1e-9
    for key in ('left_pose', 'right_pose', 'torso_pose', 'root_pose', 'left_fingers', 'right_fingers'):
        if not np.array_equal(generated[key][prefix], demo[key][prefix]):
            raise ValueError('Original prefix references must be retained through t=5.0s')
    names = record['control'].get('contact_lift_stage_names')
    stages = {s['name']: s for s in annotation['stages']}
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(n not in stages or stages[n]['kind'] != 'contact' for n in names)):
        raise ValueError('Explicit contact_lift_stage_names must identify real annotated contact stages')
    assets = json.loads(_resolve(contract, record['physics']['asset_and_initial_state_manifest']).read_text())
    if assets.get('physics_steps') != 0 or assets.get('task_sha256') != _digest(task):
        raise ValueError('Manifest task identity mismatch')
    for filename, expected in assets.get('files', {}).items():
        if not Path(filename).is_file() or _sha(filename) != expected:
            raise ValueError('Asset manifest hash mismatch: ' + filename)
    if not assets.get('files'):
        raise ValueError('Empty asset manifest')
    runtime = json.loads(_resolve(contract, record['runtime']['actual_loaded_session_evidence']).read_text())
    import importlib.metadata
    if any(importlib.metadata.version(name) != version for name, version in runtime['versions'].items()):
        raise ValueError('Current package versions differ from frozen loaded-session evidence')
    for filename, expected in runtime['model_sha256'].items():
        if _sha(Path(task['resources']['sonic_models']) / filename) != expected:
            raise ValueError('Loaded model hash does not match current model')
    return actual_hashes


def run_smoke(task, demo, generated, annotation, parquet, out, contract):
    """Run one original-scene prefix + mode1 attempt, preserving every failure."""
    started_at = _now(); total_start = time.perf_counter()
    record = validate_contract(contract)
    hashes = _check_launch_inputs(record, contract, task, demo, generated, annotation, parquet)
    out = Path(out)
    if out.exists():
        raise FileExistsError('Never overwrite or silently resume a physical attempt: ' + str(out))
    import pandas as pd
    prefix = pd.read_parquet(parquet, columns=['action.motion_token', 'action.wbc']).iloc[:250]
    tokens = np.stack(prefix['action.motion_token'])
    wbc = np.stack(prefix['action.wbc'])
    if tokens.shape != (250, 64) or wbc.shape != (250, 43) or not np.isfinite(np.r_[tokens.ravel(), wbc.ravel()]).all():
        raise ValueError('Expected 250 finite original motion tokens and 43D hand/body rows')
    from cap_sim.backends.humano import Backend
    from cap_sim.backends.native_sonic import rot
    from cap_sim.core import evaluate
    from humano.mt.base import MotorCommand
    from humano.core.action import ActionCmd
    backend = Backend()
    attempt = out/'attempts/001'
    attempt.mkdir(parents=True)
    _write(out/'task.json', task)
    _write(out/'input_hashes.json', hashes)
    failure = None; commands = []; tracking = []; modes = []
    monitor = SafetyMonitor(); handoff = None; native_validation = None
    physical_start = None; simulation_wall = 0.; planning_wall = 0.
    simulation_clock = None; executed_physics_steps = 0; unrecorded_state = None
    final_observation = {}; observation_error = None
    current_stage = 'original_token_prefix'; stage_events = []
    lift_names = record['control']['contact_lift_stage_names']
    try:
        backend.reset(copy.deepcopy(task), attempt)
        backend.d.qacc_warmstart[:] = backend.initial['qacc_warmstart']
        providers = {k: getattr(backend.sonic, k).get_providers() for k in ('encoder', 'decoder', 'planner')}
        if any(v != ['CPUExecutionProvider'] for v in providers.values()):
            raise RuntimeError('Actual loaded SONIC provider mismatch')
        _write(attempt/'actual_providers.json', providers)
        actual_physics = _physics(backend)
        expected_physics = json.loads(_resolve(contract, record['physics']['physics_parameters_manifest']).read_text())
        if actual_physics != expected_physics:
            raise RuntimeError('Actual model, initial state, PD or physics differs from frozen manifest')
        for side in ('left', 'right'):
            ids = getattr(backend.robot, side + '_hand_actuated_index')
            lo, hi = backend.m.jnt_range[ids].T
            if np.any(generated[side+'_fingers'] < lo-.001) or np.any(generated[side+'_fingers'] > hi+.001):
                raise ValueError('Measured finger reference exceeds native command limits')
        _write(attempt/'reset.json', {'time': _now(), 'observation': backend.observe()})
        physical_start = _now(); simulation_clock = time.perf_counter()
        stage_events.append({'stage': current_stage, 'start_s': 0., 'actual_mode': 'original_token_prefix'})

        def advance(target, fingers, mode, stage):
            nonlocal executed_physics_steps
            if backend.d.time + .005 > float(record['budget']['simulation_seconds_per_attempt']) + 1e-8:
                return 'execution_deadline'
            backend.fingers = [np.asarray(x).copy() for x in fingers]
            backend.robot.apply_action(ActionCmd('wbc_torque', low_cmd=MotorCommand(
                q_target=target, kp=backend.sonic.kp, kd=backend.sonic.kd), use_sensor=False,
                left_hand_cmd=MotorCommand.hand_command(backend.fingers[0]),
                right_hand_cmd=MotorCommand.hand_command(backend.fingers[1])))
            backend.mj.mj_step(backend.m, backend.d)
            executed_physics_steps += 1
            backend.mj.mj_forward(backend.m, backend.d); backend.ticks += 1
            if backend.ticks % backend.reward_stride == 0:
                info = backend.e._get_info()
                backend.native_reward = float(backend.e.task.compute_reward(info, mujoco_env=backend.e.mujoco))
                backend.native_passed = bool(backend.e.task.check_success(info, mujoco_env=backend.e.mujoco))
            contacts = backend.contacts()
            stable = bool(np.isfinite(backend.d.qpos).all() and backend.d.qpos[2] > .30
                          and rot(backend.d.qpos[3:7]).as_matrix()[2, 2] > np.cos(np.deg2rad(50)))
            stop = monitor.update(hand_force=contacts['hand_object_N'], table_force=contacts['robot_table_N'],
                                  stable=stable, warning=any(w.number for w in backend.d.warning))
            backend.hand_contact_s = monitor.contact_s
            backend.samples.append({'time': float(backend.d.time), 'dt': .005,
                'native_success': backend.native_passed, 'stable': stable, 'native_reward': backend.native_reward})
            backend.states.append(np.r_[backend.d.time, backend.d.qpos, backend.d.qvel, backend.d.ctrl])
            backend.contact_records.append(contacts)
            modes.append({'time_s': float(backend.d.time), 'stage': stage, 'mode': mode})
            commands.append({'time_s': float(backend.d.time), 'stage': stage, 'mode': mode,
                             'body_joint_target_rad': np.asarray(target).tolist(),
                             'left_fingers_rad': backend.fingers[0].tolist(),
                             'right_fingers_rad': backend.fingers[1].tolist()})
            backend.success = bool(backend.native_passed and stable and stop is None)
            return stop

        for index in range(250):
            sonic = backend.sonic; d = backend.d
            sonic.history.append({'omega': d.qvel[3:6].copy(),
                'q': (d.qpos[backend.qids] - sonic.default)[sonic.order],
                'dq': d.qvel[backend.vids][sonic.order].copy(), 'last': sonic.last.copy(),
                'gravity': rot(d.qpos[3:7]).inv().apply([0, 0, -1])})
            obs = np.concatenate([tokens[index]] + [np.concatenate([h[z] for h in sonic.history])
                                  for z in ('omega', 'q', 'dq', 'last', 'gravity')]).astype(np.float32)
            raw = sonic.decoder.run(None, {sonic.decoder.get_inputs()[0].name: obs[None]})[0][0]
            if not np.isfinite(raw).all():
                failure = 'nonfinite_sonic_output'; break
            sonic.last = raw.copy(); sonic.tick += 1
            backend.target = sonic.default + sonic.scale * raw[sonic.reverse]
            sonic.records.append({'time': float(d.time), 'mode': 'original_token_prefix',
                                  'token': tokens[index].tolist(), 'target': backend.target.tolist()})
            for _ in range(4):
                failure = advance(backend.target, [wbc[index, 29:36], wbc[index, 36:43]],
                                  'original_token_prefix', current_stage)
                if failure or backend.success:
                    break
            if failure or backend.success:
                break
        stage_events[-1]['end_s'] = float(backend.d.time)
        reference = generated
        if not failure and not backend.success:
            planning_clock = time.perf_counter()
            reference, handoff = reanchor_handoff(generated, annotation, backend.observe(), float(backend.d.time))
            planning_wall += time.perf_counter() - planning_clock
            _write(attempt/'handoff.json', handoff)
            np.savez_compressed(attempt/'executed_reference.npz', **{
                k: v for k, v in reference.items() if isinstance(v, np.ndarray)})
            times = np.asarray(reference['time_s'])
            first = int(np.searchsorted(times, 5. + 1e-9))
            for index in range(first, len(times)):
                stage = next(s for s in annotation['stages'] if s['start_frame'] <= index < s['end_frame'])
                if stage['name'] != current_stage:
                    current_stage = stage['name']
                    if stage_events:
                        stage_events[-1]['end_s'] = float(backend.d.time)
                    stage_events.append({'stage': current_stage, 'start_s': float(backend.d.time), 'actual_mode': 1})
                    failure = monitor.enter(current_stage, backend.hand_contact_s, lift_names)
                    if failure:
                        break
                if backend.d.time + .005 > float(record['budget']['simulation_seconds_per_attempt']) + 1e-8:
                    failure = 'execution_deadline'; break
                if backend.ticks % 4 == 0:
                    # Sample the desired stream at the inference instant, before
                    # advancing physics; index describes the next post-step sample.
                    target_index = max(0, index - 1)
                    wrist_targets = [(reference[s + '_pose'][target_index, :3],
                                      reference[s + '_pose'][target_index, 3:]) for s in ('left', 'right')]
                    if index == first:
                        measured = handoff['measured_observation']
                        wrist_targets = [(np.asarray(measured[s+'_position']),
                                          np.asarray(measured[s+'_quaternion_wxyz'])) for s in ('left', 'right')]
                    torso = reference['torso_pose'][target_index]
                    root = reference['root_pose'][target_index]
                    yaw = float(rot(root[3:]).as_euler('xyz')[2])
                    height = float(root[2])
                    if not .4 <= height <= .85:
                        failure = 'reference_pelvis_height_outside_interface'; break
                    backend.target = backend.sonic.step(backend.d, wrist_targets, (torso[:3], torso[3:]),
                                                        np.zeros(2), height, None, yaw)
                    if backend.sonic.records[-1]['mode'] != 1:
                        raise RuntimeError('Actual SONIC mode deviates from frozen mode1 interface')
                    row = {'time_s': float(backend.d.time), 'stage': current_stage,
                           'target_frame': target_index, 'mode': 1,
                           'root_actual': backend.d.qpos[:7].tolist(), 'root_target': root.tolist(),
                           'torso_target': torso.tolist(), 'velocity_xy_m_s': [0., 0.],
                           'pelvis_height_m': height, 'facing_yaw_rad': yaw}
                    for side, bid, goal in zip(('left', 'right'), (backend.left, backend.right), wrist_targets):
                        p, q = goal
                        row[side] = {'target_pose': np.r_[p, q].tolist(),
                            'actual_pose': np.r_[backend.d.xpos[bid], backend.d.xquat[bid]].tolist(),
                            'position_error_m': float(np.linalg.norm(backend.d.xpos[bid] - p)),
                            'orientation_error_rad': float((rot(q).inv() * rot(backend.d.xquat[bid])).magnitude())}
                    tracking.append(row)
                failure = advance(backend.target, [reference['left_fingers'][index], reference['right_fingers'][index]],
                                  1, current_stage)
                if failure or backend.success:
                    break
            if not failure and not backend.success:
                failure = 'task_not_completed'
        simulation_wall = time.perf_counter() - simulation_clock - planning_wall
    except Exception as exc:
        failure = 'executor_error: ' + type(exc).__name__ + ': ' + str(exc)
        if executed_physics_steps > len(getattr(backend, 'states', [])):
            unrecorded_state = np.r_[backend.d.time, backend.d.qpos, backend.d.qvel, backend.d.ctrl]
    finally:
        if simulation_clock is not None:
            simulation_wall = time.perf_counter() - simulation_clock - planning_wall
        export_error = verification_error = None
        if unrecorded_state is not None:
            np.savez_compressed(attempt/'unrecorded_last_state.npz', state=unrecorded_state)
        if hasattr(backend, 'sonic') and backend.sonic is not None:
            try:
                backend.export(attempt)
            except Exception as exc:
                export_error = type(exc).__name__ + ': ' + str(exc)
                failure = failure or 'evidence_export_error'
            if backend.states:
                try:
                    native_validation = backend.verify(attempt)
                except Exception as exc:
                    verification_error = type(exc).__name__ + ': ' + str(exc)
                    native_validation = {'success': False, 'error': verification_error, 'physics_steps': 0}
                    failure = failure or 'independent_verification_error'
        if hasattr(backend, 'm'):
            try:
                final_observation = backend.observe()
            except Exception as exc:
                observation_error = type(exc).__name__ + ': ' + str(exc)
                failure = failure or 'final_observation_error'
        backend.close()
        if native_validation is not None:
            _write(attempt/'independent_validation.json', native_validation)
        _write(attempt/'commands.json', commands)
        _write(attempt/'tracking.json', tracking)
        _write(attempt/'actual_modes.json', modes)
        if stage_events:
            stage_events[-1]['end_s'] = float(backend.states[-1][0]) if backend.states else 0.
        _write(attempt/'stages.json', stage_events)
        samples = getattr(backend, 'samples', [])
        _write(attempt/'samples.json', samples)
        acceptance = evaluate(samples, task['acceptance'])
        accepted = bool(failure is None and acceptance['success'] and native_validation
                        and native_validation['success'])
        values = np.array([r['right']['position_error_m'] for r in tracking])
        orientation = np.array([r['right']['orientation_error_rad'] for r in tracking])
        metrics = {'tracking_sample_count': len(tracking),
                   'right_position_rmse_m': float(np.sqrt(np.mean(values**2))) if len(values) else None,
                   'right_position_p95_m': float(np.percentile(values, 95)) if len(values) else None,
                   'right_position_max_m': float(values.max()) if len(values) else None,
                   'right_orientation_rmse_rad': float(np.sqrt(np.mean(orientation**2))) if len(orientation) else None,
                   'right_orientation_p95_rad': float(np.percentile(orientation, 95)) if len(orientation) else None,
                   'right_orientation_max_rad': float(orientation.max()) if len(orientation) else None,
                   'simulation_seconds': float(samples[-1]['time']) if samples else 0.,
                   'reference_coverage_fraction': len(samples) / len(demo['time_s'])}
        result = {'success': accepted, 'failure': failure, 'native_acceptance': acceptance,
                  'independent_native_validation': native_validation, 'metrics': metrics,
                  'attempts': 1, 'scene_count': 1, 'physics_steps': executed_physics_steps,
                  'recorded_samples': len(samples),
                  'experiment_id': record['experiment_id'], 'started_at': physical_start,
                  'ended_at': _now(), 'method': 'GPT annotated contact trajectory retargeting',
                  'original_actions_visible': True, 'out': str(out)}
        attempt_result = {**result, 'attempt': 1, 'acceptance': acceptance,
                          'observation': final_observation, 'selected': accepted}
        package_error = None
        if accepted:
            try:
                result['successful_trajectory'] = str(write_success_package(attempt, task, attempt_result))
            except Exception as exc:
                package_error = type(exc).__name__ + ': ' + str(exc)
                accepted = False; failure = 'success_packaging_error'
                result.update(success=False, failure=failure)
                attempt_result.update(success=False, failure=failure, selected=False)
        _write(attempt/'result.json', attempt_result)
        _write(out/'result.json', result)
        audit = {'contract_sha256': _sha(contract), 'input_sha256': hashes,
                 'actual_modes': sorted({str(row['mode']) for row in modes}),
                 'original_prefix_tokens_executed': sum(row['mode'] == 'original_token_prefix' for row in modes) / 4,
                 'fallback_used': False, 'ik_used': False,
                 'contact_lift_stage_names': lift_names, 'handoff_recorded': handoff is not None,
                 'native_success': acceptance['success'], 'accepted': accepted,
                 'safety_failure': failure if failure in ('hand_contact_gate', 'contact_lost',
                    'unexpected_table_contact', 'robot_fall_or_simulation_warning') else None,
                 'finger_semantics': 'Original parquet commands in prefix; measured source finger joint angles after prefix',
                 'execution_failure': failure, 'physics_steps': executed_physics_steps,
                 'recorded_samples': len(samples),
                 'missing_measurement_steps': executed_physics_steps - len(samples)}
        audit.update(export_error=export_error, verification_error=verification_error,
                     observation_error=observation_error, package_error=package_error)
        _write(out/'execution_audit.json', audit)
        _write(out/'timing.json', {'started_at': started_at, 'ended_at': _now(),
               'planning_s': planning_wall, 'simulation_wall_s': simulation_wall,
               'simulation_time_s': metrics['simulation_seconds'], 'rendering_s': 'unknown_not_rendered',
               'total_wall_s': time.perf_counter() - total_start,
               'planning_scope': 'Handoff bridge only; external annotation and generation timed separately'})
    return result
