"""Validate and freeze the controlled origin batch; never import or run physics."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = ROOT / 'experiments/20261015_1047_roboicl_origin_live'
IDLE_PROBE = ROOT / 'experiments/20261015_1205_roboicl_idle_probe'
TASK_BUDGET = {'attempts': 3, 'sim_seconds': 30, 'feedback_calls': 80,
               'max_rows': 5, 'recovery_attempts': 2, 'optimization_attempts': 0}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def validate_tests(evidence):
    if not isinstance(evidence.get('command'), str) or not evidence['command'].strip():
        raise ValueError('Preflight must record the actual test command')
    for key in ('exit_code', 'passed', 'failed', 'physics_steps'):
        if type(evidence.get(key)) is not int:
            raise ValueError('Preflight integer evidence required: ' + key)
    if evidence['exit_code'] != 0 or evidence['failed'] != 0 or evidence['passed'] <= 0:
        raise ValueError('Preflight tests did not pass')
    if evidence['physics_steps'] != 0:
        raise ValueError('Preflight tests must not launch a physics experiment')
    if evidence.get('skipped', 0) and not evidence.get('skip'):
        raise ValueError('Skipped checks require an explicit reason')


def validate_runtime(runtime, expected_hashes):
    if runtime.get('passed') is not True or runtime.get('sessions_loaded') is not True:
        raise ValueError('Actual loaded model sessions required')
    providers = runtime.get('actual_session_providers', {})
    if set(providers) != set(expected_hashes) or any(v != ['CPUExecutionProvider'] for v in providers.values()):
        raise ValueError('All three actual SONIC sessions must use only CPUExecutionProvider')
    if runtime.get('model_sha256') != expected_hashes:
        raise ValueError('SONIC model hashes differ from the frozen baseline')
    expected = {'tensorrt': '10.13.0.35', 'onnxruntime': '1.23.2',
                'mujoco': '3.3.6', 'gear_sonic': '0.1.0'}
    if any(runtime.get('versions', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('Runtime versions differ from the approved profile')
    if runtime.get('sonic_version') != '1.0' or runtime.get('tensorrt_used_for_sonic_inference') is not False:
        raise ValueError('SONIC release/inference device evidence mismatch')
    if runtime.get('physics_steps') != 0:
        raise ValueError('Runtime preflight must load models without physics')


def read_creation(out):
    created = read(out / 'created.json')
    stamp = created.get('created_at', '')
    if created.get('experiment_id') != out.name or not re.fullmatch(r'\d{8}_\d{4}_.+', out.name):
        raise ValueError('Creation evidence and experiment ID disagree')
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+08:00', stamp):
        raise ValueError('Actual second-resolution creation timestamp with +08:00 required')
    moment = datetime.fromisoformat(stamp)
    if moment.utcoffset() != timedelta(hours=8) or moment.strftime('%Y%m%d_%H%M') != out.name[:13]:
        raise ValueError('Creation timestamp minute differs from the experiment ID')
    return stamp


def validate_task(task, baseline):
    for field in ('name', 'backend', 'instruction', 'resources', 'scene', 'acceptance', 'references', 'skills'):
        if task.get(field) != baseline.get(field):
            raise ValueError('Undeclared task change: ' + field)
    if task.get('scene') != {'translation': [0, 0, 0]}:
        raise ValueError('This batch permits the origin condition only')
    if task.get('budget') != TASK_BUDGET:
        raise ValueError('Task budget must match the frozen controlled batch budget')
    checks = task.get('policy_checks', {})
    if checks.get('sonic_mode') != 1 or checks.get('require_reflection') is not True:
        raise ValueError('Encoder mode1 and between-attempt reflection are required')


def validate_train_copy(out, baseline):
    """Permit image relocation only; reject any changed measured TRAIN content."""
    old = read(baseline / 'train_memory.json')
    new = read(out / 'train_memory.json')
    expected = deepcopy(old)
    image_paths = []
    for observation in expected['observations']:
        replacements = []
        for value in observation.get('images', []):
            previous = Path(value).resolve()
            try:
                relative = previous.relative_to((baseline / 'train_images').resolve())
            except ValueError as error:
                raise ValueError('TRAIN image is outside the verified image bundle') from error
            current = (out / 'train_images' / relative).resolve()
            if not current.is_file() or sha(current) != sha(previous):
                raise ValueError('TRAIN image bytes changed: ' + str(current))
            replacements.append(str(current))
            image_paths.append(current)
        if 'images' in observation:
            observation['images'] = replacements
    if new != expected:
        raise ValueError('TRAIN observations differ from the verified sparse extraction')
    if read(out / 'source_evidence.json') != read(baseline / 'source_evidence.json'):
        raise ValueError('Source evidence changed during TRAIN relocation')
    return image_paths


def validate_contract(contract, template):
    def visit(value, shape, path):
        if value is None or value == '':
            raise ValueError('Unfilled contract field: ' + path)
        if isinstance(shape, dict):
            if not isinstance(value, dict):
                raise ValueError('Contract object required: ' + path)
            for key, child in shape.items():
                if key not in value:
                    raise ValueError('Missing contract field: ' + path + '.' + key)
                visit(value[key], child, path + '.' + key)
    visit(contract, template, 'experiment')
    if not contract['cases'] or not contract['control']['stage_interfaces']:
        raise ValueError('Cases and stage interfaces must be explicit')
    for stage in contract['control']['stage_interfaces']:
        for key in template['control']['stage_entry_required_fields']:
            if key not in stage or stage[key] is None or stage[key] == '':
                raise ValueError('Incomplete stage interface: ' + key)


def checked_resources(directory):
    if sha(directory / 'experiment.json') != (directory / 'contract.sha256').read_text().strip():
        raise ValueError('Historical contract changed: ' + str(directory))
    contract = read(directory / 'experiment.json')
    resources = read(directory / 'asset_and_code_manifest.json')
    if resources != contract['pipeline']['source_snapshot_and_hashes']:
        raise ValueError('Historical manifest differs from its frozen contract')
    changed = [p for p, h in resources.items() if not Path(p).is_file() or sha(p) != h]
    if changed:
        raise ValueError('Historical resource changed: ' + repr(changed))
    return contract, resources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=Path)
    out = parser.parse_args().experiment.resolve()
    if not out.is_dir():
        raise ValueError('Create and timestamp the experiment before preparation')
    if (out / 'contract.sha256').exists() or (out / 'experiment.json').exists():
        raise FileExistsError('Refusing to overwrite an existing contract')
    created_at = read_creation(out)
    contract, resources = checked_resources(BASELINE)
    _, idle_resources = checked_resources(IDLE_PROBE)
    for path, digest in idle_resources.items():
        if path in resources and resources[path] != digest:
            raise ValueError('Historical resource baselines disagree: ' + path)
        resources[path] = digest
    baseline_files = HERE / 'docs/baseline_files.json'
    original = read(baseline_files)
    changed = [p for p, h in original.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    if changed:
        raise ValueError('Original pipeline changed: ' + repr(changed))
    task = read(out / 'task.json')
    validate_task(task, read(BASELINE / 'task.json'))
    images = validate_train_copy(out, BASELINE)
    runtime = read(out / 'runtime.json')
    validate_runtime(runtime, contract['runtime']['model_hashes'])
    tests = read(out / 'preflight_tests.json')
    validate_tests(tests)
    closure_preflight = read(out / 'closure_profile_preflight.json')
    if closure_preflight.get('passed') is not True or closure_preflight.get('physics_steps') != 0:
        raise ValueError('Archived-scene zero-physics closure profile verification required')
    required_sources = [HERE / name for name in ('controlled_run.py', 'controlled_prompt.txt', 'dispatch_protocol.txt',
                        'repro/controlled_bridge.py', 'repro/robust_session.py', 'freeze_controlled.py')]
    if any(not p.is_file() for p in required_sources):
        raise ValueError('Controlled execution sources are incomplete')
    shutil.copyfile(HERE / 'controlled_prompt.txt', out / 'planner_prompt.txt')
    write(out / 'original_pipeline_check.json', {'checked': len(original), 'changed': [], 'passed': True,
                                                'manifest_sha256': sha(baseline_files), 'physics_steps': 0})
    write(out / 'train_copy_check.json', {'baseline': str(BASELINE), 'measured_observations_unchanged': True,
          'selected_frames': read(out / 'source_evidence.json')['selected_frames'],
          'source_evidence_unchanged': True, 'image_bytes_unchanged': True,
          'copied_images_sha256': {str(p): sha(p) for p in images}, 'physics_steps': 0})
    contract.update(experiment_id=out.name, created_at=created_at, creation_timestamp_precision='seconds; actual created.json',
                    frozen_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'), record_status='frozen')
    contract['purpose'] = {
        'hypothesis': 'Sparse measured TRAIN plus actual standing-idle and measured stability/contact feedback can support newly generated closed-loop grasp actions at origin.',
        'baseline_paths': [str(BASELINE), str(IDLE_PROBE), read(out / 'source_evidence.json')['source']],
        'planned_changes': ['Independent controlled runner and bridge; original pipeline files unchanged',
            'Standing zero-speed lower-planner tuple mode4/direction[1,0,0]/requested height -> mode0/direction[0,0,0]/height-1; encoder remains1',
            'Persistent measured stability and actual standing-idle admission before contact/lift/new finger closure',
            'Transient contact summaries and measured control status included in LIVE',
            'Bounded zero-step malformed-response repair with verified fresh-child request/response provenance',
            'Prompt explicitly identifies original COM marker and tabletop apple; source images unchanged'],
        'invariants': ['Same origin scene, G1/Dex3 assets, initial qpos/qvel/ctrl/warmstart and controller history',
            'Same 13 sparse measured TRAIN frames, four image pairs and source validation; image paths relocated only',
            'Same .005s physics, .02s encoder/decoder, .5s lower-planner refresh, PD and action bounds',
            'Same native contact/fall/collision/lift thresholds and independent acceptance',
            'Same SONIC1.0 CPU models and runtime; zero original token/action replay, no IK or encoder mode0',
            'At most3 sequential origin attempts,30 simulation seconds/attempt,80 decisions/attempt; stop on success; no optimization'],
        'known_confounders': ['This is generated-action feasibility at one origin scene, not the prescribed ten-condition comparison or isolated controller causal estimate',
            'Standing-idle tuple, admission gates, feedback and prompt change together; newly generated actions differ',
            'Idle probe did not meet the complete stability threshold; standing-idle alone is not evidence of settling',
            'TRAIN source used historical motion prefix then mode0; measured observations are not mode1 action labels',
            'Head+external RGB and privileged simulation feedback differ from original RoboICL cameras/interfaces',
            'Fresh child agents have no inherited conversation but share filesystem; instruction restriction is not an OS sandbox',
            'Sequential attempts include explicit prior-episode reflections and are not independent samples',
            'Native environment reset performs pre-restore physics before SONIC construction; current actual calls/controls and exact state restoration are audited separately. Historical reset step counts are unknown, not retroactively inferred.',
            'Simulation pauses while waiting for a model; no real-time deployment claim']}
    contract['cases'] = [{'id': 'origin', 'task': str(out / 'task.json'), 'sha256': sha(out / 'task.json'), 'translation_m': [0, 0, 0]}]
    contract['runtime'].update(actual_loaded_session_evidence=str(out / 'runtime.json'), model_hashes=runtime['model_sha256'],
                               python_version=sys.version.split()[0] + '; freeze executable ' + sys.executable,
                               gear_sonic_package_version=runtime['versions']['gear_sonic'])
    contract['physics'].update(asset_and_initial_state_manifest=str(out / 'asset_and_code_manifest.json'),
                               physics_parameters_manifest=str(out / 'source_evidence.json'))
    gate = {'sample_dt_s': .005, 'planar_speed_max_m_s': .03, 'root_local_angular_z_max_rad_s': .1,
            'continuous_duration_s': .5, 'scope': 'persistent across commands/chunks; every .005s interval must be covered by a logged actual transformed standing-idle feed and requested idle; reset each episode or on motion/gap/non-idle; repeated/zero-step reads earn no time',
            'protected_commands': 'stage contact or lift, or any new directional finger closure regardless of stage',
            'closure_definition': 'Per-joint signed change toward loaded DEX3_HAND init_qpos-to-close_qpos_left/right; tolerance1e-6rad; unknown-direction changes gated conservatively; actual model joint order verified',
            'additional_requirements': 'Requested xy velocity norm<=1e-6m/s and height>=.72m; actual latest lower-planner transformed record is mode0,target_vel-1,direction0,height-1. Adapter transforms actual zero-speed mode4 feeds only.',
            'rejection': 'Zero physics steps, measured LIVE unchanged; legal opening/retreat/unchanged-finger hold can regain stability',
            'interpretation': 'qvel[5] is root-local angular z, not exact Euler yaw derivative; closure direction proxy is not proof of contact or grasp'}
    contract['control']['admission_gate'] = gate
    interface = deepcopy(contract['control']['stage_interfaces'][0])
    interface.update(stage='approach/contact/lift/hold/recover; all permitted schema stages use the same declared interface',
        lower_body_interface='SONIC plan_legs .5s refresh: nonzero velocity walking enum1; standing zero-speed mode4,target_vel-1,height>=.72 transformed to IDLE enum0,direction[0,0,0],height-1; lower zero-speed heights remain squat enum4. Encoder stays1.',
        target_source='Each decision is a fresh fork_turns=none child using only frozen sparse TRAIN, current LIVE/RGB, declared past-episode reflection and control status',
        handoff_rule='Start every new chunk from measured wrists; cumulative target increments only inside current chunk. Recheck admission per command. Commit executed prefix and discard interrupted suffix before new inference. No fallback or source replay.')
    initialization = {'stage': 'native environment initialization before exact original-state restoration',
        'mode': 'Native reset controls before SONIC construction; no encoder/decoder mode during this stage',
        'lower_body_interface': 'Existing native environment reset; actual MuJoCo step calls, dt and controls recorded without suppressing or changing them',
        'upper_body_interface': 'Existing native environment reset controls; no model-generated action or SONIC inference',
        'action_fields_units_frames': 'Native reset controls logged in run/reset_physics.json; restore original qpos/qvel/ctrl/qacc_warmstart and controller checkpoint history before LIVE',
        'target_source': 'Unchanged original task environment reset and frozen initial checkpoint',
        'interpolation': 'Not applicable: native reset behavior before command interpolation',
        'ik_used': False, 'native_prefix_source_and_hash': 'No source trajectory/token prefix replay; initial checkpoint is hashed in asset manifest',
        'handoff_rule': 'Discard native default state, restore frozen original physical/controller state, then initialize SONIC and capture first LIVE; verify reset_physics.json restore evidence'}
    contract['control']['stage_interfaces'] = [initialization, interface]
    contract['control']['actual_mode_log'] = 'attempt_NNN/run/reset_physics.json for pre-SONIC initialization; run/controller.json for subsequent encoder; run/planner_inputs.json before/after actual lower-planner tuple; run/stability_gate.json and run/closure_profile.json for admission'
    contract['control']['action_bounds'] = {'command_duration_s': .1, 'rows_per_decision': [1, 5],
        'wrist_translation_norm_m': .03, 'wrist_world_rotvec_norm_rad': .15, 'velocity_xy_norm_m_s': .5,
        'pelvis_height_m': [.4, .85], 'standing_idle_min_height_m': .72, 'facing_yaw_rad': [-3.141592653589793, 3.141592653589793],
        'wrist_root_distance_max_m': 1.5, 'finger_targets': '7 absolute radians per hand in measured model actuated order/ranges',
        'quaternion': 'unit world wxyz, norm tolerance.05; normalized; rotvec increments left-multiply world rotation'}
    contract['pipeline'].update(entrypoint='cap-sim/scripts/humano_python.sh cap-sim/roboicl_reproduction/controlled_run.py --experiment EXP --attempt N',
        ordered_steps=['Verify frozen contract and all resource hashes', 'Audit inherited native reset physics with forwarding wrapper; restore exact original state/controller history and verify compiled physics/actual CPU providers',
            'Load frozen TRAIN and explicitly separated previous-episode reflection', 'Capture measured LIVE/RGB/control status and hash the request',
            'Fresh fork_turns=none child response with agent identity, request/response SHA and ready marker',
            'Verify transport provenance; validate complete chunk; malformed content gets bounded zero-step repair',
            'Check measured stability, actual standing-idle and closure admission before each command',
            'Execute unchanged native physics/control via independent bridge; record actual encoder/planner feeds and transient contacts',
            'Commit executed-prefix receipts; hard failure terminal; tracking interruption discards suffix',
            'Export saved states, contacts and independent zero-new-step native validation', 'Reflect before another reset; retain success and stop'],
        planner_identity_and_prompt_hash={'identity': 'Fresh isolated Codex child for every decision, fork_turns=none; exact backend serving ID unavailable',
            'prompt_sha256': sha(out / 'planner_prompt.txt'), 'dispatch_protocol_sha256': sha(HERE / 'dispatch_protocol.txt'),
            'conversation_inherited': False, 'filesystem_security_boundary': False,
            'per_decision_evidence': 'attempt_NNN/planner_provenance.json and mailbox/*_response.provenance.json; request/response SHA, child agent identity, fork_turns none'},
        external_model_calls=False, isolated_child_model_calls=True,
        external_model_calls_definition='No separately deployed external API; model decisions are actual fresh Codex child calls, not parent-authored actions',
        original_actions_visible='No raw source commands, trajectory files, tokens or keyframes in planner input; sparse measured TRAIN/RGB only. Source controller had prefix/mode0; live original-token replay length=0. Initial checkpoint controller history remains internal and is not replayed.',
        feedback_and_reflection_rules='Completed/interrupted chunks commit actual LIVE before inference. Invalid envelope/JSON/actions allow max2 zero-step corrections. Gate rejection never advances time. Hard collision/fall terminal; transport failure stops without parent planner substitution. Reflection before next original-state reset is explicit PAST_EPISODE_EXTENSION.',
        allowed_retry_changes=['New model-generated legal wrist/hand/velocity/height/torso/yaw targets only; same frozen physics, interface, gates, acceptance and initial state'],
        optimization_enabled=False)
    git = subprocess.run(['git', '-C', str(ROOT / 'cap-sim'), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    contract['pipeline']['git_commit_and_dirty_patch'] = {'commit': git.stdout.strip() if git.returncode == 0 else 'unknown; source snapshots identify actual files',
        'original_pipeline_modified': False, 'additions': 'roboicl_reproduction controlled runner/bridge/transport/freeze and tests',
        'uncommitted_code_evidence': 'asset_and_code_manifest.json plus recoverable source_snapshot_index.json'}
    for skill in contract['pipeline']['skill_versions_and_hashes']:
        if sha(skill['path']) != skill['sha256']:
            raise ValueError('Declared skill version changed: ' + skill['path'])
    contract['budget'] = {'attempts_per_case': 3, 'recovery_attempts': 2, 'simulation_seconds_per_attempt': 30,
                          'feedback_calls': 80, 'maximum_rows_per_decision': 5, 'optimization_attempts': 0,
                          'successful_attempt_rule': 'Preserve first independently accepted success; no optimization retries'}
    contract['metrics'].update(sampling_and_alignment='200Hz measured states/contacts/stability;50Hz encoder records;.5s actual lower-planner feeds; .1s command endpoint receipts, including true terminal tick',
        tracking_error_definition='Measured wrist endpoint residual relative to requested command endpoint; >.05m position or >.5rad orientation twice within a chunk interrupts its suffix. Not interpolation tracking RMSE; per-tick interpolation RMSE/P95/max unknown.',
        base_drift_definition='World root xyz displacement from stage/episode start,m; planar speed norm(qvel[:2]),m/s; absolute root-local angular z abs(qvel[5]),rad/s',
        stability_definition=gate,
        transient_contact_definition='Per-command measured peak forces and sampled intervals: hand-object>.1N, robot-table>0N; source samples200Hz. Interval evidence may end before next planner decision. Native hard collision remains>5N continuously.05s.',
        timing_definition='Actual ISO8601 +08:00 start/end per attempt; planning wait, simulation compute, simulated seconds, rendering and total wall time separate; unmeasured preparation or review durations unknown',
        physics_step_accounting='Saved action-prefix physics_steps and simulation_s exclude native pre-restore reset calls; record reset call count/dt/controls separately in run/reset_physics.json. Do not infer historical reset counts.',
        failure_truncation_rule='Actual executed prefix and elapsed time only; no padding or zero errors for unexecuted stages. Report successes/attempts and one scene, with failure stage and coverage.')
    diff = {'baseline': str(BASELINE), 'idle_evidence': str(IDLE_PROBE),
            'unchanged': contract['purpose']['invariants'], 'declared_changes': contract['purpose']['planned_changes'],
            'limitations': contract['purpose']['known_confounders'], 'admission_gate': gate,
            'historical_resources_changed': [], 'original_pipeline_files_checked': len(original),
            'same_task_resources': task['resources'], 'same_native_acceptance': task['acceptance'],
            'same_source_physics': read(out / 'source_evidence.json')['physics']}
    write(out / 'comparison_diff.json', diff)
    evidence_names = ('created.json', 'task.json', 'train_memory.json', 'source_evidence.json', 'runtime.json',
                      'planner_prompt.txt', 'preflight_tests.json', 'comparison_diff.json', 'original_pipeline_check.json', 'train_copy_check.json',
                      'initial_controller_history_audit.json', 'visual_identity_audit.json', 'closure_profile_preflight.json')
    files = [*required_sources, *HERE.glob('*.py'), *(HERE / 'repro').rglob('*.py'), *(HERE / 'tests').rglob('*.py'),
             *(HERE / 'vendor/roboicl').rglob('*.py'), *images, *[out / name for name in evidence_names],
             *out.glob('*.py'), *[p for p in [out / 'runtime_check.log'] if p.is_file()], *out.glob('*red*.txt'), *out.glob('*green*.txt'), *out.glob('*preflight*.json'),
             baseline_files, ROOT / 'AGENTS.md', ROOT / 'cap-sim/docs/experiment-record.template.json',
             ROOT / 'cap-sim/docs/experiment-records.md', ROOT / 'cap-sim/runtime/humano-cpu.json']
    for directory in (BASELINE, IDLE_PROBE):
        files.extend(directory / name for name in ('experiment.json', 'contract.sha256', 'asset_and_code_manifest.json'))
    for skill in contract['pipeline']['skill_versions_and_hashes']:
        files.append(Path(skill['path']))
    for key in ('stdout_path', 'stderr_path', 'output_path', 'red', 'green'):
        if isinstance(tests.get(key), str):
            path = Path(tests[key]); path = path if path.is_absolute() else out / path
            if not path.is_file():
                raise ValueError('Missing referenced test evidence: ' + str(path))
            files.append(path)
    resources.update({str(path.resolve()): sha(path) for path in files})
    write(out / 'asset_and_code_manifest.json', resources)
    snapshots = {}
    snapshot_dir = out / 'source_snapshots'; snapshot_dir.mkdir(exist_ok=True)
    for path, digest in resources.items():
        if Path(path).suffix in ('.py', '.json', '.sh', '.txt', '.md', '.hpp', '.h', '.cpp', '.log'):
            dest = snapshot_dir / digest
            if not dest.exists():
                shutil.copyfile(path, dest)
            if sha(dest) != digest:
                raise ValueError('Recoverable source snapshot hash mismatch: ' + str(dest))
            snapshots[path] = {'sha256': digest, 'snapshot': str(dest)}
    write(out / 'source_snapshot_index.json', snapshots)
    contract['pipeline']['source_snapshot_and_hashes'] = resources
    contract['preflight'] = {'passed': True, 'comparison_diff_path': str(out / 'comparison_diff.json'),
        'evidence_paths': [str(out / name) for name in evidence_names],
        'runtime_gate': 'Freeze checks loaded CPU session/model/version evidence; runner verifies compiled physics, providers and encoder modes before/during execution; actual lower-planner tuples and stability must authorize protected commands',
        'offline_tests': tests, 'physics_steps': 0,
        'closure_profile_preflight': str(out / 'closure_profile_preflight.json'),
        'reset_preflight_note': 'A no-mj_step sentinel blocked inherited full reset before stepping. Closure validation uses the archived compiled scene directly with zero physics. Actual runs preserve and audit the native reset step; full reset is not claimed to be zero-step.'}
    contract['execution'] = {'started_at': 'unknown until actual attempt_NNN/started.json',
        'ended_at': 'unknown until actual attempt_NNN/timing.json', 'contract_sha256': 'contract.sha256 sidecar',
        'audit_path': str(out / 'execution_audit.json'), 'timing_path': 'attempt_NNN/timing.json', 'results_path': str(out / 'results.json')}
    validate_contract(contract, read(ROOT / 'cap-sim/docs/experiment-record.template.json'))
    contract['frozen_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    write(out / 'experiment.json', contract)
    (out / 'contract.sha256').write_text(sha(out / 'experiment.json') + '\n')
    print('FROZEN', out.name)


if __name__ == '__main__':
    main()
