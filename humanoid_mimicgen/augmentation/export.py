"""Hash-checked workflow exports with explicit pre-action/post-action alignment."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np

from .contracts import require, verify_inputs
from ..workflow.contracts import WorkflowError, canonical, file_hash
from ..workflow.verification import checked_retry_groups


def read(path):
    return json.loads(Path(path).read_text())


def _success_checks(contract, audit, result, observations, commands, dt):
    if result.get('physical_task_completed') is not True:
        return False
    s = contract['workflow']['strategy']; policy = s['policy']
    require(not policy.get('grasp_trigger'), 'Runtime GPT exports need their own approval verification', 'unsupported_capability')
    require(audit['completed'] and not audit['failure'] and not result.get('verification_errors'),
            'Completion audit inconsistent', 'success_evidence')
    retry = audit.get('recovery_count', 0)
    groups, errors = checked_retry_groups(commands, retry, policy.get('recovery', {}).get('max_retries', 0))
    require(not errors, 'Retry sequence invalid', 'success_evidence')
    with np.load(contract['inputs']['targets'], allow_pickle=False) as a:
        fingers = a['fingers']
        for group in groups:
            indices = [c['source_index'] for c in group]
            require(indices == list(range(s['phases'][0]['start'], s['phases'][0]['start']+len(indices))),
                    'Source sequence incomplete', 'success_evidence')
            require(all(0 <= c['source_index'] < len(fingers) and np.array_equal(c['fingers_rad'], fingers[c['source_index']]) for c in group),
                    'Source finger command mismatch', 'success_evidence')
    require(groups[-1] and groups[-1][-1]['source_index'] == s['phases'][-1]['end']-1,
            'Final phase incomplete', 'success_evidence')
    acceptance = policy['acceptance']; hold_ids = {p['id'] for p in s['phases'] if p['requires_clearance']}
    held = [o for o in observations if o['stage'] in hold_ids and o.get('retry_index', 0) == retry]
    require(bool(held), 'Missing measured hold', 'success_evidence')
    consecutive = longest = 0.
    for o in observations:
        holding = o['stage'] in hold_ids and o.get('retry_index', 0) == retry
        valid = holding and o['clearance_m'] >= acceptance['clearance_m'] and o['contacts']['hand_object_N'] > policy['gates']['contact_force_N'] and o['contacts']['support_N'] < acceptance['support_force_N']
        require(not holding or valid, 'Hold conditions failed', 'success_evidence')
        consecutive = consecutive+dt if valid else 0.
        longest = max(longest, consecutive)
    drift = max(np.linalg.norm(np.asarray(o['root_pose'][:2])-held[0]['root_pose'][:2]) for o in held)
    require(longest+1e-9 >= acceptance['hold_duration_s'] and drift <= acceptance['root_drift_m'],
            'Hold duration or root stability failed', 'success_evidence')
    return True


def frozen_evidence(root, hashes):
    resolved = {}
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_file() or file_hash(path) != expected:
            # Historical implementation may have advanced; only its exact saved snapshot qualifies.
            parts = path.parts
            require('humanoid_mimicgen' in parts and path.suffix == '.py', 'Frozen evidence changed: '+name, 'integrity_error')
            relative = Path(*parts[parts.index('humanoid_mimicgen')+1:])
            path = root/'source_snapshot'/relative
            require(path.is_file() and file_hash(path) == expected, 'Source snapshot changed: '+name, 'integrity_error')
        resolved[str(path)] = expected
    return resolved


def _validate_existing(destination, sources):
    manifest = read(destination/'manifest.json')
    require(manifest['source_hashes'] == sources, 'Export source changed', 'integrity_error')
    verify_inputs({str(destination/p): sha for p, sha in manifest['artifact_hashes'].items()})
    return manifest['summary']


def export_case(experiment, destination):
    root = Path(experiment).resolve(); destination = Path(destination).resolve(); out = root/'execution'
    paths = [root/'experiment.json', root/'experiment.sha256', root/'native_preflight.json', root/'result.json']
    paths += [out/name for name in ('actual_initial.npz', 'states.npz', 'commands.json', 'attempted_commands.json',
                                   'observations.json', 'events.json', 'execution_audit.json')]
    videos = {name: root/'video'/(name+'.mp4') for name in ('external', 'ego') if (root/'video'/(name+'.mp4')).is_file()}
    paths += list(videos.values())
    paths += [root/'video'/name for name in ('audit.json', 'frame_mapping.json') if (root/'video'/name).is_file()]
    try:
        sources = {str(p): file_hash(p) for p in paths}
        require(file_hash(root/'experiment.json') == (root/'experiment.sha256').read_text().strip(),
                'Frozen contract changed', 'integrity_error')
        c = read(root/'experiment.json')
        require(c['record_status'] == 'frozen_ready' and c['preflight']['passed'], 'Unfrozen experiment', 'integrity_error')
        if c.get('control',{}).get('motion_noise'):
            paths += [out/'motion_noise.npz', out/'motion_noise.json']
            sources.update({str(p): file_hash(p) for p in paths[-2:]})
        resolved_evidence = frozen_evidence(root, c['preflight']['evidence_sha256'])
        if destination.exists(): return _validate_existing(destination, sources)
        meta = read(root/'native_preflight.json'); nq, nv, nu = (meta[k] for k in ('model_nq', 'model_nv', 'model_nu'))
        dt = meta['physics_dt_s']; require(dt > 0, 'Invalid physics period', 'alignment_error')
        with np.load(out/'states.npz', allow_pickle=False) as archive: states = archive['samples'].copy()
        with np.load(out/'actual_initial.npz', allow_pickle=False) as archive:
            initial = {k: archive[k].copy() for k in ('qpos', 'qvel', 'ctrl')}
        commands = read(out/'commands.json'); attempted = read(out/'attempted_commands.json')
        observations = read(out/'observations.json'); audit = read(out/'execution_audit.json'); result = read(root/'result.json')
        n = len(commands)
        require(n > 0 and states.shape == (n, 1+nq+nv+nu) and len(observations) == n and
                all(audit[k] == n for k in ('physics_steps', 'recorded_steps', 'completed_commands')),
                'Measured state/command count mismatch', 'alignment_error')
        require(all(initial[k].shape == (size,) for k, size in (('qpos', nq), ('qvel', nv), ('ctrl', nu))),
                'Initial state dimensions', 'alignment_error')
        require(len(attempted) >= n and attempted[:n] == commands, 'Attempted/executed prefix mismatch', 'alignment_error')
        times = np.array([x['time_s'] for x in commands], float)
        require(abs(times[0]) <= 1e-8, 'Missing initial-time anchor', 'alignment_error')
        require(np.allclose(states[:, 0], times+dt, atol=1e-8, rtol=0) and
                np.allclose([o['time_s'] for o in observations], states[:, 0], atol=1e-8, rtol=0) and
                (n == 1 or np.allclose(np.diff(times), dt, atol=1e-8, rtol=0)), 'Pre/post timestamps mismatch', 'alignment_error')
        require(all(x['mode'] == 1 for x in commands) and audit['actual_encoder_modes'] == [1] and
                not any(audit[k] for k in ('original_token_prefix_count', 'mode0_calls', 'midrun_state_resets', 'fallback')),
                'Executed interface differs', 'integrity_error')
        arrays = {'time_s': times, 'next_time_s': states[:, 0],
                  'qpos': np.vstack([initial['qpos'], states[:-1, 1:1+nq]]),
                  'qvel': np.vstack([initial['qvel'], states[:-1, 1+nq:1+nq+nv]]),
                  'next_qpos': states[:, 1:1+nq], 'next_qvel': states[:, 1+nq:1+nq+nv],
                  'action_ctrl': np.array([x['ctrl_Nm'] for x in commands], float),
                  'body_target_rad': np.array([x['body_target_rad'] for x in commands], float),
                  'fingers_rad': np.array([x['fingers_rad'] for x in commands], float),
                  'mode': np.array([x['mode'] for x in commands], int),
                  'source_index': np.array([x['source_index'] for x in commands], int),
                  'retry_index': np.array([x.get('retry_index', 0) for x in commands], int)}
        require(arrays['action_ctrl'].shape == (n, nu) and np.array_equal(arrays['action_ctrl'], states[:, 1+nq+nv:]),
                'Applied control differs from command', 'alignment_error')
        require(all(np.isfinite(a).all() for a in arrays.values()) and np.isfinite(states).all(), 'Nonfinite trajectory', 'alignment_error')
        if c.get('control',{}).get('motion_noise'):
            config = c['control']['motion_noise']; noise_meta = read(out/'motion_noise.json')
            require(all(noise_meta['config'].get(k)==config[k] for k in noise_meta['config']), 'Noise configuration mismatch', 'integrity_error')
            require(file_hash(out/'motion_noise.npz')==noise_meta['npz_sha256'], 'Noise record changed', 'integrity_error')
            with np.load(out/'motion_noise.npz', allow_pickle=False) as noise:
                require(len(noise['time_s'])==len(attempted) and np.array_equal(noise['time_s'][:n],times), 'Noise time alignment', 'alignment_error')
                require(np.array_equal(noise['executed_body_target_rad'][:n],arrays['body_target_rad']) and
                        np.array_equal(noise['executed_ctrl_Nm'][:n],arrays['action_ctrl']), 'Noise action alignment', 'alignment_error')
                arrays['nominal_body_target_rad']=noise['nominal_body_target_rad'][:n].copy()
                arrays['nominal_ctrl_Nm']=noise['nominal_ctrl_Nm'][:n].copy()
                arrays['noise_policy_tick']=noise['policy_tick'][:n].copy()
                arrays['body_target_noise_rad']=arrays['body_target_rad']-arrays['nominal_body_target_rad']
                indices=noise_meta['arm_target_indices']; other=[i for i in range(arrays['body_target_rad'].shape[1]) if i not in indices]
                require(len(indices)==14 and np.all(arrays['body_target_noise_rad'][:,other]==0), 'Unexpected perturbed joints', 'integrity_error')
                require(all(np.isfinite(a).all() for a in arrays.values()), 'Nonfinite noise labels', 'alignment_error')
        physical = _success_checks(c, audit, result, observations, commands, dt)
        summary = {'schema_version': 1, 'experiment_id': c['experiment_id'], 'path': str(destination),
            'physical_success': physical, 'transitions': n, 'attempted_commands': len(attempted),
            'action_representation': 'sonic_mode1_commands_and_applied_ctrl', 'sonic_tokens_available': False,
            'training_ready': result.get('training_ready') is True,
            'research_kind': 'trajectory_retargeting', 'original_sequences_accessible': True,
            'alignment': 'qpos/qvel before action; next_qpos/next_qvel after one physics step',
            'physics_steps_in_exporter': 0, 'visual_acceptance': 'unknown',
            'videos': {name: {'path': 'evidence/video/'+p.name, 'sha256': sources[str(p)],
                              'visual_acceptance': 'unknown'} for name, p in videos.items()}}
        # A result flag alone is not a visual acceptance certificate.
        summary['training_ready'] = False
        if c.get('control',{}).get('motion_noise'):
            summary['motion_noise']=c['control']['motion_noise']
            summary['intended_action_label']='nominal_body_target_rad; applied actions retained separately'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.export-', dir=destination.parent) as tmp:
            staging = Path(tmp)
            np.savez_compressed(staging/'trajectory.npz', **arrays)
            for p in paths:
                target = staging/'evidence'/p.relative_to(root); target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
            (staging/'metadata.json').write_bytes(canonical(summary))
            verify_inputs(sources)
            artifacts = {str(p.relative_to(staging)): file_hash(p) for p in staging.rglob('*') if p.is_file()}
            manifest = {'source_hashes': sources, 'artifact_hashes': artifacts, 'summary': summary,
                        'frozen_input_hashes': c['preflight']['evidence_sha256'], 'resolved_evidence_hashes': resolved_evidence}
            (staging/'manifest.json').write_bytes(canonical(manifest))
            try: os.rename(staging, destination)
            except OSError:
                if not destination.is_dir(): raise
                return _validate_existing(destination, sources)
        return summary
    except WorkflowError:
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise WorkflowError('invalid_export', type(exc).__name__) from None


def export_job(store, job_id):
    row = store.get(job_id)
    declared = row['job']['conditions']
    names = [c['case_id'] for c in declared['cases']] if declared['mode'] == 'explicit' else [f'case_{i:06d}' for i in range(declared['count'])]
    require(all(name in row['cases'] and row['cases'][name]['stage'] in ('verified', 'failed', 'exported', 'interrupted') for name in names), 'Batch still has unfinished cases', 'incomplete_workflow')
    destination = store.root/'datasets'/job_id
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        case = row['cases'].get(name)
        item = {'case_id': name, 'physical_success': False}
        if case is None or case['stage'] not in ('verified', 'failed', 'exported', 'interrupted'):
            item.update(status='unavailable', reason_code='case_not_finished')
        elif case['details'].get('experiment_path') and (Path(case['details']['experiment_path'])/'result.json').is_file():
            root = Path(case['details']['experiment_path'])
            partition = 'success' if read(root/'result.json').get('physical_task_completed') is True else 'failure'
            try: item.update(export_case(root, destination/partition/name), status='exported')
            except WorkflowError as exc: item.update(status='rejected', reason_code=exc.code)
        else:
            item.update(status='diagnostic', reason_code=case['details'].get('reason_code', 'missing_trace'), details=case['details'])
        results.append(item)
    successes = sum(x['physical_success'] for x in results)
    index = {'schema_version': 1, 'job_id': job_id, 'case_count': len(names), 'successes': successes,
             'success_rate': successes/len(names), 'cases': results, 'job': row['job'],
             'training_ready': False, 'physics_steps_in_exporter': 0}
    target = destination/'index.json'
    encoded = canonical(index)
    if target.exists():
        require(target.read_bytes() == encoded, 'Dataset index changed; use a new export revision', 'integrity_error')
    else:
        # Publish atomically without replacing a concurrently completed index.
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as f: f.write(encoded); temporary = Path(f.name)
        try:
            try: os.link(temporary, target)
            except FileExistsError: require(target.read_bytes() == encoded, 'Concurrent export conflict', 'integrity_error')
        finally: temporary.unlink()
    return {'index_path': str(target), 'case_count': len(names), 'successes': successes,
            'success_rate': successes/len(names), 'rejected': sum(x['status'] == 'rejected' for x in results)}
