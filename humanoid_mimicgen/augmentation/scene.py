"""Build initial states only; never teleport during execution."""
import copy
import json
from pathlib import Path
import shutil
import numpy as np
from .contracts import baseline_request, require
from .conditions import vector
from ..workflow.contracts import canonical, file_hash
from ..privileged import no_physics


def transform_initial(model, initial, root_body, object_body, robot_delta_xyyaw, object_delta_xy, tolerance):
    import mujoco
    vector(robot_delta_xyyaw, 3); vector(object_delta_xy, 2)
    result = {key: np.array(value, copy=True) for key, value in initial.items()}
    for name, size in (('qpos', model.nq), ('qvel', model.nv), ('ctrl', model.nu)):
        require(name in result and result[name].shape == (size,) and np.isfinite(result[name]).all(), 'Initial ' + name)
    require(np.isfinite(tolerance) and tolerance >= 0, 'Penetration tolerance')
    ids = []
    for name in (root_body, object_body):
        try:
            body = model.body(name).id
        except KeyError:
            require(False, 'Missing body ' + name)
        jid = int(model.body_jntadr[body])
        require(jid >= 0 and model.body_jntnum[body] == 1 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE,
                'Body requires a free joint: ' + name)
        ids.append(int(model.jnt_qposadr[jid]))
    require(ids[0] == 0 and ids[1] != 0, 'Distinct root and object free joints')
    q = result['qpos']
    for adr in ids:
        require(np.isclose(np.linalg.norm(q[adr+3:adr+7]), 1., atol=1e-6), 'Initial unit quaternion')
    with no_physics():
        data = mujoco.MjData(model); data.qpos[:] = q; mujoco.mj_forward(model, data)
        previous = {}
        for c in data.contact:
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            previous[pair] = min(previous.get(pair, 0.), float(c.dist))
        q[:2] += robot_delta_xyyaw[:2]
        yaw = robot_delta_xyyaw[2]
        rotation = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
        quat = np.empty(4); mujoco.mju_mulQuat(quat, rotation, q[3:7]); q[3:7] = quat
        q[ids[1]:ids[1]+2] += object_delta_xy
        # World translational root velocity follows initial yaw; local angular velocity does not.
        x, y = result['qvel'][:2].copy()
        result['qvel'][:2] = [np.cos(yaw)*x-np.sin(yaw)*y, np.sin(yaw)*x+np.cos(yaw)*y]
        data.qpos[:] = q; mujoco.mj_forward(model, data)
        for c in data.contact:
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            require(float(c.dist) >= min(0., previous.get(pair, 0.))-tolerance,
                    'New or increased initial penetration', 'initial_collision')
    return result


def materialize_case(job, condition, directory):
    import mujoco
    request = baseline_request(job)
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=False)
    files = request['files']; source_hashes = {k: file_hash(p) for k, p in files.items()}
    with no_physics():
        model = mujoco.MjModel.from_binary_path(files['model'])
        with np.load(files['initial'], allow_pickle=False) as archive:
            initial = {k: archive[k].copy() for k in archive.files}
        b = request['strategy']['bindings']
        transformed = transform_initial(model, initial, b['root_body'], b['object'],
            condition['robot_delta_xyyaw'], condition['object_delta_xy'],
            request['strategy']['policy']['planning']['penetration_tolerance_m'])
    np.savez_compressed(directory/'initial.npz', **transformed)
    # Copy evidence so queued jobs do not depend on mutable request files.
    for name, path in list(files.items()):
        dest = directory/(name+Path(path).suffix)
        if name != 'initial': shutil.copyfile(path, dest)
        files[name] = str(dest)
    task = json.loads(Path(files['task']).read_text())
    task['resources']['initial_state'] = files['initial']
    Path(files['task']).write_bytes(canonical(task))
    if 'stance_hint' in files and job['stance_hint_frame'] == 'object_relative':
        hint = json.loads(Path(files['stance_hint']).read_text())
        vector(hint['xyyaw'], 3)
        hint['xyyaw'][:2] = (np.asarray(hint['xyyaw'][:2])+condition['object_delta_xy']).tolist()
        Path(files['stance_hint']).write_bytes(canonical(hint))
    # A historical study's purpose is not the purpose of this new run.
    files.pop('study_case', None)
    study = {'purpose': {'hypothesis': 'Execute fixed-policy trajectory retargeting under declared initial planar changes',
        'baseline_paths': [job['baseline_request']], 'planned_changes': [condition]+([{'motion_noise':job['motion_noise'],'case_seed':'base seed plus case index; frozen per case'}] if 'motion_noise' in job else []),
        'invariants': ['SONIC encoder1 only', 'No runtime model decisions', 'No physical state resets after initialization'],
        'known_confounders': ['Trajectory retargeting with original source access; not a matched ten-condition controller comparison']}}
    (directory/'study_case.json').write_bytes(canonical(study)); files['study_case'] = str(directory/'study_case.json')
    # Bind the actual augmentation inputs/skill and replace inherited study metadata.
    skill = Path(__file__).parent/'skills/augmentation/SKILL.md'
    shutil.copyfile(skill, directory/'augmentation_skill.md')
    (directory/'augmentation_job.json').write_bytes(canonical(job))
    files['augmentation_skill'] = str(directory/'augmentation_skill.md')
    files['augmentation_job'] = str(directory/'augmentation_job.json')
    record = json.loads(Path(files['record']).read_text())
    record['cases'] = [{'id': condition['case_id'], 'condition': condition,
        'scene_model_sha256': file_hash(files['model']), 'initial_state_sha256': file_hash(files['initial'])}]
    assets = {'model': {'path': files['model'], 'sha256': file_hash(files['model'])},
              'initial': {'path': files['initial'], 'sha256': file_hash(files['initial'])},
              'source_baseline_request': job['baseline_request'], 'condition': condition}
    (directory/'assets.json').write_bytes(canonical(assets)); files['augmentation_assets'] = str(directory/'assets.json')
    record['physics']['asset_and_initial_state_manifest'] = files['augmentation_assets']
    record['pipeline']['skill_versions_and_hashes'].append({'path': files['augmentation_skill'], 'sha256': file_hash(files['augmentation_skill'])})
    record['pipeline']['augmentation'] = {'job': files['augmentation_job'], 'job_sha256': file_hash(files['augmentation_job']),
        'case_id': condition['case_id'], 'entrypoint': 'python -m humanoid_mimicgen.augmentation',
        'execution_model_calls': 0, 'orchestration_model_calls': 'recorded separately in agent session if used'}
    record['pipeline']['allowed_retry_changes'] = 'Existing frozen workflow recovery only; no online model changes'
    record['preflight']['evidence_sha256'].update({files[k]: file_hash(files[k]) for k in
        ('augmentation_assets', 'augmentation_job', 'augmentation_skill')})
    Path(files['record']).write_bytes(canonical(record))
    request['strategy_root'] = str(Path(job['output_root'])/'strategies')
    request['experiment_root'] = str(Path(job['output_root'])/'experiments')
    request['strategy']['provenance']['augmentation'] = {'request_id': job['request_id'], 'condition': copy.deepcopy(condition)}
    (directory/'condition.json').write_bytes(canonical({'condition': condition, 'source_hashes': source_hashes,
        'physics_steps': 0, 'stance_hint_frame': job['stance_hint_frame'],
        'initial_collision_rule': 'No new or increased penetration beyond baseline plus declared tolerance'}))
    out = directory/'request.json'; out.write_bytes(canonical(request))
    return out
