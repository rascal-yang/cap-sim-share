"""Measured contact experience, expressed in object and wrist coordinates.

Positions are metres; quaternions are wxyz. Forces are observations, not commands.
Extraction needs saved native MuJoCo evidence; mapping only needs NumPy.
"""
import hashlib
from pathlib import Path
import numpy as np
from .io import read, digest


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError('Expected a finite unit wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def map_prior(prior, observation, scale=1.):
    """Map source contacts using the CURRENT measured object pose, never stale xyz."""
    ratio = float(scale) / float(prior['reference_scale'])
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('Contact mapping requires positive finite scale')
    pose = np.asarray(observation['objects'][prior['object_key']], dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('Invalid measured object pose')
    r = rotation(pose[3:]); targets = []
    measured = observation.get('contacts', {}).get('events', [])
    wrist_r = r @ rotation(prior['wrist_quaternion_object_wxyz'])
    wrist_candidates = []
    for contact in prior['contacts']:
        local = np.asarray(contact['position_object_m'], dtype=float)
        world = pose[:3] + r @ (local * ratio)
        matches = [e for e in measured if contact['hand_body'] in e['bodies']
                   and prior.get('object_body', 'apple') in e['bodies']]
        error = min((float(np.linalg.norm(np.asarray(e['position'])-world)) for e in matches), default=None)
        targets.append({'hand_body':contact['hand_body'], 'position_object_m':(local*ratio).tolist(),
                        'position_world_m':world.tolist(), 'nearest_measured_error_m':error,
                        'measured_contacts':matches})
        wrist_candidates.append(world - wrist_r @ np.asarray(contact['position_wrist_m']))
    wrist = np.mean(wrist_candidates, axis=0) if wrist_candidates else None
    residual = max((float(np.linalg.norm(p-wrist)) for p in wrist_candidates), default=None)
    return {'object_key':prior['object_key'], 'object_pose_world_wxyz':pose.tolist(),
            'scale_ratio':ratio, 'target_contacts':targets,
            'suggested_wrist_position_world_m':None if wrist is None else wrist.tolist(),
            'rigid_wrist_fit_residual_m':residual,
            'force_semantics':'observations_only_not_control_targets',
            'note':'Rigid wrist seed only; finger adaptation, reachability and physical contact remain necessary.'}


def extract_prior(reference, *, object_body, wrist_body, object_key='target',
                  min_duration_s=.2, min_force_N=.1):
    """First interval with >=2 persistent hand links; retain local scatter and evidence."""
    reference = Path(reference).resolve(); trajectory = read(reference)
    if not trajectory.get('acceptance', {}).get('success'):
        raise ValueError('Contact extraction requires a successful reference')
    root = reference.parent
    if not read(root/'independent_validation.json')['success']:
        raise ValueError('Reference lacks independent physical success')
    if digest(read(root/'samples.json')) != trajectory['samples_sha256']:
        raise ValueError('Reference samples hash mismatch')
    import mujoco
    m = mujoco.MjModel.from_binary_path(str(root/'scene.mjb')); d = mujoco.MjData(m)
    states = np.load(root/'states.npz')['samples']; records = read(root/'contacts.json')
    if len(states) != len(records) or not len(states):
        raise ValueError('Contact/state alignment mismatch')
    if states.shape[1] != 1 + m.nq + m.nv + m.nu or not np.isfinite(states).all():
        raise ValueError('Invalid saved physics state layout')
    dt = float(m.opt.timestep)
    if not np.allclose(np.diff(states[:,0]), dt, atol=1e-7):
        raise ValueError('Noncontiguous contact timestamps')
    if not np.isfinite(min_duration_s) or min_duration_s <= 0 or not np.isfinite(min_force_N) or min_force_N <= 0:
        raise ValueError('Positive contact interval and force threshold required')
    width = max(2, int(np.ceil(min_duration_s/dt)))
    def hands(row):
        return {next(b for b in e['bodies'] if b != object_body)
                for e in row['events'] if object_body in e['bodies'] and e['force_N'] >= min_force_N
                and any('hand' in b for b in e['bodies'] if b != object_body)}
    sets = [hands(row) for row in records]; selected = None
    for start in range(len(states)-width+1):
        if len(sets[start]) < 2:continue
        common = set.intersection(*sets[start:start+width])
        if len(common) >= 2:
            selected = (start, start+width, sorted(common)); break
    if selected is None:
        raise ValueError('No persistent multi-link hand contact interval; do not fabricate a prior')
    start, end, links = selected; ob = m.body(object_body).id; wr = m.body(wrist_body).id
    points = {link:[] for link in links}; wrist_points = {link:[] for link in links}; forces = {link:[] for link in links}
    relative_quat = None
    for i in range(start, end):
        row = states[i]; d.qpos[:]=row[1:1+m.nq];d.qvel[:]=row[1+m.nq:1+m.nq+m.nv];d.ctrl[:]=row[1+m.nq+m.nv:]
        mujoco.mj_forward(m,d); ro=d.xmat[ob].reshape(3,3); rw=d.xmat[wr].reshape(3,3)
        if relative_quat is None:
            relative_quat = np.empty(4);mujoco.mju_mat2Quat(relative_quat,(ro.T@rw).reshape(9))
        for link in links:
            events=[e for e in records[i]['events'] if object_body in e['bodies'] and link in e['bodies'] and e['force_N']>=min_force_N]
            weights=np.array([e['force_N'] for e in events]);p=np.average([e['position'] for e in events],axis=0,weights=weights)
            points[link].append(ro.T@(p-d.xpos[ob]));wrist_points[link].append(rw.T@(p-d.xpos[wr]));forces[link].append(float(weights.sum()))
    contacts=[]
    for link in links:
        p=np.asarray(points[link]);w=np.asarray(wrist_points[link]);center=np.median(p,axis=0)
        contacts.append({'hand_body':link,'position_object_m':center.tolist(),
                         'position_wrist_m':np.median(w,axis=0).tolist(),
                         'local_scatter_max_m':float(np.max(np.linalg.norm(p-center,axis=1))),
                         'observed_force_N_percentiles':np.percentile(forces[link],[5,50,95]).tolist()})
    # Use actual reset geometry when available; otherwise label first post-step fallback.
    reset=Path(trajectory['source_attempt'])/'reset.json'
    if reset.exists():initial_pose=read(reset)['observation']['objects'][object_key];initial_source='reset.json'
    else:
        d.qpos[:]=states[0,1:1+m.nq];mujoco.mj_forward(m,d)
        initial_pose=np.r_[d.xpos[ob],d.xquat[ob]].tolist();initial_source='first_recorded_post_step'
    return {'schema_version':1,'object_body':object_body,'object_key':object_key,'wrist_body':wrist_body,
            'reference_scale':float(trajectory['task']['scene'].get('apple_scale',1.)),
            'reference_initial_object_pose':initial_pose,'initial_pose_source':initial_source,
            'wrist_quaternion_object_wxyz':relative_quat.tolist(),'contacts':contacts,
            'selection':{'method':'first_persistent_multi_link_contact','start_row':start,'end_row_exclusive':end,
                         'start_s':float(states[start,0]),'end_s':float(states[end-1,0]),'duration_s':width*dt,
                         'min_force_N':min_force_N,'note':'Persistence is observed; scatter is reported, not a universal safe contact region.'},
            'force_semantics':'observations_only_not_control_targets',
            'source_reference':str(reference),
            'evidence_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                               for name in ['trajectory.json','states.npz','contacts.json','scene.mjb','independent_validation.json','samples.json']}}


def enrich_request(request, priors):
    if not priors:return
    scale = request['task']['scene'].get('apple_scale',1.)
    targets = [map_prior(p,request['observation'],scale) for p in priors]
    initial = request.get('contact_privileged', {}).get('initial_targets', targets)
    request['contact_privileged'] = {'initial_targets':initial, 'source':'simulator_measured_reference_and_current_state',
        'reference_priors':priors,
        'current_targets':targets,
        'instruction':'Use object-frame reference contacts to plan. Compare current measured contacts before lifting; never count target points as achieved contacts.'}
