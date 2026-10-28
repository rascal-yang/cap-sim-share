"""Measured stage summaries and evidence-bound reflections; no physical actions."""
import math
from .io import digest


def _vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    value = value[:length]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        return None
    return value


def _distance(a, b):
    return None if a is None or b is None else math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def stage_diagnosis(action, before, result):
    after = result.get('observation', {})
    root0 = _vector(before.get('root_pose'), 3)
    root = _vector(after.get('root_pose'), 3)
    body = action.get('body_keyframes')
    mode = 'native_prefix' if action.get('stage') == 'native_prefix' else (0 if body is not None else 1)
    height = action.get('pelvis_height_m', .74) if mode == 1 else None
    if not isinstance(height,(int,float)) or not math.isfinite(height):height=None
    last_frame=_vector(body[-1],4) if isinstance(body,list) and body else None
    targets = {'pelvis_height_m': height,
               'body_reference_root_height_m': last_frame[3] if last_frame else None}
    measurements = {'actual_pelvis_height_m': root[2] if root else None,
                    'pelvis_height_error_m': root[2]-height if root and height is not None else None,
                    'base_displacement_xyz_m': [x-y for x,y in zip(root,root0)] if root and root0 else None,
                    'root_velocity': after.get('root_velocity'),
                    'contact_s': after.get('contact_s'), 'contacts': after.get('contacts')}
    for side in ('left', 'right'):
        p = _vector(action.get(side+'_position'), 3)
        actual = _vector(after.get(side+'_position'), 3)
        q = _vector(action.get(side+'_quaternion_wxyz'), 4)
        actual_q = _vector(after.get(side+'_quaternion_wxyz'), 4)
        angle = None
        if q and actual_q:
            norm = math.sqrt(sum(x*x for x in q)*sum(x*x for x in actual_q))
            if norm > 0:
                angle = 2*math.acos(min(1., abs(sum(x*y for x,y in zip(q,actual_q))/norm)))
        targets[side+'_wrist_position_m'] = p if mode == 1 else None
        measurements[side+'_wrist_position_m'] = actual
        measurements[side+'_wrist_position_error_m'] = _distance(p,actual) if mode == 1 else None
        measurements[side+'_wrist_orientation_error_rad'] = angle if mode == 1 else None
    object_moves = {}
    for name, pose in after.get('objects', {}).items():
        p0 = _vector(before.get('objects', {}).get(name),3)
        p = _vector(pose,3)
        object_moves[name] = [x-y for x,y in zip(p,p0)] if p and p0 else None
    measurements['object_displacement_xyz_m'] = object_moves
    return {'stage':action.get('stage'), 'name':action.get('name'), 'start_s':before.get('time'),
            'end_s':after.get('time'), 'requested_mode':mode, 'targets':targets,
            'measurements':measurements, 'controller':result.get('controller_diagnostics'),
            'completed':result.get('completed'), 'failure':result.get('failure')}


def check_mode(actions, checks):
    if checks.get('sonic_mode') != 1:
        return
    for a in actions:
        if isinstance(a,dict) and ('body_keyframes' in a or a.get('stage')=='native_prefix'):
            raise ValueError('mode1 policy rejects body_keyframes/native_prefix, including reused prefixes')


def validate_reflection(reflection, result, diagnostics, interface):
    """Bind facts to saved measurements, not to model confidence or prose quality."""
    if not isinstance(reflection,dict):
        raise ValueError('reflection must be an object')
    if reflection.get('source_attempt') != result['attempt'] or reflection.get('result_sha256') != digest(result):
        raise ValueError('reflection references a different attempt/result hash')
    for key in ('failed_stage','hypothesis','expected_effect','reject_if'):
        if not isinstance(reflection.get(key),str) or not reflection[key].strip():
            raise ValueError('reflection missing '+key)
    if reflection.get('hypothesis_outcome') not in ('supported','not_supported','inconclusive'):
        raise ValueError('reflection must state hypothesis outcome')
    if not isinstance(reflection.get('alternative_causes'),list):
        raise ValueError('reflection must distinguish alternative causes')
    facts = reflection.get('observed_facts')
    if not isinstance(facts,list) or not facts:
        raise ValueError('reflection requires evidence facts')
    source = {'result':result, 'diagnostics':diagnostics}
    for fact in facts:
        try:
            value=source
            for part in fact['path'].split('.'):
                value = value[int(part)] if isinstance(value,list) else value[part]
            if value is None or value != fact['value']:
                raise ValueError('reflection fact differs from saved measurement')
        except (KeyError,IndexError,TypeError,AttributeError,ValueError) as exc:
            raise ValueError('reflection fact missing or inconsistent with evidence') from exc
    change = reflection.get('next_change')
    if not isinstance(change,dict) or not isinstance(change.get('description'),str) or not change['description'].strip():
        raise ValueError('reflection requires an explicit next change')
    controls = change.get('controls')
    if not isinstance(controls,list) or not all(isinstance(x,str) for x in controls):
        raise ValueError('reflection next change controls invalid')
    allowed = interface.get('action')
    if isinstance(allowed,dict) and not set(controls) <= set(allowed):
        raise ValueError('reflection proposes unavailable control channels')
    return reflection
