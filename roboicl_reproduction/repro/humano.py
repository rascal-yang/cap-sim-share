"""Pure G1 + Dex3 action adaptation; no simulator imports or execution."""
import math
from numbers import Real

PROFILE = 'humano_g1_dex3_world_delta_10hz_v1'
REQUIRED_SONIC_MODE = 1
STAGES = ('approach', 'contact', 'lift', 'hold', 'walk', 'kick', 'sit', 'push', 'press', 'recover')


def _array_schema(size, **extra):
    return {'type': 'array', 'minItems': size, 'maxItems': size,
            'items': {'type': 'number'}, **extra}


# Shared TRAIN/LIVE row contract. Norm/finite/state/model-limit constraints
# are enforced by encode_actions, beyond JSON Schema's scalar capabilities.
ACTION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['stage', 'left_delta', 'right_delta', 'left_fingers', 'right_fingers'],
    'properties': {
        'stage': {'type': 'string', 'enum': list(STAGES)},
        **{side+'_delta': _array_schema(6, description='World dx,dy,dz (norm <= .03 m), rx,ry,rz rotvec (norm <= .15 rad)') for side in ('left', 'right')},
        **{side+'_fingers': _array_schema(7, description='Absolute radians in actual model hand actuated-joint order; actual limits required') for side in ('left', 'right')},
        'velocity_xy': _array_schema(2, description='World m/s, norm <= .5; defaults to zero'),
        'facing_yaw': {'type': 'number', 'minimum': -math.pi, 'maximum': math.pi},
        'pelvis_height_m': {'type': 'number', 'minimum': .4, 'maximum': .85, 'default': .74},
        'torso_quaternion_wxyz': _array_schema(4, description='Absolute world orientation; unit within .05, normalized'),
    },
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('Expected finite numeric value, excluding bool')
    return float(value)


def _vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'Expected {size} numeric entries')
    return [_number(x) for x in value]


def _norm(values):
    return math.hypot(*values)


def _quaternion(value):
    q = _vector(value, 4)
    norm = _norm(q)
    if norm < 1e-6 or abs(norm-1) > .05:
        raise ValueError('Quaternion must be unit length within .05, wxyz')
    return [x/norm for x in q]


def _rotate_world(rotvec, q):
    angle = _norm(rotvec)
    scale = math.sin(angle/2)/angle if angle > 1e-10 else .5
    a, b, c, d = [math.cos(angle/2), *[x*scale for x in rotvec]]
    w, x, y, z = q
    return _quaternion([a*w-b*x-c*y-d*z, a*x+b*w+c*z-d*y,
                        a*y-b*z+c*w+d*x, a*z+b*y-c*x+d*w])


def encode_actions(rows, state, joint_limits, horizon=32):
    """Validate entire chunk then return backend.execute(action, remaining) inputs.

    state uses Backend.observe() world wrist/root/torso fields. joint_limits is
    {'left': [[lo, hi], ... seven], 'right': ...} from the loaded real model.
    No target/limit clipping, gripper mapping, replay, or backend side effects.
    Caller must recheck measured state and actual mode diagnostics at execution.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 32:
        raise ValueError('horizon must be an integer in 1..32')
    if not isinstance(rows, list) or not 1 <= len(rows) <= horizon:
        raise ValueError('Invalid action chunk length')
    if not isinstance(state, dict) or not isinstance(joint_limits, dict):
        raise ValueError('Observed state and actual joint limits required')
    if set(joint_limits) != {'left', 'right'}:
        raise ValueError('Actual limits required for both hands')
    root = _vector(state.get('root_pose'), 7)[:3]
    positions, quaternions, limits = {}, {}, {}
    for side in ('left', 'right'):
        positions[side] = _vector(state.get(side+'_position'), 3)
        quaternions[side] = _quaternion(state.get(side+'_quaternion_wxyz'))
        ranges = joint_limits[side]
        if not isinstance(ranges, (list, tuple)) or len(ranges) != 7:
            raise ValueError('Seven actual joint limits required')
        limits[side] = [_vector(pair, 2) for pair in ranges]
        if any(lo >= hi for lo, hi in limits[side]):
            raise ValueError('Joint lower limit must be below upper limit')
    torso = _quaternion(state.get('torso_quaternion_wxyz'))
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row)-set(ACTION_SCHEMA['properties']) or set(ACTION_SCHEMA['required'])-set(row):
            raise ValueError('Unknown or missing action channels')
        if row['stage'] not in STAGES:
            raise ValueError('Invalid stage')
        velocity = _vector(row.get('velocity_xy', [0, 0]), 2)
        height = _number(row.get('pelvis_height_m', .74))
        if _norm(velocity) > .5 or not .4 <= height <= .85:
            raise ValueError('Invalid locomotion target')
        action = {'stage': row['stage'], 'duration_s': .1,
                  'velocity_xy': velocity, 'pelvis_height_m': height}
        if 'facing_yaw' in row:
            yaw = _number(row['facing_yaw'])
            if not -math.pi <= yaw <= math.pi:
                raise ValueError('Facing yaw outside [-pi, pi]')
            action['facing_yaw'] = yaw
        # Optional torso command holds the last specified target within chunk.
        torso = _quaternion(row.get('torso_quaternion_wxyz', torso))
        action['torso_quaternion_wxyz'] = torso[:]
        for side in ('left', 'right'):
            delta = _vector(row[side+'_delta'], 6)
            fingers = _vector(row[side+'_fingers'], 7)
            if _norm(delta[:3]) > .03 or _norm(delta[3:]) > .15:
                raise ValueError('Wrist delta exceeds norm bounds')
            if any(not lo <= f <= hi for f, (lo, hi) in zip(fingers, limits[side])):
                raise ValueError('Finger target exceeds actual model limits')
            position = [p+d for p, d in zip(positions[side], delta[:3])]
            if _norm([p-r for p, r in zip(position, root)]) > 1.5:
                raise ValueError('Accumulated wrist target outside observed root workspace')
            positions[side] = position
            quaternions[side] = _rotate_world(delta[3:], quaternions[side])
            action[side+'_position'] = position[:]
            action[side+'_quaternion_wxyz'] = quaternions[side][:]
            action[side+'_fingers'] = fingers
        result.append(action)
    return result
