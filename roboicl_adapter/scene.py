"""Read-only model inspection and rendering; never steps physics."""
import hashlib
from pathlib import Path

import numpy as np


PHYSICS_FIELDS = ('body_mass', 'body_inertia', 'body_pos', 'body_quat',
                  'geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_friction',
                  'geom_solref', 'geom_solimp', 'geom_margin', 'geom_gap',
                  'geom_contype', 'geom_conaffinity', 'dof_damping', 'dof_armature',
                  'dof_frictionloss', 'jnt_range', 'actuator_gainprm',
                  'actuator_biasprm', 'actuator_gear', 'actuator_ctrlrange')


def physics_signature(model):
    fields = {name: hashlib.sha256(np.asarray(getattr(model, name)).tobytes()).hexdigest()
              for name in PHYSICS_FIELDS}
    for name in ('timestep', 'gravity', 'iterations', 'ls_iterations', 'solver',
                 'integrator', 'cone', 'impratio', 'tolerance', 'disableflags', 'enableflags'):
        fields['opt.'+name] = np.asarray(getattr(model.opt, name)).tolist()
    return fields


def geometry(model, data):
    table = model.geom('table_geom').id
    target = model.body('apple_001').id
    return {'table_center_world_m': data.geom_xpos[table].tolist(),
            'table_half_size_m': model.geom_size[table].tolist(),
            'apple_center_world_m': data.xpos[target].tolist(),
            'apple_mass_kg': float(model.body_mass[target]),
            'frame': 'world; +X forward at initial stance, +Y left, +Z up'}


def measured_observation(model, data):
    result = {'time': float(data.time), 'root_pose': data.qpos[:7].tolist(),
              'root_velocity': data.qvel[:6].tolist(), 'geometry': geometry(model, data),
              'body_joints': data.qpos[7:36].tolist(),
              'measured_fingers': [data.qpos[36:43].tolist(), data.qpos[43:50].tolist()]}
    for side in ('left', 'right'):
        body = model.body(side+'_wrist_yaw_link').id
        result[side+'_position'] = data.xpos[body].tolist()
        result[side+'_quaternion_wxyz'] = data.xquat[body].tolist()
    return result


def render_views(model, data, directory, label):
    import mujoco
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.25, 0, .7]
    camera.distance = 2.8
    camera.azimuth = 130
    camera.elevation = -20
    result = []
    with mujoco.Renderer(model, height=360, width=640) as renderer:
        for name, view in [('head', 'head_stereo_left'), ('external', camera)]:
            renderer.update_scene(data, camera=view)
            rgb = renderer.render()
            if not np.isfinite(rgb).all() or np.std(rgb) < 2:
                raise RuntimeError('Empty camera observation')
            path = directory / (label+'_'+name+'.png')
            Image.fromarray(rgb).save(path)
            result.append(str(path.resolve()))
    return result
