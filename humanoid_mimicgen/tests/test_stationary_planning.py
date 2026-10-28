import numpy as np
import mujoco
from humanoid_mimicgen.stationary_planning import StationaryPlanner


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="torso_link" pos="0 0 1"><joint name="waist_yaw_joint" type="hinge" axis="0 0 1" range="-90 90"/><geom type="sphere" size=".01" contype="0" conaffinity="0"/><body name="right_wrist_yaw_link" pos=".4 0 0"><joint name="right_shoulder_pitch_joint" type="hinge" axis="0 1 0" range="-90 90"/><geom type="sphere" size=".01" contype="0" conaffinity="0"/></body><body name="left_wrist_yaw_link" pos="-.4 0 0"><joint name="left_shoulder_pitch_joint" type="hinge" axis="0 1 0" range="-90 90"/><geom type="sphere" size=".01" contype="0" conaffinity="0"/></body></body></worldbody></mujoco>''')


def test_already_reached_uses_arms_only_and_never_steps(monkeypatch):
    monkeypatch.setattr(mujoco,'mj_step',lambda *a,**k:(_ for _ in ()).throw(AssertionError('physics forbidden')))
    m=model();q=np.zeros(m.nq);p=StationaryPlanner(m);targets=p.wrist_poses(q);result=p.solve(q,targets,max_seeds=1)
    assert result['success'] and result['group']=='arms'
    np.testing.assert_array_equal(result['qpos'],q)
    path,info=p.connect(q,result['qpos'],result['joint_ids'])
    assert path is not None
    np.testing.assert_array_equal(q,np.zeros(m.nq))


def test_unreachable_target_reports_failure_without_changing_root():
    m=model();p=StationaryPlanner(m);q=np.zeros(m.nq);targets=p.wrist_poses(q);targets['right'][0]+=10
    result=p.solve(q,targets,max_seeds=1,max_nfev=10)
    assert result['success'] is False
    assert result['reason']=='no_stationary_solution'
    np.testing.assert_array_equal(q,np.zeros(m.nq))
