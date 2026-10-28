import numpy as np
import pytest
from humanoid_mimicgen.action_replay import stitch_controls, check_contract


def test_shared_boundary_command_is_executed_once():
    source=np.arange(10)[:,None]*10.
    clips=[(2,4,source[2:5]),(4,7,source[4:8]),(7,8,source[7:9])]
    actual=stitch_controls(source,clips)
    assert actual[:,0].tolist()==[0,10,20,30,40,50,60,70,80,90]


def test_gap_in_clips_rejected():
    a=np.arange(10)[:,None]
    with pytest.raises(ValueError):stitch_controls(a,[(2,4,a[2:5]),(5,8,a[5:9])])


def test_modified_action_rejected():
    a=np.arange(10)[:,None];clip=a[2:5].copy();clip[1]=999
    with pytest.raises(ValueError):stitch_controls(a,[(2,4,clip)])


def test_missing_frozen_contract_fails_before_physics(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):check_contract(tmp_path/'experiment.json')


def test_source_solver_warmstart_is_restored_after_forward():
    import mujoco
    from humanoid_mimicgen.action_replay import initialize_source
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    d=mujoco.MjData(m)
    initial={'qpos':m.qpos0.copy(),'qvel':np.zeros(m.nv),'ctrl':np.zeros(m.nu),'qacc_warmstart':np.arange(m.nv)+12.}
    initialize_source(m,d,initial)
    assert d.time==0
    assert d.qacc_warmstart.tolist()==[12.,13.,14.,15.,16.,17.]
