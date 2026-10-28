import copy
from collections import deque
from types import SimpleNamespace
import numpy as np
import pytest
import mujoco


def adapter():
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    s=SimpleNamespace(history=deque([{'q':np.arange(3.)}],maxlen=10),last=np.arange(3.),tick=12,ref=np.arange(72.).reshape(2,36),ref_tick=4,serial=7,navigating=True,handoff_until_tick=30,anchor=np.arange(36.))
    task=SimpleNamespace(_success_hold_count=3,reward=.2,_init_target_pos=np.array([1.,2.,3.]))
    return SimpleNamespace(m=m,d=d,sonic=s,last_target=np.arange(29.),b=SimpleNamespace(e=SimpleNamespace(task=task)))


def test_restore_preserves_solver_and_planner_history():
    from humanoid_mimicgen.augmentation import navigation_checkpoint as c
    a=adapter();a.d.qvel[:]=.3;a.d.qacc_warmstart[:]=.2
    saved=c.capture_adapter(a)
    a.d.qpos[:3]=4;a.d.qvel[:]=9;a.sonic.serial=99;a.sonic.ref[:]=0;a.sonic.history.clear();a.last_target[:]=0;a.b.e.task._success_hold_count=0
    c.restore_adapter(a,saved)
    np.testing.assert_array_equal(a.d.qvel,np.full(6,.3))
    np.testing.assert_array_equal(a.d.qacc_warmstart,np.full(6,.2))
    assert a.sonic.serial==7 and a.sonic.tick==12 and a.sonic.ref_tick==4
    np.testing.assert_array_equal(a.sonic.ref,np.arange(72.).reshape(2,36))
    assert a.sonic.history[0]['q'][2]==2 and a.b.e.task._success_hold_count==3
    assert a.last_target[-1]==28


def test_changed_navigation_cannot_inherit_navigation_but_feedback_can():
    from humanoid_mimicgen.augmentation import navigation_checkpoint as c
    old={'navigation':{'route_limits':{'speed_m_s':.5},'transition_s':4.,'checkpoint':{'enabled':True}},'feedback_gain':2.,'wrist_z_offset_m':0.,'hold_s':1.}
    new=copy.deepcopy(old);new['feedback_gain']=1.5
    c.validate_change(old,new,'parked')
    new['navigation']['route_limits']['speed_m_s']=.3
    with pytest.raises(ValueError,match='navigation'):c.validate_change(old,new,'parked')
    c.validate_change(old,new,'navigation_start')


def test_splice_requires_exact_state_boundary():
    from humanoid_mimicgen.augmentation import navigation_checkpoint as c
    prefix=np.array([[0.,1.],[.005,2.]])
    nexts=np.array([[.005,2.],[.01,3.]])
    c.verify_boundary(prefix,nexts,np.array([.01,3.]))
    with pytest.raises(ValueError,match='boundary'):c.verify_boundary(prefix,nexts,np.array([.01,4.]))


def test_model_identity_is_required_when_reusing_checkpoint_prefix():
    from humanoid_mimicgen.augmentation import navigation_checkpoint as c
    parent={'model_encoder.onnx':'old','model_decoder.onnx':'same'}
    with pytest.raises(ValueError,match='model'):c.validate_models(parent,{'model_encoder.onnx':'new','model_decoder.onnx':'same'})
    c.validate_models(parent,parent.copy())


def test_disabled_checkpoint_option_does_not_enable_capture():
    from humanoid_mimicgen.augmentation import navigation_checkpoint as c
    assert not c.options({'checkpoint':{'enabled':False}})
    assert c.options({'checkpoint':{'resume':{'case':'parent','name':'parked'}}})


def test_parked_branch_can_remove_integral_controller():
    from humanoid_mimicgen.augmentation import navigation_checkpoint as c
    c.restore_phase_state(None,None,{'parking':None,'integral':{'right':np.array([.001,0,0])}})
    integral=SimpleNamespace(bias={'right':np.zeros(3)})
    c.restore_phase_state(None,integral,{'parking':None,'integral':{'right':np.array([.001,0,0])}})
    assert integral.bias['right'][0]==.001


def test_changed_route_cannot_reuse_parked_prefix(tmp_path):
    from humanoid_mimicgen.augmentation import navigation_checkpoint as c
    parent=tmp_path/'parent';child=tmp_path/'child';parent.mkdir();child.mkdir()
    for name in ('scene.mjb','physics.json','task.json','reference.npz','initial.npz','controller_initial.npz','offline_plan.json'):
        (parent/name).write_text('same');(child/name).write_text('same')
    (child/'offline_plan.json').write_text('different route with same reference bundle')
    with pytest.raises(ValueError,match='offline_plan'):c.validate_inputs(parent,child)


def test_source_identity_uses_logical_names_and_rejects_changed_executor():
    from humanoid_mimicgen.augmentation import navigation_checkpoint as c
    p={'/a/repo/executor.py':'a','/parent/run.py':'x'}
    q={'/b/repo/executor.py':'a','/child/run.py':'y'}
    c.validate_code(p,q,'/a/repo','/b/repo')
    q['/b/repo/executor.py']='changed'
    with pytest.raises(ValueError,match='source'):c.validate_code(p,q,'/a/repo','/b/repo')


def test_navigation_start_branch_can_remove_parking():
    from humanoid_mimicgen.augmentation import navigation_checkpoint as c
    c.restore_phase_state(None,None,{'parking':{'fields':{'elapsed':0.},'velocity':np.zeros(2)},'integral':None})
