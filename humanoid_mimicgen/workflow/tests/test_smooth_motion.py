import numpy as np
from humanoid_mimicgen.workflow.motion import DistanceVelocityProfile, recovery_configuration


def test_short_and_long_moves_accelerate_brake_and_arrive():
    for distance in (.02,.5,1.5):
        controller=DistanceVelocityProfile(speed=.5,acceleration=.5,response_s=.6)
        x=np.zeros(2); history=[]
        for _ in range(1500):
            v=controller.step(np.array([distance,0])-x,.02)
            history.append(v.copy()); x+=v*.02
        assert np.linalg.norm(x-[distance,0])<1e-6
        assert max(np.linalg.norm(history,axis=1))<=.5+1e-12
        assert max(np.linalg.norm(np.diff(np.vstack([np.zeros(2),history]),axis=0),axis=1))<=.5*.02+1e-12
        assert np.min(np.array(history)[:,0])>=-1e-9


def test_target_change_limits_vector_acceleration():
    c=DistanceVelocityProfile(speed=.5,acceleration=.5,response_s=.6)
    for _ in range(100):c.step([1,0],.02)
    before=c.velocity.copy();after=c.step([-1,1],.02)
    assert np.linalg.norm(after-before)<=.01+1e-12


def test_recovery_restores_only_upper_body_in_current_root_frame():
    current=np.arange(30,dtype=float);rest=current+100
    addresses=np.array([7,8,10,16])
    goal=recovery_configuration(current,rest,addresses)
    np.testing.assert_array_equal(goal[addresses],rest[addresses])
    frozen=np.ones(30,bool);frozen[addresses]=False
    np.testing.assert_array_equal(goal[frozen],current[frozen])
    np.testing.assert_array_equal(current,np.arange(30))


def test_measured_momentum_brakes_before_arrival_region():
    c=DistanceVelocityProfile(speed=.5,acceleration=.5,response_s=.6)
    v=c.step([.2,0],.02,measured_velocity=[.4,0],arrival_radius=.1)
    np.testing.assert_array_equal(v,[0,0])


def test_leg_handoff_preserves_start_and_ends_at_stance():
    from humanoid_mimicgen.workflow.motion import blend_stance_reference
    walking=np.tile(np.r_[0,0,.8,1,0,0,0,np.ones(29)],(100,1))
    standing=np.r_[0,0,.8,np.cos(.1),0,np.sin(.1),0,np.zeros(29)]
    times=np.arange(100)/30
    result=blend_stance_reference(walking,standing,times,.6)
    np.testing.assert_allclose(result[0],walking[0])
    np.testing.assert_allclose(result[times>=.6],np.tile(standing,(sum(times>=.6),1)),atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(result[:,3:7],axis=1),1)


def test_nonzero_arrival_radius_is_reached_with_inside_target():
    c=DistanceVelocityProfile(speed=.5,acceleration=.5,response_s=.6)
    x=np.zeros(2);v=np.zeros(2)
    for _ in range(1000):
        v=c.step(np.array([.5,0])-x,.02,measured_velocity=v,arrival_radius=.1-.003)
        x+=.02*v
        if np.linalg.norm(x-[.5,0])<=.1:break
    assert np.linalg.norm(x-[.5,0])<=.1


def test_upright_reference_preserves_position_and_heading():
    from scipy.spatial.transform import Rotation
    from humanoid_mimicgen.workflow.motion import upright_root_reference
    q=np.r_[1,2,.8,np.roll(Rotation.from_euler('xyz',[.1,.2,.3]).as_quat(),1)]
    result=upright_root_reference(q)
    np.testing.assert_array_equal(result[:3],q[:3])
    np.testing.assert_allclose(Rotation.from_quat(np.roll(result[3:],-1)).as_euler('xyz'),[0,0,.3],atol=1e-12)
