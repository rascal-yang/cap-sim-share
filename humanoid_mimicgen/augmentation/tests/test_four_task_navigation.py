import numpy as np
import pytest


def test_task_bindings_and_retention_are_not_apple_for_chair_or_button():
    from humanoid_mimicgen.augmentation.four_task_navigation import task_strategy, retained
    base={'bindings':{'object':'apple','support_geoms':['table_geom'],'hands':['right']}}
    chair=task_strategy(base, {'object':'chair','support_geoms':[],'ground_geoms':['ground']}, 'chair')
    assert chair['bindings']['object']=='chair' and chair['bindings']['support_geoms']==['ground']
    assert retained('chair',[]) is None and retained('button',[]) is None
    assert base['bindings']['object']=='apple'
    ob=[dict(contacts={'hand_object_N':1.,'support_N':0.},clearance_m=.05)]
    assert retained('apple',ob) and retained('plate',ob)


def test_plane_support_height_is_finite():
    import mujoco
    from humanoid_mimicgen.workflow.native import geom_z_bounds
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1" pos="0 0 .2"/></worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    assert geom_z_bounds(m,d,0)==(.2,.2)


def test_translation_preserves_robot_initial_and_source_hand_values():
    from humanoid_mimicgen.augmentation.four_task_navigation import translate_reference
    q=np.zeros((3,16));q[:,3]=1.;q[:,7]=1.;q[:,9]=.5
    result=translate_reference(q,np.array([0,.1,0]),9)
    assert np.array_equal(q[:,1],np.zeros(3))
    assert np.allclose(result[:,1],.1) and np.allclose(result[:,10],.1)
    assert np.array_equal(result[:,3:9],q[:,3:9])


def test_api_numeric_decision_is_real_bounded_and_chair_is_not_wrist_grasp():
    import importlib.util,json
    from pathlib import Path
    path=Path(__file__).resolve().parents[3]/'scripts/run_four_task_navigation_api.py'
    spec=importlib.util.spec_from_file_location('four_api_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def raw(gain):return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({'action':'execute','feedback_gain':gain,'rationale':'numeric prior'})}]}]}
    assert module.validate_decision(raw(2),'apple')['feedback_gain']==2
    with pytest.raises(ValueError):module.validate_decision(raw(2),'chair')
    with pytest.raises(ValueError):module.validate_decision(raw(4),'plate')


def test_button_approved_success_cannot_hide_legacy_force_envelope_violation():
    from humanoid_mimicgen.augmentation.four_task_navigation import button_full_execution
    result={'completed':True,'failure':None,'audit':{'simulator_warning_count':0}}
    obs=[{'root_pose':[0,0,.7,1,0,0,0], 'contacts':{'robot_obstacle_N':0.,'events':[{'bodies':['right_wrist_yaw_link','button'],'force_N':66.}]}}]
    limits={'robot_table_force_N':50,'root_height_min_m':.3,'root_tilt_max_rad':.87,'selected_contact_force_N':65}
    assert not button_full_execution(result,obs,['right_wrist_yaw_link'],{'button'},limits,{'right_hand','right_wrist_yaw_link'})['passed']
    obs[0]['contacts']['events'][0]['force_N']=40.
    assert button_full_execution(result,obs,['right_wrist_yaw_link'],{'button'},limits,{'right_hand','right_wrist_yaw_link'})['passed']


def test_approved_button_envelope_counts_robot_table_not_other_obstacles():
    from humanoid_mimicgen.augmentation.four_task_navigation import button_full_execution
    result={'completed':True,'failure':None,'audit':{'simulator_warning_count':0}}
    ob={'root_pose':[0,0,.7,1,0,0,0],'contacts':{'robot_obstacle_N':200.,'events':[{'bodies':['right_hand','ground'],'force_N':200.},{'bodies':['button','table'],'force_N':100.}]}}
    limits={'robot_table_force_N':50,'root_height_min_m':.3,'root_tilt_max_rad':.87,'selected_contact_force_N':65}
    assert button_full_execution(result,[ob],['right_hand'],{'button'},limits,{'right_hand','right_wrist_yaw_link'})['passed']
    ob['contacts']['events'].append({'bodies':['right_hand','table'],'force_N':51.})
    assert not button_full_execution(result,[ob],['right_hand'],{'button'},limits,{'right_hand','right_wrist_yaw_link'})['passed']


def test_frozen_feedback_metadata_matches_task_active_sides():
    from humanoid_mimicgen.augmentation.four_task_navigation import feedback_metadata
    assert feedback_metadata([],2.,1.)['position_gain']==0
    assert feedback_metadata([],2.,1.)['orientation_gain']==0
    assert feedback_metadata(['left','right'],2.,1.)['active_sides']==['left','right']
    assert feedback_metadata(['right'],2.,1.)['position_gain']==2


def test_center_control_has_distinct_position_id():
    from humanoid_mimicgen.augmentation.four_task_navigation import position_label
    assert position_label(0.)=='center'
    assert position_label(-.1)=='minus10' and position_label(.1)=='plus10'
