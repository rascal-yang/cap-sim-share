import numpy as np
import pytest
from humanoid_mimicgen.workflow.contracts import WorkflowError


def test_model_geometry_mismatch_rejected():
    import mujoco
    from humanoid_mimicgen.workflow.native import validate_model_identity
    a=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size="1" type="sphere"/></worldbody></mujoco>')
    b=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size="2" type="sphere"/></worldbody></mujoco>')
    validate_model_identity(a,a)
    with pytest.raises(WorkflowError,match='model'):validate_model_identity(a,b)


def test_measured_entry_requires_declared_pose_bounds():
    from humanoid_mimicgen.workflow.execution import entry_ready
    target={'right':np.array([0,0,0,1,0,0,0.])}
    actual={'right':np.array([.03,0,0,1,0,0,0.])}
    limits={'position_m':.02,'orientation_rad':.1}
    assert not entry_ready(actual,target,['right'],limits)
    assert entry_ready(target,target,['right'],limits)


def test_nonfinite_audit_is_explicit_unknown():
    from humanoid_mimicgen.workflow.execution import finite_record
    assert finite_record({'force':float('nan'),'q':[float('inf'),1]})=={'force':None,'q':[None,1]}


def test_strategy_choice_cannot_invent_hand_or_recovery():
    from humanoid_mimicgen.workflow.selection import validate_selection
    evidence={'capabilities':{'options':[{'id':'right_demo'}]},'failure':{'reason':'contact_lost'}}
    p={'action':'select','option_id':'invented','evidence_ids':['failure'],'reason':'retry','failure_hypothesis':'lag','expected_condition':'contact'}
    with pytest.raises(WorkflowError):validate_selection(p,evidence)
    p['option_id']='right_demo'
    assert validate_selection(p,evidence)==p


def test_saved_selection_is_bound_to_observations(tmp_path):
    import json
    from humanoid_mimicgen.workflow.selection import replay_selection
    p=tmp_path/'response.json';p.write_text('{}')
    (tmp_path/'provenance.json').write_text(json.dumps({'status':'completed','evidence_sha256':'different'}))
    with pytest.raises(WorkflowError,match='context_changed'):replay_selection(p,{'scene':{}})


def test_interrupted_step_still_publishes_failure(tmp_path,monkeypatch):
    import json
    from humanoid_mimicgen.workflow import verification
    root=tmp_path/'attempt';out=root/'execution';out.mkdir(parents=True)
    targets=tmp_path/'targets.npz';np.savez(targets,fingers=np.zeros((2,14)))
    policy={'acceptance':{'clearance_m':.01,'hold_duration_s':1.,'root_drift_m':.1,'support_force_N':1},'gates':{'contact_force_N':1}}
    contract={'workflow':{'strategy_id':'test','alignment':{'command_dt_s':.005},'strategy':{'policy':policy,'phases':[{'start':0,'end':2,'id':'hold','requires_clearance':True}]}},'inputs':{'targets':str(targets)}}
    monkeypatch.setattr(verification,'check_contract',lambda _:contract)
    audit={'physics_steps':1,'original_token_prefix_count':0,'mode0_calls':0,'midrun_state_resets':0,'fallback':False,'completed':False,'failure':'execution_error','stage':'approach','actual_encoder_modes':[1], 'started_at':'test','ended_at':'test','simulation_wall_s':1.,'online_planning_wall_s':0.,'sim_seconds':.005}
    (out/'execution_audit.json').write_text(json.dumps(audit))
    for name in ('observations','commands'):(out/(name+'.json')).write_text('[]')
    np.savez(out/'states.npz',samples=np.array([]))
    (root/'request.json').write_text(json.dumps({'strategy_root':str(tmp_path/'strategies')}))
    result=verification.verify(root)
    assert result['status']=='failed' and 'incomplete_measurements' in result['verification_errors']
    assert (root/'reflection.json').exists()
