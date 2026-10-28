import copy
import numpy as np
import pytest
from types import SimpleNamespace


def test_controller_extension_roundtrip_and_missing_state_rejected():
    from humanoid_mimicgen.augmentation.rollback_state import snapshot,restore_extensions
    leg={'offset':np.array([.1,.2,.3])}
    wrist=SimpleNamespace(feedback=np.array([.01,0,-.02]))
    fingers=SimpleNamespace(integral=np.ones(7)*.2,previous=np.arange(7,dtype=np.float32))
    saved=snapshot(leg,wrist,fingers)
    leg['offset'][:]=0;wrist.feedback[:]=0;fingers.integral[:]=0;fingers.previous=None
    restore_extensions(saved,leg,wrist,fingers)
    assert np.array_equal(leg['offset'],[.1,.2,.3])
    assert np.array_equal(wrist.feedback,[.01,0,-.02])
    assert fingers.previous.dtype==np.float32 and fingers.previous[-1]==6
    with pytest.raises(ValueError):restore_extensions({},leg,wrist,fingers)
    with pytest.raises(ValueError):restore_extensions(saved,None,wrist,fingers)


def test_changed_policy_cannot_inherit_affected_prefix():
    from humanoid_mimicgen.augmentation.rollback_state import validate_branch
    parent={'contact_start':100,'contact_end':160,'tail_s':2,'wrist_delta_m':[0,0,0], 'finger_control':{'start_frame':80,'offset':0}}
    child=copy.deepcopy(parent);child['finger_control']['offset']=.3
    validate_branch(parent,child,80)
    with pytest.raises(ValueError):validate_branch(parent,child,81)
    validate_branch(parent,parent,130)
    child=copy.deepcopy(parent);child['wrist_delta_m'][2]=-.01
    validate_branch(parent,child,100)
    with pytest.raises(ValueError):validate_branch(parent,child,101)


def test_deadline_excludes_overrun_success_and_counts_shared_cost():
    from humanoid_mimicgen.augmentation.rollback_benchmark import budget_metrics
    rows=[{'elapsed_s':12,'success':False},{'elapsed_s':24,'success':True}]
    assert budget_metrics(rows,30,8)=={'attempts':1,'success':False,'first_success_s':None,'overrun_attempts':1}
    assert budget_metrics(rows,40,8)['first_success_s']==32
    with pytest.raises(ValueError):budget_metrics(rows,30,-1)


def test_compiler_changes_only_allowed_fields_and_preserves_resume():
    from humanoid_mimicgen.augmentation.rollback_plan import compile_parameters
    base={'leg_response':{'calibration_amplitude_rad':.03,'ramp_s':.8,'feedback_timescale_s':2,'max_offset_rad':.15,'max_rate_rad_s':.075},'tail_s':2,'contact_start':352,'contact_control':{'joints':[]},'resume':{'kind':'restart','case':'','frame':0},'wrist_delta_m':[0,0,0]}
    patch={'tail_s':3,'feedback_timescale_s':.6,'max_offset_rad':.22,'max_rate_rad_s':.2}
    d=compile_parameters('chair',base,patch)
    assert d['leg_response']['max_offset_rad']==.22 and d['leg_response']['ramp_s']==.8
    assert base['leg_response']['max_offset_rad']==.15 and d['resume']==base['resume']
    with pytest.raises(ValueError):compile_parameters('chair',base,patch|{'contact_start':400})
    with pytest.raises(ValueError):compile_parameters('chair',base,patch|{'max_offset_rad':9})


def test_api_queue_rejects_tail_that_truncates_annotated_phase():
    from humanoid_mimicgen.augmentation.rollback_benchmark import validate_candidates
    base={'tail_s':1,'contact_start':346,'contact_end':443,'wrist_delta_m':[0,0,.05],'contact_control':{'joints':[]}}
    spec={'profile':'apple','base_plan':base,'source_frames':394}
    cs=[{'parameters':{'tail_s':1,'wrist_delta_m':[0,0,.05-i*.001]},'rationale':'test'} for i in range(8)]
    validate_candidates({'candidates':cs},spec)
    cs[1]['parameters']['tail_s']=0
    with pytest.raises(ValueError,match='horizon'):validate_candidates({'candidates':cs},spec)


def test_queue_errors_identify_all_bad_candidates_and_duplicate_indices():
    from humanoid_mimicgen.augmentation.rollback_benchmark import validate_candidates
    base={'tail_s':1,'contact_start':346,'contact_end':443,'wrist_delta_m':[0,0,.05],'contact_control':{'joints':[]}}
    spec={'profile':'apple','base_plan':base,'source_frames':394}
    cs=[{'parameters':{'tail_s':1,'wrist_delta_m':[0,0,.05-i*.001]},'rationale':'test'} for i in range(8)]
    cs[1]['parameters']['tail_s']=1.25;cs[2]=copy.deepcopy(cs[0]);cs[6]['parameters']['tail_s']=1.75
    with pytest.raises(ValueError) as err:validate_candidates({'candidates':cs},spec)
    for message in ('candidate[1]','candidate[6]','candidate[2] duplicates candidate[0]','20ms'):
        assert message in str(err.value)


def test_larger_queue_preserves_existing_candidates_before_expanding():
    from humanoid_mimicgen.augmentation.rollback_benchmark import validate_candidates,candidate_schema
    base={'tail_s':1,'contact_start':346,'contact_end':443,'wrist_delta_m':[0,0,.05],'contact_control':{'joints':[]}}
    cs=[{'parameters':{'tail_s':1,'wrist_delta_m':[0,0,.05-i*.001]},'rationale':'test'} for i in range(12)]
    spec={'profile':'apple','base_plan':base,'source_frames':394,'candidate_count':12,'preserved_candidate_parameters':[copy.deepcopy(c['parameters']) for c in cs[:8]]}
    validate_candidates({'candidates':cs},spec)
    assert candidate_schema('apple',12)['candidates']['minItems']==12
    cs[2]['parameters']['tail_s']=2
    with pytest.raises(ValueError,match='preserved'):validate_candidates({'candidates':cs},spec)
