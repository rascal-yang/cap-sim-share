import pytest
from humanoid_mimicgen.augmentation.direct_augmentation import validate_direct, require_prerequisites, accepted


def decision():
    return {'action':'run','rationale':'small object perturbation','hypothesis':'retarget contact','episode':0,'tail_s':0,'wrist_delta_m':[0,0,0],'contact_start':10,'contact_end':20,'safety_guard':None,'object_shifts_xy_m':[[.01,0],[-.01,0]]}


def test_zero_failure_does_not_block_prerequisites():
    require_prerequisites({'success':True,'independent_success':True,'coverage':1,'failure':None},{'decision':decision()|{'action':'accept'}})
    with pytest.raises(ValueError):require_prerequisites({'success':False},{'decision':decision()})

@pytest.mark.parametrize('shifts', [[[0,0]],[[.03,0]],[[float('nan'),0]],[[.01,0],[.01,0]],[[.01,0]]*3])
def test_invalid_perturbations_rejected(shifts):
    with pytest.raises(ValueError):validate_direct(decision()|{'object_shifts_xy_m':shifts},[0])


def test_valid_pair_and_single_retry():
    assert len(validate_direct(decision(),[0])['object_shifts_xy_m'])==2
    assert validate_direct(decision()|{'action':'retry','object_shifts_xy_m':[[.01,0]]},[0])


def test_api_accept_is_not_enough():
    assert not accepted({'success':False},'accept')
    assert not accepted({'success':True,'independent_success':True,'coverage':.9},'accept')
    assert accepted({'success':True,'independent_success':True,'coverage':1,'failure':None},'accept')

@pytest.mark.parametrize('change',[{'task':'wrong'},{'episode':1},{'mode':'cartesian_source_legs'},{'object_delta_xy_m':[.01,0]}])
def test_prerequisite_identity_mismatch_blocks_before_files(change):
    from humanoid_mimicgen.augmentation.direct_augmentation import validate_binding
    base={'task':'Task','episode':0,'mode':'original_token','object_delta_xy_m':[0,0]}
    with pytest.raises(ValueError):validate_binding({'task':'Task','dataset':'/tmp/Task'},base|change,{'decision':decision()})


def test_inspection_failure_keeps_physics_record(tmp_path):
    import json
    from humanoid_mimicgen.augmentation.direct_augmentation import retain_failed_attempt
    folder=tmp_path/'attempt_00';folder.mkdir()
    r={'experiment_path':str(tmp_path/'physical'),'simulation_wall_s':2,'success':False}
    (folder/'physics.result.json').write_text(json.dumps(r))
    history=[];retain_failed_attempt(history,folder,0,'render failed')
    assert len(history)==1 and history[0]['physical_result']==r
    retain_failed_attempt(history,folder,0,'review failed')
    assert len(history)==1


def test_candidates_share_experiment_id_namespace(tmp_path):
    from humanoid_mimicgen.augmentation.direct_augmentation import physical_storage_root
    task=tmp_path/'Task'
    assert physical_storage_root(task/'candidate_00'/'attempt_00')==task
    assert physical_storage_root(task/'candidate_01'/'attempt_01')==task
