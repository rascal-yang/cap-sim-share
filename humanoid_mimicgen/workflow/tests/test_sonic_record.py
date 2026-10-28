import numpy as np
import pytest
from humanoid_mimicgen.workflow.contracts import WorkflowError
from humanoid_mimicgen.workflow.sonic_record import action_arrays, save_action_bundle


def sample():
    controllers=[dict(time=0.,mode=1,token=[.1,.2],target=[0.]*29),
                 dict(time=.02,mode=1,token=[.3,.4],target=[1.]*29)]
    commands=[dict(time_s=i*.005,stage='navigation' if i<4 else 'hold',source_index=i,
                   body_target_rad=[float(i//4)]*29,fingers_rad=[i*.01]*14,ctrl_Nm=[i*.1]*43) for i in range(5)]
    return controllers,commands


def test_complete_token_mapping_preserves_subtick_fingers_and_truncated_last_tick(tmp_path):
    controller,commands=sample();a=action_arrays(controller,commands,.005,.02)
    np.testing.assert_array_equal(a['token_index'],[0,0,0,0,1])
    assert a['tokens'].shape==(2,2)
    assert a['fingers_rad'][1,0]==.01 and a['stage'].tolist()==['navigation']*4+['hold']
    result=save_action_bundle(tmp_path,controller,commands,.005,.02,expected_physics_steps=5)
    assert result['physics_steps']==5 and result['token_count']==2
    assert result['coverage_complete'] and not result['fingers_encoded_in_token']
    with pytest.raises(FileExistsError):save_action_bundle(tmp_path,controller,commands,.005,.02,expected_physics_steps=5)


@pytest.mark.parametrize('mutation',['missing','gap','nonfinite','target','time','mode'])
def test_invalid_token_coverage_is_rejected(mutation):
    controller,commands=sample()
    if mutation=='missing':controller.pop()
    if mutation=='gap':commands[2]['time_s']+=.001
    if mutation=='nonfinite':controller[1]['token'][0]=float('nan')
    if mutation=='target':commands[3]['body_target_rad'][0]=.1
    if mutation=='time':controller[1]['time']+=.001
    if mutation=='mode':controller[0]['mode']=0
    with pytest.raises(WorkflowError):action_arrays(controller,commands,.005,.02)


def test_unexecuted_last_policy_attempt_is_identified_not_exported():
    controller,commands=sample();controller.append(dict(time=.04,mode=1,token=[.5,.6],target=[2.]*29))
    a=action_arrays(controller,commands,.005,.02)
    assert len(a['tokens'])==2 and a['unused_controller_indices'].tolist()==[2]


def test_forced_policy_tick_between_regular_ticks_has_complete_coverage():
    controller,commands=sample()
    controller.insert(1,dict(time=.015,mode=1,token=[.15,.25],target=[.5]*29))
    commands[3]['body_target_rad']=[.5]*29
    a=action_arrays(controller,commands,.005,.02)
    assert a['token_index'].tolist()==[0,0,0,1,2]


def test_missing_executed_tail_step_cannot_claim_complete_coverage(tmp_path):
    controller,commands=sample()
    with pytest.raises(WorkflowError,match='executed physics step count'):
        save_action_bundle(tmp_path,controller,commands,.005,.02,expected_physics_steps=6)
