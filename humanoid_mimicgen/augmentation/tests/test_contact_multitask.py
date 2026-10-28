import numpy as np
import pytest
from humanoid_mimicgen.augmentation.contact_evidence import reference_differences, acceptance_sources


def test_truncated_approach_is_not_contact_joint_evidence():
    assert reference_differences(np.zeros((20,2)),np.zeros((3,2)),['a','b'],10)==[]


def test_reference_differences_use_only_aligned_contact_interval():
    a=np.array([[100.,100.],[1.,2.],[3.,4.]])
    b=np.zeros((3,2))
    r=reference_differences(a,b,['a','b'],1)
    assert r[0]['replay_minus_mode1_mean_rad']==2.
    assert r[1]['replay_minus_mode1_final_rad']==4.


def test_acceptance_source_is_task_agnostic_and_contains_helpers(tmp_path):
    folder=tmp_path/'source_snapshot';folder.mkdir()
    (folder/'001_g1_wholebody_table_button_press_teleop.py').write_text('THRESHOLD=20\nclass Button:\n def _button_metrics(self): return THRESHOLD\n def check_success(self): return self._button_metrics()\n')
    result=acceptance_sources(tmp_path)
    assert '_button_metrics' in result['code'] and 'THRESHOLD=20' in result['code']
    assert len(result['hashes'])==1


def test_budget_expiration_prevents_new_transport_request():
    from humanoid_mimicgen.augmentation.contact_batch import BudgetProvider
    import time
    class Provider:
        config={'model':'test','timeout_s':120}
        def attempt_providers(self):
            raise AssertionError('must not allocate expired request')
    with pytest.raises(TimeoutError):next(BudgetProvider(Provider(),time.monotonic()-1).attempt_providers())


def test_planning_reuses_audit_without_retransmitting_images_or_source():
    from humanoid_mimicgen.augmentation.contact_review import planning_evidence
    source={'acceptance_source':'large native code','acceptance_source_sha256':{'file':'hash'},'replay':{'inspection':{'review_images':['a.jpg']},'last':{'native_success':True},'joints':['knee']},'diagnostic':{'inspection':{'review_images':['b.jpg']},'last':{'native_success':False}}}
    compact=planning_evidence(source)
    assert 'acceptance_source' not in compact
    assert 'inspection' not in compact['replay']
    assert compact['diagnostic']['last']['native_success'] is False
    assert compact['acceptance_source_sha256']=={'file':'hash'}
    assert 'inspection' in source['replay']


def test_contact_tool_advertises_executable_bounds():
    from humanoid_mimicgen.augmentation.contact_review import CONTROL,JOINT
    bounds={'ramp_s':(.1,3),'max_offset_rad':(.001,.35),'max_rate_rad_s':(.001,.5),'force_stop_N':(0,1500)}
    for key,(lo,hi) in bounds.items():
        assert CONTROL['properties'][key].get('minimum')==lo
        assert CONTROL['properties'][key].get('maximum')==hi
    assert JOINT['properties']['replay_blend']['maximum']==1
    assert CONTROL['properties']['force_range_N']['minItems']==2
    assert CONTROL['properties']['force_range_N']['maxItems']==2


def test_metric_identity_uses_matching_substep_not_final_state():
    from humanoid_mimicgen.augmentation.contact_evidence import metric_state
    trace={'state':np.array([[1.,10.],[1.005,11.],[1.01,12.]]),'next_state':np.array([[1.005,11.],[1.01,12.],[1.015,13.]])}
    assert np.array_equal(metric_state(trace,1.01),[1.01,12.])
    with pytest.raises(ValueError,match='metric timestamp'):
        metric_state(trace,1.02)
