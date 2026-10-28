import pytest


def test_precision_defaults_and_contact_summary_keep_hip_and_native_separate():
    from humanoid_mimicgen.augmentation.navigation_api import navigation_profile, summarize_contacts
    nav = navigation_profile(1.5, {'case': '/parent', 'name': 'parked'})
    assert nav['checkpoint']['resume']['name'] == 'parked'
    assert nav['parking']['leg_mode'] == 'source_replay'
    assert nav['retry']['max_retries'] == 0
    pairs = summarize_contacts([{'time_s': 0, 'contacts': {'events': [
        {'bodies': ['table', 'right_hip_link'], 'force_N': 20},
        {'bodies': ['table', 'right_hand_link'], 'force_N': 10}]}}])
    assert pairs['table / right_hip_link']['peak_N'] == 20
    assert pairs['table / right_hand_link']['duration_s'] == .005


def test_contact_duration_counts_pair_once_per_physics_step_and_clean_requires_completion():
    from humanoid_mimicgen.augmentation.navigation_api import summarize_contacts, clean_success
    event = {'bodies': ['table', 'right_hand'], 'force_N': 8}
    summary = summarize_contacts([{'time_s': 0, 'contacts': {'events': [event, event]}}])
    assert summary['table / right_hand']['duration_s'] == .005
    result = dict(native_success=True, retained_grasp=True, max_obstacle_N=0, completed=False, failure='timeout')
    assert not clean_success(result)
    assert clean_success(result | {'completed': True, 'failure': None})
