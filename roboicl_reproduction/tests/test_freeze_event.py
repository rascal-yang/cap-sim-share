"""Event contract checks use files/data only, never a physics backend."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'experiments/20261015_1350_roboicl_complete_grasp'


def module():
    path = ROOT/'cap-sim/roboicl_reproduction/freeze_event.py'
    assert path.is_file(), 'event freezer not implemented'
    spec = importlib.util.spec_from_file_location('event_freezer', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def task_fixture():
    task = json.loads((BASE/'task.json').read_text())
    task['budget'] = dict(attempts=1, sim_seconds=30, feedback_calls=40, max_rows=5,
                          recovery_attempts=2, optimization_attempts=0, wait_max_seconds=5)
    task['policy_checks'] = dict(sonic_mode=1, require_reflection=False,
                                 prior_experiment_outcomes_visible=False)
    return task


def test_event_task_rejects_extra_attempts_decisions_wait_and_prior_results():
    m = module(); baseline = json.loads((BASE/'task.json').read_text()); task = task_fixture()
    m.validate_event_task(task, baseline)
    for key, value in [('attempts',2), ('feedback_calls',41), ('sim_seconds',31), ('wait_max_seconds',6)]:
        changed = deepcopy(task); changed['budget'][key] = value
        with pytest.raises(ValueError): m.validate_event_task(changed, baseline)
    for key in ['require_reflection','prior_experiment_outcomes_visible']:
        changed = deepcopy(task); changed['policy_checks'][key] = True
        with pytest.raises(ValueError): m.validate_event_task(changed, baseline)
    changed = deepcopy(task); changed['acceptance']['hold_s'] = .5
    with pytest.raises(ValueError): m.validate_event_task(changed, baseline)


def test_planner_birth_requires_current_experiment_fresh_context_and_timestamp():
    m = module()
    good = dict(experiment_id='test',agent='/root/event_planner_1537',fork_turns='none',
                history_scope='current_episode_only',spawned_at='2026-10-15T15:40:01+08:00')
    m.validate_birth(good,'test')
    for key,value in [('experiment_id','old'),('agent',''),('fork_turns','all'),
                      ('history_scope','all_experiments'),('spawned_at','2026-10-15T15:40:01')]:
        changed = dict(good, **{key:value})
        with pytest.raises(ValueError): m.validate_birth(changed,'test')


def test_contract_replaces_old_context_budget_and_preserves_physics():
    m = module(); base = json.loads((BASE/'experiment.json').read_text()); original = deepcopy(base)
    birth = dict(agent='/root/event_planner_1537',fork_turns='none',history_scope='current_episode_only')
    result = m.event_contract(base,Path('/tmp/test_event_contract_fixture'),birth)
    assert base == original
    assert result['physics'] == base['physics']
    assert result['budget']['attempts_per_case'] == 1
    assert result['budget']['feedback_calls'] == 40
    policy = result['pipeline']['planner_context_policy']
    assert policy['prior_experiment_outcomes_visible'] is False
    assert policy['prior_experiment_reflections_visible'] is False
    assert policy['same_experiment_history_retained'] is True
    assert policy['train_source_acceptance_metadata_visible'] is False
    assert result['pipeline']['planner_identity_and_prompt_hash']['agent'] == birth['agent']
    assert result['control']['event_wait']['maximum_s'] == 5
    assert result['control']['event_wait']['nonterminal_event_latency_bound_s'] == .1


def test_freeze_stdout_is_never_collected_as_an_input(tmp_path):
    m = module()
    for name in m.EVIDENCE_FILES: (tmp_path/name).write_text('{}')
    (tmp_path/'freeze.log').write_text('mutable stdout')
    (tmp_path/'preflight_tests.log').write_text('passed')
    paths = m.evidence_files(tmp_path,{'output_path':'preflight_tests.log'})
    assert tmp_path/'preflight_tests.log' in paths
    assert tmp_path/'freeze.log' not in paths
    with pytest.raises(ValueError): m.evidence_files(tmp_path,{'output_path':'missing.log'})
