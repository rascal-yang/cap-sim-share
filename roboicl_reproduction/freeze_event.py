"""Freeze one event-wait experiment; preparation only, no simulator imports."""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

from freeze_complete import (ROOT, HERE, BASELINE, IDLE_PROBE, read, write, sha,
    read_creation, checked_resources, validate_runtime, validate_tests,
    validate_train_copy, validate_contract)

COMPLETE = ROOT/'experiments/20261015_1350_roboicl_complete_grasp'
TASK_BUDGET = {'attempts':1, 'sim_seconds':30, 'feedback_calls':40, 'max_rows':5,
               'recovery_attempts':2, 'optimization_attempts':0, 'wait_max_seconds':5}
EVIDENCE_FILES = ('created.json', 'task.json', 'train_memory.json', 'source_evidence.json',
    'runtime.json', 'runtime_check.log', 'preflight_tests.json', 'planner_birth.json',
    'planner_prompt.txt', 'comparison_diff.json', 'original_pipeline_check.json', 'train_copy_check.json',
    'closure_profile_preflight.json', 'bridge_reset_preflight.json', 'initial_controller_history_audit.json',
    'visual_identity_audit.json', 'reused_preflight_evidence.json')


def validate_event_task(task, baseline):
    for field in ('name','backend','instruction','resources','scene','acceptance','references','skills'):
        if task.get(field) != baseline.get(field):
            raise ValueError('Undeclared physical/task change: '+field)
    if task.get('scene') != {'translation':[0,0,0]} or task.get('budget') != TASK_BUDGET:
        raise ValueError('Exactly one origin attempt,30s,40 decisions and wait<=5s required')
    checks = task.get('policy_checks', {})
    if checks.get('sonic_mode') != 1 or checks.get('require_reflection') is not False:
        raise ValueError('Encoder1 and no prior-episode reflection required')
    if checks.get('prior_experiment_outcomes_visible') is not False:
        raise ValueError('Prior experiment outcomes must be excluded from planner input')


def validate_birth(birth, experiment_id):
    if (birth.get('experiment_id') != experiment_id or birth.get('fork_turns') != 'none'
            or birth.get('history_scope') != 'current_episode_only'
            or not isinstance(birth.get('agent'), str) or not birth['agent'].strip()):
        raise ValueError('Fresh isolated planner birth bound to this experiment is required')
    stamp = birth.get('spawned_at', '')
    if not isinstance(stamp,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2}',stamp):
        raise ValueError('Actual planner birth timestamp with seconds and timezone required')
    if datetime.fromisoformat(stamp).tzinfo is None:
        raise ValueError('Planner birth timestamp requires timezone')


def evidence_files(out, tests):
    """Explicit logs only: freeze stdout must never become its own hashed input."""
    paths = [out/name for name in EVIDENCE_FILES]
    for key in ('stdout_path','stderr_path','output_path','red','green'):
        value = tests.get(key)
        if isinstance(value,str):
            path = Path(value); paths.append(path if path.is_absolute() else out/path)
    for path in paths:
        if not path.is_file():
            raise ValueError('Missing freeze evidence: '+str(path))
    return paths


def event_contract(base, out, birth):
    contract = deepcopy(base)
    contract['experiment_id'] = out.name
    contract['purpose'] = {
        'hypothesis':'Same-experiment persistent model context and bounded event-return waits can reduce planning overhead while preserving measured closed-loop control at origin.',
        'baseline_paths':[str(COMPLETE),str(BASELINE),str(IDLE_PROBE)],
        'planned_changes':['One new isolated planner agent at experiment birth, retained for this experiment only; no previous experiment results/reflections enter context',
            'First request contains whitelisted measured TRAIN and initial LIVE; later requests contain current LIVE and all new receipts; same-experiment conversation retains earlier decisions/results',
            'Persistent agent writes response, chained provenance and ready marker directly; expected agent identity is frozen',
            'Event wait up to5s holds last executed absolute targets with zero velocity, executed in.1s units and returning early on measured events',
            'One authorized attempt only; model decision cap40 instead of80; no optimization'],
        'invariants':['Same origin G1/Dex3 scene, assets, initial state/controller history, warmstart, physics, PD, native acceptance and runtime models',
            'Same sparse measured TRAIN observations and image bytes; source acceptance metadata retained only for operator provenance, excluded from planner context',
            'Same encoder1 Cartesian control, standing-idle adapter, measured stability admission, continuous hard collision/lift gates and full confirmed executed-prefix feedback',
            'Physics.005s, SONIC update.02s, lower-planner refresh.5s, action execution units.1s; no source action/token replay or IK/encoder0 fallback'],
        'known_confounders':['One origin scene and one revised attempt, not the prescribed ten-condition comparison or a generalization estimate',
            'Context lifetime, decision budget and wait scheduling change together; do not attribute outcomes or wall-time differences to one isolated cause',
            'Sparse TRAIN source used a different historical controller; measured observations are not action labels',
            'Head+external RGB and privileged contact feedback differ from original RoboICL',
            'Nonterminal wait-event return is checked at the next.1s unit boundary; existing hard safety may stop within the unit',
            'Conversation isolation is not an OS filesystem sandbox; frozen prior evidence paths are operator-only',
            'Native pre-restore reset physics is audited separately; historical reset counts remain unknown',
            'Simulation pauses during model inference; no real-time robot deployment claim']}
    wait = {'maximum_s':5, 'unit_s':.1, 'duration_multiple_s':.1,
        'target_rule':'Require last executed absolute command; preserve explicit wrists/quaternions/torso/fingers/height/facing. Set stage hold,duration.1,velocity[0,0]. No initial synthetic targets or new closure targets.',
        'physics_sample_s':.005, 'nonterminal_event_latency_bound_s':.1,
        'before_step':'Return without stepping if terminal, remaining simulation budget<.1s, or already measured ready.',
        'early_return':'After every actual unit: terminal/interruption/zero-step rejection; any sampled table force>0; hand contact threshold>.1N transitions, including transient dropout or named-pair change or insufficient sample coverage; ready; timeout.',
        'contact_semantics':'Stable ongoing identical hand contact alone does not stop waiting; transient events count even when final force is zero.',
        'receipt_rule':'Retain every actual StepResult/confirmed command feedback; elapsed time is sum of actual execution only; no padding or implicit fallback.',
        'evidence':'attempt_001/wait_events.json; model_calls.json binds each decision to receipt intervals'}
    contract['control']['event_wait'] = wait
    interface = contract['control']['stage_interfaces'][1]
    interface.update(target_source='One fork_turns=none-at-birth isolated agent, same frozen identity for all current-experiment decisions; measured TRAIN/current LIVE only, no prior experiment outcomes or reflections',
        handoff_rule='Normal action chunks reanchor at latest measured wrists; wait macros retain last executed absolute targets. Commit all confirmed executed feedback before the next decision; no replay, source actions, inherited prior-session history or fallback.')
    waiting = deepcopy(interface)
    waiting.update(stage='event wait macro',target_source='Deterministic bounded wait requested by the same current-experiment model; only last executed absolute targets retained',
        action_fields_units_frames='Wait duration in seconds,.1..5 inclusive in.1 multiples; no model-supplied velocity, fingers or pose targets in wait',
        handoff_rule=wait['target_rule']+' '+wait['early_return'])
    contract['control']['stage_interfaces'] = [contract['control']['stage_interfaces'][0],interface,waiting]
    pipeline = contract['pipeline']
    pipeline.update(entrypoint='cap-sim/scripts/humano_python.sh cap-sim/roboicl_reproduction/event_run.py --experiment EXP --attempt 1',
        ordered_steps=['Verify frozen contract/resource hashes, one-attempt budget and planner birth binding',
            'Audit inherited native reset and restore exact original physical/controller state; verify actual physics/providers',
            'First request: whitelisted measured TRAIN plus initial LIVE; exclude source acceptance metadata and all prior experiment outcomes/reflections',
            'Same isolated agent writes each response, chained provenance and ready marker; verify experiment/agent/previous-response hash binding',
            'Validate act chunk or wait request; preserve original bounded zero-step corrections',
            'Execute act through unchanged control bridge, or deterministic wait units with fixed last targets and measured event return',
            'Preserve all confirmed per-command feedback and decision-to-receipt mapping before next model request',
            'Stop on terminal,30 simulation seconds,40 model decisions or first independently accepted success; no second attempt',
            'Export measured states, actual interfaces and independent zero-new-step validation'],
        planner_identity_and_prompt_hash={'identity':'One isolated Codex planner per experiment; fork_turns=none at birth, same agent thereafter; exact serving backend unavailable',
            'agent':birth['agent'],'fork_turns_at_birth':'none','history_scope':'current_episode_only',
            'conversation_inherited':False,'same_experiment_conversation_retained':True,'filesystem_security_boundary':False,
            'birth_evidence':str(out/'planner_birth.json'),
            'per_decision_evidence':'attempt_001/planner_provenance.json plus mailbox response provenance: current experiment, exact agent, request/response SHA, previous_response_sha256 chain, history_scope'},
        planner_context_policy={'prior_experiment_outcomes_visible':False,'prior_experiment_reflections_visible':False,
            'same_experiment_history_retained':True,'train_source_acceptance_metadata_visible':False,
            'train_input':'Measured observations/images and cross-interface interpretation only; omit source_accepted/source_sha256 and historical result metadata',
            'initial_request':'Full allowed TRAIN and measured initial LIVE',
            'later_requests':'Current LIVE plus all newly committed receipts/wait report; prior same-experiment context remains in the persistent agent'},
        external_model_calls=False,isolated_child_model_calls=True,
        external_model_calls_definition='Actual isolated persistent Codex child calls, no separately deployed external API; parent supplies no action content',
        original_actions_visible='No raw source actions/trajectories/tokens/keyframes and no previous experiment outcomes/reflections. Same measured TRAIN observations are allowed; source acceptance metadata is operator-only. Original action prefix replay length0; initial controller checkpoint history remains internal.',
        feedback_and_reflection_rules='Retain same-experiment decisions/results only. Full confirmed feedback for acts/waits precedes next inference. Invalid calls/actions allow max2 consecutive zero-step corrections; successful motion resets correction count. No prior-episode reflection, no second attempt, no planner identity substitution on transport failure.',
        allowed_retry_changes=['Within the single current episode, new model-generated legal actions or bounded wait duration; unchanged physical/control/gate/acceptance interfaces'],
        optimization_enabled=False)
    pipeline['git_commit_and_dirty_patch']['additions'] = 'event_run/freeze_event/event_wait/episode_transport and corresponding prompts/tests; historical frozen files unchanged'
    contract['budget'] = {'attempts_per_case':1,'recovery_attempts':2,'simulation_seconds_per_attempt':30,
        'feedback_calls':40,'maximum_rows_per_decision':5,'maximum_wait_seconds':5,'optimization_attempts':0,
        'recovery_scope':'Consecutive zero-step protocol/action corrections only; never another physics attempt',
        'successful_attempt_rule':'Retain first independently accepted success and stop'}
    contract['metrics'].update(wait_event_definition=wait,
        decision_count_definition='Each model-issued act or wait response consumes one of40 model decisions, including zero-step repairs; wait execution units are not extra model decisions.',
        planning_latency_definition='Measured waiting/inference wall time separated from actual simulation compute, LIVE rendering and total wall time; no assumed speedup or reconstructed durations.',
        failure_truncation_rule='Only executed prefix/actual elapsed; include wait early-return reason and receipt coverage. Single attempt; distinguish transport/operator abort from task failure. Unmeasured fields unknown.')
    return contract


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--experiment',required=True,type=Path)
    out = parser.parse_args().experiment.resolve()
    if not out.is_dir() or (out/'experiment.json').exists() or (out/'contract.sha256').exists():
        raise ValueError('Existing prepared, unfrozen experiment directory required')
    created = read_creation(out)
    base, resources = checked_resources(COMPLETE)
    for directory in (BASELINE,IDLE_PROBE):
        _, historical = checked_resources(directory)
        for path,digest in historical.items():
            if path in resources and resources[path] != digest: raise ValueError('Historical hashes disagree: '+path)
            resources[path] = digest
    original = read(HERE/'docs/baseline_files.json')
    changed = [p for p,h in original.items() if not (ROOT/p).is_file() or sha(ROOT/p) != h]
    if changed: raise ValueError('Original pipeline changed: '+repr(changed))
    task = read(out/'task.json'); validate_event_task(task,read(BASELINE/'task.json'))
    images = validate_train_copy(out,BASELINE)
    runtime = read(out/'runtime.json'); validate_runtime(runtime,base['runtime']['model_hashes'])
    tests = read(out/'preflight_tests.json'); validate_tests(tests)
    birth = read(out/'planner_birth.json'); validate_birth(birth,out.name)
    profile = read(out/'closure_profile_preflight.json')
    if profile.get('passed') is not True or profile.get('physics_steps') != 0: raise ValueError('Actual closure-profile evidence required')
    for key in ('scene','hand_source'):
        item = profile[key]
        if sha(item['path']) != item['sha256']: raise ValueError('Reused closure evidence changed: '+key)
    required = [HERE/name for name in ('event_run.py','freeze_event.py','event_prompt.txt','event_dispatch_protocol.txt',
        'repro/event_wait.py','repro/episode_transport.py','repro/feedback_session.py')]
    if any(not p.is_file() for p in required): raise ValueError('Event implementation incomplete')
    shutil.copyfile(HERE/'event_prompt.txt',out/'planner_prompt.txt')
    write(out/'original_pipeline_check.json',{'checked':len(original),'changed':[],'passed':True,'physics_steps':0})
    write(out/'train_copy_check.json',{'baseline':str(BASELINE),'measured_observations_unchanged':True,
        'image_bytes_unchanged':True,'source_evidence_unchanged':True,'physics_steps':0,
        'copied_images_sha256':{str(p):sha(p) for p in images},
        'planner_metadata_policy':'Frozen provenance remains operator-only; executing planner excludes source acceptance/results and all previous experiment reflections'})
    contract = event_contract(base,out,birth)
    contract.update(created_at=created,creation_timestamp_precision='seconds; actual created.json',record_status='frozen')
    contract['cases'] = [{'id':'origin','task':str(out/'task.json'),'sha256':sha(out/'task.json'),'translation_m':[0,0,0]}]
    contract['runtime']['actual_loaded_session_evidence'] = str(out/'runtime.json')
    contract['physics']['asset_and_initial_state_manifest'] = str(out/'asset_and_code_manifest.json')
    contract['physics']['physics_parameters_manifest'] = str(out/'source_evidence.json')
    identity = contract['pipeline']['planner_identity_and_prompt_hash']
    identity.update(prompt_sha256=sha(out/'planner_prompt.txt'),dispatch_protocol_sha256=sha(HERE/'event_dispatch_protocol.txt'))
    write(out/'comparison_diff.json',{'baseline':str(COMPLETE),'invariants':contract['purpose']['invariants'],
        'declared_changes':contract['purpose']['planned_changes'],'limitations':contract['purpose']['known_confounders'],
        'same_physics':base['physics'],'same_acceptance':task['acceptance'],'historical_resources_changed':[],
        'planner_context_policy':contract['pipeline']['planner_context_policy'],'event_wait':contract['control']['event_wait']})
    files = [*required,*HERE.glob('*.py'),*(HERE/'repro').rglob('*.py'),*(HERE/'tests').rglob('*.py'),
             *images,*evidence_files(out,tests),HERE/'docs/baseline_files.json']
    for directory in (BASELINE,IDLE_PROBE,COMPLETE):
        files.extend(directory/name for name in ('experiment.json','contract.sha256','asset_and_code_manifest.json'))
    for skill in contract['pipeline']['skill_versions_and_hashes']:
        if sha(skill['path']) != skill['sha256']: raise ValueError('Skill version changed')
        files.append(Path(skill['path']))
    resources.update({str(path.resolve()):sha(path) for path in files})
    write(out/'asset_and_code_manifest.json',resources)
    destination = out/'source_snapshots'; destination.mkdir(exist_ok=True); snapshots = {}
    for path,digest in resources.items():
        if Path(path).suffix in ('.py','.json','.sh','.txt','.md','.hpp','.h','.cpp','.log'):
            saved = destination/digest
            if not saved.exists(): shutil.copyfile(path,saved)
            if sha(saved) != digest: raise ValueError('Source snapshot mismatch')
            snapshots[path] = {'sha256':digest,'snapshot':str(saved)}
    write(out/'source_snapshot_index.json',snapshots)
    contract['pipeline']['source_snapshot_and_hashes'] = resources
    contract['preflight'] = {'passed':True,'comparison_diff_path':str(out/'comparison_diff.json'),
        'evidence_paths':[str(p) for p in evidence_files(out,tests)],'offline_tests':tests,'physics_steps':0,
        'runtime_gate':'Actual loaded CPU sessions/models verified; runner checks compiled physics, actual modes, birth-bound identity and hash chain; existing measured stability/hard safety gates retained',
        'reset_note':'No physics in preparation; actual run inherits native pre-restore reset steps and audits them separately'}
    contract['execution'] = {'started_at':'unknown until actual attempt_001/started.json',
        'ended_at':'unknown until actual attempt_001/timing.json','contract_sha256':'contract.sha256 sidecar',
        'audit_path':str(out/'execution_audit.json'),'timing_path':'attempt_001/timing.json','results_path':str(out/'results.json'),
        'wait_events_path':'attempt_001/wait_events.json','decision_receipts_path':'attempt_001/model_calls.json'}
    validate_contract(contract,read(ROOT/'cap-sim/docs/experiment-record.template.json'))
    contract['frozen_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    write(out/'experiment.json',contract); (out/'contract.sha256').write_text(sha(out/'experiment.json')+'\n')
    print('FROZEN',out.name)


if __name__ == '__main__': main()
