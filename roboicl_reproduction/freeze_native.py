"""Freeze a fully evidenced origin experiment, never launches physics."""
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from native_run import PROMPT

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();out=a.experiment.resolve()
 if (out/'contract.sha256').exists():raise FileExistsError('Already frozen')
 baseline=ROOT/'experiments/20261014_1957_roboicl_lateral'
 contract=json.loads((baseline/'experiment.json').read_text());resources=json.loads((baseline/'asset_and_code_manifest.json').read_text())
 changed=[f for f,h in resources.items() if not Path(f).is_file() or sha(f)!=h]
 if changed:raise ValueError('Historical resource changed: '+str(changed))
 runtime=json.loads((out/'runtime.json').read_text())
 if not runtime['passed'] or not runtime['sessions_loaded'] or any(p!=['CPUExecutionProvider'] for p in runtime['actual_session_providers'].values()):raise ValueError('Missing actual session evidence')
 (out/'planner_prompt.txt').write_text(PROMPT)
 files=[*Path(__file__).resolve().parent.glob('*.py'),*Path(__file__).resolve().parent.rglob('repro/*.py'),
        *Path(__file__).resolve().parent.glob('vendor/roboicl/**/*.py'),
        *[out/f for f in ('task.json','train_memory.json','source_evidence.json','runtime.json','planner_prompt.txt')],
        *list((out/'train_images').glob('*.png')),
        ROOT/'cap-sim/scripts/humano_python.sh',ROOT/'cap-sim/scripts/check_humano_runtime.py',ROOT/'cap-sim/roboicl_adapter/scene.py',
        *list((ROOT/'cap-sim/src').rglob('*.py'))]
 resources.update({str(f.resolve()):sha(f) for f in files})
 write(out/'asset_and_code_manifest.json',resources)
 snapshots={}
 for file,h in resources.items():
  if Path(file).suffix in ('.py','.json','.sh','.txt'):
   dest=out/'source_snapshots'/h;dest.parent.mkdir(exist_ok=True);shutil.copyfile(file,dest);snapshots[file]={'sha256':h,'snapshot':str(dest)}
 write(out/'source_snapshot_index.json',snapshots)
 contract['experiment_id']=out.name;contract['created_at']=datetime.strptime(out.name[:13],'%Y%m%d_%H%M').replace(tzinfo=ZoneInfo('Asia/Shanghai')).isoformat(timespec='minutes');contract['creation_timestamp_precision']='minute; seconds not measured';contract['frozen_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds');contract['record_status']='frozen'
 contract['purpose']={'hypothesis':'TRAIN observation memory plus fully committed LIVE receipts can generate and correct new mode1 actions at original apple position.',
  'baseline_paths':[str(baseline),str(ROOT/'experiments/20261013_apple_straight_sweep/reference_recovery/success/attempt_001')],
  'planned_changes':['Zero scene translation (user requested origin first)','New independent Session loop, .1s commands with 1..5 row chunks','Original trajectory-derived TRAIN sparse observations, NOT same-interface command demonstration','Fixed anchors+latest LIVE with zero-step repairs, partial-prefix receipts and tracking interruption','Persistent collision timer across action boundaries','Up to three sequential attempts; reflection before next reset, explicitly marked past-episode extension'],
  'invariants':['Original G1 Dex3/scene/assets/initial state and warmstart','Physics .005s, SONIC update .02s and lowerbody .5s refresh unchanged','Original PD, actual mode1, CPU providers and model hashes','Original contact/fall/lift gates and native acceptance','No original action/token prefix execution'],
  'known_confounders':['Not a matched trajectory/controller comparison; user requested generated-action feasibility at origin','TRAIN source controller differs from LIVE; learning from observations only','Head+external RGB and privileged contact feedback, not original three on-robot cameras','Feedback budget and action segmentation differ from previous pilot','Model exact serving identifier unavailable; fresh isolated session agent represents API']}
 contract['cases']=[{'id':'origin','task':str(out/'task.json'),'sha256':sha(out/'task.json'),'translation_m':[0,0,0]}]
 contract['runtime']['actual_loaded_session_evidence']=str(out/'runtime.json');contract['runtime']['model_hashes']=runtime['model_sha256']
 contract['physics']['asset_and_initial_state_manifest']=str(out/'asset_and_code_manifest.json');contract['physics']['physics_parameters_manifest']=str(out/'source_evidence.json')
 contract['physics']['safety_gates']['collision_timer_scope']='continuous across .1s commands, checked every real .005s physical step; no threshold change'
 contract['control']['stage_interfaces'][0].update(target_source='Fresh isolated API calls from original TRAIN observations and measured LIVE',action_fields_units_frames='Model: world xyz+rotvec increments, Dex3 7 absolute joint rad; bridge: absolute world wrists/unit wxyz. Fixed .1s commands,1..5 rows.',handoff_rule='New action from measured latest state; cumulative requested targets only within current chunk. No replay or mode fallback.')
 contract['control']['actual_mode_log']='attempt_NNN/run/controller.json plus stage_results.json'
 skill=Path('/home/kinetra-4090-01/.codex/skills/grasp-prior-refinement/SKILL.md')
 contract['pipeline'].update(entrypoint='cap-sim/scripts/humano_python.sh cap-sim/roboicl_reproduction/native_run.py --experiment EXP --attempt N',
  ordered_steps=['verify frozen source/resource hashes','reset original initial state and warmstart; compare compiled physics; verify actual providers','build fixed sparse TRAIN and explicit previous-episode reflection if any','capture whitelisted LIVE state + RGB','isolated fresh-context planner response','validate entire chunk','execute .1s command in unchanged Backend via measured bridge','commit actual command/time/residual/contact feedback; interrupt suffix or end on hard safety event','new inference from committed LIVE','export and independent native verifier','reflect before next attempt'],
  source_snapshot_and_hashes=resources,git_commit_and_dirty_patch={'commit':'3c9ac572b0511b39b0019d07a491958eceba3d7a','original_pipeline_modified':False,'additions':'roboicl_reproduction'},
  planner_identity_and_prompt_hash={'identity':'Codex fresh isolated subagent per request, fork_turns=none; exact model backend unavailable','prompt_sha256':sha(out/'planner_prompt.txt'),'conversation_inherited':False,'filesystem_security_boundary':False},
  skill_versions_and_hashes=[{'name':'grasp-prior-refinement','path':str(skill),'sha256':sha(skill),'scope':'operator records, not executing planner'}],
  external_model_calls=False,original_actions_visible='NO raw command sequence/tokens/keyframes to executing planner; sparse measured original observations and images visible. Preparation reads source metadata only.',
  feedback_and_reflection_rules='Every completed/interrupted chunk committed to LIVE before next inference. Invalid actions zero-step retry max2. Hard collision/fall terminal. Explicit past-episode reflection after reset, no stale state continuity.',
  allowed_retry_changes=['New model-generated wrist/hand/velocity/height/torso/yaw targets within same frozen profile and original initial state'],optimization_enabled=False)
 contract['budget']={'attempts_per_case':3,'recovery_attempts':2,'simulation_seconds_per_attempt':30,'feedback_calls':80,'optimization_attempts':0}
 contract['metrics']['sampling_and_alignment']='200Hz saved states and contacts,50Hz controller records,10Hz command receipts; interruptions include true terminal tick'
 contract['metrics']['tracking_error_definition']='Model-facing guard uses measured command endpoint residual (position>.05m or orientation>.5rad twice per chunk); not per-tick interpolation error. Per-tick interpolation RMSE unknown.'
 diff={'baseline':str(baseline),'unchanged':contract['purpose']['invariants'],'declared_changes':contract['purpose']['planned_changes'],'limitations':contract['purpose']['known_confounders'],'source_resources_checked':len(resources),'historical_resources_changed':changed}
 write(out/'comparison_diff.json',diff)
 contract['preflight']={'comparison_diff_path':str(out/'comparison_diff.json'),'passed':True,'evidence_paths':[str(out/f) for f in ('runtime.json','source_evidence.json','asset_and_code_manifest.json','preflight_tests.json')],
    'runtime_gate':'Compare actual compiled physics before first action; actual provider/mode checks required','offline_tests':'See preflight_tests.json'}
 if not (out/'preflight_tests.json').is_file():raise ValueError('Test evidence required')
 contract['execution']={'started_at':'Recorded per attempt/timing.json after launch','ended_at':'Recorded per attempt/timing.json after completion','contract_sha256':'contract.sha256 sidecar','audit_path':str(out/'execution_audit.json'),'timing_path':'attempt_NNN/timing.json','results_path':str(out/'results.json')}
 write(out/'experiment.json',contract);(out/'contract.sha256').write_text(sha(out/'experiment.json')+'\n');print('FROZEN',out.name)
if __name__=='__main__':main()
