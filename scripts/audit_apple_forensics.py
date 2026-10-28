"""Read-only verification of diagnostic evidence and controller-target identity."""
import json
from pathlib import Path
import numpy as np
from humanoid_mimicgen.workflow.contracts import file_hash,now
from humanoid_mimicgen.workflow.service import check_contract
from humanoid_mimicgen.augmentation.suite_physics import check_frozen
from humanoid_mimicgen.augmentation.agent import save
BASE=Path(__file__).resolve().parents[2]/'experiments/2026-10/26';rows=[]
for rn in ['20261026_0953_apple_navigation_forensics','20261026_0958_apple_navigation_transfer']:
 for p in sorted((BASE/rn).glob('*/summary.json')):
  s=json.loads(p.read_text())
  if 'execution' not in s:continue
  case=Path(s['prepared']['experiment_path']);check_contract(case/'experiment.json');out=case/'execution';a=json.loads((out/'execution_audit.json').read_text());commands=json.loads((out/'commands.json').read_text());controls=json.loads((out/'controller.json').read_text());j=0;identity=True
  for c in commands:
   while j+1<len(controls) and controls[j+1]['time']<=c['time_s']+1e-10:j+=1
   identity &= np.array_equal(c['body_target_rad'],controls[j]['target'])
  assert identity and a['actual_encoder_modes']==[1] and not a['mode0_calls'] and not a['original_token_prefix_count'] and not a['midrun_state_resets'] and a['api_calls_during_execution']==0
  assert s['execution']['verification_errors']==[]
  e=json.loads((out/'events.json').read_text());tracking=json.loads((out/'tracking.json').read_text());g=[t for t in tracking if t['retry_index']==0 and 1600<=t['source_index']<1880];stop=next(x for x in e if x['event']=='navigation_settled')
  row={'case':str(case),'label':p.parent.name,'body_decoder_target_exact':bool(identity),'contract_checked':True,'native':s['execution']['native']['success'],'held_success':s['execution']['physical_task_completed'],'sim_s':a['sim_seconds'],'simulation_wall_s':a['simulation_wall_s'],'online_planning_wall_s':a['online_planning_wall_s'],'retries':a['recovery_count'],'failure':a['failure'],'settled_error_cm':stop['error_m']*100,'grasp_rmse_cm':float(np.sqrt(np.mean([t['right']['position_error_m']**2 for t in g]))*100),'grasp_orientation_error_mean_deg':float(np.mean([t['right']['orientation_error_rad'] for t in g])*180/np.pi),'max_obstacle_N':s['execution']['max_obstacle_force_N']}
  rows.append(row)
for rn in ['20261026_0956_apple_early_handoff','20261026_1005_apple_early_handoff']:
 for s in json.loads((BASE/rn/'summary.json').read_text())['results']:
  case=Path(s['trial']['result']['experiment_path']);check_frozen(case);result=s['trial']['result'];controls=np.load(case/'execution/controller.npz')['records'];trace=np.load(case/'execution/trajectory.npz');identity=np.array_equal(trace['commands'][:,1:30],np.repeat(controls[:,-29:],4,axis=0));assert identity and s['prefix_exact'] and s['hands_exact'];assert result['original_token_steps']==300 and result['encoder_modes']==[1]
  row={'case':str(case),'label':rn+'/'+s['condition_id'],'body_decoder_target_exact':identity,'contract_checked':True,'native':result['success'],'max_lift_cm':s['trial']['diagnostics']['max_lift_m']*100,'original_prefix_exact':s['prefix_exact'],'hands_exact':s['hands_exact'],'simulation_wall_s':result['simulation_wall_s'],'sim_s':result['simulation_s']}
  if rn.startswith('20261026_1005'):
   enc=controls[300:,-1791:-29];anchor=np.array(result['fixed_leg_anchor_rad']);row['fixed_legs_input_exact']=bool(np.allclose(enc[:,661:781],np.tile(anchor,10),rtol=0,atol=1e-7) and np.all(enc[:,781:901]==0));assert row['fixed_legs_input_exact']
  rows.append(row)
restore=BASE/'20261026_0953_apple_navigation_forensics/restore/summary.json';s=json.loads(restore.read_text());assert s['historical_state_match']['exact'];summary={'audited_at':now(),'passed':True,'physical_runs':len(rows),'new_api_calls':0,'historical_restore_exact':True,'rows':rows};save(restore.parents[1]/'FINAL_AUDIT.json',summary);print(json.dumps(summary,ensure_ascii=False))
