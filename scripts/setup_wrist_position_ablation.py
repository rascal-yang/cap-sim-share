#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from generalization_records import conditions
p=argparse.ArgumentParser();p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--experiment-root',type=Path,required=True);a=p.parse_args();baseline=a.baseline_root.resolve();stamp=datetime.now(ZoneInfo('Asia/Shanghai'));root=a.experiment_root.resolve()/stamp.strftime('%Y-%m/%d/%Y%m%d_%H%M_apple_wrist_position_ablation');stem=root;i=1
while root.exists():root=stem.with_name(stem.name+f'_{i:02d}');i+=1
root.mkdir(parents=True)
items=[]
for row in conditions(baseline):
 if row['task_label']!='apple':continue
 if row['status']!='complete':raise ValueError('Historical condition incomplete')
 seed=row['seed'];decision=json.loads(Path(seed['decision_path']).read_text());o=decision['origin'];spec=json.loads(Path(o['spec_path']).read_text())
 assert seed['candidate_index']==0 and spec['base_plan']['wrist_delta_m']==[0,0,.05] and spec['base_plan']['tail_s']==1.5
 items.append({'condition_id':row['condition_id'],'shift':row['shift'],'spec_path':o['spec_path'],'batch_path':o['api_batch_path'],'historical_seed_case':seed['result']['experiment_path'],'historical_summary':row['_summary_path']})
assert len(items)==10
repo=Path(__file__).resolve().parents[1]
c={'output_root':str(root),'config_path':str(root/'config.json'),'created_at':stamp.isoformat(),'conditions':items,'baseline_root':str(baseline),'protocol_path':str(repo/'docs/augmentation/wrist-position-ablation.md'),'historical_feedback_contract':str(a.experiment_root.resolve()/'2026-10/19/20261019_1820_apple_nav_wrist_feedback_y_minus15/experiment.json'),'ablation':{'arms':['baseline','feedback'],'fresh_attempts_per_arm':1,'retry':False,'api_calls':0,'gain_per_s':.5,'cap_m':.02,'dt_s':.02,'feedback_target':'unchanged baseline reference including fixed API offset','feedback_activation':'first mode1 frame to end','independent_saved_token_and_hands_replay':'every successful arm','rendering':'after numeric trials; separate timing'},'benchmark':{'design':'fresh_start_feedback_ablation','order_repeats':1,'wall_budget_s':None,'simulation_horizon':'unchanged source394frames plus75tail frames','success':'native apple lift>=9cm plus unchanged safety','no_retry_or_deadline_censoring':True}}
(root/'config.json').write_text(json.dumps(c,indent=2));print(root)
