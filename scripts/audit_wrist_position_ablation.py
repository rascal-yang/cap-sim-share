#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from humanoid_mimicgen.augmentation.suite_physics import check_frozen
from humanoid_mimicgen.augmentation.rollback_benchmark import audit_result
from humanoid_mimicgen.workflow.contracts import file_hash,now
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);summary=json.loads((root/'summary.json').read_text());checks=[];timing=[]
for r in summary['results']:
    assert r['status']=='complete' and r['historical_baseline_match']['exact']
    assert all(r['comparison'][k] for k in ('prefix_exact','source_hands_exact_common_frames','nominal_world_targets_exact'))
    for arm in ('baseline','feedback'):
        row=r['arms'][arm];case=Path(row['result']['experiment_path']);record=check_frozen(case);audit=audit_result(row['result'],'apple');logs=json.loads((case/'execution/wrist_position_feedback.json').read_text())
        assert len(logs)==len(np.load(case/'execution/controller.npz')['records'])-346
        assert all(np.linalg.norm(x['bias_world_m'])<=.02+1e-12 for x in logs)
        assert all(row['result']['actual_session_providers'][k]==['CPUExecutionProvider'] for k in ('encoder','decoder','planner'))
        with np.load(case/'execution/hand_commands.npz') as h:first=int(np.flatnonzero(np.max(np.abs(h['right']),axis=1)>.01)[0])
        if arm=='feedback':
            assert json.loads((case/'execution/checkpoint_manifest.json').read_text())['version']==2
            timing.append({'condition':r['condition_id'],'first_closed_hand_command_frame':first,'feedback_first_frame':logs[0]['frame'],'lag_s':(logs[0]['frame']-first)*.02,'first_feedback':logs[0],'after_140ms':logs[7]})
        else:assert all(np.array_equal(x['bias_world_m'],[0,0,0]) for x in logs)
        checks.append({'case':str(case),'body_identity':audit['body_identity'],'frozen':True,'feedback_log_sha256':file_hash(case/'execution/wrist_position_feedback.json')})
    for arm,entry in r['replays'].items():
        assert entry['passed'] and entry['proof']['exact'] and entry['proof']['hands_exact'];case=Path(entry['trial']['result']['experiment_path']);check_frozen(case);audit=audit_result(entry['trial']['result'],'apple');assert entry['trial']['diagnostics']['feedback_ticks']==0;checks.append({'case':str(case),'body_identity':audit['body_identity'],'saved_token_replay':True,'frozen':True})
out={'experiment_id':root.name,'audited_at':now(),'passed':True,'conditions':len(summary['results']),'physical_cases':len(checks),'checks':checks,'timing_diagnostics':timing};(root/'FINAL_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ('checks','timing_diagnostics')}))
