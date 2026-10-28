#!/usr/bin/env python3
import argparse,json,statistics,time
from pathlib import Path
from generalization_records import conditions
from humanoid_mimicgen.augmentation.leg_fast_retry import exact_rollout
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);pairs=[];tick=time.monotonic()
for s in conditions(root):
    if not all(k in s['arms'] for k in ('restart','rollback')):continue
    arms={k:{r['candidate_index']:r for r in v['rows'] if not r.get('shared_seed')} for k,v in s['arms'].items()}
    for i in sorted(arms['restart'].keys()&arms['rollback'].keys()):
        x=arms['restart'][i];y=arms['rollback'][i];proof=exact_rollout(x['result']['experiment_path'],y['result']['experiment_path'])
        pairs.append({'condition':s['condition_id'],'task':s['task_label'],'candidate':i,'proof':proof,'restart_wall_s':x['attempt_wall_s'],'rollback_wall_s':y['attempt_wall_s'],'reduction_fraction':1-y['attempt_wall_s']/x['attempt_wall_s']})
summary={}
for label in ('plate','chair','button','apple'):
    xs=[p for p in pairs if p['task']==label]
    summary[label]={'matched_candidates':len(xs),'all_exact':all(p['proof']['exact'] for p in xs),'mean_attempt_wall_reduction_fraction':statistics.mean(p['reduction_fraction'] for p in xs) if xs else None}
out={'passed':bool(pairs) and all(p['proof']['exact'] for p in pairs),'pair_count':len(pairs),'wall_s':time.monotonic()-tick,'summary':summary,'pairs':pairs};(root/'MATCHED_PAIRS.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='pairs'}))
