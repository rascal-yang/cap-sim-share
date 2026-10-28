#!/usr/bin/env python3
"""Verify frozen evidence, body targets, compiled provenance and all paired rollouts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from humanoid_mimicgen.augmentation.leg_fast_retry import exact_rollout
from humanoid_mimicgen.augmentation.quality_suite import verify_origin
from humanoid_mimicgen.augmentation.rollback_plan import assemble

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);data=json.loads((root/'combined_analysis.json').read_text());cache={}
def digest(path):
 path=Path(path);st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  cache[key]=h.hexdigest()
 return cache[key]

cases={};pairproofs=[];historical=[]
for task in data['results']:
 assert task['status']=='complete' and task['verification']['passed']
 candidates=[task['seed']]+[e['trial'] for e in task['equivalence']]+[task['verification'][k] for k in ('full','token_replay')]
 for co in task['conditions']:
  left={r['candidate_index']:r for r in co['arms']['restart']['rows'] if not r.get('shared_seed')};right={r['candidate_index']:r for r in co['arms']['rollback']['rows'] if not r.get('shared_seed')}
  for k in left.keys()&right.keys():
   proof=exact_rollout(left[k]['result']['experiment_path'],right[k]['result']['experiment_path']);assert proof['exact'];pairproofs.append({'profile':task['profile'],'repeat':co['repeat'],'candidate':k,'proof':proof})
  for arm in co['arms'].values():candidates+=arm['rows']
 for row in candidates:cases[row['result']['experiment_path']]=(task['profile'],row)
 seed=task['seed'];origin=json.loads(Path(seed['decision_path']).read_text())['origin'];spec=json.loads(Path(origin['spec_path']).read_text());proof=exact_rollout(spec['source_case'],seed['result']['experiment_path']);assert proof['exact'];historical.append({'profile':task['profile'],'seed_vs_historical':proof})
checks=[]
for path,(profile,row) in cases.items():
 c=Path(path);contract=json.loads((c/'experiment.json').read_text());assert digest(c/'experiment.json')==(c/'experiment.sha256').read_text();assert contract['record_status']=='frozen_ready' and contract['preflight']['passed']
 for name,h in contract['preflight']['evidence_sha256'].items():
  q=Path(name)
  if q.exists() and digest(q)==h:continue
  assert any(digest(s)==h for s in (c/'source_snapshot').glob('*_'+q.name)),name
 r=json.loads((c/'result.json').read_text());assert all(v==['CPUExecutionProvider'] for v in r['actual_session_providers'].values());rt=json.loads((c/'runtime.json').read_text());assert rt['sessions_loaded']
 with np.load(c/'execution/trajectory.npz') as t,np.load(c/'execution/controller.npz') as ctrl,np.load(c/'execution/hand_commands.npz') as hands:
  raw=ctrl['records'];assert np.array_equal(t['commands'][:,1:30],np.repeat(raw[:,-29:],4,axis=0)[:len(t['commands'])]);assert all(hands[k].shape==(len(raw),7) for k in ('left','right'))
  modes=set(raw[:,1].astype(int));assert modes<=({-2} if r['mode']=='saved_token_replay' else {-1,1})
 rec=json.loads(Path(r['decision_path']).read_text());o=rec['origin'];batch=verify_origin(o['api_batch_path']);assert digest(o['api_batch_path'])==o['api_batch_sha256'] and digest(o['spec_path'])==o['spec_sha256'];spec=json.loads(Path(o['spec_path']).read_text());candidate=batch['decision']['candidates'][o['candidate_index']];assert assemble(spec['base_plan'],candidate['parameters'],profile,o['local_resume'],candidate['rationale'])==rec['decision'];assert any(digest(p)==o['compiler_sha256'] for p in (c/'source_snapshot').glob('*_rollback_plan.py'))
 if r.get('branch_start_frame',0):
  meta=json.loads((c/'execution/checkpoints.json').read_text())[r['branch_start_frame']];ext=meta['controller_extensions'];assert ext['version']==1 and set(ext['controllers'])==({'leg'} if profile=='chair' else {'wrist','fingers'} if profile=='button' else set())
 checks.append({'case':path,'profile':profile,'frozen_evidence':True,'compiled_provenance':True,'body_decoder_identity':True,'hands_complete':True,'actual_modes':sorted(map(int,modes)),'native_success':r['success'],'acceptance_success':row['success']})
result={'passed':True,'case_count':len(checks),'paired_exact_count':len(pairproofs),'cases':checks,'paired_proofs':pairproofs,'historical_seed_proofs':historical,'snapshot_fallback':'Historical versions verified against frozen source copies when current code has evolved'};(root/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('cases','paired_proofs','historical_seed_proofs')}))
