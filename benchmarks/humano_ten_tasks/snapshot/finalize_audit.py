"""Final evidence consistency checks; no simulation or result rewriting."""
import json,hashlib,time
from pathlib import Path
r=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
ps=sorted((r/'runs').glob('*/*/summary.json'));assert len(ps)==20,len(ps)
freeze=read(r/'no_reference_frozen.json')
for f,h in freeze['sha256'].items():assert hashlib.sha256((r/f).read_bytes()).hexdigest()==h,f
for p in ps:
 s=read(p);assert s['attempts']<=10 and s['optimization_attempts']<=1
 results=sorted((p.parent/'attempts').glob('*/result.json'));assert len(results)==s['attempts']
 for f in results:
  a=read(f);assert a['acceptance']['sim_seconds']<=30+1e-6
  if not a.get('interrupted'):
   assert read(f.parent/'independent_validation.json')['success']==a['acceptance']['success']
  if a['acceptance']['success']:
   assert (p.parent/'success'/f'attempt_{a["attempt"]:03d}'/'trajectory.json').exists()
 v=r/'videos'/p.parent.parent.name/p.parent.name;verification=read(v/'verification.json');assert verification['physics_steps']==0
 for name in ['ego','external']:assert (v/(name+'.mp4')).stat().st_size>0
 selected=s['accepted_attempt'] or s['attempts'];a=read(p.parent/'attempts'/f'{selected:03d}'/'result.json')
 assert abs(verification['source_times'][-1]-a['acceptance']['sim_seconds'])<1e-6
for case in (r/'runs/no_reference').iterdir():
 a=read(case/'native_resources.json');b=read(r/'runs/reference_augmented'/case.name/'native_resources.json')
 assert {k:v for k,v in a.items() if k!='model'}=={k:v for k,v in b.items() if k!='model'},case.name
sha_sources={hashlib.sha256(p.read_bytes()).hexdigest():str(p.relative_to(r)) for p in [r/'reference_policy.py',*sorted((r/'policy_versions').glob('*.py'))]}
versions={}
for p in (r/'mailbox/reference_augmented').glob('*/*_response.json'):
 d=read(p);h=d['planner_provenance']['policy_sha256'];assert h in sha_sources,h;versions[h]=sha_sources[h]
report={'verified_unix_s':time.time(),'completed_cases':20,'video_pairs':20,'no_reference_frozen_hashes_unchanged':True,'controller_physics_acceptance_resources_equal_between_groups_except_shifted_compiled_model':True,'all_full_attempt_independent_checks_match':True,'successful_baselines_preserved':True,'video_end_times_match_measured_states':True,'reference_policy_sources':versions}
(r/'final_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
