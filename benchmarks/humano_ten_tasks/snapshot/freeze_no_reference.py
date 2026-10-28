import hashlib,json,time
from pathlib import Path
r=Path(__file__).resolve().parent;out=r/'no_reference_frozen.json'
assert not out.exists(),'Existing freeze must remain immutable'
ps=sorted((r/'runs/no_reference').glob('*/summary.json'))
assert len(ps)==10,f'Only {len(ps)} of 10 cases complete'
files={}
for p in ps:
 s=json.loads(p.read_text());assert s['status'] in ('success','budget_exhausted','completed'),s
 for f in [p,*sorted((p.parent/'attempts').glob('*/result.json'))]:files[str(f.relative_to(r))]=hashlib.sha256(f.read_bytes()).hexdigest()
out.write_text(json.dumps({'frozen_unix_s':time.time(),'expert_rollouts_read':False,'cases':10,'sha256':files},indent=2))
print(out)
