#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from generalization_records import roots as study_roots, conditions as study_conditions
import numpy as np
from humanoid_mimicgen.augmentation.suite_physics import check_frozen
from humanoid_mimicgen.augmentation.rollback_benchmark import audit_result
from humanoid_mimicgen.workflow.contracts import file_hash,now
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);checks=[]
for path in sorted(p for r in study_roots(root) for p in r.glob('**/experiments/*/result.json')):
    r=json.loads(path.read_text());case=path.parent;errors=[]
    try:
        check_frozen(case)
        runtime=json.loads((case/'runtime.json').read_text())
        if any(v!=['CPUExecutionProvider'] for v in runtime['actual_session_providers'].values()):raise ValueError('Unexpected session provider')
        if set(r['actual_session_providers'])!={'encoder','decoder','planner'} or any(v!=['CPUExecutionProvider'] for v in r['actual_session_providers'].values()):raise ValueError('Missing loaded session evidence')
        dp=json.loads(Path(r['decision_path']).read_text());spec=json.loads(Path(dp['origin']['spec_path']).read_text())
        audit=audit_result(r,spec['profile'])
        exp=json.loads((case/'experiment.json').read_text())
        if not exp['purpose'].get('scene_only_invariant_check'):raise ValueError('Missing scene invariant check')
        checks.append({'case':str(case),'passed':True,'native_success':r['success'],'acceptance_success':audit['success'],'scene_check':exp['purpose']['scene_only_invariant_check']})
    except Exception as e:checks.append({'case':str(case),'passed':False,'error':str(e)})
conditions=study_conditions(root)
out={'audited_at':now(),'passed':bool(checks) and all(c['passed'] for c in checks),'physical_cases':len(checks),'conditions':len(conditions),'condition_errors':[{k:r.get(k) for k in ('condition_id','status','error')} for r in conditions if r['status']!='complete'],'checks':checks}
(root/'FINAL_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print(json.dumps({k:v for k,v in out.items() if k!='checks'},ensure_ascii=False))
