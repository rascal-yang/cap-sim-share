#!/usr/bin/env python3
"""Read-only versioned acceptance review; does not rewrite frozen source evidence."""
import argparse,json,time
from pathlib import Path
from humanoid_mimicgen.augmentation.button_acceptance import evaluate
from humanoid_mimicgen.augmentation.agent import save
from humanoid_mimicgen.workflow.contracts import now
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());r=Path(cfg['output_root']);start=now();tick=time.monotonic();results=[]
for entry in cfg['cases']:
 result=evaluate(entry['case'],r/(entry['name']+'.json'));results.append({'name':entry['name'],**result})
save(r/'summary.json',{'started_at':start,'ended_at':now(),'wall_s':time.monotonic()-tick,'physics_steps':0,'api_requests':0,'results':results});print([(x['name'],x['native_success'],x['revised_success'],x['revised_hold_s']) for x in results])
