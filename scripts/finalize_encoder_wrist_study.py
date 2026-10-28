#!/usr/bin/env python3
"""Preserve completed encoder-wrist physics when orchestration exits with an API error.

Usage: python scripts/finalize_encoder_wrist_study.py STUDY_ROOT
Only use after the worker has exited. Never overwrites an existing summary.
"""
import argparse,json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root)
out=root/'summary.json'
if out.exists():raise SystemExit('Existing summary preserved; no action')
history=json.loads((root/'progress.json').read_text());calls=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')]
if any(c['status']=='started' for c in calls):raise SystemExit('Unresolved request; confirm worker completion and retain unknown usage before finalizing')
first=min(c['started_at'] for c in calls);last=max(c['ended_at'] for c in calls);start=datetime.fromisoformat(first);end=datetime.fromisoformat(last)
s={'experiment_id':root.name,'started_at':first,'ended_at':last,'wall_s':(end-start).total_seconds(),'wall_scope':'first API request through last failed request; initial evidence preparation not measured','status':'api_error','attempts':history,'api_requests':len(calls),'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'unknown_usage_calls':sum(c.get('usage') is None for c in calls),'api_wall_s':sum(c.get('wall_s') or 0 for c in calls),'training_ready':False,'finalized_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'note':'Physical attempts and independent token replays preserved; next API plan failed before physics.'}
with out.open('x') as f:json.dump(s,f,ensure_ascii=False,indent=2)
print(out)
