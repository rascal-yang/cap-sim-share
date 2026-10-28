"""Read-only result accounting; native pass and grasp validity remain distinct."""
import json,time
from pathlib import Path
r=Path(__file__).resolve().parent
rows=[]
for p in sorted((r/'runs').glob('*/*/summary.json')):
 s=json.loads(p.read_text());case=p.parent
 attempts=[json.loads(x.read_text()) for x in sorted((case/'attempts').glob('*/result.json'))]
 selected=s.get('accepted_attempt');quality=None
 if selected:
  q=case/'attempts'/f'{selected:03d}'/'grasp_quality.json'
  if q.exists():quality=json.loads(q.read_text())
 native=bool(s['success']);is_pick='Pick' in case.name;valid=native and (not is_pick or bool(quality and quality.get('valid_grasp') is True));quality_pending=native and is_pick and (not quality or 'valid_grasp' not in quality)
 rows.append({'group':case.parent.name,'task':case.name,'native_success':native,'valid_task_success':valid,'quality':quality,'quality_pending':quality_pending,'attempts':len(attempts),'selected_attempt':selected,'selected_sim_s':next((a['acceptance']['sim_seconds'] for a in attempts if a['attempt']==selected),None),'sim_s_executed':sum(a['acceptance']['sim_seconds'] for a in attempts),'interrupted_attempts':[a['attempt'] for a in attempts if a.get('interrupted')],'failures':[{'attempt':a['attempt'],'reason':a.get('failure')} for a in attempts if not a['acceptance']['success']],'recorded_attempt_wall_s':sum(a.get('wall_s') or 0 for a in attempts),'initial_planning_wall_s':sum(a.get('initial_planning_wall_s') or 0 for a in attempts),'mailbox_decision_wall_s':sum(json.loads(x.read_text()).get('wait_for_GPT_wall_s',0) for x in (r/'mailbox'/case.parent.name/case.name).glob('*_timing.json')),'video_verified':(r/'videos'/case.parent.name/case.name/'verification.json').exists()})
report={'generated_unix_s':time.time(),'cases':rows,'notes':['No-reference input excludes expert rollouts; source initial state is allowed.','Wall time includes mailbox waiting; interrupted attempt wall duration is not reconstructed.','Native pick height checks alone do not establish a valid grasp.','Automatic policy decisions are GPT-authored code, not live API calls.']}
(r/'results_audit.json').write_text(json.dumps(report,indent=2))
for x in rows:print(x['group'],x['task'],x['valid_task_success'],x['attempts'],x['selected_sim_s'],x['interrupted_attempts'])
