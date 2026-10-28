import json,copy,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
r=Path(__file__).resolve().parent;old=r.parent/'20261013_cap_ten_tasks';name='G1WholebodyTableApplePickTeleop'
ref=json.loads((old/'references/candidates'/name/'episode_000000.json').read_text());df=pd.read_parquet(ref['source']);ts=np.array([float(np.asarray(t).reshape(-1)[0]) for t in df['sim.time']]);ts-=ts[0];actions=np.stack(df['action.wbc'])
for s in ref['samples']:
 i=int(np.argmin(abs(ts-s['time'])));s['finger_commands']=[actions[i,29:36].tolist(),actions[i,36:43].tolist()]
(r/'reference.json').write_text(json.dumps(ref))
t=json.loads((old/'runs/no_reference'/name/'task.json').read_text());t['references']=[str(r/'reference.json')];t['code_version']='eda87f8';t['generation_kind']='reference_generation';t['budget'].update(attempts=10,optimization_attempts=1,feedback_calls=12)
(r/'original_task.json').write_text(json.dumps(t,indent=2));t=copy.deepcopy(t);t['scene']={'translation':[0,.1,0]};(r/'target_task.json').write_text(json.dumps(t,indent=2))
(r/'protocol.json').write_text(json.dumps({'started_unix_s':time.time(),'reference_episode':0,'source_sha256':ref['source_sha256'],'source_saved_state_success':False,'source_status':'candidate requires current-interface recovery; not asserted successful','recovery_attempt_budget':10,'generation_attempt_budget':10,'attempt_sim_s':30,'feedback_calls':12,'variant_translation':[0,.1,0],'live_model_api':False,'control_changes':False,'physics_changes':False,'success_criteria_changes':False,'human_candidates':['Reference hand-wrap sequence with actual finger commands','Measured Cartesian approach with SONIC legs and closed-loop wrist correction'],'first_prediction':'Removing gratuitous alignment and reproducing complete source body preview should restore forward approach; measured root and wrist must be checked before closure.'},indent=2))
(r/'configs').mkdir(exist_ok=True)
(r/'configs/reference_recovery_001.json').write_text(json.dumps({'mode':'body','rate':1.,'diagnosis':'Old reference run used episode 1 and stopped fingers with reference time. Use episode 0 matched reset, full future body preview and commanded fingers; no artificial lateral step in original scene.'},indent=2))
