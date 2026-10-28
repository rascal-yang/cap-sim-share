import json,time,hashlib,runpy
from pathlib import Path
r=Path(__file__).resolve().parent
class Planner:
 def __init__(self,out):self.out=Path(out)
 def plan(self,q):
  phase=self.out.name;n=q['attempt'];f=q.get('feedback_index',0);key=f'{phase}_a{n:03d}_f{f:02d}';mail=r/'mailbox';mail.mkdir(exist_ok=True);start=time.time()
  request=mail/(key+'_request.json')
  if not request.exists():request.write_text(json.dumps(q))
  cfg=r/'configs'/f'{phase}_{n:03d}.json';print('DECISION',key,flush=True)
  while not cfg.exists():time.sleep(.2)
  c=json.loads(cfg.read_text());src=r/'stage_policy.py';code=src.read_bytes();h=hashlib.sha256(code).hexdigest();snap=r/'policies';snap.mkdir(exist_ok=True);(snap/(h+'.py')).write_bytes(code)
  result=runpy.run_path(str(src))['decide'](q,c,r)
  result['planner_provenance']={'kind':'GPT session authored feedback policy','live_model_call':False,'sha256':h}
  (mail/(key+'_response.json')).write_text(json.dumps(result));(mail/(key+'_timing.json')).write_text(json.dumps({'decision_wall_s':time.time()-start}));return result
