import json,sys
from pathlib import Path
r=Path(__file__).resolve().parent;n=int(sys.argv[1]);group=sys.argv[2] if len(sys.argv)>2 else 'reference_recovery'
for p in sorted((r/'mailbox').glob(f'{group}_a{n:03d}_*request.json')):
 q=json.loads(p.read_text());o=q['observation'];resp=p.with_name(p.name.replace('_request','_response'));a=json.loads(resp.read_text())['actions'][0] if resp.exists() else {}
 print(p.stem.split('_')[-1],round(o['time'],2),a.get('name','WAIT'),'root',[round(x,3) for x in o['root_pose'][:3]],'wrist',[round(x,3) for x in o['right_position']],'obj',[round(x,3) for x in o['objects']['target'][:3]],'N',round(o['contacts']['hand_object_N'],2),'contact',round(o['contact_s'],3))
p=r/'run'/group/'attempts'/f'{n:03d}'/'result.json'
if p.exists():
 d=json.loads(p.read_text());print('RESULT',d['failure'],d['acceptance'],'wall_s',d['wall_s'])
