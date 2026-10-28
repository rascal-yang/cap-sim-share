"""JSON mailbox for session decisions or the explicit GPT-authored CaP policy.
--auto-policy executes the recorded Python rules; it is not a live model call.
No external model API is used by this benchmark wrapper.
"""
import argparse,time,json,subprocess
from pathlib import Path
from cap_sim.backends.humano import Backend
from cap_sim.core import run_task
from cap_sim.io import read,write
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--group',choices=['no_reference','reference_augmented'],required=True);p.add_argument('--attempts',type=int,default=10);p.add_argument('--resume',action='store_true');p.add_argument('--auto-policy',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parent;name=a.task
out=r/'runs'/a.group/name;mail=r/'mailbox'/a.group/name;mail.mkdir(parents=True,exist_ok=True)
meta=read(r/'prepared'/name/'initial_observation.json')
task={'name':name,'backend':'humano','instruction':read(r/'prepared'/name/'environment_config.json')['description'],'resources':{'prepared':str(r/'prepared'/name),'initial_state':str(r/'dataset'/name/'sonic/episode_000000_start.npz'),'sonic_models':'/home/kinetra-4090-01/HUMANO/third_party/gear_sonic_deploy'},'scene':{} if a.group=='no_reference' else {'translation':[0,.1,0]},'acceptance':{'kind':'native','hold_s':.005,'validator_id':meta['source_sha256']},'budget':{'attempts':a.attempts,'sim_seconds':30,'optimization_attempts':1,'feedback_calls':12},'references':[],'skills':[],'generation_kind':'generated' if a.group=='no_reference' else 'augmented','planner_identity':'GPT current session via immutable JSON mailbox; no external API','code_version':subprocess.check_output(['git','-C','/home/kinetra-4090-01/code/zijian/cap-sim','rev-parse','HEAD'],text=True).strip()}
if a.group=='reference_augmented':task['references']=[str(r/'references'/f'{name}.json')]
class Mailbox:
 def plan(self,request):
  key=f"a{request['attempt']:03d}_f{request.get('feedback_index',0):02d}";q=mail/(key+'_request.json');reply=mail/(key+'_response.json');write(q,request);start=time.time();print('WAITING',q,flush=True)
  if a.auto_policy and not reply.exists():
   if a.group=="reference_augmented":
    from reference_policy import respond
   else:
    from serve_remaining import respond
   respond(q)
  while not reply.exists():
   if time.time()-start>3600:raise TimeoutError('GPT decision mailbox timed out')
   time.sleep(.2)
  result=read(reply);write(mail/(key+'_timing.json'),{'wait_for_GPT_wall_s':time.time()-start});return result
class DurableBackend(Backend):
 def execute(self,action,remaining):
  result=super().execute(action,remaining)
  import numpy as np
  np.savez_compressed(self.out/'states.npz',samples=np.asarray(self.states))
  write(self.out/'contacts.json',self.contact_records)
  write(self.out/'samples.json',self.samples)
  return result
b=DurableBackend()
try:
 result=run_task(task,Mailbox(),b,out,resume=a.resume);print(json.dumps(result),flush=True)
finally:b.close()
