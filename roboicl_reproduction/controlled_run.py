"""Original-position TRAIN/LIVE runner with declared control feedback and transport."""
import argparse
import json
from pathlib import Path
import time
import traceback
from native_run import write, now, verify_frozen, train_messages
from repro.bridge import STATE_KEYS,CAMERAS
from repro.controlled_bridge import ControlAwareBridge
from repro.humano import encode_actions,ACTION_SCHEMA,PROFILE
from repro.robust_session import RobustSession,verify_response
from roboicl.policy.action_feedback import TrackingGuard

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--attempt',type=int,required=True);a=p.parse_args()
 exp=a.experiment.resolve();contract=verify_frozen(exp);task=json.loads((exp/'task.json').read_text())
 if not 1<=a.attempt<=contract['budget']['attempts_per_case']:raise ValueError('Attempt out of budget')
 prefix=train_messages(json.loads((exp/'train_memory.json').read_text()))
 if a.attempt>1:
  previous=exp/f'attempt_{a.attempt-1:03d}/reflection.json'
  if not previous.is_file():raise ValueError('Previous reflection required')
  prefix.append({'role':'user','content':[{'type':'input_text','text':json.dumps({'source':'PAST_EPISODE_EXTENSION','episode_boundary':'NEW reset to original initial state, previous endpoint is NOT current state','reflection':json.loads(previous.read_text())})}]})
 directory=exp/f'attempt_{a.attempt:03d}';directory.mkdir(exist_ok=False);out=directory/'run';out.mkdir();mailbox=directory/'mailbox';mailbox.mkdir()
 backend=ControlAwareBridge(directory,json.loads((exp/'source_evidence.json').read_text())['physics'],standing_idle=True)
 begun=now();started=time.perf_counter();wait_s=0;session=None;outcome='unknown';calls=0;provenances=[]
 write(directory/'started.json',{'experiment_id':exp.name,'started_at':begun})
 maximum_s=contract['budget']['simulation_seconds_per_attempt'];horizon=task['budget']['max_rows']
 try:
  initial=backend.reset(task,out)
  session=RobustSession(initial,convert=lambda rows,state:encode_actions(rows,state,backend.limits,horizon=horizon),state_keys=STATE_KEYS,cameras=CAMERAS,horizon=horizon,maximum=int(round(maximum_s/.1)),anchors=8,corrections=contract['budget']['recovery_attempts'],train_messages=prefix)
  for turn in range(contract['budget']['feedback_calls']):
   if session.terminal:break
   request=mailbox/f'{turn:03d}_request.json';response=mailbox/f'{turn:03d}_response.json';provenance=mailbox/f'{turn:03d}_response.provenance.json';ready=mailbox/f'{turn:03d}_ready.json';error=mailbox/f'{turn:03d}_transport_error.json'
   write(request,{'system':(exp/'planner_prompt.txt').read_text(),'profile':PROFILE+'_standing_idle_controlled','action_schema':ACTION_SCHEMA,'joint_limits':backend.limits,'input':session.context(),'remaining_sim_seconds':maximum_s-float(backend.d.time),'remaining_calls':contract['budget']['feedback_calls']-turn,'control_status':backend.control_status(),'response_path':str(response)})
   print('WAITING_PLANNER',request,flush=True);waiting=time.perf_counter()
   try:
    while not ready.exists():
     if error.exists():raise RuntimeError('Planner transport unavailable: '+error.read_text())
     if time.perf_counter()-waiting>900:raise TimeoutError('Planner response deadline')
     time.sleep(.2)
   finally:wait_s+=time.perf_counter()-waiting
   provenances.append(verify_response(request,response,provenance,seen_agents={r['agent'] for r in provenances}));calls+=1
   backend.tracking_guard=TrackingGuard()
   session.submit_json(response.read_text(),lambda command:backend.step(command,maximum_s-float(backend.d.time)))
   write(directory/'live.json',{'step':session.step,'terminal':session.terminal,'receipts':session.receipts,'ledger':session.ledger})
   write(directory/'latest_context.json',session.context());write(directory/'planner_provenance.json',provenances)
   if backend.d.time>=maximum_s-1e-8:session.terminal=True;outcome='simulation_budget'
  if backend.success:outcome='native_success'
  elif session.receipts and session.receipts[-1]['controller_events']:outcome=session.receipts[-1]['controller_events'][-1]['type']
  elif outcome=='unknown':outcome='feedback_or_simulation_budget'
 except Exception as error:
  outcome=type(error).__name__+': '+str(error);write(directory/'error.json',{'error':outcome,'traceback':traceback.format_exc()})
 finally:
  try:
   if hasattr(backend,'sonic') and backend.sonic is not None:
    backend.export(out);verified=backend.verify(out) if backend.states else {'success':False,'states_checked':0,'physics_steps':0}
    write(out/'independent_validation.json',verified);write(out/'stage_results.json',backend.results)
    write(directory/'result.json',{'experiment_id':exp.name,'attempt':a.attempt,'outcome':outcome,'success':bool(verified['success'] and backend.success),'simulation_s':float(backend.d.time),'physics_steps':len(backend.states),'actual_modes':sorted({r['mode'] for r in backend.sonic.records}),'planner_calls':calls,'metrics':backend.metrics(),'independent_validation':verified})
   if session is not None:
    write(directory/'final_context.json',session.context());write(directory/'live.json',{'step':session.step,'terminal':session.terminal,'receipts':session.receipts,'ledger':session.ledger})
   write(directory/'planner_provenance.json',provenances)
  except Exception as error:
   outcome='finalization_error: '+str(error);write(directory/'finalization_error.json',{'error':outcome,'traceback':traceback.format_exc()})
  finally:
   write(directory/'timing.json',{'started_at':begun,'ended_at':now(),'total_wall_s':time.perf_counter()-started,'planning_wait_wall_s':wait_s,'simulation_compute_wall_s':backend.simulation_wall_s,'render_wall_s':backend.render_wall_s,'preparation_planning_s':'unknown'})
   write(directory/'done.json',{'outcome':outcome,'ended_at':now()});backend.close();print('FINISHED',outcome,flush=True)
if __name__=='__main__':main()
