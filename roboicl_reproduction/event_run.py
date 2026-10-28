"""One isolated episode planner with explicit bounded fixed-target wait macros."""
import argparse
import json
from pathlib import Path
import time
import traceback
from native_run import write, now, verify_frozen, train_messages
from repro.bridge import STATE_KEYS,CAMERAS
from repro.controlled_bridge import ControlAwareBridge
from repro.humano import encode_actions,ACTION_SCHEMA,PROFILE
from repro.feedback_session import FeedbackSession
from repro.episode_transport import verify_episode_response, observation_train
from repro.event_session import submit_wait
from roboicl.policy.action_feedback import TrackingGuard

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--attempt',type=int,required=True);a=p.parse_args()
 exp=a.experiment.resolve();contract=verify_frozen(exp);task=json.loads((exp/'task.json').read_text())
 if not 1<=a.attempt<=contract['budget']['attempts_per_case']:raise ValueError('Attempt out of budget')
 prefix=observation_train(json.loads((exp/'train_memory.json').read_text()))
 if a.attempt != 1: raise ValueError('Exactly one fresh episode is authorized')
 birth=json.loads((exp/'planner_birth.json').read_text())
 directory=exp/f'attempt_{a.attempt:03d}';directory.mkdir(exist_ok=False);out=directory/'run';out.mkdir();mailbox=directory/'mailbox';mailbox.mkdir()
 backend=ControlAwareBridge(directory,json.loads((exp/'source_evidence.json').read_text())['physics'],standing_idle=True)
 begun=now();started=time.perf_counter();wait_s=0;session=None;outcome='unknown';calls=0;provenances=[];wait_events=[];model_calls=[];receipt_cursor=0
 write(directory/'started.json',{'experiment_id':exp.name,'started_at':begun})
 maximum_s=contract['budget']['simulation_seconds_per_attempt'];horizon=task['budget']['max_rows']
 try:
  initial=backend.reset(task,out)
  session=FeedbackSession(initial,convert=lambda rows,state:encode_actions(rows,state,backend.limits,horizon=horizon),state_keys=STATE_KEYS,cameras=CAMERAS,horizon=horizon,maximum=int(round(maximum_s/.1)),anchors=8,corrections=contract['budget']['recovery_attempts'],train_messages=prefix)
  for turn in range(contract['budget']['feedback_calls']):
   if session.terminal:break
   request=mailbox/f'{turn:03d}_request.json';response=mailbox/f'{turn:03d}_response.json';provenance=mailbox/f'{turn:03d}_response.provenance.json';ready=mailbox/f'{turn:03d}_ready.json';error=mailbox/f'{turn:03d}_transport_error.json'
   new_receipts=session.receipts[receipt_cursor:]
   inputs=session.context() if turn==0 else [session._message(session.step), {'role':'user','content':[{'type':'input_text','text':json.dumps({'source':'LIVE','new_receipts':new_receipts,'uncommitted_protocol_messages':session.tail,'last_wait':({k:v for k,v in wait_events[-1].items() if k!='receipts'} if wait_events and model_calls[-1]['name']=='wait_until_stable' else None)})}]}]
   receipt_cursor=len(session.receipts)
   write(request,{'system':(exp/'planner_prompt.txt').read_text(),'profile':PROFILE+'_episode_event_wait','action_schema':ACTION_SCHEMA,'wait_schema':{'name':'wait_until_stable','arguments':{'max_wait_s':'0.1..5.0, multiple of0.1','execution_note':'string <=400 chars'}},'joint_limits':backend.limits,'input':inputs,'remaining_sim_seconds':maximum_s-float(backend.d.time),'remaining_calls':contract['budget']['feedback_calls']-turn,'control_status':backend.control_status(),'last_submitted_command':session.ledger[-1]['command'] if session.ledger else None,'episode_id':exp.name,'turn_index':turn,'expected_agent':birth['agent'],'previous_response_sha256':provenances[-1]['response_sha256'] if provenances else None,'response_path':str(response)})
   print('WAITING_PLANNER',request,flush=True);waiting=time.perf_counter()
   try:
    while not ready.exists():
     if error.exists():raise RuntimeError('Planner transport unavailable: '+error.read_text())
     if time.perf_counter()-waiting>900:raise TimeoutError('Planner response deadline')
     time.sleep(.2)
   finally:wait_s+=time.perf_counter()-waiting
   provenances.append(verify_episode_response(request,response,provenance,agent=birth['agent'],episode=exp.name,turn=turn,previous=provenances[-1]['response_sha256'] if provenances else None));calls+=1
   backend.tracking_guard=TrackingGuard()
   raw=response.read_text()
   try: call=json.loads(raw)
   except ValueError: call=None
   first_receipt=len(session.receipts);before_time=float(backend.d.time);before_step=session.step
   if isinstance(call,dict) and str(call.get('call_id','')).startswith('__compiled_wait__'):
    session._invalid_envelope('Reserved executor call_id prefix')
   elif isinstance(call,dict) and call.get('name')=='wait_until_stable':
    try:wait_events.append(submit_wait(session,call,backend,maximum_s))
    except (ValueError,KeyError,TypeError) as error:
     if session.terminal or session.step!=before_step or abs(float(backend.d.time)-before_time)>1e-9:raise
     session._invalid_envelope('Invalid wait: '+str(error))
   else:session.submit_json(raw,lambda command:backend.step(command,maximum_s-float(backend.d.time)))
   model_calls.append({'turn':turn,'name':call.get('name') if isinstance(call,dict) else 'invalid_json','call_id':call.get('call_id') if isinstance(call,dict) else None,'receipt_interval':[first_receipt,len(session.receipts)],'simulation_interval_s':[before_time,float(backend.d.time)]})
   write(directory/'wait_events.json',wait_events);write(directory/'model_calls.json',model_calls)
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
   write(directory/'wait_events.json',wait_events);write(directory/'model_calls.json',model_calls)
  except Exception as error:
   outcome='finalization_error: '+str(error);write(directory/'finalization_error.json',{'error':outcome,'traceback':traceback.format_exc()})
  finally:
   write(directory/'timing.json',{'started_at':begun,'ended_at':now(),'total_wall_s':time.perf_counter()-started,'planning_wait_wall_s':wait_s,'simulation_compute_wall_s':backend.simulation_wall_s,'render_wall_s':backend.render_wall_s,'preparation_planning_s':'unknown'})
   write(directory/'done.json',{'outcome':outcome,'ended_at':now()});backend.close();print('FINISHED',outcome,flush=True)
if __name__=='__main__':main()
