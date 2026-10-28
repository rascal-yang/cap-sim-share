"""Frozen experiment entrypoint: TRAIN observations -> fresh act -> actual LIVE."""
import argparse
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import time
import traceback
from repro.bridge import NativeBridge, STATE_KEYS, CAMERAS
from repro.humano import encode_actions, ACTION_SCHEMA, PROFILE
from repro.session import Session
from roboicl.policy.action_feedback import TrackingGuard

PROMPT='''You are the isolated online action planner for HUMANO G1 with two Dex3 hands.
Use ONLY the supplied request and referenced images. No parent conversation,
repository code, raw source trajectories, tokens, stored policy or other files.
You may read the request/images and write exactly its response JSON. Do not run physics.
Goal: move forward and pick up the current apple. TRAIN contains sparse MEASURED
observations from a successful original trajectory. Learn stance, approach,
wrist orientation, finger configuration and contact conditions. TRAIN is from a
DIFFERENT controller (native token prefix then mode0); measured changes are NOT
mode1 command labels. Generate NEW actions using current LIVE observations. No
source-timeline replay. Every displayed temporal gap is real missing context.
LIVE provides actual executed prefix, discarded suffix, end residuals and contact
feedback. Invalid actions execute zero motion: repair from SAME observation.
A tracking interruption discards the remaining chunk: reanchor at measured pose.
You control world wrist deltas: [dx,dy,dz,rx,ry,rz], rotation VECTOR, not Euler.
Within a chunk these accumulate from the prior requested target, starting from
current measured wrist pose. Translation norm <=.03m, rotation norm <=.15rad each
0.1s command. Both hands require seven absolute finger joint targets in the
provided model order/limits. fingers in LIVE are previous COMMANDS;
measured_fingers are measured angles. Holding targets does not prove grasp.
Output 1..5 rows. Robot torque updates 50Hz; physical dt=.005s. SONIC lower-body
planner refreshes only every .5s, so zero requested velocity is NOT an instantaneous
brake. Choose short horizons near obstacles. Use actual root velocity to judge
settling, and current RGB to estimate clearance. Preserve explicit wrist and torso
orientation. Avoid accumulating wrist targets far ahead of measured tracking.
Lift stage requires measured continuous hand contact >=.12s; a zero-step gate
rejection is recoverable. Hard collision/fall ends this episode. Terminal state is
not permission to continue physical motion. Contact feedback here is privileged
simulation instrumentation; cameras are head and external, not original wrist cameras.
Submit exactly one JSON function_call with type='function_call',name='act',fresh
call_id,arguments a JSON STRING containing actions and execution_note (<=400 chars).
Use the supplied action_schema. No duration/body_keyframes/tokens/code fields.
Brief execution_note states the evidence, uncertainty and intended correction.
'''

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));temp.replace(path)
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_frozen(directory):
    if sha(directory/'experiment.json')!=(directory/'contract.sha256').read_text().strip():raise ValueError('Frozen contract changed')
    contract=json.loads((directory/'experiment.json').read_text())
    if contract['record_status']!='frozen' or not contract['preflight']['passed']:raise ValueError('Preflight incomplete')
    for path,h in contract['pipeline']['source_snapshot_and_hashes'].items():
        if sha(path)!=h:raise ValueError('Frozen input changed: '+path)
    return contract


def train_messages(demo):
    result=[{'role':'user','content':[{'type':'input_text','text':json.dumps({k:v for k,v in demo.items() if k not in ('observations','source_sha256')},ensure_ascii=False)}]}]
    previous=None
    for ob in demo['observations']:
        data=deepcopy(ob);images=data.pop('images',[])
        if previous is not None:
            result.append({'role':'user','content':[{'type':'input_text','text':json.dumps({'source':'TRAIN','omitted_source_frames':[previous+1,ob['source_frame']],
                'note':'Sparse measured observations only. No command or continuous interpolation is implied.'})}]})
        result.append({'role':'user','content':[{'type':'input_text','text':json.dumps({'source':'TRAIN','observation':data})},
                                              *[{'type':'input_image','image_url':p,'detail':'original'} for p in images]]})
        previous=ob['source_frame']
    result.append({'role':'user','content':[{'type':'input_text','text':'END TRAIN OBSERVATION MEMORY. LIVE starts now; generate new actions.'}]})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True,type=Path);p.add_argument('--attempt',required=True,type=int);args=p.parse_args()
    exp=args.experiment.resolve();contract=verify_frozen(exp)
    if not 1<=args.attempt<=contract['budget']['attempts_per_case']:raise ValueError('Attempt out of budget')
    directory=exp/f'attempt_{args.attempt:03d}';directory.mkdir(exist_ok=False)
    out=directory/'run';out.mkdir();mailbox=directory/'mailbox';mailbox.mkdir()
    task=json.loads((exp/'task.json').read_text());demo=json.loads((exp/'train_memory.json').read_text())
    prefix=train_messages(demo)
    if args.attempt>1:
        reflection=exp/f'attempt_{args.attempt-1:03d}/reflection.json'
        if not reflection.is_file():raise ValueError('Previous attempt reflection required before reset')
        prefix.append({'role':'user','content':[{'type':'input_text','text':json.dumps({'source':'PAST_EPISODE_EXTENSION',
            'episode_boundary':'NEW reset to original initial state; prior endpoint is NOT current state',
            'reflection':json.loads(reflection.read_text())})}]})
    backend=NativeBridge(directory,json.loads((exp/'source_evidence.json').read_text())['physics'])
    begun=now();started=time.perf_counter();wait_s=0;session=None;outcome='unknown';calls=0
    write(directory/'started.json',{'experiment_id':exp.name,'started_at':begun})
    try:
        initial=backend.reset(task,out)
        session=Session(initial,convert=lambda rows,state:encode_actions(rows,state,backend.limits,horizon=5),
            state_keys=STATE_KEYS,cameras=CAMERAS,horizon=5,maximum=300,anchors=8,corrections=2,train_messages=prefix)
        for turn in range(contract['budget']['feedback_calls']):
            if session.terminal:break
            request=mailbox/f'{turn:03d}_request.json';response=mailbox/f'{turn:03d}_response.json'
            write(request,{'system':PROMPT,'profile':PROFILE,'action_schema':ACTION_SCHEMA,'joint_limits':backend.limits,
                'input':session.context(),'remaining_sim_seconds':30-float(backend.d.time),'response_path':str(response)})
            print('WAITING_PLANNER',request,flush=True);waiting=time.perf_counter()
            try:
                while not response.exists():
                    if time.perf_counter()-waiting>1200:raise TimeoutError('Planner response deadline')
                    time.sleep(.2)
            finally:
                wait_s+=time.perf_counter()-waiting
            calls+=1
            call=json.loads(response.read_text());backend.tracking_guard=TrackingGuard()
            session.submit(call,lambda command:backend.step(command,30-float(backend.d.time)))
            write(directory/'live.json',{'step':session.step,'terminal':session.terminal,'receipts':session.receipts,'ledger':session.ledger})
            write(directory/'latest_context.json',session.context())
            if backend.d.time>=30-1e-8:session.terminal=True
        outcome='native_success' if backend.success else (session.receipts[-1]['controller_events'][-1]['type'] if session.receipts and session.receipts[-1]['controller_events'] else 'feedback_or_simulation_budget')
    except Exception as error:
        outcome=type(error).__name__+': '+str(error)
        write(directory/'error.json',{'error':outcome,'traceback':traceback.format_exc()})
    finally:
        try:
            if hasattr(backend,'sonic') and backend.sonic is not None:
                backend.export(out)
                verified=backend.verify(out) if backend.states else {'success':False,'states_checked':0}
                write(out/'independent_validation.json',verified)
                write(out/'stage_results.json',backend.results)
                modes=sorted({r['mode'] for r in backend.sonic.records})
                write(directory/'result.json',{'experiment_id':exp.name,'attempt':args.attempt,'outcome':outcome,
                     'success':bool(verified['success'] and backend.success),'simulation_s':float(backend.d.time),
                     'physics_steps':len(backend.states),'actual_modes':modes,'planner_calls':calls,
                     'metrics':backend.metrics(),'independent_validation':verified})
            if session is not None:
                write(directory/'final_context.json',session.context())
                write(directory/'live.json',{'step':session.step,'terminal':session.terminal,'receipts':session.receipts,'ledger':session.ledger})
        except Exception as error:
            outcome='finalization_error: '+str(error)
            write(directory/'finalization_error.json',{'error':outcome,'traceback':traceback.format_exc()})
        finally:
            write(directory/'timing.json',{'started_at':begun,'ended_at':now(),'total_wall_s':time.perf_counter()-started,
                   'planning_wait_wall_s':wait_s,'simulation_compute_wall_s':backend.simulation_wall_s,'render_wall_s':backend.render_wall_s})
            write(directory/'done.json',{'outcome':outcome,'ended_at':now()});backend.close()
            print('FINISHED',outcome,flush=True)
if __name__=='__main__':main()
