"""Frozen paired controller diagnostic. No action model, TRAIN replay or IK."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from native_run import write, now, verify_frozen
from repro.bridge import NativeBridge
from repro.idle_bridge import StandingIdlePlanner
from cap_sim.backends.native_sonic import rot

def fingerprint(backend):
    values=[backend.d.qpos,backend.d.qvel,backend.d.ctrl,backend.d.qacc_warmstart,
            backend.sonic.last,backend.sonic.ref]
    for h in backend.sonic.history:
        values.extend(h[k] for k in sorted(h))
    return hashlib.sha256(b''.join((b'NONE' if v is None else np.asarray(v).tobytes()) for v in values)).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--case',choices=['legacy','standing_idle'],required=True);a=p.parse_args()
    exp=a.experiment.resolve();contract=verify_frozen(exp)
    if a.case=='standing_idle' and not (exp/'legacy/reflection.json').exists():
        raise ValueError('First case reflection required')
    directory=exp/a.case;directory.mkdir(exist_ok=False);out=directory/'run';out.mkdir()
    task=json.loads((exp/'task.json').read_text());cfg=json.loads((exp/'probe.json').read_text())
    backend=NativeBridge(directory,json.loads((exp/'source_evidence.json').read_text())['physics'])
    start=now();t=time.monotonic();outcome='unknown';stop=None;records=[]
    write(directory/'started.json',{'experiment_id':exp.name,'case':a.case,'started_at':start})
    try:
        ob=backend.reset(task,out);initial=deepcopy(ob['state']);root0=np.array(initial['root_pose'][:3])
        initial_fingerprint=fingerprint(backend)
        adapter=StandingIdlePlanner(backend.sonic.planner,a.case=='standing_idle',lambda:float(backend.d.time))
        backend.sonic.planner=adapter
        yaw=float(rot(initial['root_pose'][3:7]).as_euler('xyz')[2])
        for index in range(contract['budget']['feedback_calls']):
            state=ob['state'];tm=float(backend.d.time)
            if stop is None and (state['root_pose'][0]>=cfg['approach_stop_root_x_m'] or tm>=cfg['approach_max_s']-1e-8):
                stop={'time':tm,'state':deepcopy(state),'fingerprint':fingerprint(backend),
                      'prefix_sha256':hashlib.sha256(np.asarray(backend.states).tobytes()).hexdigest(),
                      'initial_fingerprint':initial_fingerprint}
                write(directory/'stop_boundary.json',stop)
                if a.case=='standing_idle':
                    legacy=json.loads((exp/'legacy/stop_boundary.json').read_text())
                    if any(stop[k]!=legacy[k] for k in ('time','fingerprint','prefix_sha256','initial_fingerprint')):
                        raise ValueError('Unmatched approach: refusing stop comparison')
            if stop is not None and tm>=stop['time']+cfg['stop_duration_s']-1e-8:
                outcome='completed_stop_window';break
            action={'stage':'walk' if stop is None else 'hold','duration_s':.1,
                    'velocity_xy':cfg['approach_velocity_xy'] if stop is None else [0,0],
                    'facing_yaw':yaw,'pelvis_height_m':cfg['pelvis_height_m'],
                    'torso_quaternion_wxyz':initial['torso_quaternion_wxyz']}
            for j,side in enumerate(('left','right')):
                action[side+'_position']=(np.array(initial[side+'_position'])+np.array(state['root_pose'][:3])-root0).tolist() if stop is None else stop['state'][side+'_position']
                action[side+'_quaternion_wxyz']=initial[side+'_quaternion_wxyz']
                action[side+'_fingers']=cfg['fingers'][j]
            result=backend.step(action,contract['budget']['simulation_seconds_per_attempt']-tm)
            ob=result.observation
            records.append({'index':index,'phase':'approach' if stop is None else 'stop','time':float(backend.d.time),
                            'command':action,'state':ob['state'],'feedback':result.feedback,'interrupt':result.interrupt})
            write(directory/'commands.json',records)
            if result.terminal or not result.executed:
                outcome=result.interrupt or 'zero_step';break
        else:outcome='command_budget'
    except Exception as error:
        outcome=type(error).__name__+': '+str(error)
        write(directory/'error.json',{'error':outcome,'traceback':traceback.format_exc()})
    finally:
        try:
            if hasattr(backend,'sonic'):
                backend.export(out);write(out/'stage_results.json',backend.results)
                write(directory/'planner_inputs.json',getattr(backend.sonic.planner,'records',[]))
                simtime=float(backend.d.time);metrics=backend.metrics()
                verified=backend.verify(out) if backend.states else {'success':False,'physics_steps':0}
                write(out/'independent_validation.json',verified)
                write(directory/'result.json',{'experiment_id':exp.name,'case':a.case,'outcome':outcome,
                    'simulation_s':simtime,'physics_steps':len(backend.states),'nq':backend.m.nq,'nv':backend.m.nv,
                    'actual_encoder_modes':sorted({r['mode'] for r in backend.sonic.records}),
                    'metrics':metrics,'native_grasp_validation':verified,'is_roboicl_grasp_trial':False})
        finally:
            write(directory/'timing.json',{'started_at':start,'ended_at':now(),'total_wall_s':time.monotonic()-t,
                  'action_model_planning_s':0,'preparation_planning_s':'unknown',
                  'simulation_compute_wall_s':backend.simulation_wall_s,'observation_render_wall_s':backend.render_wall_s})
            write(directory/'done.json',{'outcome':outcome,'ended_at':now()});backend.close()
    print(outcome,flush=True)
if __name__=='__main__':main()
