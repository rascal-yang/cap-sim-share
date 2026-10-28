#!/usr/bin/env python3
"""Synthetic protocol demonstration: original policy, no API or physics calls."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import time
from repro.offline import create_policy, observation, response


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'); tick=time.monotonic()
    p=create_policy(args.output,[response('invalid',[[True]*14]), response('first'),response('after_rejection'),response('after_partial')])
    p.update_obs(observation(0)); p.get_action()
    p.notify_action_rejection({'step':0,'event':{'type':'action_rejected_ik_continuity',
        'rejected_action_executed':False,'action_index':0,'discarded_commands':2}})
    actions=p.get_action()
    p.update_obs(observation(1,actions[0],{'type':'chunk_interrupted_tracking_error','discarded_commands':1}))
    actions=p.get_action();p.update_obs(observation(2,actions[0]));p.finalize()
    result={'kind':'synthetic_protocol_only','physics_executed':False,'api_requested':False,
            'start':started,'end':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            'total_elapsed_s':time.monotonic()-tick,'planning_s':None,'simulation_s':0,'rendering_s':0,
            'provider_responses':len(p.requests),'observed_command_transitions':p.step,
            'trace_directory':str(p.episode_dir),'memory':p.live_memory.stats,
            'task_success':None,'training_ready':False}
    (args.output/'requests.json').write_text(json.dumps(p.requests,indent=2))
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
