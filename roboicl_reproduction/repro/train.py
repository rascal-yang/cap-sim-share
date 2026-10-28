"""Verified command-log TRAIN for the explicitly different HUMANO profile.

Hashing establishes file integrity, not that a demonstration was physically
successful. Runtime provenance, success audit and camera calibration remain
experiment admission requirements. This builder never executes commands.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from .session import digest
from roboicl.policy.trajectory_memory import omitted_interval


def build_train(path, expected_sha256, profile, convert, *, starts, horizon, state_keys, cameras):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('TRAIN source hash mismatch')
    source = json.loads(raw)
    json.dumps(source, allow_nan=False)
    if source.get('schema') != 'humano.command_episode.v1' or source.get('profile') != profile:
        raise ValueError('TRAIN/LIVE profile mismatch')
    observations, transitions = source['observations'], source['transitions']
    if not isinstance(transitions, list) or not transitions or len(observations) != len(transitions)+1:
        raise ValueError('TRAIN requires actual before/after observations for every transition')
    times = source.get('observation_times_s')
    if (not isinstance(times, list) or len(times) != len(observations)
            or any(type(t) not in (int,float) or not math.isfinite(t) for t in times)
            or any(b <= a for a,b in zip(times,times[1:]))):
        raise ValueError('TRAIN requires finite, strictly increasing recorded observation times')
    for j,t in enumerate(transitions):
        elapsed = t.get('elapsed_s')
        if (type(elapsed) not in (int,float) or not math.isfinite(elapsed)
                or abs(times[j+1]-times[j]-elapsed)>1e-8):
            raise ValueError('TRAIN transition duration disagrees with recorded observation times')
    for ob in observations:
        if (not isinstance(ob, dict) or set(ob) != {'state','images'} or not isinstance(ob['state'],dict)
                or not isinstance(ob['images'],dict) or not ob['images']
                or any(not isinstance(v,str) or not v for v in ob['images'].values())):
            raise ValueError('invalid recorded observation')
        if set(ob['state']) != set(state_keys) or set(ob['images']) != set(cameras):
            raise ValueError('TRAIN observation whitelist differs from LIVE profile')
    if (type(horizon) is not int or horizon < 1 or not isinstance(starts,list) or not starts
            or starts[0] != 0 or any(type(s) is not int or s < 0 for s in starts)
            or any(b < a+horizon for a,b in zip(starts,starts[1:]))
            or starts[-1]+horizon > len(transitions)):
        raise ValueError('TRAIN requires frame 0 and ordered non-overlapping complete chunks')
    messages=[]; previous=0
    def observation(step, terminal=False):
        ob=observations[step]
        return {'role':'user','content':[
            {'type':'input_text','text':json.dumps({'source':'TRAIN','observation_step':step,
                'robot_state':ob['state'],'end_of_example':terminal})},
            *[{'type':'input_image','image_url':ob['images'][cam],'detail':'original'} for cam in sorted(ob['images'])]]}
    for i,start in enumerate(starts):
        end=start+horizon; selected=transitions[start:end]
        rows=[]; commands=[]
        for t in selected:
            if (t.get('control_modes') != [1] or any(type(m) is not int for m in t['control_modes']) or t.get('event') is not None
                    or type(t.get('elapsed_s')) not in (int,float) or not math.isfinite(t['elapsed_s'])
                    or abs(t['elapsed_s']-.1)>1e-8 or not isinstance(t.get('rows'),list)
                    or len(t['rows']) != 1 or not isinstance(t.get('commands'),list) or len(t['commands']) != 1):
                raise ValueError('TRAIN requires complete .1s mode1 recorded command transitions without interruption')
            rows.extend(t['rows']); commands.extend(t['commands'])
        converted=convert(deepcopy(rows),deepcopy(observations[start]['state']))
        if converted != commands:
            raise ValueError('TRAIN recorded commands differ from LIVE conversion; no endpoint-delta relabeling')
        if start>previous:
            messages.append(omitted_interval('TRAIN',previous,start,episode_id=expected_sha256))
        messages.append(observation(start))
        cid=f'train_{expected_sha256[:16]}_{i}'
        messages.append({'type':'function_call','name':'act','call_id':cid,
                         'arguments':json.dumps({'actions':rows,'execution_note':'Recorded demonstration commands'})})
        ledger=[{'step':start+j+1,'command':c,'elapsed_s':selected[j]['elapsed_s']} for j,c in enumerate(commands)]
        messages.append({'type':'function_call_output','call_id':cid,'output':json.dumps({
            'source':'TRAIN','executed_step_interval':[start,end],'predicted_steps':horizon,
            'executed_steps':horizon,'unexecuted_steps':0,'actual_elapsed_s':sum(t['elapsed_s'] for t in selected),
            'executed_command_ledger_sha256':digest(ledger),'terminal':None,'success':None,
            'controller_events':[], 'last_submitted_command':commands[-1]})})
        messages.append(observation(end, end==len(transitions)));previous=end
    if previous<len(transitions):
        messages.append(omitted_interval('TRAIN',previous,len(transitions),episode_id=expected_sha256))
        messages.append(observation(len(transitions),True))
    messages.append({'role':'user','content':[{'type':'input_text','text':
        'END OF FIXED TASK REFERENCE. Subsequent observations belong to the active episode. Generate new actions; do not replay TRAIN.'}]})
    return messages
