"""Bounded event wait over unchanged .1 s physical commands.

The bridge samples stability/contact at 200 Hz. This helper inspects that saved
evidence after each command, so nonterminal event response is at the next .1 s
command boundary (at most .1 simulated seconds), not an exact 5 ms stop. Native
hard stops may interrupt a command earlier. No time or observations are invented.
"""
from copy import deepcopy
import json
import math
from numbers import Real

from .session import StepResult

COMMAND_SECONDS = .1
MAX_WAIT_SECONDS = 5.
HAND_CONTACT_THRESHOLD_N = .1


def _number(value):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):
        raise ValueError('Finite numeric value required, excluding bool')
    return float(value)


def _duration(value):
    value = _number(value)
    if not COMMAND_SECONDS<=value<=MAX_WAIT_SECONDS or abs(value/COMMAND_SECONDS-round(value/COMMAND_SECONDS))>1e-8:
        raise ValueError('max_wait_s must be .1..5 seconds in .1 second increments')
    return value


def validate_wait_call(call, seen_call_ids=()):
    if (not isinstance(call,dict) or set(call)!={'type','name','call_id','arguments'}
            or call['type']!='function_call' or call['name']!='wait_until_stable'
            or not isinstance(call['call_id'],str) or not call['call_id'].strip()
            or call['call_id'] in seen_call_ids or not isinstance(call['arguments'],str)):
        raise ValueError('One wait_until_stable function_call with a fresh call_id required')
    try:
        args = json.loads(call['arguments'])
    except (ValueError,TypeError) as error:
        raise ValueError('Invalid wait arguments JSON') from error
    if (not isinstance(args,dict) or set(args)!={'max_wait_s','execution_note'}
            or not isinstance(args['execution_note'],str) or len(args['execution_note'])>400):
        raise ValueError('Wait only accepts max_wait_s and execution_note')
    return {'max_wait_s':_duration(args['max_wait_s']),'execution_note':args['execution_note']}


def _fixed_hold(command):
    sizes = {'torso_quaternion_wxyz':4}
    for side in ('left','right'):
        sizes.update({side+'_position':3,side+'_quaternion_wxyz':4,side+'_fingers':7})
    allowed = set(sizes)|{'stage','duration_s','velocity_xy','pelvis_height_m','facing_yaw'}
    if (not isinstance(command,dict) or set(command)-allowed
            or (set(sizes)|{'pelvis_height_m'})-set(command)):
        raise ValueError('A previous executed absolute command is required; no inferred initial targets')
    for key,size in sizes.items():
        value=command[key]
        if not isinstance(value,(list,tuple)) or len(value)!=size:
            raise ValueError('Invalid previous absolute target: '+key)
        values=[_number(v) for v in value]
        if size==4 and abs(math.hypot(*values)-1)>.001:
            raise ValueError('Previous quaternion target must be unit length')
    if not .4<=_number(command['pelvis_height_m'])<=.85:
        raise ValueError('Invalid previous pelvis target')
    if 'facing_yaw' in command and not -math.pi<=_number(command['facing_yaw'])<=math.pi:
        raise ValueError('Invalid previous facing target')
    fixed=deepcopy(command)
    fixed.update(stage='hold',duration_s=COMMAND_SECONDS,velocity_xy=[0,0])
    return fixed


def _status(callback):
    value=callback()
    if (not isinstance(value,dict) or type(value.get('terminal')) is not bool
            or not isinstance(value.get('stability'),dict) or type(value['stability'].get('ready')) is not bool
            or not isinstance(value.get('contacts'),dict)):
        raise ValueError('Measured wait status requires terminal, stability and contacts')
    for key in ('hand_object_N','robot_table_N'):
        _number(value['contacts'][key])
    if _number(value['remaining_sim_seconds'])<0:
        raise ValueError('Remaining simulation budget cannot be negative')
    return deepcopy(value)


def prepare_wait(command, max_wait_s, status=None):
    """Pure validation for callers that must commit a macro only after admission."""
    prepared={'fixed_command':_fixed_hold(command),'max_wait_s':_duration(max_wait_s)}
    if status is not None:
        _status(status)
    return prepared


def _hand_pairs(contacts):
    result=set()
    for item in contacts.get('events',contacts.get('contact_pairs',[])):
        bodies=item['bodies'];geoms=item['geoms']
        if (any('hand' in name for name in bodies)
                and not any(part in name for name in bodies+geoms for part in ('table','floor','ground'))
                # The .1 N threshold applies to aggregate hand contact, not
                # each named pair already reported by the native contact log.
                and item.get('force_N',item.get('peak_force_N',0))>0):
            result.add(tuple(sorted(zip(bodies,geoms))))
    return result


def _events(before, after, feedback, elapsed_s):
    contacts=feedback.get('command_contacts')
    if not isinstance(contacts,dict):
        return [],False
    events=[]
    if _number(contacts['peak_robot_table_N'])>0:
        events.append({'type':'robot_table_contact','peak_force_N':contacts['peak_robot_table_N']})
    intervals=contacts['hand_object_intervals']
    sample_count=contacts['sample_count']
    if (type(sample_count) is not int or sample_count!=round(elapsed_s/.005)
            or abs(_number(contacts.get('sample_dt_s'))-.005)>1e-9):
        return events,False
    active_count=sum(interval['sample_count'] for interval in intervals)
    if not 0<=active_count<=sample_count:
        return events,False
    previous=before['contacts']['hand_object_N']>HAND_CONTACT_THRESHOLD_N
    current=after['contacts']['hand_object_N']>HAND_CONTACT_THRESHOLD_N
    observed=active_count>0 or contacts['peak_hand_object_N']>HAND_CONTACT_THRESHOLD_N
    if (not previous and (observed or current)) or (previous and len(intervals)>1):
        events.append({'type':'new_hand_contact','observed_active_samples':active_count})
    if (previous and (active_count<sample_count or not current)) or (observed and (not current or len(intervals)>1)):
        events.append({'type':'lost_hand_contact','observed_active_samples':active_count,'total_samples':sample_count})
    prior_pairs=_hand_pairs(before['contacts']);observed_pairs=_hand_pairs(contacts)
    now_pairs=_hand_pairs(after['contacts'])
    identity_known='events' in before['contacts'] or 'contact_pairs' in before['contacts']
    endpoint_known='events' in after['contacts'] or 'contact_pairs' in after['contacts']
    interrupted_pair=any(tuple(sorted(zip(item['bodies'],item['geoms']))) in prior_pairs
                         and item.get('sample_count',0)<sample_count
                         for item in contacts.get('contact_pairs',[]))
    if previous and identity_known and (observed_pairs!=prior_pairs or interrupted_pair
                                        or (endpoint_known and now_pairs!=prior_pairs)):
        events.append({'type':'hand_contact_pair_changed',
                       'observed_pairs':[{'bodies':[p[0] for p in pair],'geoms':[p[1] for p in pair]} for pair in sorted(observed_pairs)]})
    return events,True


def _receipt_feedback(feedback):
    clean=deepcopy(feedback)
    if not isinstance(clean,dict):return clean
    if isinstance(clean.get('contacts'),dict):clean['contacts'].pop('events',None)
    if isinstance(clean.get('command_contacts'),dict):
        cc=clean['command_contacts']
        cc['contact_pairs']=[{k:v for k,v in item.items() if k in
            ('bodies','geoms','peak_force_N','sample_count','first_sample_s','last_sample_s')}
            for item in cc.get('contact_pairs',[])]
    for item in clean.get('executed_prefix_feedback',[]):
        item['feedback']=_receipt_feedback(item.get('feedback'))
    return clean


def execute_wait(command, max_wait_s, status, execute_once):
    """Execute fixed holds via caller-owned Session submission, never direct physics.

    status() supplies stability.ready, contacts, terminal, remaining_sim_seconds.
    execute_once(command) returns the actual StepResult. The caller owns the
    per-command Session receipts and records that these are executor_compiled_wait
    commands. No previous command means invalid wait, not an implicit hold.
    """
    prepared=prepare_wait(command,max_wait_s)
    maximum=prepared['max_wait_s'];fixed=prepared['fixed_command']
    result={'reason':None,'command_count':0,'executed_commands':0,'elapsed_s':0.,
            'receipts':[],'events':[],'fixed_command':deepcopy(fixed),
            'nonterminal_event_response_bound_s':COMMAND_SECONDS}
    def finish(reason):
        result['reason']=reason
        return result
    current=_status(status)
    for index in range(round(maximum/COMMAND_SECONDS)):
        if current['terminal']:return finish('terminal')
        if current['contacts']['robot_table_N']>0:
            result['events'].append({'type':'robot_table_contact_at_start','force_N':current['contacts']['robot_table_N']})
            return finish('robot_table_contact')
        if current['stability']['ready']:return finish('stable')
        if current['remaining_sim_seconds']<COMMAND_SECONDS-1e-9:return finish('simulation_budget')
        step=execute_once(deepcopy(fixed))
        if (not isinstance(step,StepResult) or type(step.executed) is not bool or type(step.terminal) is not bool
                or not 0<=_number(step.elapsed_s)<=COMMAND_SECONDS+1e-9
                or (step.executed and step.elapsed_s<=0) or (not step.executed and step.elapsed_s!=0)):
            raise ValueError('Wait requires actual bounded execution evidence')
        result['command_count']+=1;result['executed_commands']+=int(step.executed)
        result['elapsed_s']+=step.elapsed_s
        result['receipts'].append({'command_index':index,'executed':step.executed,'elapsed_s':step.elapsed_s,
                                  'terminal':step.terminal,'interrupt':step.interrupt,'feedback':_receipt_feedback(step.feedback)})
        after=_status(status)
        events,complete=_events(current,after,step.feedback or {},step.elapsed_s) if step.executed else ([],True)
        result['events'].extend(events)
        if step.terminal or after['terminal']:return finish('terminal')
        if not step.executed:return finish('zero_step')
        if step.interrupt:return finish('controller_interrupt')
        if not complete:return finish('feedback_incomplete')
        if any(event['type']=='robot_table_contact' for event in events):return finish('robot_table_contact')
        if events:return finish('hand_contact_changed')
        if step.elapsed_s<COMMAND_SECONDS-1e-9:return finish('partial_command')
        if after['stability']['ready']:return finish('stable')
        current=after
    return finish('timeout')
