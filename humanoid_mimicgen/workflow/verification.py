"""Independent measured-trace checks; successful process exit is not task success."""
import json
from pathlib import Path
import numpy as np
from .contracts import WorkflowError, canonical, now
from .service import check_contract, write_new


def checked_retry_groups(commands,count,budget):
    errors=[]
    if type(count) is not int or not 0<=count<=budget:
        errors.append('retry_budget_mismatch');count=0
    groups=[[] for _ in range(count+1)];previous=0
    for command in commands:
        retry=command.get('retry_index',0)
        if type(retry) is not int or not 0<=retry<=count:
            errors.append('unknown_retry_index');continue
        if retry<previous or retry>previous+1:errors.append('retry_order_mismatch')
        previous=retry
        if command['source_index']>=0:groups[retry].append(command)
    return groups,sorted(set(errors))


def verify(experiment):
    root=Path(experiment)
    if (root/'result.json').exists():return json.loads((root/'result.json').read_text())
    c=check_contract(root/'experiment.json');out=root/'execution';s=c['workflow']['strategy'];policy=s['policy']
    a=json.loads((out/'execution_audit.json').read_text());obs=json.loads((out/'observations.json').read_text());commands=json.loads((out/'commands.json').read_text())
    states=np.load(out/'states.npz')['samples'];targets=np.load(c['inputs']['targets']);dt=c['workflow']['alignment']['command_dt_s']
    errors=[]
    if len(states)!=len(obs) or len(obs)!=len(commands) or len(obs)!=a['physics_steps']:errors.append('incomplete_measurements')
    if not np.isfinite(states).all():errors.append('nonfinite_state')
    if any(x['mode']!=1 for x in commands) or a['original_token_prefix_count'] or a['mode0_calls'] or a['midrun_state_resets'] or a['fallback']:
        errors.append('actual_control_interface_mismatch')
    contact=[x for x in commands if x['source_index']>=0]
    retry_count=a.get('recovery_count',0)
    groups,retry_errors=checked_retry_groups(commands,retry_count,policy.get('recovery',{}).get('max_retries',0));errors.extend(retry_errors)
    for group in groups:
        indices=[x['source_index'] for x in group]
        expected=list(range(s['phases'][0]['start'],s['phases'][0]['start']+len(group)))
        if indices!=expected or any(not 0<=x['source_index']<len(targets['fingers']) or not np.array_equal(x['fingers_rad'],targets['fingers'][x['source_index']]) for x in group):errors.append('source_command_mismatch')
    if policy.get('grasp_trigger'):
        from .grasp_trigger import validate_judgment,validate_approval,trigger_targets
        from .contracts import digest
        decisions=sorted((out/'grasp_decisions').glob('*/provenance.json'))
        if len(decisions)!=a.get('api_calls_during_execution'):errors.append('grasp_judgment_count_mismatch')
        for path in decisions:
            proof=json.loads(path.read_text());folder=path.parent;evidence=json.loads((folder/'evidence.json').read_text())
            if proof['evidence_sha256']!=digest(evidence):errors.append('grasp_judgment_evidence_mismatch')
            if proof['status']=='completed':
                try:validate_judgment(json.loads((folder/'response.json').read_text()),evidence)
                except WorkflowError:errors.append('invalid_grasp_judgment')
        if len(decisions)>policy['grasp_trigger']['max_calls']:errors.append('grasp_judgment_budget_exceeded')
        events=json.loads((out/'events.json').read_text());tracking=json.loads((out/'tracking.json').read_text())
        for retry,group in enumerate(groups):
            if not group:continue
            try:
                event=next(e for e in events if e['event']=='contact_started' and e['retry_index']==retry)
                decision_event=next(e for e in events if e['event']=='grasp_judgment' and e['retry_index']==retry and e['time_s']==event['time_s'])
                folder=out/'grasp_decisions'/str(decision_event['query_index'])
                proof=json.loads((folder/'provenance.json').read_text());response=json.loads((folder/'response.json').read_text());evidence=json.loads((folder/'evidence.json').read_text())
                validate_approval(proof,response,evidence,event,policy['grasp_trigger']['model'])
                if response!=decision_event['decision'] or group[0]['time_s']!=event['time_s']:raise ValueError('Decision execution time mismatch')
                prior=dict(np.load(out/('gpt_pretrigger_targets_'+str(retry)+'.npz')));actual={k:np.asarray(v) for k,v in event['wrist_hold'].items()}
                for hand in s['bindings']['hands']:
                    if not np.array_equal(actual[hand],evidence['measured'][hand+'_pose']):raise ValueError('Hold pose differs from judgment')
                expected=trigger_targets(prior,actual,s['bindings']['hands'],start=s['phases'][0]['start'],lift=s['phases'][0]['end'],end=s['phases'][-1]['end'],offset=s['command_pose_offset'])
                saved=dict(np.load(out/('gpt_trigger_targets_'+str(retry)+'.npz')))
                if any(not np.array_equal(v,saved[k]) for k,v in expected.items()):raise ValueError('Grasp target artifact mismatch')
                for row in tracking:
                    if row['retry_index']==retry and row['source_index']>=0:
                        for hand in s['bindings']['hands']:
                            if not np.array_equal(row[hand]['target_pose'],saved[hand+'_pose'][row['source_index']+s['command_pose_offset']]):raise ValueError('Executed grasp target mismatch')
            except (WorkflowError,KeyError,ValueError,StopIteration,OSError):errors.append('unapproved_or_mismatched_grasp_execution')
    indices=[x['source_index'] for x in groups[-1]]
    if a['completed'] and (not indices or indices[-1]!=s['phases'][-1]['end']-1):errors.append('truncated_completion')
    # A null measurement is unknown, never evidence of successful contact.
    from .execution import finite_record
    def known(value):
        if value is None:return False
        if isinstance(value,dict):return all(known(v) for v in value.values())
        if isinstance(value,list):return all(known(v) for v in value)
        return True
    usable=[o for o in obs if known({k:o[k] for k in ('clearance_m','contacts','root_pose')})]
    if len(usable)!=len(obs):errors.append('unknown_measurements')
    held_names={p['id'] for p in s['phases'] if p['requires_clearance']};held=[o for o in usable if o['stage'] in held_names and o.get('retry_index',0)==retry_count]
    valid=[o for o in held if o['clearance_m']>=policy['acceptance']['clearance_m'] and o['contacts']['hand_object_N']>policy['gates']['contact_force_N'] and o['contacts']['support_N']<policy['acceptance']['support_force_N']]
    consecutive=longest=0.;valid_times={o['time_s'] for o in valid}
    for o in obs:
        consecutive=consecutive+dt if o['time_s'] in valid_times else 0.
        longest=max(longest,consecutive)
    drift=max((float(np.linalg.norm(np.array(o['root_pose'][:2])-held[0]['root_pose'][:2])) for o in held),default=None)
    physical=bool(not errors and not a['failure'] and a['completed'] and len(valid)==len(held) and longest>=policy['acceptance']['hold_duration_s'] and drift is not None and drift<=policy['acceptance']['root_drift_m'])
    from cap_sim.backends.humano import Backend
    from ..privileged import no_physics
    b=Backend();native=None
    try:
        if len(states) and np.isfinite(states).all():
            with no_physics():b.describe(json.loads(Path(c['inputs']['task']).read_text()));native=b.verify(out)
    except Exception as exc:
        errors.append('native_verifier_error:'+type(exc).__name__);physical=False
    finally:b.close()
    result={'schema_version':1,'experiment_id':root.name,'strategy_id':c['workflow']['strategy_id'],'status':'completed' if physical else 'failed',
        'stage':a['stage'],'reason_code':a['failure'] or (None if physical else 'verification_failed'),'physical_task_completed':physical,
        'native':native,'recovery_count':retry_count,'source_commands_per_attempt':[len(g) for g in groups],'source_commands_verified':len(contact),'all_source_commands_completed':a['completed'] and not bool(policy.get('grasp_trigger')),'all_source_finger_commands_completed':a['completed'],'grasp_execution':'gpt_triggered_measured_wrist_hold_relative_lift' if policy.get('grasp_trigger') else 'demonstration_retargeting',
        'verification_errors':errors,'hold_continuous_s':longest,'hold_valid_s':len(valid)*dt,'hold_total_s':len(held)*dt,'hold_min_clearance_m':min((x['clearance_m'] for x in held),default=None),
        'max_clearance_m':max((x['clearance_m'] for x in usable),default=None),'hold_root_drift_m':drift,
        'max_obstacle_force_N':max((x['contacts']['robot_obstacle_N'] for x in usable),default=None),
        'actual_modes':a['actual_encoder_modes'],'training_ready':False,'verified_at':now(),'physics_steps_in_verifier':0,
        'evidence':{'audit':'execution/execution_audit.json','commands':'execution/commands.json','observations':'execution/observations.json','video':None}}
    write_new(root/'result.json',result);write_new(out/'validation.json',result)
    reflection={'experiment_id':root.name,'strategy_id':c['workflow']['strategy_id'],'recorded_at':now(),'status':'observed',
                'result':result,'next_action':'Review measured tracking/contact evidence before creating a new strategy revision; no automatic parameter changes'}
    write_new(root/'reflection.json',reflection)
    with (root.parent/'workflow_attempts.jsonl').open('ab') as f:f.write(canonical(reflection)+b'\n')
    request=json.loads((root/'request.json').read_text());index=Path(request['strategy_root'])/'executions'/c['workflow']['strategy_id']
    write_new(index/(root.name+'.json'),{'experiment_id':root.name,'result_sha256':__import__('hashlib').sha256(canonical(result)).hexdigest(),'result':result})
    (root/'REPORT.md').write_text(f"# {root.name}\n\n开始 {a['started_at']}；结束 {a['ended_at']}。\n\n独立 workflow 实际控制执行；源动作可访问，属于轨迹重定向。\n\n结果：{json.dumps(result,ensure_ascii=False,indent=2)}\n\n仿真循环 {a['simulation_wall_s']:.3f}s，其中在线规划 {a['online_planning_wall_s']:.3f}s；模拟 {a['sim_seconds']:.3f}s。渲染与未测耗时 unknown。所有控制、物理、技能/代码摘要和阶段条件见冻结 experiment.json。\n")
    return result
