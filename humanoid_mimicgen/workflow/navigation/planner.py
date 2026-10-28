"""Bounded, deterministic candidate enumeration; task semantics live in data."""
import time
import numpy as np
from ..contracts import WorkflowError, positive
from .prior import finite, rotation2, transfer_stance, wrap


def plan_stances(prior, scene, current, options, evaluate, *, clock=time.monotonic):
    started=clock()
    result={'schema_version':1,'kind':'replay_navigation_plan','status':prior['status'],
            'selected':None,'candidates':[],'physics_steps':0,'api_calls':0,
            'dynamic_execution_verified':False,'original_sequences_accessible':True,
            'options':options,'scene':scene,'prior':prior}
    if prior['status']!='completed':return result
    current=finite(current,(3,),'Current xyyaw')
    frame=finite(scene['task_frame_xyzyaw'],(4,),'Scene task frame')
    count=options['max_candidates']
    if type(count) is not int or count<1:raise WorkflowError('invalid_input','Candidate budget')
    positive(options['planning_timeout_s'],'Planning timeout')
    nominal=np.asarray(transfer_stance(prior,frame));candidates=[{'id':'current','xyyaw':current.tolist()}]
    offsets=options['offsets_xyyaw']
    if not offsets:raise WorkflowError('invalid_input','Empty candidate offsets')
    for i,value in enumerate(offsets):
        delta=finite(value,(3,),'Candidate offset')
        pose=nominal+np.r_[rotation2(frame[3])@delta[:2],delta[2]];pose[2]=wrap(pose[2])
        candidates.append({'id':f'prior_{i:03d}','xyyaw':pose.tolist()})
    exhausted=False
    for candidate in candidates[:count]:
        if clock()-started>=options['planning_timeout_s']:
            exhausted=True;break
        audit=evaluate(candidate)
        # The evaluator may reject but may not silently rewrite the target.
        row={**audit,**candidate}
        if not row.get('operation_checked',False) and (row.get('feasible') is True or not row.get('reason')):
            row.update(feasible=False,reason='operation_unchecked')
        if clock()-started>=options['planning_timeout_s']:
            row.update(feasible=False,reason='planning_budget_exhausted');exhausted=True
        result['candidates'].append(row)
        if exhausted:break
        if candidate['id']=='current' and row.get('feasible') is True:
            result['selected']=row;break
    valid=[r for r in result['candidates'] if r.get('feasible') is True]
    if result['selected'] is None and valid:
        result['selected']=min(valid,key=lambda r:(r['travel_m'],abs(float(wrap(r['xyyaw'][2]-current[2]))),
                         float(np.linalg.norm(np.asarray(r['xyyaw'][:2])-nominal[:2])),r['id']))
    result['status']='completed' if result['selected'] else 'planning_budget_exhausted' if exhausted else 'no_feasible_stance'
    result['planning_wall_s']=clock()-started
    result['candidate_limit_reached']=len(candidates)>count
    return result
