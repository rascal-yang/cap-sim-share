"""Serial, deterministic, fully frozen batches; optional saved-state rendering.

Runs in one Python process to avoid repeated CLI startup. Each physical run still
gets its own fresh native adapter, runtime evidence, contract and reflection.
"""
import copy
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from .contracts import WorkflowError, canonical, digest, file_hash, now


def validate_manifest(m):
    if not isinstance(m,dict) or type(m.get('max_cases')) is not int or m['max_cases']<1:
        raise WorkflowError('invalid_input','Explicit batch case budget required')
    cases=m.get('cases')
    if not isinstance(cases,list) or not cases or len(cases)>m['max_cases']:raise WorkflowError('invalid_input','Batch case budget exceeded')
    if type(m.get('render',False)) is not bool:raise WorkflowError('invalid_input','Render must be boolean')
    ids=set()
    for c in cases:
        if not isinstance(c,dict) or set(c)!={'id','request'} or not isinstance(c['id'],str) or not re.fullmatch('[A-Za-z0-9_-]+',c['id']) or c['id'] in ids:
            raise WorkflowError('invalid_input','Unique safe case IDs required')
        if not isinstance(c['request'],str) or not c['request']:raise WorkflowError('invalid_input','Missing request path')
        ids.add(c['id'])
    if not isinstance(m.get('experiment_root'),str):raise WorkflowError('invalid_input','Batch output root')
    return m


def load_request(path):
    path=Path(path).resolve();request=json.loads(path.read_text())
    if request.get('decision',{}).get('mode')!='deterministic':raise WorkflowError('invalid_input','Batch forbids live/saved model decisions; deterministic only')
    request['files']={k:str((path.parent/v).resolve()) for k,v in request['files'].items()}
    for k in ('strategy_root','experiment_root'):request[k]=str((path.parent/request[k]).resolve())
    return request


def summarize_run(experiment,result):
    root=Path(experiment);audit=json.loads((root/'execution/execution_audit.json').read_text());events=json.loads((root/'execution/events.json').read_text())
    settled=[e for e in events if e['event']=='navigation_settled']
    return {'experiment_id':root.name,'experiment_path':str(root),'status':result['status'],'reason':result.get('reason_code'),
            'success':bool(result.get('physical_task_completed',False)),'sim_seconds':audit['sim_seconds'],
            'simulation_wall_s':audit['simulation_wall_s'],'online_planning_wall_s':audit['online_planning_wall_s'],
            'navigation_corrections':audit.get('navigation_corrections',0),'api_calls':audit.get('api_calls_during_execution'),
            'first_settled_error_m':settled[0]['error_m'] if settled else None,
            'final_settled_error_m':settled[-1]['error_m'] if settled else None,
            'navigation_settled_s':settled[-1]['time_s'] if settled else None}


def run_batch(manifest_path):
    from . import service
    from ..cli import create_experiment
    path=Path(manifest_path).resolve();m=validate_manifest(json.loads(path.read_text()));started=now();clock=time.monotonic()
    # Validate all requests before launching any physics, including API policy.
    cases=[(c,load_request(path.parent/c['request'])) for c in m['cases']]
    root=create_experiment((path.parent/m['experiment_root']).resolve(),'workflow_batch')
    manifest=copy.deepcopy(m);manifest.update(started_at=started,source_manifest_sha256=file_hash(path),
        requests_sha256={c['id']:file_hash((path.parent/c['request']).resolve()) for c,_ in cases},
        input_files_sha256={c['id']:service.input_context(r['files']) for c,r in cases},
        implementation_sha256={k:file_hash(v) for k,v in service.implementation_files().items()},planning_api_calls=0,runtime_model_decisions=any('grasp_trigger' in r['strategy']['policy'] for _,r in cases))
    service.write_new(root/'manifest.json',manifest);service.write_new(root/'manifest.sha256.json',{'sha256':file_hash(root/'manifest.json')})
    results=[]
    for case,request in cases:
        folder=root/case['id'];folder.mkdir();service.write_new(folder/'request.json',request)
        t=time.monotonic();case_start=now();timing={};experiment=None
        try:
            # Refuse edited inputs between batch declaration and use.
            if file_hash((path.parent/case['request']).resolve())!=manifest['requests_sha256'][case['id']]:raise WorkflowError('integrity_error','Batch request changed')
            if service.input_context(request['files'])!=manifest['input_files_sha256'][case['id']]:raise WorkflowError('integrity_error','Batch referenced inputs changed')
            if {k:file_hash(v) for k,v in service.implementation_files().items()}!=manifest['implementation_sha256']:raise WorkflowError('integrity_error','Batch implementation changed')
            for stage in ('plan','prepare','execute'):
                if service.input_context(request['files'])!=manifest['input_files_sha256'][case['id']]:raise WorkflowError('integrity_error','Batch referenced inputs changed')
                stage_start=now();ts=time.monotonic()
                if stage=='plan':planned=service.plan(request);value=planned
                elif stage=='prepare':prepared=service.prepare(planned['strategy_id'],request);experiment=Path(prepared['experiment_path']);value=prepared
                else:value=service.execute(experiment)
                timing[stage]={'started_at':stage_start,'ended_at':now(),'wall_s':time.monotonic()-ts}
                service.write_new(folder/(stage+'.json'),value)
            row=summarize_run(experiment,value)
            if not (experiment/'reflection.json').is_file():raise WorkflowError('integrity_error','Missing execution reflection')
        except Exception as exc:
            actual_calls=0
            if experiment:
                audit=experiment/'execution/execution_audit.json'
                if audit.is_file():actual_calls=json.loads(audit.read_text()).get('api_calls_during_execution',0)
                else:actual_calls=len(list((experiment/'execution/grasp_decisions').glob('*/provenance.json')))
            row={'status':'failed','success':False,'reason':exc.code if isinstance(exc,WorkflowError) else 'batch_case_error',
                 'detail':str(exc),'experiment_path':str(experiment) if experiment else None,'api_calls':actual_calls}
        row.update(case_id=case['id'],started_at=case_start,timing=timing)
        reflection={'case':case['id'],'experiment_path':row.get('experiment_path'),'recorded_at':now(),'result':row.copy(),
                    'next_action':'Continue next predeclared case with frozen policy; no GPT tuning','interpretation':'One measured case, not general reliability'}
        service.write_new(folder/'reflection.json',reflection)
        with (root/'reflections.jsonl').open('a') as log:log.write(canonical(reflection).decode()+'\n')
        if m.get('experience_log'):
            with (path.parent/m['experience_log']).resolve().open('a') as log:log.write(canonical(reflection).decode()+'\n')
        row['rendering_wall_s']=None
        if m.get('render',False) and experiment and (experiment/'execution/states.npz').exists():
            rt=time.monotonic()
            with (folder/'render.log').open('w') as log:
                process=subprocess.run([str(service.REPO/'scripts/humano_python.sh'),'-m','humanoid_mimicgen.render_standing',str(experiment)],cwd=service.REPO,stdout=log,stderr=subprocess.STDOUT)
            row.update(rendering_wall_s=time.monotonic()-rt,render_returncode=process.returncode)
        row.update(ended_at=now(),total_wall_s=time.monotonic()-t);service.write_new(folder/'result.json',row);results.append(row)
        print('BATCH '+case['id']+' '+row['status']+' '+str(row.get('reason')),file=sys.stderr,flush=True)
    fields=['case_id','experiment_id','success','reason','first_settled_error_m','final_settled_error_m','navigation_corrections','navigation_settled_s','sim_seconds','simulation_wall_s','online_planning_wall_s','rendering_wall_s','total_wall_s','api_calls']
    with (root/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    summary={'status':'completed','batch_path':str(root),'experiment_id':root.name,'started_at':started,'ended_at':now(),'total_wall_s':time.monotonic()-clock,
             'successes':sum(r['success'] for r in results),'case_count':len(results),'success_rate':sum(r['success'] for r in results)/len(results),
             'api_calls':sum((r.get('api_calls') or 0) for r in results),'render_requested':m.get('render',False),'rows':results}
    service.write_new(root/'summary.json',summary);return summary
