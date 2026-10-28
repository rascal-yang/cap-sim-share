"""Independent API task sessions in separate processes; physical execution remains locked."""
import argparse,json,multiprocessing,time,os
from uuid import uuid4
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from . import quality_suite as q
from .suite_provider import configured_provider
from .agent import save
from .suite_lock import exclusive
from ..workflow.contracts import now


def worker(task,cfg,provider_path):
    folder=Path(cfg['output_root'])/task
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    try:return q.run_task(Path(cfg['dataset_root'])/task,folder,configured_provider(json.loads(Path(provider_path).read_text()),Path(cfg['output_root'])/'.api_rotation.json'),cfg['episodes'],cfg.get('attempts_per_stage',3))
    except Exception as exc:
        folder.mkdir(exist_ok=True);previous=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else {}
        r={'task':task,'status':'error','reason':type(exc).__name__+': '+str(exc)[:350],'cases':previous.get('cases',[]),'stages':previous.get('stages',{}),'ended_at':now()};save(folder/'result.json',r);return r


def resume_service_errors(root,tasks):
    resumed=[]
    for task in tasks:
        path=Path(root)/task/'result.json'
        if not path.exists():continue
        result=json.loads(path.read_text())
        reason=result.get('reason','')
        allowed=('WorkflowError: api_unavailable:', 'WorkflowError: invalid_api_response:', 'RuntimeError: Incomplete API response')
        if result.get('status')!='error' or not reason.startswith(allowed):continue
        suffix=now().replace(':','').replace('+','_')+'_'+uuid4().hex[:8]
        path.rename(path.with_name('result.service_error_'+suffix+'.json'))
        resumed.append(task)
    return resumed


def _run(config,provider,jobs):
    if jobs not in (1,2,3):raise ValueError('API concurrency outside 1..3 budget')
    cfg=json.loads(Path(config).read_text());root=Path(cfg['output_root']);started=now();tick=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,t,cfg,provider) for t in cfg['tasks']]
        for f in as_completed(futures):
            r=f.result();rows.append(r);save(root/'batch_progress.json',rows);print(json.dumps({'task':r['task'],'status':r['status']},ensure_ascii=False),flush=True)
    q.run(config,provider)
    save(root/'batch_timing.json',{'started_at':started,'ended_at':now(),'wall_s':time.monotonic()-tick,'api_task_processes':jobs,'physics_parallelism':1})


def run(config,provider,jobs,resume_errors=False):
    root=json.loads(Path(config).read_text())['output_root']
    with exclusive(root,'study'):
        if resume_errors:
            suffix=now().replace(':','').replace('+','_')+'_'+uuid4().hex[:8]
            for name in ('result.json','REPORT.md','batch_timing.json','batch_progress.json'):
                path=Path(root)/name
                if path.exists():path.rename(path.with_name(path.stem+'.previous_'+suffix+path.suffix))
            resumed=resume_service_errors(root,json.loads(Path(config).read_text())['tasks'])
            save(Path(root)/('resume_'+suffix+'.json'),{'started_at':now(),'tasks':resumed,'api_routing':os.environ.get('QUALITY_SUITE_API_ROUTING','alternate'),'retry_policy':'alternate: initial plus five retries total; failover: initial plus five retries per credential'})
        return _run(config,provider,jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--provider',required=True);p.add_argument('--jobs',type=int,default=2);p.add_argument('--resume-errors',action='store_true');a=p.parse_args();run(a.config,a.provider,a.jobs,a.resume_errors)
