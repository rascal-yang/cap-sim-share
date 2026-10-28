"""Serial ten-task small-budget contact workflow; API rotation is batch-wide."""
import argparse,csv,json,signal,time,traceback
from pathlib import Path
from .agent import save
from .suite_lock import exclusive
from ..workflow.contracts import now


class DeadlineTransport:
    def __init__(self,transport,deadline):self.transport=transport;self.config=transport.config;self.deadline=deadline
    def respond(self,*args,**kwargs):
        remaining=min(self.config['timeout_s'],self.deadline-time.monotonic())
        if remaining<=0:raise TimeoutError('Task wall budget exhausted before API request')
        def expired(signum,frame):raise TimeoutError('API/task wall budget exhausted')
        old=signal.signal(signal.SIGALRM,expired);timer=signal.setitimer(signal.ITIMER_REAL,remaining)
        try:return self.transport.respond(*args,**kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL,*timer);signal.signal(signal.SIGALRM,old)


class BudgetProvider:
    def __init__(self,provider,deadline):self.provider=provider;self.deadline=deadline;self.config=provider.config;self.routing_sequence=None
    def attempt_providers(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('Task wall budget exhausted')
        schedule=self.provider.attempt_providers() if hasattr(self.provider,'attempt_providers') else iter([('primary',self.provider)]*6)
        while True:
            if time.monotonic()>=self.deadline:raise TimeoutError('Task wall budget exhausted')
            try:slot,transport=next(schedule)
            except StopIteration:return
            self.routing_sequence=getattr(self.provider,'routing_sequence',None)
            yield slot,DeadlineTransport(transport,self.deadline)


def failed_task(root,task,exc,started,elapsed):
    attempts=[]
    for p in sorted(root.glob('**/physics.result.json')):
        r=json.loads(p.read_text());cp=Path(r['experiment_path']);rp=p.parent/'result.json'
        if rp.exists():attempts.append(json.loads(rp.read_text()));continue
        attempts.append({'result':r,'accepted':False,'review':{'action':'unreviewed','rationale':'Task interrupted: '+str(exc)},'video':str(cp/'inspection/review_zh.mp4')})
    calls=[json.loads(p.read_text()) for p in root.glob('**/api/*/attempt_*.json')]
    status='budget_exhausted' if isinstance(exc,TimeoutError) else 'error'
    return {'task':task,'status':status,'error':type(exc).__name__+': '+str(exc),'started_at':started,'ended_at':now(),'total_wall_s':elapsed,'attempts':attempts,'physical_attempts':len(attempts),'native_successes':sum(h['result']['success'] for h in attempts),'accepted':sum(h.get('accepted',False) for h in attempts),'api_requests':len(calls),'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'unknown_usage_calls':sum(c.get('usage') is None for c in calls),'api_wall_s':sum(c.get('wall_s') or 0 for c in calls),'simulation_wall_s':sum(h['result']['simulation_wall_s'] for h in attempts),'rendering_wall_s':sum(json.loads(p.read_text())['rendering_and_inspection_wall_s'] for p in root.glob('**/inspection/inspection.json')),'traceback':traceback.format_exc()}


def run(config,provider_config):
    from .contact_review import run as task_run
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True);rows=[];started=now();tick=time.monotonic()
    with exclusive(root,'batch'):
        for spec in config['tasks']:
            tr=root/'tasks'/spec['task'];tr.mkdir(parents=True,exist_ok=True);cache=tr/'batch_result.json'
            if cache.exists():rows.append(json.loads(cache.read_text()));continue
            cfg={'output_root':str(tr),'study_id':root.name,'task':spec,'object_shift_xy_m':config['object_shift_xy_m'],'api_routing_path':str(root/'.api_rotation.json'),'max_physical_attempts':spec.get('remaining_physical_attempts',config.get('max_physical_attempts',3)),'initial_wall_budget_s':config.get('initial_wall_budget_s',900),'retry_wall_budget_s':config.get('retry_wall_budget_s',1200),'native_only':True}
            if 'task_wall_budget_s' in config:cfg['task_wall_budget_s']=config['task_wall_budget_s']
            save(tr/'config.json',cfg);ts=now();tt=time.monotonic()
            print(json.dumps({'event':'task_start','task':spec['task'],'started_at':ts}),flush=True)
            try:r=task_run(cfg,provider_config);r['task']=spec['task']
            except Exception as exc:r=failed_task(tr,spec['task'],exc,ts,time.monotonic()-tt);save(tr/'ERROR.json',r)
            r['task_observed_wall_s']=time.monotonic()-tt;r['task_root']=str(tr);save(cache,r);rows.append(r);save(root/'progress.json',rows)
            print(json.dumps({'event':'task_end','task':spec['task'],'status':r['status'],'attempts':r['physical_attempts'],'accepted':r['accepted'],'wall_s':r['task_observed_wall_s']},ensure_ascii=False),flush=True)
        summary={'experiment_id':root.name,'started_at':started,'ended_at':now(),'total_wall_s':time.monotonic()-tick,'tasks':rows,'task_count':len(rows),'native_success_tasks':sum(any(h['result']['success'] for h in r['attempts']) for r in rows),'accepted_tasks':sum(bool(r['accepted']) for r in rows),'first_attempt_accepted_tasks':sum(bool(r['attempts'] and r['attempts'][0].get('accepted')) for r in rows),'retried_tasks':sum(r['physical_attempts']>1 for r in rows),'retry_recovered_tasks':sum(r['physical_attempts']>1 and not r['attempts'][0].get('accepted') and bool(r['accepted']) for r in rows),'physical_attempts':sum(r['physical_attempts'] for r in rows),'known_tokens':sum(r['known_tokens'] for r in rows),'api_requests':sum(r['api_requests'] for r in rows),'unknown_usage_calls':sum(r['unknown_usage_calls'] for r in rows),'api_wall_s':sum(r['api_wall_s'] for r in rows),'simulation_wall_s':sum(r['simulation_wall_s'] for r in rows),'rendering_wall_s':sum(r['rendering_wall_s'] for r in rows),'training_ready':False,'object_shift_xy_m':config['object_shift_xy_m'],'max_physical_attempts_per_task':config.get('max_physical_attempts',3),'task_wall_budget_s':config.get('task_wall_budget_s',config.get('initial_wall_budget_s',900)+config.get('retry_wall_budget_s',1200)*(config.get('max_physical_attempts',3)-1)),'native_criteria_unchanged':True}
        save(root/'summary.json',summary)
        table=[];lines=[f'# {root.name}',f"{started} — {summary['ended_at']}",'','十任务独立 API 验收审查与接触控制测试；统一物体 X+5mm。原生标准未改；每任务最多3次物理、每次尾部最多3秒，首次阶段默认900秒，每次retry默认1200秒，总预算见配置（仅允许已开始的本地执行/导出收尾）。成功仅代表各一个微扰条件，不是稳健成功率。','原 token 前缀 + SONIC mode1 + 显式关节参考/PD/力反馈（按实际API选择）。访问成功参考指令，属于轨迹重定向；token单独不能复现。','','|任务|物理尝试|原生成功|执行验收通过|墙钟秒|token|状态|','|---|---:|---:|---:|---:|---:|---|']
        for r in rows:
            v={'task':r['task'],'attempts':r['physical_attempts'],'native_successes':r['native_successes'],'accepted':r['accepted'],'wall_s':r['task_observed_wall_s'],'tokens':r['known_tokens'],'status':r['status'],'root':r['task_root']};table.append(v)
            lines.append(f"|{r['task']}|{r['physical_attempts']}|{r['native_successes']}|{r['accepted']}|{r['task_observed_wall_s']:.1f}|{r['known_tokens']}|{r['status']}|")
        with (root/'acceptance.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        lines+=['',json.dumps({k:v for k,v in summary.items() if k!='tasks'},ensure_ascii=False,indent=2)]
        (root/'REPORT.md').write_text('\n'.join(lines));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--provider',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()),json.loads(Path(a.provider).read_text()))
