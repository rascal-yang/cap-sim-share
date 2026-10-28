"""Aggregate all-task evidence without treating native success as API acceptance."""
import argparse,json
from datetime import datetime
from pathlib import Path
from .agent import save
from ..workflow.contracts import file_hash,now


def case_grade(case,verdict):
    if not case['success']:return 'physical_failure'
    if verdict=='accept':return 'api_accepted_physical'
    if verdict is None:return 'awaiting_api_review'
    return 'api_rejected'


def read_context(path):
    req=json.loads(Path(path).read_text());content=req['input'][0]['content']
    if isinstance(content,list):content=next(c['text'] for c in content if c['type']=='input_text')
    return json.loads(content)


def reviewed_result(context):
    stage=context.get('stage','')
    if stage not in ('replay','zero','augmentation','supplementary_zero_review','supplementary_augmentation_review'):return None
    e=context.get('evidence',context)
    return e.get('result') or e.get('physical_result')


def export(root):
    root=Path(root);cfg=json.loads((root/'config.json').read_text());decisions=[];reviews={}
    for p in root.rglob('decision.json'):
        d=json.loads(p.read_text())
        if 'origin' not in d:continue
        try:ctx=read_context(d['origin']['request'])
        except (KeyError,ValueError,StopIteration):continue
        r=reviewed_result(ctx)
        if r and 'experiment_id' in r:
            key=r['experiment_id'];date=d.get('created_at','')
            if key not in reviews or date>reviews[key]['date']:reviews[key]={'date':date,'verdict':d['decision']['action'],'path':str(p)}
        decisions.append({'path':str(p),'stage':ctx.get('stage'),'decision':d['decision'],'created_at':d.get('created_at')})
    cases=[]
    for p in sorted(root.rglob('result.json')):
        c=json.loads(p.read_text())
        if 'experiment_path' not in c or 'mode' not in c:continue
        folder=p.parent;review=reviews.get(c['experiment_id'],{});grade=case_grade(c,review.get('verdict'))
        cases.append({'task':c['task'],'experiment_id':c['experiment_id'],'mode':c['mode'],'object_delta_xy_m':c['object_delta_xy_m'],'grade':grade,'api_review':review.get('path'),'success':c['success'],'training_ready':False,'result':c,'artifacts':{str(f):file_hash(f) for f in (folder/'experiment.json',folder/'execution/trajectory.npz',folder/'execution/controller.npz',p)}})
    attempts=[json.loads(p.read_text())|{'path':str(p)} for p in root.rglob('attempt_*.json')]
    render=[json.loads(p.read_text()) for p in root.rglob('inspection.json')]
    usage={k:sum((a.get('usage') or {}).get(k,0) for a in attempts) for k in ('input_tokens','output_tokens','total_tokens')}
    tasks=[]
    for name in cfg['tasks']:
        cs=[c for c in cases if c['task']==name];replay=[c for c in cs if c['mode']=='original_token'];zero=[c for c in cs if c['mode']=='cartesian_source_legs' and not any(c['object_delta_xy_m'])];aug=[c for c in cs if any(c['object_delta_xy_m'])]
        state=json.loads((root/name/'result.json').read_text()) if (root/name/'result.json').exists() else {'status':'running','reason':'pending','stages':{}}
        resumed=root/name/'supplementary_augmentation/result.json'
        if resumed.exists():state=state|{'status':'supplement_'+json.loads(resumed.read_text())['status']}
        def rate(xs):return {'passed':sum(x['success'] for x in xs),'attempts':len(xs),'api_accepted':sum(x['grade']=='api_accepted_physical' for x in xs)}
        recovery=[]
        for label,xs in [('replay',replay),('zero',zero)]:
            if xs and not xs[0]['success']:
                same=[x for x in xs[1:] if x['result']['episode']==xs[0]['result']['episode']]
                recovery.append({'stage':label,'initial_failure':True,'same_source_recovery_attempts':len(same),'recovered':any(x['success'] for x in same),'alternative_source_attempts':sum(x['result']['episode']!=xs[0]['result']['episode'] for x in xs[1:])})
        tasks.append({'task':name,'status':state['status'],'stages':state.get('stages',{}),'reason':state.get('reason'),'replay':rate(replay),'zero':rate(zero),'augmentation':rate(aug),'recovery':recovery})
    start=min((a['started_at'] for a in attempts),default=now());end=now()
    summary={'experiment_id':root.name,'started_at':start,'exported_at':end,'elapsed_since_first_api_s':(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds(),'tasks':tasks,'api_calls':len(attempts),'known_usage':usage,'unknown_usage_attempts':sum(a.get('usage') is None for a in attempts),'api_wait_summed_s':sum(a.get('wall_s') or 0 for a in attempts),'simulation_wall_s':sum(c['result']['simulation_wall_s'] for c in cases),'preparation_wall_s':sum(c['result']['preparation_wall_s'] for c in cases),'verification_wall_s':sum(c['result']['verification_wall_s'] for c in cases),'rendering_and_inspection_wall_s':sum(r['rendering_and_inspection_wall_s'] for r in render),'planning_implementation_wall_s':None,'physics_cases':len(cases),'training_ready':False,'parallel_timing_note':'API task waits may overlap; do not add them to derive elapsed wall time.'}
    save(root/'quality_summary.json',summary);save(root/'dataset_index.json',{'cases':cases,'original_actions_visible':True,'generation_type':'trajectory retargeting','state_action_alignment':'pre-state, torque action, next-state at 200Hz; source/encoded motion tokens in controller.npz at50Hz','training_ready':False});save(root/'strategy_index.json',decisions)
    lines=[f'# {root.name}',f'开始（首个 API 请求）：{start}；本报告导出：{end}','','执行原始轨迹回放、API 阶段验收和原轨迹重定向。原动作可见，不将成功等同于任务策略迁移。失败策略由真实 API 提出，本地执行器验证参数和阶段依赖。','','|任务|原token物理成功/尝试|零扰动物理成功/尝试|偏移物理成功/尝试|偏移API接受|状态|','|---|---|---|---|---|---|']
    for t in tasks:
        def fmt(k):x=t[k];return f'{x["passed"]}/{x["attempts"]}'
        lines.append(f'|{t["task"]}|{fmt("replay")}|{fmt("zero")}|{fmt("augmentation")}|{t["augmentation"]["api_accepted"]}|{t["status"]}|')
    lines+=['',f'API 请求 {len(attempts)}；已知 token {usage}；用量未知请求 {summary["unknown_usage_attempts"]}。',f'累计 API 等待 {summary["api_wait_summed_s"]:.2f}s（含并行重叠），物理计算 {summary["simulation_wall_s"]:.2f}s，准备 {summary["preparation_wall_s"]:.2f}s，独立验证 {summary["verification_wall_s"]:.2f}s，已执行渲染/检查 {summary["rendering_and_inspection_wall_s"]:.2f}s。从首个请求至本次导出 {summary["elapsed_since_first_api_s"]:.2f}s；代码设计时间未知。','','`dataset_index.json` 区分物理失败、等待 API 验收、API 拒绝、API 接受。所有样本 training_ready=false，训练就绪还需视觉/导出验收。没有执行的阶段记为阻塞或未测试，不计入该阶段尝试分母。','`strategy_index.json` 保存各阶段 API 决策路径；每个物理 experiment.json 绑定 API 请求/响应/决策哈希。历史 source_snapshot 用于验证旧版代码，不能用后来修改的当前工作区冒充历史源码。','','恢复统计：']
    for t in tasks:
        for x in t['recovery']:lines.append(f'- {t["task"]} / {x["stage"]}: 同源修复尝试 {x["same_source_recovery_attempts"]}，恢复成功 {x["recovered"]}，另换源 {x["alternative_source_attempts"]}。')
    (root/'QUALITY_REPORT.md').write_text('\n'.join(lines)+'\n');return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();r=export(a.root);print(json.dumps({k:v for k,v in r.items() if k!='tasks'},ensure_ascii=False))
