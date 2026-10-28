#!/usr/bin/env python3
"""Read-only aggregation, including deadline misses and API usage uncertainty."""
import argparse,json,statistics
from datetime import datetime
from pathlib import Path
from generalization_records import roots as study_roots, conditions as study_conditions
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);config=json.loads((root/'config.json').read_text())
rows=study_conditions(root)
labels={'plate':'拿盘子','chair':'坐椅子','button':'按按钮（掌面标准）','apple':'拿苹果'}
report=[f'# {root.name} 四任务位置泛化实验','', '本实验为原轨迹重定向＋API候选策略规划。保留原始token接近前缀；没有证明从零规划新动作。机器人初态不动，仅目标物体平移。', '', '两组各120秒，计入该位置全部API规划与共同初次执行时间。首次预算内成功后停止；独立复核、渲染不计入操作预算。成功必须通过完整重执行、token＋最终手部命令回放，并核对每条成功分支。', '', '|任务|已完成条件|从头重试有效成功/10|回溯有效成功/10|从头首次成功均时|回溯首次成功均时|额外重试次数（从头/回溯）|', '|---|---:|---:|---:|---:|---:|---:|']
tables={}
for label,name in labels.items():
 rs=[r for r in rows if r['task_label']==label];record={}
 for arm in ('restart','rollback'):
  ars=[r['arms'][arm] for r in rs if arm in r['arms']];times=[ar['metrics']['first_success_s'] for ar in ars if ar.get('verified_success')]
  record[arm]={'verified_conditions':sum(ar.get('verified_success',False) for ar in ars),'provisional_conditions':sum(ar['metrics']['success'] for ar in ars),'mean_first_success_s':statistics.mean(times) if times else None,'retries':sum(ar['retry_count'] for ar in ars),'in_budget_retry_successes':sum(ar.get('verified_success',False) and ar['retry_count']>0 for ar in ars)}
 record['physical_verified_conditions']=sum(any(v['passed'] for v in r['verification'].values()) for r in rs)
 record['initial_failure_conditions']=sum(bool(r.get('seed')) and not r['seed']['success'] for r in rs)
 for arm in ('restart','rollback'):
  record[arm]['recovered_conditions']=sum(bool(r.get('seed')) and not r['seed']['success'] and r.get('arms',{}).get(arm,{}).get('verified_success',False) for r in rs)
  record[arm]['executed_retries_including_overrun']=sum(max(0,len(r.get('arms',{}).get(arm,{}).get('rows',[]))-1) for r in rs)
 tables[label]=record
 fmt=lambda v:'—' if v is None else f'{v:.2f}s'
 report.append(f"|{name}|{len(rs)}|{record['restart']['verified_conditions']}|{record['rollback']['verified_conditions']}|{fmt(record['restart']['mean_first_success_s'])}|{fmt(record['rollback']['mean_first_success_s'])}|{record['restart']['retries']}/{record['rollback']['retries']}|")
report+=['','恢复率按“初次失败、后来预算内复核成功的条件数 / 初次失败条件数”统计，详见 ANALYSIS.json。预算内失败与物理不可行分开；超时成功仍做回放但不计预算内成功。','',*['- '+labels[k]+f"：物理复核成功 {v['physical_verified_conditions']}/10；初次失败 {v['initial_failure_conditions']}；成功恢复（从头/回溯）{v['restart']['recovered_conditions']}/{v['rollback']['recovered_conditions']}。" for k,v in tables.items()],'','## 逐位置结果','','|条件|相对原数据目标平移XY（m）|状态|初次成功|从头有效|回溯有效|完整/token复核|错误|','|---|---|---|---|---|---|---|---|']
for r in rows:
 vals=[r.get('arms',{}).get(arm,{}).get('verified_success') for arm in ('restart','rollback')]
 report.append(f"|{r['condition_id']}|{r['shift']}|{r['status']}|{r.get('seed',{}).get('success')}|{vals[0]}|{vals[1]}|{sum(v['passed'] for v in r['verification'].values())}/{len(r['verification'])}|{r.get('error','')}|")
roots=study_roots(root)+([Path(config['recovery_of'])] if config.get('recovery_of') else [])
for r in study_roots(root):
 c=json.loads((r/'config.json').read_text())
 if c.get('setup_aborted_root'):roots.append(Path(c['setup_aborted_root']))
roots=list(dict.fromkeys(roots))
usage=[]
for r in roots:
 for p in list(r.glob('**/api/*/attempt_*.json'))+list(r.glob('network_probe_*.json')):
  c=json.loads(p.read_text());usage.append({'path':str(p),'usage':c.get('usage'),'wall_s':c.get('wall_s'),'status':c.get('status')})
known=sum((c['usage'] or {}).get('total_tokens',0) for c in usage);unknown=sum(c['usage'] is None for c in usage)
physical=[json.loads(p.read_text()) for r in study_roots(root) for p in r.glob('**/experiments/*/result.json')]
summary={'conditions_done':len(rows),'tables':tables,'api_requests_with_attempt_records':len(usage),'known_tokens':known,'unknown_usage_records':unknown,'planning_wall_s':sum(r.get('planning_wall_s',0) for r in rows),'physical_runs':len(physical),'simulation_wall_s':sum(r.get('simulation_wall_s',0) or 0 for r in physical),'physical_total_wall_s':sum(r.get('total_wall_s',0) or 0 for r in physical),'training_ready':False}
starts=[json.loads((r/'run_started.json').read_text())['started_at'] for r in study_roots(root) if (r/'run_started.json').exists()]
ends=[r['ended_at'] for r in rows]
summary.update(started_at=min(starts) if starts else None,physical_conditions_ended_at=max(ends) if ends else None,api_attempt_wall_s=sum(c['wall_s'] for c in usage if isinstance(c['wall_s'],(int,float))),api_attempt_unknown_wall_records=sum(c['wall_s'] is None for c in usage),api_error_records=sum(c['status']=='error' for c in usage),executed_physics_s=sum(r.get('executed_physics_steps',r['physics_steps'])*.005 for r in physical))
if starts and ends:summary['elapsed_to_last_condition_s']=(datetime.fromisoformat(max(ends))-datetime.fromisoformat(min(starts))).total_seconds()
report+=['','## 耗时与用量','', '接口中断后的补跑使用相同冻结位置和规则，在连通性恢复后重新计120秒；故障期间API成本单列保留，因此这不是包含服务中断的端到端120秒保证。补跑按其条件列表重新交替执行两组顺序。',f"开始：{summary['started_at']}；最后条件结束：{summary['physical_conditions_ended_at']}。",f"规划累计：{summary['planning_wall_s']:.2f}s；物理执行累计（含复核）：{summary['physical_total_wall_s']:.2f}s，其中仿真步进 {summary['simulation_wall_s']:.2f}s。",f"API请求记录 {len(usage)} 条；已知token {known}；用量未知 {unknown} 条。包括前置失败批次；中断请求若无完整记录，不能据此视为零成本。",'', '按钮的原生验收与用户批准的 button_palm_mesh_v1 分开：表格使用后者，原生值保留在各trial中。任务物理通过不代表视频已可作为训练RGB。']
(root/'ANALYSIS.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False));(root/'REPORT.md').write_text('\n'.join(report)+'\n');print(json.dumps(summary,ensure_ascii=False))
