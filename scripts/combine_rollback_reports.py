#!/usr/bin/env python3
"""Aggregate immutable study roots without changing their original summaries."""
import argparse,json,statistics
from pathlib import Path
from report_rollback_benchmark import analyze

p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--apple',required=True);p.add_argument('--prior',nargs='*',default=[]);a=p.parse_args();main=Path(a.main);apple=Path(a.apple)
analyses=[analyze(main),analyze(apple)];ms=json.loads((main/'summary.json').read_text());aps=json.loads((apple/'summary.json').read_text());tasks=[t for t in ms['results'] if t['profile']!='apple']+aps['results'];rows=[r for x in analyses for r in x['conditions']];summaries=[r for x in analyses for r in x['summary'] if r['arms']['restart']['conditions']];timings=[]
for task in tasks:
 values=[]
 for c in task['conditions']:
  left={r['candidate_index']:r for r in c['arms']['restart']['rows'] if not r.get('shared_seed')};right={r['candidate_index']:r for r in c['arms']['rollback']['rows'] if not r.get('shared_seed')}
  for index in left.keys()&right.keys():values.append((left[index]['attempt_wall_s'],right[index]['attempt_wall_s'],left[index]['result']['simulation_wall_s'],right[index]['result']['simulation_wall_s']))
 means=[statistics.mean(v[i] for v in values) for i in range(4)]
 timings.append({'profile':task['profile'],'matched_candidates':len(values),'restart_attempt_wall_mean_s':means[0],'rollback_attempt_wall_mean_s':means[1],'attempt_wall_reduction':1-means[1]/means[0],'restart_physics_mean_s':means[2],'rollback_physics_mean_s':means[3],'physics_reduction':1-means[3]/means[2]})
roots=list(dict.fromkeys([str(main),str(apple),*a.prior]));calls={}
for root in roots:
 for path in Path(root).glob('**/api/*/attempt_*.json'):calls[str(path.resolve())]=json.loads(path.read_text())
usage={'requests':len(calls),'known_tokens':sum((v.get('usage') or {}).get('total_tokens',0) for v in calls.values()),'unknown_usage_calls':sum(v.get('usage') is None for v in calls.values()),'api_wall_sum_s':sum(v.get('wall_s') or 0 for v in calls.values()),'roots':roots,'deduplicated_by':'absolute request-attempt path'}
data={'experiment_id':main.name,'supplement_id':apple.name,'started_at':ms['started_at'],'ended_at':aps['ended_at'],'results':tasks,'summary':summaries,'conditions':rows,'timings':timings,'all_research_api_usage':usage,'videos':ms['videos']+aps['videos'],'superseded_roots':a.prior,'training_ready':False}
(main/'combined_analysis.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
lines=[f'# {main.name} / {apple.name}',f"\n正式比较起止：{ms['started_at']} → {aps['ended_at']}。这是两个实际实验的合并分析，原summary与冻结证据保持不变。",'\n结论须分别看任务成功率、首次成功时间、可完成的新尝试数和新验收成功数。回溯恢复完整状态，未改变同一候选的控制策略。', '\n每任务1个固定历史场景、3种预先固定队列顺序，每组120秒。顺序重复不是独立场景。chair/button共8候选；apple容量试运行发现队列先耗尽，故最终扩为12，原8候选及相对顺序保持不变，追加4个真实API候选，未按成败筛选。苹果此前失败与容量试运行的规划成本计入最终比较；试运行物理成本保留为研究开销。', '\n|任务|从头→回溯：成功条件数|新尝试数|新验收成功数|首次成功平均s|\n|---|---|---|---|---|']
for entry in summaries:
 x=entry['arms']['restart'];y=entry['arms']['rollback'];lines.append(f"|{entry['profile']}|{x['successful_conditions']}/{x['conditions']} → {y['successful_conditions']}/{y['conditions']}|{x['new_attempts']} → {y['new_attempts']}|{x['new_successes']} → {y['new_successes']}|{x['mean_first_success_s']:.2f} → {y['mean_first_success_s']:.2f}|")
lines+=['\n以上新尝试/成功数不含共享种子，不计截止后完成的尝试。按钮为既有button_palm_mesh_v1验收，原生按钮验收另存，不能宣称通过原生按钮标准。新验收成功并不表示每条都做过独立token回放；每任务选定1条回溯胜者做完整从头执行＋独立token/双手回放。', '\n|任务|相同候选平均整次执行s：从头→回溯|减少|平均物理计算s：从头→回溯|减少|\n|---|---|---|---|---|']
for t in timings:lines.append(f"|{t['profile']}|{t['restart_attempt_wall_mean_s']:.3f} → {t['rollback_attempt_wall_mean_s']:.3f}|{100*t['attempt_wall_reduction']:.1f}%|{t['restart_physics_mean_s']:.3f} → {t['rollback_physics_mean_s']:.3f}|{100*t['physics_reduction']:.1f}%|")
lines+=['\n时间口径：同一任务API一次生成共享队列，实际规划成本按3个顺序摊销，完整种子执行成本对双方全额计入。完整初始化、执行、证据保存和候选验收计入搜索；实现等价检查、渲染、选定胜者的最终从头/token证明单列。整次执行时间配对同一候选，包含已完成但搜索超时的诊断样本；不会把这些样本算入预算内成功。', '\n不摊销规划成本的任务成功条件数与新尝试数：']
for task in tasks:
 vals=[r for r in rows if r['profile']==task['profile']]
 for arm in ('restart','rollback'):
  subset=[r for r in vals if r['arm']==arm];lines.append(f"\n- {task['profile']} {arm}: 成功条件{sum(r['full_planning_charge_success'] for r in subset)}/{len(subset)}，新尝试{sum(r['full_planning_charge_new_attempts'] for r in subset)}。")
lines+=['\n实现：隔离的rollback_runtime保存MuJoCo积分状态、warmstart、SONIC历史、任务验收状态，并补齐腿部积分、手腕力反馈积分、手指积分及上一帧命令。双手命令与控制日志拼接完整继承前缀；manifest绑定。不同参数不能从已受该参数影响的时刻之后回溯；相同策略支持接触中恢复。身体保持decoder原始PD target，无decoder后微调。身体原token前缀、mode1阶段和独立手指时机分别记录；按钮手指可在身体mode1切换前开始调整。', '\nAPI只输出可变参数，固定字段由本地编译并验证来源；api_compiled明确区分本地组装与API原文。汇总全部候选错误，逐条指出20ms时基、向量范数和重复候选，减少反复请求。候选顺序与参数对两组完全匹配，故实验检验批量规划下的回溯效率，不代表独立实时GPT重规划的端到端收益。', '\n每任务早期和接触中回溯均要求状态/next_state/命令/控制器记录/双手逐数组一致；选定胜者要求完整从头执行相同，再独立token/双手回放。测试和物理证据都保留。原有其他生成管线未修改。', f"\n全研究API消耗（含失败预检查、失败规划、容量试运行，按请求路径去重）：{usage['requests']}次请求；已知服务端token{usage['known_tokens']}；未知用量{usage['unknown_usage_calls']}次；请求墙钟合计{usage['api_wall_sum_s']:.3f}s。未知用量不按零处理。开发与手工设计耗时未单独计量，未知。各正式批次数值/渲染/总墙钟见对应REPORT与summary。", '\n视频：']
for v in data['videos']:
 if v.get('video'):lines.append(f"\n- [{v['profile']} {v['label']}]({v['video']})")
lines += [f'\n- [初始正式实验报告]({main/"REPORT.md"})',f'\n- [苹果扩展队列报告]({apple/"REPORT.md"})','\n可回放性与视觉训练质量分开：最终视频仅做有限抽查，未完成原帧率时序质量认证，training_ready=false。不能从3个固定场景外推全部HUMANO任务成功率。']
(main/'COMPARISON_REPORT.md').write_text('\n'.join(lines));print(json.dumps({'summary':summaries,'timings':timings,'api':usage},indent=2))
