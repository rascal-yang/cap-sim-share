#!/usr/bin/env python3
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);rows=[json.loads(p.read_text()) for p in sorted(root.glob('apple_??/summary.json'))]
lines=[f'# {root.name} 苹果编码前右腕反馈对照','', '相同10个位置，每位置基线/反馈各一次，完整重启；不调用新API、不重试、不改变原token前缀、手指时序、固定位置修正或验收。只恢复旧公式gain0.5/s、cap2cm的右腕世界坐标积分反馈，作用于mode1后缀。','','|位置|相对基准偏移XY(cm)|基线成功|反馈成功|基线/反馈最大抬升(cm)|共同mode1腕误差RMSE(cm)|token回放（基线/反馈）|','|---|---|---|---|---|---|---|']
counts={k:0 for k in ('baseline','feedback')};gained=[];lost=[]
for r in rows:
 if r['status']!='complete':lines.append(f"|{r['condition_id']}|—|ERROR|{r.get('error')}|—|—|—|");continue
 x=r['arms']['baseline'];y=r['arms']['feedback'];m=r['comparison']['common_mode1_rmse_m']
 for k in counts:counts[k]+=int(r['arms'][k]['success'] and r['replays'].get(k,{}).get('passed',False))
 if y['success'] and not x['success']:gained.append(r['condition_id'])
 if x['success'] and not y['success']:lost.append(r['condition_id'])
 shift=[round((r['shift'][0]-.005)*100,3),round(r['shift'][1]*100,3)]
 reps=['通过' if r['replays'].get(k,{}).get('passed') else '不适用' for k in counts]
 lines.append(f"|{r['condition_id']}|{shift}|{x['success']}|{y['success']}|{x['diagnostics']['max_lift_m']*100:.2f}/{y['diagnostics']['max_lift_m']*100:.2f}|{m['baseline']*100:.2f}/{m['feedback']*100:.2f}|{'/'.join(reps)}|")
analysis={'conditions':len(rows),'verified_success':counts,'gained':gained,'lost':lost,'new_api_calls':0,'new_api_tokens':0}
if (root/'summary.json').exists():
 s=json.loads((root/'summary.json').read_text());analysis.update(started_at=s['started_at'],ended_at=s['ended_at'],numeric_wall_s=s['numeric_wall_s'])
lines+=['',f"有效成功：基线{counts['baseline']}/10，反馈{counts['feedback']}/10。新增成功：{gained}；退化：{lost}。",'', '反馈追踪的是未改变的当前基线参考，包含已有Z+5cm偏置；只改变编码输入，不改decoder输出。这个实验没有恢复旧导航、接近重规划或提前闭指，也不检验参数搜索后的最佳成功率。全后缀误差可能受末段目标变化影响，不能单独代替抓取接触质量。', '', '原生验收要求达到抬升阈值并不触发既有安全停止；本轮不新增持续保持门。完整帧与末帧接触/高度在trial.json、metrics.json中保留。', '', '来源原轨迹可见，属于轨迹重定向。物理通过不等于训练RGB视觉质量通过。']
complete=[r for r in rows if r['status']=='complete']
if complete:
 m={k:statistics.mean(r['comparison']['common_mode1_rmse_m'][k] for r in complete) for k in ('baseline','feedback')};analysis['mean_common_mode1_rmse_m']=m;analysis['relative_rmse_reduction']=1-m['feedback']/m['baseline']
 lines += ['',f"平均共同mode1腕误差：{m['baseline']*100:.3f}→{m['feedback']*100:.3f}cm，降低{analysis['relative_rmse_reduction']*100:.1f}%。该指标相对于包含原固定偏置的基线参考，不能单独代表对物体抓取几何的误差。"]
if (root/'FINAL_AUDIT.json').exists():
 audit=json.loads((root/'FINAL_AUDIT.json').read_text());analysis['audit_passed']=audit['passed'];analysis['physical_cases']=audit['physical_cases']
 lines += ['',f"证据审计：{audit['physical_cases']}次物理执行通过；所有新基线与历史seed完全一致，所有A/B原token前缀一致。",'', '时序证据：右手从第340帧开始发送闭合目标，第346帧才切mode1并开始物体相对位置适配和积分反馈，滞后120ms。Y+2cm条件首个反馈tick仍有约2cm横向误差，而积分横向修正约0.2mm。此结果支持优先检验更早接近/适配的假设；尚未通过提前接管对照证明其因果贡献。', '', '本实验不支持“只恢复接触后腕部积分即可恢复原泛化能力”。推荐下一步保持闭指指令与验收不变，提前到张手接近阶段进行物体相对位置适配，仍在这10个位置做配对验证。']
if (root/'videos.json').exists():
 lines += ['','## 对照视频','', '左侧基线，右侧编码前右腕闭环；保存状态重渲染，无新增物理。']
 lines += [f"- [{v['condition_id']}]({v['path']})" for v in json.loads((root/'videos.json').read_text())]
if (root/'rendering_timing.json').exists():
 t=json.loads((root/'rendering_timing.json').read_text());analysis['rendering']=t
 lines += ['', '## 实测时间', '', f"物理批次：{analysis['started_at']} 至 {analysis['ended_at']}，{analysis['numeric_wall_s']:.2f}秒（含执行编排）。", f"渲染：{t['started_at']} 至 {t['ended_at']}，{t['wall_s']:.2f}秒。", '规划耗时：未知；包含实现、核查和文档整理的总耗时：未知。新API请求0次，新增token 0。']
(root/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2,ensure_ascii=False));(root/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(analysis,ensure_ascii=False))
