import json,time
from pathlib import Path
r=Path(__file__).resolve().parent;rows=json.loads((r/'results_audit.json').read_text())['cases'];by={(x['group'],x['task']):x for x in rows}
labels={'BallKick':'踢球','ChairPush':'推椅子','ChairSit':'坐椅子','DollyPush':'推车','GroundBasketPick':'地面提篮','GroundBottlePick':'地面抓瓶','GroundBoxPick':'地面抓箱','TableApplePick':'桌面抓苹果','TableButtonPress':'桌面按按钮','TablePlatePick':'桌面抓盘子'}
lines=['# 10任务 CaP 实验结果','',f'更新于 {time.strftime("%Y-%m-%d %H:%M:%S %z")}。未完成的项目显示为进行中，不作为完整失败结果。','','仓库基线 `2ed4500`，原生控制适配版本 `d991dd5`。实验使用真实 MuJoCo 物理执行和 SONIC；没有在线模型 API 调用，由当前 GPT 会话及 GPT 编写的阶段反馈程序规划。','','## 结果','','每组10个任务，每项最多10次尝试、每次30秒仿真时间、最多12次阶段反馈。第一次成功后才允许1次优化，仍计入10次预算；优化失败保留原成功。参考组物体沿世界Y轴移动10cm，新增侧向对齐与接触目标调整。','','| 任务 | 无参考 | 有参考扩增 | 无参考双视角 | 有参考双视角 |','|---|---|---|---|---|']
def outcome(x):
 if not x:return '进行中'
 label='成功' if x['valid_task_success'] else ('质量待审' if x['quality_pending'] else '原生通过但非有效抓取' if x['native_success'] else '未成功')
 return f'{label}（{x["attempts"]}次）'
def videos(g,n):
 p=r/'videos'/g/n
 return f'[第一视角]({p}/ego.mp4) / [第三视角]({p}/external.mp4)' if (p/'verification.json').exists() else '待生成'
for short,label in labels.items():
 n='G1Wholebody'+short+'Teleop';lines.append(f'| {label} | {outcome(by.get(("no_reference",n)))} | {outcome(by.get(("reference_augmented",n)))} | {videos("no_reference",n)} | {videos("reference_augmented",n)} |')
lines+=['']
for group,label in [('no_reference','无参考'),('reference_augmented','有参考')]:
 xs=[x for x in rows if x['group']==group];lines.append(f'- {label}：已结束 {len(xs)}/10，有效任务成功 {sum(x["valid_task_success"] for x in xs)}/10；原生判据通过 {sum(x["native_success"] for x in xs)}/10。')
lines+=['','篮子被膝部碰起触发的原生高度通过不计有效抓取。抓取审查要求末段持续手部—目标接触、多个手部接触部位以及脱离支撑；它不是力闭合证明，也没有验证抬升后保持1秒。原生验收结果和原轨迹完整保留。','','## 用时','','数据下载：154.300秒。下表是已记录尝试的累计工作时间，包含决策、等待、执行和证据导出；不是纯物理计算时间，也不是并行任务的日历耗时。中断尝试没有完整壁钟记录，不补造。决策/信箱计时覆盖范围不同，可包含中断尝试，因此不是前列的子集，也不能与前列相加。','','| 组 | 已记录尝试壁钟秒 | 独立记录的决策/等待秒 | 实际仿真秒 |','|---|---:|---:|---:|']
for group,label in [('no_reference','无参考'),('reference_augmented','有参考')]:
 xs=[x for x in rows if x['group']==group];lines.append('| '+label+' | '+' | '.join(f'{sum(x[k] for x in xs):.3f}' for k in ['recorded_attempt_wall_s','mailbox_decision_wall_s','sim_s_executed'])+' |')
renders=[json.loads(p.read_text()) for p in (r/'videos').glob('*/*/render_timing.json')];lines+=['',f'有独立计时记录的视频渲染：{len(renders)}个任务包，累计 {sum(x["wall_s"] for x in renders):.3f}秒。各包 verification.json 记录解码、帧数与摄像机信息。','','## 解释边界','','- 无参考组全部结束并保存结果哈希后，才读取专家动作序列。9次已执行尝试遭遇会话等待超时或进程中断，单独记录；不能把它们全部归因于控制器。','- 每任务分析1条参考，首次未通过时最多再检查1条。7项在保存状态上通过当前原生判据；坐椅、苹果、按钮仍未复核通过，其运行只能称为参考条件生成。','- 源数据未保存 sim.ctrl，离线参考接触力由PD重构估计，不是原始精确力记录。生成轨迹的成功则来自本轮实际物理执行与独立复核。','- 失败后允许在已有控制接口内调整动作。策略跨任务演进，且无参考和有参考场景不同，本轮不能作为skill收益的严格因果消融。','- 视频是MuJoCo原生资产的第一/第三视角记录，不是Isaac渲染；抽帧检查不等于逐帧时序质量认证。','- 预算内失败说明当前参考表达、规划和控制组合未跑通，不能证明KPI或skill无效。','',f'详细用时、逐次失败和参考出处：[完整报告]({r}/REPORT.md)。轨迹索引：[success_catalog.json]({r}/success_catalog.json)。']
(r/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
catalog={'generated_successes':[],'augmented_successes':[],'native_passes_excluded':[]}
for x in rows:
 if not x['native_success']:continue
 path=r/'runs'/x['group']/x['task']/'success'/f'attempt_{x["selected_attempt"]:03d}'/'trajectory.json';entry={'task':x['task'],'group':x['group'],'trajectory':str(path),'quality':x['quality'],'videos':str(r/'videos'/x['group']/x['task'])}
 key=('generated_successes' if x['group']=='no_reference' else 'augmented_successes') if x['valid_task_success'] else 'native_passes_excluded';catalog[key].append(entry)
(r/'success_catalog.json').write_text(json.dumps(catalog,indent=2))
print(r/'REPORT.zh-CN.md')
