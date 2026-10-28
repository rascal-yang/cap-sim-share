import json,time,hashlib
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
summary=read(r/'run/summary.json');ps=sorted((r/'run/reference_recovery/attempts').glob('*/result.json'));assert len(ps)==10
assert summary['status']=='reference_recovery_failed' and summary['generation'] is None and not (r/'run/generation').exists()
rows=[]
for p in ps:
 d=read(p);attempt=p.parent;assert not d['acceptance']['success'];assert read(attempt/'independent_validation.json')['success']==d['acceptance']['success'];assert d['acceptance']['sim_seconds']<=30+1e-6
 assert read(attempt/'decision.json')['recorded_unix_s']<=read(attempt/'reset.json')['time']
 states=np.load(attempt/'states.npz')['samples'];contacts=read(attempt/'contacts.json');assert len(states)==len(contacts)
 video=r/'videos'/('recovery_'+attempt.name);verification=read(video/'verification.json');assert verification['physics_steps']==0
 assert abs(verification['source_times'][-1]-d['acceptance']['sim_seconds'])<1e-6
 max_hand=max(c['hand_object_N'] for c in contacts);table=max(c['robot_table_N'] for c in contacts)
 table_links=sorted(set(b for c in contacts for e in c['events'] if 'table' in e['bodies'] for b in e['bodies'] if b!='table' and b!='apple_001'))
 rows.append({'attempt':d['attempt'],'failure':d['failure'],'sim_s':d['acceptance']['sim_seconds'],'wall_s_including_decision_wait':d['wall_s'],'peak_hand_apple_N':max_hand,'peak_robot_table_N':table,'table_links':table_links,'final_root_xyz':d['observation']['root_pose'][:3],'max_apple_lift_m':float(np.max(states[:,53])-read(r/'reference.json')['samples'][0]['object'][2]),'video_ego':str((video/'ego.mp4').resolve()),'video_external':str((video/'external.mp4').resolve())})
resources=read(r/'run/reference_recovery/native_resources.json')
for name,h in resources.items():
 if name.startswith('/') and Path(name).is_file():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
report={'status':summary['status'],'recovery_attempts':10,'recovery_successes':0,'generation_attempts':0,'generation_status':'blocked_by_reference_recovery','optimization_attempts':0,'calendar_wall_s_at_audit':time.time()-read(r/'protocol.json')['started_unix_s'],'recorded_attempt_wall_s':sum(x['wall_s_including_decision_wait'] for x in rows),'sim_seconds':sum(x['sim_s'] for x in rows),'source_episode':0,'live_model_api':False,'video_pairs':len(rows),'all_independent_validations_match':True,'frozen_resource_files_unchanged':True,'cases':rows}
(r/'audit.json').write_text(json.dumps(report,indent=2))
labels={'task_not_completed':'未完成抓取','unexpected_table_contact':'碰桌触发停止','invalid_action: Wrist target outside 1.5 m workspace':'目标超出接口可达范围'}
lines=['# 苹果参考恢复与扩增实验','','结果：**参考恢复10次均未成功；按前置门禁，苹果平移10cm的扩增未启动（0次），没有成功轨迹或优化轨迹。**','','使用当前 GPT 会话分析和编写的阶段程序，无外部在线模型 API。机器人、资产、SONIC 权重、物理参数和原生验收固定；只改变现有动作接口内的参考与反馈策略。','','## 逐次结果','','| 尝试 | 改动 | 结果 | 仿真秒 | 手—苹果峰值N | 碰桌峰值N | 视频 |','|---|---|---|---:|---:|---:|---|']
for x in rows:
 c=read(r/'configs'/f"reference_recovery_{x['attempt']:03d}.json")
 lines.append(f"| {x['attempt']} | {c['diagnosis']} | {labels.get(x['failure'],x['failure'])} | {x['sim_s']:.3f} | {x['peak_hand_apple_N']:.3f} | {x['peak_robot_table_N']:.3f} | [第一视角]({x['video_ego']}) / [第三视角]({x['video_external']}) |")
lines+=['','## 用时','',f"已记录尝试累计壁钟 **{report['recorded_attempt_wall_s']:.2f}秒**（包含当前会话分析等待、仿真和证据导出），实际仿真累计 **{report['sim_seconds']:.3f}秒**。各请求等待时间保存在 mailbox/*_timing.json。日历时间包含诊断、渲染与其他收尾，不等同物理计算时间。",'','## 诊断与边界','','- 原场景源候选为 episode_000000；此前保存状态未通过当前原生高度判据，不能称为已验证成功参考。本轮尝试在同一接口下将其恢复为成功，未实现。','- 源 SONIC 文件的 encoder_obs 全为 NaN；control_phase 显示 recorded tokens / SONIC motors。实测姿态经当前 encoder 再编码，与原 token 执行不等价。没有新增 token 直放或力矩覆盖通道。','- 主要问题在接近与站位保持：参考身体姿态没有恢复前进，planner接近后又有漂移；部分动作触发髋部或手部碰桌。没有建立有效手—苹果接触，不能归因为夹力不足。','- 第1/7次还存在无接触时重复姿态修正的低效反馈；记录保留，不把它们描述为成功恢复或充分搜索。','- 第2次启动前为了更新外部策略脚本中断过一次等待；该次尚未 reset、未执行物理，原请求目录保存于 pre_reset_interruptions/，没有另计一次物理尝试。','- 每次失败后先保存诊断再提交下一次配置。所有物理尝试和失败均保留；策略源码按内容哈希归档于 policies/，首次策略另存 policy_v1.py。','- 视频是MuJoCo状态回放，完整解码验证；不推进物理。抽帧视觉检查不代表逐帧时序认证，训练就绪状态仍为false。','','下一项阻塞是验证原始参考控制输入与当前接口之间的可执行转换；恢复通过之前，不应宣称参考扩增成功率或继续批量变体。']
(r/'REPORT.md').write_text('\n'.join(lines)+'\n');print({k:v for k,v in report.items() if k!='cases'})
