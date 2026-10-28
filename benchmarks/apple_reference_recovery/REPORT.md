# 苹果参考恢复与扩增实验

结果：**参考恢复10次均未成功；按前置门禁，苹果平移10cm的扩增未启动（0次），没有成功轨迹或优化轨迹。**

使用当前 GPT 会话分析和编写的阶段程序，无外部在线模型 API。机器人、资产、SONIC 权重、物理参数和原生验收固定；只改变现有动作接口内的参考与反馈策略。

## 逐次结果

| 尝试 | 改动 | 结果 | 仿真秒 | 手—苹果峰值N | 碰桌峰值N | 视频 |
|---|---|---|---:|---:|---:|---|
| 1 | Old reference run used episode 1 and stopped fingers with reference time. Use episode 0 matched reset, full future body preview and commanded fingers; no artificial lateral step in original scene. | 未完成抓取 | 19.600 | 0.000 | 0.000 | [第一视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_001/ego.mp4) / [第三视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_001/external.mp4) |
| 2 | Attempt1 body reference did not move root. Approach measured reference stance using existing leg planner, then separate open-hand approach, commanded closure and gated lift. | 碰桌触发停止 | 3.570 | 0.000 | 314.923 | [第一视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_002/ego.mp4) / [第三视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_002/external.mp4) |
| 3 | Attempt2 walking overshot into table with left hip; brake before descent and target a stance10cm farther back. | 碰桌触发停止 | 8.055 | 0.000 | 14.400 | [第一视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_003/ego.mp4) / [第三视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_003/external.mp4) |
| 4 | Attempt3 near-zero walking caused long arm reach to pull root; use nonzero gait constant speed and early braking, with2cm raised wrist clearance. | 碰桌触发停止 | 6.410 | 0.000 | 172.245 | [第一视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_004/ego.mp4) / [第三视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_004/external.mp4) |
| 5 | Attempts2/4 coast into desk; use one short forward segment and long stop/squat before reaching, preserve hip clearance. | 碰桌触发停止 | 2.775 | 0.000 | 231.153 | [第一视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_005/ego.mp4) / [第三视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_005/external.mp4) |
| 6 | Zero velocity did not stop gait momentum in5; use bounded reverse target then settle before arm approach. | 碰桌触发停止 | 6.070 | 0.000 | 69.729 | [第一视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_006/ego.mp4) / [第三视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_006/external.mp4) |
| 7 | Repeated velocity-planner handoff drifts; test standing reference legs through existing fullbody channel with offline arm IK after one approach segment. | 未完成抓取 | 19.100 | 0.000 | 0.000 | [第一视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_007/ego.mp4) / [第三视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_007/external.mp4) |
| 8 | Fullbody static stance7 pulled robot backward. Retain planner/teleop and oppose measured forward reach drift with bounded reverse velocity during arm movement. | 碰桌触发停止 | 15.285 | 0.000 | 477.884 | [第一视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_008/ego.mp4) / [第三视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_008/external.mp4) |
| 9 | Attempt8 fixed reverse action overcorrected. Use root-error feedback during reaching, no extra gait segment, shorter contact transition. | 目标超出接口可达范围 | 13.500 | 0.000 | 0.000 | [第一视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_009/ego.mp4) / [第三视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_009/external.mp4) |
| 10 | Attempts1/7 measured-body references failed locomotion/reach; source motor targets include drive offsets. Test them as reference poses through same encoder/decoder, with original timing and contact gate; no token or torque override. | 未完成抓取 | 8.500 | 0.000 | 0.000 | [第一视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_010/ego.mp4) / [第三视角](/home/kinetra-4090-01/code/zijian/experiments/20261013_apple_recovery/videos/recovery_010/external.mp4) |

## 用时

已记录尝试累计壁钟 **495.97秒**（包含当前会话分析等待、仿真和证据导出），实际仿真累计 **102.865秒**。各请求等待时间保存在 mailbox/*_timing.json。日历时间包含诊断、渲染与其他收尾，不等同物理计算时间。

## 诊断与边界

- 原场景源候选为 episode_000000；此前保存状态未通过当前原生高度判据，不能称为已验证成功参考。本轮尝试在同一接口下将其恢复为成功，未实现。
- 源 SONIC 文件的 encoder_obs 全为 NaN；control_phase 显示 recorded tokens / SONIC motors。实测姿态经当前 encoder 再编码，与原 token 执行不等价。没有新增 token 直放或力矩覆盖通道。
- 主要问题在接近与站位保持：参考身体姿态没有恢复前进，planner接近后又有漂移；部分动作触发髋部或手部碰桌。没有建立有效手—苹果接触，不能归因为夹力不足。
- 第1/7次还存在无接触时重复姿态修正的低效反馈；记录保留，不把它们描述为成功恢复或充分搜索。
- 第2次启动前为了更新外部策略脚本中断过一次等待；该次尚未 reset、未执行物理，原请求目录保存于 pre_reset_interruptions/，没有另计一次物理尝试。
- 每次失败后先保存诊断再提交下一次配置。所有物理尝试和失败均保留；策略源码按内容哈希归档于 policies/，首次策略另存 policy_v1.py。
- 视频是MuJoCo状态回放，完整解码验证；不推进物理。抽帧视觉检查不代表逐帧时序认证，训练就绪状态仍为false。

下一项阻塞是验证原始参考控制输入与当前接口之间的可执行转换；恢复通过之前，不应宣称参考扩增成功率或继续批量变体。
