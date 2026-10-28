# 连续导航与 replay 抓苹果

入口 `humanoid_mimicgen.augmentation.replay_navigation` 已从原始远位置完成苹果 Y+5cm 的导航、平滑衔接、原 replay 抓取及保持。完整证据见 [20261027_2138_apple_navigation_pose_feedback](/home/kinetra-4090-01/code/zijian/experiments/2026-10/27/20261027_2138_apple_navigation_pose_feedback/REPORT.md)。

执行流程：数值观测与缓存计划 → 0.25m/s朝目标站位行走 → 从实测姿态做4s五次插值 → 按时间执行 replay → 保持1s。实际到达触发衔接，不等待姿态、接触或停车就绪计时；路线偏差、姿态及接触力均记录，不中止或否决原生任务结果。无校准准入要求。仅有限仿真预算和模拟器非有限状态错误结束异常执行。

身体全部通过 SONIC mode1 编码并由原始 decoder 执行；原 token 前缀0，decoder后覆盖0，运行中状态重置0。手指按200Hz复用source行，身体按50Hz生成并记录token。右腕在编码前使用位置gain2、姿态gain1的反馈；原始desired与encoder_pose分别保存，tracking相对desired计算。该路径不使用原direct_grasp的等待、姿态或接触门，配置也不保留未使用的门限字段。

所需导航配置：

```json
{
  "route_limits": {"speed_m_s": 0.25, "stop_radius_m": 0.025},
  "transition_s": 4.0,
  "simulation_budget_s": 30.0,
  "feedback_gain": 2.0,
  "orientation_feedback_gain": 1.0
}
```

调用已有离线计划、导航配置及本地数值来源记录：

```bash
PYTHONPATH=. scripts/humano_python.sh -m humanoid_mimicgen.augmentation.replay_navigation \
  "$RUN_ROOT" "$PLAN_JSON" "$NAVIGATION_JSON" "$PROVENANCE_JSON"
```

`RUN_ROOT`应为按当前北京时间创建的合规实验目录。prepare自动创建新的分钟case，核验实际加载模型与CPU providers，冻结experiment.json，然后使用同一adapter运行。provenance记录`perception_mode: privileged_only`、`transport: local`、API调用数及输入文件SHA。成功报告中提供了具体可复用命令及搬移后有效的rerun_provenance.json。

执行目录保存完整pre/post物理状态、200Hz控制和手指指令、50Hz encoder输入/token/decoder目标、分阶段目标、接触、事件和实际providers。`sonic_actions.npz`覆盖每个实际物理步；身体token不包含手指，模型和真实测量历史仍是复现条件。

此次开发迭代共4次物理执行：原控制提前减速导致未到站；定速导航后执行完成但抓取失败；位置反馈1仍滑脱；完整姿态反馈后原生成功。最终单次计算11.721s，准备9.251s，创建到物理结束22.305s，仿真13.32s，无新API请求或图像/渲染。独立原生验收通过，666个token解码结果逐位相同，2664步手指命令完全匹配；548项测试通过。

仍存在导航碰桌峰值156.34N，作为实际质量问题保留。当前证据仅覆盖一个Y+5cm条件和最终配置的一次执行，属于访问原动作的trajectory retargeting；不替代十条件对照，也不证明独立策略生成或广泛泛化。

后续推荐重试配置及导航对照见 [replay-parking-retry.md](replay-parking-retry.md)。精停未超过原导航成功率，保持可选；抓取继续复用完整源身体动作。
