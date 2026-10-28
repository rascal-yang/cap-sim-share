# Replay导航、精停与连续恢复

推荐配置为 [replay_retry.json](../../humanoid_mimicgen/augmentation/examples/replay_retry.json)：保留原成功导航和完整身体replay，新增最多2次运行内恢复重试。验证报告在 [20261028_0040_apple_replay_retry_holdout](/home/kinetra-4090-01/code/zijian/experiments/2026-10/28/20261028_0040_apple_replay_retry_holdout/REPORT.md)。

执行仍由 `augmentation.replay_navigation` 负责：原始远位导航 → 从实测全身姿态平滑接入replay → 原手指时序抓取/保持 → 若未抓住或失去持握则张手 → 在当前root/腿/物体状态下恢复手臂与腰部 → 重观物体、按平移重新锚定右手目标 → 再次接近。没有物理状态重置、原token前缀或decoder后覆盖；实际身体控制均为SONIC mode1。

重试位移在4s实测姿态衔接中按同一五次权重从零增加，避免首帧目标跳变。苹果旋转不直接带入腕姿；保留原指令序列和相对抓取方向。恢复有实测误差与有限完成窗口，`restored=false`被明确记录；流程完成不等同于实测收敛。若当前物体已落到桌外，静态手臂/腰部IK在本轮求解预算内未找到可行入口时记录`retry_target_unreachable`，不会把桌上动作直接往地面平移执行。

启用精停可使用 [replay_parking_retry.json](../../humanoid_mimicgen/augmentation/examples/replay_parking_retry.json)：复用旧版0.5m/s速度包络、0.6s直立腿参考交接，以及最多2次、每次最多3cm的固定脚位纠偏。未达目标、无进展或纠偏不可行时继续抓取，无校准准入、路线/接触/姿态中止门。停车之后恢复完整源身体动作；冻结腿部会丢失本例最多0.195rad的示范腿部运动，不作为推荐抓取策略。

本轮固定六个横向条件Y±10/15/20cm，原导航6/6成功；精停+恢复首次3/6、最终5/6，其中2例由真实重试救回，监测碰桌峰值约13–18N；原导航约16–292N。旧版腕部积分补偿加入后降为3/6，不推荐启用。成功率优先时保留原导航，不将精停作为默认替换。最终推荐组合在新增Y−12.5、+12.5、+17.5cm三个位置均首次成功；这些是小规模同场景插值验证，未证明总体成功率超过原基线。

使用已有缓存计划和本地数值provenance，传入推荐配置：

```bash
PYTHONPATH=. scripts/humano_python.sh -m humanoid_mimicgen.augmentation.replay_navigation \
  "$NEW_EXPERIMENT_ROOT" "$PLAN_JSON" \
  humanoid_mimicgen/augmentation/examples/replay_retry.json "$PROVENANCE_JSON"
```

实验根目录应按当前Asia/Shanghai时间新建，重名加序号；每次prepare创建新case并在物理步进前冻结真实模型/provider、输入、代码和接口。provenance使用`transport=local`、`perception_mode=privileged_only`、`api_calls_before_launch=0`与实际输入SHA。每个case仅初始化一次，后续retry保持同一物理和控制历史。

`execution/attempts.json`分别记录首次与重试、是否进入操作、是否完整结束、原生成功、终态持握、截断阶段和原因。`commands.json`分开保存source_desired、重新锚定后的desired、encoder_pose、retry_index与真实200Hz手指source行。身体50Hz token及每物理步映射、手指原值、目标与控制完整保存在`sonic_actions.npz`；token不编码手指。预算/异常期间的当前attempt也必须保留。

该路径无图像输入、无API等待；视频为执行后的保存状态诊断渲染。原生任务成功、碰桌质量、最终持握和视频训练可用性分开报告。始终可访问源动作，因此属于trajectory retargeting，不是独立策略生成或正式十条件IK/mode0对照。
