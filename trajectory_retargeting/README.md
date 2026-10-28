# 原轨迹骨架重定向路线

本目录保存原成功轨迹作骨架、通过 IK 等接口调整动作的历史源码快照。它与 [纯 skill 驱动生成](../skill_generation/README.md) 分开维护；两种路线的代码／准备文件均保存在 cap-sim 内。

## 源码

- [主策略](source_snapshot/20261013_apple_straight_sweep/policy.py)：从原动作阶段和关键帧出发，根据物体接触先验映射腕部目标，通过 IK 调整右臂；保留原阶段、手指和身体动作。
- [主实验入口](source_snapshot/20261013_apple_straight_sweep/run.py)：原十种位置／桌高条件的调度。
- `source_snapshot/20261013_apple_variants/run.py`：场景变体后端。
- `source_snapshot/20261013_apple_ordered_retry/conversion.py`：参考转换。
- `source_snapshot/20261014_1558_ik_teleop_comparison/run.py`：IK／teleop 对比。
- `source_snapshot/20261014_1719_stance_margin_pair/run.py`：站位余量对比。

[manifest.json](manifest.json) 记录每份源码的原位置、仓库内位置和 SHA-256；复制内容与原文件逐字节一致，原文件保留。

## 使用边界

这是历史源码保存目录，尚未迁移成可独立启动的入口。脚本中的 `__file__` 相对路径、原实验配置、成功参考轨迹、资产、模型和运行时依赖仍按历史目录组织；不要直接从快照启动仿真。实验数据与视频保留在原 experiments 目录，未复制大型数据。

后续运行前需显式适配输入／输出路径，并按统一实验记录冻结环境、接口、控制模式及验收条件；新实验输出使用分钟精度目录。此次整理没有执行仿真，也没有改变外层 pipeline。
