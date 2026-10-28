# TRAIN、LIVE 与物理执行的接入边界

本文说明 `20261015_1350_roboicl_complete_grasp` 使用的完整反馈路径及历史入口之间的关系，不报告正在进行的实验结果。逐文件职责和测试定位见 [file_map.md](file_map.md#本地入口与测试)；实际设置以该批次冻结的 `experiment.json` 为准。

## 两种 TRAIN 来源必须分开

| 路径 | 保存什么 | 能支持什么解释 |
|---|---|---|
| 当前稀疏实测观测：`prepare_native.py` | 从已独立验收的历史状态提取 13 帧观测、实测关节/接触和 4 组 head+external 图像；保留源时间与缺口。后续批次复制该 bundle 并核对观测及图像字节。 | 可供模型参考站位、接近、腕姿、手指配置和接触条件。源控制器为历史前缀加 mode0，观测变化不能冒充 mode1 动作标签。 |
| 同接口命令示范：`repro/train.py` 的 `build_train` | `humano.command_episode.v1` 中的观测、时间、单行动作、真实命令、模式和事件；重新转换动作并与记录命令核对。 | 只有完整来源、profile、时间与命令检查通过后，才可称同接口命令示范。当前稀疏观测批次未采用这条路径。 |

当前 LIVE 规划器只接收构造好的 TRAIN 观测/图像、LIVE 和显式前次反思，不接收原始动作序列、源轨迹文件、tokens 或 body keyframes。原动作前缀重放长度为 0；checkpoint 内部初始控制历史仍按冻结初态载入，不是再次执行源动作。新动作由当前观测生成，不是参考轨迹加 IK 的轨迹重定向。以上是接口约束，实际执行还需模式、字段和来源审计。

## 当前调用链

```text
冻结的稀疏实测 TRAIN + 当前 LIVE/RGB + control_status
    → 每次全新的隔离子代理 → 原始 act 响应 + SHA/身份来源记录
    → RobustSession 验证调用 → encode_actions 验证整个动作块
    → ControlAwareBridge → NativeBridge → 原 Backend.execute
    → 每条命令的 StepResult → FeedbackSession 收集已确认执行前缀
    → Session 提交回执与锚点/latest 记忆 → 下一次模型请求
```

`complete_run.py` 复用 `controlled_run.main()`，在本进程把会话类选择为 `FeedbackSession`。`FeedbackSession → RobustSession → Session` 逐层扩展，不改写历史 Session 或旧入口文件。`freeze_complete.py` 冻结对应入口、提示词和资源；`freeze_controlled.py`、`freeze_native.py` 保留各自历史路径，不能混用契约与入口。

文件 mailbox 只是传输。协调器必须为每次决策（包括修复）创建 `fork_turns=none` 的新子代理；动作代理完成后，保存响应原字节、请求/响应 SHA、代理身份，再发布 ready 标记。运行器核对来源并拒绝同尝试中重复的代理身份；不会由父会话补写动作。坏 JSON、封装或 call_id 可在同一观测有限修复；已执行动作后才推进 LIVE。会话隔离与文件读取指令不是操作系统文件沙箱，也不等于外部 API 服务部署。

## “完整执行前缀”的准确含义

每次提交动作块时，`FeedbackSession` 建立新的局部累积器。只有返回 `executed=True` 且被底层 Session 接受的命令，才作为已确认执行记录进入账本。包装器在反馈中携带其候选前缀，验证和未知执行处理仍由 Session 负责。

最终回执的 `latest_feedback.executed_prefix_feedback` 保存本块每条已确认命令的 `action_index`、`observation_step`、`elapsed_s`、`terminal`、`interrupt` 和反馈。每命令反馈可含端点误差、稳定状态、实际下肢输入、接触峰值/区间及 body/geom 名称。即使末条命令已无接触，较早命令的接触摘要仍可进入下一次请求；末条是零步拒绝时，保留此前已执行前缀，不为拒绝行造一条执行记录。

跟踪中断丢弃未执行后缀；下一块从最新实测状态重新转换。异常发生后，Session 保留已确认部分并标记未知执行和 terminal，不伪造异常命令的结果。累积器不跨块或跨修复泄漏，也不递归嵌套自己的前缀副本。

这里的“完整”限定为**当前块中已确认命令的反馈覆盖**，不是整段 episode 的永久无损历史，也不是把每个 200 Hz 物理采样直接提供给模型。长期上下文仍使用上游固定锚点/latest 记忆；摘要、锚点间缺口和原始采样证据应分别理解。

## 物理与控制接口

初始化和 LIVE 执行分开记账。原生环境 reset 在构造 SONIC 前有预恢复物理步骤；`ControlAwareBridge` 只转发并记录这些调用，不抑制或改变其行为。`run/reset_physics.json` 保存实际步数、时间、控制量及原始 qpos/qvel/ctrl/warmstart 恢复哈希。动作步数从恢复初态后单列；不能称整个 reset 为零物理，也不能倒推历史 reset 步数。

LIVE 经 `encode_actions` 整块验证，再逐命令执行：世界坐标双腕平移/旋转向量增量、绝对 wxyz 姿态、Dex3 七关节目标及允许的下肢/躯干通道。每条命令 0.1 s、物理步长 0.005 s、SONIC 更新 0.02 s，下肢 planner 约 0.5 s 更新；具体界限由冻结 profile 和实际关节范围检查。没有自动动作补齐或模式回退。

三层桥接各有边界：

- `NativeBridge` 从真实时间和新增状态算执行量，保存实际 encoder 模式和命令端点残差；跨短命令累计碰桌时间，保留真实终止 tick。端点残差不是逐 tick 插值跟踪误差。
- `StandingIdlePlanner` 只在显式启用且实际 feed 为 `mode4 / target_vel=-1 / height>=0.72` 时改为 `mode0 / direction=[0,0,0] / height=-1`，记录前后输入。行走和较低零速下蹲分支保持原语义。这里的下肢 mode0 不等于 encoder mode0，LIVE encoder 仍要求 mode1。
- `ControlAwareBridge` 在每条 contact/lift 命令或任意阶段的新闭指前检查准入。稳定证据来自连续 0.005 s 实测区间：平面速度≤0.03 m/s、根部局部 z 角速度绝对值≤0.1 rad/s，连续 0.5 s，并覆盖请求 idle 与实际已变换的 standing-idle feed。重复读取、缺口、移动或非 idle 不累计；当前命令还需满足请求与实际 idle 条件。这个角速度不是精确 Euler 偏航导数。

闭合方向由安装的 Dex3 open/close 配置按关节名称映射到实际模型顺序；未知方向变化保守门控。该方向代理、接触身份和力值都不证明抓稳。零步拒绝必须保持实测观测不变；合法开手/不新增闭合的动作可以获取后续物理证据。原碰撞终止门、提起接触门和 native 成功判据保持冻结定义；硬碰撞或跌倒后不能继续同一 episode。

## 证据与解释范围

每次完成后应核对 `result.json`、`timing.json`、`run/controller.json`、`run/planner_inputs.json`、`run/reset_physics.json`、独立验收以及请求/响应来源。`controlled_audit.json` 检查接口和实测准入，`prefix_audit.json` 检查回执前缀与确认账本一致；二者都不能替代抓取验收。准入命令数为 0 时，不能用空集合的“全部通过”声称已经验证接触阶段。

当前输入为 head+external RGB、本体状态和特权接触反馈，区别于原版 head+双腕及 25 Hz dual-ARX。原版 `rollout.py` 保留，RoboDojo/XPolicyLab 依赖尚未完整接入人形执行路径。`evidence/synthetic_protocol/` 与单元测试中的合成状态/响应只验证协议，不计入训练数据或物理成功率。

`20261015_1330_roboicl_controlled_grasp` 被启动前哈希检查拦截；`20261015_1342_roboicl_controlled_grasp` 在接近阶段主动中止，以修复末条反馈遗漏早期瞬时接触的缺口。二者与完整反馈批次分别留档，不作为本批的失败抓取。当前仅原位置 1 个场景，不能代替最近十个位置/桌高条件的公平对照。结果、阶段覆盖、成功分母和未知项只从各批次完成证据报告。
