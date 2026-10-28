# 接口与产物

## Task

必填：`name`、`backend`、`scene`、`acceptance`、`budget`。可附加自然语言 `instruction`，整体送给规划器。当前验收包含正数 `lift_m` / `hold_s`；`budget` 包含 `attempts`、`sim_seconds`、`optimization_attempts`、可选 `feedback_calls`（默认 8）。总尝试包含优化尝试，不额外送优化次数。`references` 和 `skills` 是 JSON 文件路径列表，CLI 按任务文件位置解析。整份任务、参考和经验文档会冻结；跨变体建立新 run。

内置夹爪场景：`object_position` 为世界 xy（米），`radius` 为米，`table_height` 为桌面高度。G1：`translation` 世界 xyz（米）、`table_delta` 米、`scale` 比例。G1 会覆盖初始 z 以维持桌面支撑，所以高度变化应通过 `table_delta` 指定。所有后端 z-up、米、秒、四元数 wxyz。G1 约定还依赖外部运行时。

## Planner

`plan(request) -> decision`。request 包含当前初始／阶段观测、任务、接口、参考、冻结 skill、历史失败、同场景可复用前缀、完整成功基线。decision 必须含 `analysis.evidence`、`analysis.hypothesis` 和非空 `actions`；`analysis.change` 建议明确写本次调整。

`reuse_prefix` 仅引用本 run 的前缀 id；执行器先从初态重放前缀，然后执行新的后缀。`replan_after` 要求完成阶段后再请求后续动作；反馈程序不能再次指定 `reuse_prefix`。前缀执行不算新尝试，但消耗正常仿真时间。最多 200 个已执行动作，反馈预算独立于重试次数。

异常层级：非法动作记录为失败供重试；缺少初始错误分析、非法程序、模型／基础设施不可用会中止并保留未提交目录，不伪装成正常预算耗尽。

## Backend

### 可选的严格反思与控制模式检查

新任务可设置 `policy_checks={"require_reflection":true,"sonic_mode":1}`。
第一项要求规划器提供 `reflect(request)`：每次失败（包括预算内最后一次）保存结果后，读取实测诊断，生成并验证 `reflection.json`。
下一次请求明确携带 `previous_reflection` 和程序计算的 `previous_reflection_sha256`；决策必须原样引用到 `analysis.reflection_sha256` 并说明 `analysis.change`，然后才允许执行本次 reset。
恢复运行时缺失或不一致的反思会停止，不能静默重试。

反思字段：`source_attempt`、`result_sha256`、`failed_stage`、`observed_facts:[{path,value}]`、`hypothesis`、`alternative_causes:[]`、`next_change:{description,controls:[]}`、`expected_effect`、`reject_if`、`hypothesis_outcome`（supported/not_supported/inconclusive）。
事实路径从 `result` 或 `diagnostics` 开始，按点分隔，可索引数组；值必须与保存记录一致，缺失值不能作为零观测。此检查证明来源一致，不证明因果判断正确。
`reflect` 是非物理分析调用，用时保存在 `reflection_timing.json`，不扩展物理尝试或阶段反馈预算。
`OpenAIPlanner` 和 `CommandPlanner` 支持该入口；自定义/文件规划器需实现它，未配置严格检查的旧任务仍按原契约执行。

`sonic_mode=1` 拒绝动作和复用前缀中的 `body_keyframes`/`native_prefix`，同时核对后端报告的实际模式。
阶段诊断区分目标、实际根部/腕姿、位移和位置/朝向误差；HUMANO 额外提供原始控制模式、关节目标与实测差、阶段内接触峰值及持续时间。时长仅为阶段内统计；两指触碰不是力闭合证明。
`observation.fingers` 是手指指令；实际手指姿态从 `joint_interface.*_hand[].qpos` 获取。

可配置 `backend="your_package.module:Backend"`，仅加载用户配置的可信代码。

- `interface`：支持的控制通道和物理门限，供规划器读取。
- `describe(task)`：不推进物理的初始观测。G1 通过独立临时运行时初始化和正向运动学获取。
- `reset(task, attempt_dir) -> observation`：每次从相同初态开始。
- `execute(action, remaining_seconds)`：返回 `completed`、`reusable`、`failure`、`samples`、`observation`。内部以控制器频率执行，不逐帧调用 GPT。
- `export(attempt_dir)`：保存测得状态／接触记录。
- `metrics()`：返回有限非负 `roughness`、`root_omega_rms`、`peak_table_force`。

evidence 行包含 `time`、`dt`、`lift_m`、`support_force`、`hand_force`、`stable`。时间从 0 起连续，单位秒。连续达标条件：抬升 ≥ 阈值、支撑力 < 0.1 N、手部接触力 > 0.05 N、稳定；保持足够时长才成功。具体接触位置与阶段许可由控制器执行。G1 的阶段失稳门限比最终“未跌倒”验收更严格，沿用原实验；夹爪没有自由站立的下半身。

优化候选首先必须成功，然后要求动作粗糙度至少下降 10%，根部角速度 RMS 不恶化超过 5%，机器人桌面接触峰值不超过 max(5 N, 基线)。指标定义随机器人而异，不能跨机器人直接比较。这里没有力控制契约，不能把文本里的力目标当成已实现 KPI 控制器。

## Output

```text
run/
  task.json, experience.json       # 固定输入
  external_resources.json          # G1 外部代码/模型/网格校验
  attempts/001/
    request.json, decision.json    # 先记录分析
    reset.json, program.json
    feedback_NNN_request.json, feedback_NNN_decision.json
    stage_NNN.json                 # 执行结果与观测
    samples.json, states.npz, scene.xml
    prefix.json                    # 可复用动作，包括部分成功
    result.json                    # 最后写入，作为尝试提交标志
  success/attempt_001/
    trajectory.json                # 动作、验收、指标和来源
    samples.json, states.npz, scene.xml
    contacts.json                  # G1
  progress.json, summary.json
```

成功包复制时须保留同目录 evidence；G1 XML 的外部网格仍须可访问。`states.npz.samples` 每行是 `time + qpos + qvel`，维度依场景确定。渲染只恢复测得状态并做 forward，不推进物理。夹爪第二机位是工具／俯视视角，不称为人形机器人第一视角。

`result.wall_s` 包括初始规划、执行、反馈和导出；`initial_planning_wall_s` 单列规划，`acceptance.sim_seconds` 为物理用时；`summary.wall_s_this_invocation` 包含本次 run 调用的初态检查等开销。在线模型日志另存请求用时和服务端 token usage。

恢复会校验 task、输入经验、成功证据哈希和重新计算的验收；缺失 `result.json` 的中断尝试需要检查，不会当作一次新尝试跳过预算。该版本不提供对不可信本地文件篡改的安全证明，后端也属于受信任测量边界。

## 参考恢复前置门禁

公开编排入口 `run_pipeline` / `augment_pipeline` 在参考条件生成前实际执行原场景恢复。CLI 用 `--reference-task` 提供原场景与初态；参考文件中的成功布尔值不作为门禁证据。恢复使用原有 `run_task` 的逐步物理验收与后端独立复核（若后端提供），完整保存成功包后才放行。失败返回 `reference_recovery_failed`，不创建生成／变体目录。恢复、生成各使用明确的任务预算；恢复阶段优化次数为零，先建立可执行基线。

目标任务与恢复任务必须使用相同后端、验收和控制器／资产资源；允许场景、初态、名称和预算不同。恢复成功只证明当前配置与既定验收下能够执行，不自动证明抓取质量或解决原生判据的假阳性。无外部参考的 `augment` 也先生成原场景基线。只传 skill 的 `run` 保留原执行方式，不应标为已通过参考恢复。`core.run_task` 是内部单阶段执行原语，历史实验脚本直接调用它时不具备此编排门禁。
