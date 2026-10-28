# 完整状态回溯与等时预算对照

隔离于其他路线的新入口：`scripts/run_rollback_benchmark.py`，执行模块`humanoid_mimicgen.augmentation.rollback_benchmark`。历史pipeline源码保持不变。

回溯包含MuJoCo积分状态/warmstart、SONIC历史与上次动作、任务验收计数、接触观察器状态，并补齐腿部积分、手腕力反馈积分、手指积分与上一帧目标。双手命令与控制日志保留完整继承前缀，manifest绑定来源。相同参数可以从接触中的检查点恢复；参数改变时，`validate_branch`禁止继承已经受到新参数影响的前缀。当前支持chair/button/apple三种显式profile，不推断其他控制器状态。

GPT只返回有界参数补丁和理由。本地编译固定字段、生成`api_compiled`来源记录并校验真实API调用；不伪称完整计划来自API原文。批量规划产生相同候选队列用于两种重试方式，仿真内无API调用。身体decoder输出不修改；手指命令单独保存。轨迹仍使用原动作前缀与参考，属于trajectory retargeting。

配置包括新`output_root`、自身`config_path`、逐任务`spec_paths`与`benchmark`。每份任务spec包括profile、dataset、shift、固定base_plan、source_frames、历史source_case、约束与经验；chair另外绑定成功measured replay及response_model。candidate_count默认8，可配置；已有候选扩展时用preserved_candidate_parameters冻结原参数，candidate_orders可冻结原相对顺序。禁止覆盖已启动目录。新目录使用Asia/Shanghai实际时间的experiments/YYYY-MM/DD/YYYYMMDD_HHMM_description。

运行（准备好新配置后，无需交互调整参数）：

```bash
python scripts/run_rollback_benchmark.py \
  --config /absolute/path/to/new/config.json \
  --provider /absolute/path/to/provider.json \
  --credentials '../api/openai_config.toml' \
  --secondary-credentials '../api/openai_config 2.toml' \
  --additional-credentials '../api/openai_config 3.toml'
```

启动器使用`humano_python.sh`的固定运行时，配置中的模型为gpt-6-astra low。每个物理case先冻结experiment.json与真实已加载模型/provider证据。每任务先执行共同种子，再验证阶段开始前、活动接触中的回溯结果精确一致。通过后，按相同候选顺序比较restart/rollback，并交替执行顺序。不同参数的分支一律从最早受影响的控制阶段之前恢复。

预算计入实际规划、完整种子、初始化、执行、导出与验收。默认规划按3组顺序摊销，另报全额规划成本。共享种子不冒充新尝试。达到成功仍继续队列以测量批量产出；到期停止启动新尝试，已启动尝试允许完成但若结束超时则不计预算内成功。渲染与最终full/token证明单列。只给每任务选中的成功分支做完整从头执行和独立token+双手回放，其他成功只能称验收成功，不称全部可回放训练数据。

2026-10-25验证：坐椅子、按钮、苹果各1场景、3种顺序、每组120s。新尝试合计51→69，验收成功39→55；成功条件数9/9→9/9，未显示任务成功率提升。相同候选平均整次耗时减少21.6%、10.8%、29.1%。60对共同候选完整数组一致，150案例证据审计通过，3个选定胜者full/token复核通过。281tests通过。

按钮沿用已批准button_palm_mesh_v1（精确手掌碰撞网格、20N/1mm/50帧），原生标准另列。苹果8候选试运行发现队列先耗尽，最终扩为12，原8参数/相对顺序保持不变，追加4个API候选；历史失败规划成本计入补跑。三种顺序不是独立场景；结论仅适用于共享批量API规划，不外推独立实时GPT重规划。

证据位于：
- `experiments/2026-10/25/20261025_1634_rollback_equal_budget/COMPARISON_REPORT.md`
- `experiments/2026-10/25/20261025_1709_rollback_apple_expanded_queue/`
- 前置失败和容量试运行分别保留在1628和1703目录，不混算正式成功。

汇总、画图、审计入口：`scripts/report_rollback_benchmark.py`、`scripts/combine_rollback_reports.py`、`scripts/plot_rollback_benchmark.py`、`scripts/audit_rollback_benchmark.py`。报告保留未知时长、超时尝试、原生/修订验收差异以及有限视觉审查；training_ready=false。
