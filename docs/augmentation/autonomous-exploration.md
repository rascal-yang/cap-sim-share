# API 自主探索

> 历史实验路线，已退出默认编排。当前稳定路线见 [stable-seed-first.md](stable-seed-first.md)，此文保留用于解释旧实验。

入口仍为 `scripts/run_contact_batch.py` / `run_contact_review.py`，只改独立 contact 路线。
每个任务只需要 `task`、`dataset`、成功 original-token `baseline` 和绑定它的 API `annotation`；可指定 `baseline_inspection`。
旧配置的 `failed_case`、`diagnostic_review`、`prior_criteria_decision` 不再进入初始证据或复用审查。

API 先选择 inspect / proceed / stop，再提交原有审查、计划或复核决定。
`needs_tool` 会把理由带回探索循环，Python 不自动决定调参。
可查询 replay 及本轮 attempt_XX：

- model：关节名称、qpos/dof 地址、轴和限位、执行器顺序。
- aligned：按 API 选定政策帧返回已对齐的实际关节位置、原始/最终目标、目标误差、各子步执行器控制、根姿态、前控制接触日志和末子步对应的原生指标。
- controller / trajectory：原始控制器目标和高频状态/实际控制序列。
- metrics / contact_control：保存的原生指标和接触、PD 增益记录。
- physics / result：物理配置及实际执行结果。

每次最多32行，通过 start/stop/stride 扫描再放大。aligned 的 clipping_flags 未记录时返回 null，不把缺失当零；不会声称重建了未测量的传感数据。
查询结果独立留在新运行的 exploration_NNN/tool_result.json，包含来源哈希，不写入 replay 或历史案例。
原始 API 请求、响应、决定仍由现有签名/来源检查保存。模型可访问参考动作序列，因此属于轨迹重定向，不证明技能迁移。

需要新物理证据时，API 在计划阶段选择 run 并给出实验假设和控制参数；复核阶段选择 retry 后再提交计划。
所有这种诊断实验与普通候选一样消耗物理次数、先冻结 experiment.json、使用同一个执行器、保护和验收。没有不计费的隐藏诊断仿真。
API 自己决定 restart / checkpoint。现有手指时序、29关节控制和mode1接口能力边界仍存在，不能声称任意动作都已可执行。

默认 max_exploration_decisions=24（每任务共享，包括 proceed/stop），首次900秒、retry1200秒、总预算沿用配置，物理最多3次。
耗尽预算明确报告 budget_exhausted；API 明确选择停止报告 exploration_blocked 并保留理由。原生标准不降低。

运行示例（配置和实验目录按项目记录规则创建；不要复用已有 summary 的目录）：

```bash
python scripts/run_contact_batch.py --config /absolute/new-study/config.json --provider /absolute/new-study/provider.json --credentials ../api/openai_config.toml --secondary-credentials '../api/openai_config 2.toml'
```

代码测试与真实任务成功率分开报告。本次实现验证不启动新的物理实验。
