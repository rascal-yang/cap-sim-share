# API Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended here) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** API 按 skill 调用本地 Python 工具，完成可复现的批量增广、执行、验证和导出。

**Architecture:** 独立 augmentation 包组合已有 workflow 公共接口。SQLite 持久化任务和工具调用，单物理 worker 串行执行；模型 API 与确定性执行分离。JSON CLI、Python 和模型工具共享业务逻辑。

**Tech Stack:** Python 3.10+、stdlib（sqlite3、urllib、subprocess、fcntl）、现有 NumPy/MuJoCo/SONIC、pytest；不增加 Web 服务依赖。

**Spec:** `docs/superpowers/specs/2026-10-20-api-augmentation-design.md`（用户已确认）。

## Global Constraints

- 基线 `4ca42a2`；规格提交 `1afc985`。不修改其他生成路线，不默认启用 `policy.grasp_trigger`。
- 执行仍为 G1 / Dex3 / SONIC 1.0 / MuJoCo，研究分类 `trajectory_retargeting`，原动作可访问。
- 所有真实物理运行使用 `scripts/humano_python.sh`；TensorRT 10.13.0.35、ONNX Runtime 1.23.2、SONIC 1.0 CPU，保存实际 providers 与模型哈希。
- 每次物理运行先完成并冻结 experiment.json；遵守 docs/experiment-records.md。缺少证据不启动。
- 新实验按实际 Asia/Shanghai 时间命名 `experiments/YYYYMMDD_HHMM_<description>`，冲突加序号。历史目录不改名。
- 状态、实际命令和 attempted_commands 分开；不把 mode1 命令称为 token。视觉未验收不能标记 training_ready。
- API 凭据只读指定环境变量；真实联调前不声称 API 全流程已验证。
- 所有路径相对明确的配置文件目录解析；运行日志保存绝对路径及内容哈希，不保存密钥。

## Review Focus

- 同 request_id 在并发提交或 API 超时后重试：只执行一次（任务 2、4）。
- worker 被杀、PID 被复用或锁文件遗留：依据实际进程身份和排他锁判活，不盲目重跑（任务 2、3）。
- 输入在排队期间被修改、符号链接目标变化：冻结前重新比对内容，拒绝使用变更输入（任务 1、3）。
- 无动作/部分动作、观测长度不一致、尾部 attempted 命令未执行：保留失败证据，拒绝伪造对齐（任务 5）。
- API 缺失 usage、输出拒绝或工具参数不完整：保留 unknown 和失败原因，不推断成功或零费用（任务 4）。

## 文件职责

全部新增于 `humanoid_mimicgen/augmentation/`，除非任务指出其他位置。
`contracts.py` 处理输入，`conditions.py` 生成条件，`scene.py` 生成初态；
`jobs.py` 维护事务状态，`worker.py` 执行，`tools.py` 注册工具，
`provider.py` 实现 API 传输，`agent.py` 管理工具循环，
`export.py` 输出数据，`__main__.py` 提供 JSON CLI；
`skills/augmentation/SKILL.md` 保存流程，`tests/` 验证行为。

### Task 1: 版本化任务与确定性案例生成

**Files:** 新建 `__init__.py`、`contracts.py`、`conditions.py`、`scene.py`、`tests/test_conditions.py`。

**Interfaces:**
- `load_job(path: Path) -> dict` 返回规范化绝对路径输入，含 schema_version、request_id、baseline_request、conditions、budgets、output_root、render、export。
- `validate_job(job: dict) -> dict` 返回无物理步校验结果及输入哈希；复用 workflow 的错误类型。
- `sample_conditions(job: dict) -> list[dict]` 返回 case_id、robot_delta_xyyaw、object_delta_xy 的显式列表。
- `materialize_case(job: dict, condition: dict, directory: Path) -> Path` 返回新 workflow request 文件。

- [ ] 写测试 `test_seed_reproduces_conditions`（相同 seed 输出相同列表），`test_object_shift_does_not_shift_robot`（只改 object_delta 时机器人 qpos 不变），`test_invalid_inputs_fail_before_physics`（NaN、未知字段、超预算、grasp_trigger 被拒绝），`test_changed_input_rejected`（替换引用文件后校验失败）。
- [ ] 运行 `scripts/humano_python.sh -m pytest humanoid_mimicgen/augmentation/tests/test_conditions.py -q`，确认因缺少新接口失败。
- [ ] 实现上述接口：显式列表或 NumPy Generator/PCG64 均匀采样，记录算法；初态 robot XY/yaw 和目标 free joint XY 分别变换，其他 DOF 保持；不修改源轨迹。继承基线控制预算并限制总体预算。绑定 initial/task/model，初始几何检查在 no_physics 中完成；已有 stance_hint 只在明确坐标变换规则下使用，否则拒绝不一致输入。
- [ ] 重跑测试，检查生成文件可由 workflow 校验读取；断言 mj_step 调用为零。
- [ ] 提交 `feat: add deterministic augmentation job inputs and conditions`。

### Task 2: 持久化任务和幂等操作

**Files:** 新建 `jobs.py`、`tests/test_jobs.py`。

**Interfaces:**
- `JobStore(root: Path)`；`submit(job: dict, input_hashes: dict) -> dict` 返回 job_id/status/reused。
- `get(job_id: str) -> dict`；`claim(job_id: str, owner: dict) -> bool`；`transition(job_id: str, expected: str, next_status: str, details: dict) -> None`。
- `record_case(job_id: str, case_id: str, stage: str, details: dict) -> None`。
- `lookup_call(session_id: str, call_id: str, arguments_hash: str) -> dict | None` 与 `record_call(...)`，记录参数及结果摘要；同 ID 不同参数冲突。

- [ ] 写测试 `test_duplicate_submission_reuses_job`、`test_same_id_changed_content_conflicts`、`test_concurrent_claim_has_one_owner`、`test_call_id_bound_to_arguments`、`test_transaction_survives_process_exit`，分别断言任务数/owner 数为 1、冲突显式、提交事务恢复。
- [ ] 运行该测试文件，确认失败。
- [ ] SQLite 事务/唯一约束实现接口，状态变更使用比较更新；记录 owner PID、Linux boot ID、进程启动标识。任务数据采用原子文件发布，DB 不把未发布文件标为可用。
- [ ] 重跑测试，包含两个独立进程竞争提交；验证没有重复 case 和非法状态跳转。
- [ ] 提交 `feat: persist idempotent augmentation jobs`。

### Task 3: 完整本地批次 worker 与恢复

**Files:** 新建 `worker.py`、`tests/test_worker.py`。

**Interfaces:**
- `run_job(store: JobStore, job_id: str) -> dict`：持有跨进程 flock，串行处理案例。
- `start_job(store: JobStore, job_id: str) -> dict`：固定 wrapper/模块参数启动持久 worker，返回身份。
- `reconcile_job(store: JobStore, job_id: str) -> dict`：检查进程与产物后恢复可确认的阶段。
- 消费任务 1 的 materialize_case、任务 2 的事务接口，以及 workflow.service 的 plan/prepare/execute。

- [ ] 写测试 `test_all_cases_frozen_before_execution`（每例 execute 前对应合同已冻结）、`test_duplicate_start_does_not_execute_twice`、`test_dead_worker_not_replayed`、`test_complete_artifacts_resume_verification`、`test_budget_stops_remaining_cases`、`test_changed_queued_input_blocks_launch`。
- [ ] 运行 worker 测试，先用 fake backend 观察失败，禁止测试意外推进物理。
- [ ] 实现接口及各阶段日志；每例 fresh adapter。输入/实现哈希重校验，阶段预算与总墙钟截止分开。长阶段超时时先终止并确认 worker 停止，再释放排他权；不将截断轨迹标成功。已完成执行只验证/导出，无法确认的执行标 interrupted，新 attempt 必须显式新 ID。
- [ ] 重跑测试，并用子进程测试 SIGTERM/SIGKILL、PID 身份不匹配和遗留锁文件。批次 finished 与案例 physical success 分开统计。
- [ ] 提交 `feat: execute and recover local augmentation batches`。

### Task 4: Skill、工具注册和 API 循环

**Files:** 新建 `tools.py`、`provider.py`、`agent.py`、`skills/augmentation/SKILL.md`、`tests/test_agent.py`。

**Interfaces:**
- `tool_definitions() -> list[dict]`；`dispatch(name: str, arguments: dict, context: dict) -> dict`。
- 工具为 read_skill、validate_job、submit_job、get_job、get_case、export_job；export_job 在任务 5 接通，未就绪时明确 unsupported，不伪造完成。
- `ResponsesProvider(config: dict).respond(input_items: list, tools: list, instructions: str) -> dict`。
- `run_agent(task_path: Path, config_path: Path, store: JobStore, transport=None) -> dict`；transport 可注入脚本化响应。

- [ ] 写测试 `test_model_reads_skill_and_submits_job`、`test_repeated_call_reuses_result`、`test_unknown_tool_rejected`、`test_refusal_and_incomplete_response`、`test_missing_usage_stays_unknown`、`test_credentials_absent_from_logs`、`test_running_job_does_not_poll_model`。
- [ ] 运行 agent 测试确认失败；不得使用现有账号自动替代未提供的 API key。
- [ ] 实现严格工具 schema 和本地参数校验；模型/endpoint/key_env 显式配置，HTTPS 且无嵌入凭据，拒绝未知协议。Responses function_call_output 关联 call_id，保留必要响应项；请求超时不重做已记录工具动作。按需加载 skill，实际内容哈希入日志，限制轮数/输出/时间，保存 usage 与耗时。后台 worker 完成由本地等待驱动，不循环询问模型进度。
- [ ] 重跑测试；独立检查录制 payload 的请求/结果关联及日志脱敏，mock 结果清楚标注。
- [ ] 提交 `feat: orchestrate augmentation through skills and API tools`。

### Task 5: 当前 workflow 轨迹导出

**Files:** 新建 `export.py`、`tests/test_export.py`；接通 `tools.py` 中 export_job。

**Interfaces:**
- `export_case(experiment: Path, destination: Path) -> dict` 返回文件索引、哈希、物理/视觉状态及 action representation。
- `export_job(store: JobStore, job_id: str) -> dict` 返回成败分区的数据集索引。

- [ ] 写测试 `test_measured_state_action_alignment`、`test_attempted_commands_are_separate`、`test_corrupt_or_truncated_trace_rejected`、`test_failed_cases_retained_in_denominator`、`test_mode1_is_not_token`、`test_visual_status_preserved`、`test_reexport_is_idempotent`。
- [ ] 运行测试确认失败。先读取 runner 的命令采样及 mj_step 顺序，固定实际前/后状态约定后写对齐断言，不猜测时间标签。
- [ ] 实现 NPZ 数值数组加 JSON 元数据、原始记录与 manifest；核对合同、执行日志、结果和产物哈希，不能只信已有 result.success。状态不足以形成 transition 的记录保留为诊断，不能截短补齐。临时目录原子发布；相同结果复用，不一致拒绝覆盖。
- [ ] 重跑测试，再离线导出历史真实成功/失败 workflow 各一例；记录来源 ID 与零物理步。验证训练与 token 标记不被提升。
- [ ] 提交 `feat: export verified workflow trajectory datasets`。

### Task 6: 用户入口与端到端验收

**Files:** 新建 `__main__.py`、`README.md`、`examples/job.json`、`examples/provider.json`、`tests/test_cli.py`；新增实施验收报告于本计划同目录。

**Interfaces:** JSON CLI 子命令 `validate --job`、`submit --job --store`、`worker --job-id --store`、`inspect --job-id --store`、`resume --job-id --store`、`export --job-id --store`、`agent --job --provider --store`。

- [ ] 写测试 `test_json_stdout_and_stderr_diagnostics`、`test_cli_mock_agent_to_dataset`、`test_resubmit_has_zero_new_physics`；验证无凭据的 validate/local 工具可用，而 agent 返回明确缺凭据错误。
- [ ] 运行测试确认失败；实现 CLI、完整可替换路径示例和中文操作说明。模型不能直接选择任意可执行文件。
- [ ] 运行 `scripts/humano_python.sh -m pytest humanoid_mimicgen/augmentation/tests humanoid_mimicgen/workflow/tests humanoid_mimicgen/tests tests -q`；定位失败，不把旧测试数当本次结果。
- [ ] 检查与基线的 git diff，确认其他路线源码未变。用脚本化 API 通过真实工具和真实本地 worker 做小批次：新建并冻结实验记录，至少覆盖原起点和一个已支持的变化条件，逐例保存成功/失败、时间、provider、控制模式和导出路径；按实际物理成功率报告，不要求人为把全部结果做成功。
- [ ] 需要视频时遵循相关运行/视觉技能，渲染与模拟计时分开；未验收视觉状态保持未通过。执行前读取实验记录规则及适用领域技能。
- [ ] 用户提供 API 配置后，以真实模型重复完整 skill→工具→worker→验证→导出链路，核对 usage、工具请求和结果；缺凭据时留下明确待验证项，目标不标 complete。
- [ ] 提交 `feat: expose and validate end-to-end augmentation interface`，报告测试、实验和 API 的各自验证范围。

## 自检与交接

输入/采样对应任务 1；幂等和恢复对应 2–3；skill/API 对应 4；导出对应 5；入口、回归及真实运行对应 6。五项 Review Focus 均有具体测试。source access、物理合同、实际 mode、视觉状态及 token 语义是贯穿约束。

推荐执行方式：Native（当前会话串行实现），各任务提交后再进入依赖它的任务，完成后集中独立评审。尚未执行本计划、未启动新物理实验；API 凭据待提供，模拟协议测试不能替代真实联调。
