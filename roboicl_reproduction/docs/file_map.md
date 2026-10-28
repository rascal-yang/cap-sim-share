# 文件职责与上游索引

先看[本地入口与测试](#本地入口与测试)了解 HUMANO 接入；数据流和反馈语义见 [integration.md](integration.md)。本页只说明文件职责，不代表当前物理实验已经完成或成功。

上游普通文件原样保存于 `vendor/`；下表保留逐文件定位。gitlink 只记录固定 commit，不声称已取得其内容。

| 上游文件 | 分析文档 | 本地对应/验证边界 |
|---|---|---|
| `.gitignore` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `.gitmodules` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `LICENSE` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `README.md` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `configs/adaptive_horizons.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/data.lock.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/protocols/one_shot_j12_b12.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/protocols/zero_shot_b25.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/references/one_shot_j12_b12.json` | train_and_harness.md | 原样保留；HUMANO TRAIN重建见 repro/train.py，原数据路径仍受上游约束。 |
| `configs/shared_harness.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/align_blocks.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/arrange_largest_number.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/build_tower.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/classify_objects.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/classify_objects_by_language.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/cover_blocks.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/deposit_coin.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/fasten_screws.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/fill_egg_holder.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/fill_pen_holder.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/fold_clothes.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/general_pickup.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/hang_mugs.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/imitate_sorting_sequence.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/insert_key.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/insert_tubes.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/make_kong.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/make_toast.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/match_and_pick_from_conveyor.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/organize_table.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/pack_objects_into_box.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/pick_from_conveyor_by_image.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/play_Xylophone.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/play_stacking_toy.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/play_tic_tac_toe.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/plug_in_charger.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/pour_balls_into_vase.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/pour_by_language.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/pour_liquid_into_cup.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/press_by_number.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/push_T.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/put_bottles_into_dustbin.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/solve_equation.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/sort_nesting_dolls_by_size.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/stack_blocks.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/stack_blocks_by_language.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/stack_bowls.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/store_laptop_and_headphones.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/store_tools_in_toolbox.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/swap_T.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/swap_blocks.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/tasks/sweep_blocks.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `configs/upstream.lock.json` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `images/rednote.png` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `images/teaser_benchmark.png` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `images/wechat.jpg` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/__init__.py` | 本页 | 包标记，无执行算法；原样保留，离线导入覆盖。 |
| `roboicl/config.py` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `roboicl/paths.py` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `roboicl/policy/__init__.py` | 本页 | 包标记，无执行算法；原样保留，离线导入覆盖。 |
| `roboicl/policy/action_feedback.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/anchored_cache.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/astra_policy.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/bounded_policy.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/camera_preflight.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/camera_readiness.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/dialogue_policy.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/eval_failure.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/head_calibration_capture.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/head_camera_prompt.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/history_context.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/image_budget.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/observation_images.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/provider_compat.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/reference_data.py` | train_and_harness.md | 原样保留；HUMANO TRAIN重建见 repro/train.py，原数据路径仍受上游约束。 |
| `roboicl/policy/rolling_cache.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy/train_endpoint_observations.py` | train_and_harness.md | 原样保留；HUMANO TRAIN重建见 repro/train.py，原数据路径仍受上游约束。 |
| `roboicl/policy/train_reference_bundle.py` | train_and_harness.md | 原样保留；HUMANO TRAIN重建见 repro/train.py，原数据路径仍受上游约束。 |
| `roboicl/policy/train_reference_checks.py` | train_and_harness.md | 原样保留；HUMANO TRAIN重建见 repro/train.py，原数据路径仍受上游约束。 |
| `roboicl/policy/train_tool_reference.py` | train_and_harness.md | 原样保留；HUMANO TRAIN重建见 repro/train.py，原数据路径仍受上游约束。 |
| `roboicl/policy/trajectory_memory.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/policy_server.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/reference.py` | train_and_harness.md | 原样保留；HUMANO TRAIN重建见 repro/train.py，原数据路径仍受上游约束。 |
| `roboicl/reference_setting.py` | train_and_harness.md | 原样保留；HUMANO TRAIN重建见 repro/train.py，原数据路径仍受上游约束。 |
| `roboicl/robodojo.py` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `roboicl/rollout.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `roboicl/run.py` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `roboicl/upstream.py` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `setup/constraints-sim.txt` | train_and_harness.md | 原样保留；HUMANO TRAIN重建见 repro/train.py，原数据路径仍受上游约束。 |
| `setup/fetch_assets.py` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `setup/fetch_reference.py` | train_and_harness.md | 原样保留；HUMANO TRAIN重建见 repro/train.py，原数据路径仍受上游约束。 |
| `setup/paths.sh` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `setup/requirements-policy.txt` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `setup/requirements-sim.txt` | train_and_harness.md | 原样保留；原环境启动/任务配置未移植为HUMANO。 |
| `tests/test_ik_execution.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `tests/test_reference_setting.py` | policy.md | 原样保留；离线执行主链见 repro/offline.py，HUMANO变体见 repro/session.py。 |
| `third_party/IsaacLab` | policy.md | 未初始化子模块；commit 已锁定，相关集成未验证。 |
| `third_party/RoboDojo` | policy.md | 未初始化子模块；commit 已锁定，相关集成未验证。 |
| `third_party/XPolicyLab` | policy.md | 未初始化子模块；commit 已锁定，相关集成未验证。 |
| `third_party/curobo` | policy.md | 未初始化子模块；commit 已锁定，相关集成未验证。 |

## 本地入口与测试

路径相对 `roboicl_reproduction/`。当前完整反馈链路由 `complete_run.py` 选择；旧入口保留历史行为，不能把新功能反写成旧批次已具备。

| 文件 | 职责与边界 | 对应检查 |
|---|---|---|
| `prepare_native.py` | 从保存状态提取 13 帧稀疏实测 TRAIN 和 4 组双视图；保留原时间、实测手指及接触。源为不同控制接口，不提供可执行动作标签；不重放、不推进物理。 | `test_freeze_controlled.py` 检查后续复制时观测内容与图像字节不变；提取来源另见实验 `source_evidence.json`。 |
| `repro/train.py` | 构建同接口的真实命令示范，核对观测、时间、模式及命令转换；与当前稀疏观测 TRAIN 是两条路径。 | `test_train.py`：命令来源、profile、模式、时间、白名单。 |
| `freeze_native.py` | 历史 native 批次冻结入口，保留以供复查。 | 历史契约和资源哈希；不继承新版校验结论。 |
| `freeze_controlled.py` | 校验创建时间、历史资源、TRAIN 复制、实际 CPU 加载、测试结果及门控证据；保存快照与契约。 | `test_freeze_controlled.py` 直接测试此文件的拒绝条件。 |
| `freeze_complete.py` | 为完整前缀反馈入口冻结 `complete_run.py`、`complete_prompt.txt` 和相关资源。 | 结构校验与实际冻结产物；`test_freeze_controlled.py` 不直接测试此独立文件。 |
| `native_run.py` | 历史执行循环；同时提供冻结检查、TRAIN 消息和记录辅助函数。 | `test_session.py`、`test_bridge.py` 覆盖相关组件，不等于完整物理运行验证。 |
| `controlled_run.py` | 读取冻结输入、重置、等待 mailbox、核对响应来源、执行并导出；包含时间与次数预算。 | `test_robust_session.py`、`test_controlled_bridge.py` 检查传输与桥接组件。 |
| `complete_run.py` | 复用 `controlled_run.main()`，在本进程选择 `FeedbackSession`；不改写旧入口文件。 | `test_feedback_session.py` 覆盖所选会话行为；实际入口仍需实验审计。 |
| `repro/session.py` | 管理观测白名单、上游固定锚点/latest 记忆、执行账本、有限纠错、中断后缀和 terminal。 | `test_session.py`：零步修复、部分执行、碰撞终止、未知执行异常。 |
| `repro/robust_session.py` | 包装坏 JSON/错误调用/重复 call_id 的有限零步修复；核对请求/响应 SHA、`fork_turns=none` 和子代理身份。 | `test_robust_session.py`：同观测修复、纠错预算、旧响应及重复子代理拒绝。 |
| `repro/feedback_session.py` | 将当前块每条已确认执行命令的反馈放入 `latest_feedback.executed_prefix_feedback`；末条零步拒绝仍带上此前前缀。 | `test_feedback_session.py`：早期瞬时接触、命名接触、部分中断、零步末条、跨块隔离、异常及非递归复制。 |
| `repro/humano.py` | 全块验证和转换世界坐标腕部增量、旋转向量、Dex3 绝对关节目标；不执行物理或补造动作。 | `test_humano.py`：维度、有限值、模式/字段、关节及工作空间限制。 |
| `repro/bridge.py` | 包装原 Backend，记录实测执行时间、端点误差、接触与实际 encoder 模式；碰桌计时跨命令累计。 | `test_bridge.py`：持续接触计时、重复读取、解除接触及终止分类。 |
| `repro/idle_bridge.py` | 显式启用时将站立零速下肢输入变为标准 IDLE 元组，记录变换前后 feed；不改 encoder 模式。 | `test_idle_bridge.py`：原 `Sonic.plan_legs` 输入边界、行走/下蹲不变和输入审计。 |
| `repro/controlled_bridge.py` | 接入 idle、实测稳定准入、按命名关节映射的闭合方向、每命令接触摘要；记录原生 reset 步数及初态恢复。 | `test_controlled_bridge.py`：逐 tick 稳定、实际 idle 覆盖、闭合门、接触前缀、终止锁定、reset 记账及导出。 |
| `idle_probe.py` | 固定配对停止诊断，匹配接近状态和控制器历史；不调用动作模型，不执行源动作序列。 | `test_idle_bridge.py` 检查适配器；配对物理证据另见 `20261015_1205_roboicl_idle_probe`。 |
| `controlled_prompt.txt`、`complete_prompt.txt` | 分别定义两个入口的动作模型上下文约束；完整版本说明前缀反馈字段。 | 冻结提示词 SHA；不把提示词规则当作执行证据。 |
| `dispatch_protocol.txt` | 每次决策创建新子会话，保存原响应、来源记录和 ready 标记；调度者不改写动作。 | 响应来源检查与逐尝试 provenance 审计。 |
| `repro/offline.py`、`run_offline.py` | 使用上游策略代码运行合成协议夹具。 | `test_original.py`、`test_memory_train.py`；合成轨迹不计入物理成功率。 |

表中测试文件均位于 `tests/`。测试覆盖的是列出的行为，是否通过以对应批次保存的命令和退出码为准；它们不证明抓取成功。

`docs/baseline_files.json` 是原 pipeline 的冻结文件清单，不是可随文档更新的索引。本页与 `integration.md` 的说明不替代实验契约。完整反馈批次为 `20261015_1350_roboicl_complete_grasp`，结果只以该批次完成后的独立验收与审计为准。
