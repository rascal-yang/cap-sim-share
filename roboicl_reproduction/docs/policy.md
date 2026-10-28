# RoboICL policy、执行闭环与发布物逐文件审计

审计对象为本工作区 `RoboICL/` 的实际文件，日期 2026-10-15（Asia/Shanghai）。这是静态源码审计，不是物理实验；未安装依赖、未启动 Isaac Sim、未请求模型 API。`train*.py` 与 `reference_data.py` 的实现由参考数据专项文档覆盖，本篇只说明它们在策略主链的入口。本文中的“原模块”指当前检出的 RoboICL 实现，不代表已证明与论文运行时完全一致。

## 1. 真实调用链及研究边界

```text
roboicl.run → 两个进程
  policy_server → XPolicyLab PolicyServer → dialogue_policy.Model
      继承 bounded_policy.Model → astra_policy.Model
  robodojo → upstream EvalEnv.run_eval
      将 env.eval_one_episode 替换为 rollout.eval_one_episode
      reset → 观测 → RPC update_obs → RPC get_action
      → env.take_action（上游 IK、插值、队列、物理步）
      → RGB + 本体状态 + executed_action → RPC update_obs
      → 下一次 get_action 提交上轮 receipt
      → 正常 episode 结束 → RPC finalize
```

当前可用的模型策略入口只有 `dialogue_policy.Model`。`astra_policy.Model.get_action`、`bounded_policy.Model.get_action/build_payload` 均主动抛出 legacy entrypoint removed；不能因为文件名仍在就把它们算作三种可直接运行的策略。当前工具 schema 只发布一个 `act`，没有历史检索工具，也没有自动执行任意程序的工具。

原方法不是直接播放参考关节序列：它把 TRAIN 的观测、动作张量、回执及结果图片作为模型可读 few-shot，在 LIVE 由模型生成新动作。但执行规划器**能够访问所展示的原动作窗口**，不能归入“无法访问原动作序列的纯经验技能生成”条件。当前代码没有把成功演示自动提炼为跨任务技能库、显式恢复规则库或跨 episode 长期记忆；LIVE memory 在 reset 时清空。迁移到 G1/Revo/HUMANO 需要重新定义机器人接口，双 X5 的 14 维动作不可直接复用。

## 2. `dialogue_policy.py`：策略核心（1016 行）

### 配置、输入与输出

`DialogueConfig/load_config` 从 `ASTRA_HARNESS_CONFIG` 读取 JSON，校验 horizon、相机集合、图像编码、时间/上下文预算、重试、纠错、LIVE 锚点等。类默认预测/执行 24 步、最多 0 次纠错、1 次传输尝试，不能把这些默认值当成发布协议：两个 protocol JSON 明确启用 IK guard、2 次 action correction、3 次 transport attempt；上层还配置 task-adaptive horizon 和 triptych。

动作接口：一行 `[left6,left_gripper,right6,right_gripper]` 共 14 个数；`ee_delta` 使用世界轴平移（米）和世界轴 rotation vector（弧度），不是 Euler 角。以最新观测的 link6 EE pose 为首点，chunk 内累计**前一个请求目标**；旋转为 `q_next = exp(rotvec) * q_previous`。joint 模式是两组六个绝对关节目标。夹爪为归一化位置指令 0..1，反馈夹爪状态是前次命令，不是实测开度/力/抓握成功。

`_convert_tensor` 检查 T×14、禁止 bool、有限数、固定 horizon（开启时必须精确长度）、每臂平移/旋转 L2 范数、joint 相对前一目标的最大增量，再委托 `validate_actions` 输出上游可识别的 pose/joint dict。只执行转换结果的 `execute_horizon` 前缀，预测尾段不补齐、不伪称执行。平移界限默认 0.02m、旋转 0.15rad、joint 0.15rad；不是碰撞、可达性、速度或力矩约束。

### Prompt 与 TRAIN/LIVE 初始化

`_initialize` 校验参考 instruction 或显式 aliases、source SHA 对应的 verification、固定动作窗口、25Hz、世界坐标动作空间、相机编码。bundle 要每个 episode 有对应审核。缺失源 HDF5 时可能恢复同目录 `verification.json`；这是检查过的 sidecar 信任边界，不是当前重新读取原始 HDF5。可选 head calibration 要求与 LIVE step0 state/图像尺寸一致，TRAIN 必须是独立校验的 frame0。

TRAIN 由 `tool_reference_messages` 装为固定 prefix，bundle episode 边界标 gap；LIVE calibration 置于固定 TRAIN/LIVE 边界。固定 prefix 的模型、instructions、tools、messages 和 effort 合并哈希。prompt 明示 TRAIN/LIVE 分离、原动作不可盲目照抄、intent 不等于执行/成功、RGB 才用于任务进展判断。模型不接收 object pose 或 reward；但接收了任务文档与真实机器人状态，可选已校验头相机 K/T。

### `pending` / ledger 的真实回执

`get_action` 收到唯一 `act` call 后验证新且非空的 `call_id`，保存 opaque reasoning 与 canonical function_call（去掉输出 item 的 id/status，保留 call_id），再验证 actions。合法返回建立 `pending={start,predicted,returned,call_id,...}`。执行后的 `update_obs` 才增加 ledger。

下次 `build_payload` 先 `_commit_feedback`：必须看到连续 `start+1..current_step` 的 receipt，不从经过时间推断执行；没有执行也无 rejection 时禁止重复推理；执行数不得超过已派发数。回执包含实际 interval、predicted/dispatched/executed/unexecuted、执行索引、命令 ledger SHA、最后提交命令、tracking 与 controller_events。全量命令只写本地日志，不逐条全部注入 tool output。随后放真实已采集 RGB/本体状态。

**三种失败不能混淆：**

1. 非法 tensor 在策略进程拒绝，`executed_steps=0`，原 observation 不变。附 function_call_output 要求修正同一观测，不“补偿”未执行动作；最多 `max_action_corrections` 次额外付费请求。
2. 上游 IK continuity 在执行前拒绝，经 `notify_action_rejection` 严格核对 step、action_index、discarded_commands，整个被拒绝动作及尾段未执行；此前 prefix 可能已经执行。零执行时不添加新 RGB、ledger 或 anchor，只把拒绝回执保留在 correction tail。连续零执行拒绝计数超限走 hold/失败。
3. tracking guard 是执行**之后**发现残差大，实际前缀已运动；只丢弃剩余尾段，再从新观测请求下一步。与执行前拒绝完全不同。

纠错耗尽后 `_observed_state_safety_hold` 会在有可靠 native `max_episode_steps` 且未到上限、全部 state 有效时返回**一个 joint 模式 hold**：复制最新观测六关节与已命令夹爪值，推进一步并获得新观测。它是显式本地控制模式切换；不能记录成全程 ee_delta，也不能声称物体/机器人绝对不动（控制跟踪、重力、接触仍发生）。缺数据或无步数边界就抛错，不猜关节/夹爪。日志 `safety_holds.jsonl` 明确区别被拒绝提案与实际 hold；源码给出的剩余 provider 尝试上界为 remaining_steps × (corrections+1)^2 × transport_attempts，依赖 native 步数有限。

### terminal 与请求层

`finalize()` 调 `_commit_feedback(terminal=True)`，不再发模型请求，最终一段不会因为没有下一轮推理而丢失回执。terminal 可允许最后零执行回执；正常动作回执标 terminal、success 字段明确未向模型提供。hold 有自己的 `local_safety_hold_result` 分支，没有普通 receipt 的 terminal 字段，另写 dialogue_final。`OBSERVATION_END` 只是图片结束分隔符，不是 terminal。异常离开 rollout 不保证调用 finalize，因为当前 rollout 没有 try/finally 包它。

`_request` 在 bounded transport 之上执行有限重试：失败 reservation 保留，成功响应前绝不派发机器人动作；时间预算不足不再重试。加密 reasoning 被 provider 拒绝时删除 reasoning 项重建 payload，并持续禁用该 episode 的旧 reasoning replay，同时保留可观测动作/回执。保留 reasoning 时比较原始 item 的精确有序子序列，无法解密也不宣称验证推理内容。

静态缺口：bounded 层识别 `drop_comparison_response_id` recovery，但 dialogue 层只实际实现 `drop_reasoning_items` 变换。当前 build_payload 不生成 comparison_response_id，通常触发不到；如果未来传入该字段，不能假定删字段恢复已实现。缺失/多余 act、错误 name/call_id 属硬失败，不全部纳入 tensor correction。

## 3. `bounded_policy.py`：付费请求、预算与传输（582 行）

继承 base 的 observation archive，增加 demo 加载与离线 verification sidecar 恢复。计数器 request_count/reported_tokens/pending_reserved_tokens/started 为**进程 run scoped**，reset 不清零。预算预留是 text bytes + image_count×1024 + output reservation 的保守启发式，不是 tokenizer 精确计量，不是货币硬上限；成功用 provider input/output usage 结算并释放本请求预留，失败/未知请求保留预留且 poison，只有授权重试路径解除 poison。缺 usage 拒绝执行。

`_request` 支持 urllib system 或显式 per-request direct proxy bypass。每请求有 timeout、client request UUID、payload SHA、请求与 HTTP 事件日志。HTTP 408/409/429/5xx 通常可重试，确定性 provider 错误通常不可，409 最多两次；Retry-After 只接受数值秒且上限60。HTTP200 JSON failed/incomplete 也失败，不因 HTTP200 就接受动作。SSE 仅 `response.completed` 且 status=completed 可用；流截断、DONE 无 completed、contradictory status、failed/incomplete 均拒绝。content_filter incomplete 在此实现中允许有限重传（不代表能执行被过滤的结果）；未知 terminal 默认最多两次，仍受总 transport 尝试数限制。

日志有请求 payload/图像/输出，API key 不放入 request record，错误文本做 key 替换及长度限制；这是具体实现边界，不应扩大为“所有日志自动脱敏”。非 SSE JSON 校验主要排除 failed/incomplete，未强制 status 必须 completed；后面还要 usage 与 act 校验。网络 timeout 可能服务端已经付费处理，重传可重复收费，但不会重播已执行机器人动作。

## 4. `astra_policy.py`：状态与动作公用基类（263 行）

`image_url` 将 HWC RGB/RGBA 转 uint8 RGB JPEG90 data URL；`plain` 递归转换 numpy。`update_obs` 只收六种本体字段与三相机 RGB，instruction、step；不收场景对象真值/reward。保存可选 head calibration 与 native step limit，executed_action 存在才记 ledger，并通过 `tracking_receipt` 计算跟踪误差。重复 step 直接返回，不更新状态/图像；因此 controller rejection 必须走专用 RPC，不能冒充新观测。该实现也未全面校验 step 单调增长，适配器必须提供连续、诚实的 step。

`validate_actions` 处理 joint、ee、ee_delta，检查 shape、finite、gripper、quaternion 并归一化（离 unit 偏差超过0.05拒绝），世界轴旋转左乘。joint 后不能在同 chunk 未有新 EE pose 地接 ee_delta。这个通用函数**本身没有** dialogue 的增量范数限制，也没有机器人特定关节上下限/自碰检查；直接调用时不能继承不存在的安全保证。

基类 `_tools` 仍有 inspect_history/可选 inspect_records，`history_context` 被 import，但当前子类 `_tools` 覆盖为单 act，基类 get_action 已删除。旧 `_request` 的三次重试亦被 bounded 覆盖，不是正式路径。

## 5. LIVE 保留：`trajectory_memory.py`（213 行）

`LiveTrajectoryMemory(count, fallback_interval, retain_latest)` 是确定性 episode 内记录选择器，不是视觉检索/重要性评分/LLM 摘要器。已有真实 execution 才 capture；start=end 不捕获；重复相同 interval+内容幂等；冲突覆盖或时间重叠拒绝。保存 deep copy，完整局部 archive 不裁剪。

有 maximum 时固定 anchor=`i*maximum//count` 去重排序；retain_latest 时只产生 count−1 个固定 anchor，最后一个容量保留最新完整 chunk；无 maximum 用固定 interval。anchor0 保留首个从0开始的非空 chunk，其余按 `start < anchor <= end` 保留跨过锚点的**整个实际执行交互**，不是抽一帧，也不是预定未来动作。anchor 仅本地日志，模型只看到已执行区间。

`render_with_frontier` 总保留初始观测与当前 RGB；key chunk 为起始观测→原 response/reasoning/call/receipt→结束观测，相邻边界只出现一次。未选 chunks 合并 `<TRAJECTORY_GAP>`，包含半开区间 [start,end) 和每段原始执行元数据/intent，明确不能推断跨 gap 的连续性或成功。latest slot 每轮替换；旧 latest 若非固定 anchor 就压成 gap。tail 保留当前未匹配/被拒绝 calls，零执行纠错不消耗 anchor。

稳定 cache frontier 只到最新固定 anchor 的结束观测；尚增长的 gap、滚动 latest、非固定当前 RGB、纠错 tail 都不算不变前缀。保留块数/图片数有界，**gap 内每段 summary 和本地 archive 仍随执行增长**，不能说整个系统内存或文本上下文 O(1)。

## 6. cache、历史、图像与反馈逐文件

| 文件 | 真正入口、输入→输出 | 失败机制与复用边界 |
|---|---|---|
| `anchored_cache.py`（34行） | dialogue anchored render 调 `annotate_anchored_breakpoints(inputs,fixed_length,stable_live_length)`；给稳定范围 user 内容末尾文本加 breakpoint。`observation_end` 生成文本 footer | frontier 越界或末尾非文本拒绝；只修改请求副本，不改 archive。可直接复用这种稳定边界算法，provider 是否支持扩展字段需实际验证 |
| `rolling_cache.py`（38行） | 仅非 anchored 且 visual_history_batches>0 的老路径；依据剪图批次在首张保留图前的可持久文本添加历史边界 | 不轮换旧 marker。它不是 LIVE chunk selector；正式 anchored 路径不用此函数 |
| `history_context.py`（60行） | 纯函数 recent_context 取最近16动作/4决策（可配置）与事件；retrieve_records 取 inclusive step 范围≤32 | deep copy、范围检查；只限记录数不限 token。当前仅旧 base import，无当前 act 工具接入，不能宣称模型可随时召回全部历史 |
| `provider_compat.py`（59行） | base 验证 mode/endpoint；transport deep-copy wire payload；dialogue 读 usage cache 字段 | 仅 Responses，URL 须 HTTP(S) 绝对路径且以 /responses 结尾；没有 Chat Completions 转换。cached_tokens 缺失时标无有效报告，不能当零命中；valid 未验证 cached_tokens≤input_tokens，异常 provider 数据可能得到>1比例 |
| `observation_images.py`（165行） | TRAIN/LIVE 共用 encoding/prompt_image/camera_input_parts | thumbnail JPEG90归档→最长边384 JPEG80；native JPEG90保持 source URL。triptych 固定左腕/头/右腕各640×480，横向1920×480，重新编码JPEG90 4:4:4。JPEG/RGB/base64/尺寸错误拒绝；拼图LRU128。native图像被限制≤640×640，G1新尺寸要显式改合同 |
| `image_budget.py`（126行） | 从 manifest 估算 TRAIN，再对实际prefix计数 admission；K 个 LIVE key chunks | 每观测 C 张：无 latest 上界 `(2K+1)C`，有 latest `2KC`；triptych C=1。总数超限提前拒绝，不偷偷删 TRAIN/key chunks。只限 input_image 个数，不等价 token/字节上界；dialogue 另做实测payload检查 |
| `action_feedback.py`（51行） | base/rollout 用 tracking_receipt(target,observed)；rollout 每chunk新建 TrackingGuard | EE 位置 L2、符号无关四元数角度、joint 最大误差，剔除夹爪命令伪测量。默认连续2步 position>0.05m/orientation>0.5rad/joint>0.5rad 即中断余段。不是接触、碰撞、滑落或抓牢检测器；计数每chunk重置 |
| `__init__.py`（1行） | package docstring | 无注册、无副作用、无启动逻辑 |

非 anchored 的 `_request_inputs` 只把旧 LIVE RGB 换成文本占位，所有文本/state/action receipt 仍保留；按 prune_every 批量剪图，最多 window+every−1 个LIVE图像批次。它与 anchored 的完整 chunk/gap 保留机制不可混称。

cache 请求含 prompt_cache_options、prompt_cache_key、text breakpoint、可选 reasoning.context，都是当前代码发出的接口字段。源码只能证明发送意图；真实兼容性与命中必须看 provider 返回 usage/cache.jsonl，不能从稳定哈希或“有 breakpoint”推断已缓存，更不等价直接观察 KV 或验证账单。

## 7. 相机与错误处理辅助模块

| 文件 | 输入/调用与输出 | 限制 |
|---|---|---|
| `camera_preflight.py`（31行） | `robodojo --capture-only` 实际调用；10次 env.get_obs，记录 observation 与私有 tiled annotator 的CPU/CUDA route统计，保存frame0/9 PNG、capture_probe.json | 无模型动作/评分；依赖环境 `_annotators['rgba']`、render_product 等私有API，并非通用相机校验器；其 run 本身没有断言图像非黑 |
| `camera_readiness.py`（60行） | rollout 实际使用 `blank_cameras`；检查三路存在、HWC≥3通道、有限且RGB有非零像素。另一 `capture(env,env_ids)` 可显式 render_for_capture 并最多30次 render-only retry | 当前 owned rollout 没调用 capture，而用自己的 `_ready_observation(env.get_obs)`；上游 get_obs 是否每次真正 render 需补全子模块核对。非黑不等于内容新鲜/镜头正确/无遮挡；zip list归一化也不保证缺失env检测 |
| `head_calibration_capture.py`（56行） | rollout 可选首帧 capture；上游 camera K/world pose(USD axes)减env origin→head_reference+step0 state | 单环境、K/T shape/finite、正焦距、旋转正交与det、图像尺寸校验；不是读取目标物体坐标。依赖 scipy Rotation、上游 camera/robot_manager API |
| `head_camera_prompt.py`（62行） | dialogue 可选调用；校验camera与实际head图像，将K按缩放变换、T原样输出文本 | source只能TRAIN/TEST_RUNTIME；USD +X右,+Y上,−Z前，投影翻转y/z，triptych只在prompt解释+640偏移，不重复改K。provenance形参未写入模型文本；历史标定语义仍依赖源证据 |
| `eval_failure.py`（138行） | rollout 用 `policy_call` 包RPC异常→env._policy_failure并抛PolicyServiceError；safe_error_code供transport分类 | 异常跨RPC只保留受限类别。另有 preserve_failure（视频关闭搬至interrupted_videos，不覆盖）和 opt-in finalize_policy_failure（满足最少步数/healthy条件用原生reward终评），但 owned源码无直接调用这些辅助函数；是否由上游调用当前无法证实 |

`finalize_policy_failure` 若被集成，会查询 reward 并用于**外部评估终止**，不将 reward 送进策略。它不是恢复任务控制。异常回收最终视频、官方终评成功与否不可仅由这些函数“存在”推定。

## 8. `rollout.py` 与 `policy_server.py`

`rollout.py`（131行）持有真实控制权：reset→首帧→逐action take_action→每一步新 observation/update_obs。观测metadata含step、native max_episode_steps、实际提交的action与可选controller_event。`executed_action` 保存的是提交的 EE/joint 命令，不自动等于上游实际IK关节解/驱动力，深入复现还需上游执行日志。默认tracking guard可由环境变量关闭；IK guard由配置开关安装到**解IK进程**，依赖 `src.eval_client.ik_continuity` 与上游 take_action 的原子检查。

拒绝事件到策略后 break 当前chunk，丢弃 suffix；正常 episode结束最终 finalize。`_rpc_timeout_s` 覆盖所有纠错×传输尝试与最大Retry-After，加60秒余量。max_turns达到会直接抛PolicyServiceError，此直接路径未设置 `_policy_failure`，与 policy_call 包装错误不完全相同。其他执行异常也不是一律走RPC失败保全。

`policy_server.py`（50行）配置 import paths，硬依赖 XPolicyLab `client_server.ws.model_server.PolicyServer/Config`，绑定host（默认127.0.0.1）/port，构建Dialogue Model，async serve_forever，ping30秒/timeout600秒。不包含WS实现、IK或仿真。CLI reasoning-effort传入model_cfg，但Dialogue实际 effort取 ASTRA_REASONING_EFFORT 或 harness config，不能只凭该CLI参数认定最终值；以最终payload记录为准。

## 9. tests：覆盖与缺口

`tests/test_reference_setting.py` 两个 unittest：验证8个Open零样本与22个one-shot、类别计数、reference horizon与adaptive_horizons一致（swap_blocks=10），以及manifest不混入evaluation_status/exception/bundle_source_episode。不是策略物理成功测试。

`tests/test_ik_execution.py` 八个 unittest：

- 从**当前上游 eval_env.py AST**编译 take_action_batch，mock机器人/控制器，验证任意手臂跳变原子拒绝且不改queue/counter/输入；合法周期关节解归一化；相对上一accepted target的跳变拒绝；畸形candidate关闭；插值构建失败不commit目标；joint hold刷新下次EE baseline。
- 当前rollout的中途拒绝丢弃suffix且notify一次、最终finalize；当前Dialogue Model在step0拒绝输出零执行回执、ledger为空、pending清除。

测试顶层直接 import 上游 `src.eval_client.ik_continuity`，并读取上游源码；不是独立于缺失submodule可运行的测试集。mock/scipy stub、不启动Isaac、无API，适合补齐**正确上游版本**后的离线回归；无法证明PhysX动作原子性、真实跟踪或碰撞安全。本目录没有对应 transport/SSE/LIVE锚点/图像预算/terminal异常/hold耗尽的大规模回归测试；不可将README能力描述当成已测覆盖。本审计未运行该测试集，避免把无法导入误报为测试通过。

## 10. README、License、submodules 与 images

| 文件 | 审计结果与复现用途 |
|---|---|
| `README.md` | 描述双进程/双Python环境、冻结上游与资产、固定TRAIN windows、zero-shot B25和one-shot J12/B12、dry-run/capture-only/scored运行，42任务文档与论文30任务分组不要混淆。退出码0表示评估完成而非任务成功，成功看原生result.json。论文和远程演示链接只是仓库引用，本次不把链接宣传当已复现结果 |
| `LICENSE` | 项目自有代码MIT，复用应保留copyright/license notice；第三方子模块和benchmark资产不因此自动变为MIT |
| `.gitmodules` | 四个外部库：Mosi-AI/RoboDojo、XPolicyLab/XPolicyLab、isaac-sim/IsaacLab、NVlabs/curobo。当前 `git submodule status` 四项均 `-`：未初始化，故不能审计其真实执行实现或宣称已跑通 |
| `.gitignore` | 忽略pycache/pytest_cache、data、results、eval_result、log、.env。论文运行所需资产、结果和私有配置并不随git自然提供；忽略规则也不是秘密永不入库的证明 |
| `images/teaser_benchmark.png` | 已查看实际1291×519图，四类别mean progress score柱图与多基线，用于README说明。是汇总宣传图，没有逐episode原始数据/方差/评分实现，不能替代复现实验统计 |
| `images/rednote.png` / `images/wechat.jpg` | README标为讨论群二维码/宣传图片；file元数据分别735×1139 RGBA PNG、939×1455 RGB JPEG。不是模型观测、TRAIN素材或运行依赖；未扫描/访问二维码 |

## 11. 直接复用、改造和阻塞清单

**可复用算法与记录协议：** `LiveTrajectoryMemory`、gap/observation footer、image_budget、稳定cache边界、receipt/未执行suffix语义、预算日志、SSE终态判断、图像编码合同、TRAIN/LIVE同语法。纯函数也需保持各自前置条件，不能将文件孤立拷贝后漏掉调用方保证。

**需要适配的模块：** dialogue schema/prompt/STATE_KEYS/图像尺寸/机器人坐标/步频/夹爪语义；action_feedback 的阈值和机器人自由度；安全hold是否允许joint模式；provider扩展参数支持；相机标定捕获；rollout的实际执行接口。X5的link6≠G1/Revo TCP，25Hz与双6关节布局不能默认为目标平台事实。当前用户规定HUMANO不得静默fallback，原代码的joint hold若移植必须声明实际控制模式、记录证据并遵守运行合同。

**目前阻塞完整运行：** 四个上游子模块未初始化，实际 RoboDojo IK guard/原子dispatch与终评保全未可读；模型API与其扩展字段、benchmark外部资产、reference HDF5/bundle证据、匹配的两个Python环境、Isaac兼容栈均需分别验证。不能仅补一个HTTP client就称为完整复现。

**关于“碰撞自愈”的结论：** 源码实现有限action纠错、执行前IK连续性拒绝、执行后tracking余段中断、保留真实反馈供模型重规划以及显式有界hold；没有接触力/碰撞检测器，也没有已证明的碰撞后自动恢复策略。是否从RGB/失败回执学会绕障、解卡、重新抓取需要另立实验并以真实行为与验收记录证明。
