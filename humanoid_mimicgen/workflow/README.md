# 独立的可复现操作流程

本目录是独立入口，不改 `cap_sim`、其他数据生成路线或历史 MimicGen 实验执行器。它组合策略存档、确定性几何检查、按需模型决策、冻结准备、真实 SONIC 执行和验收。

当前物理适配器明确支持 G1 / Dex3 / SONIC1.0 / 原生 MuJoCo 的单物体抓取、抬升、保持轨迹。操作手、物体、支撑面、阶段名称和帧范围均来自输入。其他机器人或操作能力不能因数据能解析就被视为支持。原动作可访问，结果属于**轨迹重定向**，不是从零生成操作策略。

## 接口

在 cap-sim 根目录使用已有 HUMANO wrapper：

```bash
./scripts/humano_python.sh -m humanoid_mimicgen.workflow plan --request request.json
./scripts/humano_python.sh -m humanoid_mimicgen.workflow prepare --request request.json --strategy-id <sha256>
./scripts/humano_python.sh -m humanoid_mimicgen.workflow execute --experiment <新实验目录>
./scripts/humano_python.sh -m humanoid_mimicgen.workflow inspect --experiment <实验目录>
```

CLI stdout 仅输出JSON，诊断写stderr；`execute` 返回0表示任务验收通过，1表示完成了失败试验，2表示输入/准备错误。Python调用同样的方法：

```python
from humanoid_mimicgen.workflow import service
proposal = service.plan(request)
prepared = service.prepare(proposal['strategy_id'], request)
result = service.execute(prepared['experiment_path'])
```

Python接口使用调用者提供的绝对路径或当前工作目录相对路径；CLI请求文件中的路径相对于该JSON文件。`inspect`不能仅凭目录存在断言进程仍活着。

## 请求数据

`files`包含 model、initial、targets、reference_states、task、alignment、record。record是有真实证据的物理实验记录，不能是未填写模板；继承的资产、物理、运行环境证据均再次校验。task中的原生环境与模型资源是部署依赖，其文件及模型版本必须保持匹配；策略包不会自动安装HUMANO或ONNX运行环境。

`strategy`含schema_version、parent_id、kind、adapter、context、bindings、phases、command_pose_offset、source_frame_count、policy、proposal、evidence、provenance。参见contracts.py的严格字段校验。

- `phases`采用命令半开区间 `[start,end)`，边界连续、无重复命令；`command_pose_offset`明确命令与测量位姿的对应。
- `bindings`按模型名称引用object/root_body/wrists/torso/feet/ground_geoms/support_geoms，hands指定实际操作手。
- `alignment`明确finger_source为executed_commands，不能将测量手指角度替代原命令。
- `policy`要求明确导航/停稳/规划/接近/物理门/任务验收/预算设置与origins，不提供针对苹果的隐式默认值。
- 确定性站位来自当前姿态和示范物体坐标系变换；其他候选来自保存的策略或模型提案。全部做IK/路径检查，不固定前移8cm、不固定偏移网格。

为了审阅已有证据，可从显式选择的历史记录构造请求：

```bash
./scripts/humano_python.sh -m humanoid_mimicgen.workflow.examples.from_record \
  --record <已验证历史实验>/experiment.json --out <新的请求目录> \
  --active-hand right --object <模型对象名> --support <支撑面geom名> \
  --speed 0.5 --stop-radius 0.10 --pose-offset -1
```

该工具只迁移已知grasp/lift/hold记录格式；它是输入导入器，**不是通用参数默认表**。速度和导航停止范围由调用者明确提供，接近速度取示范实测峰值，物理和验收标准有历史来源，调用者须审阅适用性。生成JSON使用相对文件引用，不依赖/tmp。

## 模型决策与重放

`decision.mode`可选：

- `deterministic`：只做几何计算，不能求得可行候选时返回unreachable。
- `responses`：配置model（显式GPT-6）、base_url、key_env、timeout_s。默认只在确定性候选失败时调用，request_alternatives为true时也可请求替代候选。绝不切换模型或读取其他凭据。
- `saved`：提供path和identity，提供同目录provenance.json，从保存的JSON提案读取，核对证据摘要，不调用网络。

站位模型输出接口提供 `propose_stances / needs_observation / abort`，引用scene/failed_geometry/source等证据；模型不能改变物理限制、增益、标注或执行任意代码。示范标注仍由原管线的annotation接口产生；当前决策接口不自动创造新的抓法或新的未标注接触轨迹。

另有 `resolve --request catalog.json`（Python：`selection.resolve(catalog)`）做技能选择和失败恢复提案。catalog包含：

```json
{
  "experiment_root": "experiments",
  "recovery_from": null,
  "decision": {"mode": "responses", "model": "gpt-6-astra", "base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "timeout_s": 60},
  "options": [{
    "id": "right_side_demonstration",
    "approach_description": "示范所定义的右手侧向接近",
    "initial_state_origin": "declared_initial",
    "request": {"...": "完整 plan 请求对象，包含已标注示范、操作手、源片段及其证据"}
  }]
}
```

上例是字段说明，`request`须替换成完整请求。API只能选择给定能力ID；不同手、接近方向和源片段由不同完整候选表示，不能凭空改写命令。选择后自动进行几何规划，返回 `selected_request` 和策略ID；使用该请求继续prepare。resolve不启动物理执行。`needs_observation/abort`保持为明确结果。

恢复时提供recovery_from失败实验目录；各候选parent_id必须指向失败策略，API输出failure_hypothesis和expected_condition。初态来源可为declared_initial（明确重新起始）或previous_terminal（完整检查点匹配）；无隐式重置。previous_terminal需包含qpos/qvel/ctrl/qacc_warmstart全部字段，缺少检查点则拒绝。保存的选择响应以同目录provenance.json绑定证据，离线重放不会再次请求模型。

Requests/Responses采用[官方结构化输出格式](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)，完整请求、原响应、解析结果、模型身份、耗时及错误均落盘。凭据只存在于环境变量和HTTP头。单元测试中的provider桩不等同于真实远端调用成功。

同输入复现直接调用 `prepare(saved_strategy_id, request)`，不会再次调用模型。场景/阶段/控制设置变更要求新策略；代码变化也要求恢复原版本或生成新修订。重新请求相同模型不保证相同决策，保存输出才是决策复现依据。

## 策略保存

策略保存在调用者指定的strategy_root/<sha256>/。manifest绑定策略内容与复制的输入、代码及规划证据。构建在临时目录完成后原子发布；校验失败不覆盖已有包。路径穿越、非有限数值、依赖篡改和不连续阶段被拒绝。

策略定义不写回执行结果；结果记录在strategy_root/executions/<strategy_id>/<experiment_id>.json中。parent_id用于表达新修订来源；失败的实验与反思保留。物理实验每次创建新的北京时间分钟ID，并先冻结合同；现有execution目录阻止重复执行。

## 补偿与标定

按用户后续要求，新增 `bounded_integral` 保留有效的闭环纠偏。gain_per_s、cap_m及evidence_file必须显式提供并随输入冻结；这次回归从成功实验导入0.5/s和2cm，没有成为任意任务的默认值。每个控制周期按“目标位置−实测位置”更新有界偏置，跨抓取阶段连续，记录误差和饱和。

旧实验的积分增益0.5/s和2cm上限没有成为新流程默认值。`compensation.mode=none`明确关闭额外补偿，SONIC本身仍执行闭环控制。

可选 `identified_equilibrium` 从逐周期实测指令和响应辨识局部一阶模型，再求稳态逆映射。每只操作手须提供calibration_<hand> NPZ（command_xyz、measured_xyz、scope_json、dt_s），声明max_prediction_error_m。dt_s必须匹配控制周期。模型必须有足够激励、稳定响应并通过预测误差验收。scope绑定适配器、模型、初态和对象/手绑定，超过观测指令范围返回needs_calibration，不做外推或使用旧增益。该方法是新补偿方法，不能继承旧积分补偿的物理成功结论。

## 失败与验收

按用户明确修正，取消抓取前末端位置/姿态误差门槛。接近规划执行后，仅检查底座运动和双脚支撑的连续稳定，再进入抓取；等待受approach_s预算约束。原policy.entry不再控制执行。抬升/保持的接触、离桌前置条件与连续保持验收独立记录，源时间本身不证明完成。

冻结前检查规划与实际执行的完整编译模型一致、实际三个CPU会话、模型/代码/输入哈希、编译物理参数和命名实体；真实执行仅SONIC mode1，无原token/mode0前缀，无重置或隐式回退。IK仅生成参考。

可选files.stance_hint包含xyyaw和来源，指定预设站位；同样经过IK、碰撞和导航路径验证，不再被更短但不同朝向的名义站位替代。

可选policy.recovery声明max_retries、release_min_s、release_timeout_s、retreat_timeout_s。只有接触/抬升失败且物体仍受支撑、无障碍接触时才启动物理恢复；先保存检查点与反思，张手，规划撤回到本次实测接近起点，再依据物体新位姿重定向、停稳、接近和重试。各阶段都通过SONIC执行，不写物理状态；不盲目倒放接触动作。保留已学到的腕部偏置，参数不自动调大。碰撞、失稳、仿真警告、支撑丢失或预算用尽立即结束。

重试次数、每次源命令区间与失败记录分别保留；成功验收只使用最后一次抓取的保持段，不能拼接多次失败的悬空时间。恢复功能有单元测试；是否在某次物理实验中实际触发，以recovery_count和events为准。

结果区分进程退出、轨迹执行完成、抓起保持、原生checker、视频质量。碰撞、失稳、抓前物体移动、接触丢失均停止。异常时保存已尝试与已完成命令，非有限测量写为null并拒绝成功；发布失败结果和反思。程序正常退出不代表抓取成功；未执行指标为null；训练视觉未验收时training_ready=false。

首个新入口实际运行：20261019_1946_workflow_native，真实执行1789物理步、481帧接触源命令，抬升阶段contact_lost，无碰桌，未成功抓起。它验证了执行与失败记录链路，**没有证明去掉旧经验参数后仍能成功**。旧成功配置和本次策略不同，不能作严格单一变量对照。

修订后的执行：`20261019_2004_workflow_native`，真实执行10000步（50秒），全程mode1。右手最终位置误差1.74cm、姿态误差0.0831rad；本次明确输入的姿态进入限值为0.03rad，未进入抓取而预算耗尽。这说明当时的导入夹具进入条件与控制器响应不匹配（后已按用户要求取消该门槛），不能据此断言该位置不可抓，也不能把solver容差当成普适抓取容差。后续应以接触可行域或实测标定来定义任务允许误差，不能为通过测试静默调宽。

当前26个新测试、135个历史MimicGen测试和71个仓库测试通过；模型API仅以传输桩测试，未声称真实远端调用成功。

测试：

```bash
./scripts/humano_python.sh -m pytest humanoid_mimicgen/workflow/tests -q
./scripts/humano_python.sh -m pytest humanoid_mimicgen/tests -q
./scripts/humano_python.sh -m pytest tests -q
```

两个旧测试目录存在同名模块，分别执行避免导入冲突。正式十条件方法比较仍未由一次接口回归替代。


## 已恢复的成功回归

`20261019_2050_workflow_native`：恢复预设站位、成功实验接近时序和有界积分纠偏；取消抓前末端误差门槛，保留底座稳定和实际碰撞/失稳停止。首次抓取成功，1080帧源命令完整执行，保持1.23秒，最低离桌5.575cm。与旧成功实验2584行实测状态逐元素相同，最大差0。出现5ms、2.494N轻擦桌，已保留。

对应请求：`exports/workflow_requests/20261019_2046/request.json`（相对于仓库上级目录）；策略ID：`3e3da840ead27eceae17840eae3c993f80a264b3d2dcd5bb3178f68722370c8c`。同代码与依赖下，使用该请求和策略ID执行prepare可冻结新的复现尝试，不需要重新调用模型。

本次recovery_count=0，不能算回溯成功的物理证据。当前30项新测试、135项历史MimicGen测试及71项仓库测试通过。230个历史源码文件哈希未改变。源动作仍可见，属于轨迹重定向；本次不是十条件方法比较。视觉时序未验收，training_ready=false。


## 固定苹果、不同出发位置覆盖

`experiments/20261019_2110_workflow_start_sweep`（相对仓库上级目录）：6个预声明初始XY位置，朝向和所有控制参数保持不变，每个位置一次。导航/停稳/闭环抓取进入6/6，首次及最终成功均3/6；成功耗时12.92–14.44模拟秒，均值13.67秒。恢复未增加成功例数，暴露了恢复张手碰桌、瞬时支撑门和回撤规划失败，不能宣称已有鲁棒恢复能力。停稳后9.36cm误差的例子仍成功，2.13cm误差的例子失败，不能仅从导航XY误差推断抓取可行性。完整结果、冻结证据、逐例反思和MP4均保留。

可选files.study_case可提供与默认purpose相同字段的研究目的记录，仅覆盖冻结合同的实验描述，不覆盖策略、控制或物理参数；该文件随输入绑定哈希。

## 平滑导航与正常上半身恢复（实验选项）

独立模块 `workflow/motion.py` 提供 `DistanceVelocityProfile`。每个控制周期根据当前目标位移和实测闭合速度计算速度包络，再限制速度向量变化；它限制的是命令加速度，不能保证 SONIC 的实际加速度。通过 `policy.navigation.profile="distance_velocity"` 启用，必须显式提供 `acceleration_m_s2`、`response_s`，最大速度沿用 `speed_m_s`。当前试验为 0.5 m/s、0.5 m/s²、0.6 s，属于待标定配置，不能解释成机器人通用能力。到达原 stop_radius 后将行走未来参考用五次权重逐渐过渡到站立参考，仍使用 encoder mode1；随后沿用原有实测底座稳定条件。SONIC 在线步态 planner 的模式与 encoder 模式是不同字段，分别记录。

`policy.recovery.posture_restore=true` 启用正常姿态恢复：保存第一次停稳后的手臂、腰部关节姿态；每次撤回从当前实测状态规划，到当前根部坐标系下的该上半身姿态。根部、腿、手指、物体在规划中保持当前状态；执行仍是 SONIC Cartesian mode1，不把参考关节写进仿真。必须显式设置 `joint_tolerance_rad`、`position_tolerance_m`、`angle_tolerance_rad`，参考结束后实测连续收敛才重新接近；没有添加抓取前末端误差门。接触阶段积分纠偏保留，恢复阶段不叠加旧积分偏置。

结果：`experiments/20261020_1519_workflow_blended_stop/REPORT.md`（仓库上级目录）。最新实际抓取例 `20261020_1521_workflow_native` 成功，但交接后前倾/额外位移没有改善；因此新导航不作为已验证的优选默认。独立恢复例 `20261020_1515_workflow_native` 两次完成正常姿态恢复，三次抓取仍失败。两选项本轮未联合物理测试，不能合并两项局部结果宣称联合成功。

评审后另修正交接周期重复更新速度、起点已到达时无行走参考、长交接时域截断，并增加实际步态参考记录。36 项 workflow 测试通过；无物理步的真实适配器检查覆盖无行走参考及 0.6/5 秒交接。上述评审后边界修正尚未再次物理运行；保存的成功实验绑定其当时源码快照。

后续 `navigation.standing_orientation="upright"` 可选择世界直立根部参考，保留实测XYZ/yaw和源腿参考。20261020_1539_workflow_native 单次首次抓取成功：交接后一秒pitch从旧版8.90降至5.68度；目标XY误差3.58→5.37cm，未改善；总14.02秒，无监测障碍接触。最新37项workflow测试通过；评审后正常交接修正已进入这次实际运行。完整结果见 `experiments/20261020_1539_workflow_upright_stop/REPORT.md`。仍为实验选项，尚未与新恢复功能联合测试。

## 确定性站位纠偏与批量入口（2026-10-20）

新增 `parking.py`、`batch.py`。运行期不依赖 GPT：批处理仅接受 `decision.mode="deterministic"`，每例冻结输入、源码、模型与控制合同，执行后保存反思，再进入下一例。每例新建控制器，不跨例保留隐状态。修改输入或源码会阻断后续执行。默认不渲染；需要时 manifest 设置 `render: true`。失败保留记录，不自动重跑。

在仓库目录执行本次最终三起点配置：

```bash
scripts/humano_python.sh -m humanoid_mimicgen.workflow batch \
  --manifest ../experiments/20261020_1710_workflow_parking_final/manifest.json
```

manifest 格式：`{"experiment_root":"../experiments","max_cases":3,"render":false,"cases":[{"id":"baseline","request":"baseline/request.json"}, ...]}`。request 路径相对 manifest；request 内文件路径相对 request 文件。输出新的分钟级 batch 目录，包含 `summary.json`、`results.csv`、逐例 plan/prepare/execute、输入哈希和反思。冻结/规划/仿真/渲染分别计时；未测的时间为 null。当前为串行单进程批处理，不声称并行吞吐或模型会话缓存。

`policy.navigation.correction` 提供：

- `method: "walk"`：停稳后按当前几何重新规划小步移动；受共同导航预算及 `max_duration_s` 限制。本轮单例误差没有改善，保留作实验选项。
- `method: "planted_feet"`：根据实测双脚位姿生成新的腿部 IK 参考，再通过 SONIC encoder1 执行。从实测关节开始混合，检查插值路径的脚位姿误差、障碍碰撞和目标静态 COM 支撑范围；不写入实时 qpos，不直接驱动 IK。
- `residual_feedback: true`：下一次参考位置加上实测残差，评价目标始终不变。不能跳过支撑/几何检查。
- 本次显式配置 `tolerance_m=.03, max_corrections=2, minimum_progress_m=.003, max_duration_s=4`。这些是预算和精度配置，非场景专属偏置，也不是通用最优值。优化无进展、不可行或预算结束后继续原抓取流程，不增加抓取前精度否决门。

`navigation.parking` 是独立的预测停车实验选项。先用 `calibrate-stop --experiments <历史实测目录...> --output <新文件>` 拟合停止位移与速度的响应，再显式提供 `evidence_file`、`mode="predicted_stop"`、`trigger_radius_m`、`minimum_samples`、`max_residual_m`。校验场景/物理/源轨迹/SONIC 三模型哈希、证据哈希、不同样本数和拟合残差。本轮单样本残差4.22cm，早期两例碰撞失败；最终质量检查会拒绝该不足标定，最终三例未启用预测停车。

最终评审后版本：三个出发位置误差分别 5.37→3.31、6.61→2.05、6.47→2.90 cm，任务成功2/3，整个批次不含渲染86.42秒。原起点抬升失去接触，恢复后接近不可达。较早中间版本3/3不能当作最终版本成功率。完整证据见 `../experiments/20261020_1710_workflow_parking_final/REPORT.md`（相对仓库目录）。48项 workflow、135项历史 MimicGen、71项仓库测试通过；另有真实适配器零物理步检查覆盖混合起点与时长预算。

稳定判断改用有界时间窗口，保留完整观测日志；10000次窗口选择微基准从1.684秒降至0.00218秒，仅代表该局部操作。全流程耗时仍以实测 batch 记录为准。源动作可见，抓取仍属于轨迹重定向；脚接触与参考几何约束不构成动态可行性保证。默认流程不变，新增选项尚未证明提升总体抓取成功率，训练视频尚未完成完整时序验收。

## 特权状态 GPT 抓取触发对照（可选实验分支）

`policy.grasp_trigger` 启用单独的 GPT 条件抓取。现有 replay 配置不带该字段，执行行为保持不变。批处理的 stance 决策仍要求 deterministic；显式开启本选项后，运行期模型调用会单独计数。

在接近结束前的声明窗口及后续 `grasp_probe` 中，保持手指张开，按固定模拟时间间隔将实测物体/手腕/拇指和食指、中指几何间距、接触、底座稳定信息交给模型。另提供原成功示范的几何参考，以及固定实测手腕/物体、只闭合手指的零物理步几何预测；预测不等于真实接触结果。

模型只返回 `continue / close_now / abort`。`close_now` 当即结束剩余接近，改为实测腕位姿保持，沿用原手指闭合曲线，然后将示范抬升运动相对保持姿态重定向。保留原有积分纠偏、碰撞/失稳/接触验收和恢复流程；probe也保留原有抓前物体运动/提前接触限制。没有强制接回原来的世界手腕轨迹。由此比较的是完整抓取策略，不能把结果单独归因于闭合时刻。

提供者当前为本机已登录的 `codex exec --model gpt-6-astra`，只读分类、JSON Schema输出，发现工具调用即拒绝。可执行文件、连通性测试、提示词、源码和策略绑定哈希。每次实际状态、请求、回答、token用量、事件流和等待时间保存在 `execution/grasp_decisions/`。模型错误、预算耗尽或超时不会静默回退到原策略。模型请求身份可确认，CLI没有独立的服务端模型标识字段，记录为null。

判断期间暂停物理步进，故必须同时报告模拟时间、模型等待墙钟和总墙钟，不能称为真实机器人实时控制。当前试验配置为每0.5模拟秒最多询问一次、每次执行总共最多6次、单次超时90秒；这些是显式采样/计算预算。新策略使用源手指和相对抬升先验，属于带模型触发与新腕保持动作的混合轨迹重定向，不是完全无源轨迹生成。

首次配对入口（新建实验，不覆盖原记录）：

```bash
scripts/humano_python.sh -m humanoid_mimicgen.workflow batch \
  --manifest ../experiments/20261020_1842_workflow_gpt_grasp_comparison/first_pair.json
```

第二起点对应 `second_pair.json`。独立验收检查有效 `close_now` 与确切重试/时间/实测腕位姿绑定，并重新计算触发后的目标、核对执行跟踪记录；原轨迹是否完整执行和原手指序列是否完整执行分别记录。结果详见该实验的 `REPORT.md`。
