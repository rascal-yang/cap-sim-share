# Replay 导航脚本

入口：`scripts/humano_python.sh -m humanoid_mimicgen.workflow.navigation`。

本模块将 replay 站位关系提取、新场景站位计算、整段操作参考的运动学筛选、加减速与停车控制落成 Python。正常流程不调用 GPT。现有工作流默认行为保持不变，通过 `navigation.profile="replay_navigation"` 显式启用。

## 能力与边界

- `extract`：从成功记录的实测 root/qpos/qvel、对象与环境几何中提取稳定入口窗口、对象相对站位、桌面高度/桌沿关系、关键帧与证据哈希。球形对象可使用有证据的环境朝向，避免任意 object yaw。
- `plan`：优先检查当前位置，再在任务参考系内枚举有界候选；检查地面支撑、身体与操作物体的环境碰撞、接近段、整个参考操作及相邻样本插值。规划只调用运动学，不推进物理。
- `execute`：在现有冻结合同和指定运行时下执行导航，平滑加减速和转向，持续检查稳定性，最后从实测状态重新检查操作参考。默认结果是 `navigation_handoff_only`，不会自动执行抓取、按压或落座；最终 `terminal.npz` 是审阅证据，不是供下游重置机器人继续执行的指令。显式连续操作选项见下文“全身操作交接”。

新环境若需要改变手臂或全身操作轨迹，应先由已有轨迹生成管线生成对应参考。此版本原生检查器使用完整关节参考重定位，**不会为任意桌高自动重新求解整段操作 IK**；改变桌高而原参考无法维持任务关系时返回 `operation_relation_changed`。这与导航失败、物理执行失败分别记录。

当前为轨迹重定位方法，原动作序列可访问。不能用离线检查通过推断位置泛化成功率。历史文件只读，无改写。

## 提取：可直接运行的真实记录示例

从仓库根目录执行；`--out` 必须尚不存在：

```bash
scripts/humano_python.sh -m humanoid_mimicgen.workflow.navigation extract \
  --manifest humanoid_mimicgen/workflow/tests/fixtures/navigation/apple.json \
  --options humanoid_mimicgen/workflow/tests/fixtures/navigation/extraction_options.json \
  --out /tmp/apple_navigation_prior.json
```

四份 manifest 指向本机冻结实验，不是虚构的成功样本。换机器需安装原始证据并重新建立 manifest，不能只修改已生成 prior 的路径而保留旧哈希。

manifest 的关键字段：

```json
{
  "files": {"model": "scene.mjb", "trajectory": "execution/trajectory.npz", "result": "result.json"},
  "state_key": "state",
  "entry_index": 1384,
  "end_index": 1875,
  "success_field": "native_success",
  "success_criterion": "native",
  "support_distance_m": 0.002,
  "bindings": {
    "object": "apple_001", "root_body": "pelvis",
    "feet": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "ground_geoms": ["ground"], "support_geoms": ["table_geom"],
    "frame": {"kind": "world_heading", "yaw_rad": 0, "evidence": "已知桌沿朝向"},
    "operation_bodies": ["right_wrist_yaw_link"],
    "contact_policy": [], "navigation_allowed_geom_pairs": []
  }
}
```

索引是归档**实测状态行号**，不是随意互换的 policy tick；当前原生格式为 `[time, qpos, qvel]`。`state_key` 显式支持已有 `state` 或 `samples` 键。缺少对象/阶段信息返回 `needs_annotation`；不存在合理稳定入口返回 `unsupported_entry_pattern`。稳定窗口之后若还继续走动，不会被误用为操作站位。

`contact_policy` 使用 `{"start": 行号, "end": 行号, "allowed_geom_pairs": [["具体geom A", "具体geom B"]]}`，区间含两端。物体/桌面支撑、手/物体、落座接触必须显式声明。几何放行仍受 `support_distance_m` 接触带限制，不能用允许接触掩盖深穿透。示例中空列表意味着保守拒绝未知接触，不能据此认证抓取段可行。导航期间已有物体/支撑面接触单独用 `navigation_allowed_geom_pairs` 声明。

旧记录不含足底力时间序列时，支持证据标为 `geometric_contact_not_force`；执行时仍需真实双足接触力和连续稳定条件。

## 新场景规划

scene JSON 明确提供新场景 `model`、`initial`，以及当前环境的任务系朝向。`initial.npz` 含 `qpos`；完整执行还需现有工作流要求的初始状态字段。对象位置从模型 FK 读取，不能只填一个脱离实际场景的目标坐标。

```json
{
  "files": {"model": "scene.mjb", "initial": "initial.npz"},
  "frame": {"kind": "world_heading", "yaw_rad": 0, "evidence": "当前桌沿朝向"},
  "search": {
    "max_candidates": 7, "planning_timeout_s": 30,
    "offsets_xyyaw": [[0,0,0],[-0.03,0,0],[0.03,0,0],[0,-0.03,0],[0,0.03,0],[0,0,0.05]]
  },
  "native_options": {
    "path": {"resolution": 0.06, "padding": 0.45, "max_expansions": 2500},
    "collision_tolerance_m": 0.001, "support_distance_m": 0.005,
    "corridor_margin_m": 0.04,
    "sample_translation_m": 0.02, "sample_angle_rad": 0.04,
    "max_checks": 50000, "planning_timeout_s": 30,
    "relation_tolerance_m": 0.02, "relation_tolerance_rad": 0.04
  }
}
```

这些数值是显式算法配置示例，**没有标定成机器人通用能力**。走廊与操作路径采用有限几何采样，不是连续空间或动力学安全证明。每个候选保留拒绝原因；用尽预算返回失败/不确定，不能当作不可达证明。

```bash
scripts/humano_python.sh -m humanoid_mimicgen.workflow.navigation plan \
  --prior /tmp/apple_navigation_prior.json --scene /path/to/new_scene.json \
  --out /tmp/apple_navigation_plan.json
```

## 控制接口与冻结执行

`RouteController(route, limits).step(xyyaw, [vx, vy, yaw_rate], dt)` 返回世界系平面速度、绝对世界 yaw、路点、到达状态和审计值。路点坐标为世界系米/弧度。限制包含速度、线加速度、响应时间、角速度、角加速度、到达/路点/走廊半径、低速门限、总时限和最大观测间隔，全部显式提供。

直线路点连续通过；未验证转弯空间时在拐点前平滑降至低速再转向，避免直接切角。微小角度采用离散时间制动包络避免参考过冲。目标点放在到达区域内部，避免永远逼近判定边界。

接入现有 workflow request 时：

1. `files.navigation_plan` 指向计划文件；`files.model/initial/reference_states` 的字节哈希须匹配计划。原 NPZ 不必重命名键或修改内容。
2. `policy.navigation.profile="replay_navigation"`、`navigation_only=true`、`plan_file` 为计划绝对路径；提供 `route_limits` 和明确的 14 维 `fingers_rad` 导航手指命令。站立参考取本次初始站立姿态，不用落座/抓取接触姿态充当行走站姿。
3. `control_evidence_file` 指向 request files 中的停车标定证据键，`calibration_quality` 提供 `minimum_samples` 和 `max_residual_m`。沿用现有停车标定的严格机器人、模型、物理、来源与控制参数 scope 校验；速度不得超出标定范围，响应时间不得短于拟合值，停车残差不得超过到达半径。
4. 继续使用原 workflow `plan`、`prepare` 创建策略和冻结合同。没有匹配标定只能离线分析，不能自动执行；本次未产生新的物理标定证据。
5. 在冻结合同记录该计划、路线及全部证据后执行：

```bash
scripts/humano_python.sh -m humanoid_mimicgen.workflow.navigation execute \
  --plan /path/to/navigation_plan.json --experiment /path/to/frozen_experiment
```

不允许隐式 IK/mode0/token 前缀替代。当前修正预算为零：到达后的实测操作检查失败即返回 `handoff_not_ready`，不进行不受控微调。执行的 `navigation_success` 与 `task_success` 分开，后者保持未知。后续独立经验收集器可以读取结果/审计，纯规划记录仍为零物理步。

## 当前验证

真实离线证据见 `replay-navigation-offline-evidence.json`：苹果、按钮、椅子提取成功，盘子的现有入口缺少合适的稳定前置窗口，返回明确拒绝。按钮保留 `button_palm_mesh_v1` 审核口径，不写成原生成功。

```bash
scripts/humano_python.sh -m pytest -q
scripts/humano_python.sh -m pytest -q humanoid_mimicgen/workflow/tests
```

四份历史集成测试在原证据可访问时实际读取并计算；缺少本机归档时明确 skip。当前没有执行新物理实验，没有证明四任务端到端成功率提升；连续操作接口的代码/离线测试不等于动力学验证。

## 全身操作交接（2026-10-26）

在同一冻结 workflow 中设置 `policy.navigation.navigation_only=false` 并提供 `policy.whole_body_operation`，可以连续执行导航 → 稳定 → 全身入口过渡 → 条件推进操作。`true` 保持原有导航独立验收。

`whole_body_operation` 的必需字段：

|字段|含义|
|---|---|
|transition_s|从实测全身状态到重定位参考入口的五次插值时长；按已有approach速度限制检查，不满足即拒绝|
|position_m / orientation_rad|身体部位对齐误差阈值；米/弧度|
|continuous_s / timeout_s|条件连续满足时长、每个交接门最长等待；秒|
|alignment_bodies|必须覆盖根、躯干、双脚、prior操作部位；运行时还检查物体相对参考的位置/朝向|
|open_fingers_rad|显式14维开放手命令；必须与参考入口手指命令一致|
|object_force_stop_N|机器人与物体总接触力停止上限；导航/过渡/操作每物理步持续检查|
|contact_gates|按frame排序的`{frame, groups: [[body_name,...],...], minimum_force_N}`；所有组分别达到阈值才放行|
|geometry_gates（可选）|额外纯几何交接帧。执行器还自动检测每只手离开开放命令的帧，插入合手前几何门|

所有阈值必须从任务证据/预算声明，代码没有通用经验默认值。阶段索引为源物理状态行号，遵循原`command_pose_offset`；禁止把50Hz手指数组直接搭配200Hz状态数组。`targets.fingers`必须逐源行对齐；转换源数据时显式展开原始手指命令，不在执行器里猜测采样率。整个半开阶段区间映射后必须等于导航plan检查过的闭区间。

`requires_contact`阶段必须有对应接触组门。例如苹果声明拇指、食指两组；盘子可声明双手；椅子可声明髋部与椅子的接触。接触组与对齐部位均在启动前检查模型归属。当前`requires_clearance=true`明确拒绝（该通用接口尚未统一各任务净空观测），任务成功按原生任务验收独立计算。按钮本接口报告原生验收，不自动沿用旧button_palm_mesh_v1口径。

运行时使用`NativeEvaluator`同一变换计算完整qpos参考，FK生成腕/躯干目标，SONIC mode1接收匹配的腿关节参考、根朝向与腕/躯干目标。**mode1不直接跟踪root XYZ或全部关节**；完整参考关系、实测位置门控并不等于全自由度动力学可控性。参考暂按当前帧保持，不预览越过尚未放行的接触阶段。行走仍由经过标定的实时导航控制器完成。

过渡插值只写FK副本并检查支撑/碰撞；操作前再从真实停车位置检查整段参考。源手指命令改变、门控释放和末帧只在50Hz策略周期放行，物理仍按200Hz推进。末帧实际发送并保持一个完整控制周期后才记轨迹完成；`source_index`记录实际锁存帧，`requested_source_index`单独记录请求帧。`execution/whole_body_operation.json`记录门控和推进情况。全过程不重置物理或SONIC历史、不使用原token前缀、不调用GPT补动作。

`result.scope=whole_body_operation_native`时，`task_success`要求安全完整执行且原生任务验收通过；`navigation_success`与其分开。方法仍为参考轨迹重定位。

本轮新增18项测试，包含真实苹果失败档案在frame340约10cm错位时阻止闭手，以及禁止推进物理的MuJoCo FK桥接和200Hz/50Hz节拍测试。未启动新物理实验：现有四任务+10cm预检仍为`no_feasible_stance`，且缺少匹配的停车标定；需要准备接触前入口、有效几何计划和标定证据后才能测试。原固定夹持入口（苹果1384行）已闭手，应改用有证据的合手前入口（此前候选1098行），重新执行几何规划；不能只修改索引跳过筛选。

## 显式单次诊断执行

正常request默认`navigation_execution="calibrated"`，仍要求完整匹配的合格停车标定。为采集当前控制器实际运行证据，可显式指定`navigation_execution="uncalibrated_diagnostic"`与非空`diagnostic_reason`。标定artifact必须结构完整、文件哈希正确、samples重算model完全一致；诊断仅允许scope/质量/范围不适配，不允许损坏或伪造证据。策略/合同/result保存`navigation_admission`，明确calibrated=false、diagnostic_only=true，不能宣称正常标定通过或training-ready。

该入口不改变RouteController、速度/响应参数、停车/支撑门、接触/摔倒停止条件或原生成功判定；每个冻结合同一次尝试。历史模板的已修改仓库Python文件，仅可由其冻结snapshot按原SHA精确恢复归档依赖，新代码另行绑定。物理资产改变仍拒绝。旧flat physics文件按实际model arrays、PD、dt和reward stride逐项核验，不修改旧文件或物理值。

真实诊断记录（2026-10-27）：`20261027_1945_apple_full_navigation_diagnostic`与隔离CLI进程复测`20261027_1956_apple_full_navigation_isolated`均实际执行288物理步/1.44s后触发navigation_corridor_violation。末路线偏离26.485mm、配置边界25mm，两次状态逐位相同；停车交接/抓取未执行。每例72token/288步记录完整，独立decoder重建身体目标逐位相等。该证据证明启动/记录链路可实际运行，**不证明导航抓取全程成功或标定通过**。速度、响应、走廊与接触门未调参。准备时需保留task.resources.initial_state中的真实SONIC history种子，物理初态通过NativeAdapter独立传入，不能将无history的qpos文件覆盖该资源。
