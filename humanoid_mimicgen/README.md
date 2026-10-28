# GPT-6 标注与接触轨迹复用

独立的数据生成管线。GPT-6 根据实测状态和接触证据分段，将连续接触技能转换到固定物体坐标系，复用完整接触轨迹，并生成连接段。现有 `cap_sim`、`trajectory_retargeting`、`skill_generation` 和 RoboICL 管线均不修改。

这属于**轨迹重定向**：生成器能访问原始状态、手指轨迹和动作前缀，不是纯经验生成新策略。实现借鉴 [HumanoidMimicGen](https://humanoidmimicgen.github.io/) 的接触技能复用思路；GPT 替代技能分段是本管线的扩展。当前实现没有论文中的全身避障和导航规划，不能称完整复现。

## 一条命令完成离线管线

在 `cap-sim/` 下运行。本机使用已有 HUMANO 环境和已登录的 Codex，不提取登录凭据。所有输出均禁止覆盖。

```bash
CAP_MIMIC_RUN=$(./scripts/humano_python.sh -m humanoid_mimicgen init-experiment)
./scripts/humano_python.sh -m humanoid_mimicgen run \
  --source ../experiments/20261013_apple_straight_sweep/reference_recovery/attempts/001 \
  --provider codex --model gpt-6-astra \
  --out "$CAP_MIMIC_RUN/offline"
```

`init-experiment` 使用实际北京时间 `YYYYMMDD_HHMM_description`，同分钟冲突自动加序号。`run` 的输出必须是尚不存在的子目录。也可使用 `--provider openai`，从 `OPENAI_API_KEY` 读取密钥，通过 [Responses Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses) 调用显式 GPT-6 模型。错误保留并退出，不替换模型。

默认生成历史直线抓取的十个条件：Y±10/15/20cm，桌面和苹果共同±7.5/10cm。`ten_conditions.json` 可另用 `--cases` 指定。桌高在离线产物中仅是场景约束声明，并未构建或执行变体物理场景。

输出包括 `demo.npz/json`、`annotation.json`、模型输入/原始响应/调用日志、每个变体的 `trajectory.npz/json` 与 `geometry.json`、批次汇总和分段计时。批次中途失败会保留已完成项和失败原因。`training_ready=false`，离线通过不能视为物理成功。

## 分步运行和格式

```bash
./scripts/humano_python.sh -m humanoid_mimicgen import --source SOURCE_ATTEMPT --out DEMO.npz
./scripts/humano_python.sh -m humanoid_mimicgen request --demo DEMO.npz --out REQUEST.json
./scripts/humano_python.sh -m humanoid_mimicgen annotate \
  --demo DEMO.npz --provider codex --model gpt-6-astra --out ANNOTATION.json
./scripts/humano_python.sh -m humanoid_mimicgen generate \
  --demo DEMO.npz --annotation ANNOTATION.json --translation 0 0.1 0 --out TARGET.npz
./scripts/humano_python.sh -m humanoid_mimicgen batch \
  --demo DEMO.npz --annotation ANNOTATION.json --out NEW_BATCH_DIRECTORY
```

已有 GPT 标注可通过 `annotate --provider file --response FILE --identity DESCRIPTION` 导入，仍校验源哈希和完整覆盖，明确标记本次没有模型调用。

输入为原生 HUMANO 成功包的 `attempts/NNN`，同时要求对应的 `success/attempt_NNN` 存在。由 `scene.mjb` 的维度拆分 `[time,qpos,qvel,ctrl]`，通过无物理推进的 FK 提取双腕、物体、躯干和根部位姿。保存世界坐标米、wxyz四元数、手指关节弧度及原始实测接触力。

标注为连续半开区间 `[start_frame,end_frame)`，覆盖全部样本；每段含 `kind:transit/contact`、活动手、物体、证据。真实接触样本不得落入 transit，接触段必须有实测接触证据。连续的抓持/抬升段共享首帧锚点：

```text
T_new_hand(t) = T_new_object_anchor × inverse(T_source_object_anchor) × T_source_hand(t)
```

手指数组和时序原样保留，活动手才进行坐标变换；非接触连接采用五次位置插值与最短弧四元数插值。默认保持绝对仿真时间≤5秒的原始前缀。零扰动保留原轨迹以便核验。连接段仅保证位置/姿态连续，不保证避障、速度连续、运动学可达或动力学成功。

导入的手指数据是**实测关节位置，不是原手指指令**；新轨迹中的物体位姿是期望参考，力数组是源示范证据，均不能当作新执行测量。密集接触段保存全部200Hz样本，没有把抬升阶段每帧重新锚定到移动物体。

## 原场景物理冒烟

`native.py` 当前只接受原始苹果场景；变体场景启动会拒绝。执行前必须依据仓库[实验记录规范](../docs/experiment-records.md)填写并冻结 `experiment.json`，运行环境加载检查、资产/物理清单、输入和代码哈希缺一不可。先使用：

```bash
./scripts/humano_python.sh scripts/check_humano_runtime.py --load-models --out RUNTIME.json
./scripts/humano_python.sh -m humanoid_mimicgen prepare-physics --task TASK.json --out NEW_PREFLIGHT_DIR
```

`prepare-physics` 只构造模型，不推进物理。`native.input_hashes(...)` 绑定实际输入；`preflight.evidence_sha256` 必须包含实际执行源码和证据文件。契约旁的 `experiment.sha256` 为冻结文件SHA256。物理启动入口：

```bash
./scripts/humano_python.sh -m humanoid_mimicgen smoke \
  --task TASK.json --demo DEMO.npz --generated BASELINE.npz --annotation ANNOTATION.json \
  --parquet SOURCE_EPISODE.parquet --contract EXPERIMENT/experiment.json --out NEW_PHYSICS_DIR
```

实际模式是 **5秒/250个原始motion token前缀 + SONIC mode1**。前缀后按实测腕位姿重建连接段，50Hz网络推理、200Hz物理及手指指令。运行固定 TRT10.13.0.35、ORT1.23.2、SONIC1.0 CPU，保存实际会话provider。无IK/mode0回退。

显式 `control.contact_lift_stage_names` 决定哪些标注阶段启用抬升接触门；名称含“lift”本身不是物理证据。保存逐步状态、接触、实际模式、控制目标、跟踪误差、原生验收与独立复验，失败不启动变体物理。安全停止优先于同帧任务成功。这个冒烟同时改变了执行接口、前缀后连接和手指参考语义，不能用于单独评价控制器优劣。

## 验证与结果

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./scripts/humano_python.sh -m pytest -q humanoid_mimicgen/tests
```

测试覆盖分段边界/哈希、接触覆盖、固定锚点SE(3)、不交换旋转、手指与时序保留、失败证据及冻结契约拒绝路径；这些单元测试不推进物理。

本次测试记录：[20261015_1339_humanoid_mimicgen](../../experiments/20261015_1339_humanoid_mimicgen/REPORT.md)。图表可用 `python -m humanoid_mimicgen plot --help` 重新生成。

本次实际结果：59项新管线单元测试、12项既有非物理回归通过；GPT-6真实标注与十条件离线几何检查通过。原场景物理0/1成功，8.475s左髋碰桌停止，未到8.86s接触段；未执行变体物理，尚未验证接触段物理复用成功。报告单列两项旧夹爪物理测试误在契约冻结前运行的流程偏差，它们不计入本管线的实验结论。

## 原始 HUMANO 的特权状态分段

`privileged.py` 独立读取原始苹果 Parquet，逐帧恢复完整 `qpos/qvel`，使用原场景碰撞几何和苹果碰撞网格最低点计算离桌间隙。环境初始化显式改为 `mj_forward`，`mj_step/mj_step1/mj_step2` 均被禁止；不是执行动作的新物理回放。原始数据没有接触力，几何接触不冒充实测力。

```bash
./scripts/humano_python.sh -m humanoid_mimicgen.privileged \
  --dataset ../experiments/20261013_cap_ten_tasks/dataset/G1WholebodyTableApplePickTeleop \
  --episodes 0 1 2 \
  --native ../experiments/20261013_apple_straight_sweep/reference_recovery/attempts/001 \
  --out NEW_OUTPUT_DIRECTORY
```

`--native` 可省略；指定时单独提取经过成功包校验的历史执行。输出 `grasp.npz/lift.npz/hold.npz`（仅对有证据的阶段）、`evidence.npz`、半开区间 `segments.json`、逐帧碰撞几何和编译模型。保留原始 `sim.time`、源帧索引、双腕和物体世界位姿（wxyz）、手—物相对位姿、完整状态及原始动作字段。原始数据保留 `action_wbc/action_motion_token/teleop_*_hand_joints`；历史执行保留 `recorded_ctrl` 和实测接触力，不伪造原始 token。

抓取要求持续接触；抬升由离桌和高度增长确认，再回溯离开支撑的起点；保持要求至少0.5秒持续接触、整物离桌≥3cm、垂直速度≤4cm/s、相对平移漂移≤2cm、相对旋转≤0.25rad。保持允许小幅慢速运动。没有保持就返回 `partial`，绝不复制末帧补齐；`rejected` 表示抓取/抬升证据不足。几何接触不证明力闭合，片段不能直接当作新的物理成功。

本次证据和 GPT-6 复核：[20261015_2214_apple_phase_extraction](../../experiments/20261015_2214_apple_phase_extraction/REPORT.md)。原始三条仅有抓取与抬升；历史执行单独提取出1.11秒保持。GPT-6调用与复核脚本保存在该实验目录中，默认提取命令本身为确定性分段，不自动调用模型。此输出含未知接触力的原始数据，尚未接入要求完整实测力的 `save_demo/retarget_demo` 格式，不会填零绕过原验证。

### 交互片段的开始和完成状态

`privileged.segment_phases` 提取的是接触/运动证据区间，不能直接等同于可独立连接的完整交互技能。`boundaries.boundary_clip(evidence, start, end, entry_condition, completion_condition)` 为语义边界导出器：使用真实源帧的**闭区间**，同时保留开始/完成状态、时间、完整状态数组和明确条件。相邻片段允许共享同一个交接状态；原命令行保持源对齐关系，不代表可重复执行交接控制步。可视化必须包含精确首尾，不得只取25Hz采样中最近的帧。

接触开始只是抓取内部事件。应从可连接的接触前状态开始，覆盖闭指和稳定抓持；缺少完成证据仍须标记缺失。保持的完成不等于松手。连续抓持的抓取/抬升/保持可作为一个物体中心交互技能的子阶段，复用时保留统一物体锚点，不在每个阶段随移动苹果重新锚定。

示例更正：[20261016_0913_apple_interaction_boundaries](../../experiments/20261016_0913_apple_interaction_boundaries/REPORT.md)。这是本项目对边界的实现约定，不宣称 HumanoidMimicGen 官方使用相同闭区间格式；本次无新增外部模型调用或物理执行。

### 原场景动作驱动验证

`action_replay.py` 在冻结契约通过后，按200Hz施加保存的43维执行器力矩并推进真实物理。闭区间片段共享的状态只保留一次控制步；完整接近前缀和原有尾段也保留。仅初始化一次 `qpos/qvel/ctrl`，并在 `mj_forward` 后恢复历史 `HybridBackend.reset` 使用的 `qacc_warmstart`。运行中不写参考状态、不插入IK或网络回退。源状态仅用于初始条件和事后误差比较。

```bash
./scripts/humano_python.sh -m humanoid_mimicgen.action_replay run --contract EXPERIMENT/experiment.json
./scripts/humano_python.sh -m humanoid_mimicgen.action_replay verify --contract EXPERIMENT/experiment.json
```

该入口检查契约、执行代码及输入哈希、已加载CPU模型会话证据，拒绝重复执行目录。它是**原场景记录动作回放基线**：实际执行保存的低层力矩，不运行SONIC网络、不验证新场景反馈能力。源动作来自原250-token前缀及mode0/IK历史执行，不能称为新mode1生成。独立验收分别检查原任务成功、标注保持段、接触、碰撞及每阶段覆盖。

[原场景验证记录](../../experiments/20261016_1555_apple_action_replay_warmstart/REPORT.md)：纠正求解器初态后1/1完整执行，2680物理步的qpos/qvel与源逐项一致，保持1.105秒；包含前一次遗漏warmstart导致3.36秒跌倒的失败记录。整个验证批次2次物理尝试，1次成功；不据此报告鲁棒成功率。

### 接触轨迹控制器验证

`contact_controller.py` 增加原场景接触参考验证入口：0–8秒保留已经精确验证的力矩前缀，随后以SONIC mode1执行源实测双腕/躯干参考和原手指命令。预先冻结契约，验证原指令重建及交接状态，禁止模式回退；所有物理步保存，独立验收区分轨迹覆盖与任务成功。

首次实验 `../experiments/20261016_1619_apple_contact_controller/REPORT.md` **未通过**：9.650秒左髋碰桌停止，未完成抬升/保持。不是可用的成功生成器，不支持迁移成功结论。仅新增本独立管线文件，其他管线未改。

```bash
./scripts/humano_python.sh -m humanoid_mimicgen.contact_controller run --contract /absolute/new_experiment/experiment.json
./scripts/humano_python.sh -m humanoid_mimicgen.contact_controller verify --contract /absolute/new_experiment/experiment.json
```

历史执行目录禁止覆盖；新物理实验必须重新准备并冻结完整证据。

### 指定站位与抓取交接对照

`standing_handoff.py` 提供三个显式实验方案：`standing_idle`（标准idle planner + encoder1）、`anchored_legs`（固定腿部参考 + encoder1）、`source_mode0`（历史原期望全身参考 + encoder0）。三组共用精确0..8秒前缀，先进行1..2秒实测站立门，再推进接触轨迹时钟；不固定物理脚部，不自动回退模式。

2026-10-16原场景各一次筛选见 `../experiments/20261016_1656_apple_standing_source_mode0/comparison/REPORT.md`：三组都消除了明显迈步；mode0参考基线唯一通过完整抓取/保持且零碰桌，固定腿部mode1的总骨盆漂移最小。mode0目标表示不同，不能据此声称任意腕轨迹或迁移成功。每组都必须独立冻结新experiment.json；run/verify入口为 `python -m humanoid_mimicgen.standing_handoff`。

### 接触轨迹局部平移验证

`transfer_feedback.py` 在独立管线中增加实时反馈前缀与小范围接触轨迹平移：原 token 前缀 5 秒、mode0 原全身参考 3 秒，随后 mode1 固定公共腿部参考、1 秒张手接近和站立门，再执行平移后的右腕轨迹。姿态与手指命令不变。入口：`python -m humanoid_mimicgen.transfer_feedback run --contract /absolute/new_experiment/experiment.json`；离线 `verify` 使用相同参数。每次运行必须重新冻结完整证据，不能复用历史目录。

`contact_transfer.py` 保留首次历史力矩前缀实验；该开环前缀在苹果平移后数值差异放大并跌倒，不宜作为变化场景的执行前缀。最终闭环配对见 `../experiments/20261016_1853_apple_transfer_feedback_y_plus_2cm/comparison/REPORT.md`：原位置与世界 Y +2 cm 均抓住并抬起，但均未通过完整冻结验收。平移组稳定保持 0.945 秒，原生 9 cm 中心抬升阈值未达到。每条件一次，不代表泛化成功率；仍属于轨迹复用，不是策略生成。独立安全测试 90 项通过。

### HumanoidMimicGen 规划组件复现

`skill_planning.py` 实现技能先后/同步约束、固定锚点 SE(3) 变换、动作噪声与干净标签分离，以及有预算的关节空间 RRT。`stationary_planning.py` 使用真实 MuJoCo 几何，优先仅手臂、再手臂加腰求解接近起点，保持腿部与根部不变；无法找到解时明确失败，不隐式改动站位。

```bash
./scripts/humano_python.sh -m humanoid_mimicgen.reproduce_components \
  --workspace /absolute/workspace \
  --out /absolute/workspace/experiments/YYYYMMDD_HHMM_hmg_components
```

输出目录必须是实际 Asia/Shanghai 创建时间且未存在。该入口只有 `mj_forward`，禁止 `mj_step`，不调用 SONIC，不生成物理训练样本。源码依据论文重实现，CPU IK/RRT 不是官方 cuRobo；尚缺导航执行、全身站位规划、持物碰撞与训练评估。现有冻结物理执行器保持不变。

实际结果见 [20261016_1926_hmg_components](../../experiments/20261016_1926_hmg_components/REPORT.md)：12 个真实场景几何检查中 6 个找到入口连接；Y−15 cm、Y−20 cm 的关节直连碰撞，RRT 找到通过离散碰撞检查的绕行。这里只是入口几何结果，不是动态执行或任务成功率。

### 实测规划接近 → mode1 → 接触复用

`planned_approach.py` 为显式 Y−15 cm 实验执行器：原250token/3秒mode0前缀后先站稳，从实际qpos求解arms-only IK/RRT，在独立MjData中FK，限速后输入mode1，经过到位门再复用接触片段。启动前必须有完整冻结的新实验合同：`python -m humanoid_mimicgen.planned_approach run --contract /absolute/new_run/experiment.json`，离线验证使用 `verify`。

[20261016_1943_apple_planned_approach_y_minus_15cm](../../experiments/20261016_1943_apple_planned_approach_y_minus_15cm/REPORT.md) 完整执行并抓起苹果，最高离桌6.30cm。实际中指擦桌，接近路径不能标成无碰撞；抬升/保持无桌面接触。训练候选按阶段及碰撞掩码保存，旧高度/稳定时间门限仅作诊断，视觉训练验收仍未完成。新执行器不修改历史冻结执行器，其他管线保持不变。

### 可达性优先的站位选择与实测导航

`stance_selection.py` 用特权状态生成站位候选，先过滤入口 IK 与离散碰撞，再按移动距离、转角和面向物体程度排序。选择和导航命令计算均为可复用代码，不调用 GPT；几何可行不等于动态可执行。

新实验冻结后，使用 `./scripts/humano_python.sh -m humanoid_mimicgen.stance_execution run --contract /absolute/new_run/experiment.json`，离线检查用 `verify`。该执行器取消原始动作前缀，导航使用 encoder1，随后计划从实测姿态接近并复用接触片段。`stance_execution_fullbody_nav` 是单独声明的诊断接口：导航使用 `navigation_reference.py` 产生的实时全身参考和 encoder0，到位后才切换 encoder1；不使用原始 token，不隐式回退。

[20261016_2201_apple_fullbody_nav_y_plus_10cm 多位置报告](../../experiments/20261016_2201_apple_fullbody_nav_y_plus_10cm/comparison/REPORT.md)：Y−15、+10、+15 cm 三组 encoder1 导航均在 16 秒超时；单独 Y+10 cm encoder0 诊断最终误差降到 7.98 cm，仍未通过 2.5 cm 到位门。四次均未执行抓取，不能据此计算接触复用成功率；无新成功训练数据。独立单元测试 106 项通过，动态行走与停止控制仍需解决。

### 0.5 m/s 导航与小偏差上半身补偿

`fast_navigation.py` 发出固定0.5m/s世界速度向量，进入10cm站位邻域后请求停止；这不是可达性或任务成功判定。`upper_body_approach.py` 从实测站姿先求手臂、再手臂加腰的IK/RRT，根部和腿部只在规划中固定，通过mode1执行FK参考。`stance_execution_fast.py` 保留原累计位移站稳门供对照；`stance_execution_fast_window.py` 用最近0.3秒位移判断站稳，速度、脚部受力和手腕误差阈值保持不变。都使用新冻结实验入口 `run --contract ...` / `verify --contract ...`，无原token或mode0前缀。

[20261017_1353_apple_fast_nav_y_plus_10cm](../../experiments/20261017_1353_apple_fast_nav_y_plus_10cm/REPORT.md) 在1.50秒进入停止邻域，但累计刹车位移超过旧站稳门。单独对照 [20261017_1355_apple_fast_window_y_plus_10cm](../../experiments/20261017_1355_apple_fast_window_y_plus_10cm/REPORT.md) 在4.06秒通过实测站稳，距原选定站位约5.6cm，arms-only规划找到入口连接；执行到10.03秒中指碰桌停止，尚未抓取。腰部补偿在此例未被选用，不能宣称已验证。112项独立单元测试通过；本轮没有新增成功抓取训练样本。

### 可达性优先与避碰导航流程复现

`reachability_navigation.py` 提供 `select_navigation`、`plan_base_path`、`path_command`：先在实际当前位置尝试手臂、手臂加腰IK及接近连接；成功直接跳过导航。否则检查18个附近站位/朝向候选，要求操作入口与接近连接可行、底座几何路线无碰撞，再按路线长度、转角和面向程度排序。直路受阻时使用有预算的平面A*，检查路径边并压缩路点。`reachability_execution.py` 用SONIC mode1跟随路线，保持0.5m/s和10cm终端停止邻域，再从实测站姿重新规划上半身。

入口为 `./scripts/humano_python.sh -m humanoid_mimicgen.reachability_execution run --contract /absolute/new_experiment/experiment.json`；离线核验用 `verify`。每次物理运行仍需新建完整冻结记录，不复用历史目录。

这是HMG规划流程的CPU实现，非原版cuRobo/Homie复现；腿部构型搜索仍是有限站位候选，平面静态路线不是动态足步碰撞保证。原接触序列可访问，属于轨迹重定向。没有修改其他管线或冻结的历史执行器。

118项单元测试通过。真实几何检查 [20261017_1605_hmg_navigation_geometry_detour](../../experiments/20261017_1605_hmg_navigation_geometry_detour/REPORT.md) 在实测可达站姿选择current跳过导航，并为跨桌面区域请求找到绕桌A*路径；该绕障路线尚未物理执行。首次几何请求终点位于桌内，明确被拒绝，保留于20261017_1603_hmg_navigation_geometry。

两次物理测试：[20261017_1602_apple_reachability_nav_y_plus_10cm](../../experiments/20261017_1602_apple_reachability_nav_y_plus_10cm/REPORT.md) 走停成功后接近碰桌；[20261017_1604_apple_reachability_nav_y_minus_15cm](../../experiments/20261017_1604_apple_reachability_nav_y_minus_15cm/REPORT.md) 走停并执行arms+waist接近，无碰桌，但最终右腕误差5.75cm，站立到位门超时。两次均未执行抓取，不构成接触复用成功率。

### 固定站位的接触复用范围评估

`contact_range.py` 独立准备13个XY偏移条件，复用历史冻结执行器；只平移苹果初始位置和右腕位置参考，保持机器人初态、腿部参考、腕部朝向和手指时序。明确保留5秒原SONIC token与3秒mode0全身参考前缀，此后mode1执行接近和接触。属于轨迹重定向，不能直接推广到实际导航后的任意站姿。

在仓库根目录使用 `./scripts/humano_python.sh -m humanoid_mimicgen.contact_range prepare x_plus4` 创建新的分钟级实验目录并完成预检、冻结。仅当返回 `ready: true`，才使用 `./scripts/humano_python.sh -m humanoid_mimicgen.transfer_feedback run --contract <新目录>/experiment.json`；然后依次运行同模块 `verify --contract ...`、`contact_range evaluate <新目录>` 和 `render_standing <新目录>`。冻结前检测到原前缀手物穿透的条件不启动物理。`evaluate` 的反思记录只写一次，不要重复执行。

[20261017_1722_apple_contact_range_origin 批次报告](../../experiments/20261017_1722_apple_contact_range_origin/comparison/REPORT.md)：13个条件、11次物理运行、10次接触执行；10次完成，其中4次无碰桌、6次有擦碰。3×3网格（X/Y各−4、0、+4cm）九点均完成，X+8cm也完成且无碰桌。Y+8cm站稳超时，X−8cm/Y−8cm前缀几何冲突未启动。每点仅一次，未证明连续可行半径。原点全部状态与历史基线完全一致；122项单元测试通过。

报告内提供范围图、双视角视频索引、同步对比视频及 `contact_reuse_envelope.json` 已测试偏移清单；该清单禁止插值推断。双视角阶段样本已检查，原速时序视觉验收尚未完成，所有样本 `training_ready=false`；物理任务完成、碰桌和视觉质量分别保存。

### Y+8cm 延长站稳等待对照

`transfer_feedback_wait.py` 为独立执行器，只将站稳截止从第11秒改到14秒（等待2→5秒）；姿态0.2rad、位置2.5cm、连续0.3秒及其余物理和接触条件保持不变。`contact_wait_range.py prepare y_plus8` 生成完整冻结实验；预设后续+10/+12cm仅在前一位置完整完成时启动。总仿真预算相应扩展至19.4秒，不截断接触后半段。

[20261018_1847_apple_contact_wait5_y_plus8](../../experiments/20261018_1847_apple_contact_wait5_y_plus8/REPORT.md)：1次运行仍在14秒站稳超时，未进入抓取。前11秒全部状态与旧实验完全相等；额外3秒右腕姿态误差12.19–13.50°，高于11.46°门限。最终位置误差1.11cm、姿态12.38°，无碰桌。未继续扩大范围；只延时不足以通过入口门，不代表抓取本质不可行。123项单元测试通过，完整合同、运行时和视频审计已保存，中文MP4见报告。

### 直接抓取对照与Y方向扩展

`transfer_feedback_direct.py` / `contact_direct_range.py` 按用户明确要求，在1秒接近结束（仿真第9秒）直接执行接触轨迹，跳过整个站稳/位姿准入门；碰桌、跌倒、接触丢失和抓前物体移动停止条件不变。该行为在新冻结合同中显式声明，不修改其他执行器。prepare入口使用 `humano_python.sh -m humanoid_mimicgen.contact_direct_range prepare y_plus8`；新目录冻结且ready后，使用 `transfer_feedback_direct run/verify --contract ...`，之后 `contact_direct_range evaluate <目录>`。

[20261018_1904_apple_contact_direct_y_plus8 三条件报告及中文MP4合集](../../experiments/20261018_1904_apple_contact_direct_y_plus8/comparison/REPORT.md)：Y+8/+10/+12cm各一次，3/3完成抓起保持、无碰桌，保持最低离桌分别5.45/5.37/5.79cm。Y+8cm前9秒与等待组全部状态相同；其原生成功通过，+10/+12cm未达原生高度门但满足已声明的放宽任务完成标准。本次改变了接触开始时刻并跳过整个准备门，不能称仅放宽姿态角度的实验。12cm为已测成功采样点，不是连续范围或最大边界；视觉训练验收未完成。124项单元测试通过。

### 导航后直接接触验证

`reachability_direct.py` 保留原0.5m/s导航、10cm停止邻域、走停稳定门、实测arms/waist IK/RRT接近和所有物理停止条件，仅在接近参考结束时跳过postapproach位姿准入而直接抓取；冻结合同需显式声明 `postapproach_pose_gate_bypassed=true`。原执行器未修改。

[20261019_1650_apple_nav_direct_y_minus_15cm](../../experiments/20261019_1650_apple_nav_direct_y_minus_15cm/REPORT.md)：为隔离旧到位门问题选择之前接近无碰桌的Y−15cm，而不是此前接近已碰桌的Y+10cm。1次物理、1次真实抓取、0次完成。4.18秒停稳后站位误差9.02cm，10.60秒直接抓取时站位误差10.81cm、右腕实际目标误差6.51cm；12秒因无手物接触未过抬升门。最长手物接触仅0.035秒，抓取中短暂擦桌峰值7.71N，未抬起。前10.60秒全部状态与历史基线完全相等。实际全程mode1，原token/mode0均0；125项测试通过，中文MP4见报告。结果区分导航误差与执行腕部跟踪误差，不能由此认定所有附近站位均不可抓，也不能直接套用固定源站姿的XY平移范围。


## 2026-10-19：导航后接触轨迹复用验证

[20261019_1820_apple_nav_wrist_feedback_y_minus15](../../experiments/20261019_1820_apple_nav_wrist_feedback_y_minus15/REPORT.md) 完成真实导航约40cm后抓取、抬升和保持，最终配置1/1、单位置。入口 `reachability_wrist_feedback`，独立保留历史执行器。全程SONIC mode1，无原token/mode0前缀或状态重置。实际停稳后重规划接近，原接触目标/手指时序复用；右腕XYZ积分反馈gain0.5/s、cap2cm修正实际跟踪偏差。

导航停止范围仍10cm；本次停止请求误差9.14cm、抓取时2.79cm。保持1.105s、最低离桌5.58cm；有5ms/2.49N轻擦桌。135项单元测试通过。此为轨迹重定向成功试验，不是跨位置鲁棒性或训练视频质量验收。报告附冻结准备入口、验证、双视角中文MP4及实测证据；正式十条件对照仍需独立执行。
