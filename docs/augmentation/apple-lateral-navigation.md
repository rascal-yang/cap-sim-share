# 苹果横向站位与操作规划

显式模式 `apple_lateral_v1` 从修复成功 replay 的双脚支撑窗口生成站位候选，为每个候选重新求解操作参考。它保留源骨盆运动作为姿态偏好，约束双脚和左右腕，并检查整段姿态、插值、碰撞及参考速度。正常路径没有 GPT 调用。

本功能属于 **trajectory_retargeting**：规划器可访问原动作序列；新导航/操作路径使用 SONIC mode1，原 token 前缀为 0。当前源 `20261025_1719_tableapplepick_token_prefix_mode1` 含 346 步原 token 和 preencoder 姿态反馈修正，两者不能描述成同一控制链。

## 离线入口

从仓库根目录运行，输出必须不存在：

```bash
scripts/humano_python.sh -m humanoid_mimicgen.workflow.navigation lateral-extract \
  --manifest humanoid_mimicgen/workflow/tests/fixtures/navigation/apple_lateral.json \
  --options configs/navigation/apple_lateral.json --out /tmp/apple_support_prior.json

scripts/humano_python.sh -m humanoid_mimicgen.workflow.navigation lateral-plan \
  --manifest humanoid_mimicgen/workflow/tests/fixtures/navigation/apple_lateral.json \
  --options configs/navigation/apple_lateral.json \
  --initial /absolute/path/to/successful/source/initial.npz \
  --offset-m 0.05 --out-dir /absolute/path/to/new/case
```

`--initial` 必须与 manifest 中成功源的初态匹配，包括 qvel、ctrl、历史等已有字段。只平移苹果沿桌沿切向的位置。当前桌面切向为世界 +Y，法向 +X；不使用球形苹果的 yaw 来定义左右。相对成功源的偏移会写入 `scene_change`，原场景中的既有 X 偏移保持不变。

`lateral-grid` 使用同样参数但不带 `--offset-m`，顺序记录 0、−5、+5、−10、+10、−15、+15、−20、+20 cm。每个 case 有独立 `plan.json`，父目录有逐条件 progress 和最终 summary。顶层实验路径由调用者按工作区规定提供实际 Asia/Shanghai 创建时间的 `experiments/YYYY-MM/DD/YYYYMMDD_HHMM_<description>`。

这些命令只运行 FK/IK，不推进物理。grid 的 completed 表示全部规划条件已记录，**不表示所有计划可行或抓取成功**。拒绝、超时、无有效场景会保留证据。`search_incomplete`/`candidate_limit_reached` 表示有限搜索没有遍历所有候选，不能据此证明不可达或全局最优。

## 输入证据与产物

- `hand_commands.npz` 的 left/right 命令使用同记录 controller.npz 的实际时间戳与 source state 时间作零阶保持对齐。缺覆盖、时间间隙、入口已闭手或源未成功时拒绝，不盲目重复四次。
- manifest 明确指定 entry、closure、first_contact、lift、end 的实测状态行及证据依据。接触/抬起的诊断分段阈值不替换原生成功判据。
- 支撑 prior 保存多个窗口的双脚位姿、全身代表帧、root 相对双脚关系和窗口变化量。脚踝 FK 与几何支撑不冒充真实足底力；执行仍需要原有连续稳定/支撑门。
- 成功计划保存新 initial.npz、options.json、prior.json 和 generated/reference.npz＋manifest.json。生成包包含 qpos、time_s、source_rows、fingers、已检查的 bridge_qpos/bridge_time_s；源实测轨迹单独保留，不制造“新成功 replay”。
- 接触策略显式继承冻结声明，仍受浅接触带限制。没有通用的手/桌豁免或自动放宽门限。

`configs/navigation/apple_lateral.json` 是有上限的开发配置，列出来源与未标定参数；它不是机器人能力证书。当前默认`search.mode="replay_first"`：将最近成功支撑窗口与苹果同步横移作为唯一首选，仅在失败后尝试−2.5cm、+2.5cm两个邻近站位，首个可行方案通过后立即停止。先筛入口/闭合/抬起，再验证完整动作；不做全候选最优排序。旧网格通过`mode="grid"`及`max_candidates=21`显式启用。失败中保留具体约束和数值原因。

全身求解使用 MuJoCo 位姿 Jacobian，并用有限差分回归检查导数。全身控制关节和手指命令保持模型范围，数值容差 1e-7。用于碰撞预测的**源实测手指姿态**保留显式 0.0033 rad 软限位偏差容差（该源最大观测为 0.003216528 rad），这不扩大任何手指命令或物理关节范围。全段与桥接记录逐行/逐段残差、速度和求解终止状态。

排序使用全段最小位置残差余量，最低要求由 `minimum_position_margin_m` 显式声明（初始值 0，非标定能力）。A* 展开数或 IK 迭代数耗尽均记为预算耗尽；整轮 deadline 覆盖源 FK、求解、路线和产物落盘。保留已有验证候选并标明搜索未完成。

## 绑定现有 workflow 执行

继续通过既有 workflow `plan/prepare/execute` 和冻结 experiment.json；`navigation.profile="replay_navigation"`、`navigation_only=false`。`files.reference_states` 指向**源实测轨迹**，不要替换成生成参考；`files.navigation_plan` 与 `navigation.plan_file` 指向新 plan.json。场景模型/初态与该计划一致。

阶段使用源状态行，`command_pose_offset=0`；范围必须恰好等于生成包 source_rows。`files.targets` 的 fingers 必须按同一 controller 时间映射成逐物理行命令，执行前逐值核对。`whole_body_operation.transition_s` 必须等于生成桥接时长；其他身体对齐、接触门、速度/安全/原生验收按已有工作流显式配置。

到位并满足原连续稳定条件后，执行器从实测脚位、机器人和苹果状态重新求解一次。记录仿真暂停墙钟、实测输入、派生生成包及哈希，恢复前再次核验状态没有变化。脚位无法满足操作约束、物体移动、预算耗尽均拒绝交接。第一版无自动二次微调和控制模式兜底。WholeBodyOperation 消费派生包，不再次平移源动作。

生成计划**不提供停车标定**。现有严格 scoped calibration preflight 继续生效；匹配的横向导航和停车证据缺失时返回 needs_calibration，不能直接启动抓取。无新物理结果时不报告泛化成功率。正式历史 IK/mode0 对 SONIC 比较仍需规定的十条件匹配实验。

## 验证

```bash
scripts/humano_python.sh -m pytest -q
scripts/humano_python.sh -m pytest -q humanoid_mimicgen/workflow/tests
```

测试包含命令时间对齐、固定脚/移动骨盆、关闭入口拒绝、真实 MuJoCo 支撑约束 IK、越界场景、闲置手碰撞、速度和预算拒绝、生成包篡改、执行消费与实测脚位再求解，以及旧导航/whole-body 回归。真实九位置离线结果单独归档；几何检查与单元测试不代表动力学成功。

已完成首轮离线验证：`20261027_1136_apple_lateral_navigation_offline`，0、±5、±10、±15、±20 cm 九条件均找到通过检查的计划；所有搜索均未穷尽，物理步数 0。详见 [实验报告](/home/kinetra-4090-01/code/zijian/experiments/2026-10/27/20261027_1136_apple_lateral_navigation_offline/REPORT.md)。默认测试 108 passed，workflow 测试 143 passed（运行代码 `7e392e1`）。

## 优先原样复用与完整动作记录

`reuse_source_first=true`时，先检查同步平移后的每一帧源参考。整段可行则直接保存原逐帧动作，不重插值；必要时才解IK。显式小改上限为身体关节0.15rad、root平移0.02m及旋转0.05rad（开发边界，未标定为能力）。每帧及插值后均核验，实际最大变化写入`reference_modification`。手指命令继承源值；现有实测闭合等待及末端保持照常单列。

直接抓取和连续workflow执行都会输出`execution/sonic_actions.npz`和`sonic_actions.json`：包含每次实际策略tick的SONIC body token、mode、时间，以及每个物理步的token索引、身体目标、14维手指命令、执行器命令、阶段和source row。**手指不由SONIC body token编码**，必须与token对齐单独保留。保留真实模型/provider、初始decoder history及物理状态，token不等于可脱离proprioception独立重放的整机动作。

完整性检查将command数量与实际执行步数核对，同时校验时间覆盖、有限值、mode1和decoder目标一致性。强制提前tick支持，未执行尾token单独标记；缺记录会使执行结果失败，不能标记完整。新增包不覆盖旧实验记录。

新默认版本`5e26fda`已验证Y+5cm单条件：首选站位1次检查，规划4.648s，操作参考逐帧复用、身体关节修正0；一次条件站位物理原生抬起成功，但碰桌质量未通过。1041个物理步对应261token，独立decoder逐位还原全部身体目标。连续导航仍未物理验证。[报告](/home/kinetra-4090-01/code/zijian/experiments/2026-10/27/20261027_1856_replay_first_sonic_tokens/REPORT.md)。
