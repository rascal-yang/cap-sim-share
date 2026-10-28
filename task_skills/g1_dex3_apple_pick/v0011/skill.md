# G1 / Dex3 单手抓苹果：可重新选择策略的任务 skill

版本用途：为 GPT 提供任务级经验，不替代控制器，也不是必须逐帧复现的轨迹。
更新单位：一次尝试内固定；失败或成功后基于证据生成新版本，下一次尝试可立即采用。

## 任务不变量

- 任务：从桌面抓起苹果。当前验收使用任务自带原生程序，约抬升 9 cm；没有额外验收 1 秒保持。
- 不修改物理参数、验收标准、控制权限或试验总预算。次数与时限从当前任务配置读取。
- 只有实际接触满足现有控制器门限才能抬升；目标点到位或程序完成不等于抓取成功。
- 当前后端持续手部接触门限 0.12 s；机器人碰桌力 >5 N 持续 0.05 s 会停止。
- 当前停止规则不等于对每个手指、每类障碍物的完整安全保证，不能推断“零碰撞”。
- 当前关闭减速优化；获得成功必须先保存，不因候选优化或新策略失败删除已有成功。

## 示范中关键的、可迁移的信息

1. 抓握使用右手拇指与食指在苹果相对两侧建立接触，再抬升；接触部件名和局部坐标是证据。
2. 参考中选取 8.905–9.100 s 的持续接触窗口（40 个 5 ms 物理步）：
   - 食指 `right_hand_index_1_link`，物体局部位置约 `[0.037127, -0.005921, 0.011521]` m。
   - 拇指 `right_hand_thumb_2_link`，物体局部位置约 `[-0.035341, -0.014011, 0.011089]` m。
   - 窗口内最大局部偏差分别约 1.02 mm、0.76 mm。
3. 运行时优先使用 `contact_privileged.reference_priors` 精确数据，不以本文四舍五入值替代。
   目标映射为 `p_world = object_position + R_object @ (scale_ratio * p_object)`，四元数 wxyz。
4. 接触点相对手腕的位置与物体局部位置一起给出刚性手腕初值；尺寸变化仍可能要求调整手指。
5. 参考力是实测观测，不是允许任意尺寸使用的夹持力目标。当前接口没有显式夹持力控制契约。
6. 接触之前应确认接近方向、手腕朝向和手指开合适配几何；换站位后必须重新获取物体与手的相对关系。

## 可变策略与目前证据范围

- `reference_local`（observed）：保留原前 5 秒 native token，后续参考关键帧按接触几何调整右臂。
  新接触点版本只验证了 X +10 cm 与尺寸 +10%，各一次。旧手腕偏移版本还验证了 Y −10 cm、桌高 ±5 cm，各一次。
  不把这些点解释为整个范围都稳定成功。native_prefix 只在实验 HybridBackend 明确提供时可调用。
- `reposition_then_grasp`（candidate）：物体超出当前合适操作范围时，先改变基座位置/朝向，停止后重新观察，再选择抓点与腕姿。
  当前接口提供 SONIC 速度、朝向、骨盆高度目标；大位移后抓苹果尚未验证。
- `adjust_stance_then_grasp`（candidate）：桌高或接近角度变化时，改变骨盆高度/躯干姿态，稳定后重新定位右手。
  不把接口允许的骨盆高度 0.4–0.85 m 当作该任务已验证的安全范围。
- `alternate_contact_region`（candidate）：原接触区域被遮挡、几何对应失效或抓取方向改变时，重新选可达的对向接触区、腕姿和手指动作。
  新接触点必须标为假设。苹果以外的形状、换手、换机器人不能直接沿用本示范坐标。
- 允许 GPT 提出 `new_hypothesis`，但必须说明当前证据为何支持尝试、需要哪些控制通道、如何检查失败；不得自称已验证。


## 当前默认控制模式（用户于2026-10-13指定）

下一轮苹果生成默认使用 mode1：SONIC planner 规划下肢，Cartesian/teleop 目标提供上肢动作。所有阶段默认不提供 `body_keyframes`，不插入 `native_prefix`，避免这些通道转入其他模式。参考仍可用于接触位置、接近方法、腕姿和动作阶段的经验提取。

这是任务级规划偏好，不是运行时新增模式锁；后端仍保留已有动作通道。若明确选择不同模式，必须记录偏离默认的理由与实际模式，并单独验证交接；不能沿用旧前缀而不标明切换。旧模式的成功产物和证据保留，不因默认变化否定。

整体恢复门、同接口验证门、接触门、失败分析/重试、成功保存和优化开关均不变。已有尝试快照继续使用原版本。

## 当前接口与执行限制

真正可执行的动作以本次 `interface` 为准。以下是 native HUMANO 后端支持的候选通道：
`velocity_xy`, `facing_yaw`, `pelvis_height_m`, `torso_quaternion_wxyz`,
`right_position`, `right_quaternion_wxyz`, `right_fingers`, `body_keyframes`。

- 无 `body_keyframes` 的动作使用 SONIC planner 下肢参考与 Cartesian 上身跟踪。
- 提供 `body_keyframes` 时进入全身参考模式。不得同时假定 `velocity_xy` 会重新规划下肢。
- 控制接口支持某个通道，不等于该任务上该模式已经成功。前期全轨迹重编码曾失败；保留该限制。
- 单段时长、关节限位、速度限制和动作格式由实际接口限定。先输出短阶段，不让模型逐帧控制。
- 不存在通用的 `go_to_stance()`、`grasp()` 或 `set_grip_force()` 执行 API。下文函数是推理步骤，必须编译成接口允许的 JSON 动作。
- 当前苹果局部参考适配代码不自动实现本节所有候选。选择 GPT/command 规划器才能生成不同的动作程序；固定局部适配器仍只执行原策略。

## GPT 选择策略前必须知道的内容

读取当前物体位姿/尺寸、桌面支撑边界和障碍物、机器人根位姿/朝向、腕姿、关节状态、控制通道与已有失败。
参考里的桌面尺寸、物体初始位置或可见视角不是新场景事实。缺少当前碰撞几何时指出缺项，不凭旧场景假设存在自由路径。
优先使用仿真特权几何；有实际第一/第三视角图像时辅助检查遮挡，不把未提供的视频当作已经看过。
检查苹果是否仍在桌面支撑内：旧 X −10 cm 变体因移出桌边而无效，不能通过抓地上苹果算作原桌面任务成功。

## CaP 形式的决策配方（推理伪代码）

```python
def apple_pick_skill(scene, observation, contact_prior, failure_notes, interface):
    check_current_geometry_and_table_support()
    candidates = propose_approach_stance_grasp_options(
        reference_contact_anchors=contact_prior,
        current_geometry=scene,
        current_robot=observation,
        prior_failures=failure_notes,
    )
    # 可换路线、站位、腕姿与抓点；参考动作不是必须复制的步骤。
    strategy = choose_candidate_with_reach_clearance_and_control_support(candidates)
    record_strategy_id_scope_evidence_and_expected_failure_modes(strategy)

    if strategy.requires_new_stance:
        emit_short_walk_or_stance_action_with_replan_after(interface)
        # 此次只输出下一段；实际返回后再进行以下判断。
        reobserve_root_feet_wrist_object_and_stability()
        if stance_not_achieved_or_unstable:
            stop_or_recover_with_available_controls()
            return

    choose_reachable_contact_regions_and_wrist_orientation()
    emit_short_precontact_action_with_feedback()
    compare_actual_tracking_and_clearance_before_closing_fingers()
    if object_moved_or_tracking_failed:
        reobserve_and_replan_contact_targets()
        return

    emit_bounded_finger_and_wrist_adjustment()
    compare_measured_per_link_contacts_with_target_regions()
    if actual_contact_gate_not_met:
        diagnose_approach_vs_orientation_vs_finger_failure()
        emit_one_supported_correction_or_end_attempt()
        return

    emit_lift_segment_with_feedback()
    if contact_lost_or_unstable_or_blocked:
        stop_and_enter_supported_recovery_branch()
        return
    if independent_native_validator_passes:
        preserve_success()
        # 只有任务明确启用优化，才另建候选；当前关闭。
```

阶段内反馈不算新尝试。失败后从初态重置才算下一次。所有分支仍受原始尝试数、阶段反馈次数和执行时限限制。
不能通过重新命名阶段绕过抬升接触门限，也不能假称已执行未出现的动作。

## 失败诊断与 skill 迭代

每次尝试结束后先记录：使用的 skill 版本、策略、预测、实测轨迹/接触、失败阶段、实际执行用时。
区分物体几何变化、跟踪误差、接触选择、手指闭合、步行站位能力和仿真准备错误。
下一版可以修改选择规则、适用范围、候选策略和参数建议；不得修改硬验收、权限、物理参数或预算。
下一次尝试立即可选最新版本；当前尝试内所有反馈继续使用同一版本。旧版本永久保留以便追溯。
新的“observed”条目必须引用实际试验记录和适用条件；未经执行的建议保持 candidate。
成功一次只能增加一条观测，不能写成“能处理任意桌高/任意物体位置”。

## 对模型输出的要求

`analysis.strategy` 写明：`strategy_id`, `applicability`, `approach`, `stance`, `grasp`,
`next_observation_gate`, `abort_or_recovery`。说明选择理由与未验证部分。
`actions` 仅使用运行时接口支持的动作，遇到需要反馈的阶段设 `replan_after=true`。
不能只输出伪代码或在后台加载整条专家动作来冒充重新规划。

## 已知缺口

大幅站位变化后的抓取、新抓法、复杂避障、自动手指重优化尚无通过的物理证据。
目前无外部 GPT API 规划实测；过往成功均为当前会话编写的 Python 局部策略。
skill 为新范式提供经验与决策规则，不会凭文档增加控制器没有的能力。

## 新证据：大位移尝试 1

苹果(+25cm,+30cm)仍在桌内；斜向接近确实移动了机器人，但请求零速度后仍前移，在5.04s因髋部碰桌停止。尚未抓取，不能归因为抓点错误。下一候选先沿桌外侧横移，再前进、转向和重新观察。以当前桌边和机器人外形确定通道；预留制动距离，不能把零速度命令当成已经停稳。此侧向方案尚未验证。

## 大位移尝试 2 的证据

绕到桌侧与转向已执行，25.1s 内未碰桌，但没有接触苹果。Cartesian 伸手和下蹲时基座后退，腕目标误差约16.6cm。不能据此否定新抓点，因为未到接触阶段。下一候选复用无碰桌接近前缀，把反馈预算留给靠近后的站位与全身参考交接；按实际根位置重新生成腕/臂参考，不恢复中间状态。该交接尚未验证。

## 大位移尝试 3–4

全身参考交接保持稳定，但仍有跟踪偏差。第4轮更近站位后，闭指时拇指/食指短暂接触，随后滑脱并推移苹果约1.8cm，阶段结束接触为零。必须结合完整物理子步诊断，不能只看阶段末帧。下一候选分段闭指，在中间开度观察；参考接触窗口实际关节大约[0,-.3,-.31,.5,.76,.5,.76]，这是受载姿态观测，不是通用力目标或等效驱动命令。物体移位后废弃旧接触目标。

## 2026-10-13 大位移固定场景：10 次结果

苹果平移(+0.25,+0.30,0)m，合计39.05cm；大小、桌高、质量、摩擦、控制器和原生验收固定。10次完整物理尝试，0次通过，未进行优化。

- 动作前缀：实测转身、绕到桌侧、伸手及下蹲；相同初态下多次复现接近前缀，不等于跨场景鲁棒性验证。
- 第1次髋部碰桌；第2–3次未到有效接触；第4次短暂碰到并推移苹果。
- 第5–9次分别尝试分段闭指、浅/深蹲、参考连续衔接、食指开度/拇指姿态、靠近站位与有界闭指。均在抬升时丢失接触，不能作为成功抓取经验。
- 第9次两指同时接触最长0.86s，仍未完成负载从桌面转移到手。两个链接接触不保证形成稳定抓握。
- 第6/7次为同前缀的抬升参考起点对照；改成上一段指令连续起点仍失败，不能宣称该改动已解决问题。
- 第10次从全身参考切换到Cartesian上身模式时，右手中指碰桌，停止。该交接未验证安全。
- 后端contact_s是任意手部接触时长；本批CaP另查当前拇指/食指接触，但没有实现通用力闭合控制或闭环夹持力契约。
- 规划几何、关节参考与实际腕位姿存在差别。下一轮应在固定可达站位验证腕位姿跟踪、两侧持续接触与小段负载转移，再扩大场景。不要通过修改物理参数或降低验收掩盖这一缺口。

本批由当前GPT会话读取JSON反馈并决策，底层SONIC逐帧执行；无外部GPT API调用。所有失败、动作、当前/参考接触点和用时见 experiments/20261013_apple_large_shift/REPORT.md。
结论仅为当前组合未在该场景的10次预算内跑通，不是skill/CaP无效的因果证明。

## 新诊断：为什么没有继续下蹲

只读取既有记录，未增加物理尝试。第2次mode1目标骨盆0.68m，实际达到0.682865m，但根部沿远离苹果的Y方向退了约19.67cm；腕目标误差约16.59cm，其中偏高5.25cm。第9次mode0参考骨盆约0.6864m，实际约0.7157m，跟踪差约2.93cm。尚未验证更低目标，也未隔离该跟踪偏差的具体原因，不能宣称达到物理极限。

待验证候选：在接触前固定mode1，先停止行走、观察实际根部运动与腕姿；必要时骨盆目标每段下调约2cm，在当前接口范围内执行短段，再测实际高度、根部漂移、腕位置/朝向误差及桌面接触。根部退步或腕误差变大时停止继续降低，重新诊断站位与跟踪；碰桌或失稳继续使用原停止条件。只有实测收敛才推进闭指。已建立接触时不要未经分析改变支撑姿态；失去接触按现有恢复分支处理。该候选不是已验证修复，不能保证抓取成功，也不新增反馈次数或仿真预算。

证据：[下蹲诊断](../../../../experiments/20261013_apple_control_diagnosis/REPORT.md)，原始请求/响应SHA保存在同目录diagnosis.json。

## 严格诊断、反思与重试契约（运行时已实现）

本次生成任务设置policy_checks.require_reflection=true、sonic_mode=1。不使用body_keyframes或native_prefix；复用前缀也必须mode1。当前段结束时读取diagnostics.last_stage，核对requested_mode和controller.actual_modes。缺失观测标为未知，不能写成零误差。

先检查跟踪与站位，再检查接触，最后检查负载转移：
```python
if actual_root_or_wrist_not_converged:
    compare_target_reference_command_and_measured_state()
    inspect_pelvis_height_error_base_drift_and_wrist_orientation()
    choose_one_short_stance_or_wrist_test()
elif contact_not_established:
    inspect_object_frame_contact_regions_wrist_pose_and_measured_finger_joints()
    choose_one_bounded_alignment_or_closure_test()
elif contact_lost_during_lift:
    inspect_per_link_contact_history_wrist_motion_and_table_support()
    choose_one_load_transfer_or_grasp_retention_test()
```
这些是推理分支，必须编译为当前接口动作。硬停止触发后，本次结束；不通过改阶段名绕过接触门。

模式1可尝试先稳定站位，再将骨盆目标小步降低，保持物体相对腕目标并读取根部漂移。约2cm仅为初始候选步长，不能宣称已验证；更多下蹲不能自动解决水平退步。实际腕位置/朝向和站位未收敛时，不盲目闭指和抬升。收敛阈值作为本次计划参数明确记录，不能放宽成功验收。观测fingers是指令，手指实测值读取joint_interface.right_hand[].qpos。

反思请求request_kind=reflection不执行动作。必须输出source_attempt、result_sha256、failed_stage、observed_facts（exact path/value）、hypothesis、alternative_causes、next_change（description/controls）、expected_effect、reject_if、hypothesis_outcome。先读取上一假设和执行后的实测结果，把结论标成supported/not_supported/inconclusive；一次失败不自动否定所有接触策略。数值事实引用result/diagnostics的真实路径，不能由视频猜测数值。

下一次规划读取previous_reflection，并将程序提供的previous_reflection_sha256原样复制到analysis.reflection_sha256，写明analysis.change。不要自行计算哈希。主要改变一个待验证因素；需要联动修改时说明理由和混杂限制。每段反馈同样写预测指标和否定条件。若相同调整持续不改善相应指标，则更换诊断假设或结束尝试，不无条件重复。

参考恢复阶段可按原成功接口恢复旧动作；新场景生成严格mode1，不能将旧模式成功当成mode1成功。成功先保存，当前优化预算仍为0。版本每次内冻结，反思继续读取本次实际使用的完整skill集合；下次才选新版本。

## 严格mode1批次：前3次证据

原参考重新实际恢复成功13.4s，但新场景尚未成功。第1次最近站位刹停时右髋碰桌。第2次提前结束侧向接近并刹停避免该碰撞，随后分段下蹲/伸手仍出现6–7cm腕误差；尝试下蹲时移动基座反而使骨盆上升约5cm。第3次将姿态降低与伸手分开：实际骨盆先降约3.4cm，第二个2cm指令没有继续下降；后续伸手仍产生约5.44cm后退与7.33cm腕误差。以上仅是三个固定初态单次结果。

不得持续把更低pelvis_height_m当作必然更深下蹲；同时检查planner参考高度和实际高度。位置/姿态误差增大后，禁止无条件闭指或抬升。换站位、躯干姿态或腕朝向应作为明确单项候选，记录对漂移和腕误差的预测。可以测试小幅躯干前倾改善伸手，仍未验证，不能偷偷切换全身参考模式。


## 转身站位候选（未验证）
前6次严格mode1未成功。躯干前倾可改善预接触位置，但闭指时整体后退；骨盆0.62m目标不保证实际下降；接触方向-120度曾获得1.02cm预接触位置误差，仍有0.389rad朝向误差，接触目标无可靠接触。
允许协调当前接口的左右腕位姿及躯干朝向，使两臂随实际转身调整；控制模式与物理参数固定。候选站位在桌边面向+X、右肩朝向苹果，假想根部(-.20,.60,.78)m只经过独立FK/IK检查，无桌面穿透，足底平衡与SONIC跟踪未验证。不得直接改物理根部状态；用velocity_xy/facing_yaw实际走到附近后重观测。原示范成功接触骨盆约0.785m，闭指命令[0,-.7,-.7,1,1.5,1,1.5]与受载实测手指不同。实际腕收敛且几何可行才闭指，不满足接触门禁止抬升。


## 严格mode1完整10次结果（2026-10-13）

固定苹果(+0.25,+0.30,0)m，大小/桌高/物理/验收不变。10次0成功，仿真263.525s，整轮实际5009.662s（包含本会话分析与反思，不是外部API延迟；渲染另计）。每次实测模式均1，反馈不超过12、每次仿真不超过30s。原成功保留，无优化。

关键前提缺口：原参考13.4s独立恢复成功，但使用原token前缀及mode0全身参考；新生成是mode1的下半身planner+Cartesian上身，尚未证明在原条件下能用mode1完成这条抓取。因此不能将0/10解释为skill无效。后续应先取得同一接口下的成功基线，或明确仅做当前接口能力诊断。

- 1–3：提前刹停避免持续髋碰桌；更低目标/姿态先行仍不能保证腕部收敛。
- 4–6：小幅躯干前倾、位置补偿、接触方位变化能改善部分预接触指标，但静态接触预测未转化为可靠真实抓握。0.62m骨盆目标的当前规划参考和实际高度仍约0.75m，不能认定物理极限。
- 7：实际转身到+X并移动到桌侧，无碰桌；伸手时根部X漂移约20cm、腕朝向误差约0.8rad。静态IK可解不表示SONIC可跟踪。
- 8：局部躯干pitch0.24rad配合位置补偿，名义腕误差曾降到1.58cm；朝向约0.304rad，静态闭指只发现中指接触。改善包含多个变量，不能归因于单一改动。
- 9：小幅朝向补偿略改善误差；半闭指0.35s+等待0.45s没有任何真实苹果接触。实测手指滞后于指令，不能将指令角当作到达。
- 10：采用更接近实测腕朝向的候选，朝向误差曾降到0.152rad；位置补偿后名义腕误差约1.34cm。仍无真实抓握。局部几何距离查询出现表面点越界，拒绝这些输出；有效点筛查没有可用食指候选。

任何几何优化只操作独立MjData，不改真实仿真状态。几何查询须限定碰撞体并验证最近点属于物体表面；不使用可视网格或零距离异常值宣称接触。线性IK必须报告残差、用非线性/实际跟踪复核，并检查全部手指（包含非目标手指）与桌面重叠。深穿透及静态双接触不能当力闭合。

本批任务内跟踪、接触、抬升门保持不变。失败前缀可复用完成的安全运动，但标明不是成功抓取，不恢复中间检查点。先保存所有失败、逐段诊断和反思，预算耗尽后停止。未来仅在用户授权新预算时验证mode1原场景/接触能力，不自动启动第11次。

完整证据：experiments/20261013_apple_mode1_reflection/REPORT.md、audit.json、run/attempts/NNN/diagnostics.json及reflection.json。


## 当前授权实验：直线接近的位置/桌高扫描

用户暂不要求拐弯行走，明确要求在此前小变体成功基础上测试更大的左右/上下位置。竖直上下指桌面和苹果一起升降。当前实验选择reference_local分支，使用已成功的原250-token前缀+mode0全身参考，只重定向右臂；它不同于上一批严格mode1。不得将过去mode1约束推广为本批只能mode1，也不得将本批参考模式成功宣称为mode1成功。runtime task若设置sonic_mode=1则仍必须遵守。

先执行原场景参考恢复；再做Y-10cm复验、Y+10cm、Y±15/20cm、桌高±7.5/10cm，共10个单因素条件各1次，原生验收、30s/12反馈、大小/质量/摩擦/惯量/控制器/时长/手指目标保持原值，不做优化，不新增行走转身模板。当前范围尚未验证。

```python
assert recover_original_reference_same_interface().success
for point in frozen_single_factor_points:
    validate_table_support(point)
    analyze_last_failed_point_if_any()
    reset_initial_state()
    execute_original_straight_token_prefix()
    for source_stage in successful_reference:
        map_object_and_wrist_local_contact_priors(point)
        retarget_only_right_arm_preserve_other_reference_channels()
        execute_short_stage_and_read_feedback()
        if next_stage_is_lift and not measured_contact_gate_and_current_thumb_index:
            remain_in_reference_contact_stage_within_budget()
        stop_on_existing_safety_gates()
    if native_success:
        preserve_success_before_optional_optimization()
    else:
        write_evidence_bound_reflection_before_next_point()
```

这是当前GPT生成的确定性CaP参数扫描，阶段内主要执行映射/IK及反馈门，不声称每段调用GPT或展示新模型推理收益。每点一次的完成数量不是随机重复成功率或连续可行范围。失败点不自动换mode1、不另加转身/步行、不追加第11个生成尝试；记录边界与跟踪、接触、负载转移问题即可。
