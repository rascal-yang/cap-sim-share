# G1 / Dex3 抓苹果：skill 驱动生成草稿

状态：未发布、未通过新策略物理验证。来源与证据范围见 skill_provenance.json。
本文只传递接触几何、阶段条件和失败分析，不包含专家关节/腕部时间序列或已录制 token。

## 已观察经验

- 使用右拇指和食指在苹果相对两侧建立接触，再保持接触抬升。
- 原成功示范的持续接触窗口中，苹果局部食指接触位置约 [0.037127,-0.005921,0.011521] m；拇指约 [-0.035341,-0.014011,0.011089] m。
- 这些点属于原 Dex3/苹果几何下的观察，只是新接触区域的候选种子。新腕姿必须根据当前手、苹果几何重新生成；不得复用旧腕轨迹。
- 接触映射：p_world = p_object + R_object @ (scale_ratio * p_local)。物体移动后重新观察、重新映射。
- 原示范的接触力仅是观察值，当前控制器没有显式夹持力控制能力。
- 位置到位不等于抓稳。已有失败包括：基座退移、髋部碰桌、短暂接触后滑脱、抬升丢失接触。

## 当前执行能力与限制

- 首轮使用原生 HUMANO G1/Dex3、SONIC1.0 CPU、Cartesian teleop mode1；不得隐式切换到 mode0 或调用旧轨迹。
- 可调用 velocity_xy、facing_yaw、pelvis_height_m、左右腕位置/朝向、躯干朝向、手指关节目标和有限阶段持续时间。具体范围以 backend.describe 的实际接口为准。
- 无抓取完成记录的策略仍是 candidate；不能因示范或 waypoint 测试成功就标记新抓法已验证。
- 原生验收保持约9cm抬升、hold_s=0.005s，不得描述成稳定保持1秒。接触和碰撞门限由执行器落实，不由skill降低。
- 初批先在当前可达范围内改变平面位置与桌面高度。需大幅移动/转身的场景先记录需求，不自动加入复杂走路动作。

## CaP 生成配方

```python
policy pick_apple(skill, scene, observation, interface, budget):
    analyze_scene()
    # 明确苹果几何、桌沿/障碍、机器人朝向、当前站位和手部视野。
    # 判断当前腕目标是否可达，给出站位余量、接近方向和抓法的理由。
    # 缺失信息记为unknown；不能照搬示范路线。

    select_contact_regions_from_current_geometry()
    generate_new_wrist_orientation_and_short_waypoints()
    # 所有waypoint由当前几何和skill生成；统一记录坐标系、时间和动作来源。

    execute_precontact_stage_then_observe()
    if wrist_tracking_bad or base_drift or obstacle_contact:
        diagnose_requested_vs_measured_motion()
        generate_one_testable_correction_within_interface()
        observe_again()

    execute_contact_stage_then_observe()
    if required_contact_not_confirmed:
        diagnose_contact_links_object_motion_and_finger_targets()
        adjust_approach_orientation_or_fingers_within_budget()
        # 不进入抬升，不重复无证据的等待。

    lift_in_short_segments_with_feedback()
    if contact_lost or unstable or blocked:
        stop_progress_and_follow_measured_recovery_branch()

    independent_native_acceptance()
    if accepted:
        preserve_complete_success_and_training_records()
        # 当前 optimization_attempts=0，不启动优化。
    else:
        save_error_analysis_before_reset_and_next_attempt()
```

## 版本与数据

一次尝试内冻结skill；失败后先保存事实、假设、单一拟议修改和拒绝条件，再发布下一版本。
重试可复用本条skill生成路线中已验证的阶段；不得引入外部专家动作或恢复中途成功状态。
记录真实执行指令、状态、接触、原始相机帧及时间对应关系；展示视频与训练RGB分开。
