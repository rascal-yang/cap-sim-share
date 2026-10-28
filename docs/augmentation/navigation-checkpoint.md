# 精停导航的完整状态回溯

`replay_navigation`现在支持可选阶段检查点：`navigation.checkpoint={"enabled":true}`。保存导航开始和精停完成、第一次replay衔接指令之前的完整状态，gzip JSON中保存MuJoCo积分状态、SONIC历史/规划参考/随机序列计数/交接状态、任务计数、当前控制目标和停车/执行进度。只支持这两个阶段边界，当前操作/恢复程序在这些边界均未开始。

新case配置增加`navigation.checkpoint.resume={"case":"父case绝对路径","name":"parked"}`，通过既有`direct_grasp.prepare`预检冻结，再执行`replay_navigation.run`。阶段名`navigation_start`可用于导航参数变化；`parked`仅允许在停车后才生效的反馈、姿态反馈、衔接时长和腕部积分变化。场景、物理、源参考、初态、历史初值和模型必须匹配。检查点初始化只发生一次，运行内仍只执行SONIC动作，不重置物理状态。

父轨迹到检查点之前的状态、动作、token、手指、事件继承；后缀重新执行。结果审计区分`inherited_physics_steps`、`executed_physics_steps`、`executed_sim_seconds`、`checkpoint_restorations`和`lineage`。完整轨迹长度不冒充本次新执行步数。原有物理恢复retry仍是独立机制；回溯分支由调用方提交，碰桌不会自动触发回溯。

验证实验：`experiments/2026-10/28/20261028_1137_apple_navigation_checkpoint`。Y±10cm，13次主实验：2个精停种子、3次相同参数续跑、4组不同反馈参数的restart/rollback配对。7组整条state/next_state/commands/token/hands数组完全一致。独立第14次从原始远位执行整条拼接token与手指命令，未调用encoder或planner，2916个物理步和729个token与源分支逐位一致，原生成功。最终版本569项相关测试通过；审查修复后新增种子和续跑两次物理运行，均成功且逐位一致，总计16次物理执行。

测试说明：Y−10cm种子及候选成功；Y+10cm种子及两候选均失败。回溯没有改善这组候选的成功率，证明的是等价性和重试效率。精停仍有手指碰桌，不能称全身无碰撞或训练就绪。全部新生成动作来源是原replay重定向，非独立策略迁移。

实际计时、检查点体积、碰撞与初次种子成本见实验REPORT.md。规划沿用已有完整规划；额外重渲染未执行。旧冻结证据不修改，新代码变更仅通过后续新case生效。

当前绑定采用保守范围：offline_plan在navigation_start也必须相同，更换路线或执行源码需要新建种子；仅导航控制参数变化可从navigation_start分支。enabled:false同时禁用捕获和resume。
