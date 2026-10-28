# 复现设计与验收边界

用户要求：独立目录、逐文件理解并实现 RoboICL，不改原 pipeline；HUMANO G1 + Dex3，禁止原轨迹重放。此目录分两层：

1. **原版可执行复现**：固定 c0e58e456ce4b2ea1c401d1b261d6e0327c6e04c 的 owned 源码原样保存，直接运行其 Dialogue Model、TRAIN serializer、LIVE memory、数值动作校验及回执。离线 transport 注入只替换模型网络请求，用明确标记的合成观测和预定回答检验机制，不代表模型任务能力。
2. **HUMANO 明示适配**：使用相同 observation → act → receipt → observation 语义和原版 LIVE memory；动作契约单独定义，Dex3 不压成两个标量夹爪。现有 backend 的最短动作是 0.1 秒，不能伪称支持原版 25 Hz 控制；SONIC 腿部 planner 更新还受 0.5 秒节拍限制。原版 joint safety hold 不能偷偷替换为 humanoid mode0。

TRAIN 必须是已记录的**命令**与实际前后观测；测得端点差、参考 token、原始骨架不能作为同接口动作真值。执行程序不自动消费 TRAIN 动作，只允许模型显式提出新动作。仅“数值恰巧相同”不能证明重放或生成，因此必须保存请求及执行来源。

LIVE 保留固定锚点+最新 chunk，完整本地档案不裁剪。零步非法动作留在当前 tail，允许同观测纠错；部分执行保留实际前缀并丢弃后缀；硬碰撞终止且记账，不能自动越过物理安全门。reset 开新 episode；跨 reset 学习是扩展，需要明确设计，不能冒充上游。

本次仅代码/离线复现，不启动物理，不安装环境。真实实验需独立冻结 experiment.json、同动作接口真实 TRAIN、相机校准、运行时 provider/model hash 等证据。上游四个子模块未初始化，因此不得把 owned 层测试当作完整 RoboDojo IK/仿真复现。
