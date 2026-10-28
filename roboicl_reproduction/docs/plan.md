# RoboICL Implementation Plan

> 后续接入说明：本文保留初次静态审计/离线阶段的状态；现已新增 native G1 + Dex3 桥接与跨接口 TRAIN 观测记忆路径，见 [integration.md](integration.md) 和实验 `20261015_1047_roboicl_origin_live`。原版完整环境与同接口 one-shot 仍未验证。

Goal: 在独立目录建立可追溯的上游复现与明确的 HUMANO 适配。
Architecture: immutable vendor + transport-injected original policy + protocol tests + separate humanoid action contract.
Tech Stack: existing Python, numpy/Pillow/pytest；不安装新环境。
Spec: design.md

Global Constraints: 不修改原 pipeline/历史文件；不运行未冻结的物理实验；无 trajectory replay、mode0 fallback。
Review Focus: 零步纠错是否进入下一请求；部分执行是否误报全执行；TRAIN 是否伪造动作；terminal 是否被当可恢复；是否泄露 privileged observation；vendor integrity。

- [x] 固定上游完整 owned 文件，保存 gitlink 和 SHA256；逐文件审计。
- [x] 测试先行：实际上游 DialogueModel 的非法动作、IK拒绝、部分执行、terminal、reset、TRAIN serializer、LIVE anchor/gap。
- [x] 实现离线 transport 适配及可重复演示命令，所有 fixture 明示 synthetic。
- [x] HUMANO 独立数值动作契约及测试，明确物理接入限制。
- [x] 独立闭环接口测试，非法动作与执行回执分离；不可恢复错误停机。
- [x] 新旧测试、vendor hash 和原文件不变校验，最终独立审阅。

执行记录保存在 verification.json 和文档；无未经请求的提交。

完成证据：../verification.json。独立审查发现的未知执行前缀与TRAIN时间自洽问题均已先红后绿修复。原版端点依赖测试明确skip，完整物理复现未完成且未启动。
