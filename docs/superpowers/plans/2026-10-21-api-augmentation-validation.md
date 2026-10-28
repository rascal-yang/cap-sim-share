# API 增广实施与验收记录

本地实现位于 feature/api-augmentation 分支，独立工作树 `/home/kinetra-4090-01/code/zijian/cap-sim-api-augmentation`。未合并原 main，未推送远程。原始代码快照标签保留。

已实现任务输入/确定性采样、初态构建、SQLite 幂等任务与工具调用、单设备执行锁、墙钟监督与中断恢复、Responses 工具循环、按需 skill、JSON CLI、状态/动作对齐的 NPZ/JSON 导出及可选视频证据索引。控制策略不变。

验收依据：`/home/kinetra-4090-01/code/zijian/experiments/20261021_1142_api_augmentation_final/REPORT.md`。61+53+135+71=320 项测试通过。最终物理集成两例：原起点失败、back20 成功；源实测状态与历史基线最大差均 0.0。全部导出，运行 72.107 秒；重复调用约 2.597 秒且无新物理/外部 API。真实物理执行提交 ee7e8ee，后续导出视频索引补充仅做单元和零物理步离线验证，完整代码提交见 Git 历史。

首次本地验收时尚未完成：真实 API 联调与实际 token/费用测量。没有 API 凭据；当前测试注入脚本化响应，不能宣称全目标已达成。视觉训练验收未执行，保持 false；不导出伪造 SONIC token。

## 评审与修复

独立评审指出七项重要问题，均先复现失败再修复并跑绿全套新增测试：监督进程中断时子进程存活/锁释放、输入符号链接换目标、忙设备导致任务滞留、恢复读取缓存成功结果、最终会话缓存绕过证据校验、API 超时重试少计请求/usage、前缀截断轨迹错误拼接初始状态。之后补充导出耗时统计和可选视频索引测试。没有未处理的评审 minor 项。

## 执行中的决定与限制

- 使用同级独立 worktree 完成串行实现，原目录不变；使用者必须进入该目录或后续合并分支。
- 原有测试模块重名且相互导入，四套独立执行并关闭外部 ROS pytest 插件自动加载；未改旧测试。不会验证外部插件本身。
- 初态碰撞筛查拒绝新增或恶化到超过基线容差的穿透；基线已有接触不会因此被误判为新碰撞，但并未声称修正历史重叠。
- Linux supervisor 在加载物理会话前 fork、共享文件描述符锁、进程组终止并等待；PR_SET_PDEATHSIG 约束监督进程死亡。未承诺非 Linux 支持。续跑保留已用预算，不重放中断物理案例。
- 本地 skill 是精简工具参考；脚本化响应证明工具链，真实模型是否按指导正确使用工具仍待真实 API 验证。
- 历史源代码变化只能由实验内哈希完全一致的 source_snapshot 替代；模型/数据证据损坏仍拒绝。无法恢复的旧证据会阻断导出。
- 在物理验收前集中评审，以减少修改源码导致重复冻结。物理成功、批次处理完成、数据结构可导出、视觉训练准备是四个不同字段。

## 接入

参见 `humanoid_mimicgen/augmentation/README.md` 和 examples 下两个 JSON。API 配置需要明确 protocol=responses、base_url、model、key_env；密钥通过该环境变量提供，不写入仓库和日志。其他协议需明确添加适配器，不能静默替换。


## 真实 API 验收（20261021_1607_api_augmentation_live_final）

用户指定 api/openai_config.toml、gpt-6-astra low。执行版本 b79cf08：补充可选 reasoning_effort、流式 Responses 解析、明确 User-Agent 与环境变量请求头；其他路线和控制逻辑不变。最终 5 轮响应均报告 gpt-6-astra/low，工具调用 read_skill→validate_job→submit_job→get_case，完成两例物理执行及导出。输入 6766、输出 347、合计 7113 tokens；API 等待 31.696 秒，端到端 106.841 秒，本地 worker 72.106 秒。物理 1/2 成功，两例状态与历史逐项相同。重复请求 2.563 秒，新增 API/物理均 0。322 项测试通过。

完整报告：/home/kinetra-4090-01/code/zijian/experiments/20261021_1607_api_augmentation_live_final/REPORT.md。真实 API 联调已完成；费用尚无网关价格/账单依据，保持未知。此前排障请求单独记账，总调用 10，已知 usage 11242 tokens，另 4 次 usage 未知；不把 7113 当整个排障过程总用量。视觉训练验收仍未执行。
