# 单实验持久规划与事件等待

新增路径独立于原 cap-sim pipeline。每次实验只启动一个 fork_turns=none 规划器；它保留本次决策和执行结果，不接收此前实验结果。TRAIN 为成功示范的稀疏测量信息，不包含原动作序列或 token 回放。

| 文件 | 作用 |
|---|---|
| event_run.py | 一次尝试入口；首次发送 TRAIN，后续发送当前 LIVE、全部新增执行回执和未提交协议反馈；强制同一代理及响应哈希链 |
| event_prompt.txt | 定义 TRAIN/LIVE 可见范围、动作接口、稳定性门和显式等待语义 |
| event_dispatch_protocol.txt | 代理原始响应、哈希证明、ready 原子发布的文件传输协议 |
| repro/episode_transport.py | 验证同一实验身份与响应链；只允许示范观测元数据进入 TRAIN |
| repro/event_session.py | 把模型显式等待编译为固定绝对目标的原生 hold 指令；保留每条真实执行回执，失败时保留已执行前缀 |
| repro/event_wait.py | 有上限等待；稳定、接触变化、跟踪异常、终止或预算耗尽时返回；不规划新目标 |
| freeze_event.py | 校验历史冻结资源、实际 CPU 模型加载、预检及规划器出生证据，冻结实验契约 |
| tests/test_episode_transport.py | 身份与哈希链、历史结果元数据剔除 |
| tests/test_event_wait.py | 时间预算、固定目标、瞬时接触、接触对变化、反馈完整性、停止事件 |
| tests/test_event_session.py | 真实会话账本集成、无运动拒绝与已执行前缀保存 |
| tests/test_freeze_event.py | 单次预算、出生证据与冻结证据列表 |

继续复用已冻结的 controlled_bridge.py / feedback_session.py / native bridge 来保持物理、mode1、稳定性门和验收不变。等待的每个子指令是0.1秒；非终止事件最迟下一子指令边界返回，硬停止和稳定性采样仍为0.005秒。等待不是额外模型调用，也不是固定轨迹回放；绝对目标来自本次模型先前已执行指令。

运行示例与完整证据：experiments/20261015_1537_roboicl_event_grasp。仅一个原位置物理尝试；最终结果以该目录 REPORT.md 和独立验收为准。
