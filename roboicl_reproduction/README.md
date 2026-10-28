# RoboICL 独立复现

这里与 `roboicl_adapter/`、原 `src/cap_sim/` pipeline 分开。不是把旧 pipeline 重命名：原版核心实际运行，HUMANO 的差异有单独实现与测试。

## 从这里读

1. [设计与边界](docs/design.md)：什么是原版，什么是人形适配，什么还未验证。
2. [TRAIN、运行入口、逐任务配置审计](docs/train_and_harness.md)。
3. [逐策略文件及执行闭环审计](docs/policy.md)。
4. [HUMANO 动作接口](docs/humano.md)：G1 + Dex3 双7关节，世界增量，固定0.1秒，实际 SONIC mode1。
5. [本地调用关系与接入边界](docs/integration.md)。
6. [逐文件对应索引](docs/file_map.md)。
7. [完整文件清单](upstream_manifest.json)：固定上游 commit、逐文件 SHA256、缺失子模块 gitlink。

## 已实现

- `vendor/`：上游 owned 文件原样快照，MIT LICENSE 保留。原版 `DialogueModel`、`LiveTrajectoryMemory`、动作转换及 TRAIN serializer 直接复用，不用简化近似替代。
- `repro/offline.py`：注入预定响应到原版 Model，隔离 ASTRA 环境设置，绝不调用网络。合成图像/状态仅用于协议测试。
- `repro/session.py`：独立人形闭环，实际前缀记账、未执行后缀丢弃、同观测纠错、原版固定锚点+latest记忆、terminal禁止继续。执行异常即停，不猜测执行量。`StepResult.elapsed_s`须由实际执行器测量；这里的step是命令观测转换次数，不是物理tick。
- `repro/train.py`：读取哈希锁定的真实命令记录，验证TRAIN/LIVE同profile、0.1秒实际mode1、命令转换一致、动作前后观测、遗漏区间和真实末帧。源文件hash只证明完整性，不证明任务成功或传感器真实性。不得用上一轮端点差分示范冒充输入。
- `repro/humano.py`：纯动作转换，全chunk验证后返回现有backend可接受的动作；不改backend、不重放原轨迹、不把Dex3映射成两个夹爪标量。

`Session` 通过显式 `convert` 和 `execute` 注入接口；TRAIN仅进入上下文，执行端从不遍历TRAIN发动作。调用方将 `Session.context()` 和动作schema交给隔离的API，再把唯一 `act` 交给 `submit`。硬碰撞可写terminal回执，但不会因此继续撞桌。跨reset失败学习不是上游默认机制，本实现不隐式携带。

## 无物理离线运行

从 `cap-sim` 目录，使用现有 HUMANO 环境（没有安装新依赖）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kinetra-4090-01/HUMANO/.venv/bin/python -m pytest -q roboicl_reproduction/tests
/home/kinetra-4090-01/HUMANO/.venv/bin/python roboicl_reproduction/run_offline.py --output /tmp/roboicl-protocol-check
```

输出目录必须不存在。第二条实际走“非法动作纠错 → 零步控制器拒绝 → 重新决策 → 部分执行中断 → terminal”链，保存所有请求和原版日志。固定回答是测试夹具，不能作为模型能够纠错或抓取成功的证据。

## 已接入的 G1 + Dex3 物理闭环

`prepare_native.py` 从原成功轨迹的保存状态中提取稀疏观测、实测手指关节、接触与图像，生成固定 TRAIN。它不把观测差分冒充动作标签；本轮是跨控制接口的观测记忆适配，不是原版同接口 one-shot。原轨迹的命令、token、时间序列不能交给执行器重放。`repro/train.py` 的同接口命令示范读取器仍是另一条独立路径。

`repro/bridge.py` 在原 Backend 外增加实际执行回执、持续接触检查、端点跟踪误差与真实相机观测；`native_run.py` 将这些数据加入 LIVE 后请求下一批动作。命令合法性错误和已确认零物理步的拒绝可在同一观测纠正；硬碰撞、跌倒直接终止。跨 episode 反思通过显式 `PAST_EPISODE_EXTENSION` 传入，重置后的 LIVE 从新初始状态开始。

`freeze_native.py` 核验并冻结 experiment.json、源状态、模型、运行时 provider、物理签名、代码和输入。每次运行重新校验冻结资源，禁止静默回退。物理入口必须通过 `scripts/humano_python.sh` 启动；它只接受已有完整冻结合约的实验目录，不能用空模板跳过准备。

原位置实验记录见 [`20261015_1047_roboicl_origin_live`](../../experiments/20261015_1047_roboicl_origin_live/)。其中模型由每次请求独立的新代理扮演，未继承父会话；每次请求、响应及 hash 均保存。不是外部 API 服务部署。物理成功与否以该实验的 result、独立 validation 和报告为准，不能从协议测试推断。

当前 head+external 相机及特权接触反馈与原版不同，已在合约声明。现有 backend 最短 0.1 秒、SONIC 腿 planner 约 0.5 秒更新，也不同于上游 25 Hz 双臂。短 chunk 和 LIVE 回执本身不能保证接近制动成功。

原版端点源读取测试仍因 h5py/XPolicyLab 缺失跳过；四个 RoboDojo/XPolicyLab 子模块未初始化，完整原版 IK/仿真未验证。历史离线验证保留在 [verification.json](verification.json)；本次物理接入前测试记录在实验目录 `preflight_tests.json`。

完整反馈运行入口为 `complete_run.py`，冻结入口为 `freeze_complete.py`。
该版本接入 standing-idle 和实测稳定门，并把整个已执行动作块的反馈交给下一次
LIVE。代码作用与历史版本关系见 [integration.md](docs/integration.md)。
物理实验仍须先冻结完整契约，通过 `scripts/humano_python.sh` 启动。
