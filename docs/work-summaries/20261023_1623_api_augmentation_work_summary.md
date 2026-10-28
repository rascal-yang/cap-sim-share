# HUMANO 轨迹增广与 API 自主探索：当前工作总结

- 文档最后更新时间：**2026-10-23T16:34:07+08:00**（Asia/Shanghai）。
- 当前代码目录：`/home/kinetra-4090-01/code/zijian/cap-sim-api-augmentation`。
- Git 分支：`feature/api-augmentation`。
- 本轮两个批次的流程均已结束；部分任务因接口故障未完成物理测试。本文区分流程结束与物理验收通过。
- 当前批次：**20261023_1506_api3_autonomous_remaining_tasks**；前续批次：**20261023_1447_api_autonomous_ten_tasks**。

## 1. 工作目标与研究边界

建立与仓库其他数据生成路线隔离的 HUMANO 增广 pipeline，参考 Humanoid MimicGen 的分段及交互轨迹复用思路：从成功示范中识别接近、接触、操作等阶段，在场景扰动后复用/调整接触动作，通过真实物理仿真验证，再导出可复现记录。

当前执行方法明确属于**轨迹重定向**：规划 API 可以查看原始 replay、动作目标和分段标注，执行器保留原 token 接近前缀，之后使用 SONIC mode1，并允许 API 选择显式关节参考混合、偏置、PD 增益及接触力反馈。成功不能直接证明已学到可迁移的任务策略，也不能把导出的 SONIC token 单独视为完整动作。

长期希望获得的是可迁移的站位、接近、接触条件、抓取选择与恢复经验。当前还在验证底层复用、诊断和恢复流程是否可靠。

## 2. 已确认的工作方式

1. API 自己制定任务策略；监督者检查工具、记录、执行质量，补充通用工具，不手工替 API 选任务控制参数。
2. **初始任务参考仅成功 replay 和已有分段标注**。不再向新任务初始上下文注入历史失败修复方案、历史增广成功参数或旧验收审查。
3. API 仍可读取当前任务原生验收代码、场景/机器人模型信息、控制能力边界，以及自己本轮实验产生的反馈。这些是环境信息和闭环观测，不是额外示范策略。
4. `needs_tool` 不应直接结束任务，而应让 API 进入工具探索；只有确实没有所需能力、合理停止或预算耗尽时才结束。
5. 成功与失败均保留；此前成功的任务若在新版失败，需要单独解释。视频直接保存为 MP4，不要求 ZIP。
6. 保持独立工作区，不修改原仓库其他生成路线。实验记录使用月/日/单次实验目录，保留历史冻结内容。

## 3. 当前 pipeline

```text
成功 replay + API 分段标注
        ↓
绑定检查、运行环境检查
        ↓
API 检查原生验收标准
        ↓
API 探索：inspect / proceed / stop
  ├─ inspect → Python 查询证据 → 返回来源哈希及数据 → API 再判断
  ├─ proceed → API 提交审查、执行计划或结果复核
  └─ stop → 保存具体能力缺口或停止理由
        ↓
API 选择控制参数及 restart / checkpoint
        ↓
本地校验计划、冻结 experiment.json
        ↓
物理执行、保存高频状态与控制记录
        ↓
原生验收、渲染、API 复核
  ├─ accept → 保存通过记录和导出
  ├─ retry → API 查询/分析 → 提交下一次方案
  └─ needs_tool → 回到探索循环
```

诊断物理实验走同一个执行器、冻结记录和物理次数预算，没有隐藏的免费 rollout。只读查询不执行物理积分。

### 执行接口

- 仿真：MuJoCo 3.3.6。
- 运行环境：TensorRT 10.13.0.35、ONNX Runtime 1.23.2、SONIC 1.0 发布模型。
- encoder / decoder / planner 均使用 CPUExecutionProvider；TensorRT 存在不代表本次用它推理。
- 入口通过 `scripts/humano_python.sh`。
- 接触前：原始 motion token 前缀；其时长、来源和哈希逐案例记录。
- 接触后：SONIC mode1，API 可以选择额外关节参考混合/偏置、PD 缩放和有界力反馈。
- 不存在静默 IK、mode0 或其他控制器回退。
- 当前可调主要是 29 个身体关节；手指命令沿用源轨迹时序。它还不是任意手指动作/全自由策略生成接口。

### 回溯机制

每个策略帧保存完整物理积分状态、SONIC 历史与上一动作、任务计数器和接触控制器内部状态。API 自选 checkpoint 和帧号；恢复发生在新分支初始化，不是在运行中随意瞬移。

继承前缀与新模拟后缀分开计数、导出。成功必须区分是否早已存在于继承部分。此前按钮等价性验证确认相同控制下恢复后的状态/命令与父轨迹一致；当前批次也逐次核查实际回溯。

## 4. API 可用的诊断工具

当前工具是受控 Python 查询接口，API 选择案例、数据种类、起止帧和步长，不执行任意 Python 或 shell。

| 查询类型 | 内容 |
|---|---|
| `model` | 关节名、qpos/dof 地址、轴、限位、执行器顺序 |
| `aligned` | 对齐后的实际关节位置、原始/最终目标、误差、子步控制、根姿态、前控制接触日志、后控制原生指标 |
| `controller` | 50Hz 原始 SONIC 目标 |
| `trajectory` | 200Hz 状态、下一状态和实际控制序列 |
| `metrics` | 原生任务指标和成功/保持计数 |
| `contact_control` | 接触控制目标、PD 增益和已记录的接触数据 |
| `physics` / `result` | 物理配置、实际接口、执行结果 |

单次最多 32 行，API 可先稀疏扫描再放大。缺失字段返回未知/错误，不当成零。每次查询保存 API 决策、结果和来源哈希。

**已确认的能力缺口：目前查询接口未暴露精确的 200Hz 每子步接触力与几何。** 200Hz 状态/控制数据不等于有相同频率的接触力记录；50Hz 前控制接触值也不等于保护触发时的瞬时值。拿苹果已经因此停止探索。不能把未经验证的重新计算值称为原始实测值。

## 5. 模型、三 API 轮换和预算

- 真实模型：`gpt-6-astra`，`reasoning_effort=low`。
- 凭据文件：`api/openai_config.toml`、`api/openai_config 2.toml`、`api/openai_config 3.toml`。
- 三组凭据按共享序号轮换；初次请求加最多 5 次重试，即每个决策最多 6 次传输尝试。
- 新增第三个 API 后做了独立连通性验证，已获得有效工具调用；随后它已实际参与任务决策。
- API 超时 120 秒；任务首次阶段默认 900 秒，每个 retry 阶段 1200 秒，总预算 3300 秒。
- 每任务最多 3 次物理执行、24 次共享探索决策。`inspect`、`proceed`、`stop` 都占探索决策额度。
- 保护、验收及预算由本地校验；API 不得伪造成功或自行降低原生成功门槛。

## 6. 测试设计与前续结果

上一版批次 `20261023_1102_api_ten_tasks_rollback_validation`：10 个任务走过 API 流程，其中 5 个任务进行了 8 次物理执行；坐椅子、拿苹果、按按钮通过。其余有物理失败，也有 `needs_tool` 直接终止、未执行的任务。这促成了本次自主探索改造。

本次仍使用相同任务示范及**物体 X+5mm、Y=0**条件，保持原生验收不变。它是单一微扰条件的流程测试，不是每任务随机分布上的成功率评估，也不是另行约定的历史 IK/mode0 与 SONIC 十条件苹果对照。

前续批次 `20261023_1447_api_autonomous_ten_tasks` 在只使用两组 API 时执行了坐椅子一次；期间有超时/服务器过载。用户新增第三个 API 后，在无物理执行运行时停止，审计并保留旧结果。随后新批次继续其余 9 个任务，不重复已完成的坐椅子物理执行。

## 7. 当前任务验收快照

| 任务 | 物理次数 | 当前结果 | 原因/进展 |
|---|---:|---|---|
| 坐椅子 | 1（前续批次） | 未通过，预算耗尽 | 完整执行，但缺少原生判据要求的椅面接触，保持计数为0；API复核等待耗尽首次阶段预算，未执行retry |
| 拿苹果 | 2 | exploration_blocked | 两次均触发150N接触力保护；第二次从346帧回溯。API最终要求停止子步的200Hz接触诊断，现有工具未暴露，停止探索。 |
| 按按钮 | 3 | accept | 第三次成功。两次失败后，API自主延长尾段至2.8秒并从300帧回溯；新后缀连续有效按压1.98秒。 |
| 踢球 | 0 | error | API规划阶段重试耗尽，包含服务器失败响应和固定输入绑定错误；未进入物理执行，不能记为动作物理失败。 |
| 推椅子 | 3 | reject | 三次均未通过。最后已达到0.4128m位移，但连续达标计数2/10；非最终位移不足。 |
| 推车 | 1 | budget_exhausted | 首次完整执行，末端位移约0.281m，未达0.5m；API探索/复核等待耗尽首次阶段预算，未执行retry。 |
| 拿篮子 | 1 | error | 首次完整物理执行未通过；已开始诊断查询，但随后API复核请求发生URLError，未进入retry。 |
| 拿瓶子 | 0 | error | 六次轮换请求均未取得有效结果，URLError；没有新物理执行。 |
| 拿箱子 | 0 | error | 六次轮换请求均未取得有效结果，URLError；没有新物理执行。 |
| 拿盘子 | 0 | error | 六次轮换请求均未取得有效结果，URLError；没有新物理执行。 |

截至本文快照：两个批次合计 **11 次物理执行**（前续坐椅子1次＋当前10次），覆盖 **6个实际执行任务**，有 **1个任务获API接受：按按钮**。另4个任务因API问题没有新物理执行，不能按“10个任务都经过物理测试”计算成功率。3个任务进行过retry，共5次额外物理尝试，其中按按钮恢复成功，任务级retry恢复为1/3；共有5次checkpoint分支。

### 此前成功任务的单独回归说明

- **坐椅子**：新版本首个候选失败，后续因API等待预算耗尽未恢复。不能归因于回溯错误，因为该任务本轮没有回溯执行。
- **拿苹果**：两次执行失败，工具缺口阻止进一步诊断。第二次回溯状态核查通过；失败不是已证实的checkpoint恢复故障。
- **按按钮**：首次失败、第二次仍失败，第三次由API自主调整恢复成功。成功首次发生在新模拟后缀，非继承旧成功。

这些比较同时改变了 API 的初始参考策略和实际控制参数，不能据此直接宣称 SONIC 或物理引擎发生退化。

### 已结束任务的主批次成本

| 任务 | 墙钟秒 | 已知 token |
|---|---:|---:|
| 拿苹果 | 979.3 | 1,258,450 |
| 按按钮 | 759.5 | 1,663,301 |
| 踢球 | 202.1 | 221,551 |
| 推椅子 | 1381.7 | 1,822,129 |
| 推车 | 900.0 | 674,101 |
| 拿篮子 | 390.5 | 52,396 |
| 拿瓶子 | 1.9 | 0 |
| 拿箱子 | 1.8 | 0 |
| 拿盘子 | 3.2 | 0 |

两个批次及第三接口连通性检查累计已回报 **5,977,640 tokens**；另有 **56 次已结束请求用量未知**。另有 1 条遗留的 `started` 记录未回报用量，其中可能含被中断请求；不能将其当作零用量。以上为API回报计数，未换算金额。

规划、物理仿真、渲染和API请求时间分别按实际记录汇总；独立规划总耗时未测量，标为未知。三API主批次实际执行时间为2026-10-23T15:12:07.962058+08:00至2026-10-23T16:29:07.912932+08:00，总墙钟4619.95秒（约77分钟）；物理计算49.82秒，渲染269.13秒，API请求累计4178.08秒。前续批次的成本及连通性检查另保留，见final_accounting.json。

## 8. 已经验证与尚未验证的部分

### 已经验证

- 三API轮换支持及相关代码测试：**160项通过**。
- 只读对齐查询：对10条真实replay和8条已有执行轨迹做过读取检查，无新物理积分。
- `needs_tool` 可以实际返回查询循环，而不是立即终止任务。
- API能够自主选择查询窗口、形成假设、选择checkpoint并提交修改。
- 按按钮证明至少一个任务可经这种自主流程恢复成功。
- 已完成的阶段性执行审计核查了原token前缀、mode1实际执行、无中途状态重置、CPU会话及原生验收未改。三API批次最终10个物理案例及5条checkpoint分支审计均通过；前续坐椅子案例在新增第三API代码前审计通过。
- API来源审计核查了78个签名决策、28次查询来源哈希和初始仅replay/标注的输入策略。

### 尚未验证或不能宣称

- 4个任务没有新物理执行，不能把接口失败解释为物理不可行。拿篮子虽执行一次，但API复核未完成。
- 尚未证明对更大位置、桌高或机器人出发位姿扰动稳健。
- 尚未证明少量示范能够提炼出跨任务可迁移策略。
- 稀疏关键帧复核不等于完整时序视觉质量认证。
- 当前导出仍保留 `training_ready=false`；token单独不能重现带显式关节修正的执行。

## 9. 已暴露的问题及建议后续工作

1. **精确接触诊断不够完整。** 补齐每子步接触对、法向力、接触几何及保护触发时刻；区分原始记录与重新计算的近似证据。
2. **API成本过高。** `aligned` 等查询一次返回较多原始数据，并随后续请求累积；阶段切换又会重复查询。应由Python输出针对问题的短统计、异常帧及可追溯引用，并支持按需字段读取/复用已有查询结果。
3. **接口失败与决策失败混在较粗的错误消息里。** 应分别统计服务过载、超时、不完整响应、固定绑定校验失败和物理失败；不能只看一个error状态。
4. **固定绑定字段不宜依赖模型反复抄写长路径。** 可通过受限ID/枚举约束减少格式错误，同时保持原始API决策来源可验证。
5. **有限尝试次数可能截断接近成功的候选。** 推椅子最终距离已达标但保持不足；是否追加验证应由后续明确预算和API策略决定，不在本次冻结批次偷偷增加次数或降低标准。
6. **控制接口仍有限制。** 当前主要是mode1加关节参考修正，手指时序固定；不能把所有任务失败都归结为API推理或retry机制。

以上是测试发现和待办，**不表示这些改进已落地**。本次物理批次运行期间没有修改其绑定的pipeline代码或skill；增加第三API前先安全停止并审计，随后使用新实验目录与新冻结版本。

## 10. 代码与文档入口

| 文件 | 用途 |
|---|---|
| `scripts/run_contact_batch.py` | 串行批量入口，加载两/三组凭据，启动固定HUMANO环境 |
| `humanoid_mimicgen/augmentation/contact_batch.py` | 任务预算、批量状态、统计 |
| `humanoid_mimicgen/augmentation/contact_review.py` | 验收审查、API计划、物理执行、复核与retry编排 |
| `humanoid_mimicgen/augmentation/exploration.py` | API自主查询循环及只读证据接口 |
| `humanoid_mimicgen/augmentation/suite_provider.py` | 多凭据轮换/失败切换 |
| `humanoid_mimicgen/augmentation/retry_state.py` | checkpoint保存、校验、恢复与分支 |
| `humanoid_mimicgen/augmentation/suite_physics.py` | 冻结物理契约、实际执行与记录 |
| `humanoid_mimicgen/augmentation/contact_control.py` | 显式关节参考/PD/接触力控制 |
| `humanoid_mimicgen/augmentation/skills/contact_review/SKILL.md` | API使用的探索、控制及验收规则 |
| `docs/augmentation/autonomous-exploration.md` | 自主探索流程说明 |
| `docs/augmentation/retry-rollback.md` | 回溯机制说明 |

本轮新增/修改内容仍有未提交及未跟踪文件，尚未整理为一个新的Git提交。实验复现证据依靠每次冻结的文件哈希、源码快照、配置和原始API记录；不能只用当前分支名代替版本。

### 当前批次实际启动命令

```bash
cd /home/kinetra-4090-01/code/zijian/cap-sim-api-augmentation
python scripts/run_contact_batch.py \
  --config /home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1506_api3_autonomous_remaining_tasks/config.json \
  --provider /home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1506_api3_autonomous_remaining_tasks/provider.json \
  --credentials ../api/openai_config.toml \
  --secondary-credentials '../api/openai_config 2.toml' \
  --additional-credentials '../api/openai_config 3.toml'
```

该目录用于本次续跑/查看缓存；新实验需新建带实际创建时间的实验目录，不覆盖本次结果。API密钥没有写入本文。

## 11. 实验与视频索引

- [当前三API批次最终验收报告](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1506_api3_autonomous_remaining_tasks/VALIDATION_REPORT.md)
- [当前批次结构化状态](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1506_api3_autonomous_remaining_tasks/validation_summary.json)
- [当前批次配置](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1506_api3_autonomous_remaining_tasks/config.json)
- [当前批次设计与代码哈希](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1506_api3_autonomous_remaining_tasks/DESIGN.json)
- [API来源与查询审计](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1506_api3_autonomous_remaining_tasks/exploration_audit.json)
- [最终执行审计](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1506_api3_autonomous_remaining_tasks/execution_audit.json)
- [前续批次中断及成本记录](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1447_api_autonomous_ten_tasks/INTERRUPTED.json)
- [新增第三API前的执行审计](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1447_api_autonomous_ten_tasks/execution_audit.json)
- [上一版十任务报告](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1102_api_ten_tasks_rollback_validation/VALIDATION_REPORT.md)
- [按按钮第三次成功MP4](/home/kinetra-4090-01/code/zijian/experiments/2026-10/23/20261023_1506_api3_autonomous_remaining_tasks/tasks/G1WholebodyTableButtonPressTeleop/delivery/20261023_1539_tablebuttonpress_token_prefix_mode1.mp4)

其余已产生的成功/失败MP4链接位于最终验收报告逐任务表，合计11个MP4已检查存在且可解析。未进入物理执行的任务没有新视频。

最后阶段的URLError之后做了无凭据连通性检查：DNS正常，但TLS握手/读取出现UNEXPECTED_EOF。它是连接异常的补充证据，不足以断言此前所有失败都来自同一个原因。没有因这些错误降低验收标准或用伪造API结果继续执行。
