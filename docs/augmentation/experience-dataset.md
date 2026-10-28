# 全任务经验数据集

独立模块：`humanoid_mimicgen.experience_dataset`。默认数据目录为仓库旁的 `experience-dataset/`，也可用 `--dataset` 指定。运行命令需在本仓库根目录，Python 3.10+，仅使用标准库和 SQLite FTS5，不加载机器人、不调用外部 API。

数据按一次来源事件保存，分别包含 `orchestration`（观测、诊断、指令、预期）和 `control`（实际接口、控制模式、参数来源、实测响应）。未记录的事实保持 null；报告、提议、回放和真实执行分别标记。所有来源均保留，不等于所有历史记录都能自动解释成完整经验。

## 日常使用

```bash
# 导入所有现有任务；同源同内容幂等，更新生成不可变新版本
python -m humanoid_mimicgen.experience_dataset --dataset ../experience-dataset ingest ../experiments

# 固定一次检索可见范围；名字不能重复覆盖
python -m humanoid_mimicgen.experience_dataset --dataset ../experience-dataset snapshot baseline_v1

# 编排经验（支持中文及常见任务中文别名）
python -m humanoid_mimicgen.experience_dataset --dataset ../experience-dataset query \
  --snapshot baseline_v1 --kind orchestration --text '苹果 抓取 接触失败'

# 查看被该快照绑定的具体记录版本
python -m humanoid_mimicgen.experience_dataset --dataset ../experience-dataset get \
  ATTEMPT_ID REVISION --snapshot baseline_v1

# 统计、重建可丢弃索引、输出模型工具 schema
python -m humanoid_mimicgen.experience_dataset --dataset ../experience-dataset stats
python -m humanoid_mimicgen.experience_dataset --dataset ../experience-dataset rebuild
python -m humanoid_mimicgen.experience_dataset tools
```

控制经验查询用 `--kind control`，必须同时填写 `--robot`、`--hand`、`--interface`、`--controller-version`、`--actual-modes`。这些值从当前执行配置和已核实的运行元数据获取，不能为了得到命中而借用其他案例的值。实际模式是 JSON 数组，例如 `'["original_token",1]'`。未知手型、未知模式或版本不一致不会被当成兼容。`both` 模式在控制不兼容时仅返回编排部分。

默认最多六条、12000 字符；不是精确 token 计费承诺。长内容明确截断；真实模型用量由调用者记录。结果始终包含 ID/版本和证据来源，证据哈希变化或丢失时不再参与正常检索。同来源、共享计划、原示范或回溯/回放关系形成连通组，组内去重。中文 bigram 与英文关键词走 SQLite FTS5，再按匹配程度和接口兼容性排序；不依赖嵌入服务。

## 自动记录与生命周期

```bash
# 持续观察实验目录；每轮完成后等待60秒
python -m humanoid_mimicgen.experience_dataset --dataset ../experience-dataset watch \
  ../experiments --interval 60

# 或包裹一个明确指定的生产命令；仅当你本来就准备运行该命令时使用
python -m humanoid_mimicgen.experience_dataset --dataset ../experience-dataset run \
  --root ../experiments -- YOUR_COMMAND ARGUMENTS
```

采集进程不修改现有执行器，读取其落盘文件；不在 200Hz 控制循环中写数据。`collector/status.json` 记录扫描状态和错误，`ingestion/sources.json` 保存持久导入账本，失败条目下轮重试。来源变化后追加新版本。进程 wrapper 只保存开始、PID、退出码，不保存可能含凭据的命令行；进程退出码不是任务成功/失败证据。

`watch` 是普通长期运行进程，不安装系统服务，不保证重启机器后自动启动。运行时 PID、启动日志由启动者保留，发 SIGTERM 可停止；停止不会删除记录。重启同一命令会从持久账本继续。采集器停机期间，已落盘且未被删除的文件可在下次补录；尚未落盘的内存观测无法恢复。生产器没有结果提交标记的记录保持未知，不猜测运行是否已死。

## 适配范围和覆盖账本

| 生产者 | 自动读取 | 主要边界 |
|---|---|---|
| augmentation suite | experiment/result/编译决策、指标、控制模式、token回放关联 | 同族实验脚本共享产物格式，不逐个改写执行器 |
| workflow 导航抓取 | strategy、阶段接口、native/physical结果、命令/观测证据 | 没有明确测量定义时，不推断“哪条指令导致反向响应” |
| core pipeline | request、decision、feedback决策、stage、acceptance | 继承的task单独绑定；未记录的推理保持未知 |
| 老记录/未知生产者 | 最小身份、证据、汇总、缺失字段 | 标为 partial/report；不冒充已验证物理执行 |

发现规则覆盖 result/experiment/request/decision/preflight 等标记、无后代尝试的汇总以及无映射的日期实验目录。排除依赖、源码快照、训练数据、成功复制目录、渲染目录；不跟随目录符号链接。只有视频或非标准散落文件仍需明确映射。`ingestion/latest.json` 给出扫描分母、任务分布、每条来源、错误及部分映射项，不能把数据库记录总数当物理尝试总数。

读到的 JSON 事实绑定同一批字节的 SHA256；提交前再检查来源未变化。结果引用的命令/观测及已知轨迹、关节/手指数组保存路径与哈希，避免在索引中复制密集序列。冻结合同声明的相关哈希与当前文件不一致时，经验不具备正常检索资格。跨目录引用只允许在配置 source roots 内解析；旧绝对路径经迁移映射解析，原文件不修改。

## 接 GPT

可通过 `query_experience` / `get_experience` 工具函数接现有 API 工具循环。工具 schema 从 `tools` 命令获取；宿主需自行绑定本地 Dataset 实例。查询日志写入 `queries/`，包含实际提供的经验内容和快照哈希，不包含认证配置。

现有 `plan(request)` / `reflect(request)` 规划器可使用显式适配器：

```python
from humanoid_mimicgen.experience_dataset.store import Dataset
from humanoid_mimicgen.experience_dataset.planner import ExperiencePlanner

planner = ExperiencePlanner(
    existing_planner,
    Dataset('../experience-dataset'),
    snapshot='baseline_v1',
    context={'kind': 'orchestration', 'task_id': 'G1WholebodyTableApplePickTeleop'},
)
# 在原 run_task/pipeline 调用中传入 planner；接口、预算和执行器保持原样。
```

适配器只添加 `experience_context`，不改原请求对象和返回动作。API 数值范围仍由原执行器校验。接入是 opt-in：不会静默替换正在运行的 benchmark，也没有声称现有所有入口已默认启用检索。批量候选在生成队列前检索；不能把读取新反馈后再生成候选与原固定队列实验混作同一方法。

每次读取经验后应核对：是否相同控制接口/手型/实际模式？例中指令是否真正实现？任务结果与指令结果是否不同？是否只是一条观察？有哪些差异和反例？无适用证据时按原规则继续，不虚构匹配或放宽控制权限。

## 验证收益与信息隔离

快照支持 `snapshot NAME --exclude-family FAMILY`；查询支持 `--exclude-family` 和 `--current-attempt-id`，会排除整个连通来源组。工具级 get 同样受快照版本和排除组约束；底层 Dataset.get 仅供管理，不作为模型工具暴露。严格评估应预先制作排除测试组的快照，而不依赖 GPT 自己传正确排除参数。

首次对照应固定源示范/场景分组、模型、预算、控制和验收，比较无检索与有检索。新评估记录可以入库，但固定快照不会自动看见它们。共享源示范的大量回放不能拆到训练与评估两侧；如果只剩同源材料，需补充独立来源后再做迁移结论。成功率、尝试数、重复无效指令和成本均需实际测量。

当前工具只完成存储、检索和接入基础设施，未运行检索收益的物理对照。历史反向响应缺乏对齐目标、参考系、时间窗或容差时保持 unknown；`classify_response` 支持在这些量明确时分类，不从“任务失败”推断控制方向错误。
