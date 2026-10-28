# API 验收审查与接触控制

> 当前默认已恢复稳定版，并增加已验方案优先复验，详见 [stable-seed-first.md](stable-seed-first.md)。下文保留历史接口演进记录。

独立入口 `scripts/run_contact_review.py`。当前真实任务适配验证为坐椅子；其余任务可复用关节控制工具，但尚不支持通用改写验收代码。

流程：核验成功原 token replay 和标注绑定 → 本地读取成功／失败记录的接触、碰撞距离和关节指令差 → API 审查原生验收 → API 选接触部位与关节修正 → 冻结实验 → 新物理执行 → 保留原生结果并独立复核 → 渲染 → API 验收或修复，最多三次。

API 可选择关节的原 replay 指令混合比例、静态角度修正、PD 增益比例和接触力积分修正方向／增益。源手指命令不变。接触力反馈在 50 Hz 本地执行，角度积分、关节速度与机械限位有界。停止力在 200 Hz 检查。测量为指定精确 body 对任务物体的正向法向力之和，不等于竖直承重。所有数值方案由真实 API 生成并保存。

原生验收代码不变。坐椅子允许 API 提议修订接触部位谓词，影子验收仍保持原生位置、高度、速度、椅子稳定及持续时间条件；原生／修订两列始终保留。只有审查明确提出修订时才允许以修订成功接受，并显式标识。验收审查可要求缺失工具而停止，不凭图像降低标准。

控制方式明确标注 `mode1_plus_explicit_joint_contact_residual`：原 token 前缀，之后 mode1 解码目标再叠加显式关节控制。`controller.npz` 保留 SONIC 原始输出，`trajectory.npz/commands` 保存最终目标和执行扭矩，`execution/contact_control.json` 保存每帧原／最终目标、PD 增益和力反馈状态。**仅 motion token 不足以复现这批数据**。属于轨迹重定向，不是新技能迁移；training_ready=false，需后续时序验收。

```bash
python scripts/run_contact_review.py --config /absolute/config.json \
  --provider /absolute/provider.json --credentials ../api/openai_config.toml \
  --secondary-credentials '../api/openai_config 2.toml'
```

config 含 output_root、object_shift_xy_m、task。task 含 task/dataset/baseline/annotation/failed_case 完整路径。采用已有双 API 交替及每次决策最多六次请求（首次加五次重试），不把密钥写入记录。

成功完成的 summary 可读缓存。中途重启会复用真实 API 决策和物理结果；存在 started 标记但无物理结果时阻止隐式重复。修改上下文必须使用新实验目录。所有新物理实例先冻结 experiment.json，记录技能／代码／模型／场景／API证据哈希。

完成后自动输出 delivery/ 下中文标注 MP4，并把通过的样本导出为完整动作数据集（包含执行关节目标、PD 参数、200 Hz 扭矩和状态）。可单独运行 `scripts/humano_python.sh -m humanoid_mimicgen.augmentation.contact_delivery --root /absolute/study`，无需 API 或物理重跑。

## 十任务短预算批量回归

`scripts/run_contact_batch.py` 使用同一凭证参数，config 提供 tasks 列表，每项包含成功 replay、已绑定标注与历史诊断。可显式传 baseline_inspection/diagnostic_inspection 指向独立验收目录，源哈希必须匹配。读取各任务真实验收源码和目标对象，不再按 chair 名称识别物体；关节物体通过 articulated 根 body 监测。

本批：统一物体 X+5mm；每任务至多3次物理尝试、末段至多延长3秒、墙钟预算600秒。API请求最多120秒，最多首次加5次重试，双凭证在整个批次交替。超时后不发新请求、不启动新物理；已开始的本地收尾不被中途杀掉。原生验收不改，坐椅子影子判据仅作诊断；其他任务该字段为不支持。所有成功、失败、未完成API验收分别记录，任何任务失败不会阻止其余任务。

`python scripts/run_contact_batch.py --config /abs/config.json --provider /abs/provider.json --credentials ../api/openai_config.toml --secondary-credentials '../api/openai_config 2.toml'`。批量汇总位于 summary.json、acceptance.csv、REPORT.md。每任务的 delivery/ 保存所有实际尝试视频与接受样本的完整动作数据。

## 2026-10-23：独立 retry 预算与 API 回溯

新配置默认首次阶段900秒、每轮retry1200秒，三次执行合计3300秒。`initial_wall_budget_s`、`retry_wall_budget_s`和`task_wall_budget_s`分别配置。显式旧配置600秒总上限仍生效；新实验应删掉旧上限或改为3300，不能修改历史实验配置。

API必须选择`resume`：从头执行为`{"kind":"restart","case":"","frame":0,"rationale":"..."}`；回溯为`{"kind":"checkpoint","case":"被提供的已完成case绝对路径","frame":整数,"rationale":"..."}`。每个50Hz控制帧前保存完整积分状态、SONIC历史、任务验收状态和接触控制器状态。旧数据不补造检查点。回溯点由API选定；同任务、场景、源数据、物理配置和判据必须一致。

每次回溯新建冻结case，仅初始化时恢复一次；记录完整分支及其祖先。`physics_steps`为完整分支记录长度，`executed_physics_steps`为本次真正积分的子步，`inherited_physics_steps`为祖先前缀；`simulation_s`只计新执行。数据集带`inherited_prefix`掩码和lineage，视频标注继承时长。恢复失败会阻止执行，不降级成只恢复qpos/qvel。

上一轮API复核和历次尝试摘要显式传入下一轮。`needs_tool`现已进入 API 自主证据查询循环；具体接口、预算及仅 replay/标注的初始参考策略见 [autonomous-exploration.md](autonomous-exploration.md)。未实现的物理能力仍需明确报告，不能伪造。
