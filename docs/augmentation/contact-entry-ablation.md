# 接触入口控制器对照

独立入口：`scripts/run_contact_ablation.py`。仅修改独立 augmentation worktree，不使用苹果专用恢复策略，不改变其他数据生成路线。

```bash
python scripts/run_contact_ablation.py \
  --config /absolute/path/config.json \
  --provider /absolute/path/provider.json \
  --credentials ../api/openai_config.toml \
  --secondary-credentials '../api/openai_config 2.toml'
```

配置 `output_root` 为新建的月/日/分钟实验目录；`tasks` 每项提供 `task`、`dataset`、`annotation`（历史 API 标注 decision.json）、`baseline`（已成功的原 token 物理 replay 案例目录）。provider 使用 gpt-6-astra / low。凭据仅进入环境变量，不写入实验文件。每次实际请求交替凭据；单个决策最多初次加五次重试。

API 先批准匹配参数，再分别审阅两组：整段 mode1；原 token 执行 `[0, contact_start)`，随后 mode1 执行至尾帧，包括原批准的 tail。不在 contact_end 切回，不重置历史或物理状态，不改变手指命令、场景、原生验收。该比较保留原动作访问，属于轨迹重定向，不证明新策略迁移。

每个物理案例冻结 experiment.json、运行环境、模型、API 决策和源码快照。轨迹保存 200Hz 状态和 50Hz 控制模式。诊断按 pre-control 对齐，输出世界坐标有符号根部/双腕误差、姿态旋转向量、腿部 q/dq、手关节残差和手指命令。视频从已记录状态重建，不重新积分。

`--resume-errors` 归档任务错误并复用已完成物理案例；只重试失败的处理环节。已成功审阅任务不会重新执行。每组独立保存结果，一个诊断或 API 错误不会阻断另一授权组。启动标记存在但物理结果缺失时停止该组，必须核对原案例，不能猜测并重复执行。排他锁禁止同一研究目录同时运行两个调度器。未完成的只读渲染目录归档后重建。

原生成功、独立验收、完整覆盖、无安全停止和 API 接受必须同时满足，才能记作质量通过。API 仅接受“诊断证据充分”不能覆盖原生失败；不自动标记训练就绪。切换前已成功和尾帧成功单列。

本轮 20261022_1719_contact_entry_ablation：苹果、按钮、坐椅子、踢球各两个新案例。报告的厘米误差级别仅用于描述首次越界时间，不是执行参数或验收门槛。各任务每组仅一次，不能估计跨场景成功率。
