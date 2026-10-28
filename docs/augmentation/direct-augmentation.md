# 跳过零扰动硬门的直接增广

入口与 skill 独立于原质量管线；原质量管线仍维持原有零扰动门。

```bash
python scripts/run_direct_augmentation.py \
  --config /absolute/study/config.json \
  --provider /absolute/study/provider.json \
  --credentials ../api/openai_config.toml \
  --secondary-credentials '../api/openai_config 2.toml'
```

配置含新建的 `output_root`，以及 `tasks` 中的 `task`、`dataset`、`baseline`（成功原 token 物理案例目录）、`annotation`（API decision.json）、`zero_case`（仅用于诊断）。历史 annotation.json 必须绑定同一 baseline/result.json 和哈希；API 标注请求中的接触帧与指标必须相符，原始数据哈希必须一致。

流程：成功 replay + 标注 → API 选择两个不同非零 XY 扰动 → 候选执行与审阅 → 至多一次 API 授权修复 → 审阅并导出。零扰动失败不阻断。边界为对象平移范数 2cm、额外腕部平移范数 5cm、尾部延长 3s；这些是本批实验预算，不是通用任务成功阈值。

固定 episode、contact_start、机器人初态、手指命令、物理和原生验收。接近段保留原 token；接触起点到结尾用 mode1。物体 XY 平移作用于所有 mode1 帧的双腕目标；额外腕部偏移只在 API 指定的 contact_start..contact_end 内生效。腿/脚/骨盆参考不平移。支持离线结果反馈后的新尝试，不提供在线力伺服或手指时机修改。

API 使用 gpt-6-astra low；每次实际调用交替凭据，最多初次加五次重试。Python 提取力、保持计数、物体运动和关键帧；保留全部 MP4 和 200Hz 状态。每次物理前冻结模型、已加载 CPU provider、源码/skill、API 决策、场景和参数。

重复同一命令可恢复服务/渲染错误，复用完成的物理缓存，不重跑已完成任务。无法确定是否完成的 started 标记会阻止自动重复物理。一个候选出错不阻断其他候选。物理已完成但渲染失败的尝试仍计入次数和用时。

`summary.json` 区分首次成功、最终候选成功和 retry 恢复率；`accepted_manifest.json` 仅包含原生与独立验收、完整覆盖、无安全停止且 API 接受的轨迹。导出状态仍为 `training_ready=false`，需单独完成时序视觉审核。失败轨迹不进入成功清单，但完整保存。

首次批次：20261022_1823_direct_perturbation_api；苹果、按钮、椅子、踢球共 8 个候选预算、最多 16 次物理尝试。该流程属于保留原动作的轨迹重定向，不证明新策略迁移。

主入口结束时自动生成 `accepted_dataset/manifest.json` 和明确字段的 `episode.npz`；再次运行只验证并复用相同数据，不覆盖不同内容。也可单独运行只读导出：

```bash
scripts/humano_python.sh -m humanoid_mimicgen.augmentation.direct_export --root /absolute/study
```

后续物理案例统一存于任务级 experiments/ 下，避免不同候选在同分钟生成相同 ID。本批冻结的旧路径不改名，以完整路径和候选/尝试编号区分。视频合集是验收用途，带中文参数标签；训练清单仍指向原始逐帧对齐的证据，不使用合集作为训练视频。
