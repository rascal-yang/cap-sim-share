# 后续 HUMANO 实验环境

自2026-10-14起统一使用 **TensorRT10.13＋ONNX Runtime1.23.2＋SONIC1.0 CPU**。

| 项目 | 锁定值 | 实际用途 |
| --- | --- | --- |
| Python | 3.10 | HUMANO运行环境 |
| TensorRT | 10.13.0.35 | 环境依赖；本profile不用于SONIC推理 |
| ONNX Runtime | 1.23.2 | encoder、decoder、planner均CPUExecutionProvider |
| SONIC | 1.0 / Default original release | 固定原始发布模型，按SHA256验证 |
| gear_sonic包 | 0.1.0 | 本地Python包装包版本，不是模型版本 |
| MuJoCo | 3.3.6 | 原物理参数与验收保持不变 |

NVIDIA区分Default original release、low-latency和v1.1三组模型，见[官方模型说明](https://huggingface.co/nvidia/GEAR-SONIC)。本地三个ONNX文件已与原始发布模型核对一致；此profile把用户约定的“SONIC1.0”绑定到这些明确哈希，而非重命名Python包。模型配置见[runtime/humano-cpu.json](../runtime/humano-cpu.json)。没有下载、切换或修改控制模型。

本机入口：

```bash
# 先检查环境；失败直接退出，不启动实验。
./scripts/humano_python.sh --check-only

# 后续实验使用统一入口；每次启动前检查依赖和模型文件。
./scripts/humano_python.sh -m cap_sim.cli run --task TASK.json --planner file --plans PLANS.json --out RUN_DIR

# 可保存每批实验的实际模型session后端与版本证据。
./scripts/humano_python.sh scripts/check_humano_runtime.py --load-models --out RUN_DIR/runtime.json
```

默认Python为`/home/kinetra-4090-01/HUMANO/.venv/bin/python`。TensorRT使用本机NVIDIA Debian安装包10.13.0.35解压到`~/.local/share/cap-sim/runtime/tensorrt-10.13.0.35`，由入口明确加载其Python绑定和动态库，未写系统目录。迁移机器时可设`CAP_SIM_HUMANO_ROOT`及`CAP_SIM_TRT_ROOT`，检查仍要求版本与模型哈希一致。

约束文件[runtime/humano-cpu.constraints.txt](../runtime/humano-cpu.constraints.txt)用于记录版本；模型版本通过JSON锁定。不要直接重装HUMANO全部依赖，其原pyproject仍声明旧onnxruntime-gpu且会与此profile冲突。当前仅在已有有效环境补充TensorRT，不改HUMANO源码、资产或历史轨迹。

安装TensorRT不等于启用GPU推理，EGL视频渲染也不等于SONIC使用GPU。旧实验仍保留当时的环境；不能回写为已经在TensorRT10.13下执行。此变更不改变CaP阶段顺序、动作权限、物理参数和成功判据。
