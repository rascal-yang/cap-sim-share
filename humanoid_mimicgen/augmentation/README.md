# API 调用 skill 和 Python 的批量增广

独立于其他数据生成路线。模型读取随仓库保存的 skill，再调用结构化工具；
采样、导航、稳定、闭环抓取、恢复、验证和导出由本地 Python 完成。
默认禁止运行期 GPT 抓取触发。源动作可见，方法属于轨迹重定向。

## 快速使用

新批次先创建按月／日归档的目录（北京时间、保留完整实验 ID）：

```bash
scripts/humano_python.sh -m humanoid_mimicgen.augmentation init-experiment \
  --root ../experiments --description api_augmentation
```

返回路径形如 `experiments/2026-10/21/20261021_1607_api_augmentation`。
将返回的 path 用作 job.output_root，experiment_id 用作 request_id，job/provider 配置也保存在其中。
批次内部 case 的物理证据仍保存在该批次自己的 experiments 子目录，不改变旧工作流。
历史目录已实际迁入月／日，外层不保留兼容链接；旧绝对路径在 `experiments/_index/latest-migration.json` 指向的映射中留档。冻结文件保持原文，旧配置重跑前需解析映射。
`index-history` 仅用于为后来出现的平铺旧目录建立分类入口，不是本次物理迁移命令。
目录名无法确定日期的项目列在 unclassified，不按文件修改时间猜测。

从当前分支仓库根目录执行。示例引用本机已有的苹果示范；迁移机器时替换
baseline_request 和 output_root。路径相对 JSON 所在目录解析。

```bash
scripts/humano_python.sh -m humanoid_mimicgen.augmentation validate \
  --job humanoid_mimicgen/augmentation/examples/job.json

scripts/humano_python.sh -m humanoid_mimicgen.augmentation submit \
  --job humanoid_mimicgen/augmentation/examples/job.json --store ../augmentation-store

scripts/humano_python.sh -m humanoid_mimicgen.augmentation inspect \
  --job-id <返回的job_id> --store ../augmentation-store
```

API 配置中的 model 替换为提供方实际支持的模型，base_url 为 Responses API
根地址（代码追加 `/responses`），凭据放在 key_env 指定的环境变量中。
可选配置 `reasoning_effort` 会传入 Responses 的 `reasoning.effort`，例如 `low`。
仅接受流式请求的网关可配置 `stream: true`；终止响应完整后才执行工具，断流不会推断成功。
自定义网关请求头用 `headers_env` 映射到环境变量名称，值不写入配置或日志。
没有 API key 时本地 validate/submit/inspect/export 仍可使用。
当前提供者只实现 Responses，不会静默改用 Codex CLI 登录或别的协议。

```bash
scripts/humano_python.sh -m humanoid_mimicgen.augmentation agent \
  --job humanoid_mimicgen/augmentation/examples/job.json \
  --provider humanoid_mimicgen/augmentation/examples/provider.json \
  --store ../augmentation-store
```

API 工具为 read_skill、validate_job、submit_job、get_job、get_case、export_job。
一次 submit 对应整个批次，运行中本地等待，不逐帧调用模型。
CLI stdout 是 JSON；运行库诊断在 stderr；后台工作日志位于 store/artifacts/job_id。

## 条件与复现

显式条件 `robot_delta_xyyaw` 单位是米、米、弧度；`object_delta_xy` 是世界 XY 米。
两者独立变化，不会在移动苹果时默认移动 G1。只改变新案例的初态。
stance_hint_frame 必须明确是 world 还是 object_relative；后者随目标平移站位参考。
源帧、控制器参数、验收和恢复预算沿用基线，不由模型在线调整。

随机采样可将 conditions 替换为：

```json
{"mode":"uniform","seed":17,"count":10,
 "robot_delta_xyyaw":[[-0.2,0.2],[-0.1,0.1],[0,0]],
 "object_delta_xy":[[0,0],[-0.05,0.05]]}
```

同时明确设置 budgets.max_cases。当前支持平面变化；桌高、新模型等需要单独
提供有证据的完整基线。新初态不能比原基线增加超过其容差的穿透。

request_id 相同且配置/输入/代码/skill 哈希相同会复用已有任务；同 ID 不同内容
会冲突。要做新的试验，用新的 ID。每例物理实验仍独立冻结合同并核验运行库。
同一 Unix 账号下多个 store/worktree 共享物理执行锁。

`resume --job-id ... --store ...` 仅继续未执行案例或处理完整记录；中断物理案例
保留 interrupted，不自动重跑。新 attempt 需要新任务 ID。Linux 进程身份包含
PID、boot ID 与启动时间，不能用遗留锁文件推断进程活着。

## 产物与含义

- store/jobs.sqlite3：任务、案例、工具调用和阶段记录。
- store/agents/session_id：按轮 API 请求、响应、时长与 usage（缺失为 null）。
- output_root/experiments：冻结合同、原始执行记录、独立验证和可选渲染。
- store/datasets/job_id/index.json：成功、失败、拒绝导出及无轨迹案例的完整索引。
- 每条导出包含 trajectory.npz、metadata.json、manifest.json 和原始证据。

trajectory.npz 的 qpos/qvel 是动作前状态，next_qpos/next_qvel 是该动作一步物理
积分后的状态；action_ctrl 为实际施加控制，另含 body target、手指、source_index
和 retry_index。attempted_commands 单独保留，不能混入已执行动作。
输出不是 SONIC token 格式，也不声称可直接替代 HUMANO 原始训练文件。
物理成功与视觉训练准备状态分开；未验收的视频不标成 training_ready。

整个批次 completed 表示案例处理结束，不代表全部抓取成功。成功率分母包含
失败与中断案例。导出拒绝会记录 reason_code；需要根据该字段判断产物完整性。
`export --job-id ... --store ...` 不运行新物理，同一内容重复导出复用原产物。

## 验证范围

测试中的 Scripted transport 仅验证协议和编排，不能替代真实 API 联调。
API 参数稍后提供时需要真实调用验收；未测价格记 unknown。
已有三组测试模块有重名与相互导入，分别运行，关闭环境 ROS 插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/humano_python.sh -m pytest humanoid_mimicgen/augmentation/tests -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/humano_python.sh -m pytest humanoid_mimicgen/workflow/tests -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/humano_python.sh -m pytest humanoid_mimicgen/tests -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/humano_python.sh -m pytest tests -q
```


## 运动扰动试验

可选 job.motion_noise 必须显式指定 seed、sigma_rad、interface="post_sonic_joint_target"、
resample="policy_tick"、joint_group="both_arms"。独立高斯噪声在每次 SONIC policy tick
采样，两个 tick 间保持，并按模型关节范围裁剪。作用于双臂 14 维关节目标，覆盖导航、
接近、接触和恢复；手指和腿部不加噪。逐例 seed 为任务 seed 加案例序号。
每例冻结合同声明该接口，使用 augmentation execute-case 执行。
导出同时保留 nominal_body_target_rad、带噪 body_target_rad、nominal_ctrl_Nm 和实际 action_ctrl；
无噪标签来自带扰动实测状态上的控制器输出，不声称是独立人工纠错标签。
迁移后的历史输入可用 relocation.rebase_baseline 构建新模板；先验证历史文件字节哈希，
只改派生副本中的路径，原冻结记录不变。

## 从原始按钮示范启动 API 流程

`bootstrap_agent` 是独立的任务接入入口，现阶段只支持 `G1WholebodyTableButtonPressTeleop`。
它不要求先构建苹果的 workflow request。真实 API 读取 `skills/contact_bootstrap/SKILL.md`，
选择源 episode 并调用原始状态检查、SONIC token 物理回放、接触阶段提取、mode1 小范围重定向和导出。
长任务在 Python 工具内同步执行，不为进度轮询调用 API。

从仓库根目录用系统 Python 3.11+ 启动（运行器会自动切换到已固定的 HUMANO 环境）：

```bash
python3 scripts/run_contact_bootstrap.py \
  --config /absolute/path/to/new_experiment/config.json \
  --provider /absolute/path/to/new_experiment/provider.json \
  --credentials ../api/openai_config.toml
```

config 字段：`task`、`dataset`（原始任务目录）、`episodes`（允许尝试的源 episode 列表）、
`augmentation_xy_m`（包含零偏移的固定条件列表）、`output_root`、`budget`（范围说明）。
当前试验使用最多三个候选 episode 和五个 XY 条件。正式使用前应显式给定有限列表。
provider 沿用 Responses 配置，凭据只存在进程环境中。已完成目录再次执行会返回已有结果；
新试验应使用 `init-experiment` 创建的新目录，不能覆盖旧记录。

本入口保留原任务的接触力与连续保持判据。原始 token replay 不通过 encoder；
mode1 使用示范腿部 q/dq、躯干和双腕参考，按钮独立移动，机器人初态不移动。
当前重定向执行整条示范；阶段标注作为独立产物保存，尚未拼接新规划的导航前缀。
未实现按钮任务内部的纠偏 retry，不借用苹果抓取恢复条件。输出不是 SONIC token 训练文件。
`dataset_index.json` 包含成功和失败的物理状态转换；`REPORT.md`、`metrics.json` 和 `api/`
分别记录结果、耗时和真实 API 请求/响应与 usage。未做视觉验收时 training_ready=false。

可选 `hold_tail: true` 会在每个条件末尾增加一段由原生 `hold_steps × reward_dt`
计算的保持时间。token 模式重复最后 token，mode1 固定最后 Cartesian 目标并将参考速度置零。
这是明确声明的接触补全，不是严格原始 replay。按钮首次接入实测：精确原始 replay 能对齐，
但源片段未满足当前奖励采样周期下的保持时长；补全后原 token 成功，而 mode1 五例均未成功。
不能将工具链已执行完成等同于成功增广。

网关 upstream_error、请求不可用及缺少终止事件的断流最多重试两次；错误响应中的工具不执行。
恢复会复用已保存的成功工具输出。已开始但没有结果的物理调用禁止静默重跑。
每次网络尝试都记录时长和 usage；失败响应缺少 usage 时总用量记未知，同时提供 known_usage。
