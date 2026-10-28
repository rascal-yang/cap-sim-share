# RoboICL TRAIN、配置与启动链逐文件审计

> 后续接入说明：本文保留初次静态审计/离线阶段的状态；现已新增 native G1 + Dex3 桥接与跨接口 TRAIN 观测记忆路径，见 [integration.md](integration.md) 和实验 `20261015_1047_roboicl_origin_live`。原版完整环境与同接口 one-shot 仍未验证。

审计对象是本地 `RoboICL/`，`git -C RoboICL rev-parse HEAD` 实际输出为 `c0e58e456ce4b2ea1c401d1b261d6e0327c6e04c`；读取时 `git status --short` 为空。本文依据该提交的源码与 JSON，不把文件里的论文、官网、模型或服务商名称当作已联网验证的事实。这里没有安装环境、下载资产、发起模型请求或运行物理实验。TRAIN 是上下文中的示范，不是梯度训练。

覆盖清单：`roboicl/{config,paths,reference,reference_setting,robodojo,run,upstream}.py` 七文件；`policy/reference_data.py` 与四个 `train*.py`；`setup/` 六文件；`configs/` 四个根 JSON、两个 protocol、一个 reference setting、42 个 task JSON，共 49 个配置文件。根目录下所说 config.py 等，实际位于 Python 包 `roboicl/`。配置的实际值、描述与评分条件在后文逐文件展开。

## 调用链与证据边界

`setup/fetch_assets.py → data/Assets + .roboicl-assets.json`；`setup/fetch_reference.py → reference_setting → reference.build → reference_data + train_reference_bundle + train_reference_checks → train_reference_bundle.json + verification.json`。

`python -m roboicl.run → load_profile/load_task/check → asset/reference preflight → verify_submodules → policy_server + robodojo`。policy 一侧 `bounded_policy.Model` 载入 demo，`dialogue_policy._initialize` 验证并调用 `train_tool_reference.tool_reference_messages`；sim 一侧 `robodojo.main → create_eval_env → rollout.eval_one_episode → evaluator`。TRAIN action 是给模型看的历史示范，序列化不会调机器人；LIVE action 才进入执行链。因 planner 能读原示范动作，研究时不能写成“原动作不可见的纯经验生成”。若下游直接 replay 后 IK 重定向，应明确叫轨迹重定向。

## 1. roboicl/config.py

- 调用方：run、reference、fetch_reference、bundle validation。输入 JSON 路径、任务名、variant、seed/layout；输出合并 profile、task 字典、确切路径以及官方 prompt 三元组。
- `read_json` 要求顶层 object；缺文件、JSON 解码错误直接传播。`task_names` 按配置文件名排序，不查 task registry；`load_task` 只读字典，未知任务 ValueError，并不在这里完成 task schema 校验。
- horizon 必须恰好覆盖全部 42 task，值必须是严格 int（不收 bool），范围 1..64。允许 `task_horizons` 旧式整表、相对于 profile 目录的 `adaptive_horizon_file`、默认公共整表三种来源；旧式整表和文件不能同时给。稀疏 overrides 的任务名必须已知，合并后写回 `profile['task_horizons']`。
- profile 要有非空 id、object overrides、shots 0/1/None、非负严格 int TRAIN chunks；0-shot 要 J=0、1-shot 要 J>=1；LIVE anchor 要正 int。`shot_count not in (0,1,None)` 使用 Python 相等比较，因此 bool/等值浮点的检查没有严格类型防护，勿称为完整 schema validator。
- standard/random 只映射到 `task.py` / `task_random.py`，必须存在；layout 为 `Assets/Eval_Layout/RoboDojo/arx_x5/<seed>/<actual_task>_<layout>.json`，不会扫描补号。seed/layout 必须严格非负 int。
- reference 路径为 `runtime-data/<task>/reference_1shot_train<J>_endpoint_h<H>`；H 1..64，J 正 int。路径 helper 不验证源数据。
- `render_official_task_prompt` 验证 documentation 的 version/source/description 非空与 scoring 非空且每行仅 score/condition；生成 OFFICIAL DESCRIPTION、可选 COMMENT、SCORING。评分文本被送入 prompt，真正计分仍在仿真 evaluator，二者不是同一个可执行判据。

## 2. roboicl/paths.py

- `CODE` 从模块位置解析，四个上游都在 `third_party/`。`data_root` / `results_root` 依次用显式值、ROBOICL 环境变量、仓库默认，expanduser/resolve；不创建目录。
- `python_paths('policy')` 使 XPolicyLab 在 RoboDojo 前；sim 相反。二者都有顶层 `utils`，优先级避免导错模块。列表还包括 CODE、third_party、curobo 和 IsaacLab/source 下各子目录。
- `configure_imports` 将该列表置于 sys.path 前并剔除重复路径；错误 context ValueError。没有检查路径存在，也不是依赖安装。调用方：reference、robodojo、upstream.activate；run 用 python_paths 设置子进程 PYTHONPATH。

## 3. roboicl/reference.py

- CLI `build` 接 task/source/source-sha256/output/horizon/chunks/source-id/selected-starts/reference-setting-id；`verify` 接 reference/chunks。默认 J=12；输出 JSON 报告。导入会配置 policy 路径并要求 h5py/numpy/XPolicyLab 的 decode_image_bit 可用。
- `file_sha256` 分 4 MiB 块算源哈希。`build` 要新目录；源哈希不符立即失败，不覆盖；source-id 必须非绝对、无 `..`。源频率必须恰好 25 Hz，instruction 从 HDF5 解码。
- `_valid_chunk` 用重建的 [H,14] 行，逐臂检查平移和旋转向量范数。`select_starts` 强制 frame0 和 `total-1-H` 真终点块；中间块用动态规划最小化与均匀时间目标的绝对距离，同分按最早字典序。块不重叠、全都满足动作限幅；长度不足、首尾越限、无可行路径均拒绝。复杂度含候选前驱扫描，长记录不能假定线性。
- `validate_locked_starts` 保留文献 manifest 中的选点，不静默重选；检查 J、排序唯一、窗口不重叠、边界与限幅。这个函数自身未强制首帧为 0，后续 bundle validation 会强制；也未要求末块就是末帧。
- 每个 example 含 start state、三路 RGB JPEG、H 步 action_tensor、闭 action 区间 `[start,start+H-1]`、result_frame=start+H、真实 result state/三路 RGB；另外独立存 `total-1` terminal。三相机来自 CAMERAS，不生成预测终点图。
- `reference_data.tensor_at` 重建动作，再走 `validate_actions`。统一声明 world ee_delta、25 Hz、native_jpeg/original/triptych；child 标记 `train_action_chunks_world_ee_delta_v1`，bundle 标记 `train_reference_bundle_v2`，含 episodes 与扁平 examples 两份内容。
- 写盘前把验证副本的 source_file 改成实际绝对路径并重算副本哈希，执行全量源审计，再把报告 source_sha256 绑回便携 bundle 的哈希。写 `train_reference_bundle.json`、`verification.json` 用 x 模式；如第二个文件写失败，可留下部分输出目录，重跑不会覆盖。
- `verify` 要一个 source episode，每集正好 J 个 chunk，再完整重读源；它不支持 run 的缺源 sidecar 快路。
- 已见不一致：build 只从 task override 取 `max_rotation_rad`，平移仍固定 shared 的 0.03；build_tower 与 classify_objects 的 0.031/0.034 只在 run 合并配置时生效。因此某个合法 LIVE 训练窗口仍可能在重建阶段被拒绝，必须先验证实际源才能说复现成功。
- 已见边界：`select_starts(count=2)` 被前置检查允许，但无 interior targets，`states` 保持空，随后 min(empty) 失败；默认 J=12 不触发。`train_image_count=2J` 与 `terminal_observation_is_last_chunk_result=True` 是构造时固定写入；锁定 starts 不一定以真终点结束，不应以这些 metadata 替代实际图像预算计算。

## 4. roboicl/reference_setting.py

- 调用方 fetch_reference 与 run.reference_preflight。相对 manifest 路径以 CODE 为根；schema 必须 `roboicl.reference_setting.v1`；严格要求 8 个 zero-shot、22 个 one-shot、合计 30 唯一任务与 J=B=12。
- one-shot row 禁止 evaluation_status/exception，source 禁止 bundle_source_episode；install/repository path 必须便携、非空、无 `..`。episode 必须严格 int 且与 install filename 的七位编号吻合；不会要求 repository filename 同号（Xylophone 特例见下）。
- starts 必须 12 个有序唯一 int，相邻至少 H，末个结果不能越界；H/total 要 int，但该层未全面验证正 H、非负 starts、source sha256 字符格式或 task 是否 configured。后续验证承担这些条件。返回额外 `_path`；`reference_record` 要恰好一条匹配，否则 ValueError。

## 5. roboicl/policy/reference_data.py

- `state_at(data,frame)` 从 state 组提取 left/right ee_poses、arm_joint_states、ee_joint_states，共六字段列表。要求所有 dataset 存在，缺键/越界由 h5py 抛错。
- `tensor_at(data,start,count)` 首步 previous 取真实 state/start，后续 previous 取上一 action target（不是下一 observed state）。每臂行是 `dx,dy,dz,rx,ry,rz,gripper`，双臂拼成 14 维，round 6 位。
- source quaternion 假定 `[w,x,y,z]`，转换成 scipy `[x,y,z,w]`；旋转差 `Rtarget * Rprevious^-1` 的 rotvec 是 world 旋转增量。gripper 只取 action/ee_joint_states 的第 0 通道。该实现没有 Dex3 多指关节、全身/底盘动作、速度/力矩动作语义。
- 调用方 reference.build/选择器与 source integrity。函数自身没有动作限幅或 nan 检查，调用方必须验证；scipy 会拒绝无效零四元数。

## 6. roboicl/policy/train_reference_checks.py

- `source_path` 相对路径以当前 data_root 为根；历史绝对路径不存在且有 task 时，可重定位到 `runtime-data/<task>/train/<basename>`。重定位后必须由调用方 SHA 校验；不会仅凭同名认为等价。
- `train_calibration_profile` 返回 environment_origin、原点 [0,0,0]、camera_to_world_usd；root 参数未使用。`finite` 检查形状与有限值；file_hash 分 1 MiB 读；digest 是 sort_keys JSON SHA，不同于 bundle 的 compact JSON digest。
- `source_state` 与 state_at 类似但跳过缺失 dataset；完整 bundle 的后续长度检查会补充要求。`source_cameras` 支持静态或每帧 K，外参必须 `[N,4,4]` 每帧矩阵，translation 减 origin；shape 可静态或每帧，camera_arrays 继续验证。
- `verify_training_manifest` 可递归到 bundle verifier；非 bundle 要显式 USD camera convention、3 维有限 origin、源 SHA 完全相等、world ee_delta、25 Hz、非空有序不重叠 chunks、区间/result state/起点 state/tensor 与源逐项完全相等。报告包括 head reference/calibration/chunks；只检查源一致性和标定提取，不做 FK/IK、投影一致性、接触或任务成功审计。
- `training_summary` 去掉大标定/状态内容，仅保留 schema/passed/hash/审计范围，chunk 及终点/input hashes、selected image hashes，bundle 递归。sidecar 是本地数据完整性凭据，不是签名认证、控制器日志或成功证明。

## 7. roboicl/policy/train_reference_bundle.py

- `bundle_digest` 排除自身 sha256 后，以 sort_keys/ensure_ascii=False/compact separators/allow_nan=False 整包寻址；`portable_bundle_digest` 还将每个 source_file 替成 `sha256:<sourcehash>`，可跨安装路径比较。两个哈希不能混用。
- `validate_bundle` 是纯数据结构检查：variant、64 位小写 SHA 与整包哈希、至少一 episode、支持任务、非空 instruction、25Hz world ee_delta、native triptych 编码一致。各 episode 的 COMMON_FIELDS 必须同 bundle；source 路径 casefold 后和 source hash 都不能重复。
- 每集 total>=2、1..47 chunks，各集 chunk 数相同；H 1..64；首帧 0，随后按 result 边界排序、不重叠且等 H，结果严格小于 total。tensor 必须 [H,14]、finite、原始值 int/float（不收 bool），两夹爪 [0,1]。此层不检查平移/旋转 norm。
- terminal 必须 total-1、不早于末块结果，且没有虚构 action/result_frame；每个 start/terminal 要 state 和三路 JPEG data URL，head_image 等于 cam_head；result_state 必须 dict，result images 允许旧版缺省，若有必须三相机。扁平 examples 必须逐值等于所有 episode examples 拼接。
- `verify_reference_bundle` 先结构检查，再 source SHA/动作检查、六类 state/action dataset 长度、instruction、每个 selected/terminal observation 的 state/三路原图重新 JPEG 编码完全一致与 [480,640,3] shape、每个 result state/可选 result RGB。终点标定另提取。报告显式承认历史 writer 的坐标语义不是通过 HDF5 字段名认证，且没有 kinematic/projection audit。
- 调用方 reference.build/verify、train_reference_checks、dialogue 初始化。缺源、错哈希、错字段、错误图片或标定全部异常失败；不会修补 bundle。

## 8. roboicl/policy/train_endpoint_observations.py

- `result_observations` 供 train_tool_reference 在 include_result_observations 时调用。去重 cameras，必须非空且在 CAMERAS 内，返回以 result_frame 为键的 `{frame,state,prompt_images}` 字典。
- 所有 examples 已有请求 camera 的 result_prompt_images 时走 embedded 快路，仅允许 native_jpeg；复制 result_state 并按固定六键顺序序列化。**这个分支没有重新读源、没有自行比对源哈希**，文件 docstring 的“even precomputed checked against raw recording”不是对这个函数所有分支的准确描述；它依赖外层 validate/source audit 或通过 sidecar 复用的先前审计。
- 无完整 embedded endpoint 时要求 source_file 存在、源 SHA 相等；一次开 HDF5 验证总帧数、相机帧数、interval 与 tensor 长度、result state、native shape=480x640x3，解码并重新编码；若 supplied 部分 images 与重建不一致则拒绝。禁止用起点图假装终点。
- legacy 回退调用 source_path 时不传 task，因此失效绝对路径没有本函数内 canonical task 重定位。空 examples 的 all() 为真会返回空字典，必须依赖外层 nonempty 校验。

## 9. roboicl/policy/train_tool_reference.py

- 输入单 episode demo、首段 header、与 LIVE 相同的 tensor converter、可选 consistency，以及 camera/image/endpoints 选项；输出 provider 的 user/function_call/function_call_output 消息列表。不发送 API、不执行 action。
- 未传 consistency 且有 geometry_profile 时调用 verify_training_manifest；已有 consistency 不在此再次检验 passed/hash，依赖 dialogue 的上层约束。要求 frame0、连续块内 end=start+count、块间非重叠；converter 验证每行且不能截断。
- 每块 user 放 TRAIN 标识、observation_step、robot_state 与 start 图；act 参数保留原 tensor 与空 execution_note；call id 由源 SHA 再哈希取 16 位加 index/start，来源区分稳定。
- 生成 receipt 的 executed/predicted/dispatched 数等于源标签 count，ledger hash 基于重建 commands，latest_tracking 比较末目标与 result_state。这是示范标签转成的回执，不是历史控制器真实 event log；controller_events、terminal、success 保持 null。不能把它当作“示范执行成功证据”。
- include_result_observations 时插真实 endpoint user observation 与 observation_end marker；采样块之间用 omitted_interval 显式表示缺失时间，不能假装相邻示范动作连续执行。真 terminal 若已等于末 endpoint 不重复，否则单独发；可追加固定参考结束说明，之后切换 LIVE。

## 10. roboicl/upstream.py

- `verify_submodules` 遍历 configs/upstream.lock.json；要求路径目录存在、git rev-parse HEAD 等于 lock、git status --porcelain 完全为空，输出检查 package 清单。目录缺失、commit 不匹配、dirty 都拒绝。路径存在但未 init 子模块也不能认为可执行；git 命令错误传播。
- `activate` 只 configure_imports，不验证、安装或拉取。run 的 dry-run 与真实启动都会 verify；lock 中所谓 integration 是固定 RoboDojo fork commit 的功能说明，不代表原始发布版毫无改动。

## 11. roboicl/robodojo.py

- CLI 必需 task/seed/layout/data-root/results-root/port，host 默认本机，支持 capture-only 与 AppLauncher 参数。先验证 arx_x5 exact layout，建立结果目录，设置两个 root 环境变量并 chdir，随后导入和改 env.global_configs 的资产路径，避免上游 import-time snapshot 指错目录。
- AppLauncher 启动后加载 arx_x5.yml 和 sim/scene/camera/robot/task YAML，设 num_envs=1、device_id=0、eval_batch=False、policy=RoboICL、action_type=ee、physx_monitor=False、eval_num=1。CUDA_VISIBLE_DEVICES 由上层将选卡映到设备0。
- deploy 使用 WebSocket host/port，evaluation_id 从 ROBODOJO_RUN_ID 取；直接调用 adapter 若不设这个变量仍会 KeyError。process_randomization 后 process_config，再强制两个 num_envs=1、sim.seed=[0]，camera.default_frequency 来自 observation.collect_freq（未设则0）。这里 sim.seed 与 CLI seed 不同，不能笼统说“一切随机种子就是 CLI seed”。
- create_eval_env 后替换 seed_manager 的 seed_info/list/st_idx/ed_idx/idx 和 env_seeds，精确绑定 CLI layout，reset(seed=[layout])，避免上游枚举后 gap 重编号。
- capture-only 在构造前替换 WsModelClient 为只实现 reset 的 OfflineClient，随后 camera_preflight.run；普通运行用 MethodType 将 env.eval_one_episode 换为 owned rollout，再 env.run_eval。这里不是 G1 adapter：robot/config/result 路径都是 arx_x5。
- 捕获 BaseException 时打印当前 stage 与 traceback 再抛出；finally 关闭 transport/env/simulation_app，transport/env cleanup 的异常吞掉。AppLauncher 本身创建失败在该 try 之前，不享受同一 finally。

## 12. roboicl/run.py

### 启动前检查与配置优先级

`python_path` 允许 executable path 或 PATH command，只查可执行文件，不做 Python 版本/import/provider 实测。`check` 要 profile shots 与 CLI 一致，task/variant/source/layout/两解释器存在；配置依次 shared → profile overrides → task overrides → adaptive H（预测=执行）→ CLI action-horizon。官方文档生成 prompt；instruction alias 按 variant 取唯一非空字符串。CLI reasoning/output 覆盖在 preflight 后应用。

`asset_preflight` 对完整树 manifest 检查 repo/revision/include；**不重哈希完整资产树**，记录 tree hash 而不是现算证明。explicit-files 模式则逐条验证 Assets 下安全相对路径、size、sha，且必须包含选中的 layout；仍不是 USD 所有外部依赖已闭合的证明。

`reference_preflight` 1-shot 检查 bundle canonical hash、task、episode 数、J/H、verification sidecar passed/hash、每行 14 数值+norm+gripper、要求 endpoint 时三相机齐全。源在本地则重新 SHA；缺源可依据唯一 matching passed episode sidecar 继续，记 `offline_verified_bundle`，不是本次重新看过源。对照 reference setting 的 H/starts/source SHA/episode/install basename。计算真实 reference image_count 与 anchored worst-case LIVE image budget，超过 cap=50 拒绝。0-shot 不装示范，但仍查图像预算。run 本身不做 h5py 解码，policy 正常有源时再全检，缺源时 bounded_policy 可恢复 sidecar。

### 子进程、环境与产物

- CLI shots 必需 0/1；seed/layout/gpu 默认0且非负；variant standard；两个解释器来自环境或当前 Python；profile 默认 zero_shot_b25。仅 `--shots 1` 不会自动选 1-shot profile。model 优先 CLI、ASTRA_MODEL、profile；显式 profile endpoint 覆盖继承 shell endpoint。
- horizon 1..64，max-output-tokens 256..65536，支持 omit。launch_blocked_reason 对实际 rollout 阻止、dry/capture 放行。创建端口先 bind 再释放，实际 server bind 仍有竞态可能。
- runid 用本地无时区 datetime 到微秒，附 task/shot/seed/variant/layout/reasoning/output/H/LIVE/taskdoc；这**不符合当前项目要求的 experiments/YYYYMMDD_HHMM 与时区记录**，接入 cap-sim 必须外包合规实验记录，不能拿上游 runid 冒充本项目 frozen experiment.json。
- 强制 ASTRA_TRACKING_GUARD/VALIDATE_RGB=1、NETWORK_MODE=direct、STREAM=1、RETRIEVABLE_HISTORY=0；清理部分继承变量；显式 PYTHONNOUSERSITE/PYTHONUNBUFFERED、两边各自 PYTHONPATH、GPU 映射。可选 conda run 和 sim LD_PRELOAD；存在 NVIDIA ICD 时设置 Vulkan。
- server: policy_python -m roboicl.policy_server；client: sim_python -m roboicl.robodojo + enable_cameras/headless/Kit camera 扩展。capture 不启动 server；dry-run 完成路径/资产/参考/子模块预检只打印 record，不创建结果目录、不起模拟/模型请求；它仍要求资产和子模块已存在。
- 实际运行要求 ASTRA_API_KEY/ENDPOINT（capture 除外），每 host/GPU 两个 flock slots，非阻塞占满即失败。锁文件在 results_root，锁由保活 file handle 持有至进程结束。
- 写 harness.json、run.json（参数、layout hash、lock/profile/release manifest hash、解释器、server/client command、模型服务、reference/asset preflight）、code_diff.patch、server.log/client.log。run.json 记录 code_commit，但未把每阶段开始/结束 ISO 时间、耗时和全部 dependency hashes 写成项目实验模板。

### 失败行为与成功判据

WebSocket readiness 是 HTTP 101 handshake，不调模型。启动最多等300秒；server 提前退出立即失败。client 运行中 server 退出，给 evaluator 60秒收尾；之后失败。stop 按进程组 SIGINT(30s)→SIGTERM(10s)→SIGKILL(5s)。SIGTERM 转 KeyboardInterrupt，标 interrupted/130；一般异常写 failed/1 后重新抛出，finally 清子进程。

client 返回0还必须找到精确 `_result.json`，details 恰好一条且 layout_id 匹配，复制 result.json；policy_error 时标 scored_with_error，否则 scored。**scored 和进程返回0均不等于任务成功**，必须读 evaluator score。capture 要 probe 中 observation 路由正好30条、RGB shape [480,640,3]、max 非0，标 capture_complete_no_api；该 shape/max 检查不是视觉真实性验收。最终 status.json 存 status/returncode/note。正常 rollout 没有这里统一的总运行超时；policy 预算不应被误写成 launcher 的全流程超时。

## 13. setup 逐文件

### setup/paths.sh

应在 Bash source，CODE_ROOT 从 BASH_SOURCE 定位；保留已有 DATA_ROOT/RESULTS_ROOT，默认 code/data、code/results；将 CODE_ROOT 前置 PYTHONPATH。它不创建环境、不 source conda、不设 API 密钥，非 Bash 不保证工作。

### setup/fetch_assets.py

输入 --data-root（默认 code/data）、必需 --cache，读取 data.lock.assets；全 Assets 树有意保留 USD/材质传递依赖。若已有 Assets 不是指向本 cache/Assets 的 symlink，立即保留并失败；已有 cache 非 Git 也拒绝。新 cache git init/add origin，HF_ENDPOINT 默认 huggingface，ROBODOJO_ASSET_REPO_URL 可替代 origin。

固定 HTTP/1.1，GIT_LFS_SKIP_SMUDGE=1，fetch depth1 锁定 revision、cone sparse-checkout Assets、detached checkout，确认 HEAD；lfs install/pull Assets/** 与 fsck；检查 Robots/Object/Room/Material/Sensor/Eval_Layout 六目录后链接。任何 subprocess failure 传播；不自动删除残留 cache。

`materialize_robot_configs` 读 Assets/Robots/*/curobo_tmp.yml，将 `${ASSETS_PATH}` 替换成 data_root（不是 data_root/Assets），内容不同则写 curobo.yml，意味着通过 symlink 也会改外部 cache 的生成文件。manifest 包含完整树 revision、git ls-tree 文本 SHA、生成配置列表，不是每个 LFS payload 的最终 SHA 清单。输出 `.roboicl-assets.json` 和 stdout。

### setup/fetch_reference.py

task 必须在固定 reference setting 的22项中；row 特有 revision 优先于全局 source_repository revision。data_root 与 endpoint 可选；源安装位置与 output reference_path 都从锁定配置得到，并设置 ROBOICL_DATA_ROOT。

已有源 SHA 不符保留并失败；无源先下载 `.partial`，已有 partial 就拒绝，不做自动断点续传；下载可校验 size（配置提供时）并必查 SHA，成功才 rename 成 source。失败 partial 留待检查。已有 bundle 调 reference.verify（需要源）并核对 SHA/starts/H；无则 build 锁定 starts，最后输出报告。网络/下载权限/哈希/构建异常均无隐式换源或重采样。

### setup/requirements-policy.txt

模型侧环境：numpy 1.26.0、Pillow 11.3.0、PyYAML 6.0.2、h5py 3.14.0、scipy 1.15.3、opencv-python-headless 4.11.0.86、websockets 15.0.1、msgpack 1.1.1、msgpack-numpy 0.4.8、pydantic 2.11.7、requests >=2.32,<3。requests 是范围而非冻结版本；GPT HTTP 方式不需要机器人 checkpoint，不等于无外部模型服务依赖。

### setup/requirements-sim.txt

仿真基础依赖同上 numpy/Pillow/PyYAML/h5py/scipy/opencv/msgpack/msgpack-numpy/pydantic；websockets **12.0**，另 transforms3d 0.4.2、omegaconf 2.3.0、setuptools 80.9.0、setuptools-scm 8.2.0、toml 0.10.2、wheel 0.45.1、virtualenv 20.30.0。不能与 policy 的 websocket pin 当成同一环境合并安装；IsaacLab/cuRobo 还有各自依赖。

### setup/constraints-sim.txt

面向 Isaac Sim 5.1 的 constraints：torch 2.7.0、torchvision 0.22.0、torchaudio 2.7.0；numpy1.26.0、packaging23.0、typing_extensions4.12.2、filelock3.13.1、websockets12.0、scipy1.15.3、warp-lang1.11.0、click8.1.7、psutil5.9.8、wheel0.45.1、starlette0.45.3、virtualenv20.30.0；Pillow11.3.0/opencv4.11.0.86/PyYAML6.0.2/h5py3.14.0/msgpack1.1.1/msgpack-numpy0.4.8/pydantic2.11.7；onnx1.18.0、ipython8.38.0、stable-baselines3 2.7.0；cuRobo viewer 的 viser0.2.11、plyfile1.1.3、tyro0.9.17、typeguard4.4.2。constraints 文件只是 pip 解算约束，不能单独当已安装证据，也未固定 CUDA/driver/全部传递包；无 HUMANO TensorRT10.13.0.35、ORT1.23.2、SONIC CPU provider 证明。

## 14. 配置公共机制与全量实际值

以下每个 task 文件都由 config.load_task 读入，run.check 提取 harness_overrides/instruction aliases，render_official_task_prompt 把 description/comment/scoring 拼入模型 prompt；不负责创建物体或真正计算 score。它们都使用 documentation version `robodojo-public-2026-09-23`，各自 source URL 在原值中保留。缺 version/source/description 或 malformed scoring 在启动前失败；本地 JSON 声明不证明官网内容仍相同。每个 task 的 H 从 adaptive_horizons 单独合入，默认最大平移0.03m/旋转0.15rad，后文列实际 override。

### configs/shared_harness.json

run.check 的基础字典；profile/task/CLI 可能覆盖。feedback_frames=1 与 include_endpoints 并不代表每物理帧都上模型。三路相机以 triptych 合成后计一张 context image；50 是请求图像上限。live_anchor_count=5 是基础值，发表协议覆盖为12或25。

实际完整配置：

```json
{
  "predict_horizon": 15,
  "execute_horizon": 15,
  "feedback_frames": 1,
  "include_endpoints": true,
  "feedback_cameras": [
    "cam_head",
    "cam_left_wrist",
    "cam_right_wrist"
  ],
  "endpoint_cameras": [],
  "action_space": "ee_delta",
  "delta_frame": "world",
  "max_translation_m": 0.03,
  "max_rotation_rad": 0.15,
  "max_joint_increment_rad": 0.15,
  "ik_continuity_guard": true,
  "max_action_corrections": 2,
  "reasoning_summary": "auto",
  "cache_mode": "implicit",
  "preserve_reasoning": true,
  "visual_history_batches": 0,
  "train_result_observations": true,
  "live_anchor_count": 5,
  "live_retain_latest": true,
  "live_anchor_interval": 210,
  "max_context_images": 50,
  "visual_history_prune_every": 1,
  "max_transport_attempts": 3,
  "demo_message_format": "tool_calls",
  "include_head_calibration": true,
  "fixed_predict_horizon": true,
  "demo_cameras": [
    "cam_head",
    "cam_left_wrist",
    "cam_right_wrist"
  ],
  "image_profile": "native_jpeg",
  "image_detail": "original",
  "demo_frame0_cameras": [],
  "image_layout": "triptych",
  "rolling_history_cache": true,
  "reasoning_effort": "xhigh"
}
```

### configs/adaptive_horizons.json

load_profile 读取全42任务 H，每个等于 predict/execute；25Hz 下H5/10/15/20/25分别对应0.2/0.4/0.6/0.8/1.0秒名义控制时长。不是模型响应壁钟时长。schema/source 在加载器里未做语义认证。

实际完整配置：

```json
{
  "schema": "roboicl.adaptive_horizons.v1",
  "source": "experiment-configs",
  "task_horizons": {
    "align_blocks": 5,
    "arrange_largest_number": 15,
    "build_tower": 15,
    "classify_objects": 15,
    "classify_objects_by_language": 15,
    "cover_blocks": 15,
    "deposit_coin": 5,
    "fasten_screws": 25,
    "fill_egg_holder": 10,
    "fill_pen_holder": 15,
    "fold_clothes": 10,
    "general_pickup": 5,
    "hang_mugs": 15,
    "imitate_sorting_sequence": 20,
    "insert_key": 5,
    "insert_tubes": 10,
    "make_kong": 10,
    "make_toast": 20,
    "match_and_pick_from_conveyor": 10,
    "organize_table": 15,
    "pack_objects_into_box": 20,
    "pick_from_conveyor_by_image": 10,
    "play_Xylophone": 10,
    "play_stacking_toy": 20,
    "play_tic_tac_toe": 15,
    "plug_in_charger": 5,
    "pour_balls_into_vase": 10,
    "pour_by_language": 15,
    "pour_liquid_into_cup": 5,
    "press_by_number": 10,
    "push_T": 10,
    "put_bottles_into_dustbin": 10,
    "solve_equation": 5,
    "sort_nesting_dolls_by_size": 15,
    "stack_blocks": 10,
    "stack_blocks_by_language": 5,
    "stack_bowls": 15,
    "store_laptop_and_headphones": 15,
    "store_tools_in_toolbox": 15,
    "swap_T": 5,
    "swap_blocks": 10,
    "sweep_blocks": 15
  }
}
```

### configs/upstream.lock.json

verify_submodules 消费四个 name/path/commit；URL是来源说明，验证函数不 fetch。integration 说明是固定集成提交的边界。

实际完整配置：

```json
{
  "schema": "roboicl.upstream.v2",
  "packages": [
    {
      "name": "RoboDojo",
      "path": "third_party/RoboDojo",
      "url": "https://github.com/Mosi-AI/RoboDojo.git",
      "commit": "00ef04cd245bfa76e6f1e0bd8383c82f7964f942"
    },
    {
      "name": "XPolicyLab",
      "path": "third_party/XPolicyLab",
      "url": "https://github.com/XPolicyLab/XPolicyLab.git",
      "commit": "fa431ecd893ee706883e64fe5fe1464ec8cd928d"
    },
    {
      "name": "IsaacLab",
      "path": "third_party/IsaacLab",
      "url": "https://github.com/isaac-sim/IsaacLab.git",
      "commit": "afca7b09d60d8beb9c1cb28b43066499940b969b"
    },
    {
      "name": "cuRobo",
      "path": "third_party/curobo",
      "url": "https://github.com/NVlabs/curobo.git",
      "commit": "895c6517243f8cb091c73c018c8167192d39599a"
    }
  ],
  "integration": "Submodules are loaded in place; RoboICL pins the RoboDojo integration commit containing the execution-side IK guard, camera validation, and external asset path adapters."
}
```

### configs/data.lock.json

fetch_assets 与 asset_preflight 读取 assets；references.settings 是清单索引，fetch_reference 本身直接使用固定 setting 路径。

实际完整配置：

```json
{
  "schema": "roboicl.data.v1",
  "assets": {
    "repo_id": "RoboDojo-Benchmark/RoboDojo",
    "revision": "0a709fe7863f6c869a8d141676585a883fce4270",
    "include": "Assets/**"
  },
  "references": {
    "settings": {
      "one_shot_j12_b12": "configs/references/one_shot_j12_b12.json"
    }
  }
}
```

### configs/protocols/one_shot_j12_b12.json

1-shot J12、LIVE B12、TRAIN结果图启用，request_timeout600；模型/endpoint 是配置声明，复现若替换 provider/model 必须单列差异。没有 max_output_tokens 数字上限，null 不是0。

实际完整配置：

```json
{
  "id": "roboicl-1shot-j12-b12-v1",
  "model": "gpt-6-astra-ssvip",
  "api_mode": "responses",
  "endpoint": "https://www.dmxapi.cn/v1/responses",
  "shot_count": 1,
  "train_chunk_count": 12,
  "reference_setting": "configs/references/one_shot_j12_b12.json",
  "harness_overrides": {
    "reasoning_effort": "xhigh",
    "max_output_tokens": null,
    "train_result_observations": true,
    "demo_message_format": "tool_calls",
    "live_anchor_count": 12,
    "live_retain_latest": true,
    "max_context_images": 50,
    "ik_continuity_guard": true,
    "max_action_corrections": 2,
    "max_transport_attempts": 3,
    "request_timeout_s": 600
  }
}
```

### configs/protocols/zero_shot_b25.json

0-shot J0、LIVE B25、TRAIN结果图禁用、zero_shot_only=true；run 实际依据 shot_count约束，不用 zero_shot_only 做额外任务集合筛选。其余对应模型/endpoint/retry/guard同1-shot。

实际完整配置：

```json
{
  "id": "roboicl-0shot-b25-v1",
  "model": "gpt-6-astra-ssvip",
  "api_mode": "responses",
  "endpoint": "https://www.dmxapi.cn/v1/responses",
  "shot_count": 0,
  "train_chunk_count": 0,
  "zero_shot_only": true,
  "harness_overrides": {
    "reasoning_effort": "xhigh",
    "max_output_tokens": null,
    "train_result_observations": false,
    "demo_message_format": "tool_calls",
    "live_anchor_count": 25,
    "live_retain_latest": true,
    "max_context_images": 50,
    "ik_continuity_guard": true,
    "max_action_corrections": 2,
    "max_transport_attempts": 3,
    "request_timeout_s": 600
  }
}
```

### configs/references/one_shot_j12_b12.json

调用方 fetch_reference/load_reference_setting/run.reference_preflight；固定8个zero-shot与22个one-shot，而配置目录有42任务，不能将全部42任务称为论文30任务集合。源repository默认revision为35efbc7dedfdbeeb6e95fb749bd885d73d483e41，fill_pen_holder单独覆盖为0a709fe7863f6c869a8d141676585a883fce4270。Xylophone 下载repository episode_0000001但安装episode_0000000，必须显式记录该身份转换。swap_blocks使用episode1。

LIVE定义为11个固定均匀时间锚点加最新完整交互，共12；TRAIN12块起点是锁定值，不能重新均匀采样冒充同一setting。下表的尾部间隔=total-1-(末start+H)，非零说明末块结果不是最后记录帧。

|任务|类别|H|episode|total|12个锁定starts|尾部间隔|
|---|---|---:|---:|---:|---|---:|
|cover_blocks|Memory|15|0|555|[0, 49, 98, 147, 196, 245, 294, 343, 392, 432, 490, 539]|0|
|imitate_sorting_sequence|Memory|20|0|1187|[0, 106, 212, 318, 424, 530, 636, 742, 848, 954, 1054, 1166]|0|
|match_and_pick_from_conveyor|Memory|10|0|474|[0, 42, 84, 126, 168, 210, 253, 295, 337, 379, 421, 463]|0|
|press_by_number|Memory|10|0|377|[0, 33, 67, 100, 133, 166, 200, 233, 266, 294, 333, 366]|0|
|swap_T|Memory|5|0|272|[0, 24, 48, 73, 97, 121, 145, 169, 193, 218, 242, 266]|0|
|swap_blocks|Memory|10|1|439|[0, 38, 77, 115, 154, 192, 231, 269, 308, 346, 385, 423]|5|
|build_tower|Precision|15|0|723|[0, 64, 129, 193, 257, 321, 386, 450, 514, 584, 643, 707]|0|
|deposit_coin|Precision|5|0|191|[0, 17, 34, 50, 67, 84, 101, 118, 135, 151, 168, 185]|0|
|fasten_screws|Precision|25|0|1510|[0, 135, 270, 399, 540, 675, 809, 933, 1079, 1225, 1349, 1484]|0|
|insert_key|Precision|5|0|175|[0, 15, 31, 48, 60, 77, 92, 108, 123, 138, 154, 169]|0|
|insert_tubes|Precision|10|0|319|[0, 28, 56, 84, 112, 140, 168, 196, 224, 252, 280, 308]|0|
|play_Xylophone|Precision|10|0|306|[0, 27, 54, 80, 107, 134, 161, 188, 215, 241, 268, 295]|0|
|plug_in_charger|Precision|5|0|195|[0, 17, 34, 52, 69, 86, 103, 120, 137, 155, 172, 189]|0|
|pour_balls_into_vase|Precision|10|0|355|[0, 31, 63, 94, 125, 156, 188, 219, 250, 281, 313, 344]|0|
|classify_objects|Long-Horizon|15|0|747|[0, 66, 133, 199, 266, 332, 399, 465, 532, 598, 665, 731]|0|
|fill_egg_holder|Long-Horizon|10|0|488|[0, 43, 87, 130, 173, 217, 260, 304, 347, 390, 434, 477]|0|
|fill_pen_holder|Long-Horizon|15|0|934|[0, 83, 167, 250, 334, 417, 501, 584, 668, 751, 835, 918]|0|
|make_kong|Long-Horizon|10|0|382|[0, 34, 67, 101, 135, 169, 202, 236, 270, 304, 337, 371]|0|
|organize_table|Long-Horizon|15|0|838|[0, 75, 149, 224, 299, 374, 448, 523, 598, 673, 747, 822]|0|
|play_stacking_toy|Long-Horizon|20|0|887|[0, 77, 154, 231, 308, 385, 463, 530, 617, 694, 784, 848]|18|
|play_tic_tac_toe|Long-Horizon|15|0|883|[0, 72, 155, 232, 310, 387, 474, 542, 620, 697, 775, 852]|15|
|put_bottles_into_dustbin|Long-Horizon|10|0|362|[0, 32, 64, 96, 128, 160, 191, 223, 255, 287, 319, 351]|0|

该文件完整来源字段（包括每条SHA、size、路径，不以名称代替值）：

```json
{
  "schema": "roboicl.reference_setting.v1",
  "id": "one_shot_j12_b12_paper_2026-09-29",
  "paper_protocol": "RoboICL arXiv:2609.34261, 30-task RoboDojo comparison",
  "train": {
    "shots": 1,
    "blocks_per_demonstration": 12,
    "selector": "The starts below are authoritative. For new settings, use the deterministic selector implemented in roboicl.reference.select_starts.",
    "implementation": "roboicl.reference.select_starts"
  },
  "live": {
    "retained_blocks": 12,
    "selector": "Eleven fixed, uniformly spaced time anchors retain the first complete interaction crossing each anchor; the twelfth slot retains the latest complete interaction.",
    "implementation": "roboicl.policy.trajectory_memory.LiveTrajectoryMemory"
  },
  "source_repository": {
    "repo_id": "RoboDojo-Benchmark/RoboDojo",
    "revision": "35efbc7dedfdbeeb6e95fb749bd885d73d483e41",
    "note": "The SHA256 in each row is authoritative; repository_path records the source identity used for retrieval."
  },
  "zero_shot_tasks": [
    {
      "task": "align_blocks",
      "category": "Open"
    },
    {
      "task": "classify_objects_by_language",
      "category": "Open"
    },
    {
      "task": "general_pickup",
      "category": "Open"
    },
    {
      "task": "pick_from_conveyor_by_image",
      "category": "Open"
    },
    {
      "task": "pour_by_language",
      "category": "Open"
    },
    {
      "task": "solve_equation",
      "category": "Open"
    },
    {
      "task": "stack_blocks_by_language",
      "category": "Open"
    },
    {
      "task": "store_tools_in_toolbox",
      "category": "Open"
    }
  ],
  "one_shot_references": [
    {
      "task": "cover_blocks",
      "category": "Memory",
      "reference_horizon": 15,
      "selected_action_starts": [
        0,
        49,
        98,
        147,
        196,
        245,
        294,
        343,
        392,
        432,
        490,
        539
      ],
      "source": {
        "install_path": "runtime-data/cover_blocks/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/cover_blocks/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "2fa920f75e3d126c811ea3eb0145325ce504d3d4b2ba08488a61531576bad74a",
        "total_frames": 555
      }
    },
    {
      "task": "imitate_sorting_sequence",
      "category": "Memory",
      "reference_horizon": 20,
      "selected_action_starts": [
        0,
        106,
        212,
        318,
        424,
        530,
        636,
        742,
        848,
        954,
        1054,
        1166
      ],
      "source": {
        "install_path": "runtime-data/imitate_sorting_sequence/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/imitate_sorting_sequence/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "9bfff493434772f85765229567a2f45ddd38e1cc9e98034c57a8b65354f62375",
        "size": 1298699086,
        "total_frames": 1187
      }
    },
    {
      "task": "match_and_pick_from_conveyor",
      "category": "Memory",
      "reference_horizon": 10,
      "selected_action_starts": [
        0,
        42,
        84,
        126,
        168,
        210,
        253,
        295,
        337,
        379,
        421,
        463
      ],
      "source": {
        "install_path": "runtime-data/match_and_pick_from_conveyor/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/match_and_pick_from_conveyor/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "a6b70eab16746a8ebcef310fc1ecdf61d4191975ab826aa3addd707de3389bd7",
        "total_frames": 474
      }
    },
    {
      "task": "press_by_number",
      "category": "Memory",
      "reference_horizon": 10,
      "selected_action_starts": [
        0,
        33,
        67,
        100,
        133,
        166,
        200,
        233,
        266,
        294,
        333,
        366
      ],
      "source": {
        "install_path": "runtime-data/press_by_number/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/press_by_number/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "2c96ad9ab5ad586f71ca62fed1a4b1f4a3c29fab61307891f7121ac2c4c977b0",
        "total_frames": 377
      }
    },
    {
      "task": "swap_T",
      "category": "Memory",
      "reference_horizon": 5,
      "selected_action_starts": [
        0,
        24,
        48,
        73,
        97,
        121,
        145,
        169,
        193,
        218,
        242,
        266
      ],
      "source": {
        "install_path": "runtime-data/swap_T/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/swap_T/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "8b526ba12e970161bfd5ff758164ff184fb08e0a69d06461c5ad8d93396d1e10",
        "total_frames": 272
      }
    },
    {
      "task": "swap_blocks",
      "category": "Memory",
      "reference_horizon": 10,
      "selected_action_starts": [
        0,
        38,
        77,
        115,
        154,
        192,
        231,
        269,
        308,
        346,
        385,
        423
      ],
      "source": {
        "install_path": "runtime-data/swap_blocks/train/episode_0000001.hdf5",
        "repository_path": "data/RoboDojo/swap_blocks/arx_x5/data/episode_0000001.hdf5",
        "episode": 1,
        "sha256": "025df44e23a377dcb67d758c915421986017c7750ac320f026bd935988d283f3",
        "size": 414317508,
        "total_frames": 439
      }
    },
    {
      "task": "build_tower",
      "category": "Precision",
      "reference_horizon": 15,
      "selected_action_starts": [
        0,
        64,
        129,
        193,
        257,
        321,
        386,
        450,
        514,
        584,
        643,
        707
      ],
      "source": {
        "install_path": "runtime-data/build_tower/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/build_tower/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "d7c3fdf94eceb545f313999716ec0b25e43b6571ff2990aa0986376204e38a12",
        "size": 198230849,
        "total_frames": 723
      }
    },
    {
      "task": "deposit_coin",
      "category": "Precision",
      "reference_horizon": 5,
      "selected_action_starts": [
        0,
        17,
        34,
        50,
        67,
        84,
        101,
        118,
        135,
        151,
        168,
        185
      ],
      "source": {
        "install_path": "runtime-data/deposit_coin/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/deposit_coin/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "b9093beb1142f1e15ad38003a82abdc85f96417425193b0b5c7d9128ae90cd29",
        "size": 172213297,
        "total_frames": 191
      }
    },
    {
      "task": "fasten_screws",
      "category": "Precision",
      "reference_horizon": 25,
      "selected_action_starts": [
        0,
        135,
        270,
        399,
        540,
        675,
        809,
        933,
        1079,
        1225,
        1349,
        1484
      ],
      "source": {
        "install_path": "runtime-data/fasten_screws/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/fasten_screws/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "80fb9ddbfa75d72948148c63a88d8cbddbfa127ff50dceb658849b89b1ebec37",
        "size": 1471383992,
        "total_frames": 1510
      }
    },
    {
      "task": "insert_key",
      "category": "Precision",
      "reference_horizon": 5,
      "selected_action_starts": [
        0,
        15,
        31,
        48,
        60,
        77,
        92,
        108,
        123,
        138,
        154,
        169
      ],
      "source": {
        "install_path": "runtime-data/insert_key/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/insert_key/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "56226333754583c0b86cfb4466e1a5e9a1927d4e8fac2df5028b6e7268183a02",
        "size": 155048675,
        "total_frames": 175
      }
    },
    {
      "task": "insert_tubes",
      "category": "Precision",
      "reference_horizon": 10,
      "selected_action_starts": [
        0,
        28,
        56,
        84,
        112,
        140,
        168,
        196,
        224,
        252,
        280,
        308
      ],
      "source": {
        "install_path": "runtime-data/insert_tubes/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/insert_tubes/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "a1c9d0dee5eedc6d56e5f88d98642ed4ab50b8f6315255cc8fd8492739256758",
        "size": 286528607,
        "total_frames": 319
      }
    },
    {
      "task": "play_Xylophone",
      "category": "Precision",
      "reference_horizon": 10,
      "selected_action_starts": [
        0,
        27,
        54,
        80,
        107,
        134,
        161,
        188,
        215,
        241,
        268,
        295
      ],
      "source": {
        "install_path": "runtime-data/play_Xylophone/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/play_Xylophone/arx_x5/data/episode_0000001.hdf5",
        "episode": 0,
        "sha256": "de29ee57e57e6ff391a5b5c1f4b7250e4c2c96f5d8abf1579280bdca075b9ea1",
        "size": 334968193,
        "total_frames": 306
      }
    },
    {
      "task": "plug_in_charger",
      "category": "Precision",
      "reference_horizon": 5,
      "selected_action_starts": [
        0,
        17,
        34,
        52,
        69,
        86,
        103,
        120,
        137,
        155,
        172,
        189
      ],
      "source": {
        "install_path": "runtime-data/plug_in_charger/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/plug_in_charger/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "7f16181b1697621037c2ca82c10e8506ec4c0190a615d605e3069cf41d954a6a",
        "size": 176790008,
        "total_frames": 195
      }
    },
    {
      "task": "pour_balls_into_vase",
      "category": "Precision",
      "reference_horizon": 10,
      "selected_action_starts": [
        0,
        31,
        63,
        94,
        125,
        156,
        188,
        219,
        250,
        281,
        313,
        344
      ],
      "source": {
        "install_path": "runtime-data/pour_balls_into_vase/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/pour_balls_into_vase/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "def19279bdc74357f827cd0722542b75f8fa2d778543c5acbe3d612b6060c366",
        "size": 317006855,
        "total_frames": 355
      }
    },
    {
      "task": "classify_objects",
      "category": "Long-Horizon",
      "reference_horizon": 15,
      "selected_action_starts": [
        0,
        66,
        133,
        199,
        266,
        332,
        399,
        465,
        532,
        598,
        665,
        731
      ],
      "source": {
        "install_path": "runtime-data/classify_objects/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/classify_objects/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "90c6e0651bf87358127aa431ceef08a70b63bf6fa2dc7fc77962cfeb4ea19555",
        "size": 675900439,
        "total_frames": 747
      }
    },
    {
      "task": "fill_egg_holder",
      "category": "Long-Horizon",
      "reference_horizon": 10,
      "selected_action_starts": [
        0,
        43,
        87,
        130,
        173,
        217,
        260,
        304,
        347,
        390,
        434,
        477
      ],
      "source": {
        "install_path": "runtime-data/fill_egg_holder/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/fill_egg_holder/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "9cab4b8e5d505cd10aeee80df7a9cda564bb8a6646598a5cdee35243e6355d35",
        "total_frames": 488
      }
    },
    {
      "task": "fill_pen_holder",
      "category": "Long-Horizon",
      "reference_horizon": 15,
      "selected_action_starts": [
        0,
        83,
        167,
        250,
        334,
        417,
        501,
        584,
        668,
        751,
        835,
        918
      ],
      "source": {
        "install_path": "runtime-data/fill_pen_holder/train/episode_0000000.hdf5",
        "repository_revision": "0a709fe7863f6c869a8d141676585a883fce4270",
        "repository_path": "data/RoboDojo/fill_pen_holder/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "0d33a6cc7258945b60eff9ad4186570c4f4488e2f55bd73ec0df4f8f67494542",
        "size": 245324592,
        "total_frames": 934
      }
    },
    {
      "task": "make_kong",
      "category": "Long-Horizon",
      "reference_horizon": 10,
      "selected_action_starts": [
        0,
        34,
        67,
        101,
        135,
        169,
        202,
        236,
        270,
        304,
        337,
        371
      ],
      "source": {
        "install_path": "runtime-data/make_kong/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/make_kong/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "cf87c6076a8400d7fd308155d423ca1e7ba7793189405aeaabae9e6e24847805",
        "size": 393301017,
        "total_frames": 382
      }
    },
    {
      "task": "organize_table",
      "category": "Long-Horizon",
      "reference_horizon": 15,
      "selected_action_starts": [
        0,
        75,
        149,
        224,
        299,
        374,
        448,
        523,
        598,
        673,
        747,
        822
      ],
      "source": {
        "install_path": "runtime-data/organize_table/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/organize_table/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "f35041b55aa076456c60da4d217b06c89c700add43ed4804b3b38a210f021145",
        "size": 778413312,
        "total_frames": 838
      }
    },
    {
      "task": "play_stacking_toy",
      "category": "Long-Horizon",
      "reference_horizon": 20,
      "selected_action_starts": [
        0,
        77,
        154,
        231,
        308,
        385,
        463,
        530,
        617,
        694,
        784,
        848
      ],
      "source": {
        "install_path": "runtime-data/play_stacking_toy/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/play_stacking_toy/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "c58d9e3deeb16924f0a7f1ac89d4da8fc93ab2961b494a2a1d08b9f916704298",
        "size": 907635605,
        "total_frames": 887
      }
    },
    {
      "task": "play_tic_tac_toe",
      "category": "Long-Horizon",
      "reference_horizon": 15,
      "selected_action_starts": [
        0,
        72,
        155,
        232,
        310,
        387,
        474,
        542,
        620,
        697,
        775,
        852
      ],
      "source": {
        "install_path": "runtime-data/play_tic_tac_toe/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/play_tic_tac_toe/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "324f00cef4db1548e755afdc5977121cd543ecebaf3cc2cdca4e4bc1f1e671f8",
        "size": 910524041,
        "total_frames": 883
      }
    },
    {
      "task": "put_bottles_into_dustbin",
      "category": "Long-Horizon",
      "reference_horizon": 10,
      "selected_action_starts": [
        0,
        32,
        64,
        96,
        128,
        160,
        191,
        223,
        255,
        287,
        319,
        351
      ],
      "source": {
        "install_path": "runtime-data/put_bottles_into_dustbin/train/episode_0000000.hdf5",
        "repository_path": "data/RoboDojo/put_bottles_into_dustbin/arx_x5/data/episode_0000000.hdf5",
        "episode": 0,
        "sha256": "e110130f137a2006939c47707ceee2cac489658436823555cce560545d2086b0",
        "size": 320652074,
        "total_frames": 362
      }
    }
  ]
}
```

### configs/tasks/align_blocks.json

三角尺推三块成直线，禁止抬高，必须回原位；不能用抓取堆放替代推齐。

合并后的 H=5，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/align-blocks/",
    "description": "There is a set square and three blocks. The robot needs to use the set square to push the blocks until they are aligned in parallel in a straight row.",
    "scoring": [
      {
        "score": "0",
        "condition": "The alignment, reset, or no-lift condition is not satisfied."
      },
      {
        "score": "100",
        "condition": "The three cubes are in one straight aligned row, the robot returns to origin, and no cube has been lifted above the allowed height during the episode."
      }
    ]
  }
}
```

### configs/tasks/arrange_largest_number.json

按最大数顺序摆数字，保留姿态容差；评分分别考虑四位和五位布局，描述只说四位，需以 evaluator 实际布局核实。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/arrange-largest-number/",
    "description": "There are several number tiles on the table and a pad for placement. The robot needs to determine the order that forms the largest possible number, then place the four numbers on the pad from left to right in that order.",
    "comment": "Each digit must remain within the permitted rotational tolerance relative to its initial pose.",
    "scoring": [
      {
        "score": "0",
        "condition": "No digit is placed correctly."
      },
      {
        "score": "5",
        "condition": "Any one digit is placed on its correct pad in descending-number order, with valid orientation and the gripper open."
      },
      {
        "score": "15",
        "condition": "Any two digits are correctly placed."
      },
      {
        "score": "25",
        "condition": "In a 5-digit layout, any three digits are correctly placed."
      },
      {
        "score": "30",
        "condition": "In a 4-digit layout, any three digits are correctly placed."
      },
      {
        "score": "40",
        "condition": "In a 5-digit layout, any four digits are correctly placed."
      },
      {
        "score": "100",
        "condition": "All digits are correctly ordered on the pads, have valid orientation, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/build_tower.json

板/支撑分层居中且全部直立，八件构成完整塔；放宽平移至0.031m。

合并后的 H=15，每步最大平移=0.031m、最大旋转=0.15rad；task override={"max_translation_m": 0.031}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "harness_overrides": {
    "max_translation_m": 0.031
  },
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/build-tower/",
    "description": "There are wooden blocks and wooden boards on the table. The robot needs to place each board or block on the supports below it, keep each layer centered over the layer beneath it, and keep all pieces upright.",
    "scoring": [
      {
        "score": "0",
        "condition": "No tower layer is completed."
      },
      {
        "score": "10",
        "condition": "The lower board is placed across two upright supports, remains stable and centered on them, and the gripper is open."
      },
      {
        "score": "30",
        "condition": "The middle board is placed on the completed lower layer, stays centered over it, and all lower and middle pieces remain upright."
      },
      {
        "score": "100",
        "condition": "The full tower is completed with the top pieces, all eight pieces remain upright, each layer stays centered over the layer below it, the gripper is open, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/classify_objects.json

三类物件分入任意对应篮，每篮只容纳一完整类；平移放宽至0.034m。

合并后的 H=15，每步最大平移=0.034m、最大旋转=0.15rad；task override={"max_translation_m": 0.034}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "harness_overrides": {
    "max_translation_m": 0.034
  },
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/classify-objects/",
    "description": "There are three categories of objects and three baskets. The robot needs to group the objects by category and place each category into a separate basket. Any basket can be used for any category, as long as objects of the same category are placed together.",
    "scoring": [
      {
        "score": "0",
        "condition": "No category basket is completed."
      },
      {
        "score": "15",
        "condition": "One basket contains exactly one complete object category, with no objects from other categories, and the gripper is open."
      },
      {
        "score": "40",
        "condition": "Two baskets each contain exactly one complete category."
      },
      {
        "score": "100",
        "condition": "All three categories are separated into three baskets, the objects are settled inside the baskets, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/classify_objects_by_language.json

根据语言固定左中右篮对应类别，不能沿用任意篮映射。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/classify-objects-by-language/",
    "description": "There are three baskets and three categories of unseen objects. The robot needs to understand the language instruction, identify the category of each object, and place the objects into the specified baskets from left to right.",
    "scoring": [
      {
        "score": "0",
        "condition": "No specified category is placed into its assigned basket."
      },
      {
        "score": "10",
        "condition": "One specified category is placed into its assigned basket, no other objects are in that basket, and the gripper is open."
      },
      {
        "score": "40",
        "condition": "Two specified categories are placed into their assigned baskets, with no extra objects in those baskets."
      },
      {
        "score": "100",
        "condition": "All three specified categories are placed into the left, middle, and right baskets as instructed, no basket contains objects from another category, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/cover_blocks.json

先按从左到右盖住，再按红绿蓝顺序揭开；终态照片不足以证明顺序。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/cover-blocks/",
    "description": "There are three covers and three blocks arranged in a random order. The blocks are red, green, and blue. The robot needs to cover the blocks from left to right, remember the color under each cover, and then uncover the blocks in the order of red, green, and blue.",
    "scoring": [
      {
        "score": "0",
        "condition": "No block-covering step is completed."
      },
      {
        "score": "5",
        "condition": "All three blocks are covered by cups from left to right."
      },
      {
        "score": "15",
        "condition": "After all blocks are covered, the red block is uncovered first while the green and blue blocks remain covered."
      },
      {
        "score": "30",
        "condition": "The red and green blocks are uncovered in order while the blue block remains covered."
      },
      {
        "score": "100",
        "condition": "All three blocks are first covered from left to right, then uncovered in red, green, and blue order; cup orientation remains valid and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/deposit_coin.json

硬币先抬至少8cm再准确插槽；旋转上限0.21rad。

合并后的 H=5，每步最大平移=0.03m、最大旋转=0.21rad；task override={"max_rotation_rad": 0.21}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "harness_overrides": {
    "max_rotation_rad": 0.21
  },
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/deposit-coin/",
    "description": "There is a coin placed on a holder and a coin bank. The robot needs to pick up the coin and accurately insert it into the slot of the coin bank.",
    "scoring": [
      {
        "score": "0",
        "condition": "The coin is not lifted."
      },
      {
        "score": "20",
        "condition": "The coin is lifted at least 8 cm."
      },
      {
        "score": "100",
        "condition": "The coin bounding box is inside the coin bank slot region, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/fasten_screws.json

三色匹配螺母螺栓，检查直立/对齐/旋入深度；旋转0.18rad，H25。

合并后的 H=25，每步最大平移=0.03m、最大旋转=0.18rad；task override={"max_rotation_rad": 0.18}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "harness_overrides": {
    "max_rotation_rad": 0.18
  },
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/fasten-screws/",
    "description": "There are three screws and three nuts on the table. The screws and nuts come from five possible colors: red, blue, gray, yellow, and purple. In each episode, three different colors are selected, and the robot needs to match each screw with the nut of the same color, insert it, and tighten it. The screw positions vary within a small range, while the nut positions vary within a larger range. If a handover is needed, the robot can first place the screw in the middle and then let the other arm pick it up and fasten it.",
    "scoring": [
      {
        "score": "0",
        "condition": "No nut is fastened onto its matching bolt."
      },
      {
        "score": "20",
        "condition": "Any one nut is upright, aligned with its matching bolt, rotated onto the bolt to the required depth, and the gripper is open."
      },
      {
        "score": "50",
        "condition": "Any two matching nut-bolt pairs are fastened correctly."
      },
      {
        "score": "100",
        "condition": "All three nut-bolt pairs are fastened, the gripper is open, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/fill_egg_holder.json

四蛋入盒后关盖；四蛋入盒但盖未关是90而不是100。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/fill-egg-holder/",
    "description": "There is a woven basket containing four eggs and an egg holder. The robot needs to pick up the eggs one by one, place all four into the egg holder, and then close the lid.",
    "scoring": [
      {
        "score": "0",
        "condition": "No egg is placed into the egg holder."
      },
      {
        "score": "10",
        "condition": "Any one egg is inside the egg holder and close to the holder bottom, with the gripper open."
      },
      {
        "score": "25",
        "condition": "Any two eggs are placed correctly into the egg holder."
      },
      {
        "score": "40",
        "condition": "Any three eggs are placed correctly into the egg holder."
      },
      {
        "score": "90",
        "condition": "All four eggs are placed in the holder, but the lid is not fully closed yet."
      },
      {
        "score": "100",
        "condition": "All four eggs are in the holder, the holder lid is correctly and fully closed, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/fill_pen_holder.json

一手扶笔筒另一手逐笔插入，四支笔尖朝上且筒直立；全部插好未回原位90。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/fill-pen-holder/",
    "description": "There is a pen holder and several pens. The robot needs to grasp the pen holder with one hand, use the other hand to place the pens into the holder one by one, and finally put the filled pen holder back on the table.",
    "scoring": [
      {
        "score": "0",
        "condition": "No pen is correctly inserted into the pen holder."
      },
      {
        "score": "10",
        "condition": "Any one pen is correctly inserted tip-up into the upright pen holder."
      },
      {
        "score": "25",
        "condition": "Any two pens are correctly inserted into the pen holder."
      },
      {
        "score": "40",
        "condition": "Any three pens are correctly inserted into the pen holder."
      },
      {
        "score": "90",
        "condition": "All four pens are inserted into the upright pen holder, but the robot has not returned to origin yet."
      },
      {
        "score": "100",
        "condition": "All four pens are inserted, the pen holder is upright, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/fold_clothes.json

双袖内折与下摆/肩线对齐，涉及柔性物；夹爪松开时检查袖条件。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/fold-clothes/",
    "description": "There is a piece of clothing. The robot needs to fold it neatly.",
    "scoring": [
      {
        "score": "0",
        "condition": "The sleeve-fold condition is not reached."
      },
      {
        "score": "20",
        "condition": "When the gripper opens, both sleeves are folded inward close to the opposite chest points."
      },
      {
        "score": "100",
        "condition": "Both sleeves are folded inward, the hems are close to the shoulder points in x/y, the hem line is aligned with the shoulder line within the angle threshold, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/general_pickup.json

语言选目标，目标抬至少10cm即可100，没有该文本中的回原位要求。

合并后的 H=5，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/general-pickup/",
    "description": "There are multiple objects. The robot needs to understand the language instruction, identify the target object, and pick it up.",
    "scoring": [
      {
        "score": "0",
        "condition": "The target object is not lifted high enough."
      },
      {
        "score": "100",
        "condition": "The target object is lifted at least 10 cm."
      }
    ]
  }
}
```

### configs/tasks/hang_mugs.json

三杯挂架，逐个有15/40分，全部挂好回原位100。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/hang-mugs/",
    "description": "There are three mugs and one mug rack. The robot needs to pick up each mug and hang all of them on the rack.",
    "scoring": [
      {
        "score": "0",
        "condition": "No mug is hung on the mug rack."
      },
      {
        "score": "15",
        "condition": "Any one mug is hung on the mug rack and the gripper is open."
      },
      {
        "score": "40",
        "condition": "Any two mugs are hung on the mug rack."
      },
      {
        "score": "100",
        "condition": "All three mugs are hung on the rack and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/imitate_sorting_sequence.json

观察对面五类物体入篮顺序，禁止对手机械臂结束前先动；需要时序记忆。

合并后的 H=20，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/imitate-sorting-sequence/",
    "description": "There are five categories of objects, with five objects on each side. The opposite robot places its objects into the basket on the right side in a certain order. The robot needs to observe and remember this sequence, then place its corresponding objects into the basket in the same order. This is a memory-based imitation task.",
    "scoring": [
      {
        "score": "0",
        "condition": "The first sequence step is not completed, or the policy robot moves before the opposite robot arm finishes."
      },
      {
        "score": "5",
        "condition": "The first target object is placed into the policy basket, later target objects are still outside it, all demonstration objects remain in the demo basket, and the gripper is open."
      },
      {
        "score": "15",
        "condition": "The first two target objects are placed in the correct sequence."
      },
      {
        "score": "30",
        "condition": "The first three target objects are placed in the correct sequence."
      },
      {
        "score": "50",
        "condition": "The first four target objects are placed in the correct sequence."
      },
      {
        "score": "100",
        "condition": "All five target objects are placed in sequence, all demonstration objects remain in the demo basket, the gripper is open, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/insert_key.json

钥匙抬至少5cm得15，插深/对齐/直立后转动得100；旋转0.16rad。

合并后的 H=5，每步最大平移=0.03m、最大旋转=0.16rad；task override={"max_rotation_rad": 0.16}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "harness_overrides": {
    "max_rotation_rad": 0.16
  },
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/insert-key/",
    "description": "There is a key and a keyhole on the table. The robot needs to pick up the key, hand it over to the other hand for pose adjustment, insert it accurately into the keyhole, and then turn it.",
    "scoring": [
      {
        "score": "0",
        "condition": "The key is not lifted."
      },
      {
        "score": "15",
        "condition": "The key is lifted at least 5 cm."
      },
      {
        "score": "100",
        "condition": "The key is inserted deep enough into the slot, aligned closely in x/y, kept upright, and successfully turned after insertion."
      }
    ]
  }
}
```

### configs/tasks/insert_tubes.json

三管依次插架，检查深度/直立/松手；单管20、双管40。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/insert-tubes/",
    "description": "There is a tube rack and three tubes. The robot needs to pick up each tube in sequence and insert all tubes into the rack.",
    "scoring": [
      {
        "score": "0",
        "condition": "No tube is inserted into a slot."
      },
      {
        "score": "20",
        "condition": "Any one tube is inside the slot, inserted to the required depth, upright, and the gripper is open."
      },
      {
        "score": "40",
        "condition": "Any two tubes are inserted correctly."
      },
      {
        "score": "100",
        "condition": "All three tubes are inserted correctly and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/make_kong.json

观察弃牌，三匹配牌推倒、其他牌直立，抓放目标且无非法动作；不是简单整理麻将。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/make-kong/",
    "description": "There is a Mahjong setup. The opponent first pushes out a tile. The robot needs to observe the discarded tile, identify the matching tiles on its side, and perform a valid kong action. The setup guarantees that a kong is possible.",
    "scoring": [
      {
        "score": "0",
        "condition": "The kong declaration state is not completed."
      },
      {
        "score": "100",
        "condition": "The three matching Mahjong tiles are pushed down, the non-matching tiles remain upright on the table, the target tile is correctly grasped and placed at the required position, the gripper is open, and no invalid robot motion is detected."
      }
    ]
  }
}
```

### configs/tasks/make_toast.json

两片入烤面包机并压杆；满分还要求架上恰余两片。

合并后的 H=20，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/make-toast/",
    "description": "There is a basket containing multiple slices of bread and a toaster. The robot needs to pick up two slices one by one, place them into the toaster, and then press the lever down to start toasting.",
    "scoring": [
      {
        "score": "0",
        "condition": "No bread slice is inserted into a toaster slot."
      },
      {
        "score": "25",
        "condition": "One bread slice is inserted into a toaster slot and all bread axes satisfy the required orientation."
      },
      {
        "score": "50",
        "condition": "Two bread slices are inserted into the toaster slots and all bread axes satisfy the required orientation."
      },
      {
        "score": "100",
        "condition": "Both toaster slots are filled, exactly two bread slices remain on the shelf, the toaster control is pressed down, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/match_and_pick_from_conveyor.json

记忆最先经过的对象，后续拾起匹配物至少10cm；不是图板条件任务。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/match-and-pick-from-conveyor/",
    "description": "An object first appears on the conveyor and is carried away. The robot needs to remember this object, observe the following objects on the conveyor, and pick the one that matches the first object.",
    "scoring": [
      {
        "score": "0",
        "condition": "The matching target object is not lifted high enough."
      },
      {
        "score": "100",
        "condition": "The matching target object is lifted at least 10 cm."
      }
    ]
  }
}
```

### configs/tasks/organize_table.json

描述含抽屉杂物，但配置评分只计鼠标/键盘/玩偶/闹钟四项；不能从描述反推出实际 evaluator。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/organize-table/",
    "description": "There are a computer, a keyboard, a mouse, an alarm clock, a cartoon figurine, three miscellaneous items, and a drawer. The robot needs to organize the table by placing the mouse on the mouse pad, pushing the keyboard into the frame, putting the figurine on the stand, placing the alarm clock on top of the drawer, opening the drawer, and putting all remaining miscellaneous items into it.",
    "scoring": [
      {
        "score": "0",
        "condition": "No target item is organized correctly."
      },
      {
        "score": "25",
        "condition": "Any one target item is organized correctly: mouse on mouse pad, keyboard in the frame, figurine on its stand, or alarm clock on the drawer, with the gripper open."
      },
      {
        "score": "50",
        "condition": "Any two target items are organized correctly."
      },
      {
        "score": "75",
        "condition": "Any three target items are organized correctly."
      },
      {
        "score": "100",
        "condition": "All four target items are organized correctly and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/pack_objects_into_box.json

四件入对齐盒、正面朝左；位置正确但朝向错不能视为完成。

合并后的 H=20，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/pack-objects-into-box/",
    "description": "There are several objects on the table and a box. The robot needs to pick up all the objects, place them into the box, and ensure that each object is oriented with its front side facing left.",
    "scoring": [
      {
        "score": "0",
        "condition": "No object is placed correctly into the box."
      },
      {
        "score": "10",
        "condition": "Any one object is inside the box with its front side facing left, the box is aligned, and the gripper is open."
      },
      {
        "score": "25",
        "condition": "Any two objects are placed correctly into the box."
      },
      {
        "score": "50",
        "condition": "Any three objects are placed correctly into the box."
      },
      {
        "score": "100",
        "condition": "All four objects are placed in the aligned box with correct facing direction, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/pick_from_conveyor_by_image.json

读图板指定目标，篮子抬超过8cm并目标入篮；不能只拿起传送带物体。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/pick-from-conveyor-by-image/",
    "description": "There is a board displaying an image of the target object, a basket, and a conveyor carrying multiple objects. The robot needs to first lift the basket more than 8 cm, identify the target object on the conveyor based on the image shown on the board, pick up the target object, and place it into the basket.",
    "scoring": [
      {
        "score": "0",
        "condition": "The target object is not placed into the lifted basket."
      },
      {
        "score": "100",
        "condition": "The image-specified target object is in the basket and lifted at least 8 cm."
      }
    ]
  }
}
```

### configs/tasks/play_Xylophone.json

槌头按从左到右敲全部键，每次敲完至少抬0.025m；需要事件序列。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/play-xylophone/",
    "description": "There is a xylophone and a mallet. The robot needs to pick up the mallet with one hand and strike all the xylophone keys from left to right.",
    "scoring": [
      {
        "score": "0",
        "condition": "The full left-to-right strike sequence is not completed."
      },
      {
        "score": "100",
        "condition": "The mallet tip hits all xylophone keys from left to right, and after each hit the mallet is lifted by at least 0.025 m."
      }
    ]
  }
}
```

### configs/tasks/play_stacking_toy.json

四种件数量4/3/2/1对应四柱；按完整柱组计10/30/60/100。

合并后的 H=20，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/play-stacking-toy/",
    "description": "There is a stacking toy with four pegs and four types of pieces. The numbers of pieces in the four types are 4, 3, 2, and 1, and each peg matches one type. The robot needs to place all pieces onto their corresponding pegs correctly.",
    "scoring": [
      {
        "score": "0",
        "condition": "No peg group is completed."
      },
      {
        "score": "10",
        "condition": "One peg group is completed with its pieces aligned on the correct peg and inserted into the base, with the gripper open."
      },
      {
        "score": "30",
        "condition": "Two peg groups are completed."
      },
      {
        "score": "60",
        "condition": "Three peg groups are completed."
      },
      {
        "score": "100",
        "condition": "All four peg groups are completed and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/play_tic_tac_toe.json

轮流放五子，不能对手机械臂运动时动；评分按放置数量，不以赢棋为100条件。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/play-tic-tac-toe/",
    "description": "There is a 3-by-3 tic-tac-toe board. The robot plays as the first player, while the opponent follows a random strategy. The robot and the opponent take turns placing their marks until the board is filled.",
    "scoring": [
      {
        "score": "0",
        "condition": "No player piece reaches a board cell, or the policy robot moves while the opponent robot arm is moving."
      },
      {
        "score": "10",
        "condition": "Any one player piece is placed on a board cell and the gripper is open."
      },
      {
        "score": "30",
        "condition": "Any two player pieces are placed on board cells."
      },
      {
        "score": "50",
        "condition": "Any three player pieces are placed on board cells."
      },
      {
        "score": "75",
        "condition": "Any four player pieces are placed on board cells."
      },
      {
        "score": "100",
        "condition": "All five player pieces are on board cells at the correct height and the gripper is open."
      }
    ]
  }
}
```

### configs/tasks/plug_in_charger.json

插头位置、深度、直立与回原位同时要求，二元0/100。

合并后的 H=5，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/plug-in-charger/",
    "description": "There is a charger plug and a power strip. The robot needs to pick up the charger plug and insert it into the power strip.",
    "scoring": [
      {
        "score": "0",
        "condition": "The charger is not fully plugged into the socket."
      },
      {
        "score": "100",
        "condition": "The charger is inside the socket, inserted to the required depth, upright, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/pour_balls_into_vase.json

七球全入花瓶、杯恢复直立、回原位，非液体代理。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/pour-balls-into-vase/",
    "description": "There is a cup containing many small balls and a vase. The robot needs to pick up the cup and pour all the balls into the vase.",
    "scoring": [
      {
        "score": "0",
        "condition": "Not all balls are transferred into the vase."
      },
      {
        "score": "100",
        "condition": "All seven balls are inside the vase, the cup is upright, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/pour_by_language.json

三色液体按语言和顺序倒入对应碗；前一阶段要求后续液体尚未倒。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/pour-by-language/",
    "description": "There are three colored bottles and three colored bowls on the table. The robot needs to follow the language instruction and pour the liquid from each specified bottle into the corresponding specified bowl.",
    "scoring": [
      {
        "score": "0",
        "condition": "The first target pour is not completed."
      },
      {
        "score": "20",
        "condition": "The liquid from the first specified bottle is poured into its specified bowl, and the other two liquids have not been poured into their bowls."
      },
      {
        "score": "50",
        "condition": "The first two specified liquids are poured into their bowls, and the third liquid has not yet been poured."
      },
      {
        "score": "100",
        "condition": "All three specified liquids are poured into their corresponding bowls."
      }
    ]
  }
}
```

### configs/tasks/pour_liquid_into_cup.json

杯达所需液量且瓶直立，文本未给量的数值阈值，需读取 evaluator。

合并后的 H=5，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/pour-liquid-into-cup/",
    "description": "There is a bottle containing liquid and a cup. The robot needs to pick up the bottle and pour the liquid into the cup.",
    "scoring": [
      {
        "score": "0",
        "condition": "The liquid is not successfully poured into the cup."
      },
      {
        "score": "100",
        "condition": "The cup contains the required amount of liquid and the bottle is upright."
      }
    ]
  }
}
```

### configs/tasks/press_by_number.json

精确次数和蓝确认顺序；description 说蓝键立即结束，scoring 写两次确认，二者有潜在歧义须以执行代码核实。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/press-by-number/",
    "description": "There are two number cards, two red buttons, and one blue confirmation button. The robot needs to read the numbers, press each red button the corresponding number of times, and then press the blue button to confirm. Pressing the blue button ends the task immediately.",
    "scoring": [
      {
        "score": "0",
        "condition": "The required press counts or confirm sequence is not completed exactly."
      },
      {
        "score": "100",
        "condition": "Button `0` is pressed exactly the number shown by `num0`, the blue confirm button is pressed, button `1` is pressed exactly the number shown by `num1`, the blue confirm button is pressed again, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/push_T.json

T块推到目标xy/姿态且不许超高抬升，回原位；抓取放置不等价。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/push-t/",
    "description": "There is a thin gray T-shaped pad and a T-shaped block. The robot needs to push the T-shaped block until it is precisely aligned with and fitted onto the gray pad.",
    "scoring": [
      {
        "score": "0",
        "condition": "The target pose, reset, or no-lift condition is not satisfied."
      },
      {
        "score": "100",
        "condition": "The T-shaped block is within the target xy threshold, its orientation matches the target, the robot returns to origin, and the block is never lifted above the allowed height."
      }
    ]
  }
}
```

### configs/tasks/put_bottles_into_dustbin.json

四瓶入桶，右侧因桌位需换手；逐件10/25/40/100。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/put-bottles-into-dustbin/",
    "description": "There are four bottles on the table, and they may be either standing or lying down. A dustbin is placed beside the table. The robot needs to pick up the bottles and throw them into the dustbin. Because of the shifted table layout, bottles on the right side require a handover between the two hands before being discarded.",
    "scoring": [
      {
        "score": "0",
        "condition": "No bottle is placed into the dustbin."
      },
      {
        "score": "10",
        "condition": "Any one bottle is inside the dustbin, and the gripper is open."
      },
      {
        "score": "25",
        "condition": "Any two bottles are inside the dustbin."
      },
      {
        "score": "40",
        "condition": "Any three bottles are inside the dustbin."
      },
      {
        "score": "100",
        "condition": "All four bottles are inside the dustbin and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/solve_equation.json

选择缺数字或运算符补算式，数字保持姿态容差，回原位。

合并后的 H=5，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/solve-equation/",
    "description": "There is an arithmetic equation on the table and a pad for the answer. The equation is missing either a number or an operator. The robot needs to choose the correct missing item from the randomly arranged numbers and operators on the table and place it on the pad to complete the equation.",
    "comment": "Each digit must remain within the permitted rotational tolerance relative to its initial pose.",
    "scoring": [
      {
        "score": "0",
        "condition": "The missing equation piece is not placed correctly."
      },
      {
        "score": "100",
        "condition": "The correct missing number/operator piece is placed on the missing mat and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/sort_nesting_dolls_by_size.json

五娃直立，左到右小到大并水平成一直线，回原位；没有部分分。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/sort-nesting-dolls-by-size/",
    "description": "There are five nesting dolls of different sizes on the table. The robot needs to sort them by size and arrange them in a straight row from left to right, from smallest to largest.",
    "scoring": [
      {
        "score": "0",
        "condition": "The dolls are not arranged in the required size order."
      },
      {
        "score": "100",
        "condition": "All five nesting dolls stand in a one horizontal straight row, are ordered from left to right from smallest to largest, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/stack_blocks.json

三纹理块堆叠；两块15，三块且回原位100。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/stack-blocks/",
    "description": "There are three blocks with different textures on the table. The robot needs to pick them up and stack them into a stable pile.",
    "scoring": [
      {
        "score": "0",
        "condition": "No two-block stack is completed."
      },
      {
        "score": "15",
        "condition": "Any two of the three blocks are stacked, and the gripper is open."
      },
      {
        "score": "100",
        "condition": "All three blocks are stacked together and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/stack_blocks_by_language.json

三色块按指定顺序；首两块20且底块留桌，三块回原位100。

合并后的 H=5，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/stack-blocks-by-language/",
    "description": "There are several colored blocks. The robot needs to understand the language instruction and stack the blocks in the specified color order.",
    "scoring": [
      {
        "score": "0",
        "condition": "The first ordered stack is not completed."
      },
      {
        "score": "20",
        "condition": "The first two specified blocks are stacked in the instructed order, the bottom block remains on the table, and the gripper is open."
      },
      {
        "score": "100",
        "condition": "All three specified blocks are stacked in order and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/stack_bowls.json

三碗直立嵌叠，底碗必须稳定留桌，回原位。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/stack-bowls/",
    "description": "There are three bowls. The robot needs to stack all the bowls together.",
    "scoring": [
      {
        "score": "0",
        "condition": "No two bowls are stacked correctly."
      },
      {
        "score": "15",
        "condition": "Any two bowls are stacked upright and the gripper is open."
      },
      {
        "score": "100",
        "condition": "All three bowls are stacked together, all bowls are upright, the bottom bowl is strictly and stably placed on the table, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/store_laptop_and_headphones.json

先耳机上架20，再关电脑竖插满足80，最后回原位100。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/store-laptop-and-headphones/",
    "description": "There is an open laptop on a laptop stand, a vertical laptop stand, a pair of headphones, and a headphone stand. The laptop opening angle is random but greater than 30 degrees. The robot needs to first hang the headphones on the headphone stand, then close the laptop, pick it up from the stand, and insert it into the vertical laptop stand.",
    "scoring": [
      {
        "score": "0",
        "condition": "Neither the headphones nor the laptop is stored correctly."
      },
      {
        "score": "20",
        "condition": "The headphones are placed on the stand: the headband bridge is close to the support cradle and aligned, with the gripper open."
      },
      {
        "score": "80",
        "condition": "The headphones are correctly placed on the stand, and the laptop is closed, vertically inserted into the laptop rack, and within the required range."
      },
      {
        "score": "100",
        "condition": "Both the headphones and laptop conditions are satisfied and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/store_tools_in_toolbox.json

四工具对应槽位，逐个25分；满分还含工具被工具箱覆盖与回原位。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/store-tools-in-toolbox/",
    "description": "There are several tools and a toolbox with designated slots. The robot needs to pick up each tool and place it into the corresponding position in the toolbox.",
    "scoring": [
      {
        "score": "0",
        "condition": "No tool is placed into its matching toolbox slot."
      },
      {
        "score": "25",
        "condition": "Any one tool is placed in its matching toolbox slot with the gripper open."
      },
      {
        "score": "50",
        "condition": "Any two tools are placed into their matching toolbox slots."
      },
      {
        "score": "75",
        "condition": "Any three tools are placed into their matching toolbox slots."
      },
      {
        "score": "100",
        "condition": "All four tools are placed in their matching slots, covered by the toolbox, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/swap_T.json

记住原位置和朝向，双T块互换xy与姿态；只有终态对称摆放不够。

合并后的 H=5，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/swap-t/",
    "description": "There are two T-shaped blocks on the table. The robot needs to grasp them with both hands, swap their positions, and place them back so that each block matches the original pose of the other one, including both position and orientation. This is a memory-based task.",
    "scoring": [
      {
        "score": "0",
        "condition": "The two T blocks are not swapped with the required pose accuracy."
      },
      {
        "score": "100",
        "condition": "The two T-shaped blocks swap their original xy positions, their orientations match the swapped targets, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/swap_blocks.json

利用第三空垫三步交换，每移动一次必须按一次按钮；最终位置正确但事件少不能完成。

合并后的 H=10，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/swap-blocks/",
    "description": "There are three mats, two blocks placed on two of the mats, and one button. The robot needs to swap the positions of the two blocks by using the empty mat as a temporary place. After each move, the robot must press the button.",
    "scoring": [
      {
        "score": "0",
        "condition": "The swap sequence is not completed, or the button is not pressed after each move."
      },
      {
        "score": "100",
        "condition": "The two target blocks are swapped using the empty mat, the button is pressed once after each of the three moves, the blocks finish on each other's original mats, and the robot returns to origin."
      }
    ]
  }
}
```

### configs/tasks/sweep_blocks.json

扫帚左到右手交接，左手拿簸箕；全块入簸箕、簸箕在扫帚左侧且直立稳定，回原位。random 有明确 instruction alias。

合并后的 H=15，每步最大平移=0.03m、最大旋转=0.15rad；task override={}。instruction aliases={"random": ["Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the objects."]}。

实际文件字段和所有评分级别如下；此处保留原文，避免翻译丢失约束：

```json
{
  "reference_instruction_aliases": {
    "random": [
      "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the objects."
    ]
  },
  "official_task_documentation": {
    "version": "robodojo-public-2026-09-23",
    "source": "https://robodojo-benchmark.com/doc/sim-tasks/sweep-blocks/",
    "description": "There is a broom and a dustpan on the left side, and the blocks are on the right side. The robot needs to pick up the broom, hand it over to the right hand, grasp the dustpan with the left hand, and sweep the blocks into the dustpan.",
    "scoring": [
      {
        "score": "0",
        "condition": "The blocks are not fully swept into the dustpan."
      },
      {
        "score": "100",
        "condition": "The dustpan is positioned left of the broom, stably placed upright on the table, all blocks are inside the dustpan, and the robot returns to origin."
      }
    ]
  }
}
```

## 15. HUMANO G1 + Dex3 迁移：可复用的逻辑与不可等价的接口

可复用的是“固定示范为上下文、TRAIN与LIVE明确分段、起点/动作块/真实终点成对、哈希锁源、只保留有限LIVE历史、动作解析失败有界纠正、实际执行后回执”的算法结构。原 arx_x5 的 camera/robot/action 数据不能改个路径就成为 G1 数据。

|边界|上游实际契约|G1 + Dex3 必须独立定义的差异|等价性结论|
|---|---|---|---|
|机器人与控制|arx_x5 双臂，每臂6维 world EE delta+1夹爪|G1全身、站姿/基座/腿/腰与双臂协调；Dex3多关节与接触动作|14维行不能表达多指手形和全身；单标量开合映射是有损新接口|
|末端坐标|HDF5 state/action ee_pose，wxyz quat，world左乘差|明确定义掌心、腕或指尖参考框、base/world/环境原点；动态底座如何更新|未校准转换前禁止直接套上游delta和摄像机外参|
|观测|cam_head+左右wrist，480x640 RGB；双臂六组state|G1相机安装/遮挡、手指可见性、状态schema与Dex3 joint顺序|triptych消息布局可复用，图像与状态含义不等价|
|动作采样|25Hz，H按任务5..25，target-to-target delta|SONIC实际执行频率、插值/延迟、teleop token及手指同步|改频率不只是改frequency字段；窗口时长、动作限幅、终点帧全部改变|
|IK与执行保护|arx_x5 ee控制、joint increment guard0.15rad|G1 IK如何作用于新目标、SONIC真正模式、足底/桌/障碍碰撞反馈|无声退到IK/mode0/teleop是协议改变，需逐阶段实际证据|
|source TRAIN|22条arx_x5 HDF5，源哈希、完整机器人状态与标定|同一任务G1演示，或明确注明跨机型参考转换过程与失真|保留原动作作为模型示范是跨形态ICL；直接执行转换原动作是轨迹重定向|
|任务成功|各RoboDojo evaluator含回原位、动作顺序、抬高限制|定义G1原位、平衡/站立、Dex3松手、物体接触与稳定条件|修改这些条件后的分数不能直接与原100分成功率等号|
|场景与物理|固定Assets revision、layout、arx_x5 env YAML|HUMANO场景、G1身高/桌高、手碰撞体、摩擦/惯量/solver|即使视觉相似也不是matched comparison|
|运行依赖|IsaacSim5.1约束、分离policy/sim、cuRobo|项目指定TensorRT10.13.0.35 + ORT1.23.2 + SONIC1.0 CPU，humano_python.sh|必须保存loaded-session providers和模型hash；requirements文本不构成实测|
|实验记录|本地无时区runid、run/status JSON|北京时间分钟ID、冻结experiment.json、阶段接口/证据/预算/接受阈值|上游launcher本身未满足cap-sim实验制度，不能直接启动新物理实验|

如果目标是证明从成功演示抽取可迁移经验，需要另定义“站姿选择、接近、抓型、接触条件与恢复规则”的结构化经验、执行规划器当前可见信息及是否能访问原 action sequence，并使用新位置/桌高等匹配条件检验策略生成。只确认 reference bundle 完整、动作能被IK执行、任务在同一布局成功，都不单独证明技能生成或经验迁移。

## 16. 可复核步骤与未覆盖清单

本次已完成所分配 18 个 Python/shell/依赖文件以及49个配置文件的静态读取与逐文件说明；没有以 grep 命中代替核心函数阅读。文中对 bundle/offline 的调用边界额外读了 bounded_policy.Model 与 dialogue_policy._initialize 对应段落。清单覆盖检查只用 Python 标准库读取文本，不 import Isaac Sim、不初始化 physics。

后续复现可先进行纯静态/离线检查：确认 commit 和 submodule commit/cleanliness；核对每个 JSON 与本文、TRAIN starts/horizon/哈希；在独立 policy 环境验证源 HDF5 与bundle；资产安装后可做 run --dry-run（仍需全部路径与manifest）。`--capture-only` 会启动仿真，不能当纯离线检查；执行前必须冻结项目实验记录。任何物理运行和API调用都未在本文完成。

未完成项必须保留为限制：

1. 本文范围之外的 policy/rollout/server 全文件、third_party 源码、env YAML、RoboDojo task evaluator、tests/docs 不由本子任务逐文件覆盖；相关结论仅在已读调用点范围内。它们需由总审计文档单独覆盖，不能据此声称整个上游仓库已读完。
2. 未下载/解码真实22个HDF5，因此未知记录是否能通过当前构建器平移0.03限制、元数据相机约定是否符合真实writer；未验证源episode“成功”标签与接触质量。
3. 未安装 pinned sim/policy 环境、未跑 pytest、未启动WebSocket/模型provider、未运行 dry-run/物理/渲染，因此没有成功率、API兼容性、时延、仿真/渲染耗时结果。本文规划/审计总时长未计量，记 unknown，不伪装成实验耗时。
4. 官方文档与评分配置自身的歧义（按数字按钮、整理桌面、数字数量）尚未对照任务执行源码定论；这是具体待核查项，不推定源码一定错误。
5. 未实现 G1+Dex3 adapter、相机标定、25Hz对齐、多指动作编解码、SONIC provider模式证据和公平对照；迁移表是待满足契约，不能被标为已完成适配。
6. 本次没有新实验目录，未冻结新的 experiment.json，未触发任何新物理实验。后续比较须按项目规则使用 `experiments/20261013_apple_straight_sweep` 最近十条件，清楚标注历史 IK/mode0、SONIC teleop 及保留原token前缀；上游30任务或42配置集合不能替代该指定对照。
