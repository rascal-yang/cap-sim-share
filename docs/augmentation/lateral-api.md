# API 驱动的苹果横向抓取

入口：`python scripts/run_lateral_agent.py --config /absolute/new-study/config.json --provider /absolute/new-study/provider.json --credentials ../api/openai_config.toml`。

多组凭据用 `--credentials-pool /private/api9.toml /private/api8.toml ...`。每个真实请求（包括失败重试）按持久化顺序轮换一次，保持模型和 reasoning effort 相同；attempts 和 response_origin.json 保存配置名称、顺序、端点及响应哈希。公共 api_pool.json 只含环境变量名，密钥仍只在环境中。状态以独立文件锁和原子替换保存，缺失的响应来源不能恢复为匿名授权。原生调用可显式选 `history_protocol="role_messages"`，将历史动作/结果转成普通角色消息，新的模型输出仍为真实原生 function call。

沿用 ResponsesProvider，API 真实调用 read_skill → inspect_scene → plan_lateral → prepare_grasp → execute_grasp → reflect_grasp；Python 生成全身参考并执行固定 SONIC mode1。API 选择有限范围的腕部高度修正并反思实际结果，不输出逐物理步力矩。原动作可见，方法标记为 trajectory_retargeting。

`config.json` 包含 output_root、manifest、initial、options（绝对路径）、offset_m（±0.2 m 内）、max_attempts（1–3）、max_rounds（1–20）、wall_s、experience_results 与 input_sha256。后者绑定上述输入、经验配置/接触记录及实际实现。Provider 使用已有 Responses 配置；凭据仅在环境里，不写入请求/实验文件。

每轮 API request/response、usage 与工具参数/结果分别保存。模型只能在后续请求已包含初始图片时执行，在后续请求已包含实测结果时反思。完整回复先校验再分发，防止格式错误后半段发生在物理副作用之后。同一 prepared_id 再执行返回已有结果；已启动但未保存结果的执行不能自动重跑。新准备前必须记录上一轮 retry，原生通过而碰桌超标不能 accept。至少一个有物理子步的执行和 accept/stop 反思后才能完成流程。

`decision_protocol` 显式选择 `native_tools`（默认）或 `json_action`。后者用于原生工具调用转发不可用的网关：API 返回 `{"tool":"工具名","arguments":{...}}`，harness 按相同规则验证并派发；通过普通角色消息发送工具结果和图像，不发送原生 tools 字段。原始模型 message 保存在 response.json，派发用的规范化动作另存 normalized_action.json，明确标记 `native_function_calling=false`。完成需返回 `{"tool":"finish","arguments":{"summary":"实测结论"}}`，仍受已执行/已反思门约束。不会静默切换协议。

Provider 可显式配置 `transport_python` 为系统 Python 的绝对路径，只把 HTTP 收发放在标准库子进程，凭据通过继承环境传递，避免仿真运行时的网络兼容问题。请求/响应走 JSON stdin/stdout，父进程继续核对实际模型，attempts 记录子进程 Python/SSL 版本。规划/物理仍经 humano_python.sh，物理模式和版本没有网络兜底。首次模型上下文只发送任务条件；完整输入/源码哈希保留在本地校验与冻结契约。

连通性测试已返回真实 `read_skill` 调用时，可用 `bootstrap_read_skill_probe` 显式继承其 request.json/response.json/result.json。仅允许模型、工具 schema 匹配且恰好一个 `read_skill({})` 的完成响应；物理调用不能继承。该真实请求作为已继承前缀单独计数、保留来源哈希，工具结果回传后追加新的实际实验任务；不伪造新的模型调用，也不继承任何物理执行权限。

每轮先冻结独立 experiment.json、源码/API 决策哈希和真实 loaded-session CPU provider 证据，再记录模型对初始图片的执行授权。实际运行保存 mode1、200 Hz 状态/接触和未覆盖的 decoder 输出；独立验收仅读取状态。完整解码通过的 MP4 浅层副本、源路径与 SHA 保留在所属实验目录。

第一版明确初始化在生成的操作站位，使用静止 SONIC 历史，测试站位后的条件抓取；不验证从原始远处走近。支持右腕 Z 修正 0–3 cm；手指预形状/闭合策略修改尚无工具，API 应停止并报告能力缺口。原生抬起与碰桌质量分别报告；渲染视频 training_ready=false。

## 数值快速模式（当前迭代默认）

用户要求先不加视图、缩短单次实验。配置 `execution_mode="privileged_fast"` 并提供 `cached_plan` 后，仅暴露 `run_grasp` 和 `reflect_grasp` 两个工具。主机核对缓存的源初态、平移初态、偏移、规划选项、参考与代码哈希；不把缓存复用算成新规划或 API 调用。失配必须重新规划，不能套用旧站位。

`run_grasp` 选择候选与 0–3cm 腕高修正，同一进程加载一次控制器、保存真实 provider 和模型哈希、数值检查初态并冻结 experiment.json，再直接执行。支撑、闭合对齐、接触力、摔倒和最终保持检查照常执行。独立数值审计读取保存的物理状态；初始图、接触图、MP4 均不生成或上传。每次迭代的 API 输入图片数量为零，后续按需离线渲染。

结果包括原生成功、完成状态、手物/障碍峰值力、碰桌部位及持续时间、腕部跟踪和双脚漂移。API 在下一请求中反思，记录后立即结束，无需额外总结请求。默认每会话一次物理试验、两次成功 API 响应；网络失败重试单独计数。达到预算时 `retry` 仅表示下批建议，不声称已验证或已执行重试。启动意图先持久化，同一决策重复返回已有结果，执行中断不能自动重跑。旧有看图模式仍须显式使用原流程。

数值审计的`audit_numeric`仅加载场景与原生checker，保持初态锚点及每4子步验收频率，不加载SONIC策略。实际执行阶段继续保存真实CPU provider/模型证据。离线复核可传入独立output_path保留原审计；同一轨迹全指标对照已通过。

## 精停导航与完整状态回溯（数值模式默认）

`execution_mode="privileged_fast"`现在默认`navigation_mode="precision"`：保留原始远位初态与源控制器历史，执行精停导航、replay抓取、全部SONIC token及独立手指记录。旧站位初始化需显式`navigation_mode="direct_stance"`。导航模式不采用旧站位流程的力阈值中止，接触按实际身体部位报告，原生成功和clean_success分开。

导航模式的`run_grasp`增加`execution`（restart/navigation_start/parked）、`parent_prepared_id`和0–3的`feedback_gain`。第一次必须restart且父ID为空；后续先根据已收到实测结果reflect_grasp(retry)，再选择已有检查点。parked允许换反馈增益，腕高须保持父值；修改腕高需navigation_start或restart。检查点必须在当前会话中实际保存且结果已进入API历史。更换离线plan或代码需要新种子。停稳继承的导航token和本次新执行后缀在lineage中分别计数。

无需单独检查点工具或多余总结请求：两次物理尝试通常四个成功API响应（run/reflection/run/reflection）。API选择实际分支，主机不伪造调用。碰桌不会自动触发回溯；由API读取数值结果决定。原有幂等、启动意图持久化、真实响应来源和配置池轮换继续适用。

实测记录：`experiments/2026-10/28/20261028_1244_apple_api_navigation_json/REPORT.md`。真实JSON-action API先执行原始远位精停抓取，再根据实测结果retry并执行parked回溯分支；两次原生成功、手指碰桌均未清洁通过。API最终反思请求持续超时，因此该次会话没有标为completed。

新会话使用新的output_root，明确`decision_protocol="json_action"`、`history_protocol="role_messages"`、`navigation_mode="precision"`及`max_attempts=2`；API凭据继续经`--credentials-pool`轮换，密钥不进入配置文件。实验中的prepare_navigation_api_json_validation.py已归档，生成新目录、缓存计划输入与当前源码哈希；不得直接重跑已有物理case。不要把原生工具请求失败后的JSON协议作为静默回退。

网络恢复限制：主入口根据当前tool_state生成next_step，不能保证所有中断位置都能原样续接历史请求。本次使用归档的recover_navigation_api.py从已冻结待决请求恢复，只执行缺失分支/反思，原有物理未重跑。它是本次实验的显式恢复脚本，不是入口自动恢复功能。API超时不应当隐式增加物理次数。
