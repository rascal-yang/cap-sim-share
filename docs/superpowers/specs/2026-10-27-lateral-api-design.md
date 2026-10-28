# Apple lateral API execution

The user requests the actual model API → local planning tools → physical grasp → evidence/reflection flow. Reuse the existing ResponsesProvider, repaired replay support planner and proven direct-stance mode1 physics probe. Keep work in feature/experience-dataset. The prior request to start physical experiments and omit unnecessary process steps authorizes inline implementation without another design approval.

The API reads a task skill and scene/replay experience, requests a fresh lateral plan, chooses a bounded right-wrist height adjustment, prepares a frozen experiment, sees its initial image, explicitly executes it and reflects on the measured result. Local Python performs FK/IK, verification, SONIC control and physics. These responsibilities and original-action access remain explicit. Initializing at the operation stance tests conditional manipulation; it does not test walking.

Approach: extend with a dedicated narrow tool session, using the current transport, rather than expand the old fixed-job batch schema or build a general shell agent. The choices are local engineering decisions. Keep the source controller and original frozen experiments intact. All new physics uses a new frozen contract, loaded CPU session evidence, immutable API provenance, fixed safety/acceptance and no hidden fallback. Credentials stay in environment. The API cannot select paths or edit physical properties.

At most three preparations/physical attempts and twenty model rounds; new attempts require a recorded reflection on the preceding one. Every prepared case has its own identity and immutable hashes. Duplicate execution returns saved evidence; an interrupted launch cannot silently rerun. API text alone cannot mark completion. Native lift and collision quality stay separate. At least one verified physical execution and a model reflection are required for workflow completion. Failed/incomplete API responses cannot authorize tools.

Store requests, responses, usage, tool arguments/results, selected plan, frozen contracts, physical traces, independent audits and shallow Chinese MP4 copies with SHA indexes. A model acceptance must agree with native and collision results. Unsupported finger-strategy repairs must be reported as capability gaps.

## Implemented protocol choices

Native tool forwarding was attempted first and preserved as failed connectivity evidence. An explicit json_action protocol retains real API message JSON and separately labelled normalized dispatch actions; native_function_calling=false. HTTP may run in an explicitly configured system-Python subprocess, while planner/physics always use the pinned HUMANO wrapper. Model context excludes local source hashes; those remain locally verified and frozen. Later live-session configuration uses endpoint-listed gpt-6.1-sol low; this changes no robot controller and makes no matched-model comparison claim.

## 2026-10-27 用户修订：优先数值快速迭代

采用 privileged_fast：不生成/上传视图，不强制模型看图；缓存匹配的几何站位和原参考；两次成功 API 响应（run_grasp / reflect_grasp）。主机完成只读技能/缓存校验，明确区别于新模型调用。prepare 与 physics 复用同一已验证 NativeAdapter，冻结契约后才 mj_step；独立数值审计保留。一次预算用尽后的 retry 是未来建议；无需额外 final RPC。API 配置继续逐请求轮换。训练视频验收不在本次数值流程中，training_ready=false。
