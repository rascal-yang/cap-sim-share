# Lateral API implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Steps use checkboxes.

**Goal:** Complete one real model API session controlling new lateral planning and direct-stance physical grasp tools.
**Architecture:** Promote the recorded physics probe to a bound reusable driver; add stateful narrow tools and an audited Responses loop; launch through the existing credential mechanism.
**Tech Stack:** Python, ResponsesProvider, NumPy/SciPy/MuJoCo, pinned HUMANO runtime.
**Spec:** docs/superpowers/specs/2026-10-27-lateral-api-design.md

## Global constraints
- Runtime TRT10.13.0.35/ORT1.23.2/SONIC1.0 CPU, no mode0/token fallback or live state reset.
- Every physical case freezes complete experiment.json, runtime and API provenance before launch.
- New Asia/Shanghai minute-ID experiment, preserve original evidence, indexed hash-verified shallow videos.
- API controls tool/parameter selection; Python computes references and torques. Original actions visible: trajectory_retargeting.
- Direct initialized stance is explicit; navigation remains outside this first API validation.

## Review focus
- Incomplete/malformed API replies cannot launch physics.
- Duplicate/interrupted tool execution must never silently rerun a physical case.
- Prepared reference, API decision, image and actual execution must remain bound.
- Native-only success cannot become clean success; missing reflection cannot become completed workflow.
- Credentials and account headers cannot enter saved requests or provenance.

### Task 1: Physics driver and API tool contract
- [x] Add failing tests for parameter validation, prepare/execute/reflection order, image exposure, immutable inputs and exactly-once execution.
- [x] Promote direct-stance probe to augmentation/direct_grasp.py with explicit API provenance and subprocess isolation; add independent saved-state audit.
- [x] Implement lateral_tools.py, skill and bounded tools with persisted identities and no arbitrary paths.
- [x] Verify focused tests and commit.

### Task 2: Model loop and launch
- [x] Test incomplete replies, final-text bypass, injected transport labelling and replayed call IDs.
- [x] Implement lateral_agent.py and credential-only launcher; preserve actual API request/response/usage and initial/review images.
- [x] Verify default and workflow tests; independent review and corrections before launch.

### Task 3: Real API physics evidence
- [x] Freeze code/config, create actual minute-ID root and run real gpt-6-astra low session at Y+5cm.
- [x] Verify loaded providers, tool provenance, physical traces and independent native audit; earlier image session videos indexed. Latest numeric-only session omits rendering per user.
- [x] Write reports with actual API counts, unknown usage/cost, physical results/limits and measured timing.

## Implementation evidence

- Runtime commit293f4e7; combined default/workflow/tracked augmentation suite499 passed. Unrelated untracked historical-contact test preserved; ROS auto-loaded pytest plugin disabled to avoid it importing unselected tests.
- Independent review fixes verified RED→GREEN: response-visible images/results, missing contact-focus image, full response validation before effects, refusal preservation during JSON normalization.
- Ruling: explicit JSON actions are labelled separately from native function calling; no silent protocol fallback.
- Ruling: static source/hash lists are local provenance and need not enter model context; source/decision bindings remain enforced.
- Minor review note: failed child-transport attempts can omit available child runtime metadata; successful attempts record it. No physical execution is inferred from transport probes.
- Real session root: experiments/2026-10/27/20261027_1451_apple_lateral_api_live. Earlier native/JSON connectivity diagnostics preserved in20261027_1418 and20261027_1441 roots; no new physical steps yet. One bounded300s transport wait currently pending.

## 用户修订后的收尾计划

- [x] 真实原生 API 已选择 .03 / .025m 并执行两次物理；前者触发205N手物力限制，后者原生成功但碰桌13.7N/5ms，均未宣称干净成功。
- [x] 按用户要求停止看图流程；.026m第三例仅准备，未执行，保留原契约。
- [x] 增加 privileged_fast、两工具闭环、缓存严格绑定、无渲染、同进程控制器复用；新增回归测试。
- [x] e0864fd冻结后完成20261027_1625数值闭环；2有效API+4超时，总244.008s，本地24.737s，图片为0。
- [x] 报告已核验108项冻结证据：原生成功、碰桌14.070N/10ms未通过；导航和策略迁移未验证。

后续离线优化：审计只加载场景和原生checker，避免加载SONIC；同一保存轨迹的所有审计指标（除时间）与原版精确一致，未新增物理、未覆盖旧审计。
