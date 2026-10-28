# Apple Lateral Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Generate auditable apple lateral stance/operation plans from repaired successful replay and consume regenerated references in the existing measured workflow.
**Architecture:** Add lateral extraction, bounded support-constrained IK, reference bundle and orchestration modules under workflow/navigation. Keep legacy replay navigation opt-in behavior; new plans bind distinct original and generated evidence and replan once from measured feet at handoff.
**Tech Stack:** Python, NumPy, SciPy, MuJoCo kinematics, pinned HUMANO runtime.
**Spec:** docs/superpowers/specs/2026-10-27-apple-lateral-navigation-design.md (approved by user: 开始实现).

## Global Constraints
- Apple world Y/table-tangent offsets only; preserve table/depth/height, scene physics and robot initial state.
- Source 20261025_1719 has 346 original token steps plus mode1 preencoder pose feedback; generated method has zero original prefix and no hidden fallback.
- No physics until complete frozen experiment.json and matching calibration. Offline evidence is not task success.
- Keep measured source and generated reference separate; original files read-only, source access labelled trajectory_retargeting.
- Work in current linked feature/experience-dataset worktree; preserve unrelated untracked files.
- Every generated state and interpolation must pass foot/wrist pose, support/collision and speed checks within declared budgets.

## Review Focus
- Hand commands have no own timestamps: require paired controller timestamps, forbid blind repeat-four alignment (Task 1).
- A quiet pelvis can coexist with moving feet, and fixed feet with moving pelvis; use support poses for extraction (Task 1).
- IK can satisfy keyframes but violate intermediate poses/velocity or collide with idle hand/table (Task 2).
- Measured landing and moved object invalidate offline reference; regenerate once, preserve the derived artifact, never silently retransform source (Task 3).
- Truncated search and changed evidence must never become feasibility/infeasibility certification (Tasks 2–4).

### Task 1: Support and command evidence extraction
**Files:** create navigation/lateral_prior.py, tests/test_lateral_prior.py; add optional measured body poses to navigation/archive.py.
**Interfaces:** align_hand_commands(state_times, control_times, hands, control_dt_s) -> (fingers, command_rows); extract_support_prior(trace, bindings, options, body_poses, fingers) -> JSON prior; load_lateral_source(manifest_path, options) -> (trace, prior, fingers, evidence).
- [x] Add tests for actual-time zero-order hold, absent/malformed coverage, feet stationary with pelvis moving, foot motion rejection, closed entry and unverified source.
- [x] Run scripts/humano_python.sh -m pytest -q humanoid_mimicgen/workflow/tests/test_lateral_prior.py; expect missing API failure.
- [x] Implement audited multiple support windows, table coordinate frame from geometry, command time mapping, explicit source control metadata and stage maps.
- [x] Run same test command and existing navigation archive/prior tests; expect pass. Commit task.

### Task 2: Bounded lateral planning and generated reference bundle
**Files:** create navigation/lateral_ik.py, navigation/lateral.py, navigation/reference_bundle.py; tests/test_lateral_planning.py.
**Interfaces:** LateralSolver(model, source_qpos, source_times, prior, bindings, options).solve(candidate, object_delta, current) -> reference/audit; lateral_candidates(prior,current_feet,current_root,delta,search) -> candidates; plan_lateral(manifest,initial,options,output_dir) -> plan.json; load_reference_bundle(plan) -> validated arrays.
- [x] Add real kinematic tests for foot-constrained wrist retarget, unchanged depth/height, source pelvis motion preservation, off-table rejection, bounded search, forbidden contacts, orientation/speed rejection and bundle tampering.
- [x] Run test_lateral_planning.py; expect missing API failure.
- [x] Implement full-body bounded least-squares IK with planted feet, source pelvis preference and sequential warm starts; complete state/interpolation verification and route/entry checks; immutable reference bundle and no-overwrite output.
- [x] Run both lateral test modules; expect pass. Commit task.

### Task 3: Measured handoff and execution consumption
**Files:** modify navigation/integration.py, navigation/whole_body.py, workflow/runner.py, workflow/service.py; create tests/test_lateral_execution.py.
**Interfaces:** regenerate_handoff(plan,current_qpos,output_dir) -> audit with derived bundle; validate generated plan against original reference + generated arrays; WholeBodyOperation(..., generated_reference=None) consumes arrays indexed by source row.
- [x] Add tests that generated references are used unchanged, source hash/phase/finger mismatch rejected, measured feet cause regeneration/rejection, moved object and budget expiry reject; legacy path preserved.
- [x] Run test_lateral_execution.py; expect new path failure.
- [x] Implement bundle validation and one bounded measured-state regeneration, explicit derived artifact/logged planning pause, fresh state/object checks, source/source-finger mapping and calibrated execution preflight.
- [x] Run lateral and existing whole-body/navigation tests; expect pass. Commit task.

### Task 4: CLI, real replay grid and full review
**Files:** extend navigation/__main__.py; add configs/navigation/apple_lateral.json, tests/fixtures/navigation/apple_lateral.json, tests/test_lateral_cli.py, docs/augmentation/apple-lateral-navigation.md.
**Interfaces:** navigation lateral-extract --manifest --options --out; navigation lateral-plan --manifest --initial --options --offset-m --out-dir; navigation lateral-grid --manifest --initial --options --out-dir. Grid offsets 0, ±.05, ±.10, ±.15, ±.20 m. Generated plans retain legacy execute entrypoint.
- [x] Add CLI tests for no overwrite, invalid delta, archived source integration and structured failure evidence; run to observe failure.
- [x] Implement CLI and explicit configuration with evidence-backed hand/phase mappings, normal/tangent declaration and finite budgets.
- [x] Run nine-condition offline grid in a new Asia/Shanghai minute-ID experiment directory; record source hashes, timing, candidates and honest partial/budget results. No physics.
- [x] Run default suite plus workflow suite, inspect all failures, and perform fresh code review. Fix material findings with regression tests. Commit task and docs.

## Ledger
- Execution authorized by user. Native implementation selected to keep coupled interfaces in one context; independent final code review retained.
- Design parameters are explicit starting configurations, not calibrated robot capabilities.

## Completion evidence

- Runtime implementation: `7e392e1`; default suite 108 passed, workflow suite 143 passed.
- Final offline grid: `experiments/2026-10/27/20261027_1136_apple_lateral_navigation_offline/REPORT.md`; nine of nine conditions found checked feasible references, all searches incomplete; 0 physics steps. All nine source/bundle/scene integrity checks passed.
- Matching lateral stopping calibration remains absent: physical tracking and task success unverified. No merge/push or worktree cleanup requested; retain feature/experience-dataset.

# SDD ledger — plan: docs/superpowers/plans/2026-10-27-apple-lateral-navigation.md
Task 1: complete (commits 4b01f86..f8ed89e, tests: scripts/humano_python.sh -m pytest -q humanoid_mimicgen/workflow/tests/test_lateral_prior.py humanoid_mimicgen/workflow/tests/test_navigation_archive.py humanoid_mimicgen/workflow/tests/test_navigation_prior.py → 17 passed in 1.12s)
Task 2: complete (commits f8ed89e..369df7e, tests: scripts/humano_python.sh -m pytest -q humanoid_mimicgen/workflow/tests/test_lateral_prior.py humanoid_mimicgen/workflow/tests/test_lateral_planning.py → 14 passed in 1.17s)
Task 3: complete (commits 369df7e..d2c27c1, tests: scripts/humano_python.sh -m pytest -q humanoid_mimicgen/workflow/tests/test_lateral_execution.py humanoid_mimicgen/workflow/tests/test_whole_body_operation.py humanoid_mimicgen/workflow/tests/test_navigation_cli.py → 31 passed in 1.28s)
Pre-flight: Tasks 1→2→3→4 share source-row indexed generated arrays and immutable evidence; interfaces matched.
Final: fixed joint limits, exact derived plan/source/time binding, optimizer/A* exhaustion reporting and whole-operation deadlines — regression tests RED→GREEN; default suite 108/108 and workflow 143/143.
Final: fixed independent full-path both-foot support gap — moving one foot within pose tolerance now fails; RED→GREEN.
Final: corrected real apple support projection (mesh vertices, not bounding sphere), preserved first diagnostic grid; source-mesh regression RED→GREEN.
Final: Ruling: independent review minor evidence omissions are required audit details; implemented per-row/segment residuals, declared positional margin floor and original-dataset-relative offsets.
Final: Ruling: hand collision geometry may preserve source measured soft-limit deviations up to declared 0.0033rad (observed maximum 0.003216528rad); commanded finger ranges remain unchanged and separately checked. Misinterpreting measured geometry as commanded tolerance would invalidate control claims, so both are explicitly labelled.
Final: Ruling: physical efficacy/SONIC tracking/lateral stopping calibration remain unverified; no physics launched without matching evidence, as required by AGENTS and existing runtime preflight.
Task 4: complete (commits d2c27c1..7e392e1, tests: scripts/humano_python.sh -m pytest -q humanoid_mimicgen/workflow/tests/test_lateral_cli.py → 5 passed in 1.79s)
