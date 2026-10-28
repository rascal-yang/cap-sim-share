# Replay Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Extract auditable replay stance relations, plan new scene stances and execute bounded feedback navigation without GPT.
**Architecture:** Add a focused `workflow/navigation` package with extraction, native adapters, planning and velocity control; integrate opt-in with existing workflow service/runner. Preserve existing execution defaults and frozen evidence.
**Tech Stack:** Python, NumPy, SciPy, MuJoCo kinematics; pinned HUMANO Python for native checks.
**Spec:** docs/superpowers/specs/2026-10-26-replay-navigation-design.md

## Global Constraints
- Zero GPT calls on normal navigation; bounded candidate/solver work.
- No physics launched during development. New execution requires existing frozen experiment contract checks.
- No modification of historical records; retain trajectory_retargeting labels and source-action access.
- Do not weaken existing parking calibration scope or imply dynamic reliability from kinematics.
- Reuse current linked feature worktree; preserve unrelated untracked files.

## Review Focus
- Ambiguous object axes and missing/unsuccessful replay evidence must never silently become successful priors (Task 1).
- Scene height/obstacles change independently of the manipulated object (Task 2).
- Entry IK success cannot certify the rest of the operation; unsupported moving-base contact must fail explicitly (Task 2).
- Waypoint corner cutting, heading wrap, stale state and finite stopping budgets (Task 3).
- Source/plan mutation, contract mismatch and formats of all four real tasks (Task 4).

### Task 1: Replay extraction and archive adapters
Files: create `humanoid_mimicgen/workflow/navigation/{__init__,prior,archive}.py`; tests `humanoid_mimicgen/workflow/tests/test_navigation_prior.py`.
Interfaces: `extract_prior(trace, bindings, options) -> dict`; `load_archive(manifest_path) -> (trace, bindings, evidence)`; JSON-compatible outputs, trace arrays in SI/world coordinates. Read workflow requests or explicit archive manifest; support historical four-task specs via archived observations/state and model assets.
- [x] Write tests for stable-window extraction, SE(2) relative pose, semantic/object-independent heading, absent success metadata and ambiguous phases.
- [x] Run failing tests with pinned Python; implement extraction/normalization; rerun tests.
- [x] Commit extraction and record actual source formats used.

### Task 2: Bounded stance planning
Files: create `navigation/planner.py`, `navigation/native.py`; tests `test_navigation_planner.py`; opt-in integration in `workflow/service.py`.
Interfaces: `plan_stances(prior, scene, current, options, evaluate) -> dict`; evaluator takes candidate and returns feasibility/audit. Native evaluator uses existing base planner and stationary IK/connect to validate required operation samples and connectors under explicit per-phase contacts.
- [x] Test rotated/transformed stance, unchanged table height, current-pose preference, bounded candidates, full-operation rejection and planning budget exhaustion.
- [x] Implement deterministic enumeration and native screening; propagate source and scene hashes; reject unsupported capabilities.
- [x] Run tests and commit.

### Task 3: Measured navigation controller
Files: create `navigation/control.py`; modify `workflow/runner.py`, `workflow/profile.py`; tests `test_navigation_control.py`.
Interfaces: `RouteController(route, limits).step(xyyaw, measured_velocity, dt) -> dict` emits world velocity, bounded heading reference, brake/arrival status, waypoint audit; downstream existing settle/measured-approach remains authoritative.
- [x] Test acceleration bounds, terminal braking, waypoint progression without unsafe corner cutting, yaw wrap/limits, invalid state, bounded timeout and zero-motion route.
- [x] Implement and opt-in through `navigation.profile = replay_navigation`; preserve existing profiles and strict calibration validation.
- [x] Run control and existing workflow regression tests; commit.

### Task 4: CLI, four-task evidence and complete verification
Files: `navigation/__main__.py`, `docs/augmentation/replay-navigation.md`; tests `test_navigation_cli.py`; update plan ledger.
Interfaces: extract/plan emit immutable JSON artifacts; execute binds planned strategy to existing service prepare/execute contract rather than bypassing runtime gates.
- [x] Test CLI no-overwrite, hash mutation, missing inputs, plan/contract mismatch and JSON failures.
- [x] Read real apple/plate/button/chair archived records without physics; store compact provenance fixtures and run offline extraction checks, reporting unsupported conditions honestly.
- [x] Run repository default suite plus workflow tests through pinned runtime. Review whole change, fix significant findings with regressions, commit documentation and completion ledger.

## Completion ledger

- Local implementation commits: `442f6eb`, `6b814a5`, `b375fcc`.
- Verification: repository default suite **108 passed**; workflow suite including actual four-task archive checks **91 passed**. No new physics experiment or external model call.
- Independent review: five findings (archive key, missing/deep support, object/environment collisions, yaw overshoot, missing scoped calibration) fixed with regressions. Reviewer rechecked `b375fcc`: all resolved; 25 targeted tests and 1,500 additional yaw-controller steps passed.
- Ruling: full articulated replay-reference kinematic screening replaces wrist-only screening for the standalone native evaluator, preserving chair/pelvis motion. New manipulation geometry that the reference cannot satisfy is rejected, not repaired by pretending navigation solves it.
- Ruling: execution is explicitly navigation-only. It ends after measured stability and full-reference handoff screening, saves final evidence, and reports task_success unknown. Continuous manipulation-controller integration and physical success-rate evaluation are not claimed.
- Ruling: current automatic correction budget is zero; failed measured handoff exits explicitly. No infinite micro-adjustment.
- Actual historical extraction: apple/native, button/button_palm_mesh_v1, chair/native completed. Plate returns unsupported_entry_pattern because an earlier stable interval is followed by further travel before its declared operation entry.
- Geometric contact is the available support evidence in these archives; no fabricated force measurements. Matching stopping calibration is required before automatic physical execution; none was created in this software-only work.
- Changes retained on the existing linked feature worktree. Unrelated untracked drafts preserved; no merge or push performed.
