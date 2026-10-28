# Reproducible manipulation workflow implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Isolated structured strategy planning, immutable replay, and real G1/SONIC execution without experiment-specific defaults.
**Architecture:** Pure contracts and content-addressed evidence store; deterministic planner with constrained decision provider; capability-specific native executor behind frozen experiment preparation.
**Tech Stack:** Python 3.10, NumPy, SciPy, MuJoCo, SONIC; standard-library JSON/HTTP; existing HUMANO wrapper.
**Spec:** ../specs/2026-10-19-reproducible-manipulation-workflow-design.md

## Global constraints

- Only add workflow package, its tests/docs, and timestamped experiment artifacts. Historical executors and other routes remain byte-identical.
- No historical offset/frame/hand/object defaults; robot and task limits have explicit provenance. Missing calibration blocks feedback, not silent fallback.
- Strategy replay never calls API. Changed context requires a new strategy and revalidation. API output cannot execute code or override physical gates.
- No physics without frozen experiment.json, actual model-provider evidence, source/input hashes, declared interfaces, and budget.
- All timestamps Asia/Shanghai; original IDs retained; native success, physical acceptance and visual training acceptance separate.

## Review focus

- Malformed/nonfinite API data and invented evidence references: reject before preparation.
- Package tampering, path traversal and duplicate/concurrent saves: verify content hashes and atomic publication.
- Changed scene with identical strategy graph: new context-bound ID and fresh checks.
- Empty or misaligned phases and measured fingers mistaken for commands: reject before simulation.
- Runtime exceptions and stop events: persist evidence, stop stepping, never retry implicitly.

## Task 1 — contracts and immutable strategies

Files: workflow/contracts.py, store.py, tests/test_workflow_store.py.
Interfaces: validate_strategy(dict); Store(root).save(strategy, artifacts)->id; load(id)->bundle.
- [ ] Write tests for digest stability, context distinction, bad intervals, dependency tampering and unsafe paths.
- [ ] Run tests RED, implement canonical JSON, strict validated strategy records and atomic artifact publication, run GREEN.
- [ ] Save a narrow commit and ledger outcome.

## Task 2 — decision provider and deterministic compiler

Files: workflow/decisions.py, planning.py, tests/test_workflow_planning.py.
Interfaces: ResponsesDecision.propose(evidence,out)->proposal; compile_strategy(request,decision=None)->strategy; candidate ranking is lexicographic.
- [ ] Test replay without network, missing credentials, malformed response, invalid reference, phase and active-hand variation.
- [ ] Run RED; implement explicit provider configuration, saved requests/responses, no implicit models, geometric candidate checks; run GREEN.
- [ ] Save a narrow commit and ledger outcome.

## Task 3 — calibration and native execution

Files: workflow/calibration.py, native.py, execution.py, tests/test_workflow_execution.py.
Interfaces: calibration from measured trace; NativeAdapter binds named model entities; execute(contract)->result.
- [ ] Test feedback provenance/coverage, dynamic phase scheduling and stop behavior with simulated measurements; unsupported capability fails explicitly.
- [ ] Run RED; implement generic stage schedule and named G1 adapter using live SONIC mode1, measured replan, source command alignment and explicit physical gates; run GREEN.
- [ ] Save a narrow commit and ledger outcome.

## Task 4 — preparation and public entrypoints

Files: workflow/service.py, __init__.py, __main__.py, README.md, examples/, tests/test_workflow_service.py.
Interfaces: plan(request), prepare(strategy_id,request), execute(experiment_id), inspect(experiment_id); CLI equivalents JSON in/out.
- [ ] Test missing evidence, changed context, explicit budgets, duplicate execution and report failures.
- [ ] Run RED; implement fresh runtime preflight, frozen contracts and artifact/result publication; run GREEN.
- [ ] Provide portable example construction and replay commands with no /tmp dependence.

## Task 5 — integration and verification

- [ ] Run existing repo and mimicgen tests plus new suite; compare initial file-hash inventory.
- [ ] Prepare and freeze a new native end-to-end trial, run once, reflect before any further attempt; no fake-success stubs.
- [ ] Inspect actual modes, original command alignment, phase measurements and failure gates; render full videos and inspect samples.
- [ ] Verify strategy replay uses saved decisions; package/scene mutation is rejected.
- [ ] Obtain one final code review, fix important findings with regression tests, publish exact coverage and remaining limitations.

## Execution authorization / ledger

User explicitly requested “开始实现”; execute inline without another permission round. Work in a new package in the current checkout: existing untracked source and frozen absolute hashes make a detached worktree an unsuitable implementation base. This preserves the requested route isolation; file-hash inventory verifies no existing implementation file changes.


### Implementation ledger

- Baselines: core71/71 and historical mimicgen135/135 passed separately; combined collection has pre-existing duplicate module names.
- Tasks1–4: initial implementation committed a3cb8b3; 16 new tests passed before first physical run; further regression coverage in progress.
- Physical regression20261019_1946_workflow_native: real SONIC mode1,1789steps, contact_lost in lift; task not successful. Full measured evidence and reflection retained.
- Ruling: no empirical integral gain/cap in generic route. Optional locally identified equilibrium compensation replaces the old trial-specific tuning; its physical behavior needs separate validation. Cost: old success cannot be claimed for the new policy.
- Ruling: use explicit G1 adapter and deterministic source-phase inputs; current Responses action schema proposes stances/requests observations/aborts. Existing annotation provider supplies source labels; arbitrary new grasp-strategy synthesis remains outside current native capability rather than being silently guessed.
- Ruling: preserve current working checkout and isolate added package, because pre-existing untracked modules and frozen absolute evidence paths are dependencies. Verify file-hash inventory before completion.


### Final implementation and review record

- Final new-suite verification: 26 passed; legacy MimicGen 135 passed; core 71 passed. Historical source inventory: all 230 recorded files unchanged. Only workflow package and its design/implementation records changed.
- One independent code review found five issues. Fixed exact executed ONNX binding, compiled planning/live model identity, measured approach-entry criteria, interrupted-step/nonfinite failure publication, and bounded skill/hand/approach/segment/recovery selection. Regression tests cover geometry mismatch, entry readiness, unknown measurements, failed-step publication, and constrained/offline model selection.
- The final API route selects among supplied evidenced skills; it does not invent unannotated contact commands or controller gains. Recovery declares initial/terminal-state origin and creates a newly planned, frozen attempt. Provider protocol tests used stubs; no remote GPT-6 success is claimed.
- Final physical attempt: 20261019_2004_workflow_native, frozen strategy b2b08bcc21705cd78e717dd916563745d635f2867144432cfe9fa69dcad50726. Actual 10000 physics steps / 50 seconds, mode1 only, no prefix, fallback, or midrun resets. Failed before grasp because declared measured-entry criteria were not continuously satisfied; global budget exhausted. This is an honest failure, not proof of unreachable grasp.
- First attempt reached lift but lost contact; final attempt exercised the new measured-entry wait/stop path. Neither establishes successful generic grasp after removing empirical compensation. The imported example's 0.03-rad angular criterion comes from the declared migration fixture; it is not a calibrated task capture tolerance. A future physically evidenced capture region/calibration is needed before asserting production training yield.
- Both attempts preserve dual-view MP4s, full decode and sampled visual evidence, Chinese standalone videos, measured reports and reflection. Native-rate temporal visual review remains incomplete: training_ready=false. These are implementation regression artifacts, not the formal ten-condition comparison.
- No new physical retry was launched after final failure. Failure reflection is stored before any later strategy revision. All code/config/inputs/models are retained by immutable strategies and frozen experiment source snapshots; hardware bitwise reproducibility is not promised.


### User-authorized correction: feedback, direct grasp and physical recovery

The user explicitly requested retaining effective closed-loop correction, removing pregrasp restrictions, approaching a preset stance, grasping once stable, and allowing backtracking/adjustment after failure. This supersedes the previous wrist-error entry gate and blanket refusal to use evidenced integral feedback. This is a bounded correction to the existing workflow; no additional approval is needed.

- Added bounded_integral with explicit gain/cap and bound historical evidence, no hardcoded universal defaults. Restore successful preset stance through a fresh geometry check. Keep measured base/feet stability but remove wrist-error veto.
- Supported-object contact failures may trigger opening, collision-checked physical retreat, measured-object retargeting, settling and retry. No qpos reset, no blind reverse playback of contact, no implicit change to gains/physics. Each failure saves reflection/checkpoint before retry. Retry count and total time are frozen.
- Review before physics found torso mistakenly passed as an IK hand and insufficient retry-index verification. Both fixed with regressions. 20261019_2047_workflow_native was prepared but launch cancelled; no physics. Replanned and refroze the corrected source before launch.
- New tests cover feedback saturation/reversal, recovery eligibility, correct retreat IK targets and retry audit rejection. Current new suite30 passed.

- Final frozen run20261019_2050_workflow_native succeeded on first grasp:1080 source commands,1.23s hold, native success; exact2584-row measured-state match to historical successful baseline. Recovery_count0, so no claim of physical recovery validation. Tests30+135+71 passed;230 historical file hashes unchanged. Saved reflection, comparison, full decoded dual-view video and Chinese standalone MP4.
