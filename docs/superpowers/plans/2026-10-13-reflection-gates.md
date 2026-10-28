# Reflection gates and apple retry implementation plan

> Execute inline using executing-plans and test-driven-development. User approved the bounded design and continuation; no new approval or Git commit is needed.

**Goal:** Bind retries to measured failure evidence and test mode1 apple grasping in the unchanged large-shift scene.

**Architecture:** Preserve the recovery/generation/execution/acceptance order. Add stage diagnostics and an opt-in required reflection contract to the existing core. Publish a new task skill and run a separate experiment directory.

**Tech stack:** Python, pytest, HUMANO MuJoCo, published SONIC ONNX models.

**Spec:** The approved in-chat four-part design: measured diagnosis, evidence-bound retry reflection, prioritized tracking/contact/load-transfer branches, versioned hypothesis outcomes.

## Constraints

- Preserve existing edits, all historical runs and skills, original renderer and HUMANO checkouts; no Git commit.
- Keep apple translation +0.25m/+0.30m, physics, controller weights/gains and native acceptance unchanged.
- New generation: at most 10 attempts, 30 simulated seconds and 12 stage feedback calls per attempt; optimization remains disabled.
- Default mode1; reject mode0/native-prefix actions in the new strict generation task, including reused prefixes.
- Reflect after each failed attempt, including the last; require the prior reflection before any new physical attempt.
- Preserve backward compatibility for tasks without policy_checks.

## Review focus

Wrong prior-result hash, invented numeric facts, missing measurements, hidden mode switches in reused prefixes, and resume with a missing reflection must fail closed. Diagnostics and reflections cannot assert physical success.

## Task 1: Runtime contracts

- [x] Write failing tests in tests/test_diagnostics.py and tests/test_reflection.py for numeric diagnosis, missing values, evidence binding, retry-before-reset, mode enforcement and frozen-skill reflection.
- [x] Run these tests and record RED.
- [x] Create src/cap_sim/diagnostics.py; integrate diagnostics and optional reflect(request) in core.py, planners.py and task_skills.py. Add measured controller/contact summaries in native backend without changing commands.
- [x] Run focused tests, then the full suite; record results here.

## Task 2: Skill and experiment

- [x] Publish next immutable apple skill: tracking first, then contact, then load transfer; one primary hypothesis with expected effect/rejection condition.
- [x] Prepare separate experiments/20261013_apple_mode1_reflection runner with strict mode1, immutable JSON mailbox, and source/recovery evidence audit.
- [x] Execute bounded retries; each failure reflection precedes the next reset; publish revisions only between attempts when evidence warrants changes.
- [x] Save results, times, ego/third-person videos and a final report; preserve success before considering any optional optimization.

## Progress / rulings

- Ruling: use existing experimental cap-sim branch with its uncommitted features; do not create a second checkout that would omit the existing skill/contact changes.
- Ruling: strict checks are opt-in task configuration so historical task contracts remain replayable.
- Ruling: reflection is a separate non-physical model request, not an extra stage feedback call or physical retry.

- Task1 complete: initial9 RED, focused9 GREEN; measured-mode RED→GREEN, malformed-input2 RED→GREEN, mixed-skill RED→GREEN. Full suite71 passed. Independent review P1/P2 fixed.
- Reference recovery independently passed again; source was executed using original modes, generation remains strictmode1.

- Task2 complete: reference recovery passed13.4sim seconds; strictmode1 generation0/10,263.525sim seconds,5009.662wall seconds including reflection. All10 evidence/order/mode audits passed;202 resource hashes unchanged. No optimization or Git commit. Final task skillv10 records interface mismatch and geometric-screen limits. Full dual-view video and REPORT preserved.
