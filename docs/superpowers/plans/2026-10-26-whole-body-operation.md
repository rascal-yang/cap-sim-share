# Whole-body operation implementation plan

> Execute inline using executing-plans; user requested implementation of the discussed architecture.

**Goal:** Connect calibrated replay navigation to continuous full-body operation with measured phase gates.
**Architecture:** Shared deterministic phase clock plus native full-body reference bridge. Preserve geometry gates and navigation-only compatibility.
**Tech Stack:** Python, NumPy, MuJoCo FK, SONIC mode1.
**Spec:** docs/superpowers/specs/2026-10-26-whole-body-operation-design.md

## Constraints and review focus

No live qpos writes or source-token prefix; no task-specific thresholds; freeze before physics. Test premature finger closure, nonfinite observations, phase timeout, operation range/source alignment, navigation-only acceptance isolation. Physics evidence follows workspace AGENTS.md.

- [x] Add tests then a shared operation clock: explicit phases, continuous geometric/contact conditions, bounded wait, previous-frame hold.
- [x] Add full-body reference bridge: NativeEvaluator transforms and segment checks, FK on copies, mode1 legs/wrists from same reference.
- [x] Enable opt-in in profile/integration/runner and freeze actual interfaces; native task success separate from completion.
- [x] Run targeted/default/workflow suites, review change, document physical calibration limits.


## Execution ledger

- Existing isolated worktree reused; unrelated untracked files preserved.
- RED/GREEN: missing phase clock; geometry closure bypass; excessive contact force; missing source/body bindings. Added 18 tests, including actual archived apple failure and no-physics MuJoCo bridge at 200/50 Hz.
- Review: fresh whole-change reviewer found cadence, closure, unsupported clearance, contact-force ceiling, late static checks and terminal-output acknowledgement issues. All addressed; final review found no blocking issue.
- Ruling: reject requires_clearance=true in this new path rather than silently treating a lift-specific clearance observer as generic. Native task validation remains authoritative, including native-only button acceptance.
- Ruling: do not launch physics with failed geometry plans or missing scoped calibration. No dynamics/success-rate claim; no fallback.
- Verification: default suite 108 passed; workflow suite 111 passed; git diff --check clean. Source transforms only FK copies. No new physical experiment.
