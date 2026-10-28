# Replay Parking and Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve physical apple acquisition after navigation by reusing measured parking and recovery, with all actions recorded as SONIC tokens.
**Architecture:** Optional parking/recovery helper consumed by replay_navigation; reuse legacy math/IK helpers unchanged and preserve the simple executor as the baseline.
**Tech Stack:** Python, NumPy/SciPy, MuJoCo3.3.6, SONIC mode1 CPU.
**Spec:** docs/superpowers/specs/2026-10-27-replay-parking-retry-design.md

## Global Constraints

- Original far scene/history; SONIC1.0 CPU, ORT1.23.2, TRT10.13.0.35 through scripts/humano_python.sh.
- No images/API calls, physical state resets, original-token prefix or decoder overrides.
- Freeze complete experiment.json before physics; new timestamped month/day experiment directories.
- No calibration/contact/corridor vetoes; finite correction/recovery windows record failure and continue where possible.
- Keep failed cases, all200Hz source hand commands and50Hz body tokens; separate task/contact quality and trajectory retargeting from strategy generation.

## Review Focus

- Deceleration must not recreate the old indefinite low-speed gait stall.
- A corrected standing stance must not be overwritten by replay legs during handoff.
- Recovery must preserve current physical root/legs/object state and not silently restart from the source scene.
- Per-attempt outcomes must distinguish real in-run retries from independent experiment launches.
- Raw desired wrist targets, translated retry targets and encoder feedback targets must remain distinguishable.

### Task 1: Measured parking and retained stance

**Files:** new augmentation/replay_parking.py; modify augmentation/replay_navigation.py and direct_grasp.py; new augmentation/tests/test_replay_parking.py.
**Interfaces:** ReplayParking(adapter,cfg,route,entry).navigation_command(), .begin(), .update(); output measured stage/velocity/hold poses and completion; reuse original ReplayProgram.

- [x] Write regression tests for deceleration, waypoint progression, upright blended handoff, soft settle expiry and failed-correction continuation; run RED.
- [x] Implement optional helper and retain standing legs while source arm/replay targets execute.
- [x] Bind actual source/config/interfaces in preparation, run GREEN and commit.

### Task 2: Continuous recovery and reanchored replay

**Files:** replay_parking.py, replay_navigation.py, direct_grasp.py, test_replay_parking.py.
**Interfaces:** ReplayRecovery preserves current root/legs and returns timed open/upper-body restoration; reanchored wrist translation from measured apple pose; explicit retry_index.

- [x] Write recovery/retry tests for current-state preservation, changed object translation, finite completion and original hand index reset; run RED.
- [x] Implement bounded retries without reset or additional failure vetoes; log per-attempt native/retained outcomes and targets.
- [x] Run targeted tests and tracked test suite, commit.

### Task 3: Physics validation and review

**Files:** new immutable experiment records plus final docs/report.

- [x] Freeze and physically runY+5cm; diagnose actual parking/acquisition outcomes and make evidence-supported fixes.
- [x] Freeze a declared lateral validation batch; record first-attempt and final native success, retained grasp, parking error, collision peaks, retries and all timings.
- [x] Independently audit native checker, token decoder identity and exact hand timing. Run full relevant tests and a fresh code review.
- [x] Report actual outcomes and remaining limits; update usage and commit.

Final:24 frozen physical cases; baseline6/6, optional parking/retry5/6 (two rescues), integral3/6 rejected; recommended original-navigation+retry3/3 at prospective offsets. Full source body motion retained after measured handoff.23817 tokens independently bit-exact and95268 hand steps exact. Report: experiments/2026-10/28/20261028_0040_apple_replay_retry_holdout/REPORT.md.
