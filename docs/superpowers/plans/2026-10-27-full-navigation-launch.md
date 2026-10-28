# Full navigation launch repair implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; steps use checkboxes.

**Goal:** Execute the current complete apple navigation/grasp workflow with truthful diagnostic admission and immutable source evidence.
**Architecture:** Explicit request admission plus archived-source recovery and format-compatible physical verification, using the same native runner/controllers.
**Tech Stack:** Python, MuJoCo, pinned SONIC1 CPU runtime.
**Spec:** docs/superpowers/specs/2026-10-27-full-navigation-launch-design.md

## Global Constraints
- No route/controller/contact retuning, no images/rendering, no original token/mode0 fallback or state reset.
- Default calibrated mode still rejects bad evidence; diagnostic mode is explicit and never claims calibration.
- Complete frozen experiment.json, actual loaded CPU provider and model/source hashes before physics.
- Preserve historical contracts/assets; tokens plus separate hands cover actual executed steps.

## Review Focus
- Diagnostic permission cannot mask malformed or tampered evidence.
- Changed physics/assets must not be recovered via a Python snapshot exception.
- Legacy format support must compare actual PD and timestep, not skip them.
- New results distinguish preflight failures, physical stops, task success and diagnostic calibration status.

### Task1: Admission and source/physics compatibility
- [x] RED tests for explicit diagnostic/default strictness, archive-only source recovery, legacy PD/physics mismatches.
- [x] GREEN minimal helpers and public service integration; whole applicable suite.
- [x] Independent review complete; malformed/wrong-scope model integrity fixed RED→GREEN.539 tests passed13.53s.

### Task2: Actual full workflow
- [x] New1945 study froze1951 workflow and executed288 actual steps; explicit uncalibrated diagnostic, unchanged policy.
- [x] Legacy missing inputs and overwritten SONIC seed fixed with RED→GREEN regression; zero-step preparation failures retained.
- [x] Separate-process1956 repeat is bit-exact: both navigation_corridor_violation at1.44s,26.485mm cross-track vs25mm corridor.72tokens/288steps per case independently decoded bit-exact;540 tests pass. Navigation-to-grasp physical success remains unachieved; no strategy/gate retuning.
