# Replay-first planning and SONIC records implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Steps use checkboxes.

**Goal:** Minimize replay changes and sequential stance search cost while making SONIC token coverage explicit for every executed action.
**Architecture:** Explicit replay_first candidate generator and reference-preserving solver; shared action bundle exporter consumed by direct and continuous runners.
**Tech Stack:** Python, NumPy/MuJoCo/SciPy, SONIC mode1 CPU.
**Spec:** docs/superpowers/specs/2026-10-27-replay-first-token-design.md

## Global Constraints
- Preserve frozen replay and old grid as explicit mode; no image inputs/rendering.
- TRT10.13.0.35, ORT1.23.2, SONIC1.0 CPU; no token/mode0 fallback or live state writes.
- New physics requires frozen full experiment.json with source/implementation/skill/runtime evidence and actual timestamps.
- Tokens encode body; hands remain separately aligned. Missing coverage cannot be accepted.

## Review Focus
- First successful nominal candidate must not trigger later expensive evaluations.
- Bounded search failure is inconclusive, not proof of task unreachability.
- Reused non-keyframe source samples must remain exact; small-change limits apply after interpolation too.
- Token/controller records can contain an attempted tick without a completed physics step; export only executed coverage and identify unused records.
- Hand commands can change between token ticks; preserve every physical sample rather than falsely holding fingers at50Hz.

### Task 1: Replay-first sequential planning
- [x] RED: nominal ordering/fallback, early exit, exact replay reuse and correction-limit rejection.
- [x] GREEN: lateral.py/lateral_ik.py plus configs/navigation/apple_lateral.json; keep legacy tests.
- [x] Real-source Y+5cm offline plan completed in4.648s, one nominal stance; exact translated replay, zero joint correction.

### Task 2: Complete action package
- [x] RED: coverage, timestamps, hand changes, truncated execution, missing token/target mismatch.
- [x] GREEN: workflow/sonic_record.py, direct_grasp/runner snapshots and exports.
- [x] Saved real trace exported separately:1041 physical steps /261 tokens;528 complete-suite tests passed in13.13s.
- [x] Review complete; missing executed-tail coverage fixed RED→GREEN. Frozen5e26fda physical case20261027_1900:1041 steps/261tokens, native lift0.113509m, clean quality fails14.047N/1.130s. Independent decoder check261 targets bit exact; no navigation execution. Report20261027_1856_replay_first_sonic_tokens.
