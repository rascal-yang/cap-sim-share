# Simple replay navigation implementation plan

**Goal:** Complete the actual apple task through navigation and replay without extra runtime vetoes.
**Spec:** docs/superpowers/specs/2026-10-27-simple-replay-navigation.md

- [x] Remove corridor abort and route-local timeout; regression tests cover continued control outside corridor.
- [x] Implement timed measured transition/replay/hold and retain original far initial/history in preparation.542 tests passed14.30s.
- [x] Freeze and execute Y+5cm from original far state. Four physical attempts across two owning experiment records; final native task succeeds.
- [x] Replace premature braking with direct0.25m/s goal tracking; use recorded preencoder pose feedback (position2, orientation1) while preserving replay/fingers.
- [x] Independent native checker passes; all666 tokens reproduce decoder targets bit-exactly; all2664 hand commands match source.548 tests pass in14.30s.
- [x] Report: experiments/2026-10/27/20261027_2138_apple_navigation_pose_feedback/REPORT.md. One scene/offset only; table contacts retained as diagnostics.

Final implementation and invocation: docs/augmentation/simple-replay-navigation.md. The spec above preserves the initial proposal; the measured low-speed gait stall required the direct goal controller documented in the final implementation.
