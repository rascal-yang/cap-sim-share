# Navigation checkpoint implementation and validation

Spec: docs/superpowers/specs/2026-10-28-navigation-checkpoint.md

## Task 1: checkpoint state and entry-point integration
Write failing tests for full integration/SONIC planner state roundtrip, branch policy boundary validation, and inherited prefix splice continuity. Implement optional compact phase checkpoints and one-time restore, include source contracts/evidence, complete SONIC/hand logs. Verify existing replay and new tests.

## Task 2: physical matched validation
Freeze a precision-parking seed; compare unchanged navigation-start and parked suffix restoration. Run matched changed-feedback candidates using restart/parked rollback at representative lateral positions. Audit state/time/tokens/hands equality and contacts. Independently replay a selected successful token branch from its original initial state. Report actual times, storage, source hashes and limitations. Run relevant tracked suite. Commit code and docs, leave unrelated files alone.
