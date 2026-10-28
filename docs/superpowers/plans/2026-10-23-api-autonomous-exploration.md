# API autonomous exploration implementation plan

Goal: implement the user-approved replay + annotation -> autonomous evidence queries -> API experiments/retries loop.
Architecture: a bounded structured Python evidence interface, invoked through persisted API decisions, wraps criteria/plan/review decisions. Source references are replay and annotation only; current task acceptance and own newly generated evidence remain available. No supervisor action tuning. Existing frozen physics and checkpoint execution remain authoritative.

- [x] Add tests for request bounds, case isolation, diagnostics feedback and exploration exhaustion.
- [x] Add read-only artifact catalog and bounded slices with explicit alignment, hashes and missing-data errors. No state integration or writes to source cases.
- [x] Wrap audit/plan/review with API-selected queries; needs_tool re-enters exploration. Record tool errors rather than fabricate data. New diagnostic rollouts use the existing validated plan executor and physical budget.
- [x] Remove failed_case and prior audit/review dependency from initial context; pass full own-attempt history and checkpoint catalog.
- [x] Update skill and route docs, run focused and full augmentation tests, and read-only real-artifact smoke test. No new physics is needed for code validation; do not imply measured task recovery.

Constraints: native acceptance unchanged; no arbitrary Python execution; API original decisions/provenance remain unchanged; finite wall, physical and query budgets; historical evidence immutable. Protocol capability limits are disclosed, not task policies.
