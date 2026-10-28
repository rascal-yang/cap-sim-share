# Contact retry: budget and API-selected rollback

Scope: isolated contact-review route only. The user authorized longer retry budgets and API-selected rollback on 2026-10-23.

Defaults: first phase 900 wall seconds; each retry 1200 seconds; three physical attempts, total 3300 seconds. Explicit task caps remain honored. API request timeout/five retries/credential rotation remain unchanged. A retry gets a fresh phase deadline capped by the task deadline; elapsed API failures remain accounted.

Every new contact rollout records checkpoints immediately before each 50 Hz control tick: MuJoCo integration state (including warm start and control), SONIC history/last action/tick, native task counters and initial references, contact integrator and previous target. No legacy qpos/qvel-only reconstruction is treated as a checkpoint.

API plan selects restart or a saved case plus pre-control frame and rationale. Only completed same-task attempts offered in context are allowed; model, scene, source and criteria must match. Restore once at branch initialization. Reuse the immutable ancestor prefix up to but excluding the selected tick; simulate the suffix under the new plan. Retain the full lineage and separate inherited steps from newly executed steps. A stitched branch may be accepted only after native verification of the complete branch and quality review; it is never labelled a fully new continuous physical rollout.

The retry planner receives the full preceding API review and compact attempt history. No automatic choice of rollback point, no criterion relaxation, no silent state approximation. Each branch consumes one physical attempt. needs_tool now re-enters the API-selected evidence query loop; see autonomous-exploration.md. Unsupported capabilities remain explicit and are never fabricated.

Validation: checkpoint round-trip without physics; invalid/unbound checkpoint and budget tests; same-control continuation equivalence needs a frozen physics test before claiming physical determinism. Old studies remain untouched.
