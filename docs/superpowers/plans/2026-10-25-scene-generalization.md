# Four-task object-position generalization

User approved 4 tasks × 10 frozen positions, equal-budget restart/checkpoint retry, real API proposals, independent full execution and token+hands verification for successful conditions.

Implementation is isolated in generalization_* modules and scene_generalization skill. Existing routes and historical records remain untouched. Reuse authenticated rollback_plan compiler, complete-state checkpoint schema, identity body PD and audited button palm mesh acceptance.

Positions relative to the successful object X+5mm scene: center; cardinal ±2cm; four diagonal ±2cm; X+4cm. Robot initial state and table fixed. Only target moves. Original token approach prefix remains unchanged. This is object-relative trajectory retargeting, with API candidate planning, not new skill generation.

Each scene: API queue12 (candidate0 frozen successful policy); common seed; same-scene boundary exactness; first task position also active-controller exactness. Full planning+seed cost charged to both120s budgets. Same ranked queue, alternating arm order, stop at first in-budget success; late completed attempts excluded. Explicit restart fallback if seed terminates before checkpoint. No cross-scene checkpoint reuse. Verify every unique in-budget winner and any successful seed (including deadline misses) using full execution and independent token+hands. Validation and rendering outside operational budget, separately reported.

Ruling: stop first success instead of old throughput run — this measures generalization and time-to-first-success; it cannot establish maximum successful samples per minute.
Ruling: Plate reuses apple named controller profile only — same paired Cartesian interface, actual Plate dataset/native acceptance retained. Risk: historical profile label could confuse readers; report task_label separately.
Ruling: button fixed-root scene changes checked across all model arrays with only declared target body_pos delta permitted, plus initial state/physics checks — previous exact scene hash would reject intentional perturbations.

Tasks: implementation; full tests; fresh review; serial physics40 conditions; full/token proofs; render representative outcomes shallow in root; aggregate counts/times/tokens and audit; commit.

Validation: 283 tests passed before launch, 2026-10-25. Review pending. No physics launched at this point.

Fresh review found five issues, all addressed before launch: apply scene invariant gate to all tasks; include model numeric scalars and physics options; compare every successful arm trace to fresh full execution even for a shared candidate; verify over-budget successes without counting them in budget success; expose verified_success separately from provisional operational acceptance. Scene/options negative tests added. Final suite: 285 tests passed. Validation evidence saved per case before physics. Ruling: retain both provisional and verified rates so failed replay cannot masquerade as usable generated data.

Completed:40 physical conditions;297 execution records audited;74 matching candidate trajectories exact;26 successful positions each full+token+hands verified;8 shallow Chinese-captioned videos decoded and sparse visual review recorded. Final suite288 passed. Both arms plate8,chair8,button4 approved/native0,apple6 out of10 each. No task-condition success improvement demonstrated.

Ruling: network errors require explicit separate recovery roots rather than overwriting incomplete evidence; operational budgets restart after connectivity recovery, costs remain in combined accounting — this does not support an outage-inclusive120s service guarantee.
Ruling: two API-invalid apple plans receive a versioned numeric skill/validator, with same physical interfaces/limits and no parameter clipping — comparisons across positions do not use a completely identical prompt revision.
Initial preflight included an erroneous process-address model comparison; corrected before physics and negative/unit plus12 saved-model translation checks passed. An aborted completion setup used the wrong outer directory; moved intact under experiments before any physics, with RELOCATION.json. No frozen evidence rewritten.
Metadata limitation: common evidence wrapper retains rollback_benchmark.run label in entrypoint; actual new runner/runtime/source hashes are documented in FINAL_REPORT. Body control/session/scene evidence checks passed. No other generation route modified. User-owned untracked work-summary documents excluded from commit.
