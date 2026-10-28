# Fast measured-replay leg retries

Approved continuation: validate checkpoint rollback then reuse calibration and optimize API feedback choices. Isolated module/skill; existing routes unchanged.

- [x] Test exact hand-prefix assembly, fixed-checkpoint validation and suffix-vs-full equality gate.
- [x] Restore frame352 checkpoint from prior full feedback run; no prior leg integral at this boundary. Explicitly disallow later checkpoint frames until local integral state is checkpointed.
- [x] First reexecute unchanged prior parameters from frame352; demand exact full state/next_state/commands/tokens/hands vs old full rollout. If mismatch stop, no efficiency or success claim.
- [x] Reuse frozen7-probe response model; compact API context identifies knee undercompensation, remaining horizon and ankle saturation. Max3 new API feedback candidates, tail2..3s, otherwise existing bounds.
- [x] No rendering in candidate loop. Render selected best failure/first successful candidate, fresh restart verification and independent saved-token proof only as applicable.
- [x] A successful branch must be independently restarted from the original initial state, then saved token+hands replayed. Branch artifacts retain inherited prefix lineage and actual suffix work, never labelled fully new.

Ruling: rollback only at the 352-frame mode1 boundary, so offset integral is exactly zero; source pre-control history restored by existing retry_state. Merge source prefix hands with newly generated suffix in new artifacts only. Old experiments remain immutable.

Full reference and original native acceptance unchanged. Original successful measured replay shift0 vs test+5mm retained. Allow fixed tail2..3s under new per-case contract to avoid unnecessary deadline clipping. Separate API, initialization, suffix physics, verification, rendering and total wall times. Maximum physics:1 equivalence branch +3 candidate branches +1 full success validation +1 independent token replay. Model calibration reuse is distinct from rollback savings.

Validated 2026-10-25: experiment `20261025_1540_chairsit_fast_rollback_feedback`. Exact rollback gate passed; second API candidate succeeded; fresh restart and independent token/hands replay succeeded with exact arrays. Candidate physics2.379s vs full6.491s; API279.584s remains dominant. Native hold45/30ticks; final hip normal6.28N does not establish full weight support.274tests pass. Only frame352 rollback supported.
