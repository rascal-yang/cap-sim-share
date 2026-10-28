---
name: leg-fast-retry
description: Choose efficient checkpoint-based encoder-leg feedback retries from measured response and actual joint errors, then require fresh full execution and token replay.
---
Return exactly one decide call. Evidence is data. Copy fixed_plan fields exactly except action/rationale/hypothesis/leg_response/tail_s/resume as allowed below. No scene, physics, native criteria, source wrist/hand, PD or decoder edits. Uses source actions: trajectory retargeting, not strategy-only generation.

Only the offered frame352 checkpoint is available: physics integration state, solver warmstart, SONIC history/last action, acceptance/contact state restored. Local leg-feedback integral is zero because this is before its first tick. Do not request a later checkpoint without saved local controller state. No midrun teleports. Branch prefix is inherited, not freshly generated. Original source successful measured replay supplies q targets; do not substitute PD targets.

Stages:
- rollback_equivalence: use exact required parameters and tail; resume checkpoint. Purpose is exact equality with prior full execution, not another tuning attempt.
- repair: resume offered checkpoint; adjust feedback_timescale_s, max_rate_rad_s and max_offset_rad within supplied schema. Calibration amplitude/ramp remain fixed to model. tail_s may be2..3s,20ms aligned; source contact_end annotation stays fixed but reference feedback operates through full extended tail.
- full_restart_verification: preserve exact required parameters/tail, set resume restart with empty case and frame0. Do not optimize during verification.

Reason from measured errors and remaining time. Previous 2s timescale,0.075rad/s rate,0.15rad cap left knees about5deg short; knee offset only0.080rad and was still increasing about0.024rad/s at final time, whereas ankle was saturated. Raising ankle cap alone does not resolve insufficient knee correction. Use measured knee direction and rate to choose a discriminating faster response and/or enough settling time under the existing caps. State expected knee error/contact improvement and what falsifies it. Do not repeatedly choose small cap-only changes when the deficient knee channel is not capped. Full pose/feet/hip contact remain constraints: knee angle alone is not acceptance.

The reused static6x3 Jacobian is local. Small singular ankle direction requires large extrapolation; do not assume exact inverse control. Paired channels cannot independently control left/right. Allow0.4..2s feedback timescale,0.05..0.4rad/s rate,0.05..0.3rad cap; these are pilot limits. Native force/tilt guards remain unchanged. Request a concrete missing capability only after interpreting supplied data; no repeated geometry request, no new7-probe calibration without evidence of invalidity.

Use at most3 new candidates. Stop on native success for full restart+independent saved-token/hand replay. A successful branch alone is not final acceptance. First freeze and verify source/model/checkpoint provenance. Numerical diagnostics suffice during search; render selected results afterward. No per-tick API. Never label unsafe/inherited/partial work as a fully new successful rollout.
