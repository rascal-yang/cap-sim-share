# HUMANO G1 + Dex3 adaptation

`repro.humano.PROFILE` is `humano_g1_dex3_world_delta_10hz_v1`.
This is an explicit adaptation, not the original 14-dimensional, 25 Hz robot
interface. Each row has 26 required numeric action values (two 6D wrist
increments and two 7-joint hand targets), plus a stage and optional channels.
Each backend action lasts exactly 0.1 seconds (10 Hz targets), satisfying the
existing backend's 0.1–8 second duration and 5 ms alignment constraints.

## Shared TRAIN/LIVE API

```python
from repro.humano import ACTION_SCHEMA, PROFILE, REQUIRED_SONIC_MODE, encode_actions

# obs is the current backend.observe() result.
# Limits must come from the loaded model, in actual hand actuator order.
limits = {
    'left': [j['range'] for j in obs['joint_interface']['left_hand']],
    'right': [j['range'] for j in obs['joint_interface']['right_hand']],
}
rows = [{
    'stage': 'approach',
    'left_delta': [0, 0, 0, 0, 0, 0],
    'right_delta': [0.01, 0, 0, 0, 0, 0],
    'left_fingers': obs['fingers'][0],
    'right_fingers': obs['fingers'][1],
}]
actions = encode_actions(rows, obs, limits, horizon=32)
# Only after all rows validate may a gated executor pass each action to
# backend.execute(action, remaining_seconds).
```

`ACTION_SCHEMA` is the public JSON Schema for each input row. Both TRAIN and
LIVE should present that same schema and use `encode_actions` as the final
validator. JSON Schema alone cannot enforce norm, finite-value, model-limit,
or accumulated-state constraints. A chunk must contain 1 through `horizon`
rows; `horizon` is an integer in 1..32. No inputs are mutated. Errors raise
`ValueError`, and no partial output or execution is exposed.

Both wrists require world-frame `[dx,dy,dz,rx,ry,rz]`: translation norm at most
0.03 m and rotation-vector norm at most 0.15 rad per row. These conservative
increment bounds retain the shared upstream values; the different cadence
does not imply equivalent speeds. Absolute wrist targets accumulate from the
current observation throughout the chunk. World rotation increments multiply
on the left. Output quaternions use normalized wxyz order. Observed and optional
input quaternions must be unit within 0.05 before normalization. Wrist targets
must remain within 1.5 m of the observed root; the backend rechecks the actual
root during execution, since the root can move after chunk validation.

`left_fingers` and `right_fingers` contain seven absolute joint angles in radians.
The caller must supply real finite lower/upper limits from the active model in
the same actuated-joint order. Missing, reversed, or zero-width limits fail
closed. There is no guessed hand range or scalar-gripper-to-Dex3 conversion.

Optional channels are `velocity_xy` (world m/s, norm <= 0.5, default zero),
`facing_yaw` (absolute world radians in [-pi, pi]; omission lets the backend
use measured heading), `pelvis_height_m` ([0.4, 0.85], default 0.74), and
`torso_quaternion_wxyz` (absolute world orientation; omitted rows retain the
last torso target, initially the observed torso). Supported stage names match
the existing backend. Unknown fields, including body keyframes, tokens, replay
prefixes, gripper scalars, and caller-selected duration, are rejected.
Booleans, nonfinite numbers and numeric strings are rejected in numeric fields.

## Runtime evidence and limitations

`REQUIRED_SONIC_MODE = 1` is an execution requirement, not proof of execution.
Generated actions omit full-body references; the existing `native_sonic.Sonic`
selects encoder mode 1 on that path. The executor must require task
`policy_checks.sonic_mode == 1`, save each actual
`controller_diagnostics.actual_modes`, require `[1]` with nonzero control steps,
and stop on absent or conflicting evidence. Do not inject these requirements
as unknown backend action fields. No IK/mode0/teleoperation fallback is provided.
The backend still enforces measured contact, fall, collision and task gates.

This module validates data only and imports no physics runtime. Its unit tests
do not establish physical feasibility or launch readiness. New physics runs
still require the frozen experiment record, actual providers, model/code hashes,
and runtime evidence mandated by the repository. The original action sequence
access of any executing planner must be recorded separately; this adapter alone
does not demonstrate strategy transfer.
