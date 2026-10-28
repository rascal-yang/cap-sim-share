# GPT-6 contact trajectory reuse

User intent: add an independent data generation pipeline inspired by Humanoid
MimicGen, use GPT-6 to annotate demonstration phases, reuse contact trajectories,
and perform a small real test. Existing pipelines and historical experiments are
read-only. The user delegates design and execution to the agent.

This is trajectory retargeting with access to original measured trajectories,
not evidence of strategy generation or transfer. It implements an object-centric
single-object, right-hand apple example; it does not claim full HumanoidMimicGen
reproduction or a trained navigation policy.

## Data and algorithm

`dataset.py` imports a cap-sim native successful attempt using scene.mjb and
states.npz. MuJoCo forward kinematics extracts both wrist poses, torso/root pose,
object pose and hand joints. Contacts come from aligned recorded contacts.json,
not reconstructed contact forces. Validate finite arrays, timestamps and source
acceptance/evidence hashes. No simulation step during import.

Canonical in-memory demonstration: dict with `time_s` (N), `left_pose`,
`right_pose`, `object_pose`, `root_pose`, `torso_pose` (N,7: xyz+wxyz),
`left_fingers`, `right_fingers` (N,7 radians), `hand_force_N`,
`support_force_N`, `table_force_N` (N), and `metadata` (dict). NPZ carries arrays,
adjacent JSON carries metadata. `load_demo(path)`, `save_demo(demo,path)` and
`validate_demo(demo)` are shared interfaces. Dense measured finger angles are
not original finger commands; retain that distinction in provenance.

`annotation.py` constructs a compact evidence request including sampled state,
force event intervals and source hash. GPT-6 supplies ordered half-open intervals
covering [0,N): `name`, `kind` (transit/contact), `start_frame`, `end_frame`,
`hand` (right/left/both), `object_id` (target for contact, null for transit),
`evidence` (nonempty string list). Root fields: `schema_version:1`,
`demo_sha256`, `stages`. Validate boundaries, complete coverage, real contact
evidence for every contact stage and identity binding. Model provenance is saved
separately. API (explicit GPT-6 model) and file response routes are supported;
file import must disclose its identity, never masquerade as an API call.

`retarget.py` transforms each contiguous contact group with a FIXED object pose
at the first frame: T_new_hand[t] = T_new_anchor * inverse(T_source_anchor) *
T_source_hand[t]. Consecutive lift/hold contact phases share the anchor. Transform
only annotated hands. Preserve sample times, original finger values and every
contact sample. A quintic Cartesian / shortest-arc quaternion connection fills
transit before contact, with endpoints at the fixed preceding pose and first
contact target. Initial 5s original-token prefix can be explicitly retained and
excluded from transit modification. Offline target object trajectory is only a
reference, never a simulated measurement. Geometric validation is not success.

## Execution and evidence

`native.py` is a new adapter using existing HUMANO Backend and Sonic read-only.
Execute the explicitly declared 250 original motion-token prefix from initial
state, then dense wrist references through mode1 at 50Hz, fingers at 200Hz.
No body_keyframes, IK or fallback. Initial warmstart follows historical prefix.
Save actual modes/providers, every physical state/contact, commanded references,
native acceptance and independent saved-state revalidation. Maintain original
fall/table/contact gates, record handoff and stop reasons. Generated transit is
reanchored to measured wrist pose at prefix handoff. This may expose mode1
tracking failures; never relabel these as pipeline success.

Physical launch requires a completed frozen experiment.json from the repository
template, verified hashes, loaded-session runtime evidence, physics/asset/initial
manifests and declared differences. Root prepares these before the first step.
One original-scene smoke only; no controller success-rate comparison. Offline
geometry checks use the most recent ten specified straight-sweep conditions.
No new variant physics launches if baseline fails.

## Deliverables and limits

CLI `python -m humanoid_mimicgen` provides import, request, annotate, generate,
and physical smoke entry points. README supplies repeatable commands, source
links, provenance and limitations. Tests cover malformed input, frame boundaries,
source binding, SE(3) rotation and translation, fixed-anchor lift preservation,
inactive hand, finger/time invariants, prefix and transition continuity. Existing
unit tests run as regression without changing their code. New experimental
artifacts use actual Asia/Shanghai creation time and never overwrite prior runs.
