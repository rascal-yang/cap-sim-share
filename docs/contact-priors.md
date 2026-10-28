# Reference contact priors

Opt in to extracting measured contact experience from a physically accepted native
MuJoCo reference by adding this field to a task:

```json
"contact_prior": {
  "object_body": "apple_001",
  "wrist_body": "right_wrist_yaw_link",
  "object_key": "target"
}
```

The public pipeline still recovers the original reference before generation.
Contact extraction is deferred during `reference_recovery`; generation extracts
from the newly recovered success artifact. The native apple experiments also gate
original-token replay and same-scene interface conversion before using the converted
success. Unsupported/missing evidence fails explicitly instead of inventing contacts.

Standalone extraction:

```bash
python -m cap_sim extract-contact-prior \
  --reference /path/to/success/attempt_001/trajectory.json \
  --object-body apple_001 --wrist-body right_wrist_yaw_link \
  --out /path/to/reference_contact_prior.json
```

The reference directory must contain `trajectory.json`, `scene.mjb`, `states.npz`,
`contacts.json`, `samples.json` and a successful `independent_validation.json`.
Extraction checks sample hashes, full-state layout and contiguous contact/state
alignment. It selects the first 0.2 s interval where at least two named hand links
maintain measured contact (each >=0.1 N), then saves median contact locations in
object and wrist coordinates, local scatter, observed force percentiles, the
reference reset object pose and source evidence hashes. This is a persistent
contact sample, not a learned universal safe grasp region. Current extraction
uses native `hand` link naming and direct object-body contact events; articulated
multi-body objects need an explicit adapter.

Each initial and feedback planner request gains `contact_privileged`:

- `reference_priors`: frozen source contact experience;
- `initial_targets`: contact targets and suggested wrist seed mapped at reset;
- `current_targets`: targets remapped using the current measured object pose,
  matching measured contacts per hand link and nearest-point position error;
- missing actual contact is represented by `null` error, never zero;
- force observations are explicitly **not** force-control targets.

Mapping is `p_world = object_position + R_object @ (scale_ratio * p_local)`.
Quaternions are wxyz; positions are metres. Current task scale convention is the
apple backend's uniform `scene.apple_scale`; omitted means 1.0. Nonuniform scaling
and topology/robot-hand changes are not supported by this contact mapper.
The wrist seed aligns contact centroids using the recorded object-to-wrist
orientation and wrist-local offsets. A reported fit residual exposes the fact
that scaling an object cannot generally be accommodated by wrist translation
alone. The controller must still establish actual contacts with feasible fingers.

The OpenAI planner prompt receives the same enriched request and instructions to
use these contacts, reobserve object motion, and distinguish targets from achieved
contacts. File/command planners receive identical data. No API call occurs merely
because contact priors are enabled; planner selection is unchanged.

`experience.json` freezes extracted priors and their evidence hashes. Resume
rejects changed experience. The public framework exposes contact targets; each
planner must actually consume them when generating actions. The apple validation
policy does so by deriving a wrist translation from mapped contact centroids,
replacing its previous manual scene/size offset while retaining the successful
reference's tracking compensation. It logs current per-link contact errors but
has not introduced a contact-position error threshold or automated finger optimizer.

Validation: unit tests cover translation/rotation/scale, moving-object extraction,
invalid input and state alignment, and initial/feedback request propagation.
Local physical evidence and new videos are stored outside the repository in
`experiments/20261013_apple_contact_prior/` in the parent workspace.
