# Button palm-mesh acceptance v1

User authorization on2026-10-25 permits this explicitly disclosed acceptance correction. Profile ID: `button_palm_mesh_v1`.

The current G1 hand model attaches both `right_wrist_yaw_link` wrist-shell mesh and `right_hand_palm_link` hand-palm mesh to body `right_wrist_yaw_link`. The original HUMANO checker chooses bodies by keywords(palm/thumb/index/middle/ring/little/pinky), omitting the palm mesh because of its parent body's name. This correction adds only contacts with the collision mesh `right_hand_palm_link` on that exact parent. It does not admit arbitrary wrist-shell, forearm, elbow or leg contacts; left-side mappings are not added without separate review.

Unchanged conditions: original native accepted finger/palm bodies, same button geom-name matching, same directional contact-force convention, contact count>=1, contact distance<=1mm, downward force>=20N, 50 consecutive50Hz ticks(1s). Completion additionally requires full safe execution. Timing interruptions reset the counter. Exact reviewed collision geometry must exist uniquely. Unexpected native thresholds or cadence reject evaluation. Reconstructed original per-frame hold counters must match recorded native counters before a revised verdict is accepted.

The original environment/task checker is not edited. `native_success` remains its recorded result; `revised_success` is the approved profile verdict. Keep the ID and both fields in reports/exports. Do not retroactively overwrite frozen source result.json, reports or benchmark claims. Offline reevaluation outputs belong to a new assessment record. A revised success is not an original native benchmark success.

Future contact_strategy button workflow invokes this exact-mesh evaluator, saves execution/button_acceptance.json, and uses revised acceptance plus exact token/hands replay as its completion gate. Broad all-right/wrist force remains a diagnostic only. No online force-feedback signal, motion parameters, PD gains or safety threshold changes accompany this acceptance correction. Legacy feedback's native-body omission remains a separate controller issue.

Read-only review command:

```bash
PYTHONPATH=. scripts/humano_python.sh scripts/review_button_acceptance.py --config NEW_ASSESSMENT/config.json
```

Config includes output_root and cases:[{name,case}]. Each case binds scene/trajectory/metrics/contract hashes and validator version; no physics stepping or API call. Tests cover exact palm-vs-wrist selection, force/distance/contact-count boundaries and interrupted holds. Initial review: API generation and saved-token replay pass1.76s; three short manual candidates remain failures(.02/.04/.26s); original successful replay remains a pass1.22s.
