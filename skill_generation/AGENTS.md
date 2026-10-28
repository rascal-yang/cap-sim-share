# Skill-driven generation preparation

- Preserve reference-retargeting files listed in retargeting_baseline.json. Do not relocate, modify, or delete those baselines.
- This directory is preparation only; do not present templates or candidate skills as executed/validated results.
- Keep the outer cap_sim pipeline stable. Implement future isolation as an explicit adapter; no hidden source action/token access.
- The skill author may read demonstrations. The executing GPT gets published skill, current geometry/interface/state, and measured feedback in a fresh context; no raw demonstrations.
- First control profile is SONIC1.0 CPU mode1 with TensorRT10.13.0.35/ORT1.23.2. No implicit IK/mode0 fallback. IK feasibility checks, if used, must be labelled as checks rather than executed control.
- Validate and freeze the complete experiment record before physical execution; use minute-resolution experiment folders outside this repository. Record actual API calls/model identity; no implicit model selection.
- Draft skill must be reviewed and published before launch. Same attempt budget across revisions; analysis saved before each retry; preserve any complete accepted trajectory.
- Track original observations, actual commands, state/contact feedback, frames and timestamps, and controller/model metadata. Keep failed/preview video separate from training visual acceptance.
