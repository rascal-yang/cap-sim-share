---
name: contact_bootstrap
description: Bootstrap the HUMANO table-button task from raw demonstrations through saved-state replay, physical SONIC replay, contact annotation, small retargeting trials, and evidence export.
---

Use these registered tools; no shell or frame-by-frame model reasoning is needed.
1. `inventory` validates source files and timing. `replay_saved` inspects one episode without physics. Historical accepted flags and reconstructed contacts are not fresh physical success.
2. `replay_physical` runs that episode's original motion tokens through the CPU SONIC decoder. At most three distinct episodes may be tried. Read returned failures before selecting another episode. Do not change physics, success thresholds, or claim success from a saved-state replay.
3. After a physically successful replay, `annotate_contact` saves approach, press_hold, and post_contact boundaries from measured native press conditions. Both endpoints are included; adjacent clips share a state, never duplicate an executed action. post_contact need not mean release.
4. `augment` executes the declared fixed five-case suite: mode1 baseline and object-only XY offsets ±1 cm. Source leg references and world wrist references are encoded in mode1; the robot is not translated, no original token prefix, no IK or mode0 fallback. Recorded fingers remain unchanged. This is trajectory retargeting, not a newly learned manipulation strategy. Failure is a valid exported result. Do not rerun a failed case under the same identity.
5. `export_report` writes transitions, evidence paths, success rates and timing. The host adds actual API usage after the final response. If all three physical replays fail, export the failed attempt evidence and explain the blocked later stages.

Native button acceptance: contact count and downward force sustained for the task's configured hold count. Do not borrow apple lift/grasp conditions. No online GPT control calls, no within-episode retry in this bootstrap version. A later correction requires a new experiment and explicit evidence. Keep tools compact; full state/control arrays stay on disk. Visual review is separate; exported records remain training_ready=false.

If the supplied config declares `hold_tail: true`, physical replay and every matched augmentation condition append one native acceptance window (hold_steps × reward_dt) of final intent. This is a contact-completion attempt, not exact source replay. Tokens repeat for original-token mode; Cartesian targets hold the final pose with zero reference velocity for mode1. Source data and thresholds are unchanged. Report this addition.
