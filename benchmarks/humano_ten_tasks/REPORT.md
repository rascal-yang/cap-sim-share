# HUMANO 10-task CaP experiment

Updated: 2026-10-13 10:01:57 +0800

Repository baseline: `2ed4500`; native integration: `d991dd5`.

This is a current-session GPT / GPT-authored CaP experiment, not a live external-model API benchmark. No-reference cases precede expert rollout access. Each group has 10 tasks; each case allows at most 10 attempts, 30 simulated seconds per attempt, 12 phase feedback calls and one post-success optimization within the same budget.

Reference variant: object +0.10 m along world Y. New lateral alignment and reference wrist retargeting use the same SONIC control interface. Native criteria and physics are fixed.

## no_reference

Completed: 10/10. Valid task successes: 4/10. Native passes: 5/10. Missing cases are pending, not failures.

| Task | Valid / native pass | Attempts | Selected attempt / sim s | Interrupted slots | Videos |
|---|---|---:|---|---|---|
| BallKick | True / True | 2 | 1 / 5.0 | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyBallKickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyBallKickTeleop/external.mp4) |
| ChairPush | True / True | 3 | 2 / 9.8 | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyChairPushTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyChairPushTeleop/external.mp4) |
| ChairSit | True / True | 2 | 1 / 17.2 | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyChairSitTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyChairSitTeleop/external.mp4) |
| DollyPush | True / True | 2 | 2 / 11.42 | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyDollyPushTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyDollyPushTeleop/external.mp4) |
| GroundBasketPick | False / True | 2 | 1 / 7.7 | [2] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyGroundBasketPickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyGroundBasketPickTeleop/external.mp4) |
| GroundBottlePick | False / False | 10 | None / — | [4] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyGroundBottlePickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyGroundBottlePickTeleop/external.mp4) |
| GroundBoxPick | False / False | 10 | None / — | [1, 3] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyGroundBoxPickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyGroundBoxPickTeleop/external.mp4) |
| TableApplePick | False / False | 10 | None / — | [6, 8] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyTableApplePickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyTableApplePickTeleop/external.mp4) |
| TableButtonPress | False / False | 10 | None / — | [7] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyTableButtonPressTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyTableButtonPressTeleop/external.mp4) |
| TablePlatePick | False / False | 10 | None / — | [1, 3] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyTablePlatePickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/no_reference/G1WholebodyTablePlatePickTeleop/external.mp4) |

## reference_augmented

Completed: 10/10. Valid task successes: 1/10. Native passes: 1/10. Missing cases are pending, not failures.

| Task | Valid / native pass | Attempts | Selected attempt / sim s | Interrupted slots | Videos |
|---|---|---:|---|---|---|
| BallKick | True / True | 3 | 2 / 4.16 | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyBallKickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyBallKickTeleop/external.mp4) |
| ChairPush | False / False | 10 | None / — | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyChairPushTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyChairPushTeleop/external.mp4) |
| ChairSit | False / False | 10 | None / — | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyChairSitTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyChairSitTeleop/external.mp4) |
| DollyPush | False / False | 10 | None / — | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyDollyPushTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyDollyPushTeleop/external.mp4) |
| GroundBasketPick | False / False | 10 | None / — | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyGroundBasketPickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyGroundBasketPickTeleop/external.mp4) |
| GroundBottlePick | False / False | 10 | None / — | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyGroundBottlePickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyGroundBottlePickTeleop/external.mp4) |
| GroundBoxPick | False / False | 10 | None / — | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyGroundBoxPickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyGroundBoxPickTeleop/external.mp4) |
| TableApplePick | False / False | 10 | None / — | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyTableApplePickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyTableApplePickTeleop/external.mp4) |
| TableButtonPress | False / False | 10 | None / — | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyTableButtonPressTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyTableButtonPressTeleop/external.mp4) |
| TablePlatePick | False / False | 10 | None / — | [] | [ego](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyTablePlatePickTeleop/ego.mp4) / [external](/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks/videos/reference_augmented/G1WholebodyTablePlatePickTeleop/external.mp4) |

## Reference provenance

| Task | Episode | Saved-state native check | First pass s |
|---|---:|---|---:|
| G1WholebodyBallKickTeleop | 0 | True | 2.579999999999945 |
| G1WholebodyChairPushTeleop | 0 | True | 12.00000000000081 |
| G1WholebodyChairSitTeleop | 1 | False | None |
| G1WholebodyDollyPushTeleop | 1 | True | 11.68000000000076 |
| G1WholebodyGroundBasketPickTeleop | 0 | True | 14.520000000001017 |
| G1WholebodyGroundBottlePickTeleop | 0 | True | 11.060000000000663 |
| G1WholebodyGroundBoxPickTeleop | 0 | True | 13.080000000000979 |
| G1WholebodyTableApplePickTeleop | 1 | False | None |
| G1WholebodyTableButtonPressTeleop | 1 | False | None |
| G1WholebodyTablePlatePickTeleop | 0 | True | 7.000000000000028 |

Source sim.ctrl is absent. Reference contact forces were estimated by PD reconstruction from recorded action.wbc and sampled qpos/qvel, not exact measured torque/contact records. A saved-state pass is not a demonstrated successful controller replay. ChairSit, TableApplePick and TableButtonPress remained unverified after two candidate references; their runs are reference-conditioned generation, not verified-success-experience transfer.

Reference cases with saved-state native pass: 1 valid generated successes / 7 source cases. Unverified-reference cases: 0 / 3. Pending executions are not completed failures.

Policy revisions are recorded in policy_revision.json, policy_revision_2.json and policy_revision_3.json. Target-specific bilateral contact gating was corrected before any reference pick attempt. Later task retries may switch from full-body reference targets to the existing Cartesian-upper-body plus SONIC-leg-planner API after measured tracking failure; post-reference lifts may add 3.5cm only while actual target contacts persist. No physical/controller/native acceptance parameters were changed.

Calendar elapsed since download start: 31927.220 s at report generation. This includes setup, idle/session gaps and interrupted work; it is not active simulation time.

## Timing and limits

Download wall time: 154.300 s. Attempt wall times include policy decisions, manual mailbox waits, simulation and evidence export/verification; they are not pure physics times. Decision/mailbox records have different coverage, including interrupted work, so that column is neither a subset of nor additive to completed-attempt wall time. The machine/session had infrastructure interruptions and idle gaps. Summed task work is not calendar elapsed time when runs overlap. Per-case `results_audit.json`, mailbox timing files, `direct_execution.json`, reference extraction records and per-video `render_timing.json` preserve the measured components.

| Group | Recorded completed-attempt wall s | Decision/mailbox wall s | Executed sim s |
|---|---:|---:|---:|
| no_reference | 5907.594 | 5982.828 | 1235.925 |
| reference_augmented | 2778.929 | 995.275 | 1567.235 |

Native pick checks detect height only. The basket lifted by knee collision is excluded from valid grasp/task successes; raw native success and its trajectory remain preserved. Process-loss/timeouts are reported separately and are not evidence of controller task failure.

Videos replay saved measured states through MuJoCo EGL, using head stereo left (first person) and a fixed external (third person) camera. MP4 full-decode and sampled-frame checks are recorded. They are not Isaac renders or certified training-ready video.

Failures under this budget mean this implementation did not pass the case; they do not establish that KPI or skill representations are ineffective. Native reference saved-state acceptance is not a guarantee that the reference will replay successfully through this controller.

## Failure records

- no_reference/G1WholebodyBallKickTeleop: none
- no_reference/G1WholebodyChairPushTeleop: attempt 1: task_not_completed; attempt 3: task_not_completed
- no_reference/G1WholebodyChairSitTeleop: none
- no_reference/G1WholebodyDollyPushTeleop: none
- no_reference/G1WholebodyGroundBasketPickTeleop: attempt 2: infrastructure_timeout_waiting_for_GPT
- no_reference/G1WholebodyGroundBottlePickTeleop: attempt 1: task_not_completed; attempt 2: task_not_completed; attempt 3: execution_deadline; attempt 4: infrastructure_process_lost; attempt 5: execution_deadline; attempt 6: execution_deadline; attempt 7: execution_deadline; attempt 8: invalid_action: Body reference outside joint limits; attempt 9: execution_deadline; attempt 10: execution_deadline
- no_reference/G1WholebodyGroundBoxPickTeleop: attempt 1: infrastructure_timeout_waiting_for_GPT; attempt 2: execution_deadline; attempt 3: infrastructure_process_lost; attempt 4: execution_deadline; attempt 5: execution_deadline; attempt 6: execution_deadline; attempt 7: invalid_action: Body reference outside joint limits; attempt 8: execution_deadline; attempt 9: execution_deadline; attempt 10: execution_deadline
- no_reference/G1WholebodyTableApplePickTeleop: attempt 1: unexpected_table_contact; attempt 2: unexpected_table_contact; attempt 3: unexpected_table_contact; attempt 4: unexpected_table_contact; attempt 5: contact_lost; attempt 6: infrastructure_timeout_waiting_for_GPT; attempt 7: execution_deadline; attempt 8: infrastructure_process_lost; attempt 9: execution_deadline; attempt 10: execution_deadline
- no_reference/G1WholebodyTableButtonPressTeleop: attempt 1: unexpected_table_contact; attempt 2: task_not_completed; attempt 3: unexpected_table_contact; attempt 4: unexpected_table_contact; attempt 5: unexpected_table_contact; attempt 6: unexpected_table_contact; attempt 7: infrastructure_process_lost; attempt 8: unexpected_table_contact; attempt 9: unexpected_table_contact; attempt 10: unexpected_table_contact
- no_reference/G1WholebodyTablePlatePickTeleop: attempt 1: infrastructure_timeout_waiting_for_GPT; attempt 2: task_not_completed; attempt 3: infrastructure_process_lost; attempt 4: task_not_completed; attempt 5: execution_deadline; attempt 6: execution_deadline; attempt 7: execution_deadline; attempt 8: task_not_completed; attempt 9: execution_deadline; attempt 10: execution_deadline
- reference_augmented/G1WholebodyBallKickTeleop: attempt 1: task_not_completed
- reference_augmented/G1WholebodyChairPushTeleop: attempt 1: task_not_completed; attempt 2: task_not_completed; attempt 3: task_not_completed; attempt 4: task_not_completed; attempt 5: task_not_completed; attempt 6: task_not_completed; attempt 7: task_not_completed; attempt 8: task_not_completed; attempt 9: task_not_completed; attempt 10: task_not_completed
- reference_augmented/G1WholebodyChairSitTeleop: attempt 1: task_not_completed; attempt 2: task_not_completed; attempt 3: task_not_completed; attempt 4: task_not_completed; attempt 5: task_not_completed; attempt 6: task_not_completed; attempt 7: task_not_completed; attempt 8: task_not_completed; attempt 9: task_not_completed; attempt 10: task_not_completed
- reference_augmented/G1WholebodyDollyPushTeleop: attempt 1: task_not_completed; attempt 2: task_not_completed; attempt 3: task_not_completed; attempt 4: task_not_completed; attempt 5: task_not_completed; attempt 6: task_not_completed; attempt 7: task_not_completed; attempt 8: task_not_completed; attempt 9: task_not_completed; attempt 10: task_not_completed
- reference_augmented/G1WholebodyGroundBasketPickTeleop: attempt 1: robot_fall_or_simulation_warning; attempt 2: task_not_completed; attempt 3: task_not_completed; attempt 4: task_not_completed; attempt 5: task_not_completed; attempt 6: task_not_completed; attempt 7: task_not_completed; attempt 8: task_not_completed; attempt 9: task_not_completed; attempt 10: task_not_completed
- reference_augmented/G1WholebodyGroundBottlePickTeleop: attempt 1: task_not_completed; attempt 2: task_not_completed; attempt 3: task_not_completed; attempt 4: task_not_completed; attempt 5: task_not_completed; attempt 6: task_not_completed; attempt 7: task_not_completed; attempt 8: task_not_completed; attempt 9: task_not_completed; attempt 10: task_not_completed
- reference_augmented/G1WholebodyGroundBoxPickTeleop: attempt 1: task_not_completed; attempt 2: task_not_completed; attempt 3: task_not_completed; attempt 4: task_not_completed; attempt 5: task_not_completed; attempt 6: task_not_completed; attempt 7: task_not_completed; attempt 8: task_not_completed; attempt 9: task_not_completed; attempt 10: task_not_completed
- reference_augmented/G1WholebodyTableApplePickTeleop: attempt 1: task_not_completed; attempt 2: task_not_completed; attempt 3: task_not_completed; attempt 4: task_not_completed; attempt 5: task_not_completed; attempt 6: task_not_completed; attempt 7: task_not_completed; attempt 8: task_not_completed; attempt 9: task_not_completed; attempt 10: task_not_completed
- reference_augmented/G1WholebodyTableButtonPressTeleop: attempt 1: unexpected_table_contact; attempt 2: task_not_completed; attempt 3: task_not_completed; attempt 4: task_not_completed; attempt 5: task_not_completed; attempt 6: task_not_completed; attempt 7: task_not_completed; attempt 8: task_not_completed; attempt 9: task_not_completed; attempt 10: task_not_completed
- reference_augmented/G1WholebodyTablePlatePickTeleop: attempt 1: task_not_completed; attempt 2: task_not_completed; attempt 3: task_not_completed; attempt 4: task_not_completed; attempt 5: task_not_completed; attempt 6: task_not_completed; attempt 7: task_not_completed; attempt 8: task_not_completed; attempt 9: task_not_completed; attempt 10: task_not_completed
