# Native HUMANO ten-task benchmark

This benchmark records the 2026-10-13 local CaP experiment. It is a development experiment, not an online-model evaluation or a causal skill ablation.

The initial package is committed as `2ed4500`; the frozen native SONIC adapter as `d991dd5`. Native HUMANO assets and controller weights are external dependencies. The original `isaac-state-renderer` checkout is not used as an experiment output directory.

Protocol:

1. Compile each task using its environment configuration and initial snapshot. Freeze source/assets, physics, controller permissions and native acceptance.
2. Complete ten no-reference cases before reading expert rollouts. Hash the completed result files and record a freeze timestamp.
3. Inspect one reference episode per task, at most a second candidate when the first does not pass offline checks. Preserve candidate provenance and explicitly distinguish unverified dataset references.
4. Translate the task object 10 cm along world Y. Generate new lateral alignment and retarget sparse body/wrist motion using the same SONIC interface. Source states are used for offline geometric analysis only; execution applies controller torques.
5. Diagnose before each retry; at most ten attempts per case, thirty simulation seconds per attempt, twelve feedback calls. Gate lifts on actual target contact. Preserve first success before attempting one optimization within that budget.
6. Verify generated results from saved measured states. Report native acceptance and supplementary grasp quality separately. Render first-person and third-person videos from the recorded states without stepping physics.

Planning is performed by the current GPT session and GPT-authored Python feedback programs. There are no live external model API calls. Initial manual decisions and later automatic rules differ; this experiment cannot isolate the effect of a skill document.

Raw dataset, models, states, contact traces, mailboxes and MP4s stay outside Git. The local artifact root is `/home/kinetra-4090-01/code/zijian/experiments/20261013_cap_ten_tasks`. Its `REPORT.md` and machine-readable audits are authoritative. All 20 cases are complete: valid task success is 4/10 without references and 1/10 with references. All 20 first/third-person video pairs passed decoding checks. See [Chinese report](REPORT.zh-CN.md), [full report](REPORT.md), and [final consistency audit](snapshot/final_verification.json). The snapshot contains scripts, policy versions and audits; raw evidence remains at the external artifact root.

MuJoCo 3.3.6 and the external HUMANO Python environment are required for native execution and binary-model replay. Videos use MuJoCo EGL; they are not Isaac Sim renders. The package's portable example pipeline remains independently available via the main README.
