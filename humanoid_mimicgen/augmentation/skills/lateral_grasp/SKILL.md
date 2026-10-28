---
name: lateral_grasp
description: Plan and physically test an apple lateral operation through audited local tools.
---

Read this skill, then inspect_scene and plan_lateral. Use successful replay support windows and actual geometry/physical history. This first flow initializes the robot at the generated operating stance with zero velocities and synthetic stationary SONIC history. It never establishes navigation success. Original sequences are visible: trajectory_retargeting.

The API selects tools, a checked stance candidate and a right-wrist world-Z correction in [0,.03] metres. Python generates full-body IK, checks geometry, executes SONIC mode1 and advances MuJoCo physics. You cannot change physics, PD, acceptance, original hand commands, controller mode or filenames. Unsupported finger preshape/closure-strategy changes require a new capability; stop and describe that gap instead of pretending a wrist shift implements it.

After plan_lateral, prepare_grasp with a falsifiable rationale. Inspect the initial image supplied after that call before execute_grasp, and record the visual assessment. execute_grasp is a real new physical trial, with frozen API/scene/controller provenance, actual loaded CPU providers and independent verification. No original token prefix or fallback. Never infer execution from text or an offline reference.

After every physical trial, inspect both native_success and clean_success, contact body/time/force, support, wrist tracking and saved-state images. Record reflect_grasp before another preparation. Accept only when both native and collision quality pass. Retry only with a supported change within the available tools and remaining budget. If the needed change is unsupported, stop and explain. Past successful lifts do not satisfy this new trial. Finishing this workflow honestly with measured collision failure is allowed; do not waste physical attempts merely to exhaust the budget.

Use at most the supplied preparations/attempts. No model polling while local tools run. End with a concise summary after an accept or stop reflection. Report initialization and physical scope, not a multi-position success rate. Videos are saved-state diagnostic renders, training_ready=false.
