# CaP Sim design

接受任务 JSON、可选参考轨迹与视频；输出通过独立物理验收的生成/扩增轨迹，以及所有失败证据。

Core: bounded JSON action programs (not arbitrary Python execution), simulator adapter, independent acceptance, immutable attempt directories, valid-prefix replay from initial state, successful baseline retention, then bounded optimization. A failed suffix never deletes a useful prefix. Scene changes invalidate literal world-coordinate prefixes; reference summaries may inform new plans.

Task: backend + scene + fixed acceptance + budgets + optional references. Planners: OpenAI Responses (text/vision), external JSON command, offline scripted plans, deterministic demo. Simulation: portable MuJoCo Cartesian gripper demo; actual G1/SONIC runtime imported as an external experiment adapter without copying weights/assets or hardcoded user paths into core. Backend protocol permits new robot/task adapters; arbitrary tasks are not automatically solved.

Assets: video frame extraction -> vision planner scene manifest -> metric primitive collision/visual MJCF -> geometry/physics checks. Scale, mass, friction, material, texture and pose provenance mandatory; uncertain scale remains uncalibrated. Complex mesh generation is an external command hook, never silently advertised as reconstruction fidelity. No remote publishing or credentials bundled.

Validation: fail-first tests for false success, prefix survival/replay, frozen acceptance, optimization rollback and budgets; actual MuJoCo reference/no-reference/augmentation runs; G1 recorded-state evaluation/import and a short real runtime smoke; video-asset extraction/build smoke. API transport tested with mock server; no claim of live GPT success unless actually tested.
