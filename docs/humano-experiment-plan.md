# 10-task native HUMANO integration plan

Baseline remains `2ed4500`. New branch adds the native locally available HUMANO adapter; external HUMANO source/assets remain untouched.

1. Compile all ten native tasks using episode 0 environment config and initial state only. Freeze source/assets, controller permissions and native success criteria. Record compatibility errors separately from physical failures.
2. Add a native acceptance contract, CPU SONIC encoder/decoder with sparse full-body or Cartesian stages, original torque application, state/contact logging and dual-view rendering. Both groups get the same interface and controller.
3. Run no-reference cases first. GPT in the current session supplies bounded action programs and diagnoses feedback; no online API key is available. No expert rollout or action sequence is inspected before that group is complete.
4. Read one successful reference per task. Extract object-relative contact/motion experience. Shift task object by 10 cm where the native asset supports it; adjust approach/contact/body targets and insert corrective stages. Do not label unmodified source replay as augmentation.
5. Each task/group has a 10-attempt ceiling, 30 simulated seconds per attempt. Stop after accepted success, then permit at most one optimization within the same budget. Preserve failed attempts/prefixes. If interface or startup is unsupported, mark blocked separately; do not fabricate a physics attempt.
6. Report 10-case denominators separately for no-reference and reference augmentation, elapsed setup/planning/physics/render times, all outcomes, first/third-person videos and visual review. Success rate is for this implementation/budget, not a causal proof about skill value.

Native task criteria supersede the apple-specific lift contract for this new suite. Any difference between native success and stricter contact/stability quality is reported separately. No criteria/physics changes during retries.

Native integration validation: all ten original assets/configurations compiled. Head stereo and external initial views inspected. Native SONIC CPU 0.1-second smoke, saved-state re-evaluation and dual-view MP4 decode passed. The original source uses scalar encoder mode followed by three zero values; the adapter follows it. MuJoCo MjSpec XML export contains an anonymous nested default and virtual mesh filenames, so the authoritative replay model is saved as binary `scene.mjb` (same MuJoCo version), not the diagnostic `scene_source.xml`.
