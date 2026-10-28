# Button wrist/finger joint pilot

Set `joint_wrist: true` in a new study config passed to `scripts/run_finger_study.py`. This opt-in permits the API to change `preencoder` and `finger_control` together and requires both right index joints explicitly. Other physical fields remain fixed. Existing configs keep finger-only behavior.

This is NOT explicit elbow control: the released SONIC mode1 observation schema lacks elbow targets. The 20261024_2254 model probe perturbed all fullbody joint-reference channels by +/-0.1 rad and obtained identical tokens; +/-1cm right-wrist reference changed tokens. Mode0 exposes fullbody joint references but is a different interface, not an automatic fallback.

The API receives source geometry, compact prior force/hold samples, prior finger-only outcomes if `prior_evidence` is configured, and one latest image. Its skill is `humanoid_mimicgen/augmentation/skills/wrist_finger_joint/SKILL.md`. Plans are validated before physics. Each case freezes evidence and uses the existing pinned CPU SONIC runner. Body targets and gains after decoding remain unchanged. Both index commands are saved separately from body tokens.

Budget: at most three generation candidates. Full candidates receive independent token+final hands replay; stop on native/independent generation success and exact successful replay. Native button criteria, tail, scene and safety cannot be changed by this pilot. No other tasks run. An unsuccessful pilot does not establish infeasibility. A successful fixed-scene case still requires a separately declared perturbation study before robustness claims.
