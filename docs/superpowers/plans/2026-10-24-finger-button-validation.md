# Finger button validation implementation plan

User approved fixed-wrist finger pressing experiment. Work only in independent cap-sim-api-augmentation route. Preserve previous failed wrist-only attempt_01 as baseline; same episode, X+5mm, initial history, native criteria, 2s tail, wrist algorithm/parameters, fixed body/hand PD. Wrist feedback remains closed loop, so its output may change with finger-induced forces.

New independent right-index target channel: API chooses proximal/distal offsets, smooth onset and bounded force integral. No body decoder output modification. Source fingers elsewhere unchanged. Limits come from model and explicit pilot bounds. Store exact final hand targets at each50Hz tick. Independent replay uses saved token AND final hand targets, with both feedback adapters off.

- [x] Probe actual command-to-joint mapping, limits, contact links, force and kinematic derivatives at baseline frames without physical stepping.
- [x] Test prefix, rate and limit handling; immutable source/other fingers; exact hand replay shape/nonfinite rejection.
- [x] Add finger adapter and frozen optional physics channel; persist hand targets and bind them during token replay.
- [x] Add compact API input, skill, three-credential alternating transport. At most3 physics candidates, one replay per full candidate; stop on successful generation+successful exact replay. If API unavailable after initial+5 retries, preserve terminal summary and stop; no manual parameters.
- [x] Freeze each physics contract, run serially, render and audit. Report physical success separately from replay equality, API failures and unknown token usage.

Environment fixed to registered CPU SONIC1.0 / ORT1.23.2 / TRT10.13.0.35 / MuJoCo3.3.6. No reset in rollout; no IK/mode0 fallback or threshold relaxation. This remains original-action-visible trajectory retargeting. Failed diagnostics are not exported as training-ready data.

Result: 20261024_2053_button_finger_press ran3 generation candidates and3 exact independent token+hand replays. Native success0/3, replay equality3/3. API5requests,2timeouts with unknown usage; all physical trials completed. Details in experiment REPORT.md.
