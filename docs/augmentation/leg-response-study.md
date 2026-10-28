# Measured-replay leg response and preencoder feedback

Approved experiment: use verified original-token successful replay measured q/dq as target, not PD targets; finite differences from identical handoff states; bounded local feedback before encoder; independent token+hands replay if native success. Use isolated module/skill/entrypoint; no changes to old runners.

- [x] Test central-difference response identification, coupled inverse, bounded/rate-limited integration and velocity-consistent future reference prediction.
- [x] Bind original-token success target and source, real loaded runtime and code/config hashes before every physics run. Same 352-token prefix and fixed original native criteria, PD, hands, 2s tail. Reference source shift0; tested scene shift+5mm explicitly declared.
- [x] API chooses bounded experiment parameters. Run baseline plus +/- paired hips/knees/ankles (7 full resets, no state teleport). Build 6-output/3-input local response from last10 samples. Reject incomplete/stopped calibration; disclose locality/nonlinearity and singular directions.
- [x] Up to2 feedback candidates (API selects after seeing measured response), per-tick Python error correction, no runtime GPT. Record clipping and predicted/actual behavior; freeze measured calibration response before feedback.
- [x] Successful candidate saved-token+hands replay. All videos shallow; failure/safety/API error/partial execution distinguished. No unconditional success claims.

Planning/implementation time unknown. Numeric capability bounds are declared for this pilot, not universal learned gains. Controller SVD cutoff is numerical regularization, not task-specific success tuning. No direct pelvis/foot-position command, no IK/mode0, no decoder-target or PD correction. No new acceptance revision. Target trajectory extends final successful measured pose with zero velocity after its recorded end.

Outcome: experiment20261025_1446_chairsit_measured_replay_feedback completed7 calibration and2 feedback rollouts. All9 safe/full but native sitting failed. Feedback reduced baseline six-joint error norm0.2519→0.1863/0.1895rad and pelvis0.6039→0.5926/0.5916m; hip contact remains absent. Independent successful-token replay not triggered. API chose ±0.03rad calibration, then2s response timescale with offset/rate caps0.1/0.05 and0.15/0.075. 3API calls23309tokens. 270tests and9 frozen-control audits pass.

Review fixed terminal forecast times to clamped frame intervals with zero terminal-held dq, and bound calibration trajectory/audit hashes in feedback contracts. No measured-velocity error controller claimed: q is the feedback error; successful replay dq is saved for diagnostics.

Limits: 3paired channels cannot generally match6joint targets; static Jacobian has nonzero projection residual and weak ankle direction. Saturated ankle correction, finite time, local model and left/right coupling were not independently ablated. Do not label negative result as proof of encoder-side impossibility.
