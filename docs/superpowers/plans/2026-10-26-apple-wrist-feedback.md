# Restore historical wrist integral, ten-position A/B

Approved scope: same ten apples, only encoder-input right wrist closed-loop feedback added; original prefix/hands/criteria fixed; successful data must replay through saved tokens/hands. Implement isolated scope and fresh-start runner, freeze real evidence, run10 pairs serially, independent replay, diagnostic/video/report, commit.

Ruling: reuse authenticated candidate0 and no fresh API or retries — isolates feedback without strategy/search confounds; does not measure best attainable performance with adaptive retry.
Ruling: apply exact historic0.5/s gain2cm cap to current baseline target including its fixed5cm Z offset, mode1 suffix only — preserves baseline inputs but does not restore early approach feedback or a different physical target definition.
Ruling: reject legacy rollback on feedback checkpoints instead of claiming added integral state is restored — experiment uses fresh starts; saved token replay remains independently verified.

Tests first failed on missing module, then3 tests passed: world shift once/right-position-only/immutable reference; historic cap and baseline identity; nonfinite rejection. Final review and whole suite pending. All other routes and historical evidence untouched.

Fresh review: no important defects. Minor caveat about process-wide suite_evidence hooks clarified in protocol; standalone process required. Legacy checkpoint rejection tested. Full suite292 passed before launch.

Physics complete:10positions,20 fresh A/B executions plus12 independent saved-token/hands replays. Baseline6/10,feedback6/10,no gained/lost conditions. All10 baseline trajectories exactly match historical seeds; A/B prefixes, nominal reference targets and source hands match. Mean mode1 wrist RMSE2.448→2.081cm (-15.0%).32 cases frozen-evidence/body-identity/provider audit passed. First closed-hand command frame340, feedback/scene-relative mode1 starts346 (120ms later); late adaptation is a concrete next hypothesis, not yet an early-handoff causal result. No additional strategy/API tuning performed; zero new API calls/tokens. Numeric wall398.07s. Rendering and visual review tracked separately in experiment root.
