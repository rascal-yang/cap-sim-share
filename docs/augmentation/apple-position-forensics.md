# Apple position transfer: historical control-chain investigation

2026-10-26, no external API. This is a diagnostic package, not a replacement default pipeline. Existing execution routes are unchanged. Full evidence and all11 shallow MP4 links:

[20261026_0953_apple_navigation_forensics](../../../experiments/2026-10/26/20261026_0953_apple_navigation_forensics/REPORT.md)

Two distinct historical methods must remain separate:

- The wide static sweep used250 original-token ticks,150 encoder-mode0 ticks, a1s measured-wrist open-hand bridge, then mode1 with fixed source standing legs and gradual source finger closure. No online navigation planner.
- The navigation success used live0.5m/s planner, physical settling, measured-stance IK/RRT as encoder1 references, fixed standing legs and right-wrist integral during approach/contact. No token/mode0 prefix.

Both historical executions reproduced bitwise. The static method also captured and held the exact current apple_03 position, with identical initial qpos/qvel/ctrl/warmstart and checked physical parameters. Its native9cm goal still failed: center lift8.84cm, stable hold1.00s, minimum hold clearance6.48cm, minor table contact3.63N/30ms. Do not relabel it native success or training-ready. Whole-method evidence does not isolate mode0, source trajectory, closure profile or stance as the sole cause.

Current early handoff (346→300), fixed measured-leg references, and transferred navigation speed changes did not independently rescue the tested failures. The old navigation success needs its approach/contact feedback at that tested pose; disabling it failed with two retries. Root XY precision alone did not predict capture.

## Reproduction entry points

Use `PYTHONPATH=. scripts/humano_python.sh scripts/<entry>.py ...`. Each physical case freezes its own contract before stepping; original records are read-only. Do not reuse output folders.

- `probe_historical_apple.py --variant restore|smooth|no_feedback [--root NEW_BATCH]`: rebuild the historical navigation request with verified original inputs and relocated paths; run local deterministic planning, freezing, physics and independent verification.
- `probe_apple_navigation_transfer.py --historical-request RESTORED_REQUEST --current-root WRIST_ABLATION_ROOT --condition apple_03 --stance follow_object|fixed [--root NEW_BATCH]`: preserve the robot initial state; translate apple/right-contact reference to the current condition and test object-following or fixed historical stance.
- `probe_apple_navigation_variant.py REQUEST NEW_CASE`: use the same target and contacts with the previously tested smooth navigation profile.
- `probe_apple_early_handoff.py --source WRIST_ABLATION_ROOT --handoff 300 --conditions apple_03 apple_05 apple_09`: change only handoff timing relative to the prior feedback arm.
- `probe_apple_fixed_legs.py`: same arguments as early handoff; fixes measured leg references and zero leg reference velocities at mode1 entry. Does not pin physical joints.
- `probe_apple_static_legacy.py --condition historical_y8|current_y2 [--root NEW_BATCH]`: explicit mixed-interface historical baseline. This is never an automatic fallback.

The rendering/audit/finalization scripts in this package target the recorded20261026 diagnostic batches. They are experiment utilities, not a general production benchmark interface. Renderers restore states without integrating physics. Sparse visual reviews do not certify temporal RGB quality. Static successful holds have not undergone a new independent token replay in this investigation.
