# Chair encoder-side leg reference pilot

Approved scope: add an isolated six-joint preencoder reference adapter, fresh zero-correction baseline, API-authored bounded candidates, native sitting acceptance and independent token+hands replay. Existing runners remain unchanged.

Implementation ledger:
- [x] Pure reference transformation: six named hip_pitch/knee/ankle_pitch joints, successful commanded-reference blend plus offsets; smooth activation and consistent correction velocities; identity and invalid-input tests.
- [x] Isolated context manager around suite source_poses/write_new: precompute before physics, bind reference/config/transformed arrays in frozen contract, keep all decoder/PD outputs unchanged. Restore hooks on exit.
- [x] API runner/skill: fixed full plan validated exactly, baseline zero correction, max3 repair decisions, frozen tail and native criteria, no live GPT control. Unsupported preflight is not a physical failure.
- [x] Fresh baseline, physical repair, independent replay for success; shallow videos, complete report, API usage and timing.

Rulings: reuse existing feature worktree cap-sim-api-augmentation; no changes to core suite runner or other generation routes. This pilot is trajectory retargeting: historical action targets remain visible. Historical postdecoder blending motivates reference choice but does not establish encoder-side feasibility. Do not expose nonexistent direct pelvis-height control.

Parameters: six fixed ordered joints; per-joint blend in[0,1], offset in[-0.3,0.3]rad, transition0.2..2s. Fixed activation at contact_start; remain active through episode tail. Reject total correction>0.5rad or correction slew>2rad/s, and out-of-range positions. These are explicit pilot capability bounds, not learned gains. Source velocity is retained plus discrete derivative of the correction; no unrelated source velocity recomputation. Future reference samples can preview correction; original-token prefix remains exact. PD and contact-force stop remain fixed.

Validation: 265 tests passed. Independent review findings on observer metadata and empty tracking suffix fixed with regression tests. Experiment 20261025_1230_chairsit_encoder_leg_api: zero baseline exactly reproduces historical arrays; API blends0.6/0.8/0.9 all failed native sitting contact. All4 encoder inputs and unchanged decoder targets verified. Success-replay branch not exercised because no candidate succeeded. Do not infer capability failure from this single monotone blend family. Evidence REPORT.md is in experiments/2026-10/25/20261025_1230_chairsit_encoder_leg_api/.
