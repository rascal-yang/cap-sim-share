# Right-wrist encoder-only validation

User approved the proposed button pilot. Keep independent augmentation route and existing modes unchanged. The target is successful physical execution with fixed PD, zero post-decoder changes, followed by exact saved-token replay.

Design: optional preencoder wrist adapter changes only right wrist world reference position/orientation. It may blend measured source wrist reference toward FK of the successful original-token replay's commanded right-arm joints, add API-selected position/rotation offsets, and integrate native downward-button force error into a bounded XYZ reference offset. All transforms occur before mode1 encoder. Original-token prefix, source legs, torso, left wrist, hand commands, scene, native acceptance and safety stay fixed. Source actions remain visible: this is trajectory retargeting, not strategy transfer.

API receives compact success and failed-token evidence and selects executable parameters; no manual tuning. At most 4 generation trials plus one independent token replay per full trial (including failed trials to audit action completeness); stop on native+independent success with exact token replay. Each case freezes full contract before physics. Do not relax acceptance. Original successful reference and previous experiments immutable.

- Add and run failing tests for right-only transform, untouched input/left/torso, rotation convention, bounded force feedback and parameter validation.
- Implement optional adapter and physics integration; preserve baseline PD and no postdecoder modifier. Save reference/force history and actual targets.
- Extend independent token loader for generated cases without contact_control.json.
- Add API orchestration skill and standalone launcher with 3-credential rotation, per-request timeout and persisted decisions.
- Run tests, freeze records, execute sequential trials and token replays; render and audit results.
- Report success or failure, native hold duration, exactness, timing, API usage, and limitations. No training claim.
