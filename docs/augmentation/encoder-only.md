# Encoder-only wrist validation

This optional button pilot preserves the stable contact route. Generation uses the original token prefix followed by SONIC mode1, with a right-wrist reference adapter before the encoder. It never modifies decoder targets or PD gains. The left wrist, torso, leg references and source finger commands stay fixed. Access to source actions makes this trajectory retargeting.

The API selects `preencoder`: `reference_blend`, world `translation_m`, world-left-multiplied `rotation_vector_rad`, `ramp_s`, `force_band_N`, signed world-vector `force_gain_m_per_N_s`, `max_feedback_m`, and `feedback_rate_m_s`. The reference is FK from the successful original-token replay's seven commanded right-arm targets and its recorded remaining pose. Force is the native button downward-contact measurement, not total normal force. Integration runs locally at 50 Hz and changes wrist reference before token generation. No API calls occur during physics.

A generation run must pass unchanged native and independent task checks. A separate run bypasses encoder and reference feedback, directly decodes the saved tokens with the same initial history, fixed base PD and original fingers, and checks state/next-state/command equality. Exact replay of a failed task is still a task failure.

```bash
python scripts/run_encoder_wrist_study.py \
  --config /absolute/new_study/config.json \
  --provider /absolute/new_study/provider.json \
  --credentials ../api/openai_config.toml \
  --secondary-credentials '../api/openai_config 2.toml' \
  --additional-credentials '../api/openai_config 3.toml'
```

Config contains `output_root`, `success_case`, `token_failed_case`, and `dataset`. Use a new timestamped month/day experiment root; do not rerun into an existing attempt directory. Reference cases require intact original evidence or matching frozen source snapshots. Each physical case freezes its own full experiment contract before stepping.

The pilot has four generation attempts with API-selected changes and an 1800-second API planning deadline. It tries independent token replay for each fully completed generation, retains failed evidence, and stops on successful generation plus successful exact replay. Credentials rotate across calls/retries and are loaded only into environment variables. Reports must distinguish generation success, replay equality, replay task success, and unverified training generalization.

Key files: `encoder_wrist.py` (transform/feedback), `encoder_wrist_study.py` (API orchestration), `skills/encoder_wrist/SKILL.md` (API instruction), `suite_physics.py` (optional adapter), `saved_token_replay.py` (independent decoder-only evidence).
