# Fixed-wrist finger pressing pilot

The `finger_study` route keeps the prior encoder-only wrist strategy fixed and lets the API choose right index joint targets. It changes neither SONIC decoder outputs nor body/hand PD gains. The wrist strategy remains closed loop: its force input and resulting reference may respond to new finger contact.

The actual seven-command hand order comes from the robot's `right_hand_actuated_index`, not a hardcoded nominal tuple. In the current Dex3 scene it is thumb0/1/2, middle0/1, index0/1. The API can select index0 and/or index1, a start frame, ramp, offset, signed downward-force integral gain, and offset/rate bounds. Commands are clipped to native joint limits; unselected fingers and the left hand remain source commands. Hand gains remain kp1.5/kd0.1. Both index contact links are monitored at the existing65N cap, extending the earlier proximal-only stop conservatively; the native task checker and table50N stop are unchanged.

Every rollout saves `execution/hand_commands.npz` with float32 `left` and `right` arrays [policy_frames,7]. If an independent saved-token replay finds this file, it uses these exact targets; if a finger-controlled result lacks the file, replay fails rather than silently using old fingers. Replay disables both wrist-reference and finger feedback. Body tokens do not encode hand targets; both streams are required.

Run under system Python so credential TOML parsing stays outside the pinned runtime:

```bash
python scripts/run_finger_study.py --config STUDY/config.json --provider STUDY/provider.json \
  --credentials ../api/openai_config.toml \
  --secondary-credentials '../api/openai_config 2.toml' \
  --additional-credentials '../api/openai_config 3.toml'
```

Config keys: `output_root`, `baseline_case` (completed wrist-only case), `dataset`, `geometry` (read-only current-scene finger kinematics and command-order evidence). Use a new timestamped experiment directory. The route permits three API-planned physical candidates, alternating credentials on request retries. Each case freezes its contract before physics. On transport failure it saves `ERROR.json`, completed progress and a terminal summary; no manual plan is substituted.

Task success, independent-check success, action replay equality and training readiness are distinct. Only native success plus successful exact token-and-hand replay completes this pilot. Failed but reproducible trajectories are retained as diagnostics.
