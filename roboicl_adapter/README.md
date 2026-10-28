# Independent RoboICL-style HUMANO pilot

This directory adds an independent entrypoint; existing cap-sim source, CLI,
physics/control backend and pipeline are not modified. `run.py` calls the existing
`run_pipeline` with empty `references`, using a separate mailbox planner and an
observation-only backend subclass. No reference-recovery gate or stored action
prefix is invoked.

The pilot uses G1/Dex3, the original HUMANO initial state, native acceptance,
SONIC 1.0 CPU mode1 and existing contact/collision gates. Run via
`scripts/humano_python.sh`. Do not launch without a newly frozen experiment
record and loaded-session/model-hash evidence. The entrypoint verifies the
contract and frozen sources before initialization; actual physics signatures
are checked before any generated command.

## Planner isolation

One Codex subagent per condition is started with `fork_turns="none"`. It receives
only its mailbox path and then reads each explicit system/input packet and
permitted images. It generates one finite JSON action per observed feedback.
It cannot inherit the surrounding parent conversation. File access restrictions
are instructional, not an OS sandbox. These are actual model decisions within
Codex, **not external model API calls**. No deterministic action-generation
script substitutes for the model.

`adapter.py` filters out source resource paths, references, raw tokens, body
keyframes, prefixes and hidden skills from the executing request. Runtime
validation rejects those action channels and forces measured feedback after
every short command. A bounded memory retains the first and latest committed
interactions with explicit gap markers. Task failure produces a separate
model-authored, evidence-bound reflection.

## Differences from original RoboICL

- Demonstration is extracted measured state transitions; source token/mode0
  commands cannot be truthfully labeled as mode1 actions. No full action
  equivalence or original-paper reproduction is claimed.
- Current observation includes privileged object geometry/contact data, plus
  head and external RGB. It is not the original three-camera RGB-only setting.
- The original zero-speed SONIC planner enumeration 4 is preserved. Encoder
  mode1 is a different enumeration; there is no mode0 fallback.
- The original reset NPZ includes controller history/last_action. This initial
  state is preserved, but no recorded token prefix is executed.
- Native height acceptance is preserved; it does not certify a stable grasp
  or training-ready RGB.

## Files

- `adapter.py`: context boundary, bounded interaction memory, mailbox transport,
  decision validation and frozen launch gate.
- `prepare.py`: read-only extraction/rendering of a physically validated source;
  declares four lateral conditions, +/-10 and +/-20 cm.
- `scene.py`: physics signatures and camera observation rendering; no mj_step.
- `run.py`: independent runner, observation/audit subclass, original outer loop.
- `test_adapter.py`: offline tests for forbidden commands, context leakage,
  memory receipts and frozen-code/contract tampering.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kinetra-4090-01/HUMANO/.venv/bin/python \
  -m pytest -q roboicl_adapter/test_adapter.py tests
```

Initial pilot evidence: `../experiments/20261014_1957_roboicl_lateral` relative
to the cap-sim repository root. The frozen contract, actual per-turn requests,
responses and provenance are the authoritative execution record.
