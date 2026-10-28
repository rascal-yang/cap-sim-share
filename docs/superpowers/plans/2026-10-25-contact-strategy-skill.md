# Contact strategy skill implementation plan

Goal: real API uses an evidence-based contact strategy skill to reproduce sustained button loading with exact token+hands replay, then test other tasks if the button gate passes.
Architecture: independent contact_strategy module and skill, reusable saved-state contact audit; retain native criteria and separate physical contact gate. Existing routes retain defaults. API chooses parameters and acceptance interpretation; local code freezes, executes, diagnoses, renders and replays. Same-condition experience reuse is explicitly not novel discovery. Cross-task tests must retain native task-specific gates and report limits of available interfaces.
Tech stack: Python, MuJoCo3.3.6, SONIC1.0 CPU via humano_python.sh; gpt-6-astra low with existing rotating credentials.

- [x] Implement contact signal audit: actual body coverage, force direction, continuous duration, no changed native checker; unit tests for interrupted hold and omitted-body force.
- [x] Write contact strategy skill and grounded button experience resource; preserve limitations and avoid universal numeric recipes.
- [x] Add isolated API orchestration with compact context, bounded tail up to5s, up to3 candidates; verify immutable plan and raw decoder targets. Unit-test fixed fields/acceptance interpretation.
- [x] Real API button test, frozen contract, output videos at experiment top level, exact token+hands replay, separate native/physical results and cost reporting.
- [x] If button physical>=20N1s and exact replay pass, test representative other tasks using task-native criteria and actual supported preencoder channels. Do not claim all-task success from button.
- [x] Final evidence audit, report all failures and API outages, preserve reproducible command and code commit.

No new permission required: user authorized implementation, real API and conditional cross-task experiments. Do not introduce postdecoder corrections or silently use mode0/IK. Historical manual button case had1.76s>=20N and native false due missing wrist_yaw; it used native-only feedback, so it is not a validated palm-force-feedback controller. No API source credentials appear in artifacts.

Verified results: button same-condition experience reuse1.76s>=20N and exact replay, native false. Other9 tasks reviewed,6physically executed,2native successes with exact replay (apple/plate); no new failed-task recovery. Diagnostics followup ran automatically. Final fact-review caught stale rationale; separate force aggregation fixed with182passing tests. v1 physical evidence frozen; v2 factual rule API-reviewed, no whole-batch physical rerun.
