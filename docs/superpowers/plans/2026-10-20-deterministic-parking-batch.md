# Deterministic parking and batch execution implementation plan

**Goal:** Implement the user-approved predictive stopping and post-settle correction as deterministic, auditable code; batch runs without live GPT decisions.

**Architecture:** Independent workflow only. `parking.py` fits a scoped scalar stopping-response model from measured stop/settle pairs and owns bounded correction decisions. Runner optionally predicts stopping and re-enters navigation from measured stable state with a checked path. `batch.py` executes explicit request lists serially through existing freeze/execute/reflect APIs, with optional rendering and per-case timing/accuracy/success CSV. Default modes remain unchanged.

**Tech Stack:** Python, NumPy, existing MuJoCo/SONIC mode1 adapter, content-addressed strategy store.

User authorized both methods and implementation in current turn. No further design approval needed. Shared checkout retained because legacy runtime dependencies are untracked; modify only workflow and its plan. Existing 230-file baseline hash audit protects other routes.

- [x] Add tests then measured stop-model fitting, scope/hash evidence, prediction metrics. One current same-policy sample is provisional; do not claim robust calibration. Fit zero-intercept stopping displacement=velocity*tau, record residual, sample count, timestamp and source hashes. Reject mismatched physics/controller/source scope. No site-specific XY correction constants.
- [x] Add bounded correction decision tests, checked current-state base route, reanchored upper body, explicit max attempts/time/tolerance/progress. Stable-state error controls corrections; no new pregrasp veto. No physics resets, policy mode fallback or GPT calls.
- [x] Replace full-history stable-window scans with a bounded deque; verify equivalent results. Keep all recorded evidence.
- [x] Add deterministic serial batch CLI with manifest/request hashes, fresh freeze per run, reflections before next run, machine-readable results and measured durations. Rendering optional to save batch wall time. Never treat interrupted attempts as successes or replay them silently.
- [x] Run software tests and up to three initial physical comparisons with fixed apple/initial state: prediction only, correction only, combination. Each freezes real experiment.json and provider/model evidence. Reflect after each. Further trials only for distinct informative hypotheses.
- [x] Review, render selected successes and all failures, report limitations and preserve reproducible configurations/code.

Final record: experiments/20261020_1710_workflow_parking_final/REPORT.md. Ten physical runs across informative revisions; final three-start batch 2/3 success, all parking errors reduced. Prediction quality gate now rejects insufficient calibration; no promotion to default. Explicit measured blend start, intermediate foot screening and duration limit reviewed and physically retested.
