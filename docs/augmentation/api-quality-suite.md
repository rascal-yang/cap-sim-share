# API-owned quality gates for all local HUMANO tasks

Entry point (from this worktree):

```bash
python scripts/run_quality_suite.py --config /absolute/path/config.json --provider /absolute/path/provider.json
```

The launcher reads the existing `../api/openai_config.toml` credentials into process environment variables. It does not save them into experiment records. Physics runs through `scripts/humano_python.sh` with the required SONIC CPU profile.

Config fields: `output_root`, `dataset_root`, `tasks` (explicit task directory names), `episodes` (allowed source IDs), `attempts_per_stage` (bounded physical attempts). Provider configuration uses the existing Responses transport; this experiment uses `gpt-6-astra`, reasoning effort `low`.

The local scheduler supplies the next stage's evidence to the API. Each decision includes action, hypothesis, rationale, episode, explicit tail duration, wrist translation and inclusive frame endpoints. Every executable decision is checked against the exact completed API function call, and its request/response hashes are bound into the frozen physics contract. The supervising coding assistant supplies capabilities, not hidden task-specific repair parameters.

Stages: source selection → saved-state inspection → exact original-token replay → API contact annotation → zero-offset mode1 reuse → four object-only 1 cm offsets. Every physical acceptance requires complete execution, native success, independent saved-state checker agreement, and no safety stop. The API may reject a result that satisfies these necessary checks if it is unsuitable for the next stage. Annotation includes both endpoints; it does not by itself establish a release action. Zero-offset failure blocks shifted augmentation.

The suite retains access to original trajectories and is a trajectory-retargeting experiment. Mode1 uses source leg references, source torso and object-relative world wrist targets. This is not a generic foot/pelvis retargeter, newly learned task policy, or proof of transferable strategy. Original hand commands remain unchanged. Unsupported corrections become explicit API capability requests. No threshold relaxation, changed physics or silent control fallback is allowed.

Local correction capabilities are bounded by this experiment's declared budgets: 0–3 seconds of final intent, and a common wrist translation of at most 5 cm during an API-selected reference interval. These bounds are not recommended strategies. The API chooses whether any such operation is justified. Corrections require new experiment identities. Offsets reuse exactly the accepted zero-offset parameter set.

Artifacts: task inventory, saved-state evidence, API requests/responses/attempt usage, decisions, annotation, frozen per-case physics contracts, state/control/next-state trajectories, native metrics and source snapshots. Root report includes all ten task outcomes, including blocked tasks. Data remain `training_ready=false` until visual review. The source dataset is never modified.

A completed task result is cached. An interrupted physical action must not be silently replayed under the same identity; inspect its started marker and retained evidence, then use a fresh experiment directory for a new attempt. API transport failures are recorded with unknown usage when the provider omits usage. Never infer those tokens as zero.

For batch execution with two independent API sessions and serialized physics:

```bash
python scripts/run_quality_batch.py --config /absolute/path/config.json --provider /absolute/path/provider.json
```

Study and task leases prevent concurrent schedulers from sharing an output directory. Do not remove lock files to bypass an owner. Before restarting, verify the old process has exited. Completed physics is cached separately from rendering, so rendering failures must not trigger another integration. A task with `result.json` is skipped; to resume a transport error, preserve that result under a timestamped archive name, then rerun the same configuration. Do not archive quality rejection or exhausted-budget results merely to obtain another attempt. Cached requests must match the original context and skill, and completed decisions must retain valid API provenance.

The API can explicitly choose a termination guard from the measured source posture envelope. This changes stopping classification, not native task acceptance. The chosen bounds are frozen in the experiment contract. A source posture that conflicts with the guard blocks preflight; a guard trigger alone is not proof of falling.

Every new physical case produces read-only contact reconstruction at 200 Hz and a validated three-view MP4, plus uniform and contact-focused image sheets attached to API review. Render inspection forbids integration and restores the MuJoCo entry points afterward. Stills and MP4 decode checks do not certify temporal training quality.

Export the evidence using the HUMANO environment:

```bash
bash scripts/humano_python.sh -m humanoid_mimicgen.augmentation.suite_export /absolute/path/output_root
```

`QUALITY_REPORT.md`, `quality_summary.json`, `dataset_index.json`, and `strategy_index.json` separate physical success, API acceptance, unknown request usage and unexecuted stages. API time can overlap across workers. A cross-day resumed study also contains idle time; report batch execution intervals separately from elapsed calendar time. Service overload, timeout and model mismatch are infrastructure failures, not failed robot actions. Never silently substitute another model or have the supervisor manufacture a missing strategy decision.

Credential failover and service-error resume:

```bash
python scripts/run_quality_batch.py --config /absolute/path/config.json --provider /absolute/path/provider.json --secondary-credentials "../api/openai_config 2.toml" --api-routing alternate --resume-errors
```

With a secondary configuration, the default `--api-routing alternate` allocates each actual request from a persisted study-wide sequence: primary, secondary, primary, secondary. Task processes share a file lock, so concurrent tasks and retries participate in the same sequence. Cached decisions do not allocate a request. Each decision allows one initial request plus five retries in total; failures never bypass stage acceptance. A crash after allocation can leave an unused sequence number, which is preserved rather than reusing an unknown request identity.

The earlier policy remains available as `--api-routing failover`: each credential gets one initial request plus five retries, then switches to the second credential, which stays selected within that task process. Both policies preserve model and reasoning effort. Invalid completed tool arguments have a separate three-response correction bound. Each actual request records its provider slot, routing sequence (when alternating), endpoint, timestamps, usage and outcome; secrets remain only in environment variables. Both credentials may use the same gateway, so rotation cannot guarantee recovery from a shared gateway outage.

`--resume-errors` runs under the study lease and archives only task results with explicit API service/transport errors. Completed tasks, quality rejections, other tool errors and physical caches remain intact. Earlier root summaries/timing are archived. No extra physics retry budget or relaxed quality criterion is granted by credential failover.
