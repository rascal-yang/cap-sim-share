# Restore stable baseline and revalidate proven solutions first

Goal: restore the latest comparable three-success baseline (20261023_1102) and make one behavioral improvement: preserve an authenticated successful candidate as the initial solution for an identical condition.

The user explicitly authorized restoration and delegated selection of the most important change. Chosen principle: do not discard a known feasible solution before measuring it. Alternatives were adding raw diagnostics or compacting model context; both are useful but do not directly prevent loss of successful behavior.

- [x] Preserve autonomous version in Git checkpoint 71f7c28 and archive branch.
- [x] Restore task orchestration and skill from hash-verified 1102 snapshots. Retain three-credential transport; leave autonomous query helper archived/inactive.
- [x] Add tests for authenticated seed binding, identical-condition enforcement, and no false local acceptance.
- [x] Execute the exact prior API decision locally before any new planning. Native + independent success AND identical scene/full physical trace allow a clearly labeled local revalidation, with no new API review claimed. Any failure or changed trace enters the restored API review/retry path.
- [x] Preserve immutable successful source; never overwrite it with failed candidates. Do not use a cached success as a fresh physical result.
- [x] Run unit tests and three fresh physical regressions with frozen per-case contracts; audit raw traces, mode, prefix, providers, and native criteria.

Scope: same task/episode/source/shift only. This is a regression revalidation feature, not evidence of new perturbation transfer. New conditions still use the restored API path. No threshold, control or hand-timing tuning. Actual training temporal quality remains pending.
