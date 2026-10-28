# Four-task position generalization

This isolated route compares fresh restarts and complete-state checkpoint retries at ten frozen target positions per task. It uses the existing successful replayable chair, apple, button and plate plans. It does not alter other generation routes.

`setup_generalization.py` creates a new Shanghai minute-ID experiment under month/day and freezes 40 specs and the common budget policy. Run it with the pinned `scripts/humano_python.sh`:

```bash
PYTHONPATH=. scripts/humano_python.sh scripts/setup_generalization.py \
  --workspace /home/kinetra-4090-01/code/zijian \
  --baseline-root /path/to/rollback_equal_budget \
  --apple-root /path/to/rollback_apple_expanded_queue \
  --plate-case /path/to/successful_plate_case
```

Then a single command independently performs real API planning and serial physics, without agent intervention:

```bash
python scripts/run_generalization_benchmark.py \
  --config /new/experiment/config.json --provider /new/experiment/provider.json \
  --credentials /private/primary.toml --secondary-credentials /private/secondary.toml \
  --additional-credentials /private/third.toml
```

Each task has center, four cardinal2cm offsets, four diagonal2cm offsets and +4cmX, relative to the established +5mmX target scene. Initial robot and table stay fixed; original-token approach stays fixed. All model numerical arrays/scalars, simulator options and initial dynamic state are checked against baseline, permitting only declared target translation. This is trajectory retargeting with batched API candidate planning, not online API reflection or independent skill generation.

Per condition, API emits12 ranked bounded candidates, candidate0 exact successful baseline. Both arms receive the same queue. Full actual planning and common seed cost count toward each120s deadline; each arm stops on first in-budget success. Atomic overruns finish but are not budget successes. A failed seed ending before checkpoint explicitly falls back to restart. Checkpoints are never reused across positions.

Every observed successful candidate (including late finishes) is independently executed from the beginning and replayed from saved tokens plus final hand targets. Every successful arm trace must match the full execution exactly. `metrics.success` is provisional; `verified_success` includes this gate. A shared candidate can reuse one full/token proof, but cannot omit comparing either arm's trace.

Plate's controller profile is named `apple` for compatibility with the existing paired-wrist compiler; its actual task/dataset/native criterion remains Plate. Button uses approved exact palm-mesh criterion, reports native separately, and never counts wrist-shell contact as palm contact. Body PD targets equal raw decoder targets throughout; independent fingers are saved as separate supervised commands.

Completed conditions are journaled and skipped on resume. An incomplete condition stops resumption rather than silently overwriting evidence. Unexpected preflight errors are kept as errors, not physical task failures. No automatic experiment is authorized by importing modules.

After the run:

```bash
python scripts/report_generalization.py /new/experiment
PYTHONPATH=. scripts/humano_python.sh scripts/audit_generalization.py /new/experiment
PYTHONPATH=. scripts/humano_python.sh scripts/render_generalization.py /new/experiment
```

Rendering restores saved states without physics and preserves source capture; shallow MP4s carry experiment ID and Chinese labels. It does not establish physical success or training-ready visual quality. The report includes planning/simulation costs, unknown API usage and verified task-condition rates. Review rendered images/temporal quality separately before training export.

`generalization_numeric` is a versioned prompt/validator repair for norm and control-tick errors. It changes no returned parameter values or limits. Its launcher is `scripts/run_generalization_numeric.py` with the same arguments. Keep previous planning failures and explicitly distinguish restarted operational budgets from total end-to-end API cost. Network retry and semantic validation currently share the provider schedule; transport outages can leave no repair opportunity.

Recovery roots are linked via `continuations.json`; reporting overlays only incomplete/error conditions, refusing to replace completed evidence. Use `compare_generalization_pairs.py` to compare every matching nonseed candidate, then `finalize_generalization.py` after report/audit/render/diagnostics/plot finish. These runners are process-isolated CLI entrypoints; do not invoke their module-level monkeypatch scopes concurrently in a shared API process.

The20261025 study completed40 conditions,297 physical records,74 identical matched candidate pairs and26 full/token proofs. Both arms: plate8/10,chair8/10,button4/10 approved palm criterion (native0/10),apple6/10. Equal success rates; matched attempt time reductions12.4%,20.9%,10.1%,27.6% respectively. See `/home/kinetra-4090-01/code/zijian/experiments/2026-10/25/20261025_2022_four_task_positions/FINAL_REPORT.md`.
