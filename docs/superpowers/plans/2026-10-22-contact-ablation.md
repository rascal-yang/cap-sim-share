# Contact-entry controller comparison

Approved scope: apple, button, chair sit, kick. Isolated augmentation worktree only.

- [x] Implement and test explicit token-prefix / mode1-through-end schedule, preserving decoder history and source hands.
- [x] Add read-only signed root, orientation, leg, wrist and hand diagnostics aligned to pre-control source frames.
- [x] Add serial API-owned matched A/B orchestration; immutable parameters from accepted replay and annotation; alternate credentials, five retries.
- [x] Freeze runtime/model/source/API evidence before each physics case. Run both paths fresh, render and let API review each result.
- [x] Report native and API acceptance separately, times, tokens, prefix duration and limitations.

Inputs: historical accepted replay decisions and annotated bounds. Outputs: frozen cases, traces, diagnostics, MP4, API decisions and comparison report. No offsets or task-specific repairs in this diagnostic comparison. A failure does not block the other arm: both are diagnostic experiments, not downstream augmentation. No training-ready claim.

Validation: 118 tests passed; 8 fresh physical cases audited, 4 matched pairs, 8 full MP4 decodes. Native success 1/4 tasks per arm. API 19 requests, 12 completed, 276246 known tokens, 7 unknown usage. Report: /home/kinetra-4090-01/code/zijian/experiments/2026-10/22/20261022_1719_contact_entry_ablation/REPORT.md
