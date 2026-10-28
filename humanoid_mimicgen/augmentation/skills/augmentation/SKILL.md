---
name: augmentation
description: Use when a user supplies a G1 Dex3 SONIC demonstration job for reproducible local trajectory augmentation, execution, verification, and export.
---

# Local trajectory augmentation

The task has a supplied job configuration. Use the registered tools; they run the
existing Python workflow on the local machine. No shell or source-code reading is
needed. Input XY is metres and yaw is radians. Robot and object changes are
independent. This is trajectory retargeting with source actions accessible.

1. Read this skill once. Call `validate_job` to inspect eligibility and evidence.
2. Call `submit_job` once for the full batch. Stable request identity makes retries
   idempotent. The local executor handles sampling, evidence freezing, navigation,
   grasp, recovery, verification and configured export. It retains failed cases.
3. Inspect `get_job` or `get_case` only when the returned result needs explanation.
   The host waits for running jobs locally; do not repeatedly ask for status or
   submit another job because a simulation is slow.
4. Use `export_job` if export was not requested initially. Re-export does not
   rerun physics. Report the dataset index, case count, physical success count,
   failures, timing and actual API usage, using the tools' recorded values.

On missing evidence or an unsupported condition, report the exact reason. Do not
invent input files, loosen gates or change controller settings. An interrupted
case is not a completed trial; local resume continues untouched cases and can
verify complete saved artifacts, but cannot silently replay incomplete physics.

Batch completion differs from physical success. Physical success differs from
visual training readiness. SONIC mode1 command records are not SONIC tokens.
Keep those fields distinct in the final answer. Never claim general reliability
from a small batch or API cost savings without measured usage.
