# Humanoid contact reuse implementation plan

> Agent workers use focused implementation tasks and independent review. The user
> requested autonomous completion; no additional design approval is required.

Goal: implement and test the isolated pipeline described in [design.md](design.md).
Technology: Python, NumPy, SciPy, existing MuJoCo/HUMANO runtime, GPT-6.

Global constraints: only add files inside humanoid_mimicgen; never modify other
pipelines. Tests live in humanoid_mimicgen/tests. All new experiment artifacts use
minute-resolution IDs. No physics before the complete frozen contract.

- [x] Dataset and annotation: write failure tests; implement measured-state
  importer, evidence summary, strict annotation validation and GPT-6 Responses
  adapter; test on malformed boundaries, source mismatch and absent contacts.
- [x] Retargeting: write failure tests with hand-computable transforms; implement
  fixed group anchor, transit connections and invariants; verify translation,
  rotation, consecutive lift phases and preserved prefix/fingers/timing.
- [x] Native smoke: implement explicit original-token prefix plus mode1 dense
  adapter, immutable-contract gate and recorded controls. Verify missing evidence
  blocks before constructing a backend. Root freezes actual evidence and performs
  at most one original-scene run; no fallback or variant physics after failure.
- [x] Integrate CLI and docs; import the actual successful apple demonstration;
  get a real GPT-6 annotation, validate it and generate ten offline variants.
- [x] Review and test new modules plus the existing non-physical regression suite;
  inspect plots and any rendered frames; record actual results and durations.

Review focus: q/-q equivalence, no moving-object double transform, half-open frame
coverage and final sample, immutable identity/hash evidence, simulator failure
must remain failure. Missing force data is unknown, not zero.
