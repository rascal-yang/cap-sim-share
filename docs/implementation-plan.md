# CaP Sim implementation plan

Goal: independently versioned usable pipeline, portable core and explicit simulator/model adapters.
Spec: design.md. Execute inline with executing-plans; user requested implementation, no additional design approval needed.

1. [x] Contracts, evidence evaluator and immutable artifacts. Tests: reject planned-only success, invalid state timing, altered thresholds.
2. [x] Retry engine, prefix reuse, successful baseline and optimization selection. Tests: failed suffix preserved; reset before prefix replay; failed optimization keeps success; budget enforced.
3. [x] JSON/OpenAI/command planners, task CLI, reference import and augmentation. Tests: reference-free requests omit references, augmentation re-executes scene variants.
4. [x] MuJoCo demo and G1/SONIC adapter. Real physics smoke and old trajectory import; no bundled robot assets or weights.
5. [x] Video/vision scene manifests and procedural MJCF builder. Test relative paths, finite dimensions, scale uncertainty and actual model compile/drop stability.
6. [x] Documentation, example tasks, CI, provenance, end-to-end checks and final review. Initialize and commit local repo; do not publish remote.

Decisions: use finite declarative actions rather than execute model-generated Python; GPT does planning/diagnosis, existing controllers handle physics-rate actions. Vision mesh fidelity and calibrated real-world physics are separate from pipeline correctness.
