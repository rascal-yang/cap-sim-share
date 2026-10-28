# Experience Dataset Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. User approved the design and requested implementation; continue without another plan approval.

**Goal:** Build and populate an all-task, evidence-linked experience dataset with separate orchestration/control records, bounded retrieval and automatic collection.
**Architecture:** Add an isolated standard-library package under humanoid_mimicgen/experience_dataset. Immutable JSON revisions are authoritative; SQLite is a rebuildable search cache. A durable collector observes producer artifacts without editing hash-bound controllers; an explicit planner wrapper supplies frozen retrieval context.
**Tech Stack:** Python 3.10+, SQLite, unittest/pytest, existing pinned HUMANO environment for checks only.
**Spec:** docs/superpowers/specs/2026-10-26-experience-dataset-design.md

## Global Constraints
- All tasks, all attempt outcomes; unknown is not false; report coverage denominator.
- Preserve frozen sources, historical IDs, separate orchestration from control; no implicit physics or API calls.
- Explicit interface/frame/unit/measurement-window requirements for response classification.
- Never export credentials; restrict evidence access to configured source roots.
- Freeze retrieval snapshots, group related replays/checkpoints; enforce evaluation exclusion.
- Existing untracked historical-contact files belong to another work item.

## Review Focus
- Heterogeneous summaries must not inflate attempts or turn partial records into verified success.
- Path relocation, symlink escape and missing/corrupt evidence must fail safely.
- Interrupted writes, restarts and repeated scanning must not duplicate records or lose work.
- Chinese search must work, while incompatible control interfaces never supply executable advice.
- Evaluation siblings and current attempt must not leak through snapshots or query logs.

### Task 1: Immutable records and response contracts
Files: package __init__.py, schema.py, store.py; tests/test_experience_dataset.py.
Interfaces: validate_record(dict), classify_response(...), Dataset(root).put(record), get(id,revision), rebuild(), snapshot(name).
- [x] Write failing tests for unknown outcomes, reverse/ambiguous response, immutable idempotent revisions, distinct executions and corrupted records.
- [x] Run tests and verify missing-feature failure.
- [x] Implement schema, summaries, atomic revisions, SQLite indexing and snapshot integrity.
- [x] Run tests; commit.

### Task 2: Historical inventory, adapters and collection
Files: ingest.py, collector.py; tests/test_experience_ingest.py.
Interfaces: inventory(source_roots), ingest(dataset, roots), collect_once(dataset, roots), watch(...).
- [x] Write failing tests for suite/workflow/core and unknown/rejected attempts, replay lineage, secrets, migrated paths, symlink escape, interrupted collection.
- [x] Run tests; implement bounded JSON adapters, durable ingestion ledger and deterministic summaries.
- [x] Populate workspace experience-dataset from experiments; produce task coverage and unresolved ledger without physics.
- [x] Run tests; commit.

### Task 3: Search, tools and planner integration
Files: retrieval.py, planner.py, __main__.py; tests/test_experience_retrieval.py.
Interfaces: query_experience(dataset, query), get_experience(dataset,id,revision), ExperiencePlanner(delegate,dataset,snapshot,context), CLI ingest/watch/query/get/snapshot/rebuild/stats/run.
- [x] Write failing tests for Chinese relevance, kind separation, missing/incompatible control fields, evidence mutation, group exclusion, bounded context, unchanged delegate request and frozen snapshot.
- [x] Implement metadata+character-ngram ranking, query logs, read-only tool schemas, opt-in planner wrapper, producer wrapper and CLI.
- [x] Run real CLI queries against imported data; commit.

### Task 4: Operational delivery and verification
Files: docs/augmentation/experience-dataset.md, retrieval usage policy in the same documentation (no installed Codex skill), dataset reports; tests for integration.
- [x] Document standalone collector lifecycle, adapter coverage, unknown fields, schema and evaluation policy.
- [x] Run package tests and repository default suite; report pre-existing unrelated failures separately.
- [x] Start configured local collector with persistent log/status, verify a second scan is idempotent.
- [x] Independent whole-change review, fix important issues with regressions, commit and report dataset counts and limitations.

## Ledger / rulings
- Existing linked worktree reused; implementation branch feature/experience-dataset.
- Automatic capture uses a durable filesystem collector plus explicit run wrapper rather than modifying frozen-source controllers. Already persisted stage/decision artifacts are events; abrupt exits remain unknown unless the wrapper supplies exit evidence.
- First-version summaries are deterministic and retrieval is local; no embedding or model costs.
- Preflight interfaces: Task 2 uses Task 1 put; Task 3 uses stored revisions/snapshots; Task 4 exercises all. No signature conflicts.

## Completion ledger
- Task 1 complete: 87a4b22; record/response tests watched fail then pass.
- Tasks 2–3 complete: 639b628; ingestion, retrieval, CLI and collector tests watched fail then pass.
- Task 4 complete: ed9b8b5 plus delivery docs; fresh full default suite 108 passed. Reviewer confirmed final crash-recovery fix.
- Ruling: legacy dense control response remains unknown without frame/window/tolerance; targets and measurements are retained separately instead of inventing reversal labels.
- Ruling: compatibility uses frozen source/model identity; Git commit alone is insufficient. Terminal evidence cannot be rebaselined by re-import, even after ledger loss.
- Ruling: use a hard character budget (default12000) rather than claiming exact token accounting without a model tokenizer.
- Ruling: SQLite FTS5 plus Chinese bigrams and task aliases; keep one case per source-group/task/outcome so counterexamples survive replay deduplication.
- Delivered data: ../experience-dataset; frozen snapshot experience_v1_20261026. Inventory2257, mapped1043, partial1214, zero current ingestion errors; total canonical records2282 includes retained older source discoveries.
- Running collector details: ../experience-dataset/collector/service.json. Explicitly no boot restart or automatic enabling of retrieval in existing benchmarks.
- No physical success-rate evaluation or external API calls performed.
