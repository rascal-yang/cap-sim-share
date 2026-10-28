# Task skills and strategy selection

The outer pipeline is unchanged: reference recovery, generation/augmentation,
feedback execution, physical acceptance and success preservation still use the
existing functions. This feature adds task-level experience and a planner adapter,
not a new controller or a replacement simulator loop.

## What the task skill contains

A task skill separates invariant task requirements from replaceable approach,
stance, wrist/finger configuration and contact-region choices. Store the critical
measured evidence (including object-local contact anchors), applicable geometry,
controller limitations, failure diagnosis rules and short-stage CaP reasoning.
Candidate strategies must be distinguished from observations. A candidate needs
physical validation; adding pseudocode does not make a new executable robot API.

The initial apple skill is at
[`../task_skills/g1_dex3_apple_pick/v0001/skill.md`](../task_skills/g1_dex3_apple_pick/v0001/skill.md).
It includes the validated local-reference strategy and candidate branches for
repositioning, stance/height adjustment and alternative contact regions. The
fixed native-token prefix is not universally mandatory, but replacing it requires
an available runtime interface and fresh physical evidence. `body_keyframes` use
full-body reference mode; do not assume locomotion velocity commands also drive
the legs in that mode.

The existing deterministic apple retargeting script remains a local strategy.
To make different approach/stance/grasp choices, use a GPT or command planner
that consumes the skill and emits actual supported interface actions.

## Version boundary: one attempt, not the whole experiment

- A running attempt uses one exact skill revision through all of its feedback
  steps. Observations and action parameters may still change during that attempt.
- After an attempt, analyze its evidence, revise the skill if warranted, and
  publish a new version. The next attempt immediately selects the latest
  published version. There is no requirement to finish all ten attempts first.
- A new version never erases its parent. Record revision reason, evidence IDs,
  observed/candidate/rejected status and scope. Preserve failures.
- Same total attempt budget, reset semantics, physical parameters, action
  permissions and acceptance apply. A skill revision does not reset the budget.
- `--evolving-skill` automatically selects versions; it does **not** automatically
  synthesize or publish new skill text. The session agent or an explicit skill
  authoring operation must analyze evidence and publish it between attempts.
  If no new version has been published before the next attempt starts, it uses
  the current version without inventing a revision or waiting indefinitely.

## Use an evolving task skill with the existing planner

```bash
python -m cap_sim run \
  --task target_task.json --reference-task original_task.json \
  --planner openai --model YOUR_MODEL \
  --evolving-skill task_skills/g1_dex3_apple_pick \
  --out /path/to/new_experiment
```

Existing API credentials and model configuration are required for an actual GPT
call. No real model call or large-scene physics success is implied by enabling a
skill. Current geometry, obstacle/support information and measured robot state
must be supplied by the task/backend. A skill must identify missing inputs rather
than assume that demonstration geometry still applies.

For a deliberately fixed version, use
`--skill task_skills/g1_dex3_apple_pick/v0001/skill.json` instead.
Both flags can be repeated for different task skills. An evolving task skill
replaces a fixed skill with the same task ID in the effective planner request.
An explicit `--skill` can also accompany `--no-reference`; historical behavior
without explicit skill flags is unchanged.

## What reaches GPT

The normal request still includes current observations, declared controls,
references, contacts when enabled, prior attempt results and the action budget.
The versioned Markdown and strategy records are passed in `skills`.
`task_skill_context` adds exact version hashes, candidate/observed choices,
missing runtime controls and the requested strategy decision fields:

```json
{
  "analysis": {
    "evidence": ["current geometry and measured state"],
    "hypothesis": "why this next stage is appropriate",
    "strategy": {
      "strategy_id": "reposition_then_grasp",
      "applicability": "candidate; state relevant geometry and uncertainty",
      "approach": "route and approach direction",
      "stance": "base position/heading and posture goal",
      "grasp": "contact region, wrist orientation and finger strategy",
      "next_observation_gate": "what must be measured before advancing",
      "abort_or_recovery": "supported response if tracking/contact fails"
    }
  },
  "actions": ["actual allowed action objects, not pseudocode calls"]
}
```

This is a response-format explanation, not an executable action example. The
existing executor still validates actions and physical gates. Extra explanatory
strategy fields are requested by the prompt, not a geometric feasibility proof.
The skill cannot change the runtime contract.

## Evidence and replay

`EvolvingSkillPlanner` records:

- `skill_versions/aNNN_skills.json`: exact selected artifacts, timestamp and prior
  attempt result;
- `skill_versions/aNNN_fMM_request.json`: full effective planner input;
- `skill_versions/aNNN_fMM_decision.json`: output with `skill_usage` hashes;
- the same `skill_usage` is included in the core's saved decision.

The core's `experience.json` remains the original seed experience; it is **not**
the source of truth for dynamically selected revisions. Use the per-attempt
snapshots and decision provenance to audit effective inputs. Keep these logs with
successful trajectories, whose `source_attempt` identifies their execution.
On resume, an existing attempt keeps its saved snapshot even if latest advances.

## Publish a new revision

A draft JSON contains `task_id`, `revision_reason`, `skill_markdown`,
`supported_controls`, `evidence`, `strategies` and `limitations`. Evidence items
need unique IDs, status, scope and a SHA-256 of their source artifact. Observed or
rejected strategies must cite evidence IDs; candidates remain explicitly untested.
The publisher validates structure and hashes; authors remain responsible for
whether cited evidence actually supports a claim.

```bash
python -m cap_sim publish-task-skill \
  --draft updated_draft.json --registry task_skills \
  --parent CURRENT_REVISION_SHA256
```

Omit `--parent` only when creating the first version. Parent checks prevent
accidentally overwriting a newer revision. Publishing creates a new `vNNNN`
directory and updates `latest.json`; never edit a published Markdown/JSON pair
in place. Loading checks the artifact hash and Markdown agreement.

The existing summarizer now accepts the actual interface and measured contact
prior, so draft experience need not be inferred from video or wrist motion alone:

```bash
python -m cap_sim summarize-reference \
  --reference successful_trajectory.json \
  --interface interface.json --contact-prior reference_contact_prior.json \
  --planner openai --model YOUR_MODEL --out /path/to/draft_summary
```

Summary output is an authoring artifact, not automatically a published task
revision. Add task identity, revision reason, strategy evidence links and scope
before publication. The interface requires an actual supported-control contract;
it must not be invented from desired behaviors.

## Validated scope

Tests verify version preservation, parent conflicts, hash checks, evidence status,
planner input enrichment and version selection between attempts. A synthetic run
verifies that a failed attempt on v1 can be followed by a successful fixture on
v2 without changing reset count or budget. This is software integration evidence,
not a new robot grasp result. Large changes in stand position, approach route,
object shape or grasp remain pending actual GPT planning and physics validation.
