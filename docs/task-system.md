# Task System

The task system is the unit of execution for Agentic Project Kit.

Tasks use compact metadata so agents can coordinate without spending much context.

## States

- `todo` - ready to claim.
- `doing` - claimed by one registered agent.
- `review` - implementation done, waiting for review.
- `done` - verified and complete.
- `blocked` - waiting on a decision or dependency.
- `canceled` - no longer planned.

## Owner rules

- `Owner` is a compact registered agent id.
- `Owner: none` is allowed for `todo`, `blocked`, and `canceled`.
- `doing` and `review` require a registered owner.
- `claim`, `release`, `block`, `review`, `done`, and `cancel` require `--owner`.
- Task state changes are protected by transient `.tasks/.apk.lock`.

## Agent registry

Agents register before task work:

```bash
pnpm exec apk agent register --id codex-a --platform codex --model gpt-5.5
```

Registry path:

```txt
.agentic/agents/<agent-id>.json
```

Run log path:

```txt
.agentic/runs/YYYY-MM-DD_<developer-id>_<agent-id>.jsonl
```

Legacy `.agentic/agents.jsonl` and `.agentic/runs.jsonl` are migration inputs only.

## Task rules

- Use one task file per unit of work.
- Keep allowed files narrow.
- Keep forbidden files explicit.
- Include exact context files.
- Include concrete verification commands.
- Choose checks from this repository's tools and the task's changed behavior; APK does not impose its own quality pipeline.
- Match CI to the project stage and explicit policy: discovery and MVP use relevant local checks; recommend hosted CI only when justified; preserve quality.required and approved CI contracts.
- Keep durable future ideas in the backlog only when explicitly requested; backlog cards are not executable tasks and are never auto-promoted.
- Use focused feedback during edits and a non-overlapping final verification set on the committed candidate; coverage already runs its measured tests.
- Documentation-only tasks need applicable contract, link, or example checks rather than unrelated application suites.
- Independent review inspects current trusted evidence and focused counterexamples; rerun broader checks when evidence is stale, failed, missing, or insufficient.
- Do not mark a task done until verification passes.
- Update `docs/progress.md` when task status changes.
- Use `Lane`, `Scope`, `Tags`, and `Parallel` to split work across agents.
