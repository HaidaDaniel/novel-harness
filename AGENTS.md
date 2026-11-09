# Agent Instructions

Repository docs and task files are the source of truth.

## Core Rules

- Read relevant docs before coding.
- Work on one task at a time.
- Keep changes small and reviewable.
- Stay inside the task allowed files.
- Do not add dependencies without updating docs/decisions.md.
- Update docs/progress.md when task status changes.
- Use `<repo>/tmp/apk/` for intermediate research, reproducers, temporary scripts, and notes; keep required reports, evidence, and final artifacts outside ignored scratch.
- Match CI to project stage and explicit requirements: do not mandate hosted CI for discovery or MVP without a concrete reason; preserve quality.required and approved CI contracts.
- Use the backlog for durable future ideas only when the human asks; never auto-promote backlog items or treat them as executable tasks.
- Use document proposals for optional engineering ideas, research, experiments, and future documentation plans only when the human asks to preserve information; never auto-accept a proposal or treat it as canonical or implemented.

## Default Style

- Communicate in concise, readable sentences; keep commands and technical terms exact.
- Preserve material reasons, limitations, and uncertainty: concision must not delete correctness-relevant detail or reduce necessary task verification.
- Caveman or any specialty style is only used on an explicit named human request for this task or a deliberate persistent agentStyle configuration; a generic request to be brief or save tokens is not that authorization.

## Task Workflow

- Register agent before task work: pnpm exec apk agent register --id <id> --platform <platform> --model <model>.
- Claim tasks with registered owner: pnpm exec apk claim <task-id> --owner <agent-id>.
- Before claiming another mutable task in the same checkout, release the current doing/review task first.
- Use a separate Git worktree for true parallel mutable tasks.
- Tasks in `doing` and `review` states require a registered owner.
- Use pnpm exec apk release, pnpm exec apk block, pnpm exec apk review, pnpm exec apk done, pnpm exec apk cancel with --owner.
- Task state changes are protected by .tasks/.apk.lock.
- Use the current task file as the execution contract.
- Read listed context files before editing.
- Do not touch forbidden files.
- Run verification commands before marking work done.
- During implementation, run focused checks for changed behavior. Commit the final candidate, then run canonical task verification once; repeat only after changes, failures, or missing evidence. Do not separately repeat checks already composed by that verification.
- Choose verification from the host repository and task risk. Avoid overlapping aggregate commands; coverage that executes the full suite already proves that suite ran. APK does not mandate its own test tools, full application tests for docs-only changes, or release checks for every task.
- Independent reviewers inspect the complete candidate diff and current trusted verification evidence, then run focused counterexamples for unresolved risks. Do not automatically rerun the full suite; missing, stale, failed, or untrusted evidence still requires verification.
- Human/operator decisions are explicit out-of-band authorization; request them, and record only decisions the operator actually communicated. Never self-authorize an accept-current decision, grant yourself review passes, or synthesize human consent.
- Update the task if scope must expand.
- When policy requires review, the primary agent may automatically launch a separate read-only reviewer with a different registered identity and isolated context; do not pause for routine user confirmation.
- The implementation owner cannot certify its own candidate. On changes_requested, continue fix -> verify -> fresh review; on pass, continue gate -> done.
- Commit task-owned implementation and fix changes before final verify/review/gate so candidate-bound evidence refers to the committed candidate; a successful handoff must not leave task-owned changes uncommitted.
- A task is not one commit. After the reviewed candidate commit, `apk done` may change tracked task Markdown; commit that lifecycle-only bookkeeping change separately. Never amend, rebase, or squash the candidate commit once evidence or provenance is bound to its SHA.
- Stage only files attributable to the current task. Never run `git add -A` or `git add .` to absorb unrelated or pre-existing dirty state, and never discard, rewrite, force-push, or amend unrelated work.
- Leave unrelated and pre-existing dirty changes untouched; do not include them in the task commit and do not create an empty commit when the task has no task-owned changes.
- Report the resulting commit SHA(s) in the final handoff. If a required commit fails (hook, conflict, identity, permission), surface the blocker and do not report a clean successful handoff.
- Blocked, released unfinished, canceled, or failed tasks do not require a completion commit; an explicit checkpoint commit is allowed only when policy requires it. APK never runs git commit, add, push, or rm itself.

## Worker Contract

- Use the vendor-neutral apk-worker-v1 package; role is implement, review, fix, or verify.
- Return a JSON-compatible result with runId, status, evidence references, and reason.
- Include commitIds, diffId, provenance identities, and reviewFindings when available.
- Keep vendor or harness identity separate from the worker role.

## Architecture Rules

- Keep source of truth in repository docs and config.
- Generate exported agent files from neutral policy content.
- Keep CLI commands thin and task-driven.
- Avoid tool lock-in.
- Document architecture changes in docs/decisions.md.

## Local Model Guidance

- Provide exact context files.
- Keep allowed files narrow.
- Avoid architectural inference.
- Avoid unrelated refactors.
- Prefer explicit steps and acceptance criteria.

## Context Files

- `AGENTS.md`
- `docs/project.md`
- `docs/scope.md`
- `docs/architecture.md`
- `docs/task-system.md`
- `docs/context-system.md`
- `docs/decisions.md`
