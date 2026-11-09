# Adoption Report

Agentic Project Kit was added in adopt mode.

## Guardrails

- Existing files were not overwritten.
- Application source files were not rewritten.
- Follow-up work should happen through task files.

## Observed Repository

- Name: novel-harness-public
- Stack: Python
- Application runtime: Python
- Repository tooling: none

## Pre-adoption Gaps

- Missing kit docs: 6
- Missing agent exports: 3
- Agentic config present: no
- Existing task files: 0

## Compatibility

- Contract: unconfigured
- Config schema: missing
- Task contract: none (legacy=0, gated=0, unknown=0)
- Config schema-marker migration required: yes
- Task-contract repair required: no known incompatibilities (see coverage)
- Task parseability: parseable=0; incompatible=0; unreadable=0; not-inspected=0; coverage=complete. Verification heading families alone do not prove parseability.
- Task inspection limits: 4096 entries/directory; 1024 files; 262144 bytes/file; 32 issues/file; 512 characters/issue (long text ends with …).

## Missing Kit Docs

- AGENTS.md
- docs/project.md
- docs/scope.md
- docs/task-system.md
- docs/context-system.md
- docs/progress.md

## Missing Agent Exports

- AGENTS.md
- CLAUDE.md
- GEMINI.md
