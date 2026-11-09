# Context System

The context system tells an agent exactly which files to read before working on a task.

## Context levels

### Level 1

Minimum context:

- `AGENTS.md`
- the current task file
- `docs/project.md`
- `docs/scope.md`
- `docs/architecture.md`

### Level 2

Add the relevant design docs:

- `docs/decisions.md`
- engineering docs related to the task;
- product docs related to the task;
- delivery docs related to the task.

### Level 3

Add source files and supporting repository files:

- likely affected modules;
- tests;
- configuration;
- migrations;
- generated files if relevant.

## Exclusions

The following operational files are not prompt context by default:

- `.tasks/.apk.lock`;
- `.agentic/agents.jsonl`;
- `.agentic/runs.jsonl`;
- `.agentic/agents/**`;
- `.agentic/runs/**`.

## Selection rules

- Prefer the smallest context set that still makes the task safe.
- Do not dump the entire repository into the prompt unless the task truly needs it.
- For local models, be stricter about exact file lists and allowed edits.
