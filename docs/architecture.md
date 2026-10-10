# Architecture

```mermaid
flowchart LR
  CLI[CLI / TUI] --> S[TranslationService]
  WEB[FastAPI + static Web UI] --> S
  MCP[Typed MCP v2] --> S
  S --> R[Run factory + persistence]
  S --> W[Persisted worker]
  W --> E[Stage executor]
  E --> A[Runner adapters]
  A --> P[OpenAI-compatible / OpenCode / Codex]
  S --> V[Validation + metrics]
  S --> H[Human final + Book Bible]
  H --> X[EPUB/export artifacts]
```

`TranslationService` is the application facade for every interface in the
current Japanese → Russian product scope. Adapters
validate input and delegate; they do not create a second translation engine.
The run factory freezes source and prompt hashes, chunk inventory, pipeline,
runner configuration, decoding settings, and safe execution identity. Workers
commit chunk results atomically and resume from the frozen manifest.

Credentials and raw external configuration are excluded from provenance.
Human-final text and machine output remain separate layers.
