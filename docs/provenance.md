# Provenance and review

Each frozen run records source and prompt hashes, chunk inventory, requested and
resolved model identity, decoding settings, dependency hashes, usage metrics,
and artifact hashes. Credentials and raw provider configuration are excluded.

Machine drafts and human-final text are separate layers. A deterministic PASS is
a structural gate, not literary certification. Sample approval is explicit
before a whole-book worker can run; failed and partial chunks remain inspectable,
and workers can pause, resume, retry, and continue from committed artifacts.

Unknown model identity, missing hashes, and context-horizon violations fail
closed. Provenance supports reproducibility; it does not claim that a model
interpretation is true.
