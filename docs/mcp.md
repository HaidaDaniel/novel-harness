# MCP and API surface

The typed stdio MCP v2 adapter and FastAPI REST API share the same
`TranslationService` facade. They expose library discovery, presets, runner
availability, samples, runs, progress, review, human-final edits, connections,
and EPUB export.

Read projections are bounded. They omit source bodies where a list view does not
need them, raw error text, API keys, environment contents, and private worker or
configuration paths. Connection listings expose provider type, model policy,
availability, and environment-variable names rather than credential values.

Start the REST/Web surface with `novel serve` and the stdio protocol with
`novel mcp`. Offline checks exercise both contracts with synthetic data and
credential-shaped canaries to verify that projections stay bounded.
