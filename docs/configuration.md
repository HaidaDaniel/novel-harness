# Configuration

Configuration is YAML and is loaded relative to the configuration file.
Portable examples cover a direct OpenAI-compatible runner, OpenCode, Codex, and
local Qwen-style profiles.

Important sections are:

- `novel`: source, output, books, jobs, and optional metadata paths;
- `translation`: language pair, chunk geometry, prompts, glossary, and retries;
- `validation`: structural and segment checks;
- `runners`: provider/model endpoint profiles using an environment-variable name;
- `translation_presets` and `pipelines`: user-facing workflow choices.

Credentials are read from the named environment variable or an ignored `.env`
file. They are not written to manifests, provenance, API responses, MCP
projections, or benchmark artifacts. Source chapters, glossaries, run databases,
and output directories remain explicit local inputs.
