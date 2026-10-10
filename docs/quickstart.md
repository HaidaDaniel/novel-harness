# Quickstart

Novel Harness is designed for a local checkout with source books and run data
outside version control. A normal install provides the `novel` CLI and packaged
Web assets. The [README](../README.md) shows the sample-first Web workflow.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,web]'
novel --help
```

## Prepare a local book

Use `resources/example-book/` as a metadata and glossary template. Put UTF-8
chapter files in the configured `source_dir`; do not commit real source books or
generated translations. The sample-first workflow creates a bounded draft for
review before a whole-book run.

```bash
novel --config config.quality.example.yaml library list --json
novel --config config.quality.example.yaml sample --runner local-quality --chapter 1
novel --config config.quality.example.yaml status
novel --config config.quality.example.yaml serve
```

Configure a real model id and endpoint only when making a provider call. The
examples use loopback OpenAI-compatible URLs and environment-variable names;
they contain no credentials.

For the Web workflow, open Library, choose a book and profile, start a sample,
inspect it in Review Desk, approve it explicitly, and then launch the whole-book
worker. The run remains resumable and the final layer can be exported to EPUB.

## Offline checks

```bash
python -m pytest
python -m ruff check src tests
uv build --out-dir dist
```
