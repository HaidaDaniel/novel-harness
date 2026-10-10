# Public dependency decision

The CLI declares Click directly because the Typer adapter imports Click's shell
completion types. This keeps a base package install functional without relying
on the optional Web extra to pull the transitive dependency.
