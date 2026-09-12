# Security Notes

## Credentials

Use an untracked `.env` for local API credentials and never place real values in examples,
prompts, tests, logs, or issue reports. If a key appears in a generated artifact, revoke it
before deleting or sanitizing the artifact.

## CrewAI checkpoints

CrewAI 1.15.x checkpoint snapshots can serialize an LLM object's API key and complete prompt
context. This repository therefore sets `CHECKPOINTING_ENABLED = False` without exposing an
environment switch that can turn it on. Do not enable checkpointing until the installed
CrewAI version has been independently verified to redact both credentials and sensitive
context.

## Private research data

Generated reports, raw or processed procurement exports, local caches, run audits, and custom
context files may contain sensitive commercial information or local filesystem paths. The
provided `.gitignore` excludes the expected locations, but run a secret scanner and review
`git diff --cached` before every publication.
