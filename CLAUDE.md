# CLAUDE.md: GrowthOS

Guidance for Claude Code (and other AI assistants) working in this project.

## Project Summary

An autonomous growth-experiment agent that proposes, launches and analyses marketing experiments under CAC and burn guardrails, with human approval on every spend. GrowthOS is an agent system with three loops. (1) Research: RAG over past experiments, win/loss notes, customer interviews and competitor pages; embedding-based customer segmentation. (2) Execution: MCP tools for ad platforms, email and landing-page builders; copy written by a brand-voice LoRA; every launch and budget change requires human approval. (3) Learning: Bayesian analysis of experiment results, CAC/payback per variant, and budget reallocation recommendations constrained by a burn/runway policy. Caches cut repeated research and generation costs.

## Read First

- `PRD.md`: what we are building and why; requirement IDs (FR-xx) are referenced in commits.
- `SYSTEM_ARCHITECTURE.md`: components, data flow and failure modes.
- `TECHNICAL.md`: stack, data models, APIs and algorithms.
- `DESIGN.md`: UX flows, screens and visual tokens.
- `BRAIN.md`: concept primers, experiments and the decision log.

## Commands

| Command | What it does |
|---|---|
| `uv sync && pnpm i` | Install |
| `make sim` | Run agent against simulated ad environment |
| `make dev` | Web + workers |
| `uv run pytest` | Tests |

## Code Conventions

- Spend-affecting tools are marked `requires_approval` in code and policy; tests enforce both.
- Keep functions small and typed; no silent `except`. Log with structured fields.
- Every new feature references its FR-ID in the PR description and ships with tests.
- Secrets only come from environment variables. Never hard-code keys or commit `.env`.

## Gotchas

- Ad platform conversion data is delayed 1–3 days, so account for it in stopping rules.
- Use fully loaded CAC for budget decisions.

## When Unsure

- Prefer the simplest design that satisfies the PRD; record bigger trade-offs in `BRAIN.md` → Decision Log.
- Run the eval / benchmark suite before claiming a quality or performance improvement.
- Ask before adding a new paid dependency, external service or data store.
