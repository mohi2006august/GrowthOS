# GrowthOS

> An autonomous growth-experiment agent that proposes, launches and analyses marketing experiments under CAC and burn guardrails, with human approval on every spend.

**Concepts:** `CAC`, `Burn Rate`, `LLMs`, `MCP`, `RAG`, `Embeddings`, `LoRA`, `Caches`

GrowthOS is an agent system with three loops. (1) Research: RAG over past experiments, win/loss notes, customer interviews and competitor pages; embedding-based customer segmentation. (2) Execution: MCP tools for ad platforms, email and landing-page builders; copy written by a brand-voice LoRA; every launch and budget change requires human approval. (3) Learning: Bayesian analysis of experiment results, CAC/payback per variant, and budget reallocation recommendations constrained by a burn/runway policy. Caches cut repeated research and generation costs.

## Docs

| File | Contents |
|---|---|
| [BRAIN.md](BRAIN.md) | Concept primers, research, experiments, decision log |
| [CLAUDE.md](CLAUDE.md) | Instructions for AI coding assistants |
| [DESIGN.md](DESIGN.md) | UX flows, screens, wireframes, visual language |
| [PRD.md](PRD.md) | Problem, users, requirements, metrics, roadmap |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | Components, data flow, deployment, failure modes |
| [TECHNICAL.md](TECHNICAL.md) | Stack, data models, APIs, algorithms, setup |
