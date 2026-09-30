# GrowthOS: Product Requirements Document

> An autonomous growth-experiment agent that proposes, launches and analyses marketing experiments under CAC and burn guardrails, with human approval on every spend.
>
> **Concepts:** `CAC`, `Burn Rate`, `LLMs`, `MCP`, `RAG`, `Embeddings`, `LoRA`, `Caches`

## 1. Overview

GrowthOS is an agent system with three loops. (1) Research: RAG over past experiments, win/loss notes, customer interviews and competitor pages; embedding-based customer segmentation. (2) Execution: MCP tools for ad platforms, email and landing-page builders; copy written by a brand-voice LoRA; every launch and budget change requires human approval. (3) Learning: Bayesian analysis of experiment results, CAC/payback per variant, and budget reallocation recommendations constrained by a burn/runway policy. Caches cut repeated research and generation costs.

## 2. Problem Statement

Growth teams run too few experiments because each needs research, copy, audience set-up, budget approval and analysis. Meanwhile spend drifts, CAC creeps up and nobody connects channel decisions to runway. AI can generate ad copy, but nothing closes the loop from hypothesis to measured CAC impact to budget reallocation with financial guardrails.

## 3. Target Users & Personas

| Persona | Needs |
|---|---|
| Head of growth | 10x experiment velocity without losing control. |
| Founder/CFO | Growth spend never breaks runway policy. |
| Marketer | Faster copy and set-up; focus on judgement. |

## 4. Goals

- 3x experiments per month per marketer.
- Reduce blended CAC 15% in 2 quarters.
- 0 spend changes without approval; runway policy never violated.

## 5. Non-Goals

- Fully autonomous spending.
- Replacing brand/creative strategy.

## 6. User Stories

- As head of growth, I get a weekly slate of 5 ranked experiment proposals with expected CAC impact and cost.
- As a marketer, I approve 'Test pain-point headline for SMB segment on LinkedIn, $2k, 14 days' and the agent sets it up.
- As a CFO, I set 'minimum runway 18 months' and the agent's budget recommendations respect it.
- As growth lead, the analysis reports 'Variant B: CAC −22% (94% probability better), payback 6.1 → 4.8 months'.

## 7. Functional Requirements

| ID | Priority | Feature | Description |
|---|---|---|---|
| FR-01 | P0 | Metrics foundation | Channel CAC, payback, LTV and burn/runway (CACLens + RunwayPilot style). |
| FR-02 | P0 | Experiment memory (RAG) | Past experiments, results and learnings indexed; cited in new proposals. |
| FR-03 | P0 | Proposal agent | Generates hypotheses ranked by expected value = P(win) × impact × reach − cost. |
| FR-04 | P0 | Approval workflow | Every launch/spend change requires human approval with a preview. |
| FR-05 | P1 | MCP execution tools | Google/Meta/LinkedIn Ads, email (Customer.io), landing pages (Webflow) via MCP servers. |
| FR-06 | P1 | Brand-voice LoRA | Copy generator adapter trained on approved historical copy + performance weighting. |
| FR-07 | P1 | Segmentation | Customer/lead embeddings clustered into segments with LLM-written personas. |
| FR-08 | P1 | Bayesian analysis | Beta-binomial conversion models, CAC distributions, stopping rules. |
| FR-09 | P1 | Budget guardrails | Reallocation optimiser under runway and max-CAC constraints. |
| FR-10 | P2 | Caching | Semantic cache for research queries; generation cache for copy variants. |

## 8. Non-Functional Requirements

- Full audit trail of proposals, approvals and spend.
- Spend caps enforced at tool level (not just prompt).
- Explainable recommendations with citations.

## 9. Success Metrics

| Metric | Target |
|---|---|
| Experiments/month/marketer | 3x baseline |
| Blended CAC change | −15% in 2 quarters |
| Guardrail violations | 0 |
| Proposal acceptance rate | ≥ 40% |

## 10. Release Plan

| Phase | Scope |
|---|---|
| M1 | Metrics + experiment memory + proposal agent (read-only). |
| M2 | Approval workflow + MCP ad tools (paused-by-default launches). |
| M3 | Bayesian analysis + budget guardrail optimiser. |
| M4 | Brand LoRA + segmentation + caching. |

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Agent overspends or targets wrongly | Hard caps in MCP gateway, campaigns created paused, approval required to go live, auto-pause on CAC anomaly. |
| Peeking/false positives in experiments | Bayesian decision rules with minimum sample and pre-registered metrics. |
| Brand damage from generated copy | Brand LoRA + compliance checker + human approval. |

## 12. Open Questions

- Should the agent be allowed to auto-pause (but not start) underperforming campaigns?
