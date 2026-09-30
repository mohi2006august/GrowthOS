# GrowthOS: Design Document

> An autonomous growth-experiment agent that proposes, launches and analyses marketing experiments under CAC and burn guardrails, with human approval on every spend.
>
> **Concepts:** `CAC`, `Burn Rate`, `LLMs`, `MCP`, `RAG`, `Embeddings`, `LoRA`, `Caches`

## 1. Design Principles

- Humans approve money; agents do the legwork.
- Every recommendation shows expected CAC and runway impact.

## 2. Key User Flows

### Weekly growth review

1. Open slate of proposals.
2. Read rationale + cited past experiments.
3. Approve 2, edit 1, reject 2.
4. Agent sets up paused campaigns.
5. Confirm go-live.

### Budget reallocation

1. Receive recommendation.
2. See runway before/after.
3. Approve within CFO threshold.

## 3. Screens & Surfaces

| Screen | What it shows / does |
|---|---|
| Proposal slate | Ranked cards: hypothesis, segment, EV, cost, citations. |
| Experiment detail | Posterior charts, CAC per variant, decision. |
| Guardrails | Runway policy, caps, approval thresholds. |
| Memory | Searchable experiment library. |

## 4. Wireframe (Primary Screen)

```text
Proposal #3  EV +$18.4k/qtr   cost $2,000   P(win) 0.38   [Approve] [Edit] [Reject]
"Pain-point headline ('close books in 2 days') for SMB finance segment on LinkedIn"
Based on: EXP-112 (+19% CVR, pain-point copy) · EXP-087 (LinkedIn SMB CAC $1,180)
Runway impact: −0.1 mo (policy ≥ 18 mo ✔)   Copy: brand-voice-lora@v4 (3 variants)
```

## 5. Visual Language

| Token | Value |
|---|---|
| Primary accent | `#16A34A` (green-600) |
| Neutral surface | `#0F1117` dark / `#FAFAF9` light |
| Success / Warning / Danger | `#22C55E` / `#F59E0B` / `#EF4444` |
| Type | Inter for UI, JetBrains Mono for code, tokens and numbers |
| Spacing scale | 4 / 8 / 12 / 16 / 24 / 32 px |
| Radius | 8px cards, 6px inputs, full for pills |

## 6. States

| State | Treatment |
|---|---|
| Awaiting approval | Amber badge + approver. |
| Guardrail block | Red: 'would reduce runway to 17.2 mo (< 18)'. |

## 7. Accessibility

- WCAG 2.2 AA contrast in both themes; never encode meaning with colour alone (pair with icon or label).
- Full keyboard navigation with visible focus rings; `/` focuses search, `?` opens shortcuts.
- Charts ship with a data-table fallback and `aria-describedby` summaries.
- Respect `prefers-reduced-motion`; streaming text is announced politely via `aria-live`.
