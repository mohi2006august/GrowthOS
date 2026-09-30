# GrowthOS: Technical Specification

> An autonomous growth-experiment agent that proposes, launches and analyses marketing experiments under CAC and burn guardrails, with human approval on every spend.
>
> **Concepts:** `CAC`, `Burn Rate`, `LLMs`, `MCP`, `RAG`, `Embeddings`, `LoRA`, `Caches`

## 1. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Agent | Claude with tool use + MCP | Reasoning + tools. |
| MCP servers | Custom (Ads APIs), community servers | Execution. |
| LoRA | PEFT + vLLM | Brand voice. |
| Stats | PyMC / NumPyro | Bayesian analysis. |
| Optimiser | CVXPY | Constrained allocation. |
| App | FastAPI + Next.js | UI. |

## 2. Repository Layout

```text
growthos/
├── metrics/        # cac.py, burn.py, payback.py
├── memory/         # experiment_store.py, retrieve.py
├── agent/          # orchestrator.py, prompts/, tools.py
├── mcp_servers/    # google_ads/, meta_ads/, linkedin_ads/, email/
├── gateway/        # caps.py, policy.py, audit.py
├── copy_model/     # train_lora.py, compliance.py
├── segments/       # embed.py, cluster.py
├── analysis/       # bayes.py, stopping.py
├── budget/         # optimizer.py
└── web/
```

## 3. Data Models

```python
class Experiment(BaseModel):
    id: str
    hypothesis: str
    segment_id: str
    channel: str
    variants: list[str]
    budget_usd: float
    primary_metric: Literal["cac", "cvr", "payback"]
    min_sample: int
    status: Literal["proposed", "approved", "running", "stopped", "concluded"]
    result: "ExperimentResult | None"
    learnings: str | None

class ExperimentResult(BaseModel):
    prob_best: dict[str, float]        # variant -> P(best)
    cac_posterior_mean: dict[str, float]
    expected_loss: dict[str, float]
    decision: Literal["ship", "kill", "continue"]
```

## 4. API Surface

| Method | Endpoint / Tool | Purpose |
|---|---|---|
| tool | search_experiments(query) | Cited past experiments. |
| tool | get_channel_metrics(channel, period) | CAC, payback, spend. |
| tool | propose_experiment(spec) | Store proposal for approval. |
| tool | create_campaign(spec) [approval] | Creates paused campaign via MCP. |
| tool | set_budget(campaign, usd) [approval + cap] | Budget change. |
| POST | /v1/approvals/{id} | Approve/reject. |

## 5. Core Algorithms

### Bayesian A/B

Conversions ~ Beta(1+s, 1+n−s) per variant. CAC per variant = spend / conversions, sampled from the posterior. Ship if P(best) ≥ 0.95 and expected loss < ε; stop early for futility.

### Proposal ranking

`EV = P(win) · Δmetric · reach · margin − test_cost`. P(win) comes from similar past experiments (retrieved) using a Beta prior over win rates.

### Budget reallocation

Maximise Σ new_customers_c(spend_c) (concave response curves from history) subject to Σ spend ≤ B, CAC_c ≤ max_CAC and projected runway ≥ policy months (from burn model).

## 6. Configuration

| Variable | Purpose |
|---|---|
| GOOGLE_ADS_* | Ads API |
| META_TOKEN | Ads API |
| ANTHROPIC_API_KEY | Agent |
| SPEND_CAP_DAILY_USD | Hard cap |
| MIN_RUNWAY_MONTHS | Guardrail |

## 7. Testing Strategy

- Simulated ad environment (synthetic response curves) for agent end-to-end tests.
- Guardrail tests: no path spends without approval.
- Bayesian analysis validated on simulated A/A and A/B tests (false positive rate ≤ 5%).

## 8. Performance Budgets

- Weekly proposal run < 5 min.
- Analysis refresh hourly.

## 9. Local Setup

```bash
uv sync && pnpm i
make sim
make dev
uv run pytest
```
