"""Core data models shared across the system (TECHNICAL.md section 3)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

PrimaryMetric = Literal["cac", "cvr", "payback"]
# "rejected" extends the spec's five states so a declined proposal stays on record.
ExperimentStatus = Literal["proposed", "approved", "running", "stopped", "concluded", "rejected"]
Decision = Literal["ship", "kill", "continue"]
Outcome = Literal["win", "loss", "inconclusive"]

OPEN_STATUSES: tuple[ExperimentStatus, ...] = ("proposed", "approved", "running")
FINISHED_STATUSES: tuple[ExperimentStatus, ...] = ("stopped", "concluded")


class ExperimentResult(BaseModel):
    prob_best: dict[str, float]
    # None for a variant with no conversions yet: its CAC is undefined, not zero.
    cac_posterior_mean: dict[str, float | None]
    # Relative loss (0-1) in the primary metric from choosing this variant if it is not the best.
    expected_loss: dict[str, float]
    decision: Decision

    winner: str | None = None
    reason: str = ""
    summary: str = ""
    cvr_posterior_mean: dict[str, float] = Field(default_factory=dict)
    cvr_interval: dict[str, tuple[float, float]] = Field(default_factory=dict)
    cac_interval: dict[str, tuple[float, float] | None] = Field(default_factory=dict)
    payback_months: dict[str, float | None] = Field(default_factory=dict)
    prob_beats_control: dict[str, float] = Field(default_factory=dict)
    # Relative change in the primary metric against control (negative is better for CAC).
    lift_vs_control: dict[str, float] = Field(default_factory=dict)
    samples: dict[str, int] = Field(default_factory=dict)
    conversions: dict[str, int] = Field(default_factory=dict)
    spend_usd: dict[str, float] = Field(default_factory=dict)
    look: int = 0
    ship_threshold: float | None = None
    analysed_on: date | None = None
    # Last click day included; later days are excluded until their conversions are reported.
    data_through: date | None = None


class Citation(BaseModel):
    doc_id: str
    experiment_id: str | None = None
    title: str
    note: str

    @property
    def label(self) -> str:
        return self.experiment_id or self.doc_id


class CopyVariant(BaseModel):
    variant: str
    headline: str
    body: str
    cta: str
    model_tag: str
    compliance_passed: bool = True
    violations: list[str] = Field(default_factory=list)


class ProposalMeta(BaseModel):
    """Why a proposal is on the slate: EV = P(win) * uplift * reach * value - test cost."""

    p_win: float
    p_win_interval: tuple[float, float]
    uplift: float
    reach_customers_per_quarter: float
    value_per_customer_usd: float
    ev_usd: float
    test_cost_usd: float
    rationale: str
    citations: list[Citation] = Field(default_factory=list)
    runway_before_months: float | None = None
    runway_after_months: float | None = None
    guardrail_ok: bool = True
    guardrail_reason: str | None = None
    rank: int | None = None
    generated_by: str = "heuristic"
    copy_model_tag: str | None = None
    approval_id: str | None = None


class Experiment(BaseModel):
    id: str
    hypothesis: str
    segment_id: str
    channel: str
    # The first variant is the control.
    variants: list[str] = Field(min_length=2)
    budget_usd: float = Field(gt=0)
    primary_metric: PrimaryMetric
    min_sample: int = Field(gt=0)
    status: ExperimentStatus
    result: ExperimentResult | None = None
    learnings: str | None = None

    tactic: str | None = None
    duration_days: int = Field(default=14, gt=0)
    max_sample: int | None = None
    created_on: date
    started_on: date | None = None
    ended_on: date | None = None
    campaign_id: str | None = None
    spent_usd: float = 0.0
    looks_done: int = 0
    outcome: Outcome | None = None
    observed_lift: float | None = None
    proposal: ProposalMeta | None = None
    creatives: dict[str, CopyVariant] = Field(default_factory=dict)

    @property
    def control(self) -> str:
        return self.variants[0]
