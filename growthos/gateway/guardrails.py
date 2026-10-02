"""Burn and CAC guardrails: the policy, and the runway checks every spend passes through.

Environment variables are hard limits. The stored policy (edited on the Guardrails screen)
can only be as strict or stricter: a higher runway floor, lower caps.
"""

from __future__ import annotations

import math
from datetime import date

from pydantic import BaseModel, Field

from growthos.clock import utc_now_iso
from growthos.config import Settings
from growthos.db import Database
from growthos.gateway.audit import AuditLog
from growthos.memory import ExperimentStore
from growthos.metrics import MetricsService, burn


class PolicyViolation(ValueError):
    """A requested policy would loosen a hard limit set in the environment."""


class GuardrailPolicy(BaseModel):
    min_runway_months: float = Field(gt=0)
    spend_cap_daily_usd: float = Field(gt=0)
    max_cac_usd: float = Field(gt=0)
    marketer_approval_limit_usd: float = Field(ge=0)
    allow_auto_pause: bool


class RunwayCheck(BaseModel):
    # None means the company is not burning cash.
    before_months: float | None
    after_months: float | None
    min_months: float
    ok: bool
    reason: str | None = None

    @property
    def impact_months(self) -> float | None:
        if self.before_months is None or self.after_months is None:
            return None
        return self.after_months - self.before_months


class GuardrailService:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        metrics: MetricsService,
        store: ExperimentStore,
        audit: AuditLog,
    ) -> None:
        self._db = db
        self._settings = settings
        self._metrics = metrics
        self._store = store
        self._audit = audit

    # -- policy --------------------------------------------------------------

    def hard_limits(self) -> GuardrailPolicy:
        s = self._settings
        return GuardrailPolicy(
            min_runway_months=s.min_runway_months,
            spend_cap_daily_usd=s.spend_cap_daily_usd,
            max_cac_usd=s.max_cac_usd,
            marketer_approval_limit_usd=s.marketer_approval_limit_usd,
            allow_auto_pause=s.allow_auto_pause,
        )

    def policy(self) -> GuardrailPolicy:
        with self._db.tx() as conn:
            row = conn.execute("SELECT body FROM guardrail_policy ORDER BY version DESC LIMIT 1").fetchone()
        limits = self.hard_limits()
        if row is None:
            return limits
        stored = GuardrailPolicy.model_validate_json(row["body"])
        # Re-clamp on read so a tightened environment wins over an older stored policy.
        return GuardrailPolicy(
            min_runway_months=max(stored.min_runway_months, limits.min_runway_months),
            spend_cap_daily_usd=min(stored.spend_cap_daily_usd, limits.spend_cap_daily_usd),
            max_cac_usd=min(stored.max_cac_usd, limits.max_cac_usd),
            marketer_approval_limit_usd=min(
                stored.marketer_approval_limit_usd, limits.marketer_approval_limit_usd
            ),
            allow_auto_pause=stored.allow_auto_pause and limits.allow_auto_pause,
        )

    def set_policy(self, policy: GuardrailPolicy, actor: str) -> GuardrailPolicy:
        limits = self.hard_limits()
        problems = []
        if policy.min_runway_months < limits.min_runway_months:
            problems.append(f"minimum runway cannot go below {limits.min_runway_months:g} months")
        if policy.spend_cap_daily_usd > limits.spend_cap_daily_usd:
            problems.append(f"daily spend cap cannot exceed ${limits.spend_cap_daily_usd:,.0f}")
        if policy.max_cac_usd > limits.max_cac_usd:
            problems.append(f"max CAC cannot exceed ${limits.max_cac_usd:,.0f}")
        if policy.marketer_approval_limit_usd > limits.marketer_approval_limit_usd:
            problems.append(
                f"marketer approval limit cannot exceed ${limits.marketer_approval_limit_usd:,.0f}"
            )
        if policy.allow_auto_pause and not limits.allow_auto_pause:
            problems.append("auto-pause is disabled by ALLOW_AUTO_PAUSE")
        if problems:
            self._audit.record(
                actor=actor, action="guardrails.update", outcome="denied", detail={"problems": problems}
            )
            raise PolicyViolation("; ".join(problems))
        previous = self.policy()
        with self._db.tx() as conn:
            conn.execute(
                "INSERT INTO guardrail_policy (ts, set_by, body) VALUES (?, ?, ?)",
                (utc_now_iso(), actor, policy.model_dump_json()),
            )
            self._audit.record(
                actor=actor,
                action="guardrails.update",
                detail={"from": previous.model_dump(), "to": policy.model_dump()},
            )
        return policy

    # -- runway --------------------------------------------------------------

    def commitments_usd(self, exclude_experiment_id: str | None = None) -> float:
        """Experiment money that is promised or spent but not yet in the closed books."""
        books_closed = _month_end(self._metrics.finance_snapshot().as_of_month)
        total = 0.0
        for experiment in self._store.list_experiments(["approved", "running", "stopped", "concluded"]):
            if experiment.id == exclude_experiment_id:
                continue
            if experiment.status in ("approved", "running"):
                total += experiment.budget_usd
            elif experiment.ended_on is not None and experiment.ended_on > books_closed:
                total += experiment.spent_usd
        return total

    def check_one_time(self, cost_usd: float, exclude_experiment_id: str | None = None) -> RunwayCheck:
        """Runway impact of a one-off spend such as an experiment budget."""
        finance = self._metrics.finance_snapshot()
        cash = finance.cash_usd - self.commitments_usd(exclude_experiment_id)
        before = burn.runway_months(cash, finance.net_burn_usd)
        after = burn.runway_after_one_time(cash, finance.net_burn_usd, cost_usd)
        return self._check(before, after)

    def check_recurring(self, monthly_delta_usd: float) -> RunwayCheck:
        """Runway impact of a change in monthly spend such as a budget reallocation."""
        finance = self._metrics.finance_snapshot()
        cash = finance.cash_usd - self.commitments_usd()
        before = burn.runway_months(cash, finance.net_burn_usd)
        after = burn.runway_after_recurring(cash, finance.net_burn_usd, monthly_delta_usd)
        return self._check(before, after)

    def recurring_headroom_usd(self) -> float:
        """Largest monthly spend increase the runway policy allows (negative: must cut)."""
        finance = self._metrics.finance_snapshot()
        cash = finance.cash_usd - self.commitments_usd()
        return burn.max_recurring_increase(cash, finance.net_burn_usd, self.policy().min_runway_months)

    def _check(self, before: float, after: float) -> RunwayCheck:
        minimum = self.policy().min_runway_months
        ok = after >= minimum
        return RunwayCheck(
            before_months=_finite(before),
            after_months=_finite(after),
            min_months=minimum,
            ok=ok,
            reason=None if ok else f"would reduce runway to {after:.1f} mo (< {minimum:g})",
        )


def _finite(months: float) -> float | None:
    return None if math.isinf(months) else months


def _month_end(month: str) -> date:
    year, mon = (int(part) for part in month.split("-"))
    next_month = date(year + (mon == 12), mon % 12 + 1, 1)
    return date.fromordinal(next_month.toordinal() - 1)
