"""Runway guardrails, policy hard limits and the tamper-evident audit trail."""

from __future__ import annotations

from datetime import date

import pytest

from growthos.clock import Clock
from growthos.config import Settings
from growthos.db import Database
from growthos.gateway.audit import AuditLog
from growthos.gateway.guardrails import GuardrailService, PolicyViolation
from growthos.memory import ExperimentStore
from growthos.metrics import MetricsService
from growthos.models import Experiment


@pytest.fixture
def audit(db: Database) -> AuditLog:
    return AuditLog(db)


@pytest.fixture
def guardrails(memory: ExperimentStore, db: Database, clock: Clock, audit: AuditLog) -> GuardrailService:
    return GuardrailService(db, Settings(), MetricsService(db, clock), memory, audit)


class TestAudit:
    def test_chain_verifies_and_lists_newest_first(self, audit: AuditLog) -> None:
        audit.record(actor="agent", action="proposal.created", experiment_id="EXP-113")
        audit.record(actor="maya", action="approval.approved", approval_id="APR-0001", amount_usd=2000)
        entries = audit.entries()
        assert [e.action for e in entries] == ["approval.approved", "proposal.created"]
        assert entries[0].prev_hash == entries[1].hash
        assert audit.verify() is None

    def test_editing_a_past_entry_is_detected(self, audit: AuditLog, db: Database) -> None:
        audit.record(actor="maya", action="approval.approved", amount_usd=2000)
        audit.record(actor="agent", action="campaign.created")
        with db.tx() as conn:
            conn.execute("UPDATE audit_log SET amount_usd = 200 WHERE seq = 1")
        assert audit.verify() == 1

    def test_filter_by_experiment(self, audit: AuditLog) -> None:
        audit.record(actor="agent", action="proposal.created", experiment_id="EXP-113")
        audit.record(actor="agent", action="proposal.created", experiment_id="EXP-114")
        assert [e.experiment_id for e in audit.entries(experiment_id="EXP-114")] == ["EXP-114"]


class TestPolicy:
    def test_defaults_come_from_the_environment(self, guardrails: GuardrailService) -> None:
        assert guardrails.policy() == guardrails.hard_limits()
        assert guardrails.policy().min_runway_months == 18

    def test_cfo_can_tighten_and_the_change_is_audited(
        self, guardrails: GuardrailService, audit: AuditLog
    ) -> None:
        stricter = guardrails.policy().model_copy(update={"min_runway_months": 19.5, "spend_cap_daily_usd": 2500})
        guardrails.set_policy(stricter, actor="sam")
        assert guardrails.policy() == stricter
        entry = audit.entries()[0]
        assert entry.action == "guardrails.update" and entry.actor == "sam" and entry.outcome == "ok"

    @pytest.mark.parametrize(
        "update",
        [
            {"min_runway_months": 12},
            {"spend_cap_daily_usd": 1_000_000},
            {"max_cac_usd": 5000},
            {"marketer_approval_limit_usd": 50_000},
            {"allow_auto_pause": True},
        ],
    )
    def test_hard_limits_cannot_be_loosened(
        self, guardrails: GuardrailService, audit: AuditLog, update: dict
    ) -> None:
        with pytest.raises(PolicyViolation):
            guardrails.set_policy(guardrails.policy().model_copy(update=update), actor="sam")
        assert guardrails.policy() == guardrails.hard_limits()
        assert audit.entries()[0].outcome == "denied"

    def test_tightened_environment_overrides_an_older_stored_policy(
        self, memory: ExperimentStore, db: Database, clock: Clock, audit: AuditLog
    ) -> None:
        loose = GuardrailService(db, Settings(), MetricsService(db, clock), memory, audit)
        loose.set_policy(loose.policy().model_copy(update={"min_runway_months": 18.5}), actor="sam")
        strict_env = Settings(min_runway_months=20)
        strict = GuardrailService(db, strict_env, MetricsService(db, clock), memory, audit)
        assert strict.policy().min_runway_months == 20


class TestRunway:
    def test_small_experiment_passes_and_reports_its_impact(self, guardrails: GuardrailService) -> None:
        check = guardrails.check_one_time(2000)
        assert check.ok and check.reason is None
        assert check.impact_months is not None and -0.1 < check.impact_months < 0

    def test_spend_that_breaks_policy_is_blocked_with_a_readable_reason(
        self, guardrails: GuardrailService
    ) -> None:
        check = guardrails.check_one_time(600_000)
        assert not check.ok
        assert check.after_months is not None and check.after_months < 18
        assert check.reason == f"would reduce runway to {check.after_months:.1f} mo (< 18)"

    def test_recurring_headroom_sits_exactly_on_the_policy_boundary(self, guardrails: GuardrailService) -> None:
        headroom = guardrails.recurring_headroom_usd()
        assert headroom > 0
        assert guardrails.check_recurring(headroom - 1).ok
        assert not guardrails.check_recurring(headroom + 1).ok

    def test_approved_budgets_count_as_committed_cash(
        self, guardrails: GuardrailService, memory: ExperimentStore
    ) -> None:
        before = guardrails.check_one_time(0)
        memory.save(
            Experiment(
                id=memory.new_id(),
                hypothesis="Pain-point headline for SMB finance on LinkedIn",
                segment_id="seg-smb-finance",
                channel="linkedin_ads",
                variants=["Feature overview", "Close the books in 2 days"],
                budget_usd=50_000,
                primary_metric="cac",
                min_sample=400,
                status="approved",
                created_on=date(2026, 10, 1),
            )
        )
        assert guardrails.commitments_usd() == 50_000
        after = guardrails.check_one_time(0)
        assert after.before_months is not None and before.before_months is not None
        assert after.before_months < before.before_months
