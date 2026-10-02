"""Seed for experiment memory: past experiments with results, plus research notes.

Outcomes follow the hidden market truth in `world.py` (pain-point copy works for SMB finance,
proof works for controllers, discounts only work for founders), so the memory contains real
signal for the proposal agent to find.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from growthos.analysis.bayes import AnalysisConfig, VariantData, analyse, describe
from growthos.db import Database
from growthos.memory.experiment_store import DocKind, ExperimentStore
from growthos.models import Experiment, ExperimentResult, Outcome
from growthos.sim import world
from growthos.sim.history import SIM_START

_DURATION_DAYS = 14
_MAX_CAPACITY_SHARE = 0.5
_LOAD_FACTOR = 1.34
_ARPA_USD = 260.0
# z-score the seeded winners and losers were run to; inconclusive tests stopped short.
_DECISIVE_Z = 2.6
_LAST_ENDED = SIM_START - timedelta(days=11)
_SPACING_DAYS = 19


@dataclass(frozen=True)
class PastExperiment:
    number: int
    channel: str
    segment_id: str
    tactic: str
    hypothesis: str
    control: str
    challenger: str
    lift: float
    outcome: Outcome
    learnings: str
    clicks_per_arm: int | None = None


SMB = "seg-smb-finance"
MID = "seg-midmarket-controllers"
FOUNDERS = "seg-startup-founders"
AGENCIES = "seg-bookkeeping-agencies"

PAST_EXPERIMENTS: tuple[PastExperiment, ...] = (
    PastExperiment(
        71, "meta_ads", SMB, "pain_point",
        "Pain-point creative about late-night month-end closes beats the product-tour creative for SMB finance on Meta",
        "Product tour video", "Stop closing the books at midnight",
        0.26, "win",
        "Pain-point framing lifted signup rate 26% for SMB finance teams. The late-night month-end angle "
        "outperformed every feature-led creative. Comments showed finance managers tagging colleagues.",
    ),
    PastExperiment(
        74, "meta_ads", FOUNDERS, "urgency_discount",
        "A first-month-free offer lifts trial signups from startup founders on Meta",
        "Standard free trial", "First month free for startups",
        0.17, "win",
        "Founders respond to a free first month; conversions skewed mobile. Trial-to-paid held, so the "
        "discount did not just pull in low-intent signups. Discounts work for founders only.",
    ),
    PastExperiment(
        78, "linkedin_ads", MID, "social_proof",
        "Social proof ('trusted by 400+ finance teams') beats feature copy for mid-market controllers on LinkedIn",
        "Feature overview", "Trusted by 400+ finance teams",
        0.16, "win",
        "Controllers want proof that peers at their size use it. Customer-count proof lifted CVR 16%; "
        "naming audit-ready customers in the body copy helped further.",
    ),
    PastExperiment(
        80, "linkedin_ads", MID, "urgency_discount",
        "A 20% annual-plan discount increases demo requests from mid-market controllers",
        "Standard pricing", "20% off annual plans this quarter",
        -0.11, "loss",
        "Discounting read as low-end to controllers and demo-booked rate fell 11%. Do not discount to "
        "mid-market; lead with audit readiness and proof instead.",
        clicks_per_arm=1500,
    ),
    PastExperiment(
        83, "google_ads", AGENCIES, "generic_feature",
        "An 'all-in-one bookkeeping platform' headline beats the partner-program headline for agencies on search",
        "Partner program for bookkeepers", "The all-in-one bookkeeping platform",
        -0.06, "loss",
        "Generic platform language underperformed the specific partner-program message by 6%. Agencies "
        "search for multi-client workflows, not platforms.",
        clicks_per_arm=3000,
    ),
    PastExperiment(
        85, "email", SMB, "pain_point",
        "A month-end pain-point subject line lifts click-to-trial in the SMB nurture sequence",
        "Product update newsletter", "Still reconciling spreadsheets at month-end?",
        0.21, "win",
        "Pain-point subject line and opener lifted click-to-trial 21% for SMB finance leads. Works best "
        "sent in the last week of the month when close pain is top of mind.",
    ),
    PastExperiment(
        87, "linkedin_ads", SMB, "quantified_outcome",
        "A quantified-outcome ad ('save 12 hours every month-end') beats the feature ad for SMB finance on LinkedIn",
        "Feature overview", "Save 12 hours every month-end",
        0.04, "inconclusive",
        "LinkedIn SMB CAC came in at $1,180 fully loaded. The lift was within noise at 250 clicks per arm: "
        "the test was under-powered and would need about 1,500 clicks per arm to detect a 10% lift.",
        clicks_per_arm=250,
    ),
    PastExperiment(
        90, "meta_ads", SMB, "social_proof",
        "A customer-logo carousel beats the single-image ad for SMB finance on Meta",
        "Single image ad", "Logo carousel: finance teams like yours",
        0.02, "inconclusive",
        "No detectable difference from logo carousel for SMB on Meta. Logos SMB buyers do not recognise "
        "carry little weight; proof may need to be peer stories rather than logos.",
        clicks_per_arm=900,
    ),
    PastExperiment(
        93, "google_ads", MID, "quantified_outcome",
        "A quantified headline ('close 3x faster') beats the brand headline for controller search queries",
        "Brand headline", "Close 3x faster with audit-ready books",
        0.14, "win",
        "Quantified outcomes lifted CVR 14% on controller queries. Pairing the number with 'audit-ready' "
        "mattered: the number alone tested flat in the ad-copy pre-test.",
    ),
    PastExperiment(
        96, "meta_ads", FOUNDERS, "pain_point",
        "Spreadsheet-chaos pain-point creative lifts founder signups on Meta",
        "Runway dashboard demo", "Tired of spreadsheet chaos?",
        -0.07, "loss",
        "Founders do not feel month-end close pain yet; the pain-point creative lost 7% to the runway "
        "dashboard demo. For founders, lead with runway visibility, not close pain.",
        clicks_per_arm=3000,
    ),
    PastExperiment(
        99, "email", AGENCIES, "social_proof",
        "A partner testimonial email lifts trial starts among bookkeeping agencies",
        "Feature announcement", "How a 12-person agency added 30 clients",
        0.12, "win",
        "A peer agency story lifted trial starts 12%. Agencies respond to proof from firms their own size; "
        "the partner quote outperformed the feature announcement.",
    ),
    PastExperiment(
        101, "linkedin_ads", AGENCIES, "pain_point",
        "A manual-reconciliation pain-point ad beats the partner-program ad for agencies on LinkedIn",
        "Partner program overview", "Stop re-keying client ledgers by hand",
        0.19, "win",
        "Manual re-keying pain resonated with agency partners: +19% CVR. Pain-point copy works wherever "
        "the reader personally does the manual work.",
    ),
    PastExperiment(
        104, "google_ads", FOUNDERS, "urgency_discount",
        "A free-month promotion extension lifts founder signups from search",
        "No promotion extension", "First month free for startups",
        0.13, "win",
        "Promotion extension lifted founder CVR 13% on search. Confirms EXP-074: founders are "
        "price-sensitive and respond to a free first month.",
    ),
    PastExperiment(
        106, "meta_ads", MID, "generic_feature",
        "A 'next-generation finance platform' video lifts controller signups on Meta",
        "Audit-readiness explainer", "The next-generation finance platform",
        -0.09, "loss",
        "Generic platform positioning lost 9% to the audit-readiness explainer. Controllers ignore "
        "category claims; Meta is also a weak channel for mid-market.",
        clicks_per_arm=2500,
    ),
    PastExperiment(
        108, "email", MID, "quantified_outcome",
        "A case-study email quantifying audit-prep hours saved lifts demo requests from controllers",
        "Product newsletter", "How Northgate cut audit prep by 40 hours",
        0.15, "win",
        "Quantified case study lifted controller demo requests 15%. Specific hours saved plus a named "
        "peer company is the strongest controller message so far.",
    ),
    PastExperiment(
        110, "google_ads", SMB, "social_proof",
        "A review-stars extension ('rated 4.8 by finance teams') lifts SMB search conversion",
        "No review extension", "Rated 4.8 by finance teams",
        0.05, "inconclusive",
        "Review stars showed a small positive lift that did not reach significance at 800 clicks per arm. "
        "Proof seems secondary to pain for SMB; worth a larger retest only if cheap.",
        clicks_per_arm=800,
    ),
    PastExperiment(
        89, "linkedin_ads", SMB, "generic_feature",
        "A 'powerful all-in-one finance solution' ad beats the month-end checklist offer for SMB finance on LinkedIn",
        "Month-end checklist offer", "The powerful all-in-one finance solution",
        -0.05, "loss",
        "Generic solution language lost 5% to the concrete checklist offer. SMB finance readers skip "
        "category claims and click on things that fix a specific month-end problem.",
        clicks_per_arm=2500,
    ),
    PastExperiment(
        92, "google_ads", SMB, "urgency_discount",
        "A limited-time 15% discount headline lifts SMB search conversion",
        "Standard trial headline", "Limited time: 15% off your first year",
        -0.06, "loss",
        "The discount headline lost 6% for SMB finance searchers. They are solving a time problem, not a "
        "price problem; discounting did not help and cheapened the message.",
        clicks_per_arm=3000,
    ),
    PastExperiment(
        98, "email", AGENCIES, "urgency_discount",
        "A 'today only' discount email lifts trial starts among bookkeeping agencies",
        "Standard partner invite", "Today only: 20% discount for partners",
        -0.06, "loss",
        "The today-only discount underperformed the standard invite by 6% and drew unsubscribes. "
        "Agencies buy on workflow fit, not urgency.",
        clicks_per_arm=3000,
    ),
    PastExperiment(
        102, "meta_ads", FOUNDERS, "social_proof",
        "Social proof ('join 2,000 startups') lifts founder signups on Meta",
        "Runway dashboard demo", "Join 2,000 startups tracking runway",
        0.06, "inconclusive",
        "Small positive lift, not significant at 700 clicks per arm. Founders may respond to proof but the "
        "effect looks smaller than the free-month offer.",
        clicks_per_arm=700,
    ),
    PastExperiment(
        107, "google_ads", FOUNDERS, "quantified_outcome",
        "A quantified headline ('know your runway in 5 minutes') lifts founder search conversion",
        "Brand headline", "Know your runway in 5 minutes",
        0.07, "inconclusive",
        "Promising +7% but under-powered at 600 clicks per arm. Runway visibility is the right theme for "
        "founders and deserves a properly sized retest.",
        clicks_per_arm=600,
    ),
    PastExperiment(
        112, "google_ads", SMB, "pain_point",
        "A pain-point headline about closing the books faster beats the feature headline for SMB search",
        "Accounting automation software", "Close the books in 2 days, not 10",
        0.19, "win",
        "+19% CVR from pain-point copy on SMB finance search queries. Third pain-point win for SMB "
        "(after Meta and email); not yet tested for SMB on LinkedIn.",
    ),
)

NOTES: tuple[tuple[DocKind, str, str, str], ...] = (
    (
        "win_loss", "Won: Harbor Retail (SMB, 60 employees)",
        "Chose us because month-end close dropped from 9 days to 2 in the trial. Champion was the finance "
        "manager, who found us searching 'close the books faster'. Price was not discussed until the end.",
        "crm",
    ),
    (
        "win_loss", "Lost: Northwind Logistics (mid-market, 900 employees)",
        "Picked a competitor for NetSuite integration depth and a SOC 2 Type II report. Price was not a "
        "factor. The controller asked twice which companies of their size use us.",
        "crm",
    ),
    (
        "interview", "Interview: SMB finance manager, ecommerce",
        "\"I spend the last three nights of every month in spreadsheets reconciling payouts. I'd pay for "
        "anything that gives me my evenings back. I don't care about AI, I care about closing on time.\"",
        "research",
    ),
    (
        "interview", "Interview: controller, healthcare software",
        "\"I need to know my auditors will accept it. Show me who else at my size uses it and what their "
        "audit looked like. A discount makes me nervous, honestly.\"",
        "research",
    ),
    (
        "interview", "Interview: seed-stage founder",
        "\"I don't care about month-end close. I care about knowing my runway every Monday without asking "
        "my accountant. If the first month is free I'll try it tonight.\"",
        "research",
    ),
    (
        "interview", "Interview: bookkeeping agency partner",
        "\"Every client has a different chart of accounts and we re-key everything by hand. That manual "
        "work kills our margins. I trust what other agencies my size are using.\"",
        "research",
    ),
    (
        "competitor", "Competitor page: LedgerLoop homepage",
        "Headline: 'The all-in-one finance platform for modern teams'. Feature-led, no quantified outcomes, "
        "no customer proof above the fold. Pricing from $199 per month. Targets SMB.",
        "https://ledgerloop.example/",
    ),
    (
        "competitor", "Competitor page: ClosePilot mid-market landing page",
        "Leads with 'Close in 3 days' and a wall of review badges. Targets mid-market controllers with "
        "annual-only pricing and a SOC 2 badge in the hero.",
        "https://closepilot.example/mid-market",
    ),
)


def seed_memory(db: Database, store: ExperimentStore) -> None:
    """Load past experiments and research notes into memory."""
    ordered = sorted(PAST_EXPERIMENTS, key=lambda p: p.number)
    for position, past in enumerate(ordered):
        ended = _LAST_ENDED - timedelta(days=_SPACING_DAYS * (len(ordered) - 1 - position))
        store.save(build_experiment(past, ended))
    # New experiments continue the numbering after the latest historical one.
    db.bump_counter("EXP", max(p.number for p in ordered))
    for kind, title, text, source in NOTES:
        store.add_note(kind, title, text, source=source)


def build_experiment(past: PastExperiment, ended: date) -> Experiment:
    truth = world.CHANNELS[past.channel]
    clicks = past.clicks_per_arm or _decisive_sample(truth.base_cvr, past.lift)
    # Run longer rather than push a channel past half of its click capacity.
    capacity = _MAX_CAPACITY_SHARE * truth.saturation_usd_per_day / truth.base_cpc
    duration = max(_DURATION_DAYS, math.ceil(2 * clicks / capacity))
    daily_budget = _daily_budget_for(truth, 2 * clicks / duration)
    budget = round(daily_budget * duration, 2)
    data = [
        VariantData(past.control, clicks, round(clicks * truth.base_cvr), budget / 2),
        VariantData(past.challenger, clicks, round(clicks * truth.base_cvr * (1 + past.lift)), budget / 2),
    ]
    posterior = analyse(
        data,
        control=past.control,
        config=AnalysisConfig(
            primary_metric="cvr",
            seed=past.number,
            downstream_rate=truth.signup_to_paid,
            load_factor=_LOAD_FACTOR,
            arpa_usd=_ARPA_USD,
            gross_margin=world.GROSS_MARGIN,
        ),
    )
    winner = {"win": past.challenger, "loss": past.control, "inconclusive": None}[past.outcome]
    result = ExperimentResult(
        prob_best=posterior.prob_best,
        cac_posterior_mean=posterior.cac_mean,
        expected_loss=posterior.expected_loss,
        decision="ship" if past.outcome == "win" else "kill",
        winner=winner,
        reason=f"historical record: {past.outcome}",
        summary=describe(posterior, past.challenger),
        cvr_posterior_mean=posterior.cvr_mean,
        cvr_interval=posterior.cvr_interval,
        cac_interval=posterior.cac_interval,
        payback_months=posterior.payback_months,
        prob_beats_control=posterior.prob_beats_control,
        lift_vs_control=posterior.lift_vs_control,
        samples={v.name: v.visitors for v in data},
        conversions={v.name: v.conversions for v in data},
        spend_usd={v.name: v.spend_usd for v in data},
        analysed_on=ended,
        data_through=ended,
    )
    return Experiment(
        id=f"EXP-{past.number:03d}",
        hypothesis=past.hypothesis,
        segment_id=past.segment_id,
        channel=past.channel,
        variants=[past.control, past.challenger],
        budget_usd=budget,
        primary_metric="cvr",
        min_sample=clicks,
        max_sample=clicks,
        status="concluded",
        result=result,
        learnings=past.learnings,
        tactic=past.tactic,
        duration_days=duration,
        created_on=ended - timedelta(days=duration + 5),
        started_on=ended - timedelta(days=duration),
        ended_on=ended,
        spent_usd=budget,
        outcome=past.outcome,
        observed_lift=past.lift,
    )


def _decisive_sample(base_cvr: float, lift: float) -> int:
    """Clicks per arm at which a lift of this size reaches the decisive z-score."""
    delta = base_cvr * abs(lift)
    variance = 2 * base_cvr * (1 - base_cvr)
    return math.ceil(variance * (_DECISIVE_Z / delta) ** 2 / 100) * 100


def _daily_budget_for(truth: world.ChannelTruth, clicks_per_day: float) -> float:
    """Invert clicks = spend / cpc(spend) for the saturating cost curve."""
    ceiling = truth.saturation_usd_per_day / truth.base_cpc
    if clicks_per_day >= ceiling:
        raise ValueError(f"{truth.name} cannot deliver {clicks_per_day:.0f} clicks per day")
    return clicks_per_day * truth.base_cpc / (1 - clicks_per_day / ceiling)
