"""FR-02: experiment memory and hybrid retrieval with citations."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from growthos.embeddings import HashingEmbedder, tokenize
from growthos.memory import ExperimentNotFound, ExperimentStore, HybridRetriever
from growthos.models import Experiment


def _experiment(store: ExperimentStore, **overrides) -> Experiment:
    fields = {
        "id": store.new_id(),
        "hypothesis": "Annual-billing toggle on the pricing page lifts paid conversion",
        "segment_id": "seg-smb-finance",
        "channel": "google_ads",
        "variants": ["Monthly default", "Annual default"],
        "budget_usd": 2500,
        "primary_metric": "cvr",
        "min_sample": 400,
        "status": "proposed",
        "created_on": date(2026, 10, 1),
    }
    return Experiment(**(fields | overrides))


class TestEmbedder:
    def test_rows_are_unit_length_and_deterministic(self) -> None:
        embedder = HashingEmbedder(dim=128)
        first = embedder.embed(["close the books faster", "free month for founders"])
        assert first.shape == (2, 128)
        assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
        assert np.array_equal(first, embedder.embed(["close the books faster", "free month for founders"]))

    def test_related_text_is_closer_than_unrelated(self) -> None:
        embedder = HashingEmbedder()
        anchor, related, unrelated = embedder.embed(
            [
                "pain-point headline about month-end close for finance managers",
                "month-end closing pain for a finance manager",
                "free first month promotion for startup founders",
            ]
        )
        assert anchor @ related > anchor @ unrelated

    def test_tokenizer_drops_stopwords_and_folds_simple_inflections(self) -> None:
        assert tokenize("Closing the books for controllers") == ["clos", "book", "controller"]


class TestStore:
    def test_round_trip_and_status_filter(self, store: ExperimentStore) -> None:
        proposed = store.save(_experiment(store))
        running = store.save(_experiment(store, status="running"))
        assert store.get(proposed.id) == proposed
        assert [e.id for e in store.list_experiments(["running"])] == [running.id]
        assert len(store.list_experiments()) == 2

    def test_missing_experiment_is_an_explicit_error(self, store: ExperimentStore) -> None:
        with pytest.raises(ExperimentNotFound):
            store.get("EXP-999")

    def test_only_finished_experiments_become_evidence(self, store: ExperimentStore) -> None:
        store.save(_experiment(store))
        docs, _ = store.docs()
        assert docs == []
        finished = store.save(_experiment(store, status="concluded", outcome="win", learnings="Annual default won."))
        docs, matrix = store.docs()
        assert [doc.experiment_id for doc in docs] == [finished.id]
        assert matrix.shape == (1, HashingEmbedder().dim)

    def test_resaving_updates_the_indexed_learnings(self, store: ExperimentStore) -> None:
        experiment = _experiment(store, status="concluded", learnings="First take.")
        store.save(experiment)
        store.save(experiment.model_copy(update={"learnings": "Revised take."}))
        docs, _ = store.docs()
        assert len(docs) == 1
        assert "Revised take." in docs[0].text

    def test_new_experiments_continue_after_seeded_history(self, memory: ExperimentStore) -> None:
        assert memory.new_id() == "EXP-113"


class TestRetrieval:
    @pytest.fixture
    def retriever(self, memory: ExperimentStore, embedder: HashingEmbedder) -> HybridRetriever:
        return HybridRetriever(memory, embedder)

    def test_wireframe_proposal_cites_the_same_evidence(self, retriever: HybridRetriever) -> None:
        hits = retriever.search("Pain-point headline for SMB finance segment on LinkedIn", k=5, kinds={"experiment"})
        cited = {hit.citation.experiment_id: hit.citation.note for hit in hits}
        assert "EXP-112" in cited and "EXP-087" in cited
        assert cited["EXP-112"] == "+19% CVR, pain point (win)"
        assert all(0 <= hit.similarity <= 1 for hit in hits)
        assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)

    def test_kind_filter_restricts_to_research_notes(self, retriever: HybridRetriever) -> None:
        hits = retriever.search("what do founders care about runway", k=3, kinds={"interview"})
        assert hits and all(hit.doc.kind == "interview" for hit in hits)
        assert "founder" in hits[0].doc.title.lower()
        assert hits[0].citation.experiment_id is None

    def test_new_memory_is_searchable_without_rebuilding(
        self, retriever: HybridRetriever, memory: ExperimentStore
    ) -> None:
        assert not any("tiktok" in hit.doc.text.lower() for hit in retriever.search("tiktok creator ads", k=3))
        memory.add_note("competitor", "Competitor page: TallyUp TikTok campaign", "TallyUp runs TikTok creator ads.")
        assert "TikTok" in retriever.search("tiktok creator ads", k=3)[0].doc.title

    def test_blank_query_returns_nothing(self, retriever: HybridRetriever) -> None:
        assert retriever.search("   ") == []

    def test_recall_at_five_on_labelled_queries(self, retriever: HybridRetriever) -> None:
        """Retrieval sets the ceiling: measure recall@k before tuning prompts (BRAIN.md)."""
        labelled = {
            "discount offer for mid-market controllers": "EXP-080",
            "free month promotion for startup founders on search": "EXP-104",
            "customer proof for controllers on LinkedIn": "EXP-078",
            "generic all-in-one platform messaging for agencies": "EXP-083",
            "case study email about audit prep hours saved": "EXP-108",
            "manual re-keying pain for bookkeeping agencies": "EXP-101",
            "spreadsheet pain creative for founders": "EXP-096",
            "logo carousel social proof for SMB on Meta": "EXP-090",
            "subject line about month-end pain in SMB nurture email": "EXP-085",
            "review stars extension on search for SMB": "EXP-110",
        }
        found = sum(
            expected in {hit.citation.experiment_id for hit in retriever.search(query, k=5)}
            for query, expected in labelled.items()
        )
        assert found / len(labelled) >= 0.9
