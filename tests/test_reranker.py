"""Tests for app.retrieval.reranker.

As in test_embeddings.py, a stub ``sentence_transformers`` package is installed
into ``sys.modules`` so nothing is downloaded. One test at the bottom runs the
real model and is marked ``integration``.
"""

import sys
import types

import numpy as np
import pytest

from app.retrieval.reranker import CrossEncoderReranker, Reranker, get_reranker


class FakeCrossEncoder:
    """Same surface as the real CrossEncoder, no download.

    Scores a pair by how many query words the passage contains, so relevance
    is predictable.
    """

    cached = True
    load_attempts: list[bool] = []
    max_seq_length = 512

    def __init__(self, name: str, local_files_only: bool = False) -> None:
        type(self).load_attempts.append(local_files_only)
        if local_files_only and not type(self).cached:
            raise OSError(f"{name} is not in the local cache")
        self.name = name
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, show_progress_bar=None):
        self.predict_calls.append(list(pairs))
        return np.array(
            [
                float(len(set(query.lower().split()) & set(passage.lower().split())))
                for query, passage in pairs
            ],
            dtype=np.float32,
        )


@pytest.fixture
def fake_ce(monkeypatch: pytest.MonkeyPatch) -> type[FakeCrossEncoder]:
    """Install a stub ``sentence_transformers`` package."""
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(FakeCrossEncoder, "load_attempts", [])
    monkeypatch.setattr(FakeCrossEncoder, "cached", True)
    return FakeCrossEncoder


class TestLoading:
    def test_cached_model_loads_local_only_in_one_attempt(
        self, fake_ce: type[FakeCrossEncoder]
    ) -> None:
        CrossEncoderReranker("fake/reranker")

        assert fake_ce.load_attempts == [True]

    def test_missing_model_falls_back_to_a_download(
        self, fake_ce: type[FakeCrossEncoder], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fake_ce, "cached", False)

        CrossEncoderReranker("fake/reranker")

        assert fake_ce.load_attempts == [True, False]

    def test_defaults_to_the_configured_model(
        self, fake_ce: type[FakeCrossEncoder]
    ) -> None:
        from app.config import get_settings

        assert CrossEncoderReranker().name == get_settings().reranker_model_name

    def test_reports_the_models_input_limit(
        self, fake_ce: type[FakeCrossEncoder]
    ) -> None:
        assert CrossEncoderReranker("fake/reranker").max_seq_length == 512


class TestScore:
    def test_one_score_per_passage_in_input_order(
        self, fake_ce: type[FakeCrossEncoder]
    ) -> None:
        reranker = CrossEncoderReranker("fake/reranker")

        scores = reranker.score(
            "warranty period", ["nothing here", "the warranty period is long", "warranty"]
        )

        # Input order, not sorted: 0 shared words, 2, then 1.
        assert scores == [0.0, 2.0, 1.0]

    def test_scores_are_plain_python_floats(
        self, fake_ce: type[FakeCrossEncoder]
    ) -> None:
        scores = CrossEncoderReranker("fake/reranker").score("a", ["a", "b"])

        assert all(type(value) is float for value in scores)

    def test_pairs_the_query_with_every_passage(
        self, fake_ce: type[FakeCrossEncoder]
    ) -> None:
        reranker = CrossEncoderReranker("fake/reranker")

        reranker.score("q", ["one", "two"])

        assert reranker._model.predict_calls == [[("q", "one"), ("q", "two")]]

    def test_empty_input_never_calls_the_model(
        self, fake_ce: type[FakeCrossEncoder]
    ) -> None:
        reranker = CrossEncoderReranker("fake/reranker")

        assert reranker.score("q", []) == []
        assert reranker._model.predict_calls == []


class TestProtocol:
    def test_cross_encoder_reranker_satisfies_reranker(
        self, fake_ce: type[FakeCrossEncoder]
    ) -> None:
        assert isinstance(CrossEncoderReranker("fake/reranker"), Reranker)

    def test_a_plain_object_with_score_also_satisfies_it(self) -> None:
        class Scripted:
            def score(self, query: str, passages: list[str]) -> list[float]:
                return [0.0] * len(passages)

        assert isinstance(Scripted(), Reranker)


class TestGetReranker:
    def test_is_cached(self, fake_ce: type[FakeCrossEncoder]) -> None:
        get_reranker.cache_clear()
        try:
            assert get_reranker() is get_reranker()
            assert fake_ce.load_attempts == [True]
        finally:
            get_reranker.cache_clear()


@pytest.mark.integration
class TestRealModel:
    """Loads the real cross-encoder; downloads ~90 MB on first run."""

    def test_relevant_passage_scores_above_unrelated(self) -> None:
        reranker = CrossEncoderReranker()

        relevant, unrelated = reranker.score(
            "how long is the warranty",
            [
                "The limited warranty covers defects for a period of 24 months "
                "from the date of purchase.",
                "The tenant shall not sublet the premises without written consent "
                "of the landlord.",
            ],
        )

        assert relevant > unrelated
