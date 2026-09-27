"""Tests for app.indexing.embeddings.

The real model is ~440 MB and takes seconds to load, so almost everything here
runs against a stub installed into ``sys.modules``. That works because
``EmbeddingModel`` imports ``sentence_transformers`` inside ``__init__`` rather
than at module scope. One test at the bottom exercises the real model and is
marked ``integration`` so the default run stays offline.
"""

import sys
import types

import numpy as np
import pytest

from app.indexing.embeddings import (
    ChunkWindowTooLargeError,
    EmbeddingModel,
    TextEmbedder,
    get_embedding_model,
)


class FakeTokenizer:
    """One token per whitespace-separated word — enough to assert wiring."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


class FakeSentenceTransformer:
    """Same surface as the real model, no download.

    Class attributes let a test change the model's limits before construction;
    ``monkeypatch.setattr`` restores them afterwards.
    """

    limit = 512
    dims = 4
    # Whether the model is "in the local cache". When False, a load with
    # local_files_only=True fails the way the real library does: with OSError.
    cached = True
    # Every construction attempt, as the local_files_only value it was given.
    # A list on the class, reset by the fixture, so tests can see the retry.
    load_attempts: list[bool] = []

    def __init__(self, name: str, local_files_only: bool = False) -> None:
        type(self).load_attempts.append(local_files_only)
        if local_files_only and not type(self).cached:
            raise OSError(f"{name} is not in the local cache")
        self.name = name
        self.max_seq_length = type(self).limit
        self.tokenizer = FakeTokenizer()
        self.document_calls: list[dict] = []
        self.query_calls: list[dict] = []

    def get_embedding_dimension(self) -> int:
        return type(self).dims

    def encode_document(self, texts, batch_size=32, normalize_embeddings=False):
        self.document_calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize": normalize_embeddings,
            }
        )
        return np.array([[float(len(t))] * type(self).dims for t in texts])

    def encode_query(self, text, normalize_embeddings=False):
        self.query_calls.append({"text": text, "normalize": normalize_embeddings})
        return np.array([float(len(text))] * type(self).dims)


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    """Install a stub ``sentence_transformers`` package."""
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    # A fresh list per test; monkeypatch restores the original afterwards.
    monkeypatch.setattr(FakeSentenceTransformer, "load_attempts", [])
    return FakeSentenceTransformer


class TestCacheFirstLoading:
    """The model loads from disk without contacting huggingface.co.

    Measured before the fix: 33 HTTP requests per load of an already-cached
    model, purely to check whether the cached files were still current.
    """

    def test_cached_model_loads_local_only_in_one_attempt(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        EmbeddingModel("fake/model")

        assert fake_st.load_attempts == [True]

    def test_missing_model_falls_back_to_a_download(
        self, fake_st: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fake_st, "cached", False)

        model = EmbeddingModel("fake/model")

        # Tried the cache, found nothing, then loaded normally (downloading).
        assert fake_st.load_attempts == [True, False]
        assert model.name == "fake/model"

    def test_fallback_is_logged(
        self,
        fake_st: type[FakeSentenceTransformer],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A first-run download is slow and large; saying so is the difference
        # between "it's downloading" and "it's hung".
        monkeypatch.setattr(fake_st, "cached", False)

        with caplog.at_level("INFO", logger="app.indexing.embeddings"):
            EmbeddingModel("fake/model")

        assert "downloading" in caplog.text


class TestTruncationGuard:
    """The point of the guard: oversized input is truncated, never rejected."""

    def test_window_over_the_limit_is_refused(
        self, fake_st: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fake_st, "limit", 256)

        with pytest.raises(ChunkWindowTooLargeError):
            EmbeddingModel("fake/model", max_chunk_tokens=480)

    def test_window_equal_to_the_limit_is_allowed(
        self, fake_st: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fake_st, "limit", 256)

        assert EmbeddingModel("fake/model", max_chunk_tokens=256)

    def test_message_names_the_numbers_and_the_model(
        self, fake_st: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fake_st, "limit", 256)

        with pytest.raises(ChunkWindowTooLargeError) as excinfo:
            EmbeddingModel("fake/model", max_chunk_tokens=480)

        message = str(excinfo.value)
        # A guard that fires without saying what to change is a nuisance.
        assert "480" in message
        assert "256" in message
        assert "fake/model" in message

    def test_shipped_defaults_are_a_safe_pairing(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        # Fake reports bge-base's real limit of 512; CHUNK_MAX_TOKENS is 480.
        assert EmbeddingModel("fake/model").max_seq_length == 512


class TestProtocol:
    def test_embedding_model_satisfies_text_embedder(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        assert isinstance(EmbeddingModel("fake/model"), TextEmbedder)

    def test_a_plain_object_with_the_methods_also_satisfies_it(self) -> None:
        # Structural typing: no inheritance, no registration. This is what lets
        # other tests inject a double without importing EmbeddingModel.
        class Minimal:
            def embed_documents(self, texts): return [[0.0] for _ in texts]
            def embed_query(self, text): return [0.0]
            def count_tokens(self, text): return len(text.split())

        assert isinstance(Minimal(), TextEmbedder)


class TestTokenCounting:
    def test_delegates_to_the_model_tokenizer(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        model = EmbeddingModel("fake/model")

        assert model.count_tokens("one two three") == 3

    def test_empty_text_is_zero(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        assert EmbeddingModel("fake/model").count_tokens("") == 0


class TestEmbedDocuments:
    def test_one_vector_per_text_in_order(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        model = EmbeddingModel("fake/model")

        vectors = model.embed_documents(["a", "bb", "ccc"])

        assert len(vectors) == 3
        assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]

    def test_vectors_are_plain_python_floats(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        # Chroma and JSON both reject NumPy scalars, so the conversion at the
        # boundary is load-bearing rather than cosmetic.
        vector = EmbeddingModel("fake/model").embed_documents(["a"])[0]

        assert isinstance(vector, list)
        assert all(type(value) is float for value in vector)

    def test_normalises_and_uses_the_configured_batch_size(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        model = EmbeddingModel("fake/model")

        model.embed_documents(["a"])

        call = model._model.document_calls[0]  # type: ignore[attr-defined]
        assert call["normalize"] is True
        assert call["batch_size"] == 32

    def test_empty_input_short_circuits(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        model = EmbeddingModel("fake/model")

        assert model.embed_documents([]) == []
        assert model._model.document_calls == []  # type: ignore[attr-defined]


class TestEmbedQuery:
    def test_returns_one_flat_vector(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        vector = EmbeddingModel("fake/model").embed_query("how long is the warranty")

        assert len(vector) == 4
        assert all(isinstance(value, float) for value in vector)

    def test_uses_the_query_encoder_not_the_document_encoder(
        self, fake_st: type[FakeSentenceTransformer]
    ) -> None:
        # Retrieval is asymmetric: bge is trained with a prefix on the query
        # side. Encoding a query as a document quietly degrades every result.
        model = EmbeddingModel("fake/model")

        model.embed_query("a question")

        assert len(model._model.query_calls) == 1  # type: ignore[attr-defined]
        assert model._model.document_calls == []  # type: ignore[attr-defined]

    def test_normalises(self, fake_st: type[FakeSentenceTransformer]) -> None:
        model = EmbeddingModel("fake/model")

        model.embed_query("a question")

        assert model._model.query_calls[0]["normalize"] is True  # type: ignore[attr-defined]


class TestGetEmbeddingModel:
    def test_is_cached(self, fake_st: type[FakeSentenceTransformer]) -> None:
        get_embedding_model.cache_clear()
        try:
            assert get_embedding_model() is get_embedding_model()
        finally:
            get_embedding_model.cache_clear()


@pytest.mark.integration
class TestRealModel:
    """Exercises the configured model. Downloads it on first run."""

    def test_real_model_matches_its_documented_shape(self) -> None:
        model = EmbeddingModel()

        assert model.max_seq_length == 512
        assert model.dimensions == 768

    def test_related_text_scores_above_unrelated(self) -> None:
        model = EmbeddingModel()
        query = model.embed_query("how long is the warranty")
        related, unrelated = model.embed_documents(
            [
                "The coverage period is 24 months from the date of purchase.",
                "Tenant shall not sublet the premises without written consent.",
            ]
        )

        # Vectors are normalised, so the dot product is cosine similarity.
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
        assert dot(query, related) > dot(query, unrelated)
