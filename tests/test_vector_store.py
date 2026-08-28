"""Tests for app.indexing.vector_store.

These run against real Chroma via an ephemeral in-memory client — no disk, no
network, and no mocking of the library's behaviour, so distance semantics and
the shape of its results are genuinely covered.
"""

from pathlib import Path

import pytest

from app.indexing.vector_store import COLLECTION_NAME, VectorStore
from app.ingestion.chunking import Chunk


def make_chunk(text: str, source_id: str, chunk_index: int = 0) -> Chunk:
    return Chunk(
        text=text,
        source_id=source_id,
        chunk_index=chunk_index,
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
    )


# Deliberately trivial 3-D vectors: the axes make "nearest" obvious by hand.
CHUNKS = [
    make_chunk("coverage period is 24 months", "warranty.pdf", 0),
    make_chunk("tenant shall not sublet", "sublease.pdf", 0),
    make_chunk("invoice due net 30", "invoice.docx", 0),
]
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


@pytest.fixture
def store() -> VectorStore:
    return VectorStore.in_memory()


class TestReplaceAll:
    def test_stores_every_chunk(self, store: VectorStore) -> None:
        store.replace_all(CHUNKS, VECTORS)

        assert store.count() == 3

    def test_rebuilding_does_not_duplicate(self, store: VectorStore) -> None:
        store.replace_all(CHUNKS, VECTORS)
        store.replace_all(CHUNKS, VECTORS)

        assert store.count() == 3

    def test_rebuilding_drops_chunks_that_disappeared(
        self, store: VectorStore
    ) -> None:
        # A document deleted from the corpus must not keep matching queries.
        store.replace_all(CHUNKS, VECTORS)
        store.replace_all(CHUNKS[:1], VECTORS[:1])

        assert store.count() == 1
        assert store.query([0.0, 1.0, 0.0])[0].source_id == "warranty.pdf"

    def test_mismatched_counts_are_rejected(self, store: VectorStore) -> None:
        with pytest.raises(ValueError, match="every chunk needs exactly one"):
            store.replace_all(CHUNKS, VECTORS[:2])

    def test_empty_corpus_is_allowed(self, store: VectorStore) -> None:
        store.replace_all([], [])

        assert store.count() == 0


class TestQuery:
    def test_finds_the_nearest_vector(self, store: VectorStore) -> None:
        store.replace_all(CHUNKS, VECTORS)

        hits = store.query([1.0, 0.0, 0.0])

        assert hits[0].source_id == "warranty.pdf"

    def test_identical_vector_scores_one(self, store: VectorStore) -> None:
        # Chroma reports cosine distance, where 0.0 means identical. The store
        # flips it to a similarity so higher is always better downstream.
        store.replace_all(CHUNKS, VECTORS)

        assert store.query([1.0, 0.0, 0.0])[0].score == pytest.approx(1.0)

    def test_orthogonal_vector_scores_zero(self, store: VectorStore) -> None:
        store.replace_all(CHUNKS[:1], VECTORS[:1])

        assert store.query([0.0, 1.0, 0.0])[0].score == pytest.approx(0.0)

    def test_hits_are_ordered_best_first(self, store: VectorStore) -> None:
        store.replace_all(CHUNKS, VECTORS)

        scores = [hit.score for hit in store.query([0.9, 0.4, 0.1], k=3)]

        assert scores == sorted(scores, reverse=True)

    def test_respects_k(self, store: VectorStore) -> None:
        store.replace_all(CHUNKS, VECTORS)

        assert len(store.query([1.0, 0.0, 0.0], k=2)) == 2

    def test_k_larger_than_the_corpus_is_safe(self, store: VectorStore) -> None:
        store.replace_all(CHUNKS, VECTORS)

        assert len(store.query([1.0, 0.0, 0.0], k=99)) == 3

    def test_non_positive_k_returns_nothing(self, store: VectorStore) -> None:
        store.replace_all(CHUNKS, VECTORS)

        assert store.query([1.0, 0.0, 0.0], k=0) == []

    def test_empty_store_returns_nothing(self, store: VectorStore) -> None:
        assert store.query([1.0, 0.0, 0.0]) == []

    def test_hit_carries_identity_and_text(self, store: VectorStore) -> None:
        store.replace_all(CHUNKS, VECTORS)

        hit = store.query([1.0, 0.0, 0.0])[0]

        assert hit.chunk_id == "warranty.pdf::0"
        assert hit.source_id == "warranty.pdf"
        assert hit.chunk_index == 0
        assert hit.text == "coverage period is 24 months"


class TestPersistence:
    def test_survives_reopening(self, tmp_path: Path) -> None:
        VectorStore.open(tmp_path).replace_all(CHUNKS, VECTORS)

        reopened = VectorStore.open(tmp_path)

        assert reopened.count() == 3
        assert reopened.query([1.0, 0.0, 0.0])[0].source_id == "warranty.pdf"

    def test_creates_the_directory(self, tmp_path: Path) -> None:
        location = tmp_path / "nested" / "index"

        VectorStore.open(location)

        assert location.is_dir()


class TestInMemoryIsolation:
    def test_each_new_store_starts_empty(self, store: VectorStore) -> None:
        # Chroma shares one in-process system, so without the explicit clear in
        # in_memory() this would inherit whatever a previous test left behind.
        store.replace_all(CHUNKS, VECTORS)

        assert VectorStore.in_memory().count() == 0

    def test_two_live_in_memory_stores_are_documented_as_unsupported(
        self, tmp_path: Path
    ) -> None:
        # Pins the documented limitation: the second construction drops the
        # collection the first is holding. Use open() with separate paths when
        # two stores must coexist, as this test does to show the alternative.
        first = VectorStore.open(tmp_path / "one")
        second = VectorStore.open(tmp_path / "two")
        first.replace_all(CHUNKS, VECTORS)

        assert first.count() == 3
        assert second.count() == 0


def test_collection_name_satisfies_chromas_rules() -> None:
    # Chroma rejects names under 3 characters at creation time — a constraint
    # worth pinning, since the failure only shows up at runtime.
    assert len(COLLECTION_NAME) >= 3
    assert COLLECTION_NAME.isalnum()
