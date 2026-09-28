"""Tests for app.indexing.built_index.

Every way the built indexes can be unusable must surface as one
IndexUnavailableError with a message a person can act on — never as a Chroma or
JSON traceback from wherever the problem first happened to show up.
"""

from pathlib import Path

import pytest

from app.indexing.built_index import IndexUnavailableError, open_built_indexes
from app.indexing.keyword_index import KeywordIndex
from app.indexing.vector_store import COLLECTION_NAME, VectorStore
from app.ingestion.chunking import Chunk

MODEL = "test/model"

# Four chunks, per the BM25 IDF note in CLAUDE.md.
CHUNKS = [
    Chunk(
        text=text,
        source_id=f"doc{i}.pdf",
        chunk_index=0,
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
    )
    for i, text in enumerate(
        ["warranty period", "sublease terms", "invoice schedule", "termination notice"]
    )
]
VECTORS = [[1.0 if axis == i else 0.0 for axis in range(4)] for i in range(4)]


@pytest.fixture
def index_dir(
    tmp_path: Path, fresh_settings: None, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point the settings at an empty directory configured for MODEL."""
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", MODEL)
    return tmp_path


def build(index_dir: Path, *, model_name: str = MODEL, keyword_chunks=CHUNKS) -> None:
    """Write both indexes the way scripts/build_index.py does."""
    VectorStore.open(index_dir).replace_all(CHUNKS, VECTORS, model_name=model_name)
    KeywordIndex.build(keyword_chunks).save(index_dir / "bm25_index.json")


class TestUsableIndex:
    def test_returns_both_indexes(self, index_dir: Path) -> None:
        build(index_dir)

        keyword_index, store = open_built_indexes()

        assert len(keyword_index) == 4
        assert store.count() == 4


class TestMissing:
    def test_no_index_says_to_build_first(self, index_dir: Path) -> None:
        with pytest.raises(IndexUnavailableError, match="build_index.py first"):
            open_built_indexes()

    def test_looking_does_not_create_the_vector_database(self, index_dir: Path) -> None:
        with pytest.raises(IndexUnavailableError):
            open_built_indexes()

        assert list(index_dir.iterdir()) == []

    def test_empty_vector_index(self, index_dir: Path) -> None:
        KeywordIndex.build(CHUNKS).save(index_dir / "bm25_index.json")

        with pytest.raises(IndexUnavailableError, match="empty"):
            open_built_indexes()


class TestUnreadableKeywordIndex:
    def test_other_format_version(self, index_dir: Path) -> None:
        build(index_dir)
        path = index_dir / "bm25_index.json"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"version": 1', '"version": 99'),
            encoding="utf-8",
        )

        with pytest.raises(IndexUnavailableError, match="format version 99"):
            open_built_indexes()

    def test_malformed_json(self, index_dir: Path) -> None:
        build(index_dir)
        (index_dir / "bm25_index.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(IndexUnavailableError, match="rebuild"):
            open_built_indexes()

    def test_valid_json_missing_a_field(self, index_dir: Path) -> None:
        build(index_dir)
        (index_dir / "bm25_index.json").write_text('{"version": 1}', encoding="utf-8")

        with pytest.raises(IndexUnavailableError, match="rebuild"):
            open_built_indexes()

    def test_original_error_is_chained(self, index_dir: Path) -> None:
        # `raise ... from exc` keeps the underlying error on __cause__, so a
        # developer can still see exactly what failed to parse.
        build(index_dir)
        (index_dir / "bm25_index.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(IndexUnavailableError) as raised:
            open_built_indexes()

        assert isinstance(raised.value.__cause__, ValueError)


class TestModelMismatch:
    def test_index_from_another_model_is_refused(self, index_dir: Path) -> None:
        # The dangerous case: same vector width, different model. Chroma would
        # accept the query and return confident nonsense.
        build(index_dir, model_name="some/other-model")

        with pytest.raises(IndexUnavailableError) as raised:
            open_built_indexes()

        message = str(raised.value)
        assert "some/other-model" in message
        assert MODEL in message
        assert "rebuild" in message

    def test_index_that_never_recorded_its_model_is_refused(
        self, index_dir: Path
    ) -> None:
        # Indexes built before the model was recorded cannot be trusted either.
        store = VectorStore.open(index_dir)
        store._client.delete_collection(COLLECTION_NAME)
        legacy = store._client.create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine", "embedding_dim": 4}
        )
        legacy.add(
            ids=[f"doc{i}.pdf::0" for i in range(4)],
            embeddings=VECTORS,
            documents=[c.text for c in CHUNKS],
            metadatas=[{"source_id": c.source_id, "chunk_index": 0} for c in CHUNKS],
        )
        KeywordIndex.build(CHUNKS).save(index_dir / "bm25_index.json")

        with pytest.raises(IndexUnavailableError, match="older version"):
            open_built_indexes()


class TestIndexesDisagree:
    def test_different_chunk_counts_are_refused(self, index_dir: Path) -> None:
        # An interrupted build: new vectors, but the old keyword file.
        build(index_dir, keyword_chunks=CHUNKS[:3])

        with pytest.raises(IndexUnavailableError, match="interrupted"):
            open_built_indexes()
