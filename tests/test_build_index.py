"""Tests for scripts/build_index.py.

The script lives outside the ``app`` package, so it is loaded by path. A fake
embedder stands in for the real model, keeping these offline and fast.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from app.indexing.keyword_index import KeywordIndex
from app.indexing.vector_store import VectorStore
from app.ingestion.chunking import Chunk

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_build_index():
    """Import scripts/build_index.py as a module."""
    path = REPO_ROOT / "scripts" / "build_index.py"
    spec = importlib.util.spec_from_file_location("build_index", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_index"] = module
    spec.loader.exec_module(module)
    return module


build_index = _load_build_index()


class FakeEmbeddingModel:
    """Satisfies TextEmbedder without loading anything.

    Vectors are deterministic and derived from the text, so "nearest" is
    predictable but the values are meaningless.
    """

    max_seq_length = 512
    dimensions = 8

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        counts = [0.0] * 8
        for word in text.lower().split():
            counts[hash(word) % 8] += 1.0
        norm = sum(value * value for value in counts) ** 0.5 or 1.0
        return [value / norm for value in counts]


def make_chunk(text: str, source_id: str, chunk_index: int = 0) -> Chunk:
    return Chunk(
        text=text,
        source_id=source_id,
        chunk_index=chunk_index,
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
    )


class TestRelativeSourceId:
    def test_strips_the_corpus_root(self) -> None:
        result = build_index.relative_source_id(
            str(Path("/corpus/Warranty Forms.pdf")), Path("/corpus")
        )

        assert result == "Warranty Forms.pdf"

    def test_keeps_subdirectories(self) -> None:
        result = build_index.relative_source_id(
            str(Path("/corpus/legal/nda.pdf")), Path("/corpus")
        )

        assert result == "legal/nda.pdf"

    def test_uses_posix_separators(self) -> None:
        # Chunk ids must not differ between Windows and Linux, or an index
        # built on one machine stops matching one built on another.
        result = build_index.relative_source_id(
            str(Path("/corpus/legal/nda.pdf")), Path("/corpus")
        )

        assert "\\" not in result

    def test_path_outside_the_root_falls_back_to_the_filename(self) -> None:
        result = build_index.relative_source_id(
            str(Path("/elsewhere/stray.pdf")), Path("/corpus")
        )

        assert result == "stray.pdf"


class TestCollectChunks:
    def test_skips_documents_with_no_extractable_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A scanned PDF extracts to "" without raising. Indexing it would add
        # an empty chunk that matches nothing.
        from app.ingestion.connectors.local_fs import LocalFSConnector

        (tmp_path / "empty.txt").write_text("   \n\n  ", encoding="utf-8")
        (tmp_path / "real.txt").write_text("Coverage period is 24 months.", encoding="utf-8")

        chunks = build_index.collect_chunks(
            LocalFSConnector(tmp_path), FakeEmbeddingModel()
        )

        assert {c.source_id for c in chunks} == {"real.txt"}

    def test_source_ids_are_relative_to_the_corpus(self, tmp_path: Path) -> None:
        from app.ingestion.connectors.local_fs import LocalFSConnector

        (tmp_path / "warranty.txt").write_text("Coverage period.", encoding="utf-8")

        chunks = build_index.collect_chunks(
            LocalFSConnector(tmp_path), FakeEmbeddingModel()
        )

        assert chunks
        assert all(not Path(c.source_id).is_absolute() for c in chunks)

    def test_uses_the_models_tokenizer_for_boundaries(
        self, tmp_path: Path
    ) -> None:
        from app.ingestion.connectors.local_fs import LocalFSConnector

        (tmp_path / "doc.txt").write_text("one two three four five", encoding="utf-8")

        chunks = build_index.collect_chunks(
            LocalFSConnector(tmp_path), FakeEmbeddingModel()
        )

        # FakeEmbeddingModel counts one token per word; estimate_tokens would
        # have reported ceil(5 * 1.3) == 7.
        assert chunks[0].token_count == 5


class TestIndexesAgreeOnChunkIds:
    def test_same_chunk_has_the_same_id_in_both_indexes(
        self, tmp_path: Path
    ) -> None:
        # Phase 3 merges the two result lists by id, so any divergence here
        # would silently prevent fusion from ever pairing a chunk with itself.
        #
        # Four chunks, not two, and deliberately so: BM25Okapi's IDF is
        # log(N - n + 0.5) - log(n + 0.5), which is exactly 0.0 when a term
        # appears in one of two documents. On a two-document corpus every
        # single-document term therefore scores zero and is filtered out. The
        # real index holds 69 chunks, where this never arises.
        chunks = [
            make_chunk("coverage period is 24 months", "warranty.pdf", 0),
            make_chunk("tenant shall not sublet", "sublease.pdf", 0),
            make_chunk("invoice due net 30 terms", "invoice.docx", 0),
            make_chunk("model serial number listed", "manual.pdf", 0),
        ]
        model = FakeEmbeddingModel()
        vectors = model.embed_documents([c.text for c in chunks])

        store = VectorStore.open(tmp_path / "vectors")
        store.replace_all(chunks, vectors)
        keyword = KeywordIndex.build(chunks)

        vector_ids = {h.chunk_id for h in store.query(model.embed_query("coverage"), k=5)}
        keyword_ids = {h.chunk_id for h in keyword.query("coverage", k=5)}

        assert keyword_ids
        assert keyword_ids <= vector_ids


class TestMain:
    def test_verify_without_an_index_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from app.config import Settings, get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "absent"))
        try:
            assert build_index.main(["--verify", "anything"]) == 1
            assert "Run without --verify first" in capsys.readouterr().out
        finally:
            get_settings.cache_clear()

    def test_k_reaches_verify(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            build_index,
            "verify",
            lambda query, k: seen.update(query=query, k=k) or 0,
        )

        assert build_index.main(["--verify", "warranty", "-k", "7"]) == 0
        assert seen == {"query": "warranty", "k": 7}

    def test_k_defaults_to_three(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            build_index, "verify", lambda query, k: seen.update(k=k) or 0
        )

        build_index.main(["--verify", "warranty"])

        assert seen["k"] == 3

    def test_no_arguments_runs_a_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[bool] = []
        monkeypatch.setattr(build_index, "build", lambda: called.append(True) or 0)

        assert build_index.main([]) == 0
        assert called == [True]
