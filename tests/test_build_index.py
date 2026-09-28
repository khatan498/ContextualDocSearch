"""Tests for scripts/build_index.py.

The script lives outside the ``app`` package, so it is loaded by path. The
``fake_embedder`` fixture from conftest.py stands in for the real model, keeping
these offline and fast.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from app.indexing.keyword_index import KeywordIndex
from app.indexing.vector_store import VectorStore
from app.ingestion.chunking import Chunk
from conftest import FakeEmbeddingModel

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


# main() hides the transformers progress bars, and importing transformers to do
# so costs over two seconds. The helper itself is tested in test_cli.py.
@pytest.fixture(autouse=True)
def no_transformers_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_index, "silence_model_loading_bars", lambda: None)


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
        self, tmp_path: Path, fake_embedder: FakeEmbeddingModel
    ) -> None:
        # A scanned PDF extracts to "" without raising. Indexing it would add
        # an empty chunk that matches nothing.
        from app.ingestion.connectors.local_fs import LocalFSConnector

        (tmp_path / "empty.txt").write_text("   \n\n  ", encoding="utf-8")
        (tmp_path / "real.txt").write_text("Coverage period is 24 months.", encoding="utf-8")

        chunks = build_index.collect_chunks(
            LocalFSConnector(tmp_path), fake_embedder
        )

        assert {c.source_id for c in chunks} == {"real.txt"}

    def test_source_ids_are_relative_to_the_corpus(
        self, tmp_path: Path, fake_embedder: FakeEmbeddingModel
    ) -> None:
        from app.ingestion.connectors.local_fs import LocalFSConnector

        (tmp_path / "warranty.txt").write_text("Coverage period.", encoding="utf-8")

        chunks = build_index.collect_chunks(
            LocalFSConnector(tmp_path), fake_embedder
        )

        assert chunks
        assert all(not Path(c.source_id).is_absolute() for c in chunks)

    def test_uses_the_models_tokenizer_for_boundaries(
        self, tmp_path: Path, fake_embedder: FakeEmbeddingModel
    ) -> None:
        from app.ingestion.connectors.local_fs import LocalFSConnector

        (tmp_path / "doc.txt").write_text("one two three four five", encoding="utf-8")

        chunks = build_index.collect_chunks(
            LocalFSConnector(tmp_path), fake_embedder
        )

        # FakeEmbeddingModel counts one token per word; estimate_tokens would
        # have reported ceil(5 * 1.3) == 7.
        assert chunks[0].token_count == 5


class TestBuild:
    def test_search_accepts_what_the_build_wrote(
        self,
        tmp_path: Path,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        fake_embedder: FakeEmbeddingModel,
    ) -> None:
        # The round trip: build() must record the model that actually embedded
        # the chunks, or search would refuse every index it produces — or,
        # worse, accept one whose vectors came from somewhere else.
        from app.indexing.built_index import open_built_indexes

        corpus = tmp_path / "docs"
        corpus.mkdir()
        (corpus / "warranty.txt").write_text("Coverage period is 24 months.", encoding="utf-8")
        monkeypatch.setenv("SAMPLE_DOCS_PATH", str(corpus))
        monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "index"))
        monkeypatch.setenv("EMBEDDING_MODEL_NAME", fake_embedder.name)
        monkeypatch.setattr(build_index, "EmbeddingModel", lambda: fake_embedder)

        assert build_index.build() == 0

        keyword_index, store = open_built_indexes()
        assert store.embedding_model == "fake/embedder"
        assert len(keyword_index) == store.count() == 1


class TestIndexesAgreeOnChunkIds:
    def test_same_chunk_has_the_same_id_in_both_indexes(
        self, tmp_path: Path, fake_embedder: FakeEmbeddingModel
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
        model = fake_embedder
        vectors = model.embed_documents([c.text for c in chunks])

        store = VectorStore.open(tmp_path / "vectors")
        store.replace_all(chunks, vectors, model_name="test/model")
        keyword = KeywordIndex.build(chunks)

        vector_ids = {h.chunk_id for h in store.query(model.embed_query("coverage"), k=5)}
        keyword_ids = {h.chunk_id for h in keyword.query("coverage", k=5)}

        assert keyword_ids
        assert keyword_ids <= vector_ids


class TestMain:
    def test_verify_without_an_index_exits_1_with_instructions(
        self,
        tmp_path: Path,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "absent"))

        assert build_index.main(["--verify", "anything"]) == 1
        err = capsys.readouterr().err
        assert "build_index.py" in err
        assert "Traceback" not in err

    def test_verify_refuses_an_index_from_another_model(
        self,
        tmp_path: Path,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        fake_embedder: FakeEmbeddingModel,
    ) -> None:
        # The diagnostic runs the same checks as search, so it cannot query
        # vectors from one model with a query vector from another.
        monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path))
        chunks = [make_chunk(f"chunk number {i}", f"doc{i}.pdf") for i in range(4)]
        VectorStore.open(tmp_path).replace_all(
            chunks, fake_embedder.embed_documents([c.text for c in chunks]),
            model_name="some/other-model",
        )
        KeywordIndex.build(chunks).save(tmp_path / "bm25_index.json")

        assert build_index.main(["--verify", "anything"]) == 1
        assert "some/other-model" in capsys.readouterr().err

    @pytest.mark.parametrize("k", ["0", "-1", "three"])
    def test_k_must_be_a_positive_number(self, k: str) -> None:
        # argparse reports a bad argument by exiting with code 2.
        with pytest.raises(SystemExit) as exited:
            build_index.main(["--verify", "warranty", "-k", k])

        assert exited.value.code == 2

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



class TestRefusalsAreMessagesNotTracebacks:
    """Deliberate refusals print their message and exit 2.

    Before this, `APP_MODE=personal python scripts/build_index.py` printed a
    full traceback with the explanation buried at the bottom.
    """

    def test_personal_mode(
        self,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from app.modes import PERSONAL_MODE_MESSAGE

        monkeypatch.setenv("APP_MODE", "personal")

        assert build_index.main(["--verify", "anything"]) == 2
        err = capsys.readouterr().err
        assert PERSONAL_MODE_MESSAGE in err
        assert "Traceback" not in err

    def test_invalid_configuration(
        self,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setenv("CHUNK_MIN_TOKENS", "900")
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "100")

        assert build_index.main(["--verify", "anything"]) == 2
        err = capsys.readouterr().err
        # pydantic's message names the settings at fault.
        assert "invalid configuration" in err
        assert "chunk_min_tokens" in err

    def test_chunk_window_too_large_for_the_model(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from app.indexing.embeddings import ChunkWindowTooLargeError

        def refuse() -> int:
            raise ChunkWindowTooLargeError("CHUNK_MAX_TOKENS is 900 but the model reads 512")

        monkeypatch.setattr(build_index, "build", refuse)

        assert build_index.main([]) == 2
        assert "reads 512" in capsys.readouterr().err

    def test_unexpected_errors_still_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Only the deliberate refusals are softened. A genuine bug must keep
        # its traceback, or it becomes much harder to find.
        def broken() -> int:
            raise RuntimeError("a real bug")

        monkeypatch.setattr(build_index, "build", broken)

        with pytest.raises(RuntimeError, match="a real bug"):
            build_index.main([])
