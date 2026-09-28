"""Tests for scripts/search.py.

Loaded by path, like test_build_index.py. The retriever is replaced with a
stub, so nothing here opens an index or loads a model.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from app.retrieval.results import SearchResult

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_search():
    """Import scripts/search.py as a module."""
    path = REPO_ROOT / "scripts" / "search.py"
    spec = importlib.util.spec_from_file_location("search_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["search_script"] = module
    spec.loader.exec_module(module)
    return module


search_script = _load_search()


# `autouse=True` applies this to every test in the file. main() hides the
# transformers progress bars, and importing transformers to do so costs over
# two seconds; the helper itself is tested in test_cli.py.
@pytest.fixture(autouse=True)
def no_transformers_import(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    calls: list[bool] = []
    monkeypatch.setattr(search_script, "silence_model_loading_bars", lambda: calls.append(True))
    return calls


def result(source_id: str, score: float, vector: int | None, keyword: int | None) -> SearchResult:
    return SearchResult(
        chunk_id=f"{source_id}::0",
        source_id=source_id,
        chunk_index=0,
        text=f"body of {source_id}",
        score=score,
        rrf_score=0.03,
        vector_rank=vector,
        keyword_rank=keyword,
    )


class StubRetriever:
    """Stands in for HybridRetriever; records the requested result count."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.asked: list[tuple[str, int | None]] = []

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.asked.append((query, top_k))
        return self.results


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubRetriever:
    retriever = StubRetriever(
        [
            result("Warranty Forms.pdf", 7.4, 2, 2),
            result("contract.docx", 1.2, None, 1),
        ]
    )
    monkeypatch.setattr(
        search_script.HybridRetriever, "open", classmethod(lambda cls: retriever)
    )
    return retriever


class TestOutput:
    def test_prints_results_in_order(
        self, stub: StubRetriever, capsys: pytest.CaptureFixture
    ) -> None:
        assert search_script.main(["how long is the warranty"]) == 0

        out = capsys.readouterr().out
        assert out.index("Warranty Forms.pdf#0") < out.index("contract.docx#0")

    def test_shows_where_each_index_placed_the_chunk(
        self, stub: StubRetriever, capsys: pytest.CaptureFixture
    ) -> None:
        search_script.main(["warranty"])

        lines = capsys.readouterr().out.splitlines()
        contract = next(line for line in lines if "contract.docx#0" in line)
        assert "vec   -" in contract
        assert "bm25  #1" in contract

    def test_no_results(
        self, stub: StubRetriever, capsys: pytest.CaptureFixture
    ) -> None:
        stub.results = []

        assert search_script.main(["zzz"]) == 0
        assert "(no results)" in capsys.readouterr().out


class TestArguments:
    def test_k_reaches_the_retriever(self, stub: StubRetriever) -> None:
        search_script.main(["warranty", "-k", "3"])

        assert stub.asked == [("warranty", 3)]

    def test_k_defaults_to_the_configured_count(self, stub: StubRetriever) -> None:
        search_script.main(["warranty"])

        # None tells the retriever to use SEARCH_TOP_K.
        assert stub.asked == [("warranty", None)]

    @pytest.mark.parametrize("k", ["0", "-2", "three"])
    def test_k_must_be_a_positive_number(self, stub: StubRetriever, k: str) -> None:
        # argparse reports a bad argument by exiting with code 2.
        with pytest.raises(SystemExit) as exited:
            search_script.main(["warranty", "-k", k])

        assert exited.value.code == 2
        assert stub.asked == []

    @pytest.mark.parametrize("query", ["", "   "])
    def test_blank_query_is_rejected_before_anything_loads(
        self, monkeypatch: pytest.MonkeyPatch, query: str
    ) -> None:
        opened: list[bool] = []
        monkeypatch.setattr(
            search_script.HybridRetriever,
            "open",
            classmethod(lambda cls: opened.append(True)),
        )

        with pytest.raises(SystemExit) as exited:
            search_script.main([query])

        assert exited.value.code == 2
        assert opened == []

    def test_model_loading_bars_are_hidden(
        self, stub: StubRetriever, no_transformers_import: list[bool]
    ) -> None:
        search_script.main(["warranty"])

        assert no_transformers_import == [True]


class TestRefusals:
    def test_missing_index_exits_1_with_instructions(
        self,
        tmp_path: Path,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "absent"))

        assert search_script.main(["warranty"]) == 1
        assert "build_index.py" in capsys.readouterr().err

    def test_personal_mode_exits_2_without_a_traceback(
        self,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from app.modes import PERSONAL_MODE_MESSAGE

        monkeypatch.setenv("APP_MODE", "personal")

        assert search_script.main(["warranty"]) == 2
        err = capsys.readouterr().err
        assert PERSONAL_MODE_MESSAGE in err
        assert "Traceback" not in err

    def test_invalid_configuration_exits_2(
        self,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setenv("SEARCH_TOP_K", "50")

        assert search_script.main(["warranty"]) == 2
        assert "search_top_k" in capsys.readouterr().err

    def test_chunk_window_too_large_exits_2_without_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        # Loading the embedding model enforces CHUNK_MAX_TOKENS, so a bad value
        # surfaces during open(). It must read as a message, as in build_index.
        from app.indexing.embeddings import ChunkWindowTooLargeError

        def refuse(cls):
            raise ChunkWindowTooLargeError("CHUNK_MAX_TOKENS is 600 but the model reads 512")

        monkeypatch.setattr(search_script.HybridRetriever, "open", classmethod(refuse))

        assert search_script.main(["warranty"]) == 2
        err = capsys.readouterr().err
        assert "reads 512" in err
        assert "Traceback" not in err

    def test_index_from_another_model_exits_1(
        self,
        tmp_path: Path,
        fresh_settings: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        from app.indexing.keyword_index import KeywordIndex
        from app.indexing.vector_store import VectorStore
        from app.ingestion.chunking import Chunk

        chunks = [
            Chunk(text=f"chunk {i}", source_id=f"doc{i}.pdf", chunk_index=0,
                  token_count=2, start_char=0, end_char=7)
            for i in range(4)
        ]
        vectors = [[1.0 if axis == i else 0.0 for axis in range(4)] for i in range(4)]
        VectorStore.open(tmp_path).replace_all(chunks, vectors, model_name="some/other-model")
        KeywordIndex.build(chunks).save(tmp_path / "bm25_index.json")
        monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path))

        assert search_script.main(["warranty"]) == 1
        err = capsys.readouterr().err
        assert "some/other-model" in err
        assert "Traceback" not in err

    def test_unexpected_errors_still_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(cls):
            raise RuntimeError("a real bug")

        monkeypatch.setattr(search_script.HybridRetriever, "open", classmethod(broken))

        with pytest.raises(RuntimeError, match="a real bug"):
            search_script.main(["warranty"])
