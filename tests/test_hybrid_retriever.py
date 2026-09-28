"""Tests for app.retrieval.hybrid_retriever.

The two indexes are real — a Chroma store under tmp_path and a BM25 index —
while the two models are scripted fakes, so every ranking here is exact:

- Each chunk is stored with a one-hot vector (chunk i points along axis i).
  The cosine between a query vector ``q`` and chunk i is then ``q[i] / |q|``,
  so the query vector's weights *are* the vector ranking.
- The reranker is a function of the passage text, chosen per test.

The corpus holds six chunks rather than two: BM25Okapi's IDF is zero for a term
in one of two documents, which would filter every hit out (see CLAUDE.md).
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from app.indexing.keyword_index import KeywordIndex
from app.indexing.vector_store import VectorStore
from app.ingestion.chunking import Chunk
from app.retrieval import hybrid_retriever
from app.indexing.built_index import IndexUnavailableError
from app.retrieval.hybrid_retriever import HybridRetriever

TEXTS = [
    ("warranty.pdf", "the warranty covers defects for twenty four months"),
    ("sublease.pdf", "tenant shall not sublet the premises"),
    ("services.docx", "how long the services agreement lasts is set out here"),
    ("invoice.docx", "invoice due net thirty days"),
    ("contract.docx", "termination requires thirty days written notice"),
    ("manual.pdf", "serial number is printed on the label"),
]
CHUNKS = [
    Chunk(
        text=text,
        source_id=source,
        chunk_index=0,
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
    )
    for source, text in TEXTS
]
TEXT_TO_ID = {chunk.text: f"{chunk.source_id}::0" for chunk in CHUNKS}

# Vector ranking 1..6 follows corpus order: warranty first, manual last.
DESCENDING = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def one_hot(i: int) -> list[float]:
    return [1.0 if axis == i else 0.0 for axis in range(len(CHUNKS))]


class ScriptedEmbedder:
    """Returns one fixed query vector, and records what it was asked."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("search must never embed documents")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class ScriptedReranker:
    """Scores passages with a supplied function, and records every call."""

    def __init__(self, score_fn: Callable[[str], float] = lambda passage: 0.0) -> None:
        self.score_fn = score_fn
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [self.score_fn(passage) for passage in passages]


def favour(source_id: str) -> Callable[[str], float]:
    """A reranker score function that puts one document on top."""
    return lambda passage: 1.0 if TEXT_TO_ID[passage].startswith(source_id) else 0.0


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    vector_store = VectorStore.open(tmp_path / "vectors")
    vector_store.replace_all(
        CHUNKS, [one_hot(i) for i in range(len(CHUNKS))], model_name="test/model"
    )
    return vector_store


@pytest.fixture
def keyword_index() -> KeywordIndex:
    return KeywordIndex.build(CHUNKS)


# A factory fixture: returns a function, so each test can build a retriever with
# its own settings while the indexes still come from fixtures.
@pytest.fixture
def make_retriever(
    store: VectorStore, keyword_index: KeywordIndex
) -> Callable[..., HybridRetriever]:
    def build(
        embedder: ScriptedEmbedder | None = None,
        reranker: ScriptedReranker | None = None,
        *,
        candidates: int = 6,
        rrf_k: int = 60,
        top_k: int = 5,
    ) -> HybridRetriever:
        return HybridRetriever(
            keyword_index,
            store,
            embedder or ScriptedEmbedder(DESCENDING),
            reranker or ScriptedReranker(),
            candidates_per_index=candidates,
            rrf_k=rrf_k,
            top_k=top_k,
        )

    return build


def result_ids(results: list) -> list[str]:
    return [r.chunk_id for r in results]


class TestRerankingDecidesTheOrder:
    def test_reranker_can_promote_the_last_fused_candidate(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        # manual.pdf is last in the vector ranking and absent from BM25 —
        # the lowest fused score of all. The reranker alone lifts it to first.
        retriever = make_retriever(reranker=ScriptedReranker(favour("manual.pdf")))

        results = retriever.search("termination notice")

        assert results[0].chunk_id == "manual.pdf::0"
        assert results[0].vector_rank == 6
        assert results[0].keyword_rank is None

    def test_results_are_in_descending_reranker_score(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        # Score = passage length, so the order is known in advance.
        retriever = make_retriever(reranker=ScriptedReranker(lambda p: float(len(p))))

        results = retriever.search("termination notice", top_k=6)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_equal_reranker_scores_keep_the_fused_order(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        # "zzz" matches no BM25 term, so the fused order is the vector order;
        # a reranker that scores everything 0.0 must leave it untouched.
        results = make_retriever().search("zzz", top_k=5)

        assert result_ids(results) == [
            "warranty.pdf::0",
            "sublease.pdf::0",
            "services.docx::0",
            "invoice.docx::0",
            "contract.docx::0",
        ]


class TestBothIndexesContribute:
    def test_a_keyword_only_chunk_reaches_the_reranker(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        # Two candidates per index: the vector side returns warranty and
        # sublease, which never mention termination. Only BM25 finds contract.
        reranker = ScriptedReranker(favour("contract.docx"))
        retriever = make_retriever(reranker=reranker, candidates=2)

        results = retriever.search("termination notice")

        _, passages = reranker.calls[0]
        assert "termination requires thirty days written notice" in passages
        assert results[0].chunk_id == "contract.docx::0"
        assert results[0].vector_rank is None
        assert results[0].keyword_rank == 1

    def test_a_vector_only_chunk_reaches_the_reranker(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        reranker = ScriptedReranker(favour("warranty.pdf"))
        retriever = make_retriever(reranker=reranker, candidates=2)

        results = retriever.search("termination notice")

        assert results[0].chunk_id == "warranty.pdf::0"
        assert results[0].vector_rank == 1
        assert results[0].keyword_rank is None

    def test_a_chunk_found_by_both_carries_both_ranks(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        retriever = make_retriever(reranker=ScriptedReranker(favour("contract.docx")))

        top = retriever.search("termination notice")[0]

        assert top.chunk_id == "contract.docx::0"
        assert top.vector_rank == 5
        assert top.keyword_rank == 1

    def test_no_keyword_matches_still_returns_reranked_results(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        # "INV-2024" on the real corpus: BM25 finds nothing, the vector index
        # still does. The results must still pass through the reranker.
        reranker = ScriptedReranker(favour("invoice.docx"))
        retriever = make_retriever(reranker=reranker)

        results = retriever.search("zzz")

        assert results[0].chunk_id == "invoice.docx::0"
        assert all(r.keyword_rank is None for r in results)
        assert len(reranker.calls) == 1

    def test_rrf_score_uses_the_configured_k(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        retriever = make_retriever(
            reranker=ScriptedReranker(favour("warranty.pdf")), rrf_k=10
        )

        top = retriever.search("zzz")[0]

        # Vector rank 1, no keyword rank: 1 / (10 + 1).
        assert top.rrf_score == pytest.approx(1 / 11)


class TestPipelineContract:
    """The architecture rule: nothing reaches the caller without fusion and
    reranking."""

    def test_reranker_sees_the_query_and_every_fused_candidate_once(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        reranker = ScriptedReranker()
        retriever = make_retriever(reranker=reranker)

        retriever.search("  termination notice  ")

        assert len(reranker.calls) == 1
        query, passages = reranker.calls[0]
        assert query == "termination notice"
        # Six candidates per index covers the whole corpus, so all six chunks
        # are fused, and all six are reranked.
        assert sorted(passages) == sorted(chunk.text for chunk in CHUNKS)

    def test_each_index_is_asked_for_the_candidate_count(
        self,
        make_retriever: Callable[..., HybridRetriever],
        store: VectorStore,
        keyword_index: KeywordIndex,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        asked: dict[str, int] = {}
        real_vector_query = store.query
        real_keyword_query = keyword_index.query

        # Replacing a method on one *instance* — a spy that records the call
        # and forwards it. Python allows this on any object without __slots__.
        def vector_spy(vector: list[float], k: int = 5) -> list:
            asked["vector"] = k
            return real_vector_query(vector, k=k)

        def keyword_spy(text: str, k: int = 5) -> list:
            asked["keyword"] = k
            return real_keyword_query(text, k=k)

        monkeypatch.setattr(store, "query", vector_spy)
        monkeypatch.setattr(keyword_index, "query", keyword_spy)

        make_retriever(candidates=4).search("termination notice")

        assert asked == {"vector": 4, "keyword": 4}

    def test_a_reranker_returning_the_wrong_count_is_an_error(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        class Broken:
            def score(self, query: str, passages: list[str]) -> list[float]:
                return [1.0]

        retriever = make_retriever(reranker=Broken())

        with pytest.raises(ValueError):
            retriever.search("termination notice")


class TestResultCount:
    def test_uses_the_configured_top_k(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        assert len(make_retriever(top_k=3).search("termination notice")) == 3

    def test_explicit_top_k_overrides_it(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        assert len(make_retriever(top_k=3).search("termination notice", top_k=1)) == 1

    def test_never_more_than_the_shortlist(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        results = make_retriever().search("termination notice", top_k=50)

        assert len(results) == len(CHUNKS)

    def test_zero_top_k_returns_nothing(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        assert make_retriever().search("termination notice", top_k=0) == []


class TestChunkCount:
    def test_reports_the_indexed_chunks(
        self, make_retriever: Callable[..., HybridRetriever]
    ) -> None:
        assert make_retriever().chunk_count == len(CHUNKS)


class TestBlankQuery:
    @pytest.mark.parametrize("query", ["", "   ", "\n\t"])
    def test_returns_nothing_without_consulting_either_model(
        self, make_retriever: Callable[..., HybridRetriever], query: str
    ) -> None:
        embedder = ScriptedEmbedder(DESCENDING)
        reranker = ScriptedReranker()

        assert make_retriever(embedder, reranker).search(query) == []
        assert embedder.queries == []
        assert reranker.calls == []



class TestOpen:
    def test_missing_index_is_refused_with_instructions(
        self, tmp_path: Path, fresh_settings: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index_dir = tmp_path / "absent"
        monkeypatch.setenv("VECTOR_STORE_PATH", str(index_dir))

        with pytest.raises(IndexUnavailableError, match="build_index.py"):
            HybridRetriever.open()

    def test_looking_does_not_create_the_index_directory(
        self, tmp_path: Path, fresh_settings: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index_dir = tmp_path / "absent"
        monkeypatch.setenv("VECTOR_STORE_PATH", str(index_dir))

        with pytest.raises(IndexUnavailableError):
            HybridRetriever.open()

        assert not index_dir.exists()

    def test_empty_vector_store_is_refused(
        self, tmp_path: Path, fresh_settings: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path))
        KeywordIndex.build(CHUNKS).save(tmp_path / "bm25_index.json")

        with pytest.raises(IndexUnavailableError, match="empty"):
            HybridRetriever.open()

    def test_opens_the_configured_indexes_and_settings(
        self, tmp_path: Path, fresh_settings: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path))
        monkeypatch.setenv("SEARCH_TOP_K", "2")
        # The index records the model that built it; open() refuses a mismatch.
        monkeypatch.setenv("EMBEDDING_MODEL_NAME", "test/model")
        VectorStore.open(tmp_path).replace_all(
            CHUNKS, [one_hot(i) for i in range(len(CHUNKS))], model_name="test/model"
        )
        KeywordIndex.build(CHUNKS).save(tmp_path / "bm25_index.json")
        # The real models are the shared singletons; swap in the fakes where
        # hybrid_retriever looks them up.
        monkeypatch.setattr(
            hybrid_retriever, "get_embedding_model", lambda: ScriptedEmbedder(DESCENDING)
        )
        monkeypatch.setattr(
            hybrid_retriever, "get_reranker", lambda: ScriptedReranker(favour("contract.docx"))
        )

        results = HybridRetriever.open().search("termination notice")

        assert result_ids(results) == ["contract.docx::0", "warranty.pdf::0"]

    def test_stale_index_is_refused_before_either_model_loads(
        self, tmp_path: Path, fresh_settings: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Loading both models takes about seven seconds. An index that search
        # cannot use must be reported without paying for that.
        monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path))
        monkeypatch.setenv("EMBEDDING_MODEL_NAME", "test/model")
        VectorStore.open(tmp_path).replace_all(
            CHUNKS, [one_hot(i) for i in range(len(CHUNKS))], model_name="some/other-model"
        )
        KeywordIndex.build(CHUNKS).save(tmp_path / "bm25_index.json")

        def must_not_load():
            raise AssertionError("a model was loaded before the index was checked")

        monkeypatch.setattr(hybrid_retriever, "get_embedding_model", must_not_load)
        monkeypatch.setattr(hybrid_retriever, "get_reranker", must_not_load)

        with pytest.raises(IndexUnavailableError, match="some/other-model"):
            HybridRetriever.open()


@pytest.mark.integration
class TestRealCorpus:
    """Runs against the real built index and both real models."""

    QUERY = "how long is the warranty"

    @pytest.fixture
    def settings(self):
        from app.config import get_settings

        settings = get_settings()
        if not settings.keyword_index_path.exists():
            pytest.skip("index not built; run python scripts/build_index.py")
        return settings

    def test_fusion_and_reranking_fix_the_warranty_query(self, settings) -> None:
        # The premise first, so this test cannot pass vacuously: on its own,
        # BM25 ranks an unrelated clause above the warranty document.
        keyword_top = KeywordIndex.load(settings.keyword_index_path).query(self.QUERY, k=1)
        assert keyword_top[0].source_id != "Warranty Forms.pdf"

        results = HybridRetriever.open().search(self.QUERY)

        assert results[0].source_id == "Warranty Forms.pdf"
