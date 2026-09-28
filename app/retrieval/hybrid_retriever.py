"""The search pipeline: both indexes, fused by rank, then reranked.

    query ─┬─ vector index ──┐
           │                 ├─ reciprocal rank fusion ─ cross-encoder ─ top k
           └─ keyword index ─┘

Two stages with two different jobs. The indexes and fusion are cheap and aim
for *recall*: get the right chunk somewhere into a shortlist of a few dozen.
The cross-encoder is expensive per chunk and aims for *precision*: put the best
chunk first. Neither does both jobs well alone.

There is exactly one path through :meth:`HybridRetriever.search`. No branch
returns vector hits without fusion, or fused hits without reranking — that is
the project's architecture rule, held by structure rather than by convention.
"""

import logging
from typing import Self

from app.config import get_settings
from app.indexing.built_index import open_built_indexes
from app.indexing.embeddings import TextEmbedder, get_embedding_model
from app.indexing.keyword_index import KeywordIndex
from app.indexing.vector_store import VectorStore
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker, get_reranker
from app.retrieval.results import SearchResult

logger = logging.getLogger(__name__)

# Names the two lists carry through fusion, and so the keys of FusedHit.ranks.
VECTOR = "vector"
KEYWORD = "keyword"


class HybridRetriever:
    """Searches the demo corpus.

    The two indexes are concrete classes; only the two *models* sit behind
    protocols, because only they are expensive enough that tests must fake
    them. Tests build small real indexes instead.

    Args:
        keyword_index: The BM25 index.
        vector_store: The vector index.
        embedder: Turns the query into a vector.
        reranker: Scores the fused shortlist.
        candidates_per_index: Hits taken from each index before fusion.
        rrf_k: The reciprocal rank fusion constant.
        top_k: Results returned when :meth:`search` is not told otherwise.
    """

    # The bare `*` makes every parameter after it keyword-only: callers must
    # write `rrf_k=60`, never a positional 60. Three ints side by side are
    # otherwise easy to pass in the wrong order.
    def __init__(
        self,
        keyword_index: KeywordIndex,
        vector_store: VectorStore,
        embedder: TextEmbedder,
        reranker: Reranker,
        *,
        candidates_per_index: int,
        rrf_k: int,
        top_k: int,
    ) -> None:
        self._keyword_index = keyword_index
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._candidates = candidates_per_index
        self._rrf_k = rrf_k
        self._top_k = top_k

    @classmethod
    def open(cls) -> Self:
        """Open the built indexes and load both models, as configured.

        The indexes are checked before either model loads, so a missing or
        stale index is reported immediately. The models are the shared
        singletons, so a long-running process (the API, the UI) loads each only
        once.

        Returns:
            A ready retriever.

        Raises:
            IndexUnavailableError: If the indexes are missing, unreadable, or
                were built with a different embedding model. See
                :func:`app.indexing.built_index.open_built_indexes`.
            ChunkWindowTooLargeError: If CHUNK_MAX_TOKENS exceeds what the
                embedding model can read. That setting governs indexing, but a
                configuration the next build would refuse is refused here too.
        """
        settings = get_settings()
        keyword_index, store = open_built_indexes()

        return cls(
            keyword_index,
            store,
            get_embedding_model(),
            get_reranker(),
            candidates_per_index=settings.retrieval_candidates,
            rrf_k=settings.rrf_k,
            top_k=settings.search_top_k,
        )

    @property
    def chunk_count(self) -> int:
        """Number of chunks searchable, as held by the keyword index.

        ``open()`` has already confirmed the two indexes agree on this count.
        """
        return len(self._keyword_index)

    def documents(self) -> dict[str, int]:
        """The searchable documents and how many chunks each contributed.

        Returns:
            ``{source_id: chunk_count}``, sorted by document name.
        """
        return self._keyword_index.chunks_per_source()

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Find the chunks most relevant to a query.

        Args:
            query: What the user typed.
            top_k: Results to return. Defaults to the configured count.

        Returns:
            Up to ``top_k`` results, most relevant first. A blank query returns
            an empty list without consulting either model.
        """
        limit = self._top_k if top_k is None else top_k
        text = query.strip()
        if not text or limit <= 0:
            return []

        # Stage 1 — recall. Each index ranks on its own terms.
        vector_hits = self._vector_store.query(
            self._embedder.embed_query(text), k=self._candidates
        )
        keyword_hits = self._keyword_index.query(text, k=self._candidates)
        fused = reciprocal_rank_fusion(
            {VECTOR: vector_hits, KEYWORD: keyword_hits}, k=self._rrf_k
        )
        if not fused:
            return []

        # Stage 2 — precision. Every fused candidate is reranked, not just the
        # top few: the shortlist is small, and a chunk that fusion placed low
        # is exactly the one the cross-encoder can rescue.
        scores = self._reranker.score(text, [hit.text for hit in fused])

        # `zip(..., strict=True)` pairs the lists element by element and raises
        # if their lengths differ. Plain zip would silently stop at the shorter
        # one, quietly dropping candidates if a reranker misbehaved.
        scored = list(zip(fused, scores, strict=True))

        # Python's sort is stable, `reverse=True` included, so candidates the
        # reranker scores equally keep their fused order — the same guarantee
        # LINQ's OrderBy gives.
        scored.sort(key=lambda pair: pair[1], reverse=True)

        logger.debug(
            "%r: %d vector + %d keyword hits -> %d fused -> top %d",
            text, len(vector_hits), len(keyword_hits), len(fused), limit,
        )

        return [
            SearchResult(
                chunk_id=hit.chunk_id,
                source_id=hit.source_id,
                chunk_index=hit.chunk_index,
                text=hit.text,
                score=score,
                rrf_score=hit.rrf_score,
                # dict.get returns None for a missing key rather than raising
                # — like TryGetValue collapsed into one expression.
                vector_rank=hit.ranks.get(VECTOR),
                keyword_rank=hit.ranks.get(KEYWORD),
            )
            for hit, score in scored[:limit]
        ]
