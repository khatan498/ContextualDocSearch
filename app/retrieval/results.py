"""The shape a search returns."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked search result.

    ``@dataclass(frozen=True)`` generates the constructor, equality and repr,
    and makes instances immutable — the equivalent of a C# ``record``.

    Attributes:
        chunk_id: Stable id, as built by ``make_chunk_id``.
        source_id: Document the chunk came from, relative to the corpus root.
        chunk_index: Position of the chunk within its document.
        text: The chunk body.
        score: The reranker's relevance score. Higher is better; comparable
            only against other results for the same query.
        rrf_score: The fused score the chunk reached the reranker with.
        vector_rank: Where the vector index placed it, 1-based, or ``None`` if
            the vector index did not return it.
        keyword_rank: Where the BM25 index placed it, or ``None``.
    """

    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    score: float
    rrf_score: float
    vector_rank: int | None
    keyword_rank: int | None
