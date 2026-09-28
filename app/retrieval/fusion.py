"""Reciprocal rank fusion: merge several ranked lists into one.

The vector index and the keyword index each return their own ranking, scored on
scales that cannot be compared — cosine similarity in [-1, 1], BM25 unbounded
and corpus-relative. Reciprocal rank fusion (RRF) sidesteps that entirely: it
looks only at where a chunk *placed* in each list, never at the raw score.

Each chunk scores ``sum(1 / (k + rank))`` over every list it appears in, with
``rank`` starting at 1. The constant ``k`` (60 by convention) flattens the
difference between neighbouring ranks, so a chunk that both indexes place
second beats one that a single index places first. Agreement between two
independent signals is stronger evidence than one strong opinion.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.indexing.hits import IndexHit


@dataclass(frozen=True, slots=True)
class FusedHit:
    """One chunk after fusion, remembering where each index placed it.

    Attributes:
        chunk_id: Stable id, as built by ``make_chunk_id``.
        source_id: Document the chunk came from, relative to the corpus root.
        chunk_index: Position of the chunk within its document.
        text: The chunk body.
        rrf_score: The fused score. Only comparable within one fusion.
        ranks: 1-based rank per list name, e.g. ``{"vector": 2, "keyword": 2}``.
            A list that did not return the chunk has no key.
    """

    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    rrf_score: float
    ranks: dict[str, int]


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[IndexHit]], k: int = 60
) -> list[FusedHit]:
    """Fuse ranked lists by reciprocal rank.

    Each hit's ``score`` is ignored completely; only its position counts.

    Args:
        ranked_lists: Each index's hits, best first, keyed by a name that is
            recorded in :attr:`FusedHit.ranks` (for example ``"vector"``).
        k: The RRF constant. Must be positive.

    Returns:
        Every distinct chunk from every list, best fused score first. Equal
        scores are ordered by the chunk's best single rank, then by
        ``chunk_id``, so the output is deterministic.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # `defaultdict(dict)` creates an empty dict the first time a missing key is
    # read — like C#'s `dict.TryAdd(key, new())` folded into the lookup.
    ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    hits_by_id: dict[str, IndexHit] = {}

    for list_name, hits in ranked_lists.items():
        # `enumerate(hits, start=1)` yields (1, first), (2, second), ... — a
        # 1-based index alongside each item, with no counter to maintain.
        for rank, hit in enumerate(hits, start=1):
            # A chunk listed twice keeps its first — best — rank in this list.
            ranks[hit.chunk_id].setdefault(list_name, rank)
            hits_by_id.setdefault(hit.chunk_id, hit)

    # A dict comprehension: builds {chunk_id: score} in one expression, the
    # equivalent of LINQ's ToDictionary(...).
    scores = {
        chunk_id: sum(1.0 / (k + rank) for rank in placed.values())
        for chunk_id, placed in ranks.items()
    }

    # Sorting by a tuple compares element by element, so this is
    # OrderByDescending(score).ThenBy(best rank).ThenBy(chunk_id). Negating the
    # score gives descending order within an otherwise ascending sort.
    ordered = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            min(ranks[chunk_id].values()),
            chunk_id,
        ),
    )

    return [
        FusedHit(
            chunk_id=chunk_id,
            source_id=hits_by_id[chunk_id].source_id,
            chunk_index=hits_by_id[chunk_id].chunk_index,
            text=hits_by_id[chunk_id].text,
            rrf_score=scores[chunk_id],
            ranks=dict(ranks[chunk_id]),
        )
        for chunk_id in ordered
    ]
