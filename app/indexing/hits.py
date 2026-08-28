"""The shape both indexes return, and how a chunk is identified.

The vector index and the keyword index score chunks on completely different
scales — cosine similarity sits in roughly [-1, 1], BM25 is unbounded and
corpus-dependent. They are deliberately not normalised against each other here.
Phase 3 fuses them with reciprocal rank fusion, which uses each hit's *rank*
within its own list and never compares the raw numbers.
"""

from dataclasses import dataclass


def make_chunk_id(source_id: str, chunk_index: int) -> str:
    """Build the stable identifier for one chunk.

    Both indexes derive ids the same way, so the same chunk carries the same id
    in each and the two result lists can be merged by id.

    The id must be deterministic: rebuilding the index has to overwrite the
    previous entry rather than add a second copy. That is why ``source_id`` is
    the document's path *relative* to the corpus, not the absolute path, which
    would differ on every machine.

    Args:
        source_id: Document identifier, relative to the corpus root.
        chunk_index: Position of the chunk within that document.

    Returns:
        An id of the form ``"Warranty Forms.pdf::3"``.
    """
    return f"{source_id}::{chunk_index}"


@dataclass(frozen=True, slots=True)
class IndexHit:
    """One scored result from a single index.

    Attributes:
        chunk_id: Stable id, as built by :func:`make_chunk_id`.
        source_id: Document the chunk came from, relative to the corpus root.
        chunk_index: Position of the chunk within its document.
        text: The chunk body, so a caller can show the passage without a
            second lookup.
        score: Relevance according to the index that produced it. Comparable
            only against other hits from that same index.
    """

    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    score: float
