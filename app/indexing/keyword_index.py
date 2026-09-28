"""BM25 keyword index over chunks.

BM25 scores a document by the query *words* it contains. It is a refinement of
TF-IDF with two corrections that matter in practice: term frequency saturates,
so the fiftieth "warranty" adds almost nothing over the tenth; and long
documents are penalised, so they cannot win simply by containing more words.

It understands no meaning at all — which is exactly why it belongs beside the
vector index. Embeddings blur `INV-2024-88213` into every other reference
number; BM25 matches it exactly. Each covers the other's blind spot.
"""

import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.indexing.hits import IndexHit, make_chunk_id
from app.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)

# Bumped if the on-disk shape changes, so a stale file fails loudly.
_FORMAT_VERSION = 1

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Split text into the terms BM25 matches on.

    Deliberately the only tokenizer in this module. Build time and query time
    must agree exactly: if a document is indexed as ``"warranty"`` but the query
    is tokenised to ``"Warranty"``, the term simply never matches and the
    failure is silent — no error, just nothing found.

    Args:
        text: Text to tokenise.

    Returns:
        Lowercased word tokens; punctuation is dropped.
    """
    return _WORD.findall(text.lower())


@dataclass(frozen=True, slots=True)
class _Entry:
    """One indexed chunk, with its tokens precomputed."""

    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    tokens: list[str]


class KeywordIndex:
    """A BM25 index that can be written to and read back from disk.

    ``rank_bm25`` has no persistence of its own. Rather than pickling the
    object — which breaks across library versions and is unreadable when
    something goes wrong — the tokenised corpus is stored as JSON and the BM25
    model is rebuilt on load. For a corpus this size that is instant, and the
    file stays inspectable.
    """

    def __init__(self, entries: Sequence[_Entry]) -> None:
        self._entries = list(entries)
        # BM25Okapi rejects an empty corpus, so an empty index holds no model
        # and simply answers every query with no results.
        self._bm25 = (
            BM25Okapi([entry.tokens for entry in self._entries])
            if self._entries
            else None
        )

    @classmethod
    def build(cls, chunks: Sequence[Chunk]) -> "KeywordIndex":
        """Build an index from chunks.

        Args:
            chunks: Chunks to index. ``source_id`` should already be relative
                to the corpus root.

        Returns:
            A ready-to-query index.
        """
        entries = [
            _Entry(
                chunk_id=make_chunk_id(chunk.source_id, chunk.chunk_index),
                source_id=chunk.source_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                tokens=tokenize(chunk.text),
            )
            for chunk in chunks
        ]
        logger.info("Built BM25 index over %d chunks", len(entries))
        return cls(entries)

    def save(self, path: Path) -> None:
        """Write the index to a JSON file, creating parent directories.

        Args:
            path: Destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "entries": [
                {
                    "chunk_id": entry.chunk_id,
                    "source_id": entry.source_id,
                    "chunk_index": entry.chunk_index,
                    "text": entry.text,
                    "tokens": entry.tokens,
                }
                for entry in self._entries
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote BM25 index with %d chunks to %s", len(self._entries), path)

    @classmethod
    def load(cls, path: Path) -> "KeywordIndex":
        """Read an index back from disk.

        Args:
            path: File previously written by :meth:`save`.

        Returns:
            The reconstructed index.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file was written by a different format version.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"{path} was written by format version {version!r}, but this "
                f"build expects {_FORMAT_VERSION}. Rebuild the index."
            )
        return cls(
            [
                _Entry(
                    chunk_id=item["chunk_id"],
                    source_id=item["source_id"],
                    chunk_index=item["chunk_index"],
                    text=item["text"],
                    tokens=item["tokens"],
                )
                for item in payload["entries"]
            ]
        )

    def query(self, text: str, k: int = 5) -> list[IndexHit]:
        """Return the best-scoring chunks for a query.

        Args:
            text: Query text, tokenised the same way the corpus was.
            k: Maximum number of hits to return.

        Returns:
            Up to ``k`` hits, best first. Chunks scoring zero share no terms
            with the query and are omitted rather than padding the list.
        """
        if self._bm25 is None or k <= 0:
            return []

        tokens = tokenize(text)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # `sorted` on the indices rather than the scores, so the entry each
        # score belongs to is still known afterwards.
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)

        hits: list[IndexHit] = []
        for position in ranked[:k]:
            score = float(scores[position])
            if score <= 0.0:
                break  # ranked descending, so everything after is also zero
            entry = self._entries[position]
            hits.append(
                IndexHit(
                    chunk_id=entry.chunk_id,
                    source_id=entry.source_id,
                    chunk_index=entry.chunk_index,
                    text=entry.text,
                    score=score,
                )
            )
        return hits

    def chunks_per_source(self) -> dict[str, int]:
        """How many chunks each document contributed, sorted by name, ignoring case.

        Returns:
            ``{source_id: chunk_count}``. Python dicts keep insertion order
            (guaranteed since 3.7), so building it from sorted items yields a
            sorted mapping — unlike C#'s Dictionary, where order is undefined.
        """
        # Counter is a dict specialised for tallying: Counter(["a", "b", "a"])
        # == {"a": 2, "b": 1}. Here it counts entries per document.
        counts = Counter(entry.source_id for entry in self._entries)
        # Sorted ignoring case, as a person reads a list: a plain sort puts
        # every capitalised name before "individual-svcs-agrmnt.docx".
        return dict(sorted(counts.items(), key=lambda item: item[0].casefold()))

    def __len__(self) -> int:
        """Number of indexed chunks."""
        return len(self._entries)
