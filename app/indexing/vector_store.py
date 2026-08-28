"""Chroma-backed vector index over chunks.

Stores one vector per chunk and answers "which chunks sit nearest this query
vector". Nearness is cosine similarity, so it matches on meaning rather than on
shared words — the half of retrieval that BM25 cannot do.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.indexing.hits import IndexHit, make_chunk_id
from app.indexing.embeddings import Vector
from app.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)

# Chroma requires 3-512 characters from [a-zA-Z0-9._-], starting and ending
# alphanumeric — a one-character name is rejected at creation.
COLLECTION_NAME = "chunks"

# Turn off Chroma's usage telemetry. The project promises nothing leaves the
# machine, and it also keeps the test suite from touching the network.
_CHROMA_SETTINGS = ChromaSettings(anonymized_telemetry=False)


class VectorStore:
    """A Chroma collection holding one vector per chunk.

    Construct via :meth:`open` for the on-disk store, or :meth:`in_memory` for
    tests.
    """

    def __init__(self, client: chromadb.api.ClientAPI) -> None:
        self._client = client
        self._collection = self._get_or_create()

    def _get_or_create(self) -> chromadb.api.models.Collection.Collection:
        # `hnsw:space` picks the distance metric and can only be set when the
        # collection is created. Cosine pairs with the normalised vectors the
        # embedding model produces.
        return self._client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    @classmethod
    def open(cls, path: Path | None = None) -> "VectorStore":
        """Open (or create) the persistent store on disk.

        Args:
            path: Directory to store the database in. Defaults to the
                configured ``VECTOR_STORE_PATH``.

        Returns:
            A store backed by files under ``path``.
        """
        location = get_settings().vector_store_path if path is None else path
        location.mkdir(parents=True, exist_ok=True)
        logger.info("Opening vector store at %s", location)
        return cls(
            chromadb.PersistentClient(path=str(location), settings=_CHROMA_SETTINGS)
        )

    @classmethod
    def in_memory(cls) -> "VectorStore":
        """Create a throwaway store that never touches disk. For tests.

        Chroma caches its in-memory system, so repeated ``EphemeralClient()``
        calls within one process hand back the *same* underlying store — a
        second "fresh" client would otherwise still see the first one's data.
        Dropping the collection here gives each new instance a clean slate.

        Because that store is shared, **only one in-memory VectorStore may be
        alive at a time**: constructing a second one drops the collection the
        first is holding, and the first then raises ``NotFoundError``. Tests
        take one per test, which is the normal pattern. If you genuinely need
        two side by side, use :meth:`open` with separate ``tmp_path``
        directories instead.
        """
        client = chromadb.EphemeralClient(settings=_CHROMA_SETTINGS)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - absent on the first call, which is fine
            logger.debug("No existing in-memory collection to clear")
        return cls(client)

    def replace_all(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        """Replace the entire collection with these chunks.

        Dropping and recreating rather than upserting means a rebuild after
        documents are removed or re-chunked cannot leave orphaned vectors
        behind, silently matching text that no longer exists.

        Args:
            chunks: Chunks to store; ``source_id`` should be relative to the
                corpus root.
            vectors: One vector per chunk, in the same order.

        Raises:
            ValueError: If the counts do not match.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(vectors)} vectors; "
                f"every chunk needs exactly one"
            )

        # delete_collection raises if it does not exist yet, which is normal on
        # a first run — nothing to clean up in that case.
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - chroma raises varied types here
            logger.debug("No existing collection to delete")
        self._collection = self._get_or_create()

        if not chunks:
            logger.info("No chunks to index")
            return

        self._collection.add(
            ids=[make_chunk_id(c.source_id, c.chunk_index) for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            # Chroma metadata values must be str/int/float/bool — no nesting.
            metadatas=[
                {"source_id": c.source_id, "chunk_index": c.chunk_index}
                for c in chunks
            ],
        )
        logger.info("Indexed %d vectors into %s", len(chunks), COLLECTION_NAME)

    def query(self, vector: Vector, k: int = 5) -> list[IndexHit]:
        """Return the chunks nearest a query vector.

        Args:
            vector: Query embedding, produced by ``embed_query``.
            k: Maximum number of hits.

        Returns:
            Up to ``k`` hits, most similar first.
        """
        if k <= 0 or self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[list(vector)], n_results=min(k, self.count())
        )

        # Chroma accepts a batch of queries, so every field comes back wrapped
        # one level deeper than you want. Unwrap the single query here so no
        # caller has to know.
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            IndexHit(
                chunk_id=chunk_id,
                source_id=str(metadata["source_id"]),
                chunk_index=int(metadata["chunk_index"]),
                text=document,
                # Chroma reports cosine *distance*: 0.0 is identical. Flip it
                # once, here, so nothing downstream has to remember which
                # direction is better.
                score=1.0 - float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]

    def count(self) -> int:
        """Number of vectors currently stored."""
        return int(self._collection.count())
