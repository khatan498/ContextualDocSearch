"""Turn text into vectors with a local sentence-transformers model.

An *embedding* is a fixed-length list of floats that places a piece of text in a
space where distance means difference in meaning. Two passages about warranty
periods land near each other even with no words in common, which is what lets a
search for "how long is the warranty" find "coverage period: 24 months".

Everything here runs locally. The model is downloaded once from Hugging Face and
cached; no text ever leaves the machine.
"""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.config import get_settings
from app.model_cache import load_cache_first

if TYPE_CHECKING:
    # Imported for type checking only — this branch never runs. It keeps the
    # annotation below honest without paying the import cost at runtime.
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# A single embedding. Naming it makes the signatures below readable.
Vector = list[float]


class ChunkWindowTooLargeError(RuntimeError):
    """Raised when the configured chunk size exceeds the model's input limit.

    Embedding models do not reject oversized input — they silently truncate it
    and return a vector for the part that fitted. The lost text simply never
    becomes searchable, with nothing in any log to say so. This error converts
    that silent failure into a loud one.
    """


@runtime_checkable
class TextEmbedder(Protocol):
    """What the indexing pipeline needs from an embedding model.

    ``Protocol`` is Python's structural typing: any class with these three
    methods satisfies it, with no base class and no ``implements`` clause. The
    closest C# analogue is an interface, except conformance is judged by shape
    rather than declaration — so a test double satisfies it without importing
    anything from here.

    That is exactly why this exists: :class:`EmbeddingModel` downloads ~440 MB
    and takes seconds to load, so tests inject a lightweight fake instead and
    the suite stays offline.

    (``@runtime_checkable`` permits ``isinstance`` checks, but note it only
    verifies that the method *names* exist — not their signatures.)
    """

    def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...

    def count_tokens(self, text: str) -> int: ...


class EmbeddingModel:
    """A sentence-transformers model, wrapped for this project's needs.

    Args:
        model_name: Hugging Face model id. Defaults to the configured
            ``EMBEDDING_MODEL_NAME``.
        max_chunk_tokens: Chunk ceiling to validate against the model's limit.
            Defaults to the configured ``CHUNK_MAX_TOKENS``.

    Raises:
        ChunkWindowTooLargeError: If the chunk ceiling exceeds what the model
            can actually read.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        max_chunk_tokens: int | None = None,
    ) -> None:
        # Deferred import: `sentence_transformers` pulls in PyTorch, which costs
        # seconds. Importing it here rather than at module scope keeps
        # `import app.indexing.embeddings` cheap, so tests that only need the
        # Protocol never pay for it. C# has no direct equivalent — assemblies
        # load lazily on first use anyway.
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self._name = (
            settings.embedding_model_name if model_name is None else model_name
        )
        self._batch_size = settings.embedding_batch_size

        # Cache first: no network traffic at all once the model is downloaded.
        self._model: SentenceTransformer = load_cache_first(SentenceTransformer, self._name)
        self._silence_length_warning()

        ceiling = (
            settings.chunk_max_tokens if max_chunk_tokens is None else max_chunk_tokens
        )
        self._assert_chunks_fit(ceiling)

    @staticmethod
    def _silence_length_warning() -> None:
        """Mute the tokenizer's "sequence longer than the maximum" warning.

        Chunking calls :meth:`count_tokens` on whole paragraphs precisely to
        find the oversized ones and split them, so that warning fires routinely
        during a normal build. Its text — "running this sequence through the
        model will result in indexing errors" — describes something that never
        happens here: nothing over the limit is ever embedded, which
        :meth:`_assert_chunks_fit` and the chunk packer between them guarantee.

        Configuring logging is normally the application's job rather than a
        library's. This is a deliberate exception: the message is actively
        misleading about this code, and only this module knows that.
        """
        logging.getLogger("transformers.tokenization_utils_base").setLevel(
            logging.ERROR
        )

    def _assert_chunks_fit(self, max_chunk_tokens: int) -> None:
        """Refuse to run if chunks can outgrow what the model will read."""
        limit = self.max_seq_length
        if max_chunk_tokens > limit:
            raise ChunkWindowTooLargeError(
                f"CHUNK_MAX_TOKENS is {max_chunk_tokens} but {self._name} reads "
                f"at most {limit} tokens and truncates the rest silently. "
                f"Lower CHUNK_MAX_TOKENS to {limit} or below, or choose a model "
                f"with a longer input limit."
            )
        logger.info(
            "Chunk ceiling %d fits %s (limit %d)", max_chunk_tokens, self._name, limit
        )

    @property
    def name(self) -> str:
        """The Hugging Face model id in use."""
        return self._name

    @property
    def max_seq_length(self) -> int:
        """Longest input the model reads, in tokens. Beyond this it truncates."""
        return int(self._model.max_seq_length)

    @property
    def dimensions(self) -> int:
        """Length of every vector this model produces.

        The vector store is fixed to this width, so changing model means
        rebuilding the index from scratch.
        """
        return int(self._model.get_embedding_dimension())

    def count_tokens(self, text: str) -> int:
        """Count tokens the way the model itself counts them.

        Passed to ``chunk_text(count_tokens=...)`` at index time so chunk
        boundaries are measured in the model's own units. The word-based
        ``estimate_tokens`` default under-counts badly on tables, dates and
        reference numbers — precisely the content a chunk must not lose.

        Args:
            text: Text to measure.

        Returns:
            Token count excluding the special tokens added at encode time.
            Chunking allows headroom for those: a 480-token chunk reaches 482
            once the model wraps it, against a 512 limit.
        """
        # `tokenize` rather than `encode(add_special_tokens=False)`: identical
        # counts, and it skips converting tokens to ids, which is wasted work
        # when only the length is wanted.
        return len(self._model.tokenizer.tokenize(text))

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed passages for storage in the index.

        Args:
            texts: Chunk bodies.

        Returns:
            One vector per input, in the same order.
        """
        if not texts:
            return []

        # `encode_document` rather than plain `encode`: retrieval is asymmetric.
        # A short question and a long passage are different shapes of text, and
        # bge models are trained with a prefix on the query side to account for
        # it. These methods apply the right treatment to each, so the prefix
        # never has to be hand-written here.
        vectors = self._model.encode_document(
            texts,
            batch_size=self._batch_size,
            # Scale every vector to length 1, which makes cosine similarity and
            # the dot product the same operation — cheaper to compare, and
            # scores land in a predictable range.
            normalize_embeddings=True,
        )
        # sentence-transformers returns a NumPy array; `.tolist()` converts it
        # to plain Python floats, which is what Chroma and JSON both want.
        return [list(map(float, row)) for row in vectors.tolist()]

    def embed_query(self, text: str) -> Vector:
        """Embed a search query.

        Args:
            text: The user's query.

        Returns:
            A single vector, directly comparable with document vectors.
        """
        vector = self._model.encode_query(text, normalize_embeddings=True)
        return [float(value) for value in vector.tolist()]


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Return the shared embedding model, loading it on first call.

    Same lazy-singleton pattern as ``get_settings()`` in ``app.config``: the
    model is large and slow to load, so it is built once and reused. Tests that
    need a different instance should construct :class:`EmbeddingModel` directly
    or call ``get_embedding_model.cache_clear()``.
    """
    return EmbeddingModel()
