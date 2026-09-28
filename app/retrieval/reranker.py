"""Re-score a shortlist of chunks with a cross-encoder.

The embedding model is a *bi-encoder*: it turns the query and each chunk into
vectors separately, and search compares the vectors. That is fast, because
every chunk vector is computed once at index time, but the model never sees the
query and the chunk together.

A *cross-encoder* reads the pair as one input — ``[query] [SEP] [chunk]`` — and
outputs a single relevance number. Its attention can connect "how long" in the
query directly to "24 months" in the chunk, which makes it markedly more
accurate. The price is that nothing can be precomputed: every search runs the
model once per (query, chunk) pair. So it only ever sees the short list that
fusion produced, never the whole corpus.

Everything runs locally; the model is downloaded once and cached.
"""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.config import get_settings
from app.model_cache import load_cache_first

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """What retrieval needs from a reranker.

    Structural, like ``TextEmbedder``: any class with a matching ``score``
    method satisfies it, so tests pass a lightweight fake and never download a
    model.
    """

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """A sentence-transformers cross-encoder, wrapped for this project.

    Args:
        model_name: Hugging Face model id. Defaults to the configured
            ``RERANKER_MODEL_NAME``.
    """

    def __init__(self, model_name: str | None = None) -> None:
        # Deferred import, as in EmbeddingModel: sentence_transformers pulls in
        # PyTorch, and importing this module should stay cheap.
        from sentence_transformers import CrossEncoder

        self._name = get_settings().reranker_model_name if model_name is None else model_name
        self._model: CrossEncoder = load_cache_first(CrossEncoder, self._name)
        logger.info("Reranker %s reads up to %d tokens", self._name, self.max_seq_length)

    @property
    def name(self) -> str:
        """The Hugging Face model id in use."""
        return self._name

    @property
    def max_seq_length(self) -> int:
        """Longest (query + passage) input the model reads, in tokens.

        The pair shares this window. A 480-token chunk plus a short query and
        the three special tokens fits within 512; a long query pushes the end
        of the largest chunks out of view. That affects this *score* only —
        the index is untouched — so it is accepted rather than refused.
        """
        return int(self._model.max_seq_length)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score each passage's relevance to the query.

        Args:
            query: The user's query.
            passages: Chunk bodies to score.

        Returns:
            One score per passage, in the same order — not sorted; ordering is
            the caller's job. Higher is more relevant. The numbers are raw
            model outputs (logits): meaningful for ordering within one query,
            not as probabilities or as a threshold across queries.
        """
        if not passages:
            return []

        # A list comprehension building (query, passage) tuples — the input
        # shape CrossEncoder.predict expects.
        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs, show_progress_bar=False)
        # predict returns a NumPy array; convert to plain Python floats.
        return [float(value) for value in scores]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """Return the shared reranker, loading it on first call.

    Same lazy-singleton pattern as ``get_embedding_model()``.
    """
    return CrossEncoderReranker()
