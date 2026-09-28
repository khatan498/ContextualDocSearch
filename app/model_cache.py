"""Load a Hugging Face model from the local cache, downloading only if absent.

Shared by the embedding model and the cross-encoder reranker, so the promise
"a cached model loads without touching the network" is kept in one place.
"""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# A type variable makes the function generic: whatever `factory` builds is what
# comes back — `load_cache_first<T>(Func<..., T> factory)` in C# terms.
T = TypeVar("T")


def load_cache_first(factory: Callable[..., T], name: str, **kwargs: Any) -> T:
    """Construct a model from the local cache, falling back to a download.

    Left to its defaults, the Hugging Face client contacts huggingface.co on
    *every* load to check whether the cached files are still current — 33
    requests per load, measured, even with the model fully cached. No document
    or query text is ever in them, but they are network traffic the project
    promises not to make, and they slow startup and make it depend on the
    network being up.

    ``local_files_only=True`` forbids that. When the model genuinely is not
    cached it raises ``OSError`` immediately, still without touching the
    network, and only then is a normal (downloading) load attempted.

    Args:
        factory: The model class, e.g. ``SentenceTransformer`` or
            ``CrossEncoder``. Passed in rather than imported here so each
            caller keeps its own deferred import of the heavy library.
        name: Hugging Face model id.
        **kwargs: Further constructor arguments, passed on both attempts.
            ``**kwargs`` collects any extra keyword arguments into a dict —
            roughly C#'s ``params``, but for named arguments.

    Returns:
        The constructed model.
    """
    try:
        model = factory(name, local_files_only=True, **kwargs)
        logger.info("Loaded %s from local cache", name)
        return model
    except OSError:
        # EAFP — "easier to ask forgiveness than permission" — is the idiomatic
        # Python pattern here: try the cheap path and handle its failure,
        # rather than first checking whether the cache is complete. Checking
        # first would duplicate the library's own cache logic.
        logger.info("%s not cached; downloading (first run)", name)
        return factory(name, **kwargs)
