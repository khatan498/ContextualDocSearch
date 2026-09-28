"""Open the indexes that scripts/build_index.py wrote, and check they are usable.

Everything that reads the indexes — search, the ``--verify`` diagnostic, and
later the API — goes through :func:`open_built_indexes`. Each way the indexes
can be unusable then produces the same kind of error, carrying a message
written for a person, instead of a Chroma or JSON traceback from wherever the
problem first happened to surface.

Every check here runs before any model is loaded, so a stale index is reported
in well under a second rather than after a seven-second model load.
"""

from app.config import get_settings
from app.indexing.keyword_index import KeywordIndex
from app.indexing.vector_store import VectorStore

_REBUILD = "Run python scripts/build_index.py to rebuild it."


class IndexUnavailableError(RuntimeError):
    """The indexes are missing, unreadable, or do not match the configuration.

    A deliberate refusal, like ``PersonalModeUnavailableError``: callers print
    its message rather than a traceback. Rebuilding the index fixes every case.
    """


def open_built_indexes() -> tuple[KeywordIndex, VectorStore]:
    """Open both indexes, refusing any that search could not use correctly.

    Returns:
        The keyword index and the vector store, in that order. Returning a
        tuple and unpacking it at the call site — ``keyword, store =
        open_built_indexes()`` — is the idiomatic Python alternative to an out
        parameter or a small result class.

    Raises:
        IndexUnavailableError: If the index has not been built; if the keyword
            index file cannot be read; if the vector index is empty; if the
            vectors came from a different embedding model than the configured
            one; or if the two indexes hold different numbers of chunks.
    """
    settings = get_settings()
    keyword_path = settings.keyword_index_path

    # Checked before opening the vector store, which would otherwise create an
    # empty database directory as a side effect of looking.
    if not keyword_path.exists():
        raise IndexUnavailableError(
            f"No search index at {settings.vector_store_path}. "
            f"Run python scripts/build_index.py first."
        )

    try:
        keyword_index = KeywordIndex.load(keyword_path)
    # ValueError covers both a file from another format version and malformed
    # JSON (json.JSONDecodeError is a ValueError subclass); KeyError covers
    # valid JSON missing an expected field.
    except (ValueError, KeyError) as exc:
        raise IndexUnavailableError(
            f"The keyword index at {keyword_path} cannot be used: {exc}\n{_REBUILD}"
        ) from exc

    store = VectorStore.open(settings.vector_store_path)
    stored = store.count()
    if stored == 0:
        raise IndexUnavailableError(f"The vector index is empty. {_REBUILD}")

    # A query vector is only comparable with vectors from the same model. A
    # different width fails inside Chroma; a different model of the *same*
    # width fails silently, returning confident nonsense. Both are caught here.
    built_with = store.embedding_model
    configured = settings.embedding_model_name
    if built_with != configured:
        origin = (
            "by an older version of this project, which did not record its model"
            if built_with is None
            else f"with {built_with}"
        )
        raise IndexUnavailableError(
            f"The index was built {origin}, but EMBEDDING_MODEL_NAME is "
            f"{configured}. Queries would be compared against vectors from a "
            f"different model. {_REBUILD}"
        )

    # The builder writes the vectors first and the keyword file second, so an
    # interrupted build can leave the two describing different corpora.
    if len(keyword_index) != stored:
        raise IndexUnavailableError(
            f"The keyword index holds {len(keyword_index)} chunks but the vector "
            f"index holds {stored}, so a build was probably interrupted. {_REBUILD}"
        )

    return keyword_index, store
