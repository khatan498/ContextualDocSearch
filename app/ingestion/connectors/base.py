"""The contract every document source must satisfy.

This module is the architectural seam required by CLAUDE.md: local disk,
Google Drive, and anything added later all look identical to the rest of the
pipeline. No source-specific concept (a filesystem path, a Drive file id, an
API client) may leak past this interface.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

# Note on imports: `Iterator` comes from `collections.abc`, not `typing`.
# The `typing` versions of the container types have been deprecated since
# Python 3.9 — you will see the old `typing.Iterator` in a lot of tutorials.


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Describes one source document, independent of where it came from.

    ``@dataclass`` generates ``__init__``, ``__eq__`` and ``__repr__`` from the
    field declarations below — the equivalent of a C# ``record``.
    ``frozen=True`` makes instances immutable (init-only properties), and
    ``slots=True`` drops the per-instance ``__dict__`` Python would otherwise
    allocate, which matters once a scan produces tens of thousands of these.

    Attributes:
        source_id: Stable unique identifier for the document. Deliberately not
            named ``path``: local files use an absolute path, but Google Drive
            has opaque file ids and no path. Naming it ``path`` would bake a
            filesystem assumption into the shared interface.
        source_type: Which connector produced this, e.g. ``"local_fs"``.
        title: Human-readable name, normally the filename.
        extension: Lowercased file extension including the dot, e.g. ``".pdf"``.
        size_bytes: Size of the raw document content.
        modified_at: Last-modified timestamp, if the source exposes one.
    """

    source_id: str
    source_type: str
    title: str
    extension: str
    size_bytes: int
    modified_at: datetime | None = None


class SourceConnector(ABC):
    """Abstract base class for anything that can supply documents.

    Python has no ``interface`` keyword. The equivalent is an abstract base
    class: subclassing ``ABC`` and marking methods ``@abstractmethod`` makes
    Python raise ``TypeError`` if you try to *construct* a subclass that hasn't
    implemented them all. That check happens at instantiation time rather than
    compile time, but it does exist.
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Short identifier for this source, stored on every document's metadata.

        Returns:
            A stable slug such as ``"local_fs"`` or ``"google_drive"``.
        """

    @abstractmethod
    def iter_documents(self) -> Iterator[tuple[bytes, DocumentMetadata]]:
        """Yield each available document as raw bytes plus its metadata.

        Implementations must be lazy generators rather than returning a list:
        a large scan should never hold every document in memory at once.

        Decoding bytes into text is deliberately *not* done here — that is
        ``loaders.extract_text``'s job, so connectors stay focused on fetching
        and every source gets identical parsing behaviour.

        Yields:
            Tuples of ``(raw_bytes, metadata)``, one per document.
        """
