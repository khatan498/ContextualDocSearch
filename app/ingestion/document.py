"""The data type that describes one source document.

Lives in its own module rather than alongside the connector so that
``loaders.py`` can depend on it without depending on the filesystem layer.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Describes one source document.

    ``@dataclass`` generates ``__init__``, ``__eq__`` and ``__repr__`` from the
    field declarations below — the equivalent of a C# ``record``.
    ``frozen=True`` makes instances immutable (init-only properties), and
    ``slots=True`` drops the per-instance ``__dict__`` Python would otherwise
    allocate, which matters once a scan produces a lot of these.

    Attributes:
        source_id: Stable unique identifier for the document. Currently always
            an absolute filesystem path. Named generically rather than ``path``
            so the field stays accurate if a future personal mode adds sources
            that identify documents some other way.
        source_type: Which connector produced this. Always ``"local_fs"`` while
            the app is demo-only.
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
