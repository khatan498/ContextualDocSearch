"""Turn raw document bytes into plain text.

One small loader per file type, selected through a dispatch table. Anything
unreadable raises :class:`DocumentLoadError` so a batch ingestion run can log
it and carry on instead of dying on a single corrupt file.
"""

import io
import logging
import re
from collections.abc import Callable

import docx
from docx.opc.exceptions import PackageNotFoundError
from pypdf import PdfReader
from pypdf.errors import DependencyError, PyPdfError

from app.ingestion.document import DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Raised when a document's bytes cannot be turned into text."""


# Tried in order. `utf-8-sig` goes first because it strips the byte-order mark
# Windows editors prepend; plain `utf-8` would decode that BOM into a stray
# ﻿ character leading the first chunk.
_TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252")

# Module-level compiled regexes: compiled once at import rather than on every
# call. Equivalent to a `static readonly Regex` field in C#.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)


def _load_pdf(data: bytes) -> str:
    """Extract text from a PDF.

    Args:
        data: Raw PDF bytes.

    Returns:
        Page texts joined by blank lines.

    Raises:
        DocumentLoadError: If the file is unparseable or encrypted.
    """
    # `io.BytesIO` wraps bytes in a file-like object — a MemoryStream. pypdf
    # wants something with .read()/.seek(), not a bytes object.
    try:
        reader = PdfReader(io.BytesIO(data))
    except (PyPdfError, DependencyError) as exc:
        # `raise ... from exc` chains the original exception, the same idea as
        # passing an innerException in C#. Without `from`, the original
        # traceback is lost.
        raise DocumentLoadError(f"Could not parse PDF: {exc}") from exc

    if reader.is_encrypted:
        raise DocumentLoadError(
            "PDF is encrypted; password-protected files are not supported in v1"
        )

    pages: list[str] = []
    # `enumerate(x, start=1)` yields (index, item) pairs with a 1-based counter.
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            # extract_text() can return None for image-only pages.
            pages.append(page.extract_text() or "")
        except (PyPdfError, DependencyError) as exc:
            # One damaged page shouldn't cost us the other 200.
            logger.warning("Skipping unreadable page %d: %s", page_number, exc)

    return "\n\n".join(page for page in pages if page.strip())


def _load_docx(data: bytes) -> str:
    """Extract text from a .docx file, including table contents.

    Tables are extracted deliberately: contracts, tax documents and warranties
    put the values that matter (dates, amounts, terms) inside tables, and
    paragraph-only extraction drops them silently.

    Note:
        Paragraphs are emitted before tables rather than in true document
        order. Recovering the real interleaved order means walking the
        underlying XML body, which is not worth the complexity here — chunking
        keeps both, so retrieval still finds the content.

    Args:
        data: Raw .docx bytes.

    Returns:
        Paragraph text followed by table text, separated by blank lines.

    Raises:
        DocumentLoadError: If the file is not a readable .docx package.
    """
    try:
        document = docx.Document(io.BytesIO(data))
    except (PackageNotFoundError, ValueError, KeyError) as exc:
        raise DocumentLoadError(f"Could not parse DOCX: {exc}") from exc

    # List comprehension: `[expr for x in xs if cond]` builds a list in one
    # expression. Equivalent to LINQ's .Where(...).Select(...).ToList().
    blocks: list[str] = [
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    ]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            # Keep the row on one line with a separator so the association
            # between a label and its value survives into the chunk.
            line = " | ".join(cell for cell in cells if cell)
            if line:
                blocks.append(line)

    return "\n\n".join(blocks)


def _load_txt(data: bytes) -> str:
    """Decode a plain-text file, trying several encodings.

    Args:
        data: Raw file bytes.

    Returns:
        The decoded text. This function does not raise: the final fallback
        replaces undecodable bytes rather than failing.
    """
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    # latin-1 maps all 256 byte values to characters, so it cannot raise.
    # Text may be mangled, but garbled text beats losing the document.
    logger.warning("Falling back to latin-1 with replacement characters")
    return data.decode("latin-1", errors="replace")


# The dispatch table. Adding a format is one entry here plus one function —
# no `if/elif` chain to edit. Closest C# equivalent:
# Dictionary<string, Func<byte[], string>>.
#
# It must be defined *after* the functions it references: Python executes a
# module top to bottom, and the names have to already exist.
_LOADERS: dict[str, Callable[[bytes], str]] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_txt,
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_LOADERS)


def extract_text(data: bytes, metadata: DocumentMetadata) -> str:
    """Extract plain text from a document's bytes.

    Args:
        data: Raw document bytes, as yielded by
            :meth:`~app.ingestion.connectors.local_fs.LocalFSConnector.iter_documents`.
        metadata: Document metadata; ``extension`` selects the loader.

    Returns:
        Whitespace-normalised plain text. May be empty for a document with no
        extractable text, such as a scanned image-only PDF.

    Raises:
        DocumentLoadError: If the extension is unsupported or parsing fails.
    """
    loader = _LOADERS.get(metadata.extension.lower())
    if loader is None:
        raise DocumentLoadError(
            f"Unsupported extension {metadata.extension!r} for {metadata.source_id}"
        )

    try:
        text = loader(data)
    except DocumentLoadError:
        raise  # already the right type — let it through untouched
    except Exception as exc:
        # Deliberately broad. These are third-party parsers running over
        # arbitrary user files; they can raise almost anything on malformed
        # input. At an ingestion boundary, converting every failure into one
        # typed error is what lets the caller skip-and-continue reliably.
        raise DocumentLoadError(
            f"Failed to extract text from {metadata.source_id}: {exc}"
        ) from exc

    return _normalise_whitespace(text)


def _normalise_whitespace(text: str) -> str:
    """Tidy extracted text so chunk boundaries follow content, not layout.

    PDF extraction in particular produces ragged trailing spaces and long runs
    of blank lines. Since chunking splits paragraphs on blank lines, that noise
    would otherwise decide where chunks break.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WHITESPACE.sub("", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()
