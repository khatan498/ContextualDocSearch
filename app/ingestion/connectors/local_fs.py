"""Local filesystem source connector.

Walks a directory tree and yields the raw bytes of every file worth indexing,
skipping anything hidden, unsupported, or oversized.
"""

import logging
import os
import stat as stat_module
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.ingestion.connectors.base import DocumentMetadata, SourceConnector

# One logger per module, named after the module. Configuring logging is the
# application's job (scripts/, api/) — libraries just emit.
logger = logging.getLogger(__name__)

# `frozenset` is an immutable HashSet<string>: O(1) membership, and being
# immutable it is safe to share as a module-level default.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".txt"})


class LocalFSConnector(SourceConnector):
    """Yields documents found under a local directory tree.

    Args:
        root: Directory to scan recursively.
        max_file_size_bytes: Skip files larger than this. Defaults to the
            configured ``MAX_FILE_SIZE_BYTES``.
        supported_extensions: Lowercased extensions to accept, including the
            leading dot. Defaults to ``SUPPORTED_EXTENSIONS``.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_file_size_bytes: int | None = None,
        supported_extensions: frozenset[str] | None = None,
    ) -> None:
        # Everything after `*` is keyword-only: callers must write
        # `LocalFSConnector(root, max_file_size_bytes=1024)`. This stops
        # anonymous positional numbers/booleans appearing at call sites.
        settings = get_settings()

        self._root = Path(root).expanduser().resolve()
        # `if x is not None` rather than `or`: a caller passing 0 is a
        # legitimate (if useless) limit, and `0 or default` would silently
        # replace it with the default. This trips people up constantly.
        self._max_file_size_bytes = (
            max_file_size_bytes
            if max_file_size_bytes is not None
            else settings.max_file_size_bytes
        )
        self._supported_extensions = (
            supported_extensions
            if supported_extensions is not None
            else SUPPORTED_EXTENSIONS
        )

    @property
    def source_type(self) -> str:
        """See :class:`SourceConnector`."""
        return "local_fs"

    @property
    def root(self) -> Path:
        """The resolved absolute directory being scanned."""
        return self._root

    def iter_documents(self) -> Iterator[tuple[bytes, DocumentMetadata]]:
        """Yield ``(raw_bytes, metadata)`` for each indexable file under root.

        A file that cannot be read (locked, permission denied) is logged and
        skipped rather than aborting the scan.

        Yields:
            Tuples of the file's bytes and its :class:`DocumentMetadata`.
        """
        for path, file_stat in self._iter_candidate_files():
            try:
                data = path.read_bytes()
            except OSError as exc:
                logger.warning("Skipping unreadable file %s: %s", path, exc)
                continue

            yield data, self._build_metadata(path, file_stat, len(data))

    def _iter_candidate_files(self) -> Iterator[tuple[Path, os.stat_result]]:
        """Walk the tree and yield files that pass every skip rule.

        Yields the ``stat`` result alongside the path so callers get the
        modification time without a second syscall.
        """
        if not self._root.is_dir():
            logger.warning("Scan root is not a directory: %s", self._root)
            return  # a bare `return` in a generator just ends iteration

        # `followlinks=False` is os.walk's default; stated explicitly because it
        # is what stops a symlink cycle turning into an infinite scan.
        for dirpath, dirnames, filenames in os.walk(self._root, followlinks=False):
            current_dir = Path(dirpath)

            # Prune hidden directories so we never descend into .git/, .venv/
            # and friends at all.
            #
            # The `[:]` is essential. Slice assignment mutates the list os.walk
            # is holding; os.walk re-reads it after this loop body to decide
            # where to recurse. Plain `dirnames = [...]` would rebind a local
            # name and silently do nothing.
            dirnames[:] = [
                name
                for name in dirnames
                if not self._is_hidden_name(name)
                and not self._has_hidden_attribute(current_dir / name)
            ]

            for filename in filenames:
                path = current_dir / filename

                # Checked in cost order: the two free string checks first, then
                # a single stat() that answers both remaining questions.
                #
                # Note we test `filename`, not the absolute path. Directory
                # pruning above already excluded hidden ancestors *inside* the
                # tree, and ignoring everything above the root is deliberate: a
                # root of `D:\...\.private\docs` must still scan normally.
                if self._is_hidden_name(filename):
                    continue

                if path.suffix.lower() not in self._supported_extensions:
                    continue

                try:
                    file_stat = path.stat()
                except OSError as exc:
                    logger.warning("Skipping unstattable file %s: %s", path, exc)
                    continue

                if self._is_hidden_stat(file_stat):
                    continue

                if file_stat.st_size > self._max_file_size_bytes:
                    logger.info(
                        "Skipping oversized file %s (%d bytes > %d limit)",
                        path,
                        file_stat.st_size,
                        self._max_file_size_bytes,
                    )
                    continue

                yield path, file_stat

    def _build_metadata(
        self, path: Path, file_stat: os.stat_result, size_bytes: int
    ) -> DocumentMetadata:
        """Construct metadata for one discovered file."""
        return DocumentMetadata(
            source_id=str(path),
            source_type=self.source_type,
            title=path.name,
            extension=path.suffix.lower(),
            size_bytes=size_bytes,
            modified_at=datetime.fromtimestamp(file_stat.st_mtime, tz=UTC),
        )

    @staticmethod
    def _is_hidden_name(name: str) -> bool:
        """Whether a bare file or directory name is dot-prefixed."""
        return name.startswith(".")

    @staticmethod
    def _is_hidden_stat(file_stat: os.stat_result) -> bool:
        """Whether a stat result carries the Windows hidden attribute.

        A file hidden through Explorer does not start with a dot, so the
        dot-prefix check alone misses it on Windows.

        Both ``getattr`` calls supply a default of 0 so this is a harmless
        no-op on Linux and macOS, where neither the stat field nor the constant
        exists. ``0 & 0`` is falsey, so nothing is ever hidden there.
        """
        attributes = getattr(file_stat, "st_file_attributes", 0)
        hidden_flag = getattr(stat_module, "FILE_ATTRIBUTE_HIDDEN", 0)
        return bool(attributes & hidden_flag)

    @classmethod
    def _has_hidden_attribute(cls, path: Path) -> bool:
        """Whether a directory carries the Windows hidden attribute."""
        try:
            return cls._is_hidden_stat(path.stat())
        except OSError:
            # Unreadable directory: let the walk skip it quietly.
            return True
