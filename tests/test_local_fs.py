"""Tests for app.ingestion.connectors.local_fs."""

import ctypes
import sys
from pathlib import Path

import pytest

from app.ingestion.connectors.base import DocumentMetadata, SourceConnector
from app.ingestion.connectors.local_fs import LocalFSConnector

FILE_ATTRIBUTE_HIDDEN = 0x02


def write(path: Path, content: str = "content") -> Path:
    """Create a file and any missing parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def titles(connector: LocalFSConnector) -> set[str]:
    """Filenames the connector yields."""
    return {metadata.title for _, metadata in connector.iter_documents()}


class TestInterface:
    def test_is_a_source_connector(self) -> None:
        assert issubclass(LocalFSConnector, SourceConnector)

    def test_source_type(self, tmp_path: Path) -> None:
        assert LocalFSConnector(tmp_path).source_type == "local_fs"

    def test_iter_documents_is_lazy(self, tmp_path: Path) -> None:
        write(tmp_path / "a.txt")

        result = LocalFSConnector(tmp_path).iter_documents()

        # A generator, not a materialised list — a large scan must not load
        # every document into memory up front.
        assert not isinstance(result, list)
        assert next(iter(result))


class TestDiscovery:
    def test_finds_supported_files_recursively(self, tmp_path: Path) -> None:
        write(tmp_path / "top.txt")
        write(tmp_path / "nested" / "deep" / "buried.txt")

        assert titles(LocalFSConnector(tmp_path)) == {"top.txt", "buried.txt"}

    def test_yields_the_file_contents(self, tmp_path: Path) -> None:
        write(tmp_path / "a.txt", "hello from disk")

        (data, _), = list(LocalFSConnector(tmp_path).iter_documents())

        assert data == b"hello from disk"

    def test_missing_root_yields_nothing(self, tmp_path: Path) -> None:
        connector = LocalFSConnector(tmp_path / "does-not-exist")

        assert list(connector.iter_documents()) == []

    def test_root_that_is_a_file_yields_nothing(self, tmp_path: Path) -> None:
        target = write(tmp_path / "a.txt")

        assert list(LocalFSConnector(target).iter_documents()) == []

    def test_empty_directory_yields_nothing(self, tmp_path: Path) -> None:
        assert list(LocalFSConnector(tmp_path).iter_documents()) == []


class TestSkipRules:
    def test_unsupported_extensions_are_skipped(self, tmp_path: Path) -> None:
        write(tmp_path / "keep.txt")
        write(tmp_path / "skip.xlsx")
        write(tmp_path / "skip.png")
        write(tmp_path / "skip")  # no extension at all

        assert titles(LocalFSConnector(tmp_path)) == {"keep.txt"}

    def test_extension_matching_is_case_insensitive(self, tmp_path: Path) -> None:
        write(tmp_path / "shouty.TXT")
        write(tmp_path / "mixed.PdF")

        assert titles(LocalFSConnector(tmp_path)) == {"shouty.TXT", "mixed.PdF"}

    def test_dot_hidden_files_are_skipped(self, tmp_path: Path) -> None:
        write(tmp_path / "visible.txt")
        write(tmp_path / ".hidden.txt")

        assert titles(LocalFSConnector(tmp_path)) == {"visible.txt"}

    def test_dot_hidden_directories_are_not_descended_into(self, tmp_path: Path) -> None:
        write(tmp_path / "visible.txt")
        write(tmp_path / ".git" / "config.txt")
        write(tmp_path / ".venv" / "lib" / "module.txt")

        assert titles(LocalFSConnector(tmp_path)) == {"visible.txt"}

    def test_oversized_files_are_skipped(self, tmp_path: Path) -> None:
        write(tmp_path / "small.txt", "x" * 10)
        write(tmp_path / "large.txt", "x" * 5000)

        connector = LocalFSConnector(tmp_path, max_file_size_bytes=1000)

        assert titles(connector) == {"small.txt"}

    def test_size_limit_boundary_is_inclusive(self, tmp_path: Path) -> None:
        write(tmp_path / "exact.txt", "x" * 100)

        connector = LocalFSConnector(tmp_path, max_file_size_bytes=100)

        assert titles(connector) == {"exact.txt"}

    def test_supported_extensions_can_be_overridden(self, tmp_path: Path) -> None:
        write(tmp_path / "a.txt")
        write(tmp_path / "b.pdf")

        connector = LocalFSConnector(tmp_path, supported_extensions=frozenset({".pdf"}))

        assert titles(connector) == {"b.pdf"}


class TestHiddenRoot:
    def test_a_hidden_root_directory_is_still_scanned(self, tmp_path: Path) -> None:
        """Hidden-ness is judged relative to the root, not absolutely.

        Scanning `D:/archive/.private/docs` must work normally. Testing the
        absolute path components instead would reject every file underneath a
        dot-prefixed ancestor the user explicitly pointed us at.
        """
        root = tmp_path / ".private"
        write(root / "contract.txt")
        write(root / "sub" / "invoice.txt")

        assert titles(LocalFSConnector(root)) == {"contract.txt", "invoice.txt"}

    def test_hidden_children_of_a_hidden_root_are_still_skipped(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / ".private"
        write(root / "visible.txt")
        write(root / ".hidden.txt")

        assert titles(LocalFSConnector(root)) == {"visible.txt"}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only file attribute")
class TestWindowsHiddenAttribute:
    def test_attribute_hidden_file_is_skipped(self, tmp_path: Path) -> None:
        """A file hidden via Explorer has no leading dot."""
        visible = write(tmp_path / "visible.txt")
        hidden = write(tmp_path / "attribute-hidden.txt")

        assert ctypes.windll.kernel32.SetFileAttributesW(
            str(hidden), FILE_ATTRIBUTE_HIDDEN
        ), "could not set the hidden attribute"

        assert titles(LocalFSConnector(tmp_path)) == {visible.name}

    def test_attribute_hidden_directory_is_not_descended_into(
        self, tmp_path: Path
    ) -> None:
        write(tmp_path / "visible.txt")
        hidden_dir = tmp_path / "system"
        write(hidden_dir / "buried.txt")

        assert ctypes.windll.kernel32.SetFileAttributesW(
            str(hidden_dir), FILE_ATTRIBUTE_HIDDEN
        ), "could not set the hidden attribute"

        assert titles(LocalFSConnector(tmp_path)) == {"visible.txt"}


class TestMetadata:
    def test_metadata_fields(self, tmp_path: Path) -> None:
        target = write(tmp_path / "sub" / "Report.PDF", "abcde")

        (_, metadata), = list(
            LocalFSConnector(tmp_path, supported_extensions=frozenset({".pdf"}))
            .iter_documents()
        )

        assert isinstance(metadata, DocumentMetadata)
        assert metadata.source_type == "local_fs"
        assert metadata.title == "Report.PDF"
        assert metadata.extension == ".pdf", "extension is normalised to lowercase"
        assert metadata.size_bytes == 5
        assert Path(metadata.source_id) == target.resolve()

    def test_modified_at_is_timezone_aware(self, tmp_path: Path) -> None:
        write(tmp_path / "a.txt")

        (_, metadata), = list(LocalFSConnector(tmp_path).iter_documents())

        assert metadata.modified_at is not None
        # A naive datetime would compare incorrectly against anything else.
        assert metadata.modified_at.tzinfo is not None

    def test_source_ids_are_unique_across_the_tree(self, tmp_path: Path) -> None:
        write(tmp_path / "a" / "same-name.txt")
        write(tmp_path / "b" / "same-name.txt")

        ids = [metadata.source_id for _, metadata in LocalFSConnector(tmp_path).iter_documents()]

        assert len(ids) == 2
        assert len(set(ids)) == 2, "identically named files must not collide"
