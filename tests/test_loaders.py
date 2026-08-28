"""Tests for app.ingestion.loaders."""

import io

import docx
import pytest
from pypdf import PdfWriter

from app.ingestion.document import DocumentMetadata
from app.ingestion.loaders import DocumentLoadError, extract_text


def make_metadata(extension: str, source_id: str = "doc") -> DocumentMetadata:
    """Minimal metadata; only ``extension`` affects loader selection."""
    return DocumentMetadata(
        source_id=source_id,
        source_type="test",
        title=f"sample{extension}",
        extension=extension,
        size_bytes=0,
    )


def build_pdf(body: str) -> bytes:
    """Construct a tiny single-page PDF containing ``body`` as real text.

    Written by hand rather than committed as a binary fixture: nothing in the
    dependency set can *generate* a text-bearing PDF (that needs reportlab),
    and a ~600 byte builder is preferable to adding a dependency or checking in
    an opaque blob.
    """
    stream = f"BT /F1 12 Tf 20 100 Td ({body}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_position = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(out)


def build_docx(paragraphs: list[str], table_rows: list[tuple[str, str]]) -> bytes:
    """Construct a .docx in memory with the given paragraphs and a 2-column table."""
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)

    if table_rows:
        table = document.add_table(rows=len(table_rows), cols=2)
        for row_index, (left, right) in enumerate(table_rows):
            table.cell(row_index, 0).text = left
            table.cell(row_index, 1).text = right

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class TestPlainText:
    def test_utf8(self) -> None:
        data = "Hello world. Ünicode survives.".encode("utf-8")

        assert extract_text(data, make_metadata(".txt")) == "Hello world. Ünicode survives."

    def test_utf8_with_bom_strips_the_marker(self) -> None:
        data = "Hello world.".encode("utf-8-sig")

        result = extract_text(data, make_metadata(".txt"))

        assert result == "Hello world."
        assert "\ufeff" not in result, "BOM must not leak into the first chunk"

    def test_cp1252_fallback(self) -> None:
        # Valid cp1252, invalid UTF-8.
        data = b"Caf\xe9 receipt"

        assert extract_text(data, make_metadata(".txt")) == "Café receipt"

    def test_undecodable_bytes_never_raise(self) -> None:
        # 0x81 is undefined in cp1252, so only the latin-1 fallback can handle it.
        data = b"\x81\x81 trailing text"

        result = extract_text(data, make_metadata(".txt"))

        assert "trailing text" in result


class TestWhitespaceNormalisation:
    def test_runs_of_blank_lines_are_collapsed(self) -> None:
        data = b"First para.\n\n\n\n\nSecond para."

        assert extract_text(data, make_metadata(".txt")) == "First para.\n\nSecond para."

    def test_trailing_spaces_and_crlf_are_cleaned(self) -> None:
        data = b"Line one.   \r\n\r\nLine two.\t\r\n"

        assert extract_text(data, make_metadata(".txt")) == "Line one.\n\nLine two."

    def test_paragraph_breaks_are_preserved(self) -> None:
        """Chunking splits on blank lines, so they must survive extraction."""
        data = b"One.\n\nTwo.\n\nThree."

        assert extract_text(data, make_metadata(".txt")).count("\n\n") == 2


class TestDocx:
    def test_paragraphs_are_extracted(self) -> None:
        data = build_docx(["First paragraph.", "Second paragraph."], [])

        result = extract_text(data, make_metadata(".docx"))

        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_table_cells_are_extracted(self) -> None:
        """Contracts and tax documents keep the important values in tables."""
        data = build_docx(
            ["Warranty terms follow."],
            [("Coverage period", "24 months"), ("Excess", "$150")],
        )

        result = extract_text(data, make_metadata(".docx"))

        assert "24 months" in result
        assert "Coverage period" in result
        assert "$150" in result

    def test_table_row_stays_on_one_line(self) -> None:
        """A label must stay attached to its value, or retrieval loses the link."""
        data = build_docx([], [("Coverage period", "24 months")])

        result = extract_text(data, make_metadata(".docx"))

        assert "Coverage period | 24 months" in result

    def test_empty_paragraphs_are_dropped(self) -> None:
        data = build_docx(["Real text.", "", "   ", "More text."], [])

        result = extract_text(data, make_metadata(".docx"))

        assert result == "Real text.\n\nMore text."

    def test_corrupt_docx_raises(self) -> None:
        with pytest.raises(DocumentLoadError):
            extract_text(b"definitely not a docx", make_metadata(".docx"))


class TestPdf:
    def test_text_is_extracted(self) -> None:
        data = build_pdf("Hello chunk world")

        assert "Hello chunk world" in extract_text(data, make_metadata(".pdf"))

    def test_corrupt_pdf_raises(self) -> None:
        with pytest.raises(DocumentLoadError):
            extract_text(b"this is not a pdf at all", make_metadata(".pdf"))

    def test_empty_bytes_raise(self) -> None:
        with pytest.raises(DocumentLoadError):
            extract_text(b"", make_metadata(".pdf"))

    def test_encrypted_pdf_raises_rather_than_returning_nothing(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt("secret")
        buffer = io.BytesIO()
        writer.write(buffer)

        with pytest.raises(DocumentLoadError, match="encrypted"):
            extract_text(buffer.getvalue(), make_metadata(".pdf"))

    def test_image_only_pdf_yields_empty_text_without_raising(self) -> None:
        """A scanned page has no extractable text; that is not an error."""
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)

        assert extract_text(buffer.getvalue(), make_metadata(".pdf")) == ""


class TestDispatch:
    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(DocumentLoadError, match="Unsupported extension"):
            extract_text(b"content", make_metadata(".xlsx"))

    def test_extension_matching_is_case_insensitive(self) -> None:
        data = "Shouty filename.".encode("utf-8")

        assert extract_text(data, make_metadata(".TXT")) == "Shouty filename."

    def test_error_message_identifies_the_document(self) -> None:
        metadata = make_metadata(".xlsx", source_id="D:/docs/budget.xlsx")

        with pytest.raises(DocumentLoadError, match="budget.xlsx"):
            extract_text(b"content", metadata)

    def test_load_error_chains_the_original_exception(self) -> None:
        """`raise ... from exc` must preserve the underlying cause."""
        with pytest.raises(DocumentLoadError) as exc_info:
            extract_text(b"not a pdf", make_metadata(".pdf"))

        assert exc_info.value.__cause__ is not None
