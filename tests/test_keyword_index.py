"""Tests for app.indexing.keyword_index."""

import json
from pathlib import Path

import pytest

from app.indexing.keyword_index import KeywordIndex, tokenize
from app.ingestion.chunking import Chunk


def make_chunk(text: str, source_id: str = "doc.pdf", chunk_index: int = 0) -> Chunk:
    """A Chunk carrying the given text; offsets are irrelevant to BM25."""
    return Chunk(
        text=text,
        source_id=source_id,
        chunk_index=chunk_index,
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
    )


CORPUS = [
    make_chunk("The coverage period is 24 months from the date of purchase.", "warranty.pdf", 0),
    make_chunk("Invoice INV-2024-88213 is due on 2024-11-30 with Net 30 terms.", "invoice.docx", 0),
    make_chunk("Tenant shall not sublet the premises without written consent.", "sublease.pdf", 0),
]


class TestTokenize:
    def test_lowercases(self) -> None:
        assert tokenize("Warranty PERIOD") == ["warranty", "period"]

    def test_drops_punctuation(self) -> None:
        assert tokenize("months, from: the date.") == ["months", "from", "the", "date"]

    def test_splits_reference_numbers_consistently(self) -> None:
        # Whatever it does to INV-2024-88213, it must do identically at build
        # and query time — that parity is what the next test pins down.
        assert tokenize("INV-2024-88213") == ["inv", "2024", "88213"]

    def test_empty_text_is_empty(self) -> None:
        assert tokenize("") == []
        assert tokenize("   !!!  ") == []


class TestBuildAndQuery:
    def test_length_matches_the_corpus(self) -> None:
        assert len(KeywordIndex.build(CORPUS)) == 3

    def test_finds_the_chunk_containing_the_terms(self) -> None:
        index = KeywordIndex.build(CORPUS)

        hits = index.query("coverage period")

        assert hits
        assert hits[0].source_id == "warranty.pdf"

    def test_matches_an_exact_reference_number(self) -> None:
        # The reason BM25 sits alongside the vector index: embeddings blur one
        # invoice number into every other one.
        index = KeywordIndex.build(CORPUS)

        hits = index.query("INV-2024-88213")

        assert hits
        assert hits[0].source_id == "invoice.docx"

    def test_respects_k(self) -> None:
        index = KeywordIndex.build(CORPUS)

        assert len(index.query("the", k=2)) <= 2

    def test_non_positive_k_returns_nothing(self) -> None:
        assert KeywordIndex.build(CORPUS).query("coverage", k=0) == []

    def test_zero_scoring_chunks_are_omitted(self) -> None:
        index = KeywordIndex.build(CORPUS)

        hits = index.query("coverage period", k=10)

        # Only the warranty chunk shares terms; padding the list to k with
        # unrelated chunks would be worse than returning fewer.
        assert all(hit.score > 0 for hit in hits)
        assert len(hits) < len(CORPUS)

    def test_query_with_no_known_terms_returns_nothing(self) -> None:
        assert KeywordIndex.build(CORPUS).query("zzzz qqqq") == []

    def test_empty_query_returns_nothing(self) -> None:
        assert KeywordIndex.build(CORPUS).query("   ") == []

    def test_hits_are_ordered_best_first(self) -> None:
        index = KeywordIndex.build(CORPUS)

        scores = [hit.score for hit in index.query("the", k=10)]

        assert scores == sorted(scores, reverse=True)

    def test_hit_carries_chunk_identity_and_text(self) -> None:
        hit = KeywordIndex.build(CORPUS).query("coverage period")[0]

        assert hit.chunk_id == "warranty.pdf::0"
        assert hit.chunk_index == 0
        assert "coverage period" in hit.text


class TestEmptyIndex:
    def test_builds_without_error(self) -> None:
        # BM25Okapi itself rejects an empty corpus, so this is a real edge.
        assert len(KeywordIndex.build([])) == 0

    def test_query_returns_nothing(self) -> None:
        assert KeywordIndex.build([]).query("anything") == []


class TestPersistence:
    def test_round_trip_preserves_results(self, tmp_path: Path) -> None:
        path = tmp_path / "bm25_index.json"
        original = KeywordIndex.build(CORPUS)
        original.save(path)

        reloaded = KeywordIndex.load(path)

        assert len(reloaded) == len(original)
        assert [h.chunk_id for h in reloaded.query("coverage period")] == [
            h.chunk_id for h in original.query("coverage period")
        ]

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "bm25_index.json"

        KeywordIndex.build(CORPUS).save(path)

        assert path.is_file()

    def test_file_is_readable_json(self, tmp_path: Path) -> None:
        # Chosen over pickle so a human can inspect it when a query misbehaves.
        path = tmp_path / "bm25_index.json"
        KeywordIndex.build(CORPUS).save(path)

        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["version"] == 1
        assert len(payload["entries"]) == 3
        assert "tokens" in payload["entries"][0]

    def test_stale_format_version_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bm25_index.json"
        path.write_text(json.dumps({"version": 99, "entries": []}), encoding="utf-8")

        with pytest.raises(ValueError, match="Rebuild the index"):
            KeywordIndex.load(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            KeywordIndex.load(tmp_path / "absent.json")

    def test_tokenisation_survives_the_round_trip(self, tmp_path: Path) -> None:
        # The silent-failure mode this guards: tokens stored one way and the
        # query tokenised another finds nothing, with no error anywhere.
        path = tmp_path / "bm25_index.json"
        KeywordIndex.build(CORPUS).save(path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        stored = payload["entries"][0]["tokens"]

        assert stored == tokenize(CORPUS[0].text)


class TestChunksPerSource:
    def test_counts_chunks_per_document_sorted_by_name(self) -> None:
        chunks = [
            make_chunk("warranty terms", "warranty.pdf", 0),
            make_chunk("sublease terms", "sublease.pdf", 0),
            make_chunk("more warranty", "warranty.pdf", 1),
            make_chunk("even more warranty", "warranty.pdf", 2),
        ]

        counts = KeywordIndex.build(chunks).chunks_per_source()

        assert counts == {"sublease.pdf": 1, "warranty.pdf": 3}
        assert list(counts) == ["sublease.pdf", "warranty.pdf"]

    def test_order_ignores_case(self) -> None:
        # Found in the real UI: "individual-svcs-agrmnt.docx" was listed after
        # "Warranty Forms.pdf" because capitals sort first.
        chunks = [
            make_chunk("a", "Warranty Forms.pdf"),
            make_chunk("b", "individual-svcs-agrmnt.docx"),
            make_chunk("c", "Sample Contract.docx"),
            make_chunk("d", "texas.pdf"),
        ]

        order = list(KeywordIndex.build(chunks).chunks_per_source())

        assert order == [
            "individual-svcs-agrmnt.docx",
            "Sample Contract.docx",
            "texas.pdf",
            "Warranty Forms.pdf",
        ]

    def test_empty_index(self) -> None:
        assert KeywordIndex.build([]).chunks_per_source() == {}
