"""Tests for app.api.schemas."""

import pytest
from pydantic import ValidationError

from app.api.schemas import MAX_QUERY_CHARS, SearchHit, SearchRequest
from app.retrieval.results import SearchResult


class TestSearchRequest:
    def test_minimal_request(self) -> None:
        request = SearchRequest(query="warranty")

        assert request.query == "warranty"
        assert request.top_k is None

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert SearchRequest(query="  how long is the warranty \n").query == (
            "how long is the warranty"
        )

    @pytest.mark.parametrize("query", ["", "   ", "\n\t"])
    def test_blank_query_is_rejected(self, query: str) -> None:
        with pytest.raises(ValidationError, match="query"):
            SearchRequest(query=query)

    def test_missing_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="query"):
            SearchRequest()  # type: ignore[call-arg]

    def test_query_at_the_limit_is_accepted(self) -> None:
        assert SearchRequest(query="a" * MAX_QUERY_CHARS)

    def test_over_long_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="query"):
            SearchRequest(query="a" * (MAX_QUERY_CHARS + 1))

    def test_length_is_measured_after_stripping(self) -> None:
        # Padding must not push an acceptable query over the limit.
        assert SearchRequest(query="  " + "a" * MAX_QUERY_CHARS + "  ")

    @pytest.mark.parametrize("top_k", [1, 5, 100])
    def test_top_k_in_range(self, top_k: int) -> None:
        assert SearchRequest(query="q", top_k=top_k).top_k == top_k

    @pytest.mark.parametrize("top_k", [0, -1, 101])
    def test_top_k_out_of_range(self, top_k: int) -> None:
        with pytest.raises(ValidationError, match="top_k"):
            SearchRequest(query="q", top_k=top_k)

    def test_unknown_fields_are_rejected(self) -> None:
        # A typo must fail loudly rather than be silently ignored.
        with pytest.raises(ValidationError, match="topk"):
            SearchRequest(query="q", topk=3)  # type: ignore[call-arg]


class TestSearchHit:
    def test_from_result_copies_every_field(self) -> None:
        result = SearchResult(
            chunk_id="Warranty Forms.pdf::1",
            source_id="Warranty Forms.pdf",
            chunk_index=1,
            text="within a period of 24 months",
            score=-0.68,
            rrf_score=0.0313,
            vector_rank=1,
            keyword_rank=None,
        )

        hit = SearchHit.from_result(3, result)

        assert hit.model_dump() == {
            "rank": 3,
            "chunk_id": "Warranty Forms.pdf::1",
            "source_id": "Warranty Forms.pdf",
            "chunk_index": 1,
            "text": "within a period of 24 months",
            "score": -0.68,
            "rrf_score": 0.0313,
            "vector_rank": 1,
            "keyword_rank": None,
        }

    def test_every_result_field_is_mapped(self) -> None:
        # If SearchResult gains a field, this fails until the API decides
        # whether to expose it.
        from dataclasses import fields

        result_fields = {f.name for f in fields(SearchResult)}
        assert result_fields <= set(SearchHit.model_fields)
