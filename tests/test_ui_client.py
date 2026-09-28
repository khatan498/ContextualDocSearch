"""Tests for app.ui.client.

``httpx.MockTransport`` answers every request from a function in the test, so
no server runs and nothing touches the network.
"""

import json

import httpx
import pytest

from app.ui.client import ApiError, ApiUnavailableError, SearchClient, SearchRejectedError

HIT = {
    "rank": 1,
    "chunk_id": "Warranty Forms.pdf::1",
    "source_id": "Warranty Forms.pdf",
    "chunk_index": 1,
    "text": "within a period of 24 months",
    "score": -0.68,
    "rrf_score": 0.0313,
    "vector_rank": 1,
    "keyword_rank": 7,
}


def client_answering(handler) -> tuple[SearchClient, list[httpx.Request]]:
    """A client whose every request is answered by `handler`."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = SearchClient("http://api.test", transport=httpx.MockTransport(record))
    return client, seen


class TestSearch:
    def test_parses_the_response(self) -> None:
        client, _ = client_answering(
            lambda r: httpx.Response(200, json={"query": "warranty", "took_ms": 1117, "results": [HIT]})
        )

        response = client.search("warranty")

        assert response.took_ms == 1117
        assert response.results[0].source_id == "Warranty Forms.pdf"
        assert response.results[0].keyword_rank == 7

    def test_posts_the_query_in_the_body(self) -> None:
        client, seen = client_answering(
            lambda r: httpx.Response(200, json={"query": "q", "took_ms": 1, "results": []})
        )

        client.search("how long is the warranty", top_k=3)

        request = seen[0]
        assert request.method == "POST"
        assert request.url.path == "/search"
        # Never in the URL, where the server's access log would record it.
        assert "warranty" not in str(request.url)
        assert json.loads(request.content) == {"query": "how long is the warranty", "top_k": 3}

    def test_omitted_top_k_is_not_sent(self) -> None:
        client, seen = client_answering(
            lambda r: httpx.Response(200, json={"query": "q", "took_ms": 1, "results": []})
        )

        client.search("q")

        assert json.loads(seen[0].content) == {"query": "q"}


class TestFailures:
    def test_422_becomes_a_readable_rejection(self) -> None:
        client, _ = client_answering(
            lambda r: httpx.Response(
                422,
                json={"detail": [{"loc": ["body", "query"], "msg": "String should have at least 1 character", "type": "string_too_short"}]},
            )
        )

        with pytest.raises(SearchRejectedError, match="query: String should have at least 1 character"):
            client.search(" ")

    def test_unreadable_422_still_rejects(self) -> None:
        client, _ = client_answering(lambda r: httpx.Response(422, text="not json"))

        with pytest.raises(SearchRejectedError, match="rejected"):
            client.search("q")

    @pytest.mark.parametrize("status", [413, 500, 503])
    def test_other_errors_become_api_errors(self, status: int) -> None:
        client, _ = client_answering(lambda r: httpx.Response(status, json={"detail": "x"}))

        with pytest.raises(ApiError, match=str(status)):
            client.search("q")

    def test_connection_refused_means_unavailable(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client, _ = client_answering(refuse)

        with pytest.raises(ApiUnavailableError, match="isn't answering"):
            client.search("q")

    def test_timeout_means_unavailable(self) -> None:
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        client, _ = client_answering(slow)

        with pytest.raises(ApiUnavailableError):
            client.health()

    def test_a_response_of_the_wrong_shape_is_an_api_error(self) -> None:
        client, _ = client_answering(lambda r: httpx.Response(200, json={"unexpected": True}))

        with pytest.raises(ApiError, match="could not read"):
            client.search("q")

    def test_a_non_json_response_is_an_api_error(self) -> None:
        client, _ = client_answering(lambda r: httpx.Response(200, text="<html>"))

        with pytest.raises(ApiError):
            client.health()

    def test_the_real_cause_is_chained(self) -> None:
        # `raise ... from exc` keeps the transport error for the terminal log.
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client, _ = client_answering(refuse)

        with pytest.raises(ApiUnavailableError) as raised:
            client.search("q")

        assert isinstance(raised.value.__cause__, httpx.ConnectError)


class TestHealthAndDocuments:
    def test_health(self) -> None:
        client, seen = client_answering(
            lambda r: httpx.Response(
                200,
                json={"status": "ready", "chunks": 69, "embedding_model": "e", "reranker_model": "r"},
            )
        )

        health = client.health()

        assert (health.status, health.chunks) == ("ready", 69)
        assert (seen[0].method, seen[0].url.path) == ("GET", "/health")

    def test_documents(self) -> None:
        client, seen = client_answering(
            lambda r: httpx.Response(
                200, json={"documents": [{"source_id": "Warranty Forms.pdf", "chunks": 3}]}
            )
        )

        documents = client.documents().documents

        assert [(d.source_id, d.chunks) for d in documents] == [("Warranty Forms.pdf", 3)]
        assert seen[0].url.path == "/documents"


@pytest.mark.integration
class TestAgainstTheRealApi:
    """The real client, talking to the real app over the real index."""

    def test_warranty_query_reaches_the_ui_layer(self) -> None:
        from fastapi.testclient import TestClient

        from app.api.main import create_app
        from app.config import get_settings

        if not get_settings().keyword_index_path.exists():
            pytest.skip("index not built; run python scripts/build_index.py")

        # TestClient is an httpx.Client wired to the app in-process; its
        # transport lets SearchClient talk to the app with no port bound.
        with TestClient(create_app()) as app_client:
            client = SearchClient("http://testserver", transport=app_client._transport)

            response = client.search("how long is the warranty")
            documents = client.documents().documents

        assert response.results[0].source_id == "Warranty Forms.pdf"
        assert len(documents) == 4
