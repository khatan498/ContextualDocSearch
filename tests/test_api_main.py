"""Tests for app.api.main.

``TestClient`` drives the app in-process — the equivalent of
``WebApplicationFactory`` plus its ``HttpClient`` — and every app here is built
around a stub retriever, so no model is loaded and nothing touches the network.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.main import create_app
from app.indexing.built_index import IndexUnavailableError
from app.retrieval.results import SearchResult


def result(source_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=f"{source_id}::0",
        source_id=source_id,
        chunk_index=0,
        text=f"body of {source_id}",
        score=score,
        rrf_score=0.03,
        vector_rank=1,
        keyword_rank=None,
    )


class StubRetriever:
    """Stands in for HybridRetriever and records every search it receives."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results if results is not None else [
            result("Warranty Forms.pdf", 7.4),
            result("contract.docx", 1.2),
        ]
        self.calls: list[tuple[str, int | None]] = []

    @property
    def chunk_count(self) -> int:
        return 69

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results


@pytest.fixture
def stub() -> StubRetriever:
    return StubRetriever()


@pytest.fixture
def client(stub: StubRetriever) -> TestClient:
    return TestClient(create_app(stub))


class TestSearch:
    def test_returns_results_in_order_with_ranks(
        self, client: TestClient, stub: StubRetriever
    ) -> None:
        response = client.post("/search", json={"query": "how long is the warranty"})

        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "how long is the warranty"
        assert [hit["rank"] for hit in body["results"]] == [1, 2]
        assert [hit["source_id"] for hit in body["results"]] == [
            "Warranty Forms.pdf",
            "contract.docx",
        ]

    def test_every_field_reaches_the_response(self, client: TestClient) -> None:
        hit = client.post("/search", json={"query": "warranty"}).json()["results"][0]

        assert hit == {
            "rank": 1,
            "chunk_id": "Warranty Forms.pdf::0",
            "source_id": "Warranty Forms.pdf",
            "chunk_index": 0,
            "text": "body of Warranty Forms.pdf",
            "score": 7.4,
            "rrf_score": 0.03,
            "vector_rank": 1,
            "keyword_rank": None,
        }

    def test_reports_how_long_it_took(self, client: TestClient) -> None:
        body = client.post("/search", json={"query": "warranty"}).json()

        assert isinstance(body["took_ms"], int)
        assert body["took_ms"] >= 0

    def test_top_k_reaches_the_retriever(
        self, client: TestClient, stub: StubRetriever
    ) -> None:
        client.post("/search", json={"query": "warranty", "top_k": 3})

        assert stub.calls == [("warranty", 3)]

    def test_omitted_top_k_leaves_the_default_to_the_retriever(
        self, client: TestClient, stub: StubRetriever
    ) -> None:
        client.post("/search", json={"query": "warranty"})

        # None means "use SEARCH_TOP_K", decided in one place: the retriever.
        assert stub.calls == [("warranty", None)]

    def test_the_retriever_receives_the_stripped_query(
        self, client: TestClient, stub: StubRetriever
    ) -> None:
        response = client.post("/search", json={"query": "  warranty  "})

        assert stub.calls == [("warranty", None)]
        assert response.json()["query"] == "warranty"

    def test_no_results(self, stub: StubRetriever, client: TestClient) -> None:
        stub.results = []

        response = client.post("/search", json={"query": "zzz"})

        assert response.status_code == 200
        assert response.json()["results"] == []


class TestValidation:
    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"query": ""},
            {"query": "   "},
            {"query": "a" * 501},
            {"query": "warranty", "top_k": 0},
            {"query": "warranty", "top_k": 101},
            {"query": "warranty", "topk": 3},
            {"query": 42},
        ],
    )
    def test_invalid_body_is_422_and_never_searches(
        self, client: TestClient, stub: StubRetriever, body: dict
    ) -> None:
        response = client.post("/search", json=body)

        assert response.status_code == 422
        assert stub.calls == []

    def test_422_names_the_bad_field(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "warranty", "top_k": 0})

        locations = [error["loc"] for error in response.json()["detail"]]
        assert ["body", "top_k"] in locations

    def test_422_does_not_echo_the_rejected_value(self, client: TestClient) -> None:
        # FastAPI's default copies each rejected value back under "input", so
        # an oversized query came straight back — 20 MB in, 20 MB out.
        rejected = "confidential " * 40  # over the 500-character limit

        response = client.post("/search", json={"query": rejected})

        assert response.status_code == 422
        assert "confidential" not in response.text
        error = response.json()["detail"][0]
        assert error["loc"] == ["body", "query"]
        assert error["ctx"] == {"max_length": 500}
        assert "input" not in error

    def test_malformed_json_is_still_a_clear_422(self, client: TestClient) -> None:
        response = client.post(
            "/search", content=b'{"query": ', headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "json_invalid"

    def test_oversized_body_is_413_and_never_searches(
        self, client: TestClient, stub: StubRetriever
    ) -> None:
        response = client.post("/search", json={"query": "x" * 1_000_000})

        assert response.status_code == 413
        assert len(response.content) < 200
        assert stub.calls == []

    def test_query_string_is_not_accepted_as_a_search(self, client: TestClient) -> None:
        # Queries travel in the body so they stay out of URLs and access logs.
        assert client.get("/search", params={"query": "warranty"}).status_code == 405


class SlowStub(StubRetriever):
    """Takes a while per search and records how many calls overlap."""

    def __init__(self) -> None:
        super().__init__()
        self._guard = threading.Lock()
        self.inside = 0
        self.most_inside = 0

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        with self._guard:
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
        time.sleep(0.05)
        with self._guard:
            self.inside -= 1
        return super().search(query, top_k)


class TestConcurrency:
    def test_searches_run_one_at_a_time(self) -> None:
        slow = SlowStub()
        client = TestClient(create_app(slow))

        # Eight requests from four threads at once. FastAPI runs each on its
        # own worker thread, so without the lock they would overlap.
        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(
                pool.map(
                    lambda i: client.post("/search", json={"query": f"q{i}"}).status_code,
                    range(8),
                )
            )

        assert statuses == [200] * 8
        assert len(slow.calls) == 8
        assert slow.most_inside == 1

    def test_a_failed_search_releases_the_lock(self) -> None:
        # `with lock:` must release on an exception, or one bug would hang
        # every later request.
        class FailsOnce(StubRetriever):
            failed = False

            def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("a real bug")
                return super().search(query, top_k)

        client = TestClient(create_app(FailsOnce()), raise_server_exceptions=False)

        assert client.post("/search", json={"query": "q"}).status_code == 500
        assert client.post("/search", json={"query": "q"}).status_code == 200


class TestErrors:
    def test_a_genuine_bug_is_a_500(self) -> None:
        class Broken(StubRetriever):
            def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
                raise RuntimeError("a real bug")

        client = TestClient(create_app(Broken()), raise_server_exceptions=False)

        response = client.post("/search", json={"query": "warranty"})

        assert response.status_code == 500
        # The details stay in the server log, not in the response.
        assert "a real bug" not in response.text


class TestPrivacy:
    def test_the_query_text_is_never_logged(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "my confidential search phrase"

        with caplog.at_level(logging.DEBUG):
            client.post("/search", json={"query": secret})

        assert caplog.records, "expected the search to log something"
        assert secret not in caplog.text


class TestHealth:
    def test_reports_the_corpus_and_models(self, client: TestClient) -> None:
        from app.config import get_settings

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "chunks": 69,
            "embedding_model": get_settings().embedding_model_name,
            "reranker_model": get_settings().reranker_model_name,
        }


class TestDocumentation:
    def test_root_redirects_to_the_interactive_docs(self, client: TestClient) -> None:
        response = client.get("/", follow_redirects=False)

        assert response.status_code in (302, 307)
        assert response.headers["location"] == "/docs"

    def test_openapi_lists_the_endpoints(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]

        assert "post" in paths["/search"]
        assert "get" in paths["/health"]
        assert "/" not in paths

    def test_openapi_warns_that_scores_are_per_query(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()["components"]["schemas"]

        assert "same query" in schema["SearchHit"]["properties"]["score"]["description"]


class TestLifespan:
    """`uvicorn app.api.main:app` builds the app with no retriever; the
    lifespan must open one, once, before the first request."""

    def test_opens_the_retriever_once_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[StubRetriever] = []

        def fake_open(cls) -> StubRetriever:
            opened.append(StubRetriever())
            return opened[-1]

        monkeypatch.setattr(api_main.HybridRetriever, "open", classmethod(fake_open))
        app = create_app()

        assert opened == []  # building the app loads nothing

        # `with` runs the lifespan: startup on entry, shutdown on exit.
        with TestClient(app) as client:
            assert len(opened) == 1
            client.post("/search", json={"query": "one"})
            client.post("/search", json={"query": "two"})

        assert len(opened) == 1
        assert opened[0].calls == [("one", None), ("two", None)]

    def test_a_refusal_at_startup_stops_the_app_and_logs_why(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def refuse(cls) -> StubRetriever:
            raise IndexUnavailableError("No search index. Run build_index.py first.")

        monkeypatch.setattr(api_main.HybridRetriever, "open", classmethod(refuse))

        with pytest.raises(IndexUnavailableError):
            with TestClient(create_app()):
                pass

        assert "Run build_index.py first" in caplog.text

    def test_a_supplied_retriever_is_used_without_opening_another(
        self, monkeypatch: pytest.MonkeyPatch, stub: StubRetriever
    ) -> None:
        def must_not_open(cls) -> StubRetriever:
            raise AssertionError("opened a second retriever")

        monkeypatch.setattr(api_main.HybridRetriever, "open", classmethod(must_not_open))

        with TestClient(create_app(stub)) as client:
            assert client.post("/search", json={"query": "q"}).status_code == 200


@pytest.mark.integration
class TestRealIndex:
    """The real app, started the way `uvicorn app.api.main:app` starts it,
    over the real built index and both real models."""

    @pytest.fixture
    def client(self):
        from app.config import get_settings

        if not get_settings().keyword_index_path.exists():
            pytest.skip("index not built; run python scripts/build_index.py")
        with TestClient(create_app()) as client:
            yield client

    def test_warranty_query_over_http(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "how long is the warranty"})

        assert response.status_code == 200
        assert response.json()["results"][0]["source_id"] == "Warranty Forms.pdf"

    def test_health_reports_the_real_corpus(self, client: TestClient) -> None:
        body = client.get("/health").json()

        assert body["status"] == "ready"
        assert body["chunks"] > 0
