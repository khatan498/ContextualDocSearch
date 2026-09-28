"""The search API: the Phase 3 pipeline behind HTTP.

    python scripts/serve.py              # the supported way to run it
    uvicorn app.api.main:app             # also works, from the repo root

Endpoints:
    POST /search   run a search (the query travels in the JSON body)
    GET  /health   what is loaded
    GET  /docs     interactive documentation, generated from the schemas

The server only ever starts with a working retriever. scripts/serve.py opens it
before binding the port, so every deliberate refusal — personal mode, invalid
configuration, a missing or stale index — is printed as a message and the
server never comes up half-working.
"""

import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from app.api.body_limit import BodySizeLimitMiddleware
from app.api.schemas import HealthResponse, SearchHit, SearchRequest, SearchResponse
from app.config import get_settings
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.results import SearchResult

logger = logging.getLogger(__name__)


class Searcher(Protocol):
    """What the API needs from the retrieval layer.

    ``HybridRetriever`` satisfies it; tests pass a lightweight stub, so the
    API is tested without loading any model.
    """

    @property
    def chunk_count(self) -> int: ...

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]: ...


def get_retriever(request: Request) -> Searcher:
    """FastAPI dependency: the retriever this app was started with.

    Declared as a parameter default with ``Depends(get_retriever)``, FastAPI
    calls this for each request and passes the result in — the equivalent of
    constructor injection of a singleton in ASP.NET Core.
    """
    return request.app.state.retriever


def create_app(retriever: Searcher | None = None) -> FastAPI:
    """Build the API around a retriever.

    An application factory rather than one module-level app, so the launcher
    and the tests can each supply their own retriever — the role
    ``WebApplicationFactory`` plays in ASP.NET Core tests.

    Args:
        retriever: An already-open retriever. scripts/serve.py passes the one
            it opened before binding the port; tests pass a stub. When
            omitted — ``uvicorn app.api.main:app`` — one is opened at startup.

    Returns:
        The configured application.
    """

    # `@asynccontextmanager` turns a generator into a context manager: the code
    # before `yield` runs once at startup, the code after it at shutdown — like
    # IHostedService.StartAsync / StopAsync.
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if retriever is None:
            try:
                app.state.retriever = HybridRetriever.open()
            except Exception as exc:
                # Log the reason as one readable line before startup fails.
                # uvicorn then reports the failure with its own traceback;
                # scripts/serve.py avoids that by checking before it starts.
                logger.error("Cannot start the search API: %s", exc)
                raise
        yield

    app = FastAPI(
        title="ContextualDocSearch",
        summary="Hybrid keyword + vector search over the bundled demo documents.",
        lifespan=lifespan,
    )
    if retriever is not None:
        app.state.retriever = retriever

    # Refuse oversized bodies before anything reads them into memory.
    app.add_middleware(BodySizeLimitMiddleware)

    # FastAPI's default 422 response copies each rejected value back to the
    # client under "input" — the whole value, whatever its size. Measured: a
    # 20 MB query drew a 20 MB error response. This handler keeps where the
    # problem is, what it is, and its limits ("ctx"), and drops the echo.
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {key: value for key, value in error.items() if key != "input"}
            for error in exc.errors()
        ]
        # jsonable_encoder converts anything JSON cannot hold natively (an
        # exception object inside "ctx", for instance) into plain values.
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})

    # FastAPI runs each plain `def` endpoint on a worker thread, so two
    # searches can arrive at once. They are run one at a time: measured, the
    # embedding model and reranker already use every CPU core for a single
    # search, so running searches in parallel gained only ~15% — not worth
    # relying on the models being thread-safe.
    search_lock = threading.Lock()

    # The routes are defined inside the factory so each app gets its own lock
    # and retriever. A decorator registers the function with the app — the
    # equivalent of app.MapPost("/search", handler).
    @app.post("/search", response_model=SearchResponse, summary="Search the documents")
    def search(
        request: SearchRequest,
        searcher: Annotated[Searcher, Depends(get_retriever)],
    ) -> SearchResponse:
        """Run the full pipeline: both indexes, rank fusion, then reranking.

        A plain `def`, not `async def`, deliberately: a search is about a
        second of CPU work, and inside `async def` it would block the event
        loop and stall every other request, health checks included.
        """
        # `with` acquires the lock and guarantees its release, even if the
        # search raises — the equivalent of C#'s `lock (obj) { ... }`.
        with search_lock:
            started = time.perf_counter()
            results = searcher.search(request.query, top_k=request.top_k)
            took_ms = round((time.perf_counter() - started) * 1000)

        # Deliberately no query text in the log: the project promises that
        # nothing about your searches leaves the machine or lingers in logs.
        logger.info("search: %d results in %d ms", len(results), took_ms)

        return SearchResponse(
            query=request.query,
            took_ms=took_ms,
            results=[
                SearchHit.from_result(rank, result)
                for rank, result in enumerate(results, start=1)
            ],
        )

    @app.get("/health", response_model=HealthResponse, summary="What is loaded")
    def health(searcher: Annotated[Searcher, Depends(get_retriever)]) -> HealthResponse:
        """Report the searchable corpus and the models in use."""
        settings = get_settings()
        return HealthResponse(
            status="ready",
            chunks=searcher.chunk_count,
            embedding_model=settings.embedding_model_name,
            reranker_model=settings.reranker_model_name,
        )

    # `include_in_schema=False` keeps this convenience redirect out of /docs.
    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Send a browser to the interactive documentation."""
        return RedirectResponse(url="/docs")

    return app


# For `uvicorn app.api.main:app`. Cheap to build: nothing loads until startup.
app = create_app()
