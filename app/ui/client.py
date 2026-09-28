"""A typed client for the search API, used by the Streamlit page.

The responses are parsed with the very pydantic models the server uses to
produce them (``app/api/schemas.py``), so if the API contract changes, the
client changes with it — the two sides cannot drift apart silently. That is
the same idea as sharing a DTO assembly between an ASP.NET Core API and its
client.

Every failure is turned into one of three exceptions whose messages are
written for the person using the page, so the UI never has to show a stack
trace.
"""

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.api.schemas import DocumentsResponse, HealthResponse, SearchResponse


# A type variable makes _parse generic: the model class passed in decides the
# type returned — `T Parse<T>(...) where T : BaseModel` in C# terms.
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class ApiUnavailableError(RuntimeError):
    """The search API could not be reached: not running, or not answering."""


class SearchRejectedError(RuntimeError):
    """The API refused the search as invalid (HTTP 422), e.g. a blank query."""


class ApiError(RuntimeError):
    """The API answered, but with an error or a response the UI cannot read."""


class SearchClient:
    """Talks to the search API over HTTP.

    Args:
        base_url: Where the API listens, e.g. ``http://127.0.0.1:8000``.
        timeout: Seconds to wait for an answer. Searches take about a second,
            but run one at a time on the server, so a busy moment can queue.
        transport: Replaces the network layer. Tests pass
            ``httpx.MockTransport`` so no server is needed.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        """Run a search.

        Args:
            query: What to search for.
            top_k: Results wanted, or ``None`` for the server's default.

        Returns:
            The parsed response.

        Raises:
            ApiUnavailableError: If the API is not reachable.
            SearchRejectedError: If the API rejected the query.
            ApiError: For any other failure.
        """
        body: dict[str, Any] = {"query": query}
        if top_k is not None:
            body["top_k"] = top_k
        return self._parse(SearchResponse, self._request("POST", "/search", json=body))

    def health(self) -> HealthResponse:
        """What the API has loaded.

        Raises:
            ApiUnavailableError: If the API is not reachable.
            ApiError: For any other failure.
        """
        return self._parse(HealthResponse, self._request("GET", "/health"))

    def documents(self) -> DocumentsResponse:
        """The searchable documents and their passage counts.

        Raises:
            ApiUnavailableError: If the API is not reachable.
            ApiError: For any other failure.
        """
        return self._parse(DocumentsResponse, self._request("GET", "/documents"))

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        # httpx.TransportError covers everything below HTTP: connection refused,
        # DNS failure, timeouts, a dropped connection.
        except httpx.TransportError as exc:
            raise ApiUnavailableError(
                f"The search service at {self.base_url} isn't answering."
            ) from exc

        if response.status_code == 422:
            raise SearchRejectedError(_rejection_message(response))
        if response.status_code >= 400:
            raise ApiError(f"The search service answered with an error ({response.status_code}).")
        return response

    @staticmethod
    def _parse(model: type[ResponseModel], response: httpx.Response) -> ResponseModel:
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ApiError("The search service sent a response the page could not read.") from exc


def _rejection_message(response: httpx.Response) -> str:
    """Turn FastAPI's 422 body into one readable sentence.

    FastAPI reports each problem as ``{"loc": ["body", "query"], "msg": ...}``.
    """
    try:
        first = response.json()["detail"][0]
        field = first["loc"][-1]
        return f"{field}: {first['msg']}"
    except (ValueError, KeyError, IndexError, TypeError):
        return "The search service rejected the request."
