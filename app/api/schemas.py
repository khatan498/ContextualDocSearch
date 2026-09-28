"""The HTTP contract: request and response bodies for the search API.

These pydantic models are the API's data-transfer objects. They are kept apart
from the retrieval layer's ``SearchResult`` dataclass so the wire format and the
domain type can each change without dragging the other along — the same split
as DTOs and domain entities in an ASP.NET Core project.

FastAPI validates every request body against these models before the endpoint
runs, and publishes them, descriptions included, in the OpenAPI schema at
``/openapi.json`` and the interactive page at ``/docs``.
"""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.retrieval.results import SearchResult

# Bounds the work one request can ask for. The embedding model and the reranker
# read only ~512 tokens anyway, and a normal question is well under 100.
MAX_QUERY_CHARS = 500

# `Annotated[T, ...]` attaches extra metadata to a type — here, constraints that
# pydantic enforces: whitespace is stripped first, then the length is checked,
# so "   " is rejected as empty. Declaring it as a type alias lets it be reused
# and keeps the model below readable. The closest C# analogue is a validation
# attribute bundle applied to a property.
QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS),
]


class SearchRequest(BaseModel):
    """Body of ``POST /search``."""

    # `extra="forbid"` rejects unknown fields, so a typo such as "topk" is a
    # 422 naming the field rather than a silently ignored setting.
    model_config = ConfigDict(extra="forbid")

    query: QueryText = Field(
        description="What to search for. Surrounding whitespace is ignored.",
        examples=["how long is the warranty"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description=(
            "Results to return. Defaults to the server's SEARCH_TOP_K. Fewer come "
            "back if the fused shortlist is smaller."
        ),
    )


class SearchHit(BaseModel):
    """One result in a search response."""

    rank: int = Field(description="Position in this response, starting at 1.")
    chunk_id: str = Field(description='Stable chunk id, e.g. "Warranty Forms.pdf::1".')
    source_id: str = Field(description="Document the passage came from.")
    chunk_index: int = Field(description="Position of the passage within its document.")
    text: str = Field(description="The passage itself.")
    score: float = Field(
        description=(
            "The reranker's relevance score. Higher is better, but it is only "
            "comparable with other results for the same query: it is not a "
            "probability and cannot tell a good match from no match."
        )
    )
    rrf_score: float = Field(description="Fused score the passage reached the reranker with.")
    vector_rank: int | None = Field(
        description="Where vector search placed it, 1-based; null if it was not found there."
    )
    keyword_rank: int | None = Field(
        description="Where BM25 keyword search placed it, 1-based; null if not found there."
    )

    @classmethod
    def from_result(cls, rank: int, result: SearchResult) -> Self:
        """Map a retrieval result onto the wire format.

        Written out field by field, rather than letting pydantic read
        attributes automatically, so the mapping is visible in one place.

        Args:
            rank: 1-based position in the response.
            result: The retrieval layer's result.

        Returns:
            The API representation.
        """
        return cls(
            rank=rank,
            chunk_id=result.chunk_id,
            source_id=result.source_id,
            chunk_index=result.chunk_index,
            text=result.text,
            score=result.score,
            rrf_score=result.rrf_score,
            vector_rank=result.vector_rank,
            keyword_rank=result.keyword_rank,
        )


class SearchResponse(BaseModel):
    """Body returned by ``POST /search``."""

    query: str = Field(description="The query as searched, whitespace stripped.")
    took_ms: int = Field(description="Time spent searching, in milliseconds.")
    results: list[SearchHit] = Field(description="Most relevant first.")


class HealthResponse(BaseModel):
    """Body returned by ``GET /health``."""

    status: str = Field(description='Always "ready": the server only starts once search works.')
    chunks: int = Field(description="Passages available to search.")
    embedding_model: str
    reranker_model: str


class DocumentInfo(BaseModel):
    """One searchable document."""

    source_id: str = Field(description='Document name, e.g. "Warranty Forms.pdf".')
    chunks: int = Field(description="Passages the document was split into for search.")


class DocumentsResponse(BaseModel):
    """Body returned by ``GET /documents``."""

    documents: list[DocumentInfo] = Field(description="Sorted by document name.")
