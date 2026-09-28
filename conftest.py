"""Pytest configuration for the repository root.

This file exists mainly for a side effect: pytest prepends the directory
containing the rootdir ``conftest.py`` onto ``sys.path``. That is what makes
``import app.ingestion.chunking`` resolve when tests run.

Without it, pytest would instead put ``tests/`` on ``sys.path`` and every
``from app...`` import would fail with ModuleNotFoundError.

(Python has no .csproj-style project file; ``sys.path`` is the equivalent of
the assembly probing path, and it is assembled at runtime.)

It also holds the fakes shared across test files. Fixtures defined here are
auto-discovered by pytest and injected into tests by parameter name. The fake
classes themselves can be imported (``from conftest import ...``) for type
hints, which works for the same sys.path reason.
"""

import pytest


class FakeEmbeddingModel:
    """Satisfies TextEmbedder without loading anything.

    Vectors are deterministic and derived from the text, so "nearest" is
    predictable but the values are meaningless. Tokens are counted one per
    word, so a test can predict a chunk's token_count exactly.
    """

    name = "fake/embedder"
    max_seq_length = 512
    dimensions = 8

    def __init__(self) -> None:
        # Every query embedded, so a test can assert whether the model was
        # consulted at all.
        self.queries: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        counts = [0.0] * 8
        for word in text.lower().split():
            counts[hash(word) % 8] += 1.0
        norm = sum(value * value for value in counts) ** 0.5 or 1.0
        return [value / norm for value in counts]


# A fixture is injected into any test that names it as a parameter — pytest's
# form of constructor injection. Defining it here, in the root conftest.py,
# makes it available to every test file without an import.
@pytest.fixture
def fake_embedder() -> FakeEmbeddingModel:
    """A fresh offline stand-in for the embedding model."""
    return FakeEmbeddingModel()


@pytest.fixture
def fresh_settings():
    """Clear the cached Settings before and after the test.

    ``get_settings()`` is memoised, so environment changes made with
    ``monkeypatch.setenv`` are invisible until the cache is cleared — and a
    Settings built from them must not leak into any later test.
    """
    from app.config import get_settings

    get_settings.cache_clear()
    # `yield` splits the fixture: the code before it is setup, the code after
    # it teardown — pytest's equivalent of a using/IDisposable block.
    yield
    get_settings.cache_clear()
