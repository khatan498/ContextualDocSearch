"""Pytest configuration for the repository root.

This file exists mainly for a side effect: pytest prepends the directory
containing the rootdir ``conftest.py`` onto ``sys.path``. That is what makes
``import app.ingestion.chunking`` resolve when tests run.

Without it, pytest would instead put ``tests/`` on ``sys.path`` and every
``from app...`` import would fail with ModuleNotFoundError.

(Python has no .csproj-style project file; ``sys.path`` is the equivalent of
the assembly probing path, and it is assembled at runtime.)

Shared fixtures can be added here later — they are auto-discovered by pytest
and injected into tests by parameter name.
"""
