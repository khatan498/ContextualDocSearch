"""Serve the search API over HTTP.

    python scripts/serve.py
    python scripts/serve.py --port 8080
    python scripts/serve.py --host 0.0.0.0     # reachable from other machines

Opens the indexes and loads both models *before* binding the port, so every
deliberate refusal — personal mode, invalid configuration, a missing or stale
index — is printed as a message and the server never starts half-working.
Build the indexes first with scripts/build_index.py, and restart this server
after rebuilding them: the keyword index is read into memory at startup.
"""

import argparse
import logging
import sys
from pathlib import Path

# Same reason as in the other scripts: make `import app` work when this file
# is run directly from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

from app.api.main import create_app  # noqa: E402
from app.cli import port_number, run_with_refusals, silence_model_loading_bars  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


def serve(host: str | None, port: int | None) -> int:
    """Open the retriever, then serve the API until interrupted.

    Args:
        host: Address to bind, or ``None`` for the configured ``API_HOST``.
        port: Port to bind, or ``None`` for the configured ``API_PORT``.

    Returns:
        Process exit code, once the server stops.
    """
    # Reading the settings is itself a check: invalid configuration or personal
    # mode raises here, before anything else happens.
    settings = get_settings()
    bind_host = settings.api_host if host is None else host
    bind_port = settings.api_port if port is None else port

    print("loading models and checking the index...")
    retriever = HybridRetriever.open()
    print(f"ready  : {retriever.chunk_count} chunks searchable")
    print(f"serving: http://{bind_host}:{bind_port}  (interactive docs at /docs)")
    print()

    # Passing the app object rather than an "app.api.main:app" string means the
    # retriever opened above is the one that serves every request.
    uvicorn.run(create_app(retriever), host=bind_host, port=bind_port)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, for testing. Defaults to ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", help="address to bind (default API_HOST, 127.0.0.1)")
    parser.add_argument(
        "--port", type=port_number, help="port to listen on (default API_PORT, 8000)"
    )
    args = parser.parse_args(argv)

    # Same logging setup as build_index.py: this project's messages at INFO,
    # the chatty libraries held to WARNING, all on stdout so the order holds.
    # uvicorn configures its own "uvicorn.*" loggers, including the access log.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("chromadb", "sentence_transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    silence_model_loading_bars()

    # Deliberate refusals print their message and exit 1 or 2 without ever
    # binding the port; anything unexpected keeps its traceback.
    return run_with_refusals(lambda: serve(args.host, args.port))


if __name__ == "__main__":
    raise SystemExit(main())
