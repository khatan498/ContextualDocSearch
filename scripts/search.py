"""Search the bundled demo corpus.

    python scripts/search.py "how long is the warranty"
    python scripts/search.py "termination notice period" -k 3

Runs the full retrieval pipeline — both indexes, reciprocal rank fusion, then
cross-encoder reranking — and prints the results best first. Build the indexes
first with scripts/build_index.py.
"""

import argparse
import logging
import sys
from pathlib import Path

# Same reason as in build_index.py: make `import app` work when this file is
# run directly from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from app.cli import positive_int, silence_model_loading_bars  # noqa: E402
from app.indexing.built_index import IndexUnavailableError  # noqa: E402
from app.indexing.embeddings import ChunkWindowTooLargeError  # noqa: E402
from app.modes import PersonalModeUnavailableError  # noqa: E402
from app.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from app.retrieval.results import SearchResult  # noqa: E402


def _provenance(result: SearchResult) -> str:
    """Say which index found a result, and where it placed it."""
    # A conditional expression — `a if condition else b` — is Python's ternary,
    # C#'s `condition ? a : b` with the operands in a different order.
    vector = f"#{result.vector_rank}" if result.vector_rank is not None else "-"
    keyword = f"#{result.keyword_rank}" if result.keyword_rank is not None else "-"
    return f"vec {vector:>3}  bm25 {keyword:>3}"


def print_results(query: str, results: list[SearchResult]) -> None:
    """Render search results, one line each.

    Args:
        query: The query, echoed as a header.
        results: Results, best first.
    """
    print(f"query  : {query!r}")
    print()
    if not results:
        print("  (no results)")
        return

    for position, result in enumerate(results, start=1):
        location = f"{result.source_id}#{result.chunk_index}"
        snippet = " ".join(result.text.split())[:60]
        print(
            f"  {position:>2}  {result.score:7.3f}  {location:<34} "
            f"{_provenance(result)}  {snippet}..."
        )
    print()
    print("  score: reranker relevance, comparable only within this query")
    print("  vec / bm25: where each index placed the chunk before fusion (- = not found)")


def search(query: str, k: int | None) -> int:
    """Run one search and print it.

    Args:
        query: Query text.
        k: Results to show, or ``None`` for the configured ``SEARCH_TOP_K``.

    Returns:
        Process exit code.
    """
    retriever = HybridRetriever.open()
    print_results(query, retriever.search(query, top_k=k))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, for testing. Defaults to ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help="what to search for")
    parser.add_argument(
        "-k",
        type=positive_int,
        default=None,
        help="results to show, at least 1 (default SEARCH_TOP_K)",
    )
    args = parser.parse_args(argv)
    if not args.query.strip():
        # Rejected here, before either model loads: a blank query would
        # otherwise cost a seven-second model load to print "no results".
        # parser.error prints usage and exits with code 2, like any other
        # argument error.
        parser.error("the query must not be blank")

    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    silence_model_loading_bars()

    # The same pattern as build_index.main(): deliberate refusals print their
    # message; anything unexpected keeps its traceback.
    try:
        return search(args.query, args.k)
    except IndexUnavailableError as exc:
        # Exit 1, like build_index --verify without an index: a state that
        # rebuilding fixes, not a configuration the user got wrong.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (PersonalModeUnavailableError, ChunkWindowTooLargeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"error: invalid configuration\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
