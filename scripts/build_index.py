"""Build the search indexes over the bundled demo corpus.

    python scripts/build_index.py
    python scripts/build_index.py --verify "how long is the warranty"

Reads every document in the demo corpus, splits it into chunks measured with the
embedding model's own tokenizer, and writes two indexes side by side: a Chroma
vector store for meaning, and a BM25 file for exact terms.
"""

import argparse
import logging
import sys
from pathlib import Path

# The project is not pip-installable (no pyproject.toml), and Python puts the
# *script's* directory on sys.path rather than the working directory. Without
# this, `import app` fails when run as `python scripts/build_index.py`. The test
# suite solves the same problem through the root conftest.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli import positive_int, run_with_refusals, silence_model_loading_bars  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.indexing.built_index import open_built_indexes  # noqa: E402
from app.indexing.embeddings import EmbeddingModel  # noqa: E402
from app.indexing.hits import IndexHit  # noqa: E402
from app.indexing.keyword_index import KeywordIndex  # noqa: E402
from app.indexing.vector_store import VectorStore  # noqa: E402
from app.ingestion.chunking import Chunk, chunk_text  # noqa: E402
from app.ingestion.connectors.local_fs import LocalFSConnector  # noqa: E402
from app.ingestion.loaders import DocumentLoadError, extract_text  # noqa: E402

logger = logging.getLogger("build_index")


def relative_source_id(source_id: str, root: Path) -> str:
    """Shorten an absolute document path to one relative to the corpus.

    Chunk ids are built from this, and they must be identical on every machine
    for a rebuild to overwrite rather than duplicate. It also makes ``--verify``
    output readable.

    Args:
        source_id: Absolute path as stamped by the connector.
        root: Corpus root directory.

    Returns:
        A path relative to ``root``, or the bare filename if it lies outside.
    """
    try:
        return Path(source_id).relative_to(root).as_posix()
    except ValueError:
        return Path(source_id).name


def collect_chunks(connector: LocalFSConnector, model: EmbeddingModel) -> list[Chunk]:
    """Read every document in the corpus and chunk it.

    Args:
        connector: Source of documents.
        model: Supplies the tokenizer used to measure chunk boundaries.

    Returns:
        Every chunk from every readable document, in corpus order.
    """
    chunks: list[Chunk] = []

    for raw_bytes, metadata in connector.iter_documents():
        try:
            text = extract_text(raw_bytes, metadata)
        except DocumentLoadError as exc:
            logger.warning("Skipping %s: %s", metadata.title, exc)
            continue

        if not text.strip():
            logger.warning("Skipping %s: no extractable text", metadata.title)
            continue

        source_id = relative_source_id(metadata.source_id, connector.root)
        # The real tokenizer, not the word-based estimate. Boundaries are then
        # measured in the same units the model will read them in, which is what
        # makes the chunk-size guard meaningful.
        document_chunks = list(
            chunk_text(text, source_id=source_id, count_tokens=model.count_tokens)
        )
        logger.info("%-32s %3d chunks", source_id, len(document_chunks))
        chunks.extend(document_chunks)

    return chunks


def build() -> int:
    """Build both indexes from scratch.

    Returns:
        Process exit code.
    """
    settings = get_settings()
    connector = LocalFSConnector()

    print(f"corpus : {connector.root}")
    print(f"model  : {settings.embedding_model_name}")
    print("loading model (first run downloads it)...")

    # Constructing the model is also what enforces the chunk-size guard: it
    # raises ChunkWindowTooLargeError if CHUNK_MAX_TOKENS exceeds what the
    # model can read, rather than letting text be truncated invisibly.
    model = EmbeddingModel()
    print(f"         limit {model.max_seq_length} tokens, {model.dimensions} dims")
    print()

    chunks = collect_chunks(connector, model)
    if not chunks:
        print("No indexable content found.")
        return 1

    largest = max(chunk.token_count for chunk in chunks)
    print()
    print(f"chunks : {len(chunks)}  (largest {largest} tokens)")

    print("embedding...")
    vectors = model.embed_documents([chunk.text for chunk in chunks])

    VectorStore.open().replace_all(chunks, vectors, model_name=model.name)
    bm25_path = settings.keyword_index_path
    KeywordIndex.build(chunks).save(bm25_path)

    print()
    print(f"wrote  : {settings.vector_store_path}  (vectors)")
    print(f"         {bm25_path}  (bm25)")
    return 0


def _print_hits(title: str, hits: list[IndexHit]) -> None:
    """Render one index's results."""
    print(f"  {title}")
    if not hits:
        print("    (no matches)")
        return
    for hit in hits:
        # Collapse the snippet onto one line so the ranking stays scannable.
        snippet = " ".join(hit.text.split())[:70]
        print(f"    {hit.score:7.3f}  {hit.source_id}#{hit.chunk_index}  {snippet}...")


def verify(query: str, k: int) -> int:
    """Query each index separately and print what it returns.

    This is a diagnostic, not search. Real results come from
    scripts/search.py, where the two lists are merged by reciprocal rank fusion
    and then reranked — the project's rule is that raw vector hits are never
    returned on their own.

    Args:
        query: Query text.
        k: Hits to show per index.

    Returns:
        Process exit code.

    Raises:
        IndexUnavailableError: If the indexes are missing, unreadable, or were
            built with a different embedding model.
    """
    # The same checks search runs, so the diagnostic never queries an index
    # that search would refuse.
    keyword_index, store = open_built_indexes()
    model = EmbeddingModel()

    print("[diagnostic - each index on its own; for real results use scripts/search.py]")
    print()
    print(f"query  : {query!r}")
    print(f"indexed: {store.count()} chunks")
    print()

    _print_hits(
        "vector index (cosine similarity, 1.0 = identical)",
        store.query(model.embed_query(query), k=k),
    )
    print()
    _print_hits(
        "keyword index (BM25 score, corpus-relative)",
        keyword_index.query(query, k=k),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, for testing. Defaults to ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify",
        metavar="QUERY",
        help="query the existing indexes and print each one's hits",
    )
    parser.add_argument(
        "-k", type=positive_int, default=3, help="hits to show per index (default 3)"
    )
    args = parser.parse_args(argv)

    # Configuring logging is the application's job; the library modules only
    # emit. Chroma and sentence-transformers are chatty at INFO, so they are
    # held to WARNING to keep this output readable.
    # Log to stdout, not the default stderr: the two streams are buffered
    # separately, so mixing them scrambles the order of what the user reads.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("chromadb", "sentence_transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    silence_model_loading_bars()

    # Deliberate refusals print their message and exit 1 or 2; anything
    # unexpected keeps its traceback. See run_with_refusals for the mapping.
    # `lambda:` builds a small anonymous function — like C#'s `() => ...` — so
    # the choice between verify and build is made inside the handler.
    return run_with_refusals(
        lambda: verify(args.verify, args.k) if args.verify else build()
    )

if __name__ == "__main__":
    raise SystemExit(main())
