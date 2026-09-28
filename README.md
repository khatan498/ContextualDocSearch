# ContextualDocSearch

Hybrid (keyword + vector) search over a small set of sample documents bundled with
the repo — contracts, service agreements, a sublease, and warranty forms.

Ask *"how long is the warranty"* and get back the paragraph that says
*"coverage period: 24 months from date of purchase"* — even though it shares no
keywords with the question.

**This is a demo.** It searches the documents in `data/sample_docs/` and nothing
else. Connecting your own files is a planned future release — see [Modes](#modes).

> ## Work in progress — runs locally, not packaged for deployment
>
> Ingestion, indexing, hybrid retrieval, the search API and the search page all
> work and are tested. `python scripts/run_local.py` opens the search page in your
> browser. It is built to run on your own machine; there is no deployment setup.
>
> This repo is public to track progress in the open, not because it is ready to use.
> Read [Known issues](#known-issues) before building on it.

---

## What works now

Ingestion over the bundled sample documents:

- **Walk the corpus directory** recursively, skipping hidden files and directories
  (including Windows hidden-attribute files), unsupported types, and anything over a
  configurable size limit.
- **Extract plain text** from `.pdf`, `.docx`, and `.txt`, including **DOCX table
  contents** — the cells where contracts and tax forms keep the numbers that matter.
- **Split that text into overlapping, size-bounded chunks**, each carrying its source
  identifier, position in the document, and character offsets.
- **Skip and log** unreadable files rather than aborting the run, so one corrupt PDF
  does not kill the whole scan.
- **Refuse to start in personal mode**, explaining what that mode will eventually do
  rather than failing obscurely.

Then build two indexes over those chunks:

- **Embed every chunk locally** with `bge-base-en-v1.5` through
  `sentence-transformers` — 768 dimensions, no external API, nothing leaves the
  machine.
- **Measure chunk boundaries with the model's own tokenizer**, and refuse to build
  at all if the configured chunk size exceeds what the model can read. Embedding
  models truncate oversized input silently, so this failure has to be made loud.
- **Store vectors in Chroma** under cosine distance, with deterministic chunk ids so
  a rebuild replaces the previous index rather than accumulating duplicates.
- **Store a BM25 keyword index** as plain JSON — inspectable, and portable across
  library versions in a way a pickle would not be.

And search them:

- **Query both indexes** for a shortlist of candidates each.
- **Fuse the two lists by rank** with reciprocal rank fusion, so neither index's
  score scale can drown out the other.
- **Rerank the shortlist with a cross-encoder** (`ms-marco-MiniLM-L6-v2`, local),
  which reads the query and each passage together. Vector similarity alone is never
  the final answer.

And serve it:

- **A FastAPI search endpoint** — `POST /search`, `GET /health`, and interactive
  documentation at `/docs`, generated from the request and response schemas.
- **Refuse to start rather than start broken** — every configuration or index
  problem is reported as a message before the port is ever bound.

And put a page in front of it:

- **A Streamlit search page** — examples to click, highlighted matches, and a
  plain-language line on *why* each passage was found.
- **One command** — `python scripts/run_local.py` starts the API and the page, and
  Ctrl+C stops both.

566 tests cover this, and the default run is offline in about twenty seconds.

## What is planned

Nothing further is scheduled for v1 — it is feature-complete as a local demo.

Out of scope for v1: any chat or LLM answer-synthesis layer. This is search — it
returns passages, not generated answers. Personal mode — searching your own files
through a connected cloud drive — is also a future release, not part of v1.

---

## Modes

| Mode | Status | What it searches |
|---|---|---|
| `demo` | **Working**, and the default | The documents in `data/sample_docs/`, committed to this repo |
| `personal` | **Not implemented** | Would search your own files through a connected cloud drive |

Setting `APP_MODE=personal` makes the app refuse to start, with:

> This build is demo-only and searches a fixed set of sample documents bundled with
> the app. A future release will add personal mode, letting you connect your own
> cloud drives (Google Drive, OneDrive) and search your own files.

There is no cloud connector and no credential handling in this build at all. It
cannot reach your files — hosted or local — because it only ever reads its own
sample corpus.

---

## Requirements

- **Python 3.11+** — the code uses `typing.Self` and `datetime.UTC`.
  Developed and tested on 3.14.7.
- Roughly 3 GB of disk for dependencies (`sentence-transformers` pulls in PyTorch).

## Setup

```bash
git clone https://github.com/<you>/ContextualDocSearch.git
cd ContextualDocSearch

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
```

Every setting has a working default, so `.env` is optional until you want to change
chunk sizes. See [`.env.example`](.env.example) for what is tunable.

## Building the indexes

The sample documents are already in `data/sample_docs/`, so there is nothing to add:

```bash
python scripts/build_index.py
```

The first run downloads the embedding model (~440 MB) into the Hugging Face cache;
later runs reuse it. Output:

```
corpus : .../data/sample_docs
model  : BAAI/bge-base-en-v1.5
         limit 512 tokens, 768 dims

individual-svcs-agrmnt.docx       38 chunks
Sample Contract.docx              24 chunks
Texas-Sublease-Agreement.pdf       4 chunks
Warranty Forms.pdf                 3 chunks

chunks : 69  (largest 480 tokens)
```

Then query each index — a diagnostic, not the search API:

```bash
python scripts/build_index.py --verify "how long is the warranty"
```

```
  vector index (cosine similarity, 1.0 = identical)
      0.580  Warranty Forms.pdf#1  GUARANTEE/WARRANTY for ______________ We hereby...
      0.535  Warranty Forms.pdf#0  GUARANTEE/WARRANTY FORM FOR EQUIPMENT OR COMPO...

  keyword index (BM25 score, corpus-relative)
      7.856  individual-svcs-agrmnt.docx#33  THIS IS AN EXAMPLE ONLY. Please contact...
      7.719  Warranty Forms.pdf#0  GUARANTEE/WARRANTY FORM FOR EQUIPMENT OR COMPO...
```

That output is a compact argument for why this project is hybrid. The vector index
gets the warranty document right from a question sharing almost no words with it.
BM25's top hit is an unrelated clause, dragged there by "how", "long" and "the" —
but BM25 is the half that will match `INV-2024-88213` exactly, which embeddings blur
into every other reference number. Search fuses the two so each covers the other's
blind spot.

## Using the search page

```bash
python scripts/run_local.py
```

This starts the search API, waits until it has loaded its models (about ten
seconds), and opens the search page at `http://127.0.0.1:8501`. If an API is
already running it is reused and left running when you quit. Ctrl+C stops
everything the command started. If the API cannot start — no index, personal mode,
invalid settings — you get its message and the page never opens. If port 8000 or
8501 is already taken by another program, it says so straight away, before any
model loads; `API_PORT` and `UI_PORT` in `.env` move them.

What is on the page:

- **Search box** with a results selector (3, 5 or 10). Enter or **Search** runs it.
- **Example buttons** — "How long is the warranty?", "Who pays for utilities?",
  "FERPA", "GA-48" and more — chosen to show both halves of hybrid search: questions
  answered by *meaning*, and codes found by *exact term*.
- **One card per result**: the document, which passage of it ("passage 2 of 3"), the
  passage itself with your query's words highlighted, and a **Why** line in plain
  words — "meaning match #1 · keyword match #7" — saying which half of the search
  found it and where each placed it.
- **Details** under each card: the raw scores, with a reminder that they only rank
  results within one search and are not a confidence measure. That is why they are
  not on the card itself.
- **Sidebar**: whether the search service is connected, the four documents with
  their passage counts, a short "How search works", and the models in use.
- **A plain footer** under the results: these are the closest passages in the demo
  documents, and the answer may not be among them. Search always returns its best
  candidates, even for questions the documents cannot answer.

If the search service stops, the page says so and shows the command to start it,
rather than an error trace.

**Three Streamlit defaults are switched off**, because they break this project's
promises. Out of the box, Streamlit sends usage statistics to its developers,
listens on every network interface, and stops its first launch to ask for an email
address. `run_local.py` overrides all three on the command line, and
`.streamlit/config.toml` does the same for anyone running
`streamlit run ui/streamlit_app.py` by hand from the repo root.

## Searching from the command line

```bash
python scripts/search.py "how long is the warranty"
python scripts/search.py "termination notice period" -k 3
```

The first search downloads the reranker (~90 MB); after that both models load from
the local cache with no network traffic at all.

```
query  : 'how long is the warranty'

   1   -0.680  Warranty Forms.pdf#1               vec  #1  bm25  #7  GUARANTEE/WARRANTY for ______...
   2   -4.159  individual-svcs-agrmnt.docx#6      vec #11  bm25   -  If University Records are sub...
   3   -4.513  Sample Contract.docx#3             vec  #5  bm25   -  Contractor will submit invoic...
```

`score` is the reranker's relevance judgement — higher is better, but the numbers
only mean something relative to each other within one query. `vec` and `bm25` show
where each index placed the chunk before fusion (`-` means that index did not find
it). BM25's #1 from the diagnostic above is gone: vector search did not rate it, so
fusion ranked it low and the reranker agreed.

A warm search takes about one second on a laptop CPU, almost all of it the
cross-encoder. `RETRIEVAL_CANDIDATES` (default 20 per index) is the lever: fewer
candidates, faster reranking, more risk of missing the right chunk.

Search refuses — with a message and exit code 1, before loading either model —
when the index is missing or unreadable, when it was built with a different
`EMBEDDING_MODEL_NAME` than the one configured, or when the two indexes disagree
(an interrupted build). A query vector from one embedding model is meaningless
against vectors from another, and two models of the same width would otherwise
return confident nonsense without any error. Rebuilding fixes all of these.

### Why this reranker

Three cross-encoders were measured on 12 queries whose answer chunk can be picked
out by its text:

| Reranker | Top result holds the answer | Warm search (CPU) | Download |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` (default) | 12/12 | 0.65–1.3 s | 90 MB |
| `BAAI/bge-reranker-base` | 10/12 | 4.6–7.8 s | 1.1 GB |
| `BAAI/bge-reranker-v2-m3` | 12/12 | 13–29 s | 2.3 GB |

`bge-reranker-base` tokenizes differently and counts the largest chunk as 544
tokens, past its 512 limit, so the ends of long chunks go unread. `v2-m3` reads up
to 8192 tokens and never truncates, but is only practical with a GPU. The set is
small enough that the default and `v2-m3` tie; it separates bad from good, not good
from better. `RERANKER_MODEL_NAME` switches models without a rebuild — the reranker
reads chunk text, not stored vectors.

## Running the API

```bash
python scripts/serve.py                 # http://127.0.0.1:8000
python scripts/serve.py --port 8080
python scripts/serve.py --host 0.0.0.0  # reachable from other machines — deliberately opt-in
```

Startup opens the index and loads both models before the port is bound — about ten
seconds — so a server that is running can always search. Every problem search would
refuse is reported instead, and the server does not start:

| Problem | Exit code |
|---|---|
| Index missing, unreadable, or built with another embedding model | 1 |
| Personal mode, invalid settings, or a chunk size the model cannot read | 2 |

Open `http://127.0.0.1:8000` in a browser for the interactive documentation, or:

```bash
curl -X POST http://127.0.0.1:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "how long is the warranty", "top_k": 3}'
```

```json
{
  "query": "how long is the warranty",
  "took_ms": 1280,
  "results": [
    {"rank": 1, "chunk_id": "Warranty Forms.pdf::1", "source_id": "Warranty Forms.pdf",
     "chunk_index": 1, "text": "GUARANTEE/WARRANTY for ...", "score": -0.68,
     "rrf_score": 0.0313, "vector_rank": 1, "keyword_rank": 7}
  ]
}
```

- `query` is required, at most 500 characters, and may not be blank. `top_k` is
  optional (1–100, default `SEARCH_TOP_K`). Anything else — including a misspelt
  field — is a `422` naming the problem. The error says where and what, but never
  echoes the rejected value back.
- Request bodies over 16 KiB are refused with `413` before they are read. The
  largest valid request is about 6 KB, and neither uvicorn nor FastAPI limits body
  size on its own.
- `GET /health` returns the number of searchable chunks and the models in use.
- **Search is `POST`, not `GET`, on purpose.** Anything in a URL is written to the
  server's access log — even a request the server rejects — so a `GET` API would log
  every query in plain text. Queries travel in the body and are never logged.
- Searches run one at a time. One search already keeps every CPU core busy;
  measured, running them in parallel gained only about 15%.
- **Restart the server after rebuilding the index.** The keyword index is read into
  memory at startup.

`uvicorn app.api.main:app`, run from the repo root, also works. It prints the same
refusal messages, but uvicorn follows them with its own traceback; `serve.py` is the
clean path.

## Trying the ingestion pipeline directly

Chunking without touching the index. Save this at the **repo root** and run it from
there — the project is not yet pip-installable, so `app` is only importable when the
repo root is your working directory:

```python
from app.ingestion.chunking import chunk_text
from app.ingestion.connectors.local_fs import LocalFSConnector
from app.ingestion.loaders import DocumentLoadError, extract_text

# No argument: the connector defaults to the bundled demo corpus.
connector = LocalFSConnector()

for raw_bytes, metadata in connector.iter_documents():
    try:
        text = extract_text(raw_bytes, metadata)
    except DocumentLoadError as exc:
        print(f"skipped {metadata.title}: {exc}")
        continue

    chunks = list(chunk_text(text, source_id=metadata.source_id))
    print(f"{metadata.title}: {len(chunks)} chunks")
```

## Tests

```bash
pytest                    # 566 tests, about twenty seconds, no network required
pytest -m integration     # 7 more that load the real models and read the built index
```

The default run is offline. Chunking tests use a word-based token estimate rather
than a real tokenizer, and the embedding and reranker tests run against stubs injected
into `sys.modules` — which works because both model classes import
`sentence_transformers` inside `__init__` rather than at module scope. Vector-store
and retrieval tests use real Chroma and real BM25, so ranking behaviour is genuinely
covered rather than mocked; only the two models are faked. The search page is
tested headlessly with Streamlit's `AppTest`, against a fake API client.

Every source file is also parsed with the Python 3.11 grammar, so syntax newer than
the supported minimum cannot slip in unnoticed from a 3.14 development machine.

The integration run includes the regression that motivated hybrid search: it asserts
that BM25 alone gets "how long is the warranty" wrong, and that the full pipeline
gets it right.

Any change to chunking or retrieval logic needs a corresponding test.

---

## How it fits together

```
app/
├── config.py                    working — typed settings (pydantic-settings, .env)
├── modes.py                     working — demo/personal modes, and the message
├── cli.py                       working — shared command-line helpers
├── model_cache.py               working — cache-first model loading, no network once cached
├── ingestion/
│   ├── document.py              working — DocumentMetadata
│   ├── connectors/
│   │   └── local_fs.py          working — recursive walk, skip rules
│   ├── loaders.py               working — PDF/DOCX/TXT to plain text
│   └── chunking.py              working — overlapping, size-bounded chunks
├── indexing/
│   ├── built_index.py           working — opens the built indexes, refuses stale ones
│   ├── embeddings.py            working — bge-base behind a TextEmbedder Protocol
│   ├── hits.py                  working — IndexHit, the shape both indexes return
│   ├── keyword_index.py         working — BM25, persisted as JSON
│   └── vector_store.py          working — Chroma, cosine, deterministic ids
├── retrieval/
│   ├── fusion.py                working — reciprocal rank fusion
│   ├── reranker.py              working — cross-encoder behind a Reranker Protocol
│   ├── results.py               working — SearchResult
│   └── hybrid_retriever.py      working — both indexes → fusion → reranking
├── api/
│   ├── schemas.py               working — request/response models (the HTTP contract)
│   ├── body_limit.py            working — refuses request bodies over 16 KiB
│   └── main.py                  working — FastAPI app: /search, /health, /documents
└── ui/
    ├── client.py                working — typed API client used by the page
    └── text.py                  working — safe markdown, highlighting, the "why" line
ui/streamlit_app.py              working — the search page (layout only)
scripts/build_index.py           working — index builder + --verify diagnostic
scripts/search.py                working — command-line search
scripts/serve.py                 working — runs the API
scripts/run_local.py             working — runs the API and the page together
.streamlit/config.toml           Streamlit privacy settings for a hand-run page
data/sample_docs/                the demo corpus (4 documents)
data/index/                      built artifacts (gitignored, rebuildable)
```

Indexing and search are two straight lines:

```
LocalFSConnector → (bytes, DocumentMetadata) → extract_text() → str
    → chunk_text(count_tokens=model.count_tokens) → Chunk
        → embed_documents() → VectorStore
        → KeywordIndex

query ─┬─ embed_query() → VectorStore.query() ─┐
       └─ KeywordIndex.query() ────────────────┴─ reciprocal_rank_fusion()
            → Reranker.score() → SearchResult
```

### Design decisions worth knowing

**No source abstraction, deliberately.** An earlier design put a `SourceConnector`
interface in front of every source so local files and cloud drives would look
identical. With the scope cut to demo-only there is exactly one source, so
`LocalFSConnector` is a plain concrete class and the interface was removed rather
than kept as ceremony. `DocumentMetadata.source_id` keeps its generic name instead
of becoming `path`, so it stays accurate if personal mode brings a second source
back.

**Chunks overlap by about 15%.** A hard cut lands mid-thought. If one chunk ends
with "…the manufacturer's warranty expires" and the next begins "after 24 months…",
neither answers the question alone. Overlap repeats the tail so the complete
statement survives intact somewhere.

**Token counting is pluggable.** `chunk_text()` accepts a `count_tokens` callable,
defaulting to a word-based estimate (about 1.3 tokens per word). That keeps
`chunking.py` free of any model dependency and the tests offline, while
`build_index.py` passes the model's real tokenizer so the boundaries that end up in
the index are measured in the units the model reads. The estimate is not good enough
for that job: it under-counts by roughly 12% on mixed prose and by 44–74% on table
rows, dates and reference numbers — exactly the chunks holding the facts a query is
usually after.

**The truncation trap, and why the guard sits where it does.** Embedding models have
hard input limits and **silently truncate** past them — no error, just missing text.
`bge-base-en-v1.5` caps at 512, so `CHUNK_MAX_TOKENS` defaults to 480, leaving room
for the special tokens the model adds (a 480-token chunk arrives as 482).
`EmbeddingModel` refuses to construct if the configured ceiling exceeds the model's
limit. That check lives there rather than in `Settings` deliberately: reading
`max_seq_length` means loading the model, and reading configuration should never
trigger a 440 MB download.

**The two indexes are not score-comparable, on purpose.** Cosine similarity sits in
roughly [-1, 1]; BM25 is unbounded and depends on the corpus. Nothing normalises one
against the other, because reciprocal rank fusion works on each hit's *rank* within
its own list and never compares the raw numbers.

**Retrieval is two stages with two jobs.** The indexes and fusion are cheap and aim
for recall: get the right chunk into a shortlist of 20–40. The cross-encoder is
accurate but runs once per (query, chunk) pair on every search, so it only ever sees
that shortlist. There is one path through `HybridRetriever.search()` — no branch
returns unfused or unreranked results.

**The reranker shares its 512-token window between query and passage.** A 480-token
chunk plus a typical query fits. A query longer than about 29 tokens pushes the end
of the largest chunks out of view — measured: the tokenizer trims the passage and
keeps the query. That affects the rerank score only, never the index, so it is
documented rather than refused.

---

## Known issues

- **DOCX loses document order.** Paragraphs are extracted before tables rather than
  interleaved as they appear. Content is preserved; position is not.
- **Scanned PDFs yield nothing.** Text extraction only, no OCR. An image-only PDF
  returns empty text without raising.
- **Not pip-installable.** There is no `pyproject.toml`, so `import app` only works
  with the repo root as the working directory. Tests pass only because the root
  `conftest.py` puts it on `sys.path`, and both scripts add it themselves.
  Configured *data* paths are unaffected: `sample_docs_path`, `vector_store_path` and
  `.env` resolve against the repo root, so the script works from any directory.
- **No upper version bounds** in `requirements.txt`, so a future breaking release of
  a dependency can break a fresh install.
- **AES-encrypted PDFs are skipped.** Encrypted PDFs that open without a password —
  forms locked only against printing or editing — are read, but only when they use
  the older RC4 scheme. AES needs the optional `cryptography` package, which is
  deliberately not installed; those files are skipped with a message saying so.
  PDFs that genuinely require a password are always skipped.
- **Garbled PDFs extract silently.** A PDF with no ToUnicode CMap yields mojibake
  instead of text, and `extract_text()` does not raise — so skip-and-log never fires
  and the junk reaches the index. One sample document hit this and was removed from
  the corpus. Still undetected; worth fixing before the corpus grows.
- **Only one in-memory vector store may be alive per process.** Chroma caches its
  in-memory system, so constructing a second `VectorStore.in_memory()` drops the
  collection the first is holding. Use `VectorStore.open()` with separate directories
  when two must coexist. Documented on the method and pinned by a test.
- **Every search returns results, even for nonsense.** Vector search always finds
  *something* nearest, so there is no "no good match" answer yet. The reranker's
  scores cannot supply one: they only rank results within a single query. The
  correct chunk for "who pays for heat and other utilities" scores −9.15, *below*
  the best hit for `INV-2024` (−8.12), which appears nowhere in the corpus.
- **The API must be restarted after an index rebuild.** The keyword index is held in
  memory, so a running server keeps serving the corpus it started with.
- **A force-killed launcher leaves the app running.** Ctrl+C, or closing the
  terminal window, stops everything `run_local.py` started — Windows delivers those
  to every process in the console. Ending the launcher alone from Task Manager skips
  its clean-up, so the API and page keep running on ports 8000 and 8501; end the
  `python` processes too. The next `run_local.py` then reports the port as taken.
- **No CI**, and no `LICENSE` file yet — see below.

## A note on privacy

The app reads the documents bundled in `data/sample_docs/` and nothing else. There
is no cloud connector, no credential handling, and no upload path in this build, so
it cannot reach your own files whether you run it locally or deploy it.

Everything stays on your machine: embeddings are computed locally through
`sentence-transformers`, with no external API calls. The only network access is the
one-time model download from Hugging Face on the first index build. After that the
model is loaded strictly from the local cache (`local_files_only=True`) — left to its
defaults, the Hugging Face client would otherwise send dozens of requests to
huggingface.co on every load to check the cached files are current. Chroma's usage
telemetry is switched off explicitly. `.gitignore` covers `.env` and the built index.

The search API listens on `127.0.0.1` unless you choose otherwise, and takes queries
in the request body rather than the URL, so what you search for never reaches the
server's access log. The search page also listens only on `127.0.0.1`, with
Streamlit's usage statistics switched off.

## License

Not yet licensed. Until a `LICENSE` file is added, default copyright applies and no
permission is granted for reuse.
