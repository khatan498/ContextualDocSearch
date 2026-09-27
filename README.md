# ContextualDocSearch

Hybrid (keyword + vector) search over a small set of sample documents bundled with
the repo — contracts, service agreements, a sublease, and warranty forms.

Ask *"how long is the warranty"* and get back the paragraph that says
*"coverage period: 24 months from date of purchase"* — even though it shares no
keywords with the question.

**This is a demo.** It searches the documents in `data/sample_docs/` and nothing
else. Connecting your own files is a planned future release — see [Modes](#modes).

> ## Work in progress — no search API yet
>
> Ingestion and indexing work and are tested. Both indexes build over the sample
> documents, and `scripts/build_index.py --verify` will query them — but that is a
> diagnostic that reports each index separately, not search. The layer that merges
> and reranks them does not exist yet, and there is no API or UI. Several files in
> the tree are deliberately empty placeholders.
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

202 tests cover this, and the default run is offline in about two seconds.

## What is planned

- **Hybrid retrieval** — merge keyword and vector results through reciprocal rank
  fusion, then rerank with a cross-encoder. Vector similarity alone is never the
  final answer.
- **A search API and UI** — a FastAPI endpoint and a Streamlit front end.

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
into every other reference number. Phase 3 fuses the two so each covers the other's
blind spot.

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
pytest                    # 202 tests, about 2 seconds, no network required
pytest -m integration     # 2 more that load the real model
```

The default run is offline. Chunking tests use a word-based token estimate rather
than a real tokenizer, and the embedding tests run against a stub injected into
`sys.modules` — which works because `EmbeddingModel` imports `sentence_transformers`
inside `__init__` rather than at module scope. Vector-store tests use real Chroma
through an in-memory client, so distance semantics are genuinely covered rather than
mocked.

Any change to chunking or retrieval logic needs a corresponding test.

---

## How it fits together

```
app/
├── config.py                    working — typed settings (pydantic-settings, .env)
├── modes.py                     working — demo/personal modes, and the message
├── ingestion/
│   ├── document.py              working — DocumentMetadata
│   ├── connectors/
│   │   └── local_fs.py          working — recursive walk, skip rules
│   ├── loaders.py               working — PDF/DOCX/TXT to plain text
│   └── chunking.py              working — overlapping, size-bounded chunks
├── indexing/
│   ├── embeddings.py            working — bge-base behind a TextEmbedder Protocol
│   ├── hits.py                  working — IndexHit, the shape both indexes return
│   ├── keyword_index.py         working — BM25, persisted as JSON
│   └── vector_store.py          working — Chroma, cosine, deterministic ids
├── retrieval/                   planned — rank fusion, reranking
└── api/main.py                  planned — search endpoint
ui/streamlit_app.py              planned — search UI
scripts/build_index.py           working — index builder + --verify diagnostic
data/sample_docs/                the demo corpus (4 documents)
data/index/                      built artifacts (gitignored, rebuildable)
```

The working part of the pipeline is a straight line:

```
LocalFSConnector → (bytes, DocumentMetadata) → extract_text() → str
    → chunk_text(count_tokens=model.count_tokens) → Chunk
        → embed_documents() → VectorStore
        → KeywordIndex
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

---

## Known issues

- **DOCX loses document order.** Paragraphs are extracted before tables rather than
  interleaved as they appear. Content is preserved; position is not.
- **Scanned PDFs yield nothing.** Text extraction only, no OCR. An image-only PDF
  returns empty text without raising.
- **Not pip-installable.** There is no `pyproject.toml`, so `import app` only works
  with the repo root as the working directory. Tests pass only because the root
  `conftest.py` puts it on `sys.path`, and `scripts/build_index.py` adds it itself.
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

## License

Not yet licensed. Until a `LICENSE` file is added, default copyright applies and no
permission is granted for reuse.
