# ContextualDocSearch

Hybrid (keyword + vector) search over a small set of sample documents bundled with
the repo — contracts, service agreements, a sublease, and warranty forms.

Ask *"how long is the warranty"* and get back the paragraph that says
*"coverage period: 24 months from date of purchase"* — even though it shares no
keywords with the question.

**This is a demo.** It searches the documents in `data/sample_docs/` and nothing
else. Connecting your own files is a planned future release — see [Modes](#modes).

> ## Work in progress — not usable as a search engine yet
>
> Document ingestion works and is tested. Nothing is embedded, indexed, or
> searchable yet, and there is no API or UI. Several files in the tree are
> deliberately empty placeholders.
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

100 tests cover this, and they run offline in about a second.

## What is planned

- **Fix the known chunking issues** below and add regression tests for them.
- **Embeddings and indexing** — local sentence-transformers embeddings, a Chroma
  vector store, and a BM25 keyword index built by `scripts/build_index.py`.
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

## Trying the ingestion pipeline

This is the only working entry point right now. The sample documents are already in
`data/sample_docs/`, so there is nothing to add. Save the following at the **repo
root** and run it from there — the project is not yet pip-installable, so `app` is
only importable when the repo root is your working directory:

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
pytest              # 100 tests, about 1 second, no network required
```

Chunking tests use a word-based token estimate rather than a real tokenizer, which
keeps the suite offline and fast. Any change to chunking or retrieval logic needs a
corresponding test.

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
├── indexing/                    planned — embeddings, vector store, BM25
├── retrieval/                   planned — rank fusion, reranking
└── api/main.py                  planned — search endpoint
ui/streamlit_app.py              planned — search UI
scripts/build_index.py           planned — index builder
data/sample_docs/                the demo corpus (4 documents)
```

The working part of the pipeline is a straight line:

```
LocalFSConnector → (bytes, DocumentMetadata) → extract_text() → str → chunk_text() → Chunk
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
defaulting to a word-based estimate (about 1.3 tokens per word). This keeps
ingestion free of any model dependency and the tests offline. The embedding layer
can later pass the real model's tokenizer without touching `chunking.py`.

**The truncation trap.** Embedding models have hard input limits and **silently
truncate** past them — no error, just missing text. `all-MiniLM-L6-v2` caps at 256
tokens, `bge-base-en-v1.5` at 512. The default 500–800 token chunks would be cut off
by either. Chunk sizes are configurable for exactly this reason, and the embedding
layer must assert `chunk_max_tokens <= model.max_seq_length` at startup.

---

## Known issues

Found by randomised invariant testing. The existing unit tests do not catch these.

- **Chunks can exceed `max_tokens`,** by up to roughly double. After a chunk closes,
  the next is seeded with the overlap tail, but `max_tokens` is not re-checked before
  the following segment is appended. Combined with `_overlap_tail()` taking its first
  segment regardless of the overlap budget, a 300-token limit has produced a
  521-token chunk. This feeds directly into the truncation trap described above.
- **Duplicate chunks.** The same mechanism can emit a chunk whose content is entirely
  contained in its predecessor — wasted index space, and the same text competing with
  itself in results.
- **DOCX loses document order.** Paragraphs are extracted before tables rather than
  interleaved as they appear. Content is preserved; position is not.
- **Scanned PDFs yield nothing.** Text extraction only, no OCR. An image-only PDF
  returns empty text without raising.
- **Not pip-installable.** There is no `pyproject.toml`, so `import app` only works
  with the repo root as the working directory. Tests pass only because the root
  `conftest.py` puts it on `sys.path`.
- **No upper version bounds** in `requirements.txt`, so a future breaking release of
  a dependency can break a fresh install.
- **Garbled PDFs extract silently.** A PDF with no ToUnicode CMap yields mojibake
  instead of text, and `extract_text()` does not raise — so skip-and-log never fires
  and the junk would reach the index. One sample document hit this and was removed
  from the corpus. Detecting it is a Phase 2 follow-up.
- **No CI**, and no `LICENSE` file yet — see below.

## A note on privacy

The app reads the documents bundled in `data/sample_docs/` and nothing else. There
is no cloud connector, no credential handling, and no upload path in this build, so
it cannot reach your own files whether you run it locally or deploy it.

Everything stays on your machine regardless: embeddings will run locally through
`sentence-transformers`, with no external API calls. `.gitignore` covers `.env` and
the built index.

## License

Not yet licensed. Until a `LICENSE` file is added, default copyright applies and no
permission is granted for reuse.
