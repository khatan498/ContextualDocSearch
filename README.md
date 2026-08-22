# ContextualDocSearch

Hybrid (keyword + vector) search over your own documents — local files today,
Google Drive later. Built for the kind of paperwork that actually piles up:
contracts, tax documents, warranties, manuals.

Ask *"how long is the warranty on the dishwasher"* and get back the paragraph that
says *"coverage period: 24 months from date of purchase"* — even though it shares no
keywords with the question.

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

Point it at a folder and it will:

- **Walk the directory tree** recursively, skipping hidden files and directories
  (including Windows hidden-attribute files), unsupported types, and anything over a
  configurable size limit.
- **Extract plain text** from `.pdf`, `.docx`, and `.txt`, including **DOCX table
  contents** — the cells where contracts and tax forms keep the numbers that matter.
- **Split that text into overlapping, size-bounded chunks**, each carrying its source
  identifier, position in the document, and character offsets.
- **Skip and log** unreadable files rather than aborting the run, so one corrupt PDF
  does not kill a scan of ten thousand documents.

73 tests cover this, and they run offline in about a second.

## What is planned

- **Fix the known chunking issues** below and add regression tests for them.
- **Embeddings and indexing** — local sentence-transformers embeddings, a Chroma
  vector store, and a BM25 keyword index built by `scripts/build_index.py`.
- **Hybrid retrieval** — merge keyword and vector results through reciprocal rank
  fusion, then rerank with a cross-encoder. Vector similarity alone is never the
  final answer.
- **A search API and UI** — a FastAPI endpoint and a Streamlit front end.
- **Google Drive as a source**, alongside the local filesystem.

Out of scope for v1: any chat or LLM answer-synthesis layer. This is search — it
returns passages, not generated answers.

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
chunk sizes or wire up Google Drive. See [`.env.example`](.env.example) for what is
tunable.

## Trying the ingestion pipeline

This is the only working entry point right now. Drop a few `.pdf`, `.docx`, or
`.txt` files into `data/sample_docs/`, then save the following at the **repo root**
and run it from there — the project is not yet pip-installable, so `app` is only
importable when the repo root is your working directory:

```python
from pathlib import Path

from app.ingestion.chunking import chunk_text
from app.ingestion.connectors.local_fs import LocalFSConnector
from app.ingestion.loaders import DocumentLoadError, extract_text

connector = LocalFSConnector(Path("data/sample_docs"))

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
pytest              # 73 tests, about 1 second, no network required
```

Chunking tests use a word-based token estimate rather than a real tokenizer, which
keeps the suite offline and fast. Any change to chunking or retrieval logic needs a
corresponding test.

---

## How it fits together

```
app/
├── config.py                    working — typed settings (pydantic-settings, .env)
├── ingestion/
│   ├── connectors/
│   │   ├── base.py              working — SourceConnector ABC + DocumentMetadata
│   │   ├── local_fs.py          working — recursive walk, skip rules
│   │   └── google_drive.py      planned
│   ├── loaders.py               working — PDF/DOCX/TXT to plain text
│   └── chunking.py              working — overlapping, size-bounded chunks
├── indexing/                    planned — embeddings, vector store, BM25
├── retrieval/                   planned — rank fusion, reranking
└── api/main.py                  planned — search endpoint
ui/streamlit_app.py              planned — search UI
scripts/build_index.py           planned — index builder
```

The working part of the pipeline is a straight line:

```
LocalFSConnector → (bytes, DocumentMetadata) → extract_text() → str → chunk_text() → Chunk
```

### Design decisions worth knowing

**Every source hides behind `SourceConnector`.** Connectors yield
`(bytes, DocumentMetadata)` and nothing else. `DocumentMetadata.source_id` is
deliberately not named `path`: Drive has opaque file ids and no paths, and calling
it `path` would bake a filesystem assumption into the shared interface.

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
- **No CI**, and no `LICENSE` file yet — see below.

## A note on privacy

Connectors read whatever directory you point them at, and everything stays on your
machine: embeddings run locally through `sentence-transformers`, with no external API
calls. `.gitignore` covers `.env`, credential files, and the built index.

If this is ever deployed publicly it must only ever index `data/sample_docs/` — never
real local paths or Drive credentials.

## License

Not yet licensed. Until a `LICENSE` file is added, default copyright applies and no
permission is granted for reuse.
