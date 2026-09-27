# CLAUDE.md

## Project Overview
A hybrid (keyword + vector) search demo over a fixed set of sample documents
bundled with the repo (contracts, agreements, warranty forms). Anyone can run it
and search that corpus — that is the whole delivered feature set. Search-only:
no chat/answer-synthesis layer.

The app also names a "personal" mode that is deliberately NOT implemented.
Selecting it fails at startup with app.modes.PERSONAL_MODE_MESSAGE, explaining
that a future release will let users connect their own cloud drives (Google
Drive, OneDrive) to search their own files. Until then the app has no cloud
connector, no credential handling, and reads nothing outside its own corpus.

## Tech Stack
- Python 3.11+ (StrEnum, typing.Self, datetime.UTC); developed on 3.14
- Embeddings: sentence-transformers (local, no external API)
- Vector store: Chroma (dev), Pinecone/Qdrant (deployed)
- Keyword index: rank_bm25
- Reranking: sentence-transformers cross-encoder

## Developer Context
I'm an experienced developer (~7 years, strong in .NET/C# and some Java/Spring
Boot) but new to Python and to the AI/ML ecosystem (embeddings, vector stores,
RAG). Assume I understand software engineering concepts, OOP, and architecture
well — don't over-explain those. DO explain Python-specific idioms and any
AI/ML-specific library or concept the first time it appears.

When implementing:
- Add brief inline comments explaining Python idioms that differ from C#/Java
  (generators/yield, abstract base classes, decorators, context managers,
  list/dict comprehensions, type hints)
- After completing a file, give me a short plain-language summary of what it
  does and why, so I can learn as we go and explain it later
- If there's a Pythonic way and a more C#/Java-familiar way, use the Pythonic
  one but tell me what the familiar equivalent would be
- Prefer clarity over cleverness — this is a learning project

## Code Style
- Type hints on every function signature
- Docstrings in Google style
- Small, single-purpose modules — one class/concern per file

## Architecture Rules
- The app only ever reads sample_docs_path. LocalFSConnector defaults its root
    to it, so the rule holds by construction. No cloud connector, credential
    handling, upload path, or access to a user's own files may be added while the
    scope is demo-only
- Personal mode stays unimplemented. app/modes.py owns the single user-facing
  message; config rejects the mode at startup. Anything that has to explain the
  limitation uses PERSONAL_MODE_MESSAGE rather than inventing its own wording
- There is deliberately no SourceConnector abstraction — LocalFSConnector is the
  only source and is a plain concrete class. Do not reintroduce an interface
  until personal mode genuinely needs a second implementation
- Retrieval always merges BM25 + vector results via reciprocal rank fusion
  before reranking — never return raw vector-similarity results alone
- Chunk boundaries must be measured with the embedding model's own tokenizer
  in the indexing path — pass it as chunk_text(count_tokens=...). The built-in
  estimate_tokens (words x 1.3) is a dev/test default only

## Testing
- pytest, one test file per module in app/
- Any change to chunking or retrieval logic needs a corresponding test

## Things to Avoid
- No hardcoded file paths or API keys — use app/config.py and .env
- Don't add a chat/LLM-answer layer — this is v1, search-only

## Current Phase
Phase 2 complete — both indexes build over the demo corpus. Next: Phase 3 —
reciprocal rank fusion over the two result lists, then cross-encoder reranking.

What Phase 2 delivered:
- app/indexing/embeddings.py — bge-base-en-v1.5 behind a TextEmbedder Protocol,
  so tests inject a fake and stay offline
- app/indexing/keyword_index.py — BM25 persisted as JSON, rebuilt on load
- app/indexing/vector_store.py — Chroma, cosine, deterministic chunk ids
- app/indexing/hits.py — IndexHit, the shape both indexes return
- scripts/build_index.py — builds both; `--verify "query"` is a diagnostic that
  prints each index's hits separately, never a fused result

Both Phase 2 carry-forwards are done: build_index passes the model's real
tokenizer as chunk_text's count_tokens, and EmbeddingModel raises
ChunkWindowTooLargeError when chunk_max_tokens exceeds model.max_seq_length.
That check lives in EmbeddingModel rather than Settings deliberately — reading
max_seq_length means loading the model, and config must never trigger a 440 MB
download just to be read.

Measured on the real corpus: 4 documents, 69 chunks, largest 480 tokens (482
once the model adds special tokens, against its 512 limit).

Carried into Phase 3:
1. Fuse with RRF on *rank*, not score. Cosine similarity sits in [-1, 1] while
   BM25 is unbounded and corpus-relative; the two are deliberately not
   normalised against each other. app/indexing/hits.py explains this.
2. BM25 needs a real corpus to behave. BM25Okapi's IDF is
   log(N - n + 0.5) - log(n + 0.5), which is exactly 0 for a term appearing in
   one of two documents — so on tiny corpora every term scores zero and is
   filtered out. Write fusion tests against a corpus of four or more chunks.
3. Common query words drag BM25 badly. "how long is the warranty" ranks an
   unrelated clause above Warranty Forms.pdf on BM25 alone, while the vector
   index gets it right. This is the case fusion has to fix, and a good
   regression test for Phase 3.

Carried into Phase 4 (API) and Phase 5 (UI):
1. Startup refusals must reach the user as messages. PersonalModeUnavailableError,
   ChunkWindowTooLargeError and pydantic's ValidationError are deliberate, and
   each carries text written for a person. scripts/build_index.py main() shows
   the pattern: catch exactly those, print the message, exit 2; let anything
   else raise with its traceback. The API must not turn them into a bare 500.
2. Launch from anywhere is already safe. Data paths and .env resolve against
   app.config.PROJECT_ROOT, not the working directory, so uvicorn or Streamlit
   can start from any folder. Do not reintroduce CWD-relative paths.
3. The model loads offline once cached (local_files_only), and downloads only
   when absent. A fresh deployment still downloads ~440 MB on first start, so
   build the index — which caches the model — as part of deploying.

## Known Chunking Behaviours
Re-measured 2026-08-28 against the fixed packer and the 350/480 defaults, using
estimate_tokens. Current behaviour, not bugs — a baseline for retrieval tuning
in Phase 3. Changing any of these needs a test (see Testing).

Two defects that used to sit here were fixed at the start of Phase 2 and are now
covered by TestPackerRegressions in tests/test_chunking.py: chunks could reach
1.81x chunk_max_tokens, and a chunk could be emitted holding nothing but
carried-over overlap. Both came from re-seeding a chunk with the overlap tail
without re-checking what that tail cost or whether it held anything new.

Root cause of (2), (3) and (4): overlap is quantised to whole segments —
normally paragraphs — so it can never be finer-grained than what it is cut from.

1. Chunks settle near chunk_min_tokens, not chunk_max_tokens. The packer closes
   at the first paragraph break past min, so min is the attractor and max only a
   ceiling. Measured: 350/480 defaults over ~58-token paragraphs produced every
   full chunk at 359 tokens, never near 480.
   Lever: raise CHUNK_MIN_TOKENS to get larger chunks.

2. A chunk made of a single segment gets zero overlap. _overlap_tail returns []
   when len(segments) <= 1; that is required for forward progress, since a tail
   as long as the chunk would restart the next chunk at the same offset.
   Less prevalent under the tighter 480 ceiling than it was under 800: a
   paragraph over max is split into sentences, so the chunk spans several
   segments and overlap returns. Measured on 572-token paragraphs: 0/5
   boundaries overlapped at 500/800, but 5/11 at 350/480.
   Lever: keep CHUNK_MAX_TOKENS below the document's typical paragraph size, or
   CHUNK_MIN_TOKENS above it — either makes a chunk span more than one segment.

3. Where overlap does occur it can still exceed chunk_overlap_ratio. The tail's
   first segment is appended regardless of the budget, so one large paragraph is
   repeated whole. Measured: a 240-token budget produced a 575-token overlap,
   2.4x over. Treat the ratio as a target, not a cap — index size can run above
   the nominal +15%.

4. Overlap is dropped where it will not fit. When the carried-over tail leaves
   no room for the segment that forced the chunk to close, the tail is trimmed
   from the front and may be discarded entirely, so that boundary has reduced or
   no overlap. This is the trade the max_tokens guarantee is bought with:
   silently truncated text is worse than a boundary without overlap.

5. A chunk can close below chunk_min_tokens. min is a target and max a hard
   limit; when they conflict, max wins. If the chunk so far is under min but
   adding the next segment would breach max, it closes where it is. Measured on
   the real corpus (2026-09-27, real tokenizer): 10 of 69 non-final chunks sit
   under 350 tokens, the smallest at 124. Pinned by
   test_chunk_closes_below_min_when_the_next_segment_will_not_fit.
   Lever: more headroom between min and max makes it rarer; splitting the
   incoming segment to top the chunk up would remove it, at the cost of
   breaking a paragraph that currently survives whole.
