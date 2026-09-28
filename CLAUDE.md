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
Phase 5 complete — v1 is feature-complete as a local demo. Run it with
`python scripts/run_local.py`. It is not packaged for deployment; the user runs it
locally only.

What Phase 5 delivered:
- ui/streamlit_app.py — the page, layout only. Calls the API; loads no models.
  Takes its client from st.session_state["client"], which is how AppTest injects
  a fake. Search outcomes live in session_state so reruns redisplay, not
  re-search. Personal mode / invalid settings render as st.error, not a trace
- app/ui/client.py — SearchClient over httpx; parses responses with the same
  pydantic models the API uses (app/api/schemas.py), so the two cannot drift.
  Errors: ApiUnavailableError, SearchRejectedError (422), ApiError
- app/ui/text.py — escape_markdown (every ASCII punctuation char, which CommonMark
  allows), normalize_whitespace, highlight (whole words via keyword_index.tokenize,
  stopwords skipped), explain_match (the "why" line)
- GET /documents on the API — {source_id: chunks}, from
  KeywordIndex.chunks_per_source() via HybridRetriever.documents()
- scripts/run_local.py — reuses a running API or starts serve.py and waits on
  /health; passes serve.py's exit code through on refusal; starts Streamlit with
  --server.address 127.0.0.1, --browser.gatherUsageStats false,
  --server.showEmailPrompt false; on exit stops only what it started
- run_local.py also checks the page port, and the API port when it is not
  reusing an API, before starting anything: a clash is reported at once instead
  of after ~10 s of model loading followed by Streamlit's terse "Port 8501 is not
  available". Its progress lines use say() = print(flush=True), because
  redirected stdout is block-buffered (found end to end: the lines appeared only
  after shutdown). serve.py's prints are flushed for the same reason
- port_is_free() connects first, then binds. Bind alone is not enough on
  Windows: a program listening on 0.0.0.0 does not stop 127.0.0.1 being bound on
  the same port (measured), so a bind-only check said "free" and Streamlit
  silently shared the port. SO_EXCLUSIVEADDRUSE on the probe does not help (only
  the holder can set it). A free port costs ~0.5 s to confirm (Windows is slow to
  refuse local connections), under 1 s of a ~12 s startup
- .streamlit/config.toml — the same three overrides for a hand-run page
- tests/test_compatibility.py — parses every source file with the Python 3.11
  grammar (ast feature_version). Development is on 3.14, where 3.12+ syntax
  parses fine; Phase 5 briefly introduced a PEP 695 generic, caught in review

Why those Streamlit overrides: its defaults send usage statistics to Streamlit,
bind every network interface, and block the first launch on an email prompt.
Each breaks a project promise; each is pinned by a test.

Rendering rule: passage text is always escaped before st.markdown. Streamlit
markdown treats $ as LaTeX (contracts are full of dollar amounts), and
*, _, [..](..), :color[..] as formatting. Never pass unsafe_allow_html.

Score is never shown on a result card, only in its Details expander, with a
note that it is not confidence (Phase 3 carry-forward 2 still holds).

Windows process behaviour, measured end to end (2026-09-27):
- .venv\Scripts\python.exe is a stub that starts the real interpreter as a child
  (two PIDs per script). terminate() on the stub also ends that child — verified —
  so run_local's stop() is sufficient.
- A real Ctrl+C reaches every process on the console: launcher, API and page all
  stopped within ~4 s, nothing left behind.
- A force-killed launcher (Task Manager) skips its finally block, so the API and
  page keep running. Documented in README Known issues; a Windows job object with
  KILL_ON_JOB_CLOSE would close the gap if it ever matters.
- Testing gotcha: processes started from an agent or CI harness can inherit
  "Ctrl+C disabled" (SetConsoleCtrlHandler(NULL, TRUE) is inherited). A faithful
  Ctrl+C test must re-enable it in a wrapper inside a new console first.

Phase 4 — search is served over HTTP.

What Phase 4 delivered:
- app/api/schemas.py — pydantic request/response models, kept apart from the
  SearchResult dataclass. SearchRequest strips the query, rejects blank or
  >500-char queries and unknown fields (extra="forbid"); top_k 1-100
- app/api/main.py — create_app(retriever=None). POST /search, GET /health,
  GET / -> /docs. Searches run under a threading.Lock (measured: parallel
  searches gained ~15%, since one search already uses every core). The search
  endpoint is a plain `def` so FastAPI runs it on a worker thread. With no
  retriever supplied, the lifespan opens one at startup
- app/api/body_limit.py — pure-ASGI middleware refusing bodies over 16 KiB
  with 413, by Content-Length or by counting a chunked body as it arrives.
  Neither uvicorn nor Starlette caps body size; a 20 MB body was read in full
- A RequestValidationError handler in main.py drops pydantic's "input" field
  from 422s: the default echoed the whole rejected value (20 MB in, 20 MB out)
- scripts/serve.py — opens the retriever *before* uvicorn binds the port, so
  every refusal is a message and the server never starts half-working
- app/cli.py — run_with_refusals(), the single refusal-to-exit-code mapping
  used by all three scripts (index problems 1; configuration refusals 2);
  port_number for --port
- Settings: api_host (127.0.0.1 — loopback unless deliberately widened),
  api_port (8000)

Search is POST with a JSON body, not GET, deliberately: uvicorn's access log
writes every URL, even for rejected requests (verified — a GET with ?query=
was logged verbatim). Queries are never logged; test_api_main pins this.

Measured 2026-09-27: startup to first answer ~9.3 s; a warm search over HTTP
~1.05 s; four simultaneous searches all 200 and correct. HTTP results match
scripts/search.py exactly.

Carried into Phase 5 (UI):
1. The UI can call POST /search on a running serve.py, or import
   HybridRetriever directly. Either way, build one retriever and reuse it —
   open() takes ~7-9 s. If it calls the API, show 422 details and a clear
   "API not running" message rather than a stack trace.
2. The server must be restarted after rebuilding the index (the keyword index
   is held in memory). A UI that embeds the retriever has the same constraint.
3. Still no relevance floor (see Phase 3 carry-forward 2): nonsense queries
   return top_k results, and scores cannot separate them. Don't present score
   as a confidence.
4. vector_rank / keyword_rank make "why did this match?" answerable in the UI.
5. Starlette 1.3 warns that its TestClient's use of httpx is deprecated in
   favour of `httpx2`. Test-only and harmless today; switch when convenient.
6. `uvicorn app.api.main:app` works but follows a refusal message with
   uvicorn's own traceback (exit 3). serve.py is the clean path.

Phase 3 — search works end to end from the command line.

What Phase 3 delivered:
- app/retrieval/fusion.py — reciprocal_rank_fusion over named IndexHit lists;
  rank only, raw scores ignored; deterministic tie-break (best rank, then id)
- app/retrieval/reranker.py — CrossEncoderReranker (ms-marco-MiniLM-L6-v2)
  behind a Reranker Protocol; score() returns scores in input order, unsorted
- app/retrieval/results.py — SearchResult, carrying vector_rank/keyword_rank
- app/retrieval/hybrid_retriever.py — HybridRetriever.open()/search(); one path,
  no branch skips fusion or reranking
- app/indexing/built_index.py — open_built_indexes(), the single gate every
  reader of the indexes goes through. Raises IndexUnavailableError when the
  index is missing, unreadable, empty, built with a different embedding model,
  or when the two indexes disagree on chunk count (an interrupted build). All
  checks run before any model loads
- The vector store records embedding_model (and embedding_dim) in its
  collection metadata; replace_all(model_name=...) is required. An index built
  before this change has no recorded model and is refused until rebuilt
- app/model_cache.py — load_cache_first, shared by both models (0 HTTP requests
  on a cached load, measured for both)
- app/cli.py — positive_int for -k, and silence_model_loading_bars()
- scripts/search.py — CLI search; blank query and -k < 1 are usage errors
  (exit 2); IndexUnavailableError -> exit 1; PersonalModeUnavailableError,
  ChunkWindowTooLargeError, ValidationError -> exit 2
- Settings: reranker_model_name, retrieval_candidates (20 per index), rrf_k (60),
  search_top_k (5, must be <= retrieval_candidates), keyword_index_path property

Measured on the real corpus (2026-09-27): the fused shortlist is 20-36 of 69
chunks; a warm search takes 0.65-1.3 s on CPU, nearly all of it reranking.
"how long is the warranty" now returns Warranty Forms.pdf first — pinned by an
integration test that also asserts BM25 alone gets it wrong.

Reranker choice, measured 2026-09-27 on 12 queries whose answer chunk can be
identified by its text (top-1 chunk contains the answer / warm CPU latency):
- cross-encoder/ms-marco-MiniLM-L6-v2: 12/12, 0.65-1.3 s, 90 MB — chosen
- BAAI/bge-reranker-base: 10/12, 4.6-7.8 s, 1.1 GB. Its XLM-R tokenizer counts
  the largest chunk as 544 tokens, over its 512 window, so chunk tails are cut
- BAAI/bge-reranker-v2-m3: 12/12, 13-29 s, 2.3 GB. Reads 8192 tokens, so never
  truncates; only practical with a GPU
The eval is small and saturated — it separates bad from good, not good from
better. A larger eval set is the prerequisite for revisiting this choice.

Carried into Phase 4 (API) and Phase 5 (UI), in addition to those below
(1 and the refusal handling are now done in Phase 4):
1. Build one HybridRetriever at startup and reuse it. open() loads both models
   (~7 s cold); search() is then ~1 s. IndexUnavailableError is a deliberate
   refusal like the others — surface its message, not a 500.
2. No relevance floor, and reranker scores cannot provide one. The vector index
   always returns candidates, so nonsense queries still get top_k results. The
   scores are logits, comparable only within one query: a correct answer can
   score below a nonsense query's best hit (the right chunk for "who pays for
   heat and other utilities" scores -9.15; "INV-2024", absent from the corpus,
   scores -8.12). Any "no good match" signal needs a different approach.
3. The reranker shares its 512-token window between query and passage. A query
   over ~29 tokens trims the tail of the largest chunks (measured: the passage is
   trimmed, the query kept). Rerank score only; documented, not refused.

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

Carried into Phase 3 (all three now handled — kept as the reasoning behind
the tests in tests/test_fusion.py and tests/test_hybrid_retriever.py):
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

Carried into Phase 4 (API) and Phase 5 (UI) from Phase 2:
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
