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
Phase 1 complete — demo-corpus ingestion (LocalFSConnector, loaders, chunking)
plus the demo/personal mode split. Next: Phase 2 — embeddings + vector store +
BM25 index over the sample corpus.

Carried into Phase 2 — these are one decision, not two:

1. Pass the embedding model's real tokenizer as chunk_text's count_tokens in
   scripts/build_index.py. Keep estimate_tokens as the signature default so the
   test suite stays offline and fast.
2. Assert `chunk_max_tokens <= model.max_seq_length` at startup. Embedding
   models silently truncate longer input — all-MiniLM-L6-v2 caps at 256,
   bge-base-en-v1.5 at 512, so the default 800 overflows both with no error.

The assert is only meaningful with (1) in place. It compares *configured* token
counts against the model limit, so heuristic-driven boundaries can pass the
assert while individual chunks still overflow. The assert alone gives false
confidence.

Why (measured against all-MiniLM-L6-v2's tokenizer, 2026-08-25): estimate_tokens
under-counts by ~12% on a realistic mixed document, and by 44-74% on table rows,
dates, currency, serial numbers, and non-English text. It over-counts on plain
prose (+22%), which is the harmless direction. Under-counting is the dangerous
one, and the error concentrates in exactly the chunks holding the precise facts
a query is usually after — amounts, dates, reference numbers. Tokenizer load
from cache is ~1.6s and happens at index time, not query time, so the cost is
not a reason to avoid it.

## Known Chunking Behaviours
Measured 2026-08-25 against app/ingestion/chunking.py using estimate_tokens.
Current behaviour, not bugs — recorded as a baseline for retrieval tuning in
Phase 3. Changing any of these needs a test (see Testing).

Distinct from README's "Known issues", which lists genuine defects to fix
(chunks exceeding max_tokens, duplicate chunks). The three below are intended
behaviour that is merely surprising; do not "fix" them without a decision.

Root cause of (2) and (3): overlap is quantised to whole segments — normally
paragraphs — so it can never be finer-grained than the paragraphs it is cut from.

1. Chunks settle near chunk_min_tokens, not chunk_max_tokens. The packer closes
   at the first paragraph break past min, so min is the attractor and max is only
   a ceiling. Measured: 500/800 defaults over ~68-token paragraphs produced every
   full chunk at 562 tokens, never near 800.
   Lever: raise CHUNK_MIN_TOKENS to get larger chunks.

2. A chunk made of a single segment gets zero overlap. _overlap_tail returns []
   when len(segments) <= 1; this is required for forward progress, since a tail
   as long as the chunk would restart the next chunk at the same offset. So a
   document whose paragraphs each exceed chunk_min_tokens gets no overlap at all.
   Measured: 676-token paragraphs at 500/800 gave 0/5 boundaries overlapping,
   versus 5/5 for 68-token paragraphs.
   Lever: set CHUNK_MIN_TOKENS above the document's typical paragraph size so a
   chunk always spans more than one paragraph.

3. Where overlap does occur it can far exceed chunk_overlap_ratio. The tail's
   first segment is appended unconditionally, so one large paragraph is repeated
   whole. Measured: a 240-token budget produced a 676-token overlap — 2.8x over.
   Treat the ratio as a target, not a cap: index size can exceed the nominal +15%.
