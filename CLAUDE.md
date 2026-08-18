# CLAUDE.md

## Project Overview
A hybrid (keyword + vector) document search engine over local files and
Google Drive. Search-only for v1 — no chat/answer-synthesis layer yet.
General-purpose: should work for contracts, tax documents, warranties, etc.

## Tech Stack
- Python 3.14+, FastAPI, Streamlit
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
- All document sources implement the SourceConnector interface in
  app/ingestion/connectors/base.py — never hardcode a source-specific call
  outside a connector
- Retrieval always merges BM25 + vector results via reciprocal rank fusion
  before reranking — never return raw vector-similarity results alone
- The public demo deployment only ever indexes data/sample_docs/ — never
  wire real local paths or Drive credentials into anything deployed publicly

## Testing
- pytest, one test file per module in app/
- Any change to chunking or retrieval logic needs a corresponding test

## Things to Avoid
- No hardcoded file paths or API keys — use app/config.py and .env
- Don't add a chat/LLM-answer layer — this is v1, search-only

## Current Phase
Phase 1 complete — local ingestion (SourceConnector, LocalFSConnector, loaders,
chunking). Next: Phase 2 — embeddings + vector store + BM25 index.

Carried into Phase 2: assert `chunk_max_tokens <= model.max_seq_length` at
startup. Embedding models silently truncate longer input, so the default
500-800 token chunks would be quietly cut off by e.g. all-MiniLM-L6-v2 (256)
or bge-base-en-v1.5 (512) with no error raised.