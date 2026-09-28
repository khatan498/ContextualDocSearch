"""The search page.

    python scripts/run_local.py            # starts the API and this page together

Streamlit runs this whole file from top to bottom every time the user does
anything — types and submits, clicks an example, opens an expander. Anything
that must survive between those runs (the API client, the last results) lives
in `st.session_state`, Streamlit's per-browser-tab storage.

All the logic is in app/ui/ and is tested without Streamlit; this file only
lays the page out.
"""

import sys
from pathlib import Path

# `streamlit run` puts this file's folder (ui/) on sys.path, not the repo root,
# so `import app` would fail without this — the same fix the scripts use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.api.schemas import MAX_QUERY_CHARS, DocumentInfo, SearchResponse  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.modes import PERSONAL_MODE_MESSAGE, PersonalModeUnavailableError  # noqa: E402
from app.ui.client import (  # noqa: E402
    ApiError,
    ApiUnavailableError,
    SearchClient,
    SearchRejectedError,
)
from app.ui.text import escape_markdown, explain_match, highlight  # noqa: E402

START_COMMAND = "python scripts/run_local.py"
RESULT_CHOICES = (3, 5, 10)

# Chosen to show both halves of hybrid search at work.
EXAMPLES: tuple[tuple[str, str], ...] = (
    ("How long is the warranty?", "meaning"),
    ("Who pays for utilities?", "meaning"),
    ("Security deposit", "meaning + keyword"),
    ("FERPA", "exact term"),
    ("GA-48", "exact term"),
    ("Chapter 2260", "exact term"),
)

st.set_page_config(page_title="ContextualDocSearch", page_icon="🔎", layout="wide")

# --- Configuration ------------------------------------------------------------
# The launcher refuses these before the page ever starts, but a hand-run
# `streamlit run` reaches here, so they are shown as messages, not tracebacks.
try:
    settings = get_settings()
except PersonalModeUnavailableError:
    st.error(PERSONAL_MODE_MESSAGE)
    st.stop()  # ends this run of the script here, like an early return
except ValidationError as exc:
    st.error(f"The configuration is invalid:\n\n{escape_markdown(str(exc))}")
    st.stop()

# One client per browser tab, created on the first run and reused after.
# Tests put a fake here before the page runs.
if "client" not in st.session_state:
    st.session_state.client = SearchClient(settings.api_base_url)
client: SearchClient = st.session_state.client


# --- Searching --------------------------------------------------------------
def run_search(query: str, top_k: int) -> None:
    """Search, and store the outcome for this and later reruns to display."""
    query = query.strip()
    if not query:
        st.session_state.outcome = ("warning", "Type something to search for.")
        return
    try:
        with st.spinner("Searching…"):
            st.session_state.outcome = ("results", client.search(query, top_k=top_k))
    except ApiUnavailableError:
        st.session_state.outcome = (
            "error",
            f"The search service isn't running. Start everything with `{START_COMMAND}`.",
        )
    except SearchRejectedError as exc:
        st.session_state.outcome = ("warning", f"That search can't be run — {exc}")
    except ApiError as exc:
        st.session_state.outcome = ("error", str(exc))


def choose_example(example: str) -> None:
    """Button callback: put the example in the box and search for it.

    Callbacks run *before* the next rerun of the script, so setting the text
    box's session-state key here makes the box show the example.
    """
    st.session_state.query_input = example
    st.session_state.pending_example = example


# --- Sidebar: status and what can be searched -------------------------------
documents: list[DocumentInfo] = []
with st.sidebar:
    try:
        health = client.health()
        documents = client.documents().documents
    except (ApiUnavailableError, ApiError):
        health = None
        st.markdown(":red[●] **Not connected**")
        st.caption(f"Start everything with `{START_COMMAND}`.")
    else:
        st.markdown(":green[●] **Connected**")
        st.caption(f"{health.chunks} passages · {len(documents)} documents")
        st.subheader("Documents")
        for document in documents:
            st.markdown(f"{escape_markdown(document.source_id)} — {document.chunks} passages")

    with st.expander("How search works"):
        st.markdown(
            "1. **Keyword search** (BM25) and **meaning search** (embeddings) each "
            "shortlist the passages they rate best.\n"
            "2. The two shortlists are **merged by rank**, so a passage both rate "
            "highly rises to the top.\n"
            "3. A **reranker** reads each shortlisted passage together with your "
            "question and puts them in final order.\n\n"
            "Rebuilt the index? Restart with the launcher so the new one is loaded."
        )
    if health is not None:
        with st.expander("Models"):
            st.text(f"Embeddings: {health.embedding_model}\nReranker:   {health.reranker_model}")

passage_totals = {document.source_id: document.chunks for document in documents}


# --- Main area: header and search form --------------------------------------
st.title("🔎 ContextualDocSearch")
st.caption("Hybrid keyword + meaning search over a set of sample contracts and forms.")
corpus = f"the {len(documents)} bundled sample documents" if documents else "the bundled sample documents"
st.info(f"Demo mode: searches {corpus} only.")
with st.expander("Search your own files?"):
    st.write(PERSONAL_MODE_MESSAGE)

# A form sends its inputs together: typing does not trigger a rerun, only
# pressing Enter or the Search button does.
with st.form("search", border=False):
    box, size, button = st.columns([6, 1, 1], vertical_alignment="bottom")
    query = box.text_input(
        "Search",
        key="query_input",
        max_chars=MAX_QUERY_CHARS,
        placeholder="e.g. how long is the warranty",
        label_visibility="collapsed",
    )
    top_k = size.selectbox("Results", RESULT_CHOICES, index=1, label_visibility="collapsed")
    submitted = button.form_submit_button("Search", type="primary", width="stretch")

st.caption("Try an example:")
columns = st.columns(len(EXAMPLES))
for column, (example, kind) in zip(columns, EXAMPLES):
    column.button(example, on_click=choose_example, args=(example,), help=f"Tests {kind} search")

# `pop` reads and removes in one step, so an example runs exactly once.
pending = st.session_state.pop("pending_example", None)
if submitted:
    run_search(query, top_k)
elif pending is not None:
    run_search(pending, top_k)


# --- Results ------------------------------------------------------------------
def show_results(response: SearchResponse) -> None:
    """One bordered card per result, then the honest footer."""
    count = len(response.results)
    st.caption(f"{count} result{'' if count == 1 else 's'} · {response.took_ms / 1000:.1f} s")
    if not response.results:
        st.info("No passages found.")
        return

    for hit in response.results:
        with st.container(border=True):
            total = passage_totals.get(hit.source_id)
            position = f"passage {hit.chunk_index + 1}" + (f" of {total}" if total else "")
            st.markdown(f"**{hit.rank}** · **{escape_markdown(hit.source_id)}** · {position}")
            st.markdown(highlight(hit.text, response.query))
            st.caption(
                "Why: "
                + explain_match(hit.vector_rank, hit.keyword_rank, settings.retrieval_candidates)
            )
            # The score lives only in here: it ranks results within this one
            # search and says nothing about whether the answer is right.
            with st.expander("Details"):
                st.caption("Scores only compare results within this search — they are not confidence.")
                st.text(
                    f"reranker score  {hit.score:.3f}\n"
                    f"fused score     {hit.rrf_score:.4f}\n"
                    f"chunk           {hit.chunk_id}"
                )

    st.caption(
        "These are the closest passages in the demo documents; "
        "the answer may not be among them."
    )


outcome = st.session_state.get("outcome")
if outcome is None:
    st.write("Type a question above, or try one of the examples.")
else:
    kind, value = outcome
    if kind == "results":
        show_results(value)
    elif kind == "warning":
        st.warning(value)
    else:
        st.error(value)
