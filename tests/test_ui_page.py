"""Tests for ui/streamlit_app.py.

``AppTest`` runs the real page script headlessly — no browser, no server —
and exposes what it rendered as a tree of elements. A fake client is placed in
``session_state`` before the first run, exactly where the page looks for its
client, so nothing talks to the network.
"""

import pytest
from streamlit.testing.v1 import AppTest

from app.api.schemas import (
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
)
from app.config import PROJECT_ROOT
from app.modes import PERSONAL_MODE_MESSAGE
from app.ui.client import ApiError, ApiUnavailableError, SearchRejectedError

# Absolute: AppTest resolves a relative path against the calling test file.
PAGE = str(PROJECT_ROOT / "ui" / "streamlit_app.py")


def hit(rank: int, source_id: str, chunk_index: int, text: str, **ranks) -> SearchHit:
    return SearchHit(
        rank=rank,
        chunk_id=f"{source_id}::{chunk_index}",
        source_id=source_id,
        chunk_index=chunk_index,
        text=text,
        score=-0.68 * rank,
        rrf_score=0.03,
        vector_rank=ranks.get("vector_rank"),
        keyword_rank=ranks.get("keyword_rank"),
    )


RESULTS = [
    hit(1, "Warranty Forms.pdf", 1, "defects within a period of 24 months; fee $1,000", vector_rank=1, keyword_rank=7),
    hit(2, "Sample Contract.docx", 3, "Contractor will submit invoices", vector_rank=5),
]


class FakeClient:
    """Stands in for SearchClient. Records searches; can be told to fail."""

    base_url = "http://fake.test"

    def __init__(self, *, down: bool = False, search_error: Exception | None = None) -> None:
        self.down = down
        self.search_error = search_error
        self.searches: list[tuple[str, int | None]] = []

    def health(self) -> HealthResponse:
        if self.down:
            raise ApiUnavailableError("down")
        return HealthResponse(status="ready", chunks=69, embedding_model="emb/model", reranker_model="rr/model")

    def documents(self) -> DocumentsResponse:
        if self.down:
            raise ApiUnavailableError("down")
        return DocumentsResponse(
            documents=[
                DocumentInfo(source_id="Sample Contract.docx", chunks=24),
                DocumentInfo(source_id="Warranty Forms.pdf", chunks=3),
            ]
        )

    def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        self.searches.append((query, top_k))
        if self.down:
            raise ApiUnavailableError("down")
        if self.search_error is not None:
            raise self.search_error
        return SearchResponse(query=query, took_ms=1100, results=RESULTS)


def open_page(client: FakeClient) -> AppTest:
    """Run the page once, with `client` in place of the real API client."""
    # The first run imports pydantic, numpy and friends; allow for it.
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.session_state["client"] = client
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def search(at: AppTest, query: str) -> AppTest:
    at.text_input(key="query_input").input(query)
    submit = next(b for b in at.button if b.label == "Search")
    submit.click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def card_headers(at: AppTest) -> list[str]:
    return [m.value for m in at.main.markdown if m.value.startswith("**")]


class TestFirstVisit:
    def test_shows_the_examples_and_no_results(self) -> None:
        at = open_page(FakeClient())

        labels = [b.label for b in at.button]
        assert "How long is the warranty?" in labels
        assert "FERPA" in labels
        assert card_headers(at) == []
        assert any("try one of the examples" in m.value for m in at.main.markdown)

    def test_demo_notice_counts_the_documents(self) -> None:
        at = open_page(FakeClient())

        assert "Demo mode" in at.info[0].value
        assert "2 bundled sample documents" in at.info[0].value

    def test_personal_mode_is_explained_with_the_shared_message(self) -> None:
        at = open_page(FakeClient())

        expander = next(e for e in at.expander if e.label == "Search your own files?")
        assert expander.markdown[0].value == PERSONAL_MODE_MESSAGE

    def test_nothing_is_searched_until_asked(self) -> None:
        client = FakeClient()

        open_page(client)

        assert client.searches == []


class TestSidebar:
    def test_connected_status_and_documents(self) -> None:
        at = open_page(FakeClient())

        sidebar = [m.value for m in at.sidebar.markdown]
        assert any("Connected" in value for value in sidebar)
        assert "Sample Contract\\.docx — 24 passages" in sidebar
        assert "Warranty Forms\\.pdf — 3 passages" in sidebar
        assert any("69 passages · 2 documents" in c.value for c in at.sidebar.caption)

    def test_models_are_listed(self) -> None:
        at = open_page(FakeClient())

        models = next(e for e in at.expander if e.label == "Models")
        assert "emb/model" in models.text[0].value
        assert "rr/model" in models.text[0].value

    def test_api_down_shows_not_connected_with_the_command(self) -> None:
        at = open_page(FakeClient(down=True))

        assert any("Not connected" in m.value for m in at.sidebar.markdown)
        assert any("run_local.py" in c.value for c in at.sidebar.caption)


class TestSearching:
    def test_one_card_per_result_in_order(self) -> None:
        at = search(open_page(FakeClient()), "how long is the warranty")

        assert card_headers(at) == [
            "**1** · **Warranty Forms\\.pdf** · passage 2 of 3",
            "**2** · **Sample Contract\\.docx** · passage 4 of 24",
        ]

    def test_summary_line(self) -> None:
        at = search(open_page(FakeClient()), "warranty")

        assert any(c.value == "2 results · 1.1 s" for c in at.main.caption)

    def test_passages_are_highlighted_and_escaped(self) -> None:
        at = search(open_page(FakeClient()), "how long is the warranty period")

        passage = next(m.value for m in at.main.markdown if "defects" in m.value)
        assert ":orange-background[period]" in passage
        # A literal dollar sign, not the start of a LaTeX formula.
        assert "\\$1\\,000" in passage

    def test_why_line_explains_both_indexes(self) -> None:
        at = search(open_page(FakeClient()), "warranty")

        why = [c.value for c in at.main.caption if c.value.startswith("Why:")]
        assert why == [
            "Why: meaning match #1 · keyword match #7",
            "Why: meaning match #5 · keyword search: not in its top 20",
        ]

    def test_score_appears_only_inside_details(self) -> None:
        at = search(open_page(FakeClient()), "warranty")

        visible = [m.value for m in at.main.markdown] + [c.value for c in at.main.caption]
        assert not any("-0.680" in value or "-1.360" in value for value in visible)
        details = [e for e in at.expander if e.label == "Details"]
        assert len(details) == 2
        assert "reranker score  -0.680" in details[0].text[0].value
        assert "not confidence" in details[0].caption[0].value

    def test_honest_footer(self) -> None:
        at = search(open_page(FakeClient()), "warranty")

        assert any("may not be among them" in c.value for c in at.main.caption)

    def test_the_chosen_result_count_is_sent(self) -> None:
        client = FakeClient()
        at = open_page(client)
        at.selectbox[0].select(3)

        search(at, "warranty")

        assert client.searches == [("warranty", 3)]

    def test_results_survive_a_rerun(self) -> None:
        client = FakeClient()
        at = search(open_page(client), "warranty")

        at.run()  # any later interaction reruns the whole script

        assert len(card_headers(at)) == 2
        assert len(client.searches) == 1  # shown again, not searched again

    def test_blank_query_warns_without_searching(self) -> None:
        client = FakeClient()

        at = search(open_page(client), "   ")

        assert "Type something" in at.warning[0].value
        assert client.searches == []


class TestExamples:
    def test_clicking_an_example_fills_the_box_and_searches(self) -> None:
        client = FakeClient()
        at = open_page(client)

        next(b for b in at.button if b.label == "FERPA").click().run()

        assert not at.exception
        assert client.searches == [("FERPA", 5)]
        assert at.text_input(key="query_input").value == "FERPA"
        assert len(card_headers(at)) == 2

    def test_an_example_runs_only_once(self) -> None:
        client = FakeClient()
        at = open_page(client)
        next(b for b in at.button if b.label == "GA-48").click().run()

        at.run()

        assert client.searches == [("GA-48", 5)]


class TestFailures:
    def test_api_down_explains_how_to_start_it(self) -> None:
        at = search(open_page(FakeClient(down=True)), "warranty")

        assert "isn't running" in at.error[0].value
        assert "run_local.py" in at.error[0].value

    def test_rejected_search_is_a_warning_with_the_reason(self) -> None:
        client = FakeClient(search_error=SearchRejectedError("query: String should have at most 500 characters"))

        at = search(open_page(client), "warranty")

        assert "at most 500 characters" in at.warning[0].value

    def test_other_api_errors_are_shown_plainly(self) -> None:
        client = FakeClient(search_error=ApiError("The search service answered with an error (500)."))

        at = search(open_page(client), "warranty")

        assert "(500)" in at.error[0].value


class TestConfigurationRefusals:
    def test_personal_mode_shows_the_message_instead_of_the_page(
        self, fresh_settings: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_MODE", "personal")

        at = open_page(FakeClient())

        assert at.error[0].value == PERSONAL_MODE_MESSAGE
        assert [b.label for b in at.button] == []

    def test_invalid_settings_are_a_message_not_a_traceback(
        self, fresh_settings: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_TOP_K", "50")

        at = open_page(FakeClient())

        assert "configuration is invalid" in at.error[0].value
