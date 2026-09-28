"""Tests for app.ui.text.

Rendering is checked two ways: the exact markdown produced, and — for the
cases that matter most — that removing the backslash escapes gives back the
original text, which is what "renders literally" means.
"""

import re

import pytest

from app.ui.text import (
    STOPWORDS,
    escape_markdown,
    explain_match,
    highlight,
    highlight_terms,
    normalize_whitespace,
)

HIGHLIGHT = re.compile(r":orange-background\[(.*?)\]")


def unescape(markdown: str) -> str:
    """What the reader sees: highlight markup removed, escapes resolved."""
    return re.sub(r"\\(.)", r"\1", HIGHLIGHT.sub(r"\1", markdown))


def highlighted_words(markdown: str) -> list[str]:
    return [unescape(word) for word in HIGHLIGHT.findall(markdown)]


class TestEscapeMarkdown:
    @pytest.mark.parametrize(
        "text",
        [
            "Fee of $1,000,000 and a $500 deposit",  # LaTeX math delimiters
            "*bold* _italic_ **strong**",
            "# Not a heading",
            "[click](http://example.com)",
            ":red[styled] :orange-background[fake]",
            "<script>alert(1)</script>",
            "a | table | row",
            "back\\slash and `code`",
            "1. not a list - nor this + either",
            "~strike~ and {braces} and !image",
        ],
    )
    def test_text_renders_literally(self, text: str) -> None:
        escaped = escape_markdown(text)

        assert unescape(escaped) == text

    def test_every_special_character_is_escaped(self) -> None:
        escaped = escape_markdown("$*_[]()#:<>|`\\")

        # No special character is left without a backslash before it.
        assert re.sub(r"\\.", "", escaped) == ""

    def test_dollar_signs_cannot_open_math(self) -> None:
        assert "$" not in escape_markdown("$5 and $10").replace("\\$", "")

    def test_plain_words_are_unchanged(self) -> None:
        assert escape_markdown("warranty period") == "warranty period"


class TestNormalizeWhitespace:
    def test_indentation_cannot_become_a_code_block(self) -> None:
        assert normalize_whitespace("    indented line") == "indented line"

    def test_line_breaks_inside_a_paragraph_are_joined(self) -> None:
        assert normalize_whitespace("one\ntwo\n   three") == "one two three"

    def test_paragraph_breaks_are_kept(self) -> None:
        assert normalize_whitespace("first\n\n\n  second") == "first\n\nsecond"

    def test_blank_input(self) -> None:
        assert normalize_whitespace("  \n\n  ") == ""


class TestHighlightTerms:
    def test_uses_keyword_tokenization_and_drops_stopwords(self) -> None:
        assert highlight_terms("How long is the Warranty?") == {"long", "warranty"}

    def test_reference_numbers_split_like_bm25(self) -> None:
        assert highlight_terms("GA-48") == {"ga", "48"}

    def test_only_stopwords_gives_nothing(self) -> None:
        assert highlight_terms("what is the") == frozenset()

    def test_contract_words_are_not_stopwords(self) -> None:
        assert {"shall", "term", "notice"}.isdisjoint(STOPWORDS)


class TestHighlight:
    def test_matching_words_are_highlighted(self) -> None:
        markdown = highlight("The warranty period is 24 months.", "how long is the warranty")

        assert highlighted_words(markdown) == ["warranty"]
        assert unescape(markdown) == "The warranty period is 24 months."

    def test_case_is_ignored(self) -> None:
        markdown = highlight("WARRANTY and Warranty", "warranty")

        assert highlighted_words(markdown) == ["WARRANTY", "Warranty"]

    def test_whole_words_only(self) -> None:
        assert highlighted_words(highlight("the warrantyholder", "warranty")) == []

    def test_stopwords_are_never_highlighted(self) -> None:
        markdown = highlight("the term of the lease", "the term")

        assert highlighted_words(markdown) == ["term"]

    def test_repeated_terms_are_all_highlighted(self) -> None:
        markdown = highlight("notice, notice and notice", "notice")

        assert highlighted_words(markdown) == ["notice"] * 3

    def test_unicode_words(self) -> None:
        markdown = highlight("La garantía cubre 24 meses", "garantía")

        assert highlighted_words(markdown) == ["garantía"]

    def test_dollar_amounts_stay_literal_around_highlights(self) -> None:
        text = "A deposit of $1,000 is due; the deposit refund is $500."

        markdown = highlight(text, "deposit")

        assert unescape(markdown) == text
        assert highlighted_words(markdown) == ["deposit", "deposit"]
        # Every $ is escaped, so none can open a math block.
        assert "$" not in markdown.replace("\\$", "")

    def test_underscores_inside_words_are_escaped(self) -> None:
        markdown = highlight("see section_4 and term_sheet", "term_sheet")

        assert unescape(markdown) == "see section_4 and term_sheet"
        assert "\\_" in markdown

    def test_injected_directives_render_literally(self) -> None:
        text = ":red[fake highlight] and [a link](http://evil.example)"

        markdown = highlight(text, "zzz")

        assert unescape(markdown) == text
        assert HIGHLIGHT.findall(markdown) == []

    def test_no_query_terms_means_no_highlights(self) -> None:
        assert highlighted_words(highlight("the warranty", "the")) == []

    def test_paragraphs_survive(self) -> None:
        assert "\n\n" in highlight("first para\n\nsecond para", "para")


class TestExplainMatch:
    def test_found_by_both(self) -> None:
        assert explain_match(1, 7, 20) == "meaning match #1 · keyword match #7"

    def test_found_only_by_meaning(self) -> None:
        assert explain_match(11, None, 20) == "meaning match #11 · keyword search: not in its top 20"

    def test_found_only_by_keyword(self) -> None:
        assert explain_match(None, 1, 20) == "meaning search: not in its top 20 · keyword match #1"
