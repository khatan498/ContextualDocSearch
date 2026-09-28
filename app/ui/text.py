"""Turn document text into safe, highlighted Streamlit markdown.

Passages are shown with ``st.markdown`` so matched words can be highlighted —
but Streamlit's markdown is not plain text, and document text is not written
with that in mind:

- ``$`` starts LaTeX math. "$1,000,000 ... $500" in a contract would render as
  a garbled formula.
- ``*``, ``_``, ``#``, ``[text](url)`` and Streamlit's ``:color[...]``
  directives would restyle the passage or inject links.
- A line indented four spaces becomes a code block.

So every passage is normalised and escaped *before* highlight markup is added,
and only the highlight markup itself is ever unescaped.
"""

import re
import string

from app.indexing.keyword_index import tokenize

# Words too common to be worth highlighting: without this, "how long is the
# warranty" would light up every "is" and "the" in every passage. Deliberately
# small and ordinary — "shall", "term" and the like carry meaning in contracts
# and are kept.
STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been but by can could did do does for from had has
    have how i if in into is it its may me my no not of on or our so than that
    the their them then there these they this to was we were what when where
    which who whom why will with would you your""".split()
)

# CommonMark allows *any* ASCII punctuation character to be backslash-escaped,
# and an escaped character always renders literally. Escaping all of them is
# simpler — and provably complete — compared with tracking which ones matter
# where.
_PUNCTUATION = re.compile("([" + re.escape(string.punctuation) + "])")

_WORD = re.compile(r"\w+")
_BLANK_LINE = re.compile(r"\n\s*\n")


def escape_markdown(text: str) -> str:
    """Make text render literally in Streamlit markdown.

    Args:
        text: Arbitrary text.

    Returns:
        The same text with every ASCII punctuation character backslash-escaped.
    """
    # `\1` in the replacement means "the character the group matched".
    return _PUNCTUATION.sub(r"\\\1", text)


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace inside paragraphs; keep the breaks between them.

    Extracted PDF text is full of hard line breaks and indentation. Left in
    markdown, indentation turns lines into code blocks.

    Args:
        text: Raw passage text.

    Returns:
        Paragraphs separated by one blank line, each on a single line.
    """
    paragraphs = (" ".join(part.split()) for part in _BLANK_LINE.split(text))
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def highlight_terms(query: str) -> frozenset[str]:
    """The query words worth highlighting.

    Uses the keyword index's own ``tokenize``, so a highlighted word is always
    one BM25 could match, then drops the stopwords.

    Args:
        query: What the user searched for.

    Returns:
        Lowercased terms.
    """
    return frozenset(token for token in tokenize(query) if token not in STOPWORDS)


def highlight(text: str, query: str) -> str:
    """Render a passage as safe markdown with the query's words highlighted.

    Matching is by whole word and ignores case, exactly as keyword search
    tokenises: "warranty" highlights "Warranty" but not "warrantyholder".

    Args:
        text: The passage.
        query: What the user searched for.

    Returns:
        Markdown ready for ``st.markdown``.
    """
    terms = highlight_terms(query)
    text = normalize_whitespace(text)
    pieces: list[str] = []
    position = 0

    # `finditer` walks the words; the gaps between them (spaces, punctuation)
    # are escaped and kept as they are.
    for match in _WORD.finditer(text):
        pieces.append(escape_markdown(text[position : match.start()]))
        word = match.group()
        # The word is escaped too: \w matches "_", which markdown would read
        # as emphasis.
        escaped = escape_markdown(word)
        if word.lower() in terms:
            pieces.append(f":orange-background[{escaped}]")
        else:
            pieces.append(escaped)
        position = match.end()

    pieces.append(escape_markdown(text[position:]))
    return "".join(pieces)


def explain_match(vector_rank: int | None, keyword_rank: int | None, shortlist: int) -> str:
    """Say in plain words which half of hybrid search found a passage.

    "Meaning" is the vector index, "keyword" the BM25 index. A missing rank
    means the passage was outside that index's shortlist, not necessarily
    that it shares no words with the query.

    Args:
        vector_rank: Where vector search placed it, or None.
        keyword_rank: Where keyword search placed it, or None.
        shortlist: How many candidates each index contributes
            (RETRIEVAL_CANDIDATES).

    Returns:
        e.g. "meaning match #1 · keyword match #7".
    """
    meaning = (
        f"meaning match #{vector_rank}"
        if vector_rank is not None
        else f"meaning search: not in its top {shortlist}"
    )
    keyword = (
        f"keyword match #{keyword_rank}"
        if keyword_rank is not None
        else f"keyword search: not in its top {shortlist}"
    )
    return f"{meaning} · {keyword}"
