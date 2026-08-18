"""Split document text into overlapping, size-bounded chunks.

Chunks are the unit that later gets embedded and searched. Two reasons they
exist rather than embedding whole documents:

1. Every embedding model has a hard input limit measured in tokens, and
   silently truncates anything longer — no error, just missing text.
2. One vector for an 80-page contract is an average of everything in it, so it
   matches every query weakly and none strongly. Smaller pieces match sharply.

Consecutive chunks overlap so a sentence split across a boundary still survives
whole in at least one chunk.
"""

import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

from app.config import get_settings

# A type alias for "any function taking a str and returning an int".
# Roughly a delegate type: `delegate int TokenCounter(string text)`.
TokenCounter = Callable[[str], int]

# Subword tokenizers split rare words into pieces, so token count runs above
# word count. ~1.3x is the usual figure for English prose.
_TOKENS_PER_WORD = 1.3

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Approximate the token count of a piece of text, without a real tokenizer.

    This is the default so that chunking has no model dependency: loading a
    real tokenizer would download a model on first use and make every chunking
    test slow and network-bound. Phase 2 can pass the embedding model's actual
    tokenizer as ``count_tokens`` instead — no change needed here.

    Args:
        text: Text to measure.

    Returns:
        Estimated token count. Real subword counts vary with content; numbers
        and tables run higher than prose.
    """
    # `str.split()` with no argument splits on runs of any whitespace and drops
    # empty pieces — different from `split(" ")`, which would keep them.
    word_count = len(text.split())
    return math.ceil(word_count * _TOKENS_PER_WORD) if word_count else 0


@dataclass(frozen=True, slots=True)
class Chunk:
    """One embeddable piece of a document.

    Attributes:
        text: The chunk's text, sliced verbatim from the source document.
        source_id: Identifier of the document this came from, carried through
            unchanged from :class:`DocumentMetadata`.
        chunk_index: Position in the document's chunk sequence, starting at 0.
        token_count: Token count of ``text`` per the counter in use.
        start_char: Start offset of ``text`` within the source document.
        end_char: End offset (exclusive) within the source document.
    """

    text: str
    source_id: str
    chunk_index: int
    token_count: int
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class _Segment:
    """An indivisible unit of text that chunks are assembled from.

    Normally a paragraph; a sentence or a run of words when a paragraph alone
    exceeds the token limit.
    """

    text: str
    start: int
    end: int
    token_count: int
    ends_paragraph: bool


def chunk_text(
    text: str,
    *,
    source_id: str,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    overlap_ratio: float | None = None,
    count_tokens: TokenCounter = estimate_tokens,
) -> Iterator[Chunk]:
    """Split text into overlapping chunks within a token size window.

    Packing is boundary-aware rather than a fixed window: a chunk is closed at
    a paragraph break once it is large enough, and only forced mid-paragraph
    when the next segment would breach ``max_tokens``. Specifying a *range*
    rather than a single size is what makes that possible.

    Args:
        text: Plain text of one document, as returned by ``extract_text``.
        source_id: Identifier stamped onto every chunk.
        min_tokens: Size at which a paragraph break becomes a valid place to
            close a chunk. Defaults to the configured value.
        max_tokens: Hard upper bound on chunk size. Defaults to the configured
            value.
        overlap_ratio: Fraction of ``max_tokens`` repeated from the end of one
            chunk at the start of the next. Defaults to the configured value.
        count_tokens: How to measure tokens. Swap in a model tokenizer to get
            exact counts.

    Yields:
        :class:`Chunk` objects in document order, ``chunk_index`` from 0.
        Yields nothing for empty or whitespace-only input.

    Raises:
        ValueError: If the size window or overlap ratio is not usable.
    """
    settings = get_settings()
    # `if x is None` rather than `or`: passing 0 explicitly must not be
    # silently replaced by the default.
    min_tokens = settings.chunk_min_tokens if min_tokens is None else min_tokens
    max_tokens = settings.chunk_max_tokens if max_tokens is None else max_tokens
    overlap_ratio = (
        settings.chunk_overlap_ratio if overlap_ratio is None else overlap_ratio
    )

    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if min_tokens > max_tokens:
        raise ValueError(
            f"min_tokens ({min_tokens}) must be <= max_tokens ({max_tokens})"
        )
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(
            f"overlap_ratio must be in [0.0, 1.0), got {overlap_ratio}"
        )

    segments = _segment(text, max_tokens, count_tokens)
    if not segments:
        return  # a bare `return` in a generator simply ends iteration

    overlap_budget = int(max_tokens * overlap_ratio)

    chunk_index = 0
    current: list[_Segment] = []
    current_tokens = 0
    segments_since_emit = 0

    for segment in segments:
        # Hard rule: never breach max_tokens. Close first, then start the next
        # chunk with the overlap tail already in place.
        if current and current_tokens + segment.token_count > max_tokens:
            yield _build_chunk(text, current, source_id, chunk_index, count_tokens)
            chunk_index += 1
            current = _overlap_tail(current, overlap_budget)
            current_tokens = sum(item.token_count for item in current)
            segments_since_emit = 0

        current.append(segment)
        current_tokens += segment.token_count
        segments_since_emit += 1

        # Soft rule: a paragraph break is a good place to stop once the chunk
        # is big enough. This is what keeps chunks from splitting mid-thought.
        if segment.ends_paragraph and current_tokens >= min_tokens:
            yield _build_chunk(text, current, source_id, chunk_index, count_tokens)
            chunk_index += 1
            current = _overlap_tail(current, overlap_budget)
            current_tokens = sum(item.token_count for item in current)
            segments_since_emit = 0

    # Emit the remainder — but only if it holds content not already emitted.
    # After a close, `current` holds just the overlap tail; without this guard
    # that tail would be published again as a duplicate final chunk.
    if current and segments_since_emit > 0:
        yield _build_chunk(text, current, source_id, chunk_index, count_tokens)


def _build_chunk(
    text: str,
    segments: list[_Segment],
    source_id: str,
    chunk_index: int,
    count_tokens: TokenCounter,
) -> Chunk:
    """Assemble one chunk from the segments currently accumulated.

    Segments are always contiguous in the source, so the chunk body is a plain
    slice of the original text. That keeps the document's own formatting rather
    than re-joining segments with invented separators, and makes
    ``start_char``/``end_char`` exact.
    """
    start = segments[0].start
    end = segments[-1].end
    body = text[start:end]
    return Chunk(
        text=body,
        source_id=source_id,
        chunk_index=chunk_index,
        token_count=count_tokens(body),
        start_char=start,
        end_char=end,
    )


def _overlap_tail(segments: list[_Segment], overlap_budget: int) -> list[_Segment]:
    """Pick the trailing segments to repeat at the start of the next chunk.

    Termination note: the result is always strictly shorter than ``segments``.
    If it were allowed to contain everything, the next chunk would begin where
    this one began and the scan would never advance.
    """
    if overlap_budget <= 0 or len(segments) <= 1:
        return []

    tail: list[_Segment] = []
    total = 0
    # `reversed()` walks the list back to front without copying it.
    for segment in reversed(segments):
        if tail and total + segment.token_count > overlap_budget:
            break
        tail.append(segment)
        total += segment.token_count

    tail.reverse()

    if len(tail) >= len(segments):
        tail = tail[1:]

    return tail


def _segment(
    text: str, max_tokens: int, count_tokens: TokenCounter
) -> list[_Segment]:
    """Break text into the smallest units chunks are assembled from.

    Paragraphs are preferred. A paragraph over the token limit is split into
    sentences, and a sentence still over the limit is split on word boundaries.
    """
    segments: list[_Segment] = []

    for para_start, para_end in _split_spans(text, (0, len(text)), _PARAGRAPH_BREAK):
        paragraph = text[para_start:para_end]
        paragraph_tokens = count_tokens(paragraph)

        if paragraph_tokens <= max_tokens:
            segments.append(
                _Segment(
                    text=paragraph,
                    start=para_start,
                    end=para_end,
                    token_count=paragraph_tokens,
                    ends_paragraph=True,
                )
            )
            continue

        # Paragraph is too large — descend to sentences.
        pieces: list[_Segment] = []
        for sent_start, sent_end in _split_spans(
            text, (para_start, para_end), _SENTENCE_BREAK
        ):
            sentence = text[sent_start:sent_end]
            sentence_tokens = count_tokens(sentence)

            if sentence_tokens <= max_tokens:
                pieces.append(
                    _Segment(
                        text=sentence,
                        start=sent_start,
                        end=sent_end,
                        token_count=sentence_tokens,
                        ends_paragraph=False,
                    )
                )
            else:
                pieces.extend(
                    _hard_split(text, sent_start, sent_end, max_tokens, count_tokens)
                )

        if pieces:
            # Only the final piece genuinely ends the paragraph.
            # `replace()` returns a modified copy of a frozen dataclass — the
            # same idea as C#'s `record with { ... }` expression.
            pieces[-1] = replace(pieces[-1], ends_paragraph=True)
            segments.extend(pieces)

    return segments


def _hard_split(
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> Iterator[_Segment]:
    """Split an oversized run of text on word boundaries.

    The fallback of last resort, for content with no paragraph or sentence
    structure — a giant table row, a minified blob, a wall of text with no
    punctuation.

    Yields:
        Segments at or under ``max_tokens``, except where a single word
        exceeds it on its own and cannot be divided further.
    """
    words = list(_WORD.finditer(text, start, end))
    if not words:
        return

    # First guess by token density, so the common case needs no re-splitting.
    # The 0.9 leaves headroom for density varying across the span.
    span_tokens = max(1, count_tokens(text[start:end]))
    tokens_per_word = span_tokens / len(words)
    words_per_piece = max(1, int(max_tokens * 0.9 / tokens_per_word))

    # `range(start, stop, step)` walking a list in fixed-size groups.
    for index in range(0, len(words), words_per_piece):
        group = words[index : index + words_per_piece]
        # `yield from` delegates to another generator — it re-yields everything
        # that one produces, rather than yielding the generator object itself.
        yield from _emit_group(text, group, max_tokens, count_tokens)


def _emit_group(
    text: str,
    group: list[re.Match[str]],
    max_tokens: int,
    count_tokens: TokenCounter,
) -> Iterator[_Segment]:
    """Emit one group of words, halving it while it still exceeds the limit.

    Halving guarantees termination: each recursion strictly shrinks the group,
    and a single word is emitted regardless of size since it cannot be split.
    """
    piece_start = group[0].start()
    piece_end = group[-1].end()
    piece = text[piece_start:piece_end]
    tokens = count_tokens(piece)

    if tokens > max_tokens and len(group) > 1:
        middle = len(group) // 2
        yield from _emit_group(text, group[:middle], max_tokens, count_tokens)
        yield from _emit_group(text, group[middle:], max_tokens, count_tokens)
        return

    yield _Segment(
        text=piece,
        start=piece_start,
        end=piece_end,
        token_count=tokens,
        ends_paragraph=False,
    )


def _split_spans(
    text: str, span: tuple[int, int], pattern: re.Pattern[str]
) -> list[tuple[int, int]]:
    """Split a region of text on a separator, returning absolute offsets.

    Works in offsets rather than substrings so every segment can be traced back
    to its exact position in the original document — which is what makes
    ``Chunk.start_char``/``end_char`` meaningful.

    Args:
        text: The full document text.
        span: ``(start, end)`` region of ``text`` to split.
        pattern: Compiled separator pattern.

    Returns:
        Non-empty ``(start, end)`` spans with surrounding whitespace trimmed.
    """
    start, end = span
    raw_spans: list[tuple[int, int]] = []
    cursor = start

    # finditer's pos/endpos arguments search a slice of the string without
    # copying it.
    for match in pattern.finditer(text, start, end):
        raw_spans.append((cursor, match.start()))
        cursor = match.end()
    raw_spans.append((cursor, end))

    trimmed: list[tuple[int, int]] = []
    for span_start, span_end in raw_spans:
        piece = text[span_start:span_end]
        leading = len(piece) - len(piece.lstrip())
        trailing = len(piece) - len(piece.rstrip())
        span_start += leading
        span_end -= trailing
        if span_end > span_start:
            trimmed.append((span_start, span_end))

    return trimmed
