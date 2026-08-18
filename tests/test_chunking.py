"""Tests for app.ingestion.chunking."""

import pytest

from app.ingestion.chunking import Chunk, chunk_text, estimate_tokens

# Most tests pass min/max/overlap explicitly rather than relying on the
# configured defaults, so a change to .env can never silently alter what these
# assert.
WINDOW = {"min_tokens": 100, "max_tokens": 200, "overlap_ratio": 0.15}


def make_document(paragraphs: int = 30, sentences: int = 6) -> str:
    """Build predictable multi-paragraph text."""
    sentence = "The quick brown fox jumps over the lazy dog."
    return "\n\n".join(
        f"Paragraph {index}. " + " ".join([sentence] * sentences)
        for index in range(paragraphs)
    )


class TestEstimateTokens:
    def test_empty_text_is_zero(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens("   \n\t ") == 0

    def test_scales_above_word_count(self) -> None:
        text = " ".join(["word"] * 100)
        # ~1.3 tokens per word, so it must exceed the raw word count.
        assert estimate_tokens(text) > 100


class TestEmptyInput:
    @pytest.mark.parametrize("text", ["", "   ", "\n\n\n", "\t  \n  \t"])
    def test_blank_input_yields_no_chunks(self, text: str) -> None:
        assert list(chunk_text(text, source_id="doc", **WINDOW)) == []


class TestChunkStructure:
    def test_short_document_yields_single_chunk(self) -> None:
        chunks = list(chunk_text("One short sentence.", source_id="doc", **WINDOW))

        assert len(chunks) == 1
        assert chunks[0].text == "One short sentence."
        assert chunks[0].chunk_index == 0

    def test_chunk_indices_are_contiguous_from_zero(self) -> None:
        chunks = list(chunk_text(make_document(), source_id="doc", **WINDOW))

        assert len(chunks) > 1, "test needs a document large enough to split"
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_source_id_is_preserved_on_every_chunk(self) -> None:
        chunks = list(chunk_text(make_document(), source_id="contracts/nda.pdf", **WINDOW))

        assert chunks
        assert all(c.source_id == "contracts/nda.pdf" for c in chunks)

    def test_offsets_slice_back_to_the_source_text(self) -> None:
        document = make_document()
        chunks = list(chunk_text(document, source_id="doc", **WINDOW))

        assert chunks
        for chunk in chunks:
            assert chunk.text == document[chunk.start_char : chunk.end_char]

    def test_chunks_are_immutable(self) -> None:
        chunk = next(iter(chunk_text("Hello world.", source_id="doc", **WINDOW)))

        # frozen dataclass — assignment raises rather than silently mutating.
        with pytest.raises(Exception):
            chunk.text = "changed"  # type: ignore[misc]


class TestSizeWindow:
    def test_no_chunk_exceeds_max_tokens(self) -> None:
        chunks = list(chunk_text(make_document(), source_id="doc", **WINDOW))

        assert chunks
        assert all(c.token_count <= WINDOW["max_tokens"] for c in chunks)

    def test_every_chunk_but_the_last_reaches_min_tokens(self) -> None:
        chunks = list(chunk_text(make_document(), source_id="doc", **WINDOW))

        assert len(chunks) > 1
        # The final chunk is whatever remains and may legitimately be short.
        assert all(c.token_count >= WINDOW["min_tokens"] for c in chunks[:-1])

    def test_whole_document_is_covered(self) -> None:
        document = make_document()
        chunks = list(chunk_text(document, source_id="doc", **WINDOW))

        assert chunks[0].start_char == 0
        assert chunks[-1].end_char == len(document.rstrip())


class TestOverlap:
    def test_consecutive_chunks_overlap_in_the_source(self) -> None:
        chunks = list(chunk_text(make_document(), source_id="doc", **WINDOW))

        assert len(chunks) > 1
        for earlier, later in zip(chunks, chunks[1:]):
            assert later.start_char < earlier.end_char, "chunks must overlap"

    def test_overlapping_text_actually_repeats(self) -> None:
        document = make_document()
        chunks = list(chunk_text(document, source_id="doc", **WINDOW))

        first, second = chunks[0], chunks[1]
        shared = document[second.start_char : first.end_char]

        assert shared.strip(), "overlap region must be non-empty"
        assert shared in first.text
        assert shared in second.text

    def test_zero_overlap_produces_adjacent_chunks(self) -> None:
        document = make_document()
        chunks = list(
            chunk_text(
                document,
                source_id="doc",
                min_tokens=100,
                max_tokens=200,
                overlap_ratio=0.0,
            )
        )

        assert len(chunks) > 1
        for earlier, later in zip(chunks, chunks[1:]):
            assert later.start_char >= earlier.end_char


class TestPluggableTokenCounter:
    def test_custom_counter_changes_the_boundaries(self) -> None:
        document = make_document()

        default_chunks = list(chunk_text(document, source_id="doc", **WINDOW))
        # Reporting 10x the tokens forces paragraphs to be split internally.
        inflated_chunks = list(
            chunk_text(
                document,
                source_id="doc",
                count_tokens=lambda text: len(text.split()) * 10,
                **WINDOW,
            )
        )

        assert len(inflated_chunks) > len(default_chunks)

    def test_counter_affects_reported_size_even_when_boundaries_match(self) -> None:
        """Documents a real property of the paragraph-boundary rule.

        A chunk closes at the first paragraph break past ``min_tokens``. When
        paragraphs are small relative to the window, the same number of them
        fits either way, so a merely *proportional* change to the counter
        shifts the reported sizes without moving any boundary.
        """
        document = make_document()

        default_chunks = list(chunk_text(document, source_id="doc", **WINDOW))
        word_chunks = list(
            chunk_text(
                document,
                source_id="doc",
                count_tokens=lambda text: len(text.split()),
                **WINDOW,
            )
        )

        assert [c.start_char for c in word_chunks] == [
            c.start_char for c in default_chunks
        ]
        # A pure word count is ~1.3x lower than the subword estimate.
        assert word_chunks[0].token_count < default_chunks[0].token_count

    def test_token_count_is_reported_by_the_supplied_counter(self) -> None:
        chunks = list(
            chunk_text(
                "One two three four five.",
                source_id="doc",
                count_tokens=lambda text: len(text.split()),
                **WINDOW,
            )
        )

        assert chunks[0].token_count == 5


class TestPathologicalInput:
    def test_unbroken_blob_is_split_and_terminates(self) -> None:
        """No paragraph breaks, no sentence punctuation — the hard-split path."""
        blob = " ".join(f"word{i}" for i in range(4000))

        chunks = list(chunk_text(blob, source_id="blob", **WINDOW))

        assert len(chunks) > 1
        assert all(c.token_count <= WINDOW["max_tokens"] for c in chunks)

    def test_single_word_longer_than_the_window_still_emits(self) -> None:
        chunks = list(
            chunk_text(
                "supercalifragilisticexpialidocious",
                source_id="doc",
                min_tokens=1,
                max_tokens=1,
                overlap_ratio=0.0,
            )
        )

        # An indivisible word is emitted even though it cannot be made to fit.
        assert len(chunks) == 1

    def test_tiny_window_with_heavy_overlap_still_advances(self) -> None:
        """Regression guard: a large overlap must not stall the scan.

        If the overlap tail were allowed to contain every segment of the
        previous chunk, the next chunk would start where the last one started
        and the loop would never move forward.
        """
        chunks = list(
            chunk_text(
                make_document(paragraphs=10),
                source_id="doc",
                min_tokens=1,
                max_tokens=10,
                overlap_ratio=0.9,
            )
        )

        assert chunks
        for earlier, later in zip(chunks, chunks[1:]):
            assert later.start_char > earlier.start_char

    def test_paragraph_longer_than_max_is_split_by_sentence(self) -> None:
        sentence = "This is a sentence with several words in it."
        single_paragraph = " ".join([sentence] * 60)

        chunks = list(chunk_text(single_paragraph, source_id="doc", **WINDOW))

        assert len(chunks) > 1
        assert all(c.token_count <= WINDOW["max_tokens"] for c in chunks)


class TestInvalidArguments:
    def test_min_above_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_tokens"):
            list(chunk_text("text", source_id="doc", min_tokens=500, max_tokens=100))

    def test_non_positive_max_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            list(chunk_text("text", source_id="doc", min_tokens=0, max_tokens=0))

    @pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5])
    def test_out_of_range_overlap_raises(self, ratio: float) -> None:
        with pytest.raises(ValueError, match="overlap_ratio"):
            list(
                chunk_text(
                    "text",
                    source_id="doc",
                    min_tokens=1,
                    max_tokens=10,
                    overlap_ratio=ratio,
                )
            )


class TestLaziness:
    def test_chunk_text_is_a_generator(self) -> None:
        """Nothing should be computed until the caller iterates."""
        result = chunk_text(make_document(), source_id="doc", **WINDOW)

        assert not isinstance(result, list)
        assert isinstance(next(iter(result)), Chunk)
