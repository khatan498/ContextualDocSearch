"""Tests for app.retrieval.fusion."""

import pytest

from app.indexing.hits import IndexHit
from app.retrieval.fusion import reciprocal_rank_fusion


def hit(chunk_id: str, score: float = 1.0) -> IndexHit:
    """A hit whose id doubles as its source, e.g. ``"doc.pdf::0"``."""
    source_id, index = chunk_id.split("::")
    return IndexHit(
        chunk_id=chunk_id,
        source_id=source_id,
        chunk_index=int(index),
        text=f"text of {chunk_id}",
        score=score,
    )


def ids(fused: list) -> list[str]:
    return [f.chunk_id for f in fused]


class TestScoring:
    def test_first_in_both_lists_scores_two_over_k_plus_one(self) -> None:
        fused = reciprocal_rank_fusion(
            {"vector": [hit("a::0")], "keyword": [hit("a::0")]}, k=60
        )

        assert fused[0].rrf_score == pytest.approx(2 / 61)

    def test_each_list_contributes_its_own_rank(self) -> None:
        fused = reciprocal_rank_fusion(
            {
                "vector": [hit("x::0"), hit("a::0")],
                "keyword": [hit("y::0"), hit("z::0"), hit("a::0")],
            },
            k=60,
        )

        a = next(f for f in fused if f.chunk_id == "a::0")
        assert a.rrf_score == pytest.approx(1 / 62 + 1 / 63)
        assert a.ranks == {"vector": 2, "keyword": 3}

    def test_k_changes_the_score(self) -> None:
        fused = reciprocal_rank_fusion({"vector": [hit("a::0")]}, k=10)

        assert fused[0].rrf_score == pytest.approx(1 / 11)


class TestRankNotScore:
    """The whole point of RRF: raw scores from different indexes are never
    compared, only positions. Cosine sits in [-1, 1]; BM25 is unbounded."""

    def test_raw_scores_do_not_affect_the_result(self) -> None:
        modest = reciprocal_rank_fusion(
            {
                "vector": [hit("a::0", 0.9), hit("b::0", 0.8)],
                "keyword": [hit("b::0", 4.0), hit("c::0", 3.0)],
            }
        )
        extreme = reciprocal_rank_fusion(
            {
                "vector": [hit("a::0", 0.01), hit("b::0", -0.5)],
                "keyword": [hit("b::0", 9000.0), hit("c::0", 0.001)],
            }
        )

        assert ids(modest) == ids(extreme)
        assert [f.rrf_score for f in modest] == [f.rrf_score for f in extreme]

    def test_a_huge_score_in_one_list_does_not_dominate(self) -> None:
        fused = reciprocal_rank_fusion(
            {
                "vector": [hit("a::0", 0.5), hit("b::0", 0.4)],
                "keyword": [hit("b::0", 1_000_000.0), hit("a::0", 0.1)],
            }
        )

        # Both chunks hold ranks {1, 2}, so they tie on score whatever the
        # BM25 numbers say.
        assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)


class TestAgreement:
    def test_second_in_both_beats_first_in_one(self) -> None:
        # The Phase 3 regression, in miniature. For "how long is the warranty"
        # BM25 ranks an unrelated clause first while the warranty chunk sits
        # second in both lists. Fusion must put the warranty chunk on top.
        fused = reciprocal_rank_fusion(
            {
                "vector": [hit("warranty::1"), hit("warranty::0"), hit("other::5")],
                "keyword": [hit("services::33"), hit("warranty::0"), hit("other::5")],
            }
        )

        assert fused[0].chunk_id == "warranty::0"
        assert ids(fused).index("warranty::0") < ids(fused).index("services::33")


class TestCoverage:
    def test_a_chunk_found_by_one_list_still_appears(self) -> None:
        fused = reciprocal_rank_fusion(
            {"vector": [hit("a::0")], "keyword": [hit("b::0")]}
        )

        assert set(ids(fused)) == {"a::0", "b::0"}
        b = next(f for f in fused if f.chunk_id == "b::0")
        assert b.ranks == {"keyword": 1}

    def test_one_empty_list_keeps_the_others_order(self) -> None:
        # "INV-2024" matches no BM25 term at all, so fusion degenerates to the
        # vector order — still fused, just with a single contributor.
        vector = [hit("c::0"), hit("a::0"), hit("b::0")]

        fused = reciprocal_rank_fusion({"vector": vector, "keyword": []})

        assert ids(fused) == ["c::0", "a::0", "b::0"]

    def test_all_lists_empty(self) -> None:
        assert reciprocal_rank_fusion({"vector": [], "keyword": []}) == []

    def test_no_lists(self) -> None:
        assert reciprocal_rank_fusion({}) == []

    def test_every_chunk_appears_once(self) -> None:
        fused = reciprocal_rank_fusion(
            {
                "vector": [hit("a::0"), hit("b::0")],
                "keyword": [hit("b::0"), hit("a::0")],
            }
        )

        assert sorted(ids(fused)) == ["a::0", "b::0"]

    def test_hit_fields_are_carried_through(self) -> None:
        fused = reciprocal_rank_fusion({"vector": [hit("doc.pdf::3")]})

        assert fused[0].source_id == "doc.pdf"
        assert fused[0].chunk_index == 3
        assert fused[0].text == "text of doc.pdf::3"


class TestEdgeCases:
    def test_duplicate_in_one_list_keeps_its_best_rank(self) -> None:
        fused = reciprocal_rank_fusion(
            {"vector": [hit("a::0"), hit("b::0"), hit("a::0")]}
        )

        a = next(f for f in fused if f.chunk_id == "a::0")
        assert a.ranks == {"vector": 1}
        assert a.rrf_score == pytest.approx(1 / 61)

    def test_ties_break_by_best_rank_then_id(self) -> None:
        # c and d mirror each other, so they hold identical rank sets.
        fused = reciprocal_rank_fusion(
            {
                "vector": [hit("d::0"), hit("c::0")],
                "keyword": [hit("c::0"), hit("d::0")],
            }
        )

        # Same score, same best rank (1), so chunk_id decides.
        assert ids(fused) == ["c::0", "d::0"]

    def test_tie_prefers_the_chunk_with_a_better_single_rank(self) -> None:
        # e holds ranks {1, 4}, f holds {2, 2}: with k=2 both score
        # 1/3 + 1/6 = 1/4 + 1/4 = 0.5, so the tiebreak is exercised.
        fused = reciprocal_rank_fusion(
            {
                "vector": [hit("e::0"), hit("f::0")],
                "keyword": [hit("x::0"), hit("f::0"), hit("y::0"), hit("e::0")],
            },
            k=2,
        )

        e = next(f for f in fused if f.chunk_id == "e::0")
        f = next(f for f in fused if f.chunk_id == "f::0")
        assert e.rrf_score == pytest.approx(f.rrf_score)
        assert ids(fused).index("e::0") < ids(fused).index("f::0")

    def test_output_is_deterministic(self) -> None:
        lists = {
            "vector": [hit(f"v::{i}") for i in range(10)],
            "keyword": [hit(f"k::{i}") for i in range(10)],
        }

        assert ids(reciprocal_rank_fusion(lists)) == ids(reciprocal_rank_fusion(lists))

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_is_rejected(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            reciprocal_rank_fusion({"vector": [hit("a::0")]}, k=k)
