import pytest

from services.search.fusion import reciprocal_rank_fusion


def test_documented_worked_example():
    bm25 = ["A", "B"]
    vector = ["C", "X", "A"]
    fused = dict(reciprocal_rank_fusion([bm25, vector], k=60))
    assert fused["A"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["B"] == pytest.approx(1 / 62)
    assert fused["C"] == pytest.approx(1 / 61)
    order = [d for d, _ in reciprocal_rank_fusion([bm25, vector], k=60)]
    assert order[:3] == ["A", "C", "B"]


def test_documents_in_both_lists_beat_single_list_winners():
    fused = reciprocal_rank_fusion([["x", "both"], ["y", "both"]], k=60)
    assert fused[0][0] == "both"


def test_weights_and_k():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=10, weights=[2.0, 1.0]))
    assert fused["a"] == pytest.approx(2 / 11) and fused["b"] == pytest.approx(1 / 11)


def test_empty_and_deterministic_ties():
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([["b"], ["a"]]) == reciprocal_rank_fusion([["b"], ["a"]])
