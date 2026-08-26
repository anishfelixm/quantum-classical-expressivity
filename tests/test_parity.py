"""
PARAMETER PARITY.

The primary comparison claims the quantum and classical heads have EQUAL
capacity. If that stops being true, every "no advantage at equal parameters"
statement in the manuscript becomes false, and nothing else in the pipeline
would notice - parameter counts are never asserted at runtime.

These tests fail loudly instead.

WHAT IS CHECKED
---------------
1. At d=4, quantum_vqc, quantum_reupload, matched_param_fullrank and
   low_rank(rank=2) all have exactly 24 head parameters.
2. low_rank(rank=2) holds parity at d=8 (48) and d=16 (96) too, which is why it
   exists - MatchedParamFullRankHead cannot.
3. MatchedParamFullRankHead is DOCUMENTED as failing above d=4. That is asserted
   here so the limitation cannot be quietly forgotten and used anyway.
4. quantum_reupload has the same parameter count but a wider spectrum, which is
   the entire basis of the Q4 contrast.
5. matched_param is rank-limited, and stays flagged as such.

Run:  python -m pytest tests/test_parity.py -v -s
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.registry import build_arm, count_head_params   # noqa: E402


def head_params(arm, d, **kw):
    m = build_arm(arm, d=d, num_classes=2, seed=42, build_backbone=False, **kw)
    return count_head_params(m)["head"], m


# ------------------------------------------------------------------ d = 4
PARITY_ARMS_D4 = [
    ("quantum_vqc", {}),
    ("quantum_reupload", {}),
    ("matched_param_fullrank", {}),
    ("low_rank", {"head_rank": 2}),
    ("matched_param", {}),
]


@pytest.mark.parametrize("arm,kw", PARITY_ARMS_D4)
def test_head_params_are_24_at_d4(arm, kw):
    """
    24 = 3 * L * d with L=2, d=4. Every arm in the parity family must hit it
    exactly - not approximately, not on average.
    """
    n, _ = head_params(arm, 4, **kw)
    print(f"\n  {arm:24s} d=4  head params = {n}")
    assert n == 24, f"{arm} has {n} head parameters at d=4, expected 24"


def test_all_parity_arms_agree_with_each_other():
    """A pairwise check, so a failure names the disagreement rather than a number."""
    counts = {arm: head_params(arm, 4, **kw)[0] for arm, kw in PARITY_ARMS_D4}
    print("\n  " + "  ".join(f"{a}={n}" for a, n in counts.items()))
    assert len(set(counts.values())) == 1, f"parity broken across arms: {counts}"


# ------------------------------------------------------------------ any d
@pytest.mark.parametrize("d,expected", [(4, 24), (8, 48), (16, 96)])
def test_low_rank_holds_parity_at_every_dimension(d, expected):
    """
    2*d*rank + 2*d = 6d  <=>  rank = 2, independent of d.

    This is the property that unblocks d=8 and d=16: the dense full-rank head
    cannot match 6d above d=4 (at d=8 a d x d matrix alone is 64 > 48).
    """
    n, m = head_params("low_rank", d, head_rank=2)
    desc = m.head.describe()
    print(f"\n  low_rank rank=2 d={d:<3d} params={n} target={desc['target_params']}")
    assert n == expected
    assert desc["exact_match"], f"low_rank(rank=2) not exact at d={d}"


def test_dense_fullrank_head_fails_above_d4_as_documented():
    """
    NOT a bug - a documented limitation, asserted so it cannot be forgotten.

    If this ever starts passing, MatchedParamFullRankHead was changed and the
    d>4 restriction in its docstring is stale.
    """
    for d in (8, 16):
        n, m = head_params("matched_param_fullrank", d)
        target = m.head.target_params
        print(f"\n  matched_param_fullrank d={d:<3d} params={n} target={target} "
              f"(mismatch expected)")
        assert not m.head.exact_match, (
            f"matched_param_fullrank unexpectedly matches at d={d}; the "
            f"documented d=4-only limitation may be stale")


# ------------------------------------------------------------------ capacity axis
@pytest.mark.parametrize("rank,expected", [(0, 8), (1, 16), (2, 24), (4, 40), (8, 72)])
def test_low_rank_capacity_axis_at_d4(rank, expected):
    """The capacity sweep's x-axis: 2*d*rank + 2*d at d=4."""
    n, _ = head_params("low_rank", 4, head_rank=rank)
    print(f"\n  low_rank rank={rank:<2d} d=4  params={n}")
    assert n == expected


def test_low_rank_is_never_rank_limited():
    """
    I + U V^T is generically invertible at every rank including 0, so capacity
    varies without rank varying. That separation is the whole point of using
    this head for the mechanism test rather than a width sweep.
    """
    for rank in (0, 1, 2, 4, 8):
        _, m = head_params("low_rank", 4, head_rank=rank)
        assert m.head.rank_limited is False


def test_matched_param_is_flagged_rank_limited():
    """The diagnostic arm must keep advertising its own defect."""
    _, m = head_params("matched_param", 4)
    print(f"\n  matched_param d=4 hidden_width={m.head.hidden_width} "
          f"rank_limited={m.head.rank_limited}")
    assert m.head.rank_limited is True
    for d in (8, 16):
        _, m = head_params("matched_param", d)
        assert m.head.rank_limited is True, (
            f"matched_param must stay flagged at d={d}; comparisons against it "
            f"above d=4 are not valid")


# ------------------------------------------------------------------ spectrum
def test_reupload_matches_params_but_not_spectrum():
    """
    The Q4 contrast in one assertion: same parameters, wider spectrum. If the
    parameter counts ever diverge, the arms stop being a controlled contrast.
    """
    _, a = head_params("quantum_vqc", 4)
    _, b = head_params("quantum_reupload", 4)
    print(f"\n  quantum_vqc      params={a.head.n_quantum_params} "
          f"spectrum={a.head.spectrum_size}")
    print(f"  quantum_reupload params={b.head.n_quantum_params} "
          f"spectrum={b.head.spectrum_size}")
    assert a.head.n_quantum_params == b.head.n_quantum_params == 24
    assert a.head.spectrum_size == 81
    assert b.head.spectrum_size == 625


# ------------------------------------------------------------------ capacity share
def test_frozen_bottleneck_makes_the_head_dominant():
    """
    The motivation for 12_bottleneck_ablation, as an executable statement.

    With a learned projection the head holds ~2% of the trainable budget, so the
    "frozen backbone" experiment is not isolating the head. Freezing the
    projection raises the head's share above half.
    """
    _, learned = head_params("quantum_vqc", 4, bottleneck_policy="learned")
    _, frozen = head_params("quantum_vqc", 4, bottleneck_policy="random")
    cl, cf = learned.capacity_report(), frozen.capacity_report()
    print(f"\n  learned bottleneck: head {100 * cl['head_share']:.1f}% "
          f"of {cl['total']} trainable")
    print(f"  frozen  bottleneck: head {100 * cf['head_share']:.1f}% "
          f"of {cf['total']} trainable")
    assert cl["head_share"] < 0.10, "learned projection should dominate capacity"
    assert cf["head_share"] > 0.50, "frozen projection should leave the head dominant"


if __name__ == "__main__":
    for arm, kw in PARITY_ARMS_D4:
        test_head_params_are_24_at_d4(arm, kw)
    for d, e in ((4, 24), (8, 48), (16, 96)):
        test_low_rank_holds_parity_at_every_dimension(d, e)
    test_frozen_bottleneck_makes_the_head_dominant()
    print("\nParity verified.")
