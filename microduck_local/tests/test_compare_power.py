"""The instrument has to say what it could not have seen.

Every "measured off" verdict in this repo's soccer work was reported as a
p-value alone, over batteries of 12-24 seeds.  A p-value alone cannot tell
"no effect" from "no instrument", and the measurement that settles which is
the minimum detectable effect -- which `compare_pitch.py` was already
printing, in the `95%` column, unread.  Over the thirteen A/B batteries on
disk the median MDE is 28% of baseline on kicks and 19% on ballAdvance, so a
real 10% improvement was invisible by construction and came out as a null.

These lock the reading rules that stop that: the MDE identity, the three-way
verdict, the seed budget, and the measured fact that the paired design buys
essentially nothing here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_pitch import (  # noqa: E402
    UNQUOTABLE,
    load,
    paired,
    pairing_gain,
    seeds_for,
    t_ppf975,
    t_sf,
    value,
    verdict,
)

RUNS = Path(__file__).resolve().parents[1] / "runs"


def test_half_width_is_exactly_the_minimum_detectable_effect():
    """The MDE is not a new statistic -- it is the 95% half-width, which this
    script always printed.  A difference is significant exactly when it
    exceeds it, so nudging the mean either side of it must flip p at 0.05."""
    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 2.0, 24)
    # A difference series with a fixed spread and a mean we can slide.  Shifting
    # it does not change the spread, so the half-width stays put while p moves.
    d0 = rng.normal(0.0, 2.0, 24)
    d0 -= d0.mean()
    _, half, _, _ = paired(a, a + d0)

    _, _, p_under, _ = paired(a, a + d0 + half * 0.95)
    _, _, p_over, _ = paired(a, a + d0 + half * 1.05)
    assert p_under > 0.05 >= p_over, (p_under, p_over, half)


def test_a_small_real_effect_under_the_mde_is_no_result_not_null():
    """The failure this whole file exists for: a genuine improvement smaller
    than the battery can resolve must NOT be written up as a null."""
    rng = np.random.default_rng(1)
    a = rng.normal(6.0, 4.0, 24)                      # kicks/run, real spread
    b = a * 1.10 + rng.normal(0.0, 4.0, 24)           # a true +10%
    _, half, p, _ = paired(a, b)
    pct = 100.0 * half / abs(a.mean())

    assert p > 0.05, "the effect is genuinely undetectable at this size"
    assert pct > 15.0, f"MDE {pct:.0f}% should be far too wide to deny a 10% effect"
    assert verdict(p, pct) == "NO RESULT"


def test_a_tight_zero_effect_is_a_real_null():
    """Symmetrically: when the battery COULD have seen it and did not, `null`
    is the honest word, and the rule must still say so."""
    rng = np.random.default_rng(2)
    a = rng.normal(10.0, 0.2, 24)
    b = a + rng.normal(0.0, 0.2, 24)
    _, half, p, _ = paired(a, b)
    pct = 100.0 * half / abs(a.mean())

    assert p > 0.05 and pct <= 15.0
    assert verdict(p, pct) == "null"


def test_a_resolved_effect_is_called_an_effect():
    rng = np.random.default_rng(3)
    a = rng.normal(10.0, 0.5, 24)
    assert verdict(*(lambda r: (r[2], 100.0 * r[1] / 10.0))(paired(a, a + 2.0))) == "effect"


@pytest.mark.parametrize("mde_pct,expect", [(5.0, "null"), (14.9, "null"),
                                            (15.1, "NO RESULT"), (300.0, "NO RESULT")])
def test_the_tight_threshold_is_where_null_becomes_no_result(mde_pct, expect):
    assert verdict(0.5, mde_pct) == expect


def test_a_significant_row_is_an_effect_however_wide_its_mde():
    assert verdict(0.01, 900.0) == "effect"


def test_pairing_gain_is_one_when_the_arms_are_independent():
    """What this harness actually measures: the sim diverges, so the shared
    seed cancels nothing and the gain sits at 1."""
    rng = np.random.default_rng(4)
    a, b = rng.normal(6.0, 4.0, 400), rng.normal(6.0, 4.0, 400)
    assert pairing_gain(a, b) == pytest.approx(1.0, abs=0.1)


def test_pairing_gain_rises_when_the_seed_really_does_carry_over():
    """And it is kept because it can still pay for a knob that barely fires
    -- `t9 hunt` measured r = 0.6.  A shared layout term must show up."""
    rng = np.random.default_rng(5)
    layout = rng.normal(0.0, 4.0, 400)
    a = 6.0 + layout + rng.normal(0.0, 1.0, 400)
    b = 6.0 + layout + rng.normal(0.0, 1.0, 400)
    assert pairing_gain(a, b) > 2.0


def test_seeds_for_scales_as_the_square_of_the_effect_ratio():
    """MDE falls as 1/sqrt(n), so resolving half the effect costs 4x."""
    n, base = 24, 6.0
    half = 0.28 * base
    assert seeds_for(half, n, base, 28.0) == pytest.approx(n, abs=1)
    assert seeds_for(half, n, base, 14.0) == pytest.approx(4 * n, abs=2)
    assert seeds_for(half, n, base, 7.0) == pytest.approx(16 * n, abs=4)


def test_seeds_for_is_degenerate_safe():
    assert seeds_for(float("inf"), 24, 6.0) == 0
    assert seeds_for(1.0, 24, 0.0) == 0


def test_ball_progress_is_unquotable():
    """MDE has never been under 100% of baseline on a real battery -- the
    median seed budget for a 10% change is ~21,000."""
    assert "ballProgress" in UNQUOTABLE


@pytest.mark.parametrize("base,arm", [("t7-base3", "t7-color3"), ("t8-base", "t8-comp"),
                                      ("t9-nohunt", "t9-hunt"), ("led2-base24", "led2-i8-24")])
def test_the_real_batteries_could_not_have_resolved_a_tenth_of_the_kick_rate(base, arm):
    """The finding itself, locked to the batteries it was measured on: at the
    size these ran, a 10% change in kicks needs seeds in the hundreds.  Read
    off the real pair, so the difference spread is the one that was actually
    there -- not a synthetic arm, which would be perfectly correlated and hide
    exactly the divergence that costs the power.  If a future change to the
    pitch makes this false, that is a WIN: re-measure this test, don't delete
    it."""
    pa, pb = RUNS / f"{base}.jsonl", RUNS / f"{arm}.jsonl"
    if not (pa.exists() and pb.exists()):
        pytest.skip(f"{base}/{arm} not on disk")
    A, B = load(str(pa)), load(str(pb))
    seeds = sorted(set(A) & set(B))
    x = np.array([value(A[s], "kickCount", "sum", None) or 0.0 for s in seeds])
    y = np.array([value(B[s], "kickCount", "sum", None) or 0.0 for s in seeds])
    assert len(seeds) >= 12 and x.mean() > 0

    _, half, p, _ = paired(x, y)
    pct = 100.0 * half / x.mean()
    assert pct > 15.0, f"{base} vs {arm}: MDE {pct:.0f}% -- re-measure this test"
    assert verdict(p, pct) in {"effect", "NO RESULT"}, "a null here would be unearned"
    assert seeds_for(half, len(seeds), x.mean(), 10.0) > 80


@pytest.mark.parametrize("base,arm", [("t7-base3", "t7-color3"), ("t8-base", "t8-comp"),
                                      ("led2-base24", "led2-i8-24")])
def test_the_shared_seeds_bought_essentially_nothing_on_the_real_batteries(base, arm):
    """Why the MDE is that wide: the pairing the docstring used to lean on is
    decorative.  Measured gain sits at ~1, so the seeds cancel no variance."""
    pa, pb = RUNS / f"{base}.jsonl", RUNS / f"{arm}.jsonl"
    if not (pa.exists() and pb.exists()):
        pytest.skip(f"{base}/{arm} not on disk")
    A, B = load(str(pa)), load(str(pb))
    seeds = sorted(set(A) & set(B))
    x = np.array([value(A[s], "kickCount", "sum", None) or 0.0 for s in seeds])
    y = np.array([value(B[s], "kickCount", "sum", None) or 0.0 for s in seeds])
    assert pairing_gain(x, y) < 1.35, "pairing started paying -- re-measure the docstring claim"


def test_every_field_the_table_prints_has_a_verdict_rule():
    """A metric can be an effect, a null, no result, or unquotable -- never
    printed bare, which is how the nulls got quoted."""
    for pct in (0.0, 5.0, 15.0, 50.0, 1e6):
        for p in (0.0, 0.049, 0.051, 1.0):
            assert verdict(p, pct) in {"effect", "null", "NO RESULT"}


def test_mde_is_finite_for_a_battery_with_no_spread():
    a = np.full(8, 3.0)
    d, half, p, _ = paired(a, a)
    assert d == 0.0 and math.isfinite(half) and p == 1.0


# --- Student's t, written out because scipy is not a dependency here -------
# The old fallback took the half-width from a t-table and the p-value from
# `erfc`, i.e. the NORMAL.  Since scipy has never been installed in this
# workspace, that mixed pair is what produced every soccer p-value on record.

@pytest.mark.parametrize("df,want", [(1, 12.706), (2, 4.303), (5, 2.571), (11, 2.201),
                                     (23, 2.069), (30, 2.042), (60, 2.000), (200, 1.972)])
def test_t_critical_values_match_the_published_table(df, want):
    assert t_ppf975(df) == pytest.approx(want, abs=0.001)


def test_t_tends_to_the_normal_in_the_limit():
    assert t_ppf975(10_000_000) == pytest.approx(1.960, abs=0.002)
    assert 2.0 * t_sf(1.96, 10_000_000) == pytest.approx(0.05, abs=0.001)


@pytest.mark.parametrize("t,df,want", [(2.0, 23, 0.0571), (1.0, 10, 0.3409),
                                       (3.0, 5, 0.0301), (0.0, 8, 1.0)])
def test_t_two_sided_p_matches_the_published_table(t, df, want):
    assert 2.0 * t_sf(t, df) == pytest.approx(want, abs=0.001)


def test_the_normal_would_have_called_a_borderline_row_significant():
    """The size of the old error, at the sizes these batteries run: t = 2.0 on
    24 seeds is p = 0.057 under Student's t and p = 0.046 under the normal.
    One is a null, the other is an `effect`, and the repo was taking the
    second."""
    normal_p = math.erfc(2.0 / math.sqrt(2))
    assert normal_p < 0.05 <= 2.0 * t_sf(2.0, 23)


def test_the_interval_and_the_test_now_use_one_distribution():
    """The identity the whole verdict rests on: significant exactly when the
    difference exceeds the half-width.  It was false while the half came from
    a t-table and p came from the normal."""
    rng = np.random.default_rng(7)
    for n in (6, 12, 24, 48):
        a = rng.normal(10.0, 2.0, n)
        d0 = rng.normal(0.0, 2.0, n)
        d0 -= d0.mean()
        _, half, _, _ = paired(a, a + d0)
        _, _, p_under, _ = paired(a, a + d0 + half * 0.95)
        _, _, p_over, _ = paired(a, a + d0 + half * 1.05)
        assert p_under > 0.05 >= p_over, (n, p_under, p_over)


def test_t_survival_is_monotone_and_bounded():
    for df in (1, 5, 23):
        vals = [t_sf(t, df) for t in (0.0, 0.5, 1.0, 2.0, 4.0, 10.0)]
        assert vals == sorted(vals, reverse=True)
        assert all(0.0 <= v <= 0.5 for v in vals)


# --- the audit that produced the diagnosis --------------------------------

from audit_power import METRICS, PAIRS, normal_p, series  # noqa: E402


def test_normal_p_is_anti_conservative_against_students_t():
    """The size of the old error, across the sizes these batteries run: the
    normal always returns a SMALLER p, so it only ever over-claims."""
    for t in (1.5, 2.0, 2.5, 3.0):
        for df in (5, 11, 23, 47):
            assert normal_p(t) < 2.0 * t_sf(t, df)


def test_the_untrustworthy_band_at_24_seeds():
    """Concretely: a paired reading at 24 seeds whose reported p fell between
    0.039 and 0.05 was significant under the normal and is not under t.  Three
    such readings survive on disk (t4 clamp spread, t7 colour possession,
    t8 comp goals)."""
    df = 23
    edge = t_ppf975(df)
    assert normal_p(edge) == pytest.approx(0.0386, abs=0.001)
    assert 2.0 * t_sf(edge, df) == pytest.approx(0.05, abs=1e-6)


def test_the_audit_pairs_all_name_real_metrics():
    fields = {f for f, _ in METRICS}
    assert {"kickCount", "ballAdvance", "falls", "goals", "possession"} <= fields


def test_series_returns_none_rather_than_raising_on_an_old_row():
    """Batteries written before a field existed must be skipped, not crash the
    audit -- half these files predate several of the metrics."""
    rows = {0: {"seed": 0}, 1: {"seed": 1}}
    assert series(rows, [0, 1], "falls", "sum") is None
    assert series(rows, [0, 1], "possession", "sum") is None


@pytest.mark.parametrize("label,base,arm", PAIRS)
def test_every_audited_pair_is_readable_if_it_is_on_disk(label, base, arm):
    pa, pb = RUNS / f"{base}.jsonl", RUNS / f"{arm}.jsonl"
    if not (pa.exists() and pb.exists()):
        pytest.skip(f"{label}: files not on disk")
    A, B = load(str(pa)), load(str(pb))
    assert len(set(A) & set(B)) >= 3, f"{label}: arms share too few seeds to compare"


def test_the_audit_runs_end_to_end(capsys):
    import audit_power
    old = sys.argv
    try:
        sys.argv = ["audit_power.py", "--old-normal"]
        audit_power.main()
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert "variance reduction" in out and "med MDE%" in out


# --- the gym carries the same rule on its proportion test -----------------

from kick_gym import (  # noqa: E402
    TARGET_PP,
    TIGHT_PP,
    swings_for,
    verdict_prop,
)
from kick_gym import two_proportions as gym_two_proportions  # noqa: E402


def test_the_gym_mde_is_the_boundary_of_its_own_significance():
    """Same identity as the pitch: the whiff shift is significant exactly when
    it exceeds the MDE.  Walk a second arm's count until p crosses 0.05 and
    check the shift crosses the MDE at the same event."""
    n = 400
    x1 = 100
    crossed_p = crossed_mde = None
    for x2 in range(100, 200):
        d, p, mde = gym_two_proportions(x1, n, x2, n)
        if crossed_p is None and p < 0.05:
            crossed_p = x2
        if crossed_mde is None and abs(d) > mde:
            crossed_mde = x2
    assert crossed_p is not None and crossed_p == crossed_mde


def test_a_small_real_whiff_shift_on_few_swings_is_no_result():
    """The gym's version of the failure: 40 swings an arm cannot see 10 points,
    so it must not report a null."""
    d, p, mde = gym_two_proportions(10, 40, 14, 40)
    assert p > 0.05 and mde > TIGHT_PP
    assert verdict_prop(p, mde) == "NO RESULT"


def test_the_same_shift_on_enough_swings_resolves():
    d, p, mde = gym_two_proportions(100, 400, 140, 400)
    assert verdict_prop(p, mde) == "effect"


def test_a_tight_zero_shift_is_a_real_null():
    d, p, mde = gym_two_proportions(250, 1000, 252, 1000)
    assert p > 0.05 and mde <= TIGHT_PP
    assert verdict_prop(p, mde) == "null"


def test_swings_for_scales_quadratically():
    _, _, mde = gym_two_proportions(100, 400, 100, 400)
    assert swings_for(mde, 400, 400, mde) == pytest.approx(400, abs=2)
    assert swings_for(mde, 400, 400, mde / 2) == pytest.approx(1600, abs=8)


def test_gym_degenerate_counts_do_not_claim_a_null():
    for args in ((0, 0, 0, 0), (5, 5, 5, 5), (0, 10, 0, 10)):
        d, p, mde = gym_two_proportions(*args)
        assert verdict_prop(p, mde) != "null" or mde <= TIGHT_PP
    assert swings_for(float("inf"), 10, 10) == 0


def test_the_gym_thresholds_are_stricter_than_the_pitch_because_events_are_cheap():
    """The gym buys ~19x the events per CPU-second, so it has no excuse for a
    loose null: 8 points, against the pitch's 15% of baseline."""
    assert 0 < TIGHT_PP < 0.15 and 0 < TARGET_PP <= TIGHT_PP * 2


# --- a knob that changes nothing is BROKEN, not null (playbook rule 0) -----
# Found by this session's own contest arm: `contest_margin=0.15` reproduced
# the baseline episode for episode, because the rule is gated on `use_color`,
# which is off by default.  The MDE machinery happily called that a null.

from kick_gym import is_identical, outcome_key  # noqa: E402


def _rows(travels):
    return [{"swing": t is not None, "travel": t, "whiff": (t or 0.0) < 0.10}
            for t in travels]


def test_an_arm_that_reproduces_the_baseline_is_flagged_broken():
    base = _rows([0.2, None, 0.05, 0.31])
    assert is_identical(base, _rows([0.2, None, 0.05, 0.31]))


def test_a_single_changed_episode_is_enough_to_be_a_real_arm():
    base = _rows([0.2, None, 0.05, 0.31])
    assert not is_identical(base, _rows([0.2, None, 0.05, 0.32]))


def test_a_changed_swing_pattern_is_a_real_arm():
    base = _rows([0.2, None, 0.05])
    assert not is_identical(base, _rows([0.2, 0.4, 0.05]))


def test_differing_episode_counts_are_never_called_identical():
    assert not is_identical(_rows([0.2, 0.3]), _rows([0.2, 0.3, 0.4]))


def test_outcome_key_ignores_fields_that_are_not_the_physics():
    a = [{"swing": True, "travel": 0.25, "arm": "x", "seed": 1}]
    b = [{"swing": True, "travel": 0.25, "arm": "y", "seed": 2}]
    assert outcome_key(a) == outcome_key(b)


def test_a_no_swing_episode_carries_no_travel_into_the_key():
    assert outcome_key([{"swing": False, "travel": 9.9}]) == ((False, None),)
