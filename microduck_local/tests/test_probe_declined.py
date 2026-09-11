"""The declined-swing probe must not move the thing it measures (12au (3)).

`scripts/probe_declined_swings.py` answers "does the corrected kick sidecar
make the selector refuse swings?" by running the gym and, at every plan,
re-scoring the SAME candidate fan under the OTHER arm's exit model. That second
roll-out draws from `Chase._kick_rng` — the generator the live brain is about to
draw from — so if its state is not restored the probe changes every subsequent
kick it is meant to observe, and the arm it reports is not the arm it ran.

Locked here:

  * the counterfactual consumes NO randomness: with it on and off, every event
    row of the same seed is identical, field for field.
  * installing the patch twice in one process wraps the PRISTINE method, not
    the patch. Nesting it tripled the select census and would have inflated
    every "per plan" rate by the nesting depth (caught by `--selfcheck`
    before any block was run, and kept honest here).
  * the rows carry what the item is quoted from: the per-episode census keys
    the "reachable set" table is computed out of, and a per-event row with the
    selector's own price on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

probe = pytest.importorskip("probe_declined_swings")

# The exits of the two arms 12au compared: the pair the SIDECARS carry today
# (read, not pinned — the owner set the left to its in-play +0.209 on
# 2026-09-11, which is when a pinned constant went red) and the pre-correction
# bench pair as the counterfactual. Named so a test failure says which it ran.
import json as _json
from pathlib import Path as _Path
_KICKS = _Path(__file__).resolve().parents[1] / "policies" / "kick"
SHIPPED = tuple(_json.loads((_KICKS / f"kick_{f}.json").read_text())["exit_rad"] for f in ("left", "right"))
CORRECTED = (-0.225, -0.036)   # the bench pair the shipped sidecar carried before 9ca8d9d

EPISODE_KEYS = {"kind", "ep", "seed", "swings", "declines", "pushes", "t_first_swing",
                "place", "place_board", "selects", "select_none", "select_pinned",
                "select_p_own_pos", "cf_none", "cf_diff_foot", "cf_diff_line",
                "exits_live", "cf_exits"}


def _events(rows):
    """Every row minus the counterfactual, which is the only thing that may
    differ between the two runs."""
    return [{k: v for k, v in r.items() if not k.startswith("cf_") and k != "sel"} for r in rows]


@pytest.fixture(scope="module")
def pair():
    plain = probe.run(0, 2, 0.8, 0.0, None)
    with_cf = probe.run(0, 2, 0.8, 0.0, CORRECTED)
    return plain, with_cf


def test_the_patch_does_not_outlive_a_run(pair):
    """After `run`, the class method is the pristine one — a leaked wrapper
    doubles the selector fan for every later brain in the process (CI went
    red on `test_spot_reach` for exactly this)."""
    from microduck_local.brain.controllers import Chase
    assert probe._CF["orig"] is None
    assert Chase._select_kick_line.__name__ == "_select_kick_line"


def test_the_counterfactual_consumes_no_randomness(pair):
    plain, with_cf = pair
    assert _events(plain) == _events(with_cf)


def test_installing_the_patch_twice_does_not_nest_it(pair):
    """Two `run`s in one process: the second must not see a select census
    inflated by wrapping the patch."""
    plain, with_cf = pair
    a = [r for r in plain if r["kind"] == "episode"]
    b = [r for r in with_cf if r["kind"] == "episode"]
    assert a and len(a) == len(b)
    assert [r["selects"] for r in a] == [r["selects"] for r in b]


def test_the_episode_row_carries_the_reachable_set_census(pair):
    plain, _ = pair
    eps = [r for r in plain if r["kind"] == "episode"]
    assert eps, "no episode rows"
    for r in eps:
        assert set(r) == EPISODE_KEYS
        assert r["selects"] >= r["select_none"] >= 0
        assert r["selects"] >= r["select_pinned"] >= 0
        assert tuple(r["exits_live"]) == SHIPPED      # the shipped pair, read off the CONSTRUCTED brain


def test_every_event_row_carries_the_selector_price_and_a_fate(pair):
    _, with_cf = pair
    evs = [r for r in with_cf if r["kind"] in ("swing", "decline", "push")]
    assert evs, "no event rows"
    for r in evs:
        assert {"t", "ball", "mouth_range", "mouth_bearing", "ball_board", "duck", "sel"} <= set(r)
        assert {"p_own", "p_goal", "value", "detour", "aim_max",
                "cf_p_own", "cf_p_goal", "cf_value", "cf_foot"} <= set(r["sel"])
    # …and at least one of them closed its 4 s window inside the episode
    assert any("fate_dx" in r for r in evs)
