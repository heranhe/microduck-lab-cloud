"""The DUEL, as an EVENT the gym can place and score (roadmap C.4, second half).

C.4's first attempt at the duel rule was judged by a whole-match ledger and
resolved nothing about the duel: the best instrument it had was an MDE of 3% on
possession against a firing rate of 10%, which prices the RUN and not the
event. C.4's own "what would settle it next" is this file's subject — build the
duel as a gym episode and score *the opponent's next touch*.

An instrument that decides a rule's fate has to be locked harder than the rule.
What is locked here:

  * THE PLACEMENT IS WHAT IT CLAIMS. The whole event is its definition: an
    opponent NEARER the ball than this duck, at contact range, while this duck
    goes for it. A draw that ever puts our duck nearer is not a duel, and every
    number measured on it would be about something else.
  * THE POSITIVE CONTROL. Remove the opponent and the same draw must read
    "ours" on essentially every episode. This is the check that separates "the
    opponent took the ball" from "our duck never reaches a ball in six
    seconds", and it is what set `DUEL_S` (58% at 4 s, 94% at 6 s).
  * THE ATTRIBUTION. In a duel both bodies are inside the touch radius, so
    nearest-wins would credit whichever duck the ball was kicked AT. The rule
    is "the ball leaves the body that struck it", and its sign is the one thing
    here that fails silently.
  * THE ROW AND THE BACKWARD COMPATIBILITY. `duel_summ` on a swing row file
    must be `{}` and `exit_summ` on a duel row file must be `{}` — both
    directions, because `scripts/compare_gym.py` is pointed at whatever
    `runs/*.jsonl` is to hand.
  * THE REACHABLE SET, read off the CONSTRUCTED brain, never a fresh
    `ChaseParams()` (verification rule 0). The arm C.4 names — a standoff
    INSIDE `duck_touch` — is precisely the one that can compute itself every
    tick and never reach the elif chain, and the row has to be able to say so.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kick_gym import (  # noqa: E402
    DUEL_APART,
    DUEL_MINE,
    DUEL_S,
    DUEL_THEIRS,
    TOUCH_R,
    _board_rect,
    _place_duel,
    _touch_by,
    duel_summ,
    exit_summ,
    gym_scenario,
    is_identical,
    run_duel,
)

from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer  # noqa: E402
from microduck_local.brain.controllers import ChaseParams  # noqa: E402
from microduck_local.world import World  # noqa: E402


@pytest.fixture(scope="module")
def world():
    sc = gym_scenario(opponents=1)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    return World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=0)


@pytest.fixture(autouse=True)
def _no_ambient_knobs():
    """`run_duel` writes `MICRODUCK_CHASE` into this process (it is designed to
    run in a pool worker). A test that left one set would silently arm every
    test after it."""
    before = os.environ.get("MICRODUCK_CHASE")
    yield
    if before is None:
        os.environ.pop("MICRODUCK_CHASE", None)
    else:
        os.environ["MICRODUCK_CHASE"] = before


# --- the placement is the definition ----------------------------------------

def test_the_opponent_is_always_nearer_the_ball_than_we_are(world):
    """The event, checked on the draw itself over 200 episodes: our duck in
    `DUEL_MINE` of the ball, the opponent in `DUEL_THEIRS`, and the opponent
    strictly nearer. The bands do not overlap, so "nearer" cannot fail — but
    the bands are what a later edit would loosen, so both are asserted."""
    w = world
    rng = np.random.default_rng(7)
    bx_h, by_h = _board_rect(w)
    for _ in range(200):
        q, _v, place = _place_duel(w, rng, ["o0"])
        ball = (float(w.data.qpos[q]), float(w.data.qpos[q + 1]))
        assert ball == pytest.approx(place["ball"], abs=1e-3)
        mine = math.dist(_xy(w, "d0"), ball)
        theirs = math.dist(_xy(w, "o0"), ball)
        assert DUEL_MINE[0] - 0.02 <= mine <= DUEL_MINE[1] + 0.02
        assert DUEL_THEIRS[0] - 0.02 <= theirs <= DUEL_THEIRS[1] + 0.02
        assert theirs < mine, "an opponent that is not nearer the ball is not a duel"
        assert math.dist(_xy(w, "d0"), _xy(w, "o0")) >= DUEL_APART - 0.02
        for did in ("d0", "o0"):
            x, y = _xy(w, did)
            assert abs(x) < bx_h and abs(y) < by_h, f"{did} spawned in the boards"


def test_the_ball_starts_clear_of_every_board(world):
    """An `advance` over the carry window that is really a rebound is not a
    touch's advance. The draw keeps the ball at least 0.5 m off every board."""
    w = world
    rng = np.random.default_rng(11)
    bx_h, by_h = _board_rect(w)
    for _ in range(100):
        q, _v, _p = _place_duel(w, rng, ["o0"])
        bx, by = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
        assert min(bx_h - abs(bx), by_h - abs(by)) >= 0.45


def test_our_duck_is_pointed_at_the_ball_so_it_is_going_for_it(world):
    """"…while this duck goes for it" is half the definition. The spawn yaw is
    the ball's bearing plus the same +-0.25 rad jitter every other placer in
    this file uses."""
    w = world
    rng = np.random.default_rng(3)
    for _ in range(60):
        q, _v, _p = _place_duel(w, rng, ["o0"])
        bx, by = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
        dx, dy = _xy(w, "d0")
        want = math.atan2(by - dy, bx - dx)
        got = w.ducks["d0"].yaw(w.data)
        assert abs(math.atan2(math.sin(got - want), math.cos(got - want))) <= 0.26


def test_the_control_draw_places_our_duck_identically_and_nobody_else(world):
    """The positive control must differ from the duel in ONE thing — that
    nobody contests it. Same seed, same draw for the ball and for our duck."""
    w = world
    a = _place_duel(w, np.random.default_rng(21), ["o0"])
    ball_a, mine_a = a[2]["ball"], a[2]["mine"]
    duck_a = _xy(w, "d0")
    b = _place_duel(w, np.random.default_rng(21), [])
    assert b[2]["ball"] == ball_a and b[2]["mine"] == mine_a
    assert b[2]["theirs"] is None
    assert _xy(w, "d0") == pytest.approx(duck_a, abs=1e-6)


# --- the attribution --------------------------------------------------------

def test_the_touch_is_credited_to_the_body_the_ball_LEAVES(world):
    """The sign that fails silently. Both ducks are inside `TOUCH_R` — which is
    what a duel IS — so nearest-wins would credit whichever duck the ball was
    kicked AT, reversing every row in the file."""
    w = world
    _park(w, "d0", -0.30, 0.0, 0.0)
    _park(w, "o0", 0.15, 0.0, math.pi)
    ball = (0.0, 0.0)
    # The ball running toward -x is running AWAY from o0 and AT d0: o0 hit it.
    assert _touch_by(w, ball, (-1.2, 0.0)) == "o0"
    # ...and the other way round.
    assert _touch_by(w, ball, (1.2, 0.0)) == "d0"


def test_a_ball_nothing_is_near_is_nobodys_touch(world):
    w = world
    _park(w, "d0", -1.2, -1.0, 0.0)
    _park(w, "o0", 1.2, 1.0, math.pi)
    assert _touch_by(w, (0.0, 0.0), (1.0, 0.0)) is None
    # …and a ball that is not moving has no direction to be credited by.
    assert _touch_by(w, (0.0, 0.0), (0.0, 0.0)) is None


def test_only_a_duck_inside_the_touch_radius_can_be_credited(world):
    w = world
    _park(w, "d0", -(TOUCH_R + 0.1), 0.0, 0.0)
    _park(w, "o0", 1.2, 1.0, math.pi)
    assert _touch_by(w, (0.0, 0.0), (1.0, 0.0)) is None
    _park(w, "d0", -(TOUCH_R - 0.15), 0.0, 0.0)
    assert _touch_by(w, (0.0, 0.0), (1.0, 0.0)) == "d0"


# --- the row, and what it is read with --------------------------------------

ROW_COLUMNS = ("duel", "ep", "seed", "arm", "live", "reach", "place", "first", "t_first",
               "ours_touched", "theirs_touched", "t_ours", "t_theirs", "their_advance",
               "our_advance", "falls_us", "falls_them", "state05", "ticks", "fire_ticks",
               "act_ticks", "avoid_ticks", "n_unattr")


def test_every_row_carries_every_column():
    rows = run_duel(seed=0, episodes=2)
    assert len(rows) == 2
    for r in rows:
        assert set(ROW_COLUMNS) <= set(r), sorted(set(ROW_COLUMNS) - set(r))
        assert r["duel"] is True
        assert r["first"] in (None, "ours", "theirs")
        assert (r["t_first"] is None) == (r["first"] is None)
        assert r["ticks"] > 0
        # The state at 0.5 s is one of the brain's, never a number or a blank.
        assert isinstance(r["state05"], str) and r["state05"]


def test_the_reachable_set_is_read_off_the_constructed_brain():
    """Verification rule 0. `assert ChaseParams().duel == 0.3` would pass while
    the brain that ran had 0.0; this reads the value back off the brain the
    episode was driven with, and flags C.4's own variant — a standoff inside
    `duck_touch`, where `avoid` owns the last metres."""
    shipped = run_duel(seed=0, episodes=1)[0]["reach"]
    assert shipped["duel"] == 0.0 and shipped["inside_touch"] is False
    assert shipped["duck_touch"] == pytest.approx(ChaseParams().duck_touch)
    wide = run_duel(seed=0, episodes=1, knobs="duel=0.3")[0]["reach"]
    assert wide["duel"] == pytest.approx(0.3) and wide["inside_touch"] is False
    tight = run_duel(seed=0, episodes=1, knobs="duel=0.15")[0]["reach"]
    assert tight["duel"] == pytest.approx(0.15)
    assert tight["inside_touch"] is True, "0.15 < duck_touch 0.22: `avoid` owns the last metres"


def test_a_knob_that_fires_is_not_read_as_broken():
    """`is_identical` keys a swing arm on its travel, which a duel row does not
    have — so without the duel key every duel arm would read BROKEN, which is
    the check reporting the opposite of the truth."""
    a = run_duel(seed=0, episodes=2)
    b = run_duel(seed=0, episodes=2, knobs="duel=0.3")
    assert is_identical(a, a)
    assert not is_identical(a, b), "duel=0.3 acts on this population; it must not read identical"


def test_the_firing_rate_is_zero_on_the_shipped_brain_and_high_with_the_knob():
    """The number C.4 was missing. The placement is a duel by construction, so
    a rule that fires on the duel must fire on most of these ticks — and the
    shipped brain, which has no rule, must be exactly 0."""
    off = duel_summ(run_duel(seed=0, episodes=3))
    assert off["fire"] == 0.0 and off["act"] == 0.0
    on = duel_summ(run_duel(seed=0, episodes=3, knobs="duel=0.3"))
    assert on["fire"] > 0.5
    assert 0.0 < on["act"] <= on["fire"], "the branch cannot run more often than the rule computes"


# --- the positive control ---------------------------------------------------

def test_with_nobody_to_contest_it_we_touch_first(world):  # noqa: ARG001
    """The control that makes every contested number readable: the identical
    draw for our duck, no opponent, and the episode must resolve to OURS. It
    also fixes `DUEL_S` — at 4 s this read 58%, which would have put a 40%
    floor of "nobody" under every arm and called it a contest."""
    rows = run_duel(seed=0, episodes=16, opponents=0)
    s = duel_summ(rows)
    assert s["theirs_first"] == 0, "there is no opponent: nothing can touch it but us"
    assert s["their_advance"] is None
    assert s["ours_first"] / s["n"] >= 0.85, (
        f"the unopposed duck resolved only {s['ours_first']}/{s['n']} in {DUEL_S:g} s — "
        "the window, not the contest, is deciding these episodes")


# --- backward compatibility, both directions --------------------------------

def test_a_swing_row_file_has_no_duel_summary():
    """`compare_gym.py` is routinely pointed at row files written weeks ago.
    A duel summary of zeros on a swing file would read as "we never touch the
    ball first", which is a finding about nothing."""
    old = [{"swing": True, "foot": "kick_left", "ahead": 0.09, "side": 0.05,
            "travel": 0.42, "whiff": False, "seed": 0},
           {"swing": False, "ep": 1}]
    assert duel_summ(old) == {}
    assert duel_summ([]) == {}


def test_a_duel_row_file_has_no_exit_summary():
    """And the other way: the exit table is per swing, and a duel row has no
    swing. It must be empty rather than a row of Nones."""
    rows = [{"duel": True, "ep": 0, "first": "theirs", "t_first": 1.2, "ticks": 300}]
    assert exit_summ(rows) == {}
    assert duel_summ(rows)["n"] == 1


def test_the_summary_counts_every_episode_in_every_rate():
    """A rate's denominator must come from the same population as its numerator
    (AGENTS.md): an episode nobody touched the ball in is an OUTCOME of the
    duel, so it is in `n` — dropping it would score the rule on the episodes it
    already survived."""
    rows = [{"duel": True, "first": "ours", "t_first": 1.0, "ours_touched": True,
             "theirs_touched": False, "our_advance": 0.4, "falls_us": 0, "falls_them": 0,
             "ticks": 100, "fire_ticks": 50, "act_ticks": 25, "avoid_ticks": 5, "state05": "chase"},
            {"duel": True, "first": "theirs", "t_first": 2.0, "ours_touched": False,
             "theirs_touched": True, "their_advance": 0.2, "falls_us": 1, "falls_them": 0,
             "ticks": 100, "fire_ticks": 50, "act_ticks": 25, "avoid_ticks": 5, "state05": "avoid"},
            {"duel": True, "first": None, "t_first": None, "ours_touched": False,
             "theirs_touched": False, "falls_us": 0, "falls_them": 2,
             "ticks": 100, "fire_ticks": 0, "act_ticks": 0, "avoid_ticks": 0, "state05": "duel"}]
    s = duel_summ(rows)
    assert (s["n"], s["ours_first"], s["theirs_first"], s["none_first"]) == (3, 1, 1, 1)
    assert s["ours_touched"] == pytest.approx(1 / 3)
    assert s["our_advance"] == pytest.approx(0.4)      # scored on the one episode we touched
    assert s["their_advance"] == pytest.approx(0.2)
    assert (s["fell_us"], s["fell_them"]) == (1, 1)
    assert s["fire"] == pytest.approx(100 / 300) and s["act"] == pytest.approx(50 / 300)
    assert s["states"] == {"chase": 1, "avoid": 1, "duel": 1}
    assert s["t_first"] == pytest.approx(1.5)


def _xy(w: World, did: str) -> tuple[float, float]:
    p = w.ducks[did].trunk_pos(w.data)
    return float(p[0]), float(p[1])


def _park(w: World, did: str, x: float, y: float, yaw: float) -> None:
    """A duck at a known place, with the forward kinematics run. `_respawn`
    writes qpos and stops; `trunk_pos` reads `data.xpos`, which is stale until
    `mj_forward` — so without this every attribution test reads the duck's
    PREVIOUS position and passes or fails for the wrong reason."""
    w.ducks[did].spawn = (x, y, yaw)
    w._respawn(w.ducks[did])
    mujoco.mj_forward(w.model, w.data)
    assert _xy(w, did) == pytest.approx((x, y), abs=1e-3)
