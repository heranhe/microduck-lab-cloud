"""The gym could not ask the boards question, and now it can.

`board_margin` — the rule that lines a kick up ALONG a board instead of into
it — measured null on the pitch, and that null is the one case the triage rule
says is worth re-running: the population is ample (12m measures a third of a
run within 0.40 m of a board) while the instrument was not (MDE 28% of
baseline on kicks at 24 seeds, against an observed difference of 0.2).

The gym is the right instrument for it and could not pose it. Measured on the
shipped open-play draw, 200k samples: the ball lands within 0.40 m of a board
on **6.2%** of episodes and within 0.15 m on **1.0%** — the duck spawns at
x = -0.9 and draws 0.45-1.4 m ahead, which never reaches the end boards and
reaches the side boards only in the tail. An arm about the boards, run in that
gym, would have measured the baseline with extra steps: the same
null-about-nothing as `contest_margin`, one level up from the knob.

These lock the placement mode that fixes it, and — as much as anything — lock
the OPEN-PLAY draw against drifting, since the two modes are only comparable
while the duck's walk-in is the same in both.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kick_gym import _board_rect, _place, _place_at_boards, gym_scenario  # noqa: E402

from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer  # noqa: E402
from microduck_local.world import World  # noqa: E402

WALK_IN = (0.45, 1.40)          # the range BOTH modes draw the duck at


@pytest.fixture(scope="module")
def world():
    sc = gym_scenario()
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    return World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=0)


def ball_xy(w, q):
    return float(w.data.qpos[q]), float(w.data.qpos[q + 1])


def to_board(w, bx, by):
    hx, hy = _board_rect(w)
    return min(hx - abs(bx), hy - abs(by))


def test_the_board_rect_is_the_walls_not_the_floor(world):
    """The floor is 0.25 m larger than the boards on every side. Reading the
    floor and calling it the boards is a quarter-metre error in the one number
    this whole mode is about."""
    hx, hy = _board_rect(world)
    assert (hx, hy) == pytest.approx((1.5, 1.25))
    fx, fy = world.scenario.floor
    assert hx < fx / 2 and hy < fy / 2


@pytest.mark.parametrize("margin", [0.10, 0.25, 0.40])
def test_every_ball_lands_within_the_margin_of_a_board(world, margin):
    rng = np.random.default_rng(3)
    for _ in range(120):
        q, _v = _place_at_boards(world, rng, margin)
        d = to_board(world, *ball_xy(world, q))
        assert 0.0 < d <= margin + 1e-6, d


def test_the_ball_is_never_placed_inside_a_board(world):
    """A ball at or past the board line is not a placement, it is a bug that
    would settle as an immediate ball-out."""
    rng = np.random.default_rng(4)
    r = world.scenario.balls[0].radius
    for _ in range(120):
        q, _v = _place_at_boards(world, rng, 0.25)
        assert to_board(world, *ball_xy(world, q)) >= r, "ball is in the board"


def test_the_duck_always_spawns_on_the_pitch(world):
    """A spawn behind a board is not a spawn. The bearing is retried against
    the board rectangle, with a straight-in fallback."""
    rng = np.random.default_rng(5)
    hx, hy = _board_rect(world)
    for _ in range(200):
        _q, _v = _place_at_boards(world, rng, 0.25)
        dx, dy, _yaw = world.ducks["d0"].spawn
        assert abs(dx) < hx and abs(dy) < hy, (dx, dy)


def test_the_walk_in_matches_open_play_so_the_modes_are_comparable(world):
    """The two modes must differ in WHERE THE BALL IS and not in how far the
    duck walks — otherwise a difference between them is the approach length,
    which is already known to drive the whiff (the plan's age at the swing)."""
    rng = np.random.default_rng(6)
    for _ in range(200):
        q, _v = _place_at_boards(world, rng, 0.25)
        bx, by = ball_xy(world, q)
        dx, dy, _yaw = world.ducks["d0"].spawn
        assert WALK_IN[0] - 1e-6 <= math.dist((dx, dy), (bx, by)) <= WALK_IN[1] + 1e-6


def test_the_duck_starts_facing_the_ball(world):
    rng = np.random.default_rng(7)
    for _ in range(120):
        q, _v = _place_at_boards(world, rng, 0.25)
        bx, by = ball_xy(world, q)
        dx, dy, yaw = world.ducks["d0"].spawn
        off = abs(math.atan2(by - dy, bx - dx) - yaw)
        assert min(off, 2 * math.pi - off) <= 0.26, "the jitter is +-0.25 rad"


def test_all_four_boards_are_sampled(world):
    """A mode that only ever used one board would answer a narrower question
    than the one asked, and the end boards are where the goal is."""
    rng = np.random.default_rng(8)
    hx, hy = _board_rect(world)
    sides = set()
    for _ in range(400):
        q, _v = _place_at_boards(world, rng, 0.25)
        bx, by = ball_xy(world, q)
        sides.add(("x" if hx - abs(bx) < hy - abs(by) else "y", bx > 0 if hx - abs(bx) < hy - abs(by) else by > 0))
    assert len(sides) == 4, sides


def test_the_open_play_draw_still_almost_never_reaches_a_board(world):
    """The measurement that motivates the mode, locked so it cannot drift
    silently: if `_place` ever starts producing a boards population of its own,
    this mode's premise changes and the comparison has to be re-thought."""
    rng = np.random.default_rng(9)
    near = 0
    n = 600
    for _ in range(n):
        q, _v = _place(world, rng, 0.8)
        if to_board(world, *ball_xy(world, q)) <= 0.40:
            near += 1
    assert near / n < 0.15, f"open play now reaches the boards {100 * near / n:.0f}% of the time"


def test_a_tiny_margin_still_produces_a_legal_placement(world):
    """Degenerate input must not put the ball in the board or hang the retry."""
    rng = np.random.default_rng(10)
    r = world.scenario.balls[0].radius
    for _ in range(40):
        q, _v = _place_at_boards(world, rng, 0.0)
        assert to_board(world, *ball_xy(world, q)) >= r
