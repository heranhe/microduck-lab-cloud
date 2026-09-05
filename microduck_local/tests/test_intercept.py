"""The block: geometry, trigger, who goes, and that the knob reaches the brain.

Every test here is written against the REAL types — `ChaseParams`, a real
`Chase` built the way `brain/team.py` builds one, a real `Team` — because the
playbook's rule 0 was earned by a stub that carried a field the real type
lacked and made a dead code path look alive.
"""

from __future__ import annotations

import math
import os

import pytest

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.intercept import Interceptor, block_point, walk_time
from microduck_local.brain.runtime import Senses
from microduck_local.brain.team import Team, brain_kwargs
from microduck_local.sensors.detector import Detection, DetectionFrame

GOAL = (-1.7, 0.0)          # a duck attacking +x defends this one


def _p(**kw) -> ChaseParams:
    base = {"intercept_eta": 8.0}
    base.update(kw)
    return ChaseParams(**base)


# -- geometry ----------------------------------------------------------------

def test_block_point_sits_between_the_ball_and_the_goal():
    ball, me = (0.0, 0.0), (0.0, 0.6)
    pt = block_point(ball, GOAL, me, ahead=0.25, keep=0.45)
    assert pt is not None
    # on the ball→goal line (which is the −x axis here), goal-side of the ball
    assert pt[0] < ball[0] and abs(pt[1]) < 1e-9
    # and no nearer the goal than `keep`
    assert math.dist(pt, GOAL) >= 0.45 - 1e-9
    assert math.dist(pt, ball) >= 0.25 - 1e-9


def test_block_point_is_the_nearest_reachable_point_on_the_line():
    ball = (0.0, 0.0)
    near = block_point(ball, GOAL, (-0.4, 0.5), 0.25, 0.45)
    far = block_point(ball, GOAL, (-1.0, 0.5), 0.25, 0.45)
    assert near is not None and far is not None
    assert far[0] < near[0]          # a duck further down the line blocks further down it


def test_block_point_refuses_when_there_is_no_room():
    """The first thing this got wrong: clamped into the gap instead, the spot
    lands centimetres goal-side of a ball already at the mouth, and walking to
    it walks the ball in. `ahead + keep` = 0.70 m of clearance or nothing."""
    assert block_point((-1.2, 0.0), GOAL, (0.0, 0.0), 0.25, 0.45) is None
    assert block_point((-1.05, 0.0), GOAL, (0.0, 0.0), 0.25, 0.45) is None
    assert block_point((-0.9, 0.0), GOAL, (0.0, 0.0), 0.25, 0.45) is not None


def test_walk_time_charges_the_turn_but_not_the_free_bearing():
    ahead = walk_time((0.0, 0.0, 0.0), (1.0, 0.0))
    behind = walk_time((0.0, 0.0, math.pi), (1.0, 0.0))
    assert ahead == pytest.approx(1.0 / 0.45, rel=1e-6)      # no turn, no cold start
    assert behind > ahead + 3.0                              # pi rad of turning, plus the cold second


# -- the trigger -------------------------------------------------------------

def _roll(it: Interceptor, p, x0: float, vx: float, t0: float = 0.0, n: int = 40,
          dt: float = 0.05, odom=(0.0, 0.8, 0.0), mates=()) -> tuple[float, float] | None:
    """Feed a ball rolling along −x past the interceptor, one sighting a step."""
    out = None
    for k in range(n):
        t = t0 + k * dt
        b = (x0 + vx * (t - t0), 0.0)
        out = it.update(t, p, b, b, GOAL, odom, "d0", list(mates))
    return out


def test_a_ball_rolling_at_our_goal_fires_the_block():
    it = Interceptor()
    assert _roll(it, _p(), 0.4, -0.3) is not None
    assert it.rate == pytest.approx(0.3, abs=0.02)          # the SCALAR closing rate, not a velocity


def test_a_ball_rolling_away_does_not():
    it = Interceptor()
    assert _roll(it, _p(), -0.4, +0.3) is None


def test_a_ball_barely_moving_does_not():
    it = Interceptor()
    assert _roll(it, _p(), 0.4, -0.05) is None               # under `intercept_vmin`


def test_the_master_knob_off_is_off():
    it = Interceptor()
    assert _roll(it, _p(intercept_eta=0.0), 0.4, -0.3) is None
    assert it.target is None


def test_only_sightings_feed_the_trigger():
    """A coasting track's odometry-frame position walks with the duck holding
    it, so differencing one measures the walk and calls it a rolling ball.
    With `sight` None the history never grows and nothing fires."""
    it, p = Interceptor(), _p()
    for k in range(40):
        b = (0.4 - 0.3 * k * 0.05, 0.0)
        assert it.update(k * 0.05, p, b, None, GOAL, (0.0, 0.8, 0.0), "d0", []) is None


def test_a_stale_history_stops_firing_but_the_hold_carries():
    it, p = Interceptor(), _p(intercept_hold=1.0)
    assert _roll(it, p, 0.4, -0.3) is not None
    ball = (-0.8, 0.0)
    # nothing new seen: committed for `intercept_hold`, then done
    assert it.update(2.5, p, ball, None, GOAL, (0.0, 0.8, 0.0), "d0", []) is not None
    assert it.update(4.0, p, ball, None, GOAL, (0.0, 0.8, 0.0), "d0", []) is None


def test_one_duck_goes():
    """A nearer teammate takes it; a further one does not. Both ducks run this
    arithmetic on the same blackboard, so the answers agree without a
    message."""
    p = ChaseParams(intercept_eta=8.0)
    mine, far = Interceptor(), Interceptor()
    assert _roll(mine, p, 0.4, -0.3, odom=(0.0, 0.8, 0.0),
                 mates=[("d1", (-2.5, 2.0, 0.0))]) is not None
    assert _roll(far, p, 0.4, -0.3, odom=(0.0, 0.8, 0.0),
                 mates=[("d1", (-0.5, 0.05, math.pi))]) is None


# -- the brain ---------------------------------------------------------------

def test_the_knob_reaches_a_roster_brain():
    """Rule 0 / rule 10: assert on the object that RUNS. `brain_kwargs` hands
    a two-a-side roster its own `p`, and for a while that `p` was the bare
    defaults, so every knob a battery set was silently discarded."""
    old = os.environ.get("MICRODUCK_CHASE")
    os.environ["MICRODUCK_CHASE"] = "intercept_eta=6.5,intercept_clear=0.3"
    try:
        assert ChaseParams.from_env().intercept_eta == 6.5
        b = Chase()                                   # the lab / benchmark path: no `p` given
        assert b.p.intercept_eta == 6.5 and b.p.intercept_clear == 0.3
    finally:
        if old is None:
            os.environ.pop("MICRODUCK_CHASE", None)
        else:
            os.environ["MICRODUCK_CHASE"] = old
    assert ChaseParams().intercept_eta == 0.0         # …and it ships OFF


def test_brain_kwargs_carries_the_knob_onto_a_2v2_roster():
    from microduck_local.world import World, make_pitch
    old = os.environ.get("MICRODUCK_CHASE")
    os.environ["MICRODUCK_CHASE"] = "intercept_eta=6.5"
    try:
        sc = make_pitch(per_side=2)
        w = World(sc, seed=0)
        teams: dict = {}
        for d in sc.ducks:
            kw = brain_kwargs(d, w, teams)
            assert Chase(**kw).p.intercept_eta == 6.5
    finally:
        if old is None:
            os.environ.pop("MICRODUCK_CHASE", None)
        else:
            os.environ["MICRODUCK_CHASE"] = old


def _drive(brain: Chase, xs: list[tuple[float, float]], odom, t0=0.0, dt=0.1):
    """Show the brain a ball at the given odometry-frame positions."""
    last = None
    for k, (bx, by) in enumerate(xs):
        t = t0 + k * dt
        bear = math.atan2(by - odom[1], bx - odom[0]) - odom[2]
        rng = math.hypot(bx - odom[0], by - odom[1])
        det = DetectionFrame(t, [Detection("ball", "ball0", bear, -0.3, 0.05, rng, 0.9)])
        last = brain.step(Senses(t=t, det=det, det_age=0.0, speed=0.0, odom=odom))
    return last


def test_chase_enters_block_on_a_ball_rolling_at_its_own_goal():
    """The end-to-end check: a real `Chase` with a real goal, shown a real
    detection stream of a ball rolling toward the goal it defends, leaves the
    play and stands on the line."""
    odom = (0.0, 0.7, -1.2)
    off = Chase(p=ChaseParams(), goal=(1.7, 0.0), bounds=(1.7, 1.425), goal_w=0.7)
    on = Chase(p=_p(), goal=(1.7, 0.0), bounds=(1.7, 1.425), goal_w=0.7)
    roll = [(0.6 - 0.06 * k, 0.0) for k in range(20)]
    _drive(off, roll, odom)
    _drive(on, roll, odom)
    assert on.state == "block"
    assert off.state != "block"                 # …and the shipped default never does
    # …aimed between the ball and the goal it defends (−x), not at the ball
    assert on.blocker.target is not None and on.blocker.target[0] < on.blocker.ball[0]


def test_block_never_fires_without_a_goal_to_defend():
    """Off a pitch (`goal` None) there is no own goal, so there is nothing to
    stand in front of and the knob is inert."""
    b = Chase(p=_p())
    _drive(b, [(0.6 - 0.06 * k, 0.0) for k in range(20)], (0.0, 0.7, -1.2))
    assert b.state != "block"


def test_block_is_in_the_bump_stand_states():
    """Standing on the line is a turn in place with a body possibly beside it,
    which is the shape 12 of 13 traced 3v3 falls had."""
    assert "block" in ChaseParams().bump_stand_states


def test_kickoff_forgets_the_block():
    b = Chase(p=_p(), goal=(1.7, 0.0))
    _drive(b, [(0.6 - 0.06 * k, 0.0) for k in range(20)], (0.0, 0.7, -1.2))
    b.kickoff()
    assert b.blocker.hist == [] and b.blocker.target is None


def test_board_ball_reads_the_freshest_claim():
    """A supporter cannot see the ball (fresh detection on 22% of threat ticks
    over the baseline battery); the board is how it learns one is coming."""
    tm = Team("cream")
    b = Chase(p=_p(), goal=(1.7, 0.0), team=tm, duck_id="d0")
    tm.claim("d1", 1.0, 0.5, (0.3, 0.1), (0.2, 0.1, 0.0))
    tm.claim("d2", 1.4, 0.6, (0.1, 0.1), (0.4, 0.1, 0.0))
    xy, age = b._board_ball(1.5)
    assert xy == (0.1, 0.1) and age == pytest.approx(0.1)
