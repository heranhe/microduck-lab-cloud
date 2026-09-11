"""The duel (`ChaseParams.duel`, roadmap C.4's second half): geometry, the
trigger, the state transitions, and that OFF is the shipped chain to the bit.

Written against the real types — a real `Chase` built the way `brain/team.py`
builds one, real `Detection` frames — because rule 0 here was earned by a stub
that made a dead path look alive. The knob-off test is the one that matters
most: three previous duel rules were measured and two of them turned out to be
gated off, so "the arm ran the shipped path and reported a null" is the local
failure mode (`brain/knob_gates.py`).
"""

from __future__ import annotations

import math
import os

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.knob_gates import warning_for
from microduck_local.brain.runtime import Senses
from microduck_local.sensors.detector import Detection, DetectionFrame

BOUNDS = (1.7, 1.425)
GOAL = (1.7, 0.0)             # attacked; the own goal is (-1.7, 0.0)
STANDOFF = 0.30


def _brain(**knobs) -> Chase:
    return Chase(ChaseParams(**knobs), goal=GOAL, bounds=BOUNDS, goal_w=0.7, duck_id="d0")


def _frame(t: float, odom, ball, ducks=()) -> DetectionFrame:
    """A detection frame that puts `ball` and each of `ducks` where they are,
    from a duck at `odom`."""
    def det(cls: str, name: str, xy) -> Detection:
        bear = math.atan2(xy[1] - odom[1], xy[0] - odom[0]) - odom[2]
        rng = math.hypot(xy[0] - odom[0], xy[1] - odom[1])
        return Detection(cls, name, math.atan2(math.sin(bear), math.cos(bear)),
                         -0.3, 0.05, rng, 0.9)
    out = [det("ball", "ball0", ball)]
    out += [det("duck", f"o{k}", d) for k, d in enumerate(ducks)]
    return DetectionFrame(t, out)


def _drive(b: Chase, odom, ball, ducks=(), n: int = 6, dt: float = 0.1, t0: float = 0.0):
    """Show the brain a still scene for `n` frames and return the last Intent."""
    last = None
    for k in range(n):
        t = t0 + k * dt
        last = b.step(Senses(t=t, det=_frame(t, odom, ball, ducks), det_age=0.0,
                             speed=0.0, odom=odom))
    return last


# -- it ships off ------------------------------------------------------------

def test_the_knob_ships_off_and_its_radius_is_the_probes_number():
    assert ChaseParams().duel == 0.0
    assert ChaseParams().duel_near == 0.35


def test_the_knob_reaches_the_brain_that_runs_and_is_not_gated():
    """Rule 0: assert on the object the battery builds. `contest_margin` was
    measured twice on a brain where it could not act."""
    old = os.environ.get("MICRODUCK_CHASE")
    os.environ["MICRODUCK_CHASE"] = f"duel={STANDOFF}"
    try:
        assert ChaseParams.from_env().duel == STANDOFF
        assert Chase().p.duel == STANDOFF            # the lab / benchmark path: no `p` given
    finally:
        if old is None:
            os.environ.pop("MICRODUCK_CHASE", None)
        else:
            os.environ["MICRODUCK_CHASE"] = old
    assert warning_for(f"duel={STANDOFF}") is None   # nothing else has to be set for it to act


def test_brain_kwargs_carries_it_onto_a_2v2_roster():
    from microduck_local.brain.team import brain_kwargs
    from microduck_local.world import World, make_pitch
    old = os.environ.get("MICRODUCK_CHASE")
    os.environ["MICRODUCK_CHASE"] = f"duel={STANDOFF}"
    try:
        sc = make_pitch(per_side=2)
        w = World(sc, seed=0)
        teams: dict = {}
        for d in sc.ducks:
            assert Chase(**brain_kwargs(d, w, teams)).p.duel == STANDOFF
    finally:
        if old is None:
            os.environ.pop("MICRODUCK_CHASE", None)
        else:
            os.environ["MICRODUCK_CHASE"] = old


# -- the geometry ------------------------------------------------------------

def test_the_spot_is_goal_side_of_the_ball_on_the_line_to_our_own_goal():
    """Ball at the origin, our goal at −x: the block spot is `duel` metres
    along −x of the ball, so the body ends between the two."""
    b = _brain(duel=STANDOFF)
    odom = (0.9, 0.0, math.pi)                       # 0.9 m the goal side, facing the ball
    _drive(b, odom, (0.0, 0.0), ducks=[(-0.15, 0.0)])
    assert b.dueling and b.duel_spot is not None
    assert b.duel_spot[0] == -STANDOFF and abs(b.duel_spot[1]) < 1e-9


def test_the_spot_is_never_laid_in_the_boards():
    """A ball on our own goal line: the spot would be 0.3 m inside the end
    board, and is clamped to `support_margin` inside it, like every other
    post this brain lays."""
    b = _brain(duel=STANDOFF)
    ball = (-BOUNDS[0] + 0.10, 0.0)
    _drive(b, (0.6, 0.0, math.pi), ball, ducks=[(ball[0] - 0.12, 0.0)])
    assert b.dueling and b.duel_spot is not None
    assert b.duel_spot[0] >= -BOUNDS[0] + b.p.support_margin - 1e-9


# -- the trigger -------------------------------------------------------------

def test_it_fires_only_when_the_opponent_is_the_nearer_of_the_two():
    b = _brain(duel=STANDOFF)
    _drive(b, (0.9, 0.0, math.pi), (0.0, 0.0), ducks=[(-0.15, 0.0)])
    assert b.dueling                                  # they are 0.15 m off it, we are 0.90

    b = _brain(duel=STANDOFF)
    _drive(b, (0.20, 0.0, math.pi), (0.0, 0.0), ducks=[(-0.30, 0.0)])
    assert not b.dueling                              # we are nearer: that is the CONTEST's case, not this


def test_an_opponent_merely_on_the_pitch_is_not_a_duel():
    """`duel_near` is a radius about the BALL, not about this duck."""
    b = _brain(duel=STANDOFF)
    _drive(b, (0.9, 0.0, math.pi), (0.0, 0.0), ducks=[(-0.60, 0.35)])
    assert not b.dueling                              # 0.69 m off the ball, outside `duel_near`


def test_the_board_says_who_is_ours_and_a_teammate_is_not_duelled():
    """`_opponents` drops a duck track the board places on a teammate, which
    is the only sense this brain has with the colour vote off. Without it the
    rule would block against its own side."""
    from microduck_local.brain.team import Team
    team = Team("cream")
    odom, ball, mate = (0.9, 0.0, math.pi), (0.0, 0.0), (-0.15, 0.0)
    b = Chase(ChaseParams(duel=STANDOFF), goal=GOAL, bounds=BOUNDS, goal_w=0.7,
              duck_id="d0", team=team)
    for k in range(6):
        t = k * 0.1
        team.claim("d1", t, 0.15, ball, (mate[0], mate[1], 0.0), 0.05)
        b.step(Senses(t=t, det=_frame(t, odom, ball, [mate]), det_age=0.0, speed=0.0, odom=odom))
    assert not b.dueling


def test_off_the_flag_never_sets():
    b = _brain(duel=0.0)
    _drive(b, (0.9, 0.0, math.pi), (0.0, 0.0), ducks=[(-0.15, 0.0)])
    assert not b.dueling and b.duel_spot is None


# -- the state machine -------------------------------------------------------

def test_it_takes_the_state_from_the_chase_and_walks_to_the_spot():
    on = _brain(duel=STANDOFF)
    off = _brain(duel=0.0)
    odom, ball, opp = (0.55, 0.75, -2.2), (0.0, 0.0), (-0.15, 0.0)
    ion = _drive(on, odom, ball, ducks=[opp])
    ioff = _drive(off, odom, ball, ducks=[opp])
    assert on.state == "duel" and off.state != "duel"
    assert on.spot is None                            # a duel drops the kick line-up
    assert ion.twist != ioff.twist                    # ...and steers somewhere else, not at the ball


def test_a_settle_is_never_interrupted():
    """A settle is a swing about to happen; stealing it costs the touch. Same
    exemption the `yield` rule carries."""
    b = _brain(duel=STANDOFF)
    b.state = "settle"
    _drive(b, (0.9, 0.0, math.pi), (0.0, 0.0), ducks=[(-0.15, 0.0)], n=1)
    assert b.dueling                                  # the situation is read...
    assert b.state != "duel"                          # ...and the swing is left alone


def test_a_supporter_keeps_its_post():
    """The role branch is ahead of the duel in the chain, so a duck the board
    has given a post never abandons it to block."""
    b = _brain(duel=STANDOFF)
    b.role = "support"
    odom, ball, opp = (0.9, 0.0, math.pi), (0.0, 0.0), (-0.15, 0.0)
    for k in range(3):
        t = k * 0.1
        b.role = "support"                            # no team board here: re-stamp it each tick
        b.step(Senses(t=t, det=_frame(t, odom, ball, [opp]), det_age=0.0, speed=0.0, odom=odom))
    assert b.state in ("support", "wait") and b.state != "duel"


def test_on_the_spot_it_stands_and_faces_the_ball():
    b = _brain(duel=STANDOFF)
    odom = (-STANDOFF, 0.0, 0.0)                      # already on the spot, facing the ball
    it = _drive(b, odom, (0.0, 0.0), ducks=[(-0.15, 0.25)])
    assert b.state == "duel"
    assert it.twist[0] == 0.0 and it.twist[2] == 0.0  # standing, square on the ball


# -- off is the shipped chain, to the bit ------------------------------------

def test_knob_off_is_byte_identical_to_the_shipped_brain():
    """Row for row over a rollout that spends its time in exactly the states
    the duel would have taken (a ball with an opponent all over it). `duel=0`
    must not change one command, or every A/B against it is measuring two
    changes."""
    def rollout(**knobs) -> list[tuple]:
        b = _brain(**knobs)
        rows = []
        for k in range(120):
            t = k * 0.05
            a = 0.6 * math.sin(0.25 * t)
            ball = (0.3 * math.cos(0.4 * t), 0.3 * math.sin(0.4 * t))
            odom = (0.9 + 0.1 * math.cos(t), 0.2 * math.sin(t), math.pi + a)
            opp = (ball[0] - 0.12, ball[1] + 0.05)
            it = b.step(Senses(t=t, det=_frame(t, odom, ball, [opp]), det_age=0.0,
                               speed=0.2, odom=odom))
            rows.append((b.state, it.twist, it.head, it.skill))
        return rows

    shipped = rollout()
    assert rollout(duel=0.0) == shipped
    assert rollout(duel=STANDOFF) != shipped          # ...and the knob is not inert
