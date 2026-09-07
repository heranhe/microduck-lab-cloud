"""Locks for outcome-simulated kick selection (brain/kickselect.py, roadmap
Track 4 §6 A.3): the roll-out labels a sample by where it stops, the
selector refuses lines with an own-goal sample and prefers lines that score,
the potential field slopes toward the goal we attack, and with the knob off
`Chase._plan` is the planner as it was before the selector shipped."""

import math

import numpy as np

from microduck_local.brain.controllers import Chase, ChaseParams, _wrap
from microduck_local.brain.kickselect import (
    GOALOPP,
    GOALOWN,
    INFIELD,
    KickModel,
    Pitch,
    evaluate,
    potential,
    roll_out,
    select,
)
from microduck_local.brain.runtime import Senses
from microduck_local.sensors.detector import Detection, DetectionFrame

PITCH = Pitch(half_x=1.5, half_y=1.25, goal_w=0.7, attack_sign=1.0)
EXACT = KickModel(speed=1.4, speed_sd=0.0, dir_sd=0.0, decel=0.3)        # no scatter: the geometry alone


def test_roll_out_labels_a_sample_by_where_it_stops():
    rng = np.random.default_rng(0)
    # Straight at the attacked mouth from 0.5 m out: 1.4 m/s over 0.3 m/s^2 rolls 3.3 m, so it crosses.
    (label, end), = roll_out((1.0, 0.0), 0.0, EXACT, PITCH, rng, 1)
    assert label == GOALOPP and abs(end[0] - 1.5) < 1e-9
    # The same, aimed past the post: it reaches the board and stops there, in the field of play.
    (label, end), = roll_out((1.0, 0.0), math.atan2(0.5, 0.5), EXACT, PITCH, rng, 1)
    assert label == INFIELD and abs(end[0] - 1.5) < 1e-9 and end[1] > 0.35
    # Back toward our own mouth from in front of it.
    (label, end), = roll_out((-1.0, 0.0), math.pi, EXACT, PITCH, rng, 1)
    assert label == GOALOWN and abs(end[0] + 1.5) < 1e-9
    # A gentle kick that stops before any board.
    soft = KickModel(speed=0.3, speed_sd=0.0, dir_sd=0.0, decel=0.3)     # rolls 0.15 m
    (label, end), = roll_out((0.0, 0.0), 0.0, soft, PITCH, rng, 1)
    assert label == INFIELD and abs(end[0] - 0.15) < 1e-9


def test_the_potential_slopes_toward_the_goal_we_attack_and_away_from_our_own():
    assert potential(1.2, 0.0, PITCH) > potential(0.0, 0.0, PITCH) > potential(-1.2, 0.0, PITCH)
    assert potential(-1.2, 0.0, PITCH) < potential(-1.2, 1.0, PITCH)    # in front of our own mouth is the worst place
    mirrored = Pitch(1.5, 1.25, 0.7, attack_sign=-1.0)
    assert potential(-1.2, 0.0, mirrored) > potential(1.2, 0.0, mirrored)


def test_select_refuses_an_own_goal_line_and_prefers_a_line_that_scores():
    rng = np.random.default_rng(1)
    model = KickModel(speed=1.4, speed_sd=0.2, dir_sd=0.15, decel=0.3)
    # In front of our own mouth, the line of sight pointing into it: the
    # own-goal line is refused, and the chosen line sends the ball up the pitch.
    ball = (-1.0, 0.0)
    cands = [(_wrap(math.pi - EXACT.exit_left), "kick_left"),          # outcome straight into our net
             (_wrap(math.pi / 2 - EXACT.exit_left), "kick_left"),      # along the mouth: safe, sideways
             (_wrap(0.0 - EXACT.exit_left), "kick_left")]              # up the pitch
    v = select(ball, cands, model, PITCH, rng, n=40)
    assert v is not None and v.p_own == 0.0
    assert abs(_wrap(v.heading - (0.0 - EXACT.exit_left))) < 1e-9     # the up-pitch line wins on the potential
    # In front of THEIR mouth: the straight line scores most and is chosen.
    ball = (1.0, 0.0)
    cands = [(_wrap(0.0 - EXACT.exit_left), "kick_left"),
             (_wrap(0.9 - EXACT.exit_left), "kick_left"),
             (_wrap(-0.9 - EXACT.exit_left), "kick_left")]
    v = select(ball, cands, model, PITCH, rng, n=40)
    assert v is not None and abs(_wrap(v.heading - (0.0 - EXACT.exit_left))) < 1e-9 and v.p_goal > 0.6
    # Every candidate risky: None, so the caller keeps its own line.
    ball = (-1.2, 0.0)
    v = select(ball, [(_wrap(math.pi - EXACT.exit_left), "kick_left")], model, PITCH, rng, n=40)
    assert v is None


def test_evaluate_applies_the_foot_exit_angle_to_the_intended_line():
    rng = np.random.default_rng(2)
    # A line aimed 23.6 deg RIGHT of straight with the LEFT foot leaves straight: the exit angle cancels.
    v = evaluate((1.0, 0.0), -EXACT.exit_left, "kick_left", EXACT, PITCH, rng, 5)
    assert v.p_goal == 1.0
    # The same intended line with the RIGHT foot leaves 52 deg right: past the post, in the field.
    v = evaluate((1.0, 0.0), -EXACT.exit_left, "kick_right", EXACT, PITCH, rng, 5)
    assert v.p_goal == 0.0 and v.p_own == 0.0


def _senses(t, ball, odom):
    det = DetectionFrame(t, [Detection("ball", "ball0", ball[0], -0.3, 0.12, ball[1], 0.9)])
    return Senses(t=t, det=det, det_age=0.0, speed=0.3, odom=odom)


def test_with_the_knob_off_the_plan_is_unchanged_and_on_it_stays_inside_the_aim_window():
    """`_plan` with kick_select off is the shipped planner to the bit; on, the
    chosen line is inside `aim_max` of the line of sight (the same walk-round
    the clamp allows), and a duck facing its own goal is sent up the pitch."""
    def plan(p, odom, bearing, rng_range=1.0):
        b = Chase(p, goal=(1.5, 0.0), duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7)
        b.step(_senses(0.0, (bearing, rng_range), odom))
        b.step(_senses(0.1, (bearing, rng_range), odom))
        ball = b.tracker.best("ball", 0.1, min_hits=1)
        return b, b._plan(odom, ball)
    assert ChaseParams().kick_select is True                        # ships on since 2026-09-07 (confirmed on fresh seeds)
    off = ChaseParams(kick_select=False)                             # the pre-2026-09-07 planner
    b0, plan_off = plan(off, (0.0, 0.0, 0.0), 0.2)
    b1, plan_off2 = plan(ChaseParams(kick_select=False), (0.0, 0.0, 0.0), 0.2)
    assert plan_off == plan_off2 and b0.last_select is None and b1.last_select is None
    on = ChaseParams()
    # Facing our OWN mouth from 0.4 m out (the ball 0.5 m ahead of a duck at
    # x = -0.6): straight ahead puts two thirds of kicks in our net, the
    # edge lines of the aim window 7-9% (measured, see the knob). The
    # selector picks inside the window and under the tolerance, never the
    # straight line.
    odom = (-0.6, 0.0, math.pi)
    b, (sx, sy, foot, h, mode) = plan(on, odom, 0.0, 0.5)
    los = math.pi
    v = b.last_select
    assert v is not None and v.p_own <= on.kick_select_t_own
    assert abs(_wrap(v.heading - los)) <= on.aim_max + 1e-9
    assert abs(_wrap(v.heading - los)) > 0.9                            # the edge of the window, not straight at our net
    assert mode == "kick" and foot == v.foot                            # the planner laid the spot for the CHOSEN foot
    # The foot is the lever: the outcome line (exit angle applied) points
    # well away from our mouth, which the planner's own foot on that line
    # could not manage - its exit angle bends the kick back toward the ball's
    # heading (measured: 50-83% own goals with the planner's foot, 7% with
    # the other).
    exit_a = on.kick_exit_left if v.foot == "kick_left" else on.kick_exit_right
    assert abs(_wrap(v.heading + exit_a - los)) > 1.2
    # Mellmann's zero tolerance would refuse every line there and keep the clamp's.
    strict = ChaseParams(kick_select=True, kick_select_t_own=0.0)
    b, _ = plan(strict, odom, 0.0, 0.5)
    assert b.last_select is None
