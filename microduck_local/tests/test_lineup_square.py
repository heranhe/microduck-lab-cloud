"""The square-up at a reached spot (`lineup_square`, roadmap 12ao): inside it
of a kick spot and off heading, the line-up turns in place to the heading
instead of servoing at the spot; squared but short, it walks the last
centimetres straight along the heading. Off, the servo as before."""

from __future__ import annotations

import math

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.runtime import Senses

BOUNDS = (1.5, 1.25)


def _tick(spot, **knobs):
    b = Chase(ChaseParams(gaze_still=True, **knobs), goal=(BOUNDS[0], 0.0), bounds=BOUNDS, goal_w=0.7, duck_id="d0")
    b.state, b.lined, b.t_state = "lineup", True, 5.0
    b.spot = spot
    out = b.step(Senses(t=5.0, det=None, det_age=None, odom=(0.0, 0.0, 0.0), speed=0.0))
    return b, out.twist


def test_the_default_is_off():
    assert ChaseParams().lineup_square == 0.0


def test_near_and_off_heading_it_turns_to_the_heading_not_the_spot():
    """The spot 5 cm ahead-left; the kick heading 60 deg to the RIGHT. The
    servo turns left toward the spot; the rule turns right to the heading."""
    spot = (0.06, 0.04, "kick_right", -math.radians(60), "kick")     # 7 cm off: outside lineup_tol 0.05, inside the window
    _, off = _tick(spot)
    _, on = _tick(spot, lineup_square=0.08)
    assert on[2] < 0.0 and on[0] <= 0.2 + 1e-9                         # the kicked turn (TURN_KICK creep), RIGHT to the heading
    assert off[2] > 0.0                                                # the servo: LEFT, toward the spot


def test_near_and_squared_it_closes_the_spot_holonomically():
    ahead = (0.07, 0.0, "kick_right", 0.0, "kick")                      # 7 cm straight ahead, squared
    _, on = _tick(ahead, lineup_square=0.08)
    assert on[0] > 0.2 and on[1] == 0.0 and abs(on[2]) < 0.05            # forward at the close's speed, no crab, no turn
    beside = (0.0, 0.07, "kick_right", 0.0, "kick")                     # 7 cm to the LEFT, squared
    _, on = _tick(beside, lineup_square=0.08)
    assert on[1] > 0.2 and abs(on[0]) < 1e-9 and abs(on[2]) < 0.05       # a crab left at the close's speed, no walk, no turn
    _, off = _tick(beside)
    assert off[1] == 0.0                                               # the shipped brain never crabs


def test_outside_the_window_or_for_a_push_the_servo_is_untouched():
    far = (0.30, 0.10, "kick_right", -math.radians(60), "kick")
    _, off = _tick(far)
    _, on = _tick(far, lineup_square=0.08)
    assert on == off
    push = (0.06, 0.04, None, -math.radians(60), "push")
    _, off = _tick(push)
    _, on = _tick(push, lineup_square=0.08)
    assert on == off
