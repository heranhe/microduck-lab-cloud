"""Reachability as a constraint in the kick selector (`spot_reach`, roadmap
12v / 12aa, 2026-09-10): a candidate line whose stand spot is closer to a
board than the walking body's extent is not offered; a fan with no reachable
candidate (a corner) is left whole, so the plan is never worse than the
old one. At 0 the fan is untouched and the counters stay at zero; it ships
at the body extent (12al)."""

from __future__ import annotations

import math

from microduck_local.brain.controllers import Chase, ChaseParams

BOUNDS = (1.5, 1.25)
BODY = 0.129


def _brain(**knobs) -> Chase:
    return Chase(ChaseParams(**knobs), goal=(1.5, 0.0), bounds=BOUNDS, goal_w=0.7, duck_id="d0")


def test_the_default_is_the_body_extent():
    assert ChaseParams().spot_reach == BODY                       # ships on (roadmap 12al)


def test_the_spot_geometry_is_the_plans_own():
    """A ball 0.15 m off the +y board, the line along the board (+x): the
    right foot stands to the ball's +y side (0.06 in), 0.09 m from the wall,
    the left foot on the open side, 0.21 m from it."""
    b = _brain()
    ball = (0.0, BOUNDS[1] - 0.15)
    assert not b._spot_clear(ball, 0.0, "kick_right", BODY)
    assert b._spot_clear(ball, 0.0, "kick_left", BODY)
    assert b._spot_clear(ball, 0.0, "kick_right", 0.05)          # a smaller body would fit
    assert not b._spot_clear(ball, -math.pi / 2, "push", BODY)   # a push AWAY from the board stands in the board
    assert b._spot_clear(ball, 0.0, "push", BODY)                # a push along it stands 0.16 m behind, 0.15 m in
    assert Chase(ChaseParams())._spot_clear(ball, 0.0, "kick_right", BODY)   # off a pitch: always


def test_on_the_selector_offers_only_spots_the_body_fits():
    ball = (0.0, BOUNDS[1] - 0.15)
    odom = (-0.6, ball[1], 0.0)                                   # 0.6 m behind it, facing +x along the board
    los = math.atan2(ball[1] - odom[1], ball[0] - odom[0])
    b = _brain(spot_reach=BODY)
    b._senses = None
    got = b._select_kick_line(odom, ball, los, los)
    assert got is not None
    u, foot = got
    assert b._spot_clear(ball, u, foot, BODY), (u, foot)
    assert b.unreach_dropped > 0 and b.unreach_corners == 0


def test_off_the_fan_is_untouched():
    ball = (0.0, BOUNDS[1] - 0.15)
    odom = (-0.6, ball[1], 0.0)
    los = math.atan2(ball[1] - odom[1], ball[0] - odom[0])
    b = _brain(spot_reach=0.0)
    b._senses = None
    b._select_kick_line(odom, ball, los, los)
    assert b.unreach_dropped == 0 and b.unreach_corners == 0


def test_a_corner_leaves_the_fan_whole_and_the_plan_as_it_was():
    """5 cm from two boards: no line has a reachable spot (the body extent
    exceeds the ball's offset plus kick_side), so the constraint drops
    nothing and the verdict is the shipped one, seed for seed."""
    ball = (BOUNDS[0] - 0.05, BOUNDS[1] - 0.05)
    odom = (ball[0] - 0.5, ball[1] - 0.5, math.pi / 4)
    los = math.atan2(ball[1] - odom[1], ball[0] - odom[0])
    on, off = _brain(spot_reach=BODY), _brain(spot_reach=0.0)
    on._senses = off._senses = None
    a, c = on._select_kick_line(odom, ball, los, los), off._select_kick_line(odom, ball, los, los)
    assert on.unreach_corners == 1 and on.unreach_dropped == 0
    assert a == c
