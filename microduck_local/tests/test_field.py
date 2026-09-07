"""Locks for the supporter potential field (brain/field.py, roadmap Track
4 §6 D.2): a supporter stands ahead of the ball BESIDE the carrier's lane
on its own side, crosses the lane when a teammate or an opponent has the
spot, is held by hysteresis, keeps the boards' margin, its zone and the
attacker's room, and — through `ChaseParams.support_field` — replaces a
striker's, a midfielder's or a plain supporter's post while a defender's
and a keeper's posts are untouched. Ships off until measured."""

import math

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.field import Field, FieldParams, lane_unit, potential_grid
from microduck_local.brain.kickselect import Pitch, potential
from microduck_local.brain.runtime import Senses
from microduck_local.brain.team import Team
from microduck_local.brain.tracker import Track

BOUNDS = (1.75, 1.25)
PITCH = Pitch(BOUNDS[0], BOUNDS[1], 0.7, 1.0)
U = (1.0, 0.0)                                              # the lane: from the ball toward the +x mouth


def _field() -> Field:
    return Field(BOUNDS, 0.35)


def test_the_grid_potential_is_the_selectors_potential():
    f = _field()
    pg = potential_grid(f.x, f.y, PITCH)
    for i in (0, len(f.x) // 3, len(f.x) - 1):
        assert abs(pg[i] - potential(float(f.x[i]), float(f.y[i]), PITCH)) < 1e-9


def test_a_supporter_stands_ahead_of_the_ball_beside_the_lane_on_its_own_side():
    f = _field()
    for my_y, side in ((0.8, 1), (-0.8, -1)):
        x, y = f.spot((0.0, 0.0), U, 0.8, [], [], PITCH, me=(0.5, my_y))
        assert 0.6 <= x <= 1.1, (x, y)                       # ahead by about the role's number
        assert side * y >= 0.4, (x, y)                       # beside the lane, not in it, on my side
    # Behind the ball (a plain supporter's negative `ahead`): behind, and
    # the lane is no obstacle back there.
    x, y = f.spot((0.0, 0.0), U, -0.7, [], [], PITCH, me=(-0.5, 0.1))
    assert -0.9 <= x <= -0.5 and abs(y) <= 0.7


def test_a_teammate_or_an_opponent_on_the_spot_moves_it_across_the_lane():
    f = _field()
    ball, me = (0.0, 0.0), (0.5, 0.8)
    x0, y0 = f.spot(ball, U, 0.8, [], [], PITCH, me=me)
    assert y0 > 0
    x1, y1 = f.spot(ball, U, 0.8, [(x0, y0)], [], PITCH, me=me)          # a mate is standing there
    assert y1 < -0.3 and math.hypot(x1 - x0, y1 - y0) > 0.8
    # An opponent BETWEEN the ball and the spot closes the lane just the same.
    ox, oy = 0.6 * x0, 0.6 * y0
    x2, y2 = f.spot(ball, U, 0.8, [], [(ox, oy)], PITCH, me=me)
    assert y2 < -0.3, (x2, y2)


def test_hysteresis_holds_the_spot_against_a_small_move_and_lets_a_big_one_through():
    f = _field()
    me = (0.5, 0.8)
    s0 = f.spot((0.0, 0.0), U, 0.8, [], [], PITCH, me=me)
    s1 = f.spot((0.03, 0.02), U, 0.8, [], [], PITCH, me=me, prev=s0)
    assert s1 == s0
    s2 = f.spot((-0.6, 0.0), U, 0.8, [], [], PITCH, me=me, prev=s0)
    assert s2 != s0 and s2[0] < s0[0]


def test_the_boards_margin_the_zone_and_the_attackers_room_are_hard():
    f = _field()
    # Every candidate spot is inside the margin.
    assert abs(f.x).max() <= BOUNDS[0] - 0.35 + 1e-9 and abs(f.y).max() <= BOUNDS[1] - 0.35 + 1e-9
    # A striker's zone (the attacking third) with the ball in our own third:
    # the spot is still in the third, not 0.8 m ahead of the ball.
    x, y = f.spot((-1.0, 0.0), U, 0.8, [], [], PITCH, zone=(1 / 3, 1.0), me=(0.0, 0.5))
    assert x >= BOUNDS[0] / 3 - 1e-9
    # A midfielder level with a ball near the mouth keeps the attacker's room.
    x, y = f.spot((1.3, 0.0), U, 0.0, [], [], PITCH, keep_out=0.45, me=(1.0, 0.2))
    assert math.hypot(x - 1.3, y) >= 0.45
    # A zone nothing on the grid satisfies: None, and the caller keeps its post.
    assert f.spot((0.0, 0.0), U, 0.8, [], [], PITCH, zone=(2.0, 3.0)) is None


def test_lane_unit_points_from_the_ball_to_the_goal_and_falls_back_on_the_goal_line():
    ux, uy = lane_unit((0.0, 0.5), (1.75, 0.0), 0.0)
    assert abs(math.hypot(ux, uy) - 1.0) < 1e-9 and ux > 0 and uy < 0
    assert lane_unit((1.75, 0.0), (1.75, 0.0), math.pi / 2) == (math.cos(math.pi / 2), 1.0)


def _roster(p: ChaseParams):
    tm = Team("left")
    tm.jobs = {"d0": "defender", "d1": "midfielder", "d2": "striker"}
    ducks = {k: Chase(p, goal=(1.75, 0.0), team=tm, duck_id=k, bounds=BOUNDS, goal_w=0.7, role=tm.jobs[k])
             for k in tm.jobs}
    for b in ducks.values():
        b._senses = Senses(t=10.0)
    return tm, ducks


def test_the_field_replaces_the_striker_and_midfielder_posts_and_leaves_the_back_line_alone():
    assert ChaseParams().support_field is False                    # ships off until measured
    ball = (0.2, -0.3)
    tm_on, on = _roster(ChaseParams(support_field=True))
    tm_off, off = _roster(ChaseParams())
    for tm in (tm_on, tm_off):
        tm.claim("d1", 10.0, 0.2, ball, (0.1, -0.3, 0.0))          # the mid has the ball
        tm.claim("d0", 10.0, 1.5, None, (-1.2, 0.0, 0.0))
        tm.claim("d2", 10.0, 1.0, None, (0.9, 0.5, 0.0))
        assert tm.attacker(10.0) == "d1"
    # The defender's between-post is byte for byte the same with the field on.
    assert on["d0"]._hold_target(ball, (-1.2, 0.0, 0.0)) == off["d0"]._hold_target(ball, (-1.2, 0.0, 0.0))
    # The striker's spot is the field's: ahead of the ball along the lane
    # to the goal, beside it on the striker's own side, inside its third.
    sx, sy = on["d2"]._hold_target(ball, (0.9, 0.5, 0.0))
    px, py = off["d2"]._hold_target(ball, (0.9, 0.5, 0.0))
    assert (sx, sy) != (px, py)
    assert sx - ball[0] >= 0.5 and sy > 0.1 and sx >= BOUNDS[0] / 3 - 1e-9, (sx, sy)
    # The carrier is not a repulsor: the striker stays within a push's reach and a bit.
    assert math.hypot(sx - ball[0], sy - ball[1]) <= 1.2
    # Hysteresis lives on the brain and a kickoff forgets it.
    assert on["d2"]._field_prev == (sx, sy)
    on["d2"].kickoff()
    assert on["d2"]._field_prev is None


def test_opponents_come_from_duck_tracks_that_the_board_does_not_own():
    tm, on = _roster(ChaseParams(support_field=True))
    ball = (0.0, 0.0)
    tm.claim("d1", 10.0, 0.2, ball, (-0.1, 0.0, 0.0))
    tm.claim("d0", 10.0, 1.5, None, (-1.2, 0.0, 0.0))
    tm.claim("d2", 10.0, 1.0, None, (0.5, 0.8, 0.0))
    s = on["d2"]
    x0, y0 = s._hold_target(ball, (0.5, 0.8, 0.0))
    assert y0 > 0
    # A duck track on the board as a teammate is not an opponent: nothing moves.
    s._field_prev = None
    s.tracker.tracks.append(Track(id=1, cls="duck", bearing=0.0, elevation=0.0, width=0.3, range=1.0, conf=0.9,
                                  born_t=9.0, last_t=9.9, xy=(-1.15, 0.05)))
    assert s._hold_target(ball, (0.5, 0.8, 0.0)) == (x0, y0)
    # A stranger between the ball and the spot: the spot crosses the lane.
    s._field_prev = None
    s.tracker.tracks.append(Track(id=2, cls="duck", bearing=0.0, elevation=0.0, width=0.3, range=0.5, conf=0.9,
                                  born_t=9.0, last_t=9.9, xy=(0.6 * x0, 0.6 * y0)))
    x1, y1 = s._hold_target(ball, (0.5, 0.8, 0.0))
    assert y1 < 0, (x1, y1)


def test_field_params_are_read_off_the_environment():
    p = ChaseParams.from_env("support_field=1,field_wide=0.4,field_lane=0.25,field_mid_ahead=-0.5")
    assert p.support_field is True and p.field_wide == 0.4 and p.field_lane == 0.25
    assert p.field_mid_ahead == -0.5 and ChaseParams().field_mid_ahead == 0.0
    assert FieldParams().wide == 0.5 and FieldParams().lane_w == 0.2


def test_the_midfielders_field_spot_follows_field_mid_ahead():
    ball = (0.2, 0.0)
    spots = {}
    for a in (0.0, -0.5):
        tm, on = _roster(ChaseParams(support_field=True, field_mid_ahead=a))
        tm.claim("d2", 10.0, 0.2, ball, (0.1, 0.0, 0.0))                 # the striker has the ball
        tm.claim("d0", 10.0, 1.5, None, (-1.2, 0.0, 0.0))
        tm.claim("d1", 10.0, 1.0, None, (-0.2, 0.5, 0.0))
        spots[a] = on["d1"]._hold_target(ball, (-0.2, 0.5, 0.0))
    assert spots[-0.5][0] < spots[0.0][0] <= ball[0] + 0.15                # level with the ball, then behind it
    assert all(abs(s[1]) >= 0.3 for s in spots.values())                    # beside the lane either way
