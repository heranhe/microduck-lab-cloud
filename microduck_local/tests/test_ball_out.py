"""Locks for roadmap Track 4 item 11b: the ball-out rule (a World knob, off
by default, on for the lab's pitches) and the kick line along the boards."""

import math

import mujoco
import numpy as np

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.tracker import Track
from microduck_local.world import World, make_pitch


def _park_ball_at_the_boards(w: World) -> tuple[int, float, float]:
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    hx, hy = w.scenario.floor[0] / 2 - 0.25, w.scenario.floor[1] / 2 - 0.25
    w.data.qpos[q:q + 2] = [0.3, hy - 0.05]                          # against the side board, out of the mouths
    v = int(w.model.jnt_dofadr[j])
    w.data.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(w.model, w.data)
    return q, hx, hy


def test_ball_out_is_off_by_default_and_places_the_ball_in_when_on():
    sc = make_pitch()
    off = World(sc, seed=1)
    assert off.ball_out_s == 0.0 and off.soccer_score()["ballOuts"] == 0
    q, hx, hy = _park_ball_at_the_boards(off)
    for _ in range(int(2.0 / 0.02)):
        off.step()
    assert off.ball_outs == 0 and abs(float(off.data.qpos[q + 1]) - (hy - 0.05)) < 0.03   # still at the boards
    on = World(sc, seed=1, ball_out_s=1.0)
    q, hx, hy = _park_ball_at_the_boards(on)
    for _ in range(int(0.8 / 0.02)):
        on.step()
    assert on.ball_outs == 0                                          # not yet: it has to REST there for ball_out_s
    for _ in range(int(0.5 / 0.02)):
        on.step()
    assert on.ball_outs == 1 and on.soccer_score()["ballOuts"] == 1
    x, y = float(on.data.qpos[q]), float(on.data.qpos[q + 1])
    assert hy - abs(y) >= on.ball_out_in - 0.02 and abs(x - 0.3) < 0.05   # placed in from the wall, same x
    on.reset()
    assert on.ball_outs == 0


def test_the_lab_pitches_play_under_the_rule_and_rooms_do_not():
    from microduck_local.world_server import PITCH_BALL_OUT_S, WorldState
    st = WorldState(None)
    st.preload("pitch-2v2")
    assert st.world.ball_out_s == PITCH_BALL_OUT_S > 0
    room = WorldState(None)
    room.preload("living-room")
    assert room.world.ball_out_s == 0.0


def _seen_ball(b: Chase, odom, bx, by, t=1.0):
    """A track for a ball at (bx, by), as the brain would hold it."""
    rng = math.hypot(bx - odom[0], by - odom[1])
    bearing = math.atan2(by - odom[1], bx - odom[0]) - odom[2]
    tr = Track.__new__(Track)
    tr.bearing, tr.range, tr.vel_hits = bearing, rng, 0
    return tr


def test_a_ball_at_the_boards_gets_a_spot_along_them_on_the_open_side():
    p = ChaseParams(board_margin=0.08)                                 # ships off; the lab's pitches use the ball-out rule
    hx, hy = 1.5, 1.25
    b = Chase(p, goal=(hx, 0.0), bounds=(hx, hy), goal_w=0.7)
    odom = (-0.5, hy - 0.4, 0.0)
    bx, by = 0.0, hy - 0.04                                            # against the +y side board
    x, y, foot, h, mode = b._plan(odom, _seen_ball(b, odom, bx, by))
    assert mode == "kick"
    assert hy - abs(y) >= p.board_margin                               # the spot is off the boards
    assert abs(h) < 0.3                                                # the line runs UP the pitch, along the wall
    assert y < by                                                      # the body on the open side of the ball
    # The same ball with the margin off: the spot lands inside the boards.
    b0 = Chase(ChaseParams(board_margin=0.0), goal=(hx, 0.0), bounds=(hx, hy), goal_w=0.7)
    _, y0, _, _, _ = b0._plan(odom, _seen_ball(b0, odom, bx, by))
    assert hy - abs(y0) < p.board_margin
    # The OWN end board (the goal line points away from it, so the spot
    # behind the ball is inside the wall): the line runs toward the middle.
    odom = (-hx + 0.6, 0.9, math.pi)
    bx, by = -hx + 0.04, 0.9
    x, y, foot, h, mode = b._plan(odom, _seen_ball(b, odom, bx, by))
    assert hx - abs(x) >= p.board_margin and abs(h + math.pi / 2) < 0.3
    # The attacked end board beside the mouth: the goal line's own spot is
    # clear (it lies back up the pitch), so the plan is the ordinary one.
    odom = (hx - 0.6, 0.9, 0.0)
    bx, by = hx - 0.04, 0.9
    x, y, foot, h, mode = b._plan(odom, _seen_ball(b, odom, bx, by))
    assert hx - abs(x) >= p.board_margin and h < -0.3
    # In the open the plan is untouched by the margin (fresh brains: the
    # foot hysteresis remembers the last spot).
    odom = (-0.5, 0.0, 0.0)
    b1 = Chase(p, goal=(hx, 0.0), bounds=(hx, hy), goal_w=0.7)
    b2 = Chase(ChaseParams(board_margin=0.0), goal=(hx, 0.0), bounds=(hx, hy), goal_w=0.7)
    a = b1._plan(odom, _seen_ball(b1, odom, 0.0, 0.0))
    c = b2._plan(odom, _seen_ball(b2, odom, 0.0, 0.0))
    assert np.allclose(a[:2], c[:2]) and a[2:] == c[2:]
