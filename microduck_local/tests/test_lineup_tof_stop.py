"""The line-up's own stop (`lineup_tof_stop` / `lineup_tof_within`, roadmap
12am): inside the window of a kick spot the selector passed as clear of every
board by the body extent, the walk stops at the shorter distance instead of
`tof_stop`, so a line-up can reach a spot 0.13-0.20 m from a wall. At 0 the
old bumper everywhere; it ships at 0.12 and applies ONLY in lineup/settle,
only near the spot, only for a kick spot, and only when that spot is
body-clear."""

from __future__ import annotations

import numpy as np

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.runtime import Senses
from microduck_local.sensors.tof import TofFrame

BOUNDS = (1.5, 1.25)


def _tof(t: float, ahead_mm: int) -> TofFrame:
    f = np.full((8, 8), 4000, np.uint16)
    f[2:5, 3:5] = ahead_mm                                          # the ahead columns
    return TofFrame(t=t, depth_mm=f, valid=np.ones((8, 8), bool))


def _walk(state: str, spot, ahead_mm: int, **knobs) -> float:
    """The forward speed the brain commands this tick: at the origin facing
    +x, a laid spot, something body-height `ahead_mm` in front."""
    b = Chase(ChaseParams(gaze_still=True, **knobs), goal=(BOUNDS[0], 0.0), bounds=BOUNDS, goal_w=0.7, duck_id="d0")
    b.state = state
    b.spot = spot
    b.lined = True
    b.t_state = 5.0
    out = b.step(Senses(t=5.0, tof=_tof(5.0, ahead_mm), tof_age=0.0, det=None, det_age=None,
                        odom=(0.0, 0.0, 0.0), speed=0.0))
    return float(out.twist[0])


NEAR = (0.30, 0.0, "kick_right", 0.0, "kick")                       # 0.3 m ahead, mid-pitch: body-clear


def test_the_default_is_the_shipped_stop():
    p = ChaseParams()
    assert p.lineup_tof_stop == 0.12 and p.lineup_tof_within == 0.45   # ships on (roadmap 12am)


def test_off_the_bumper_stops_a_line_up_at_tof_stop():
    assert _walk("lineup", NEAR, 200, lineup_tof_stop=0.0) == 0.0    # a wall at 0.20 < tof_stop 0.30: no walking
    assert _walk("lineup", NEAR, 400, lineup_tof_stop=0.0) > 0.0     # nothing near: walking


def test_on_the_line_up_walks_to_its_spot_past_the_bumper():
    assert _walk("lineup", NEAR, 200, lineup_tof_stop=0.12) > 0.0    # 0.20 >= 0.12: walk
    assert _walk("lineup", NEAR, 100, lineup_tof_stop=0.12) == 0.0   # 0.10 < 0.12: still stops
    on_spot = (0.0, 0.0, "kick_right", 0.0, "kick")                  # a settle IS on its spot: it stands, knob or not
    assert _walk("settle", on_spot, 200, lineup_tof_stop=0.12) == 0.0


def test_the_shorter_stop_needs_the_spot_near_and_body_clear():
    far = (0.80, 0.0, "kick_right", 0.0, "kick")                     # outside lineup_tof_within 0.45
    assert _walk("lineup", far, 200, lineup_tof_stop=0.12) == 0.0
    in_wall = (BOUNDS[0] - 0.05, BOUNDS[1] - 0.05, "kick_right", 0.0, "kick")   # the body cannot occupy it
    b = Chase(ChaseParams(), bounds=BOUNDS, goal=(BOUNDS[0], 0.0), goal_w=0.7)
    assert not b._spot_body_clear(*in_wall[:2]) and b._spot_body_clear(*NEAR[:2])
    push = (0.30, 0.0, None, 0.0, "push")                            # a body-clear PUSH spot walks too (12g's board push needs it)
    assert _walk("lineup", push, 200, lineup_tof_stop=0.12) > 0.0
    assert _walk("chase", None, 200, lineup_tof_stop=0.12) == 0.0    # not a line-up: the shipped bumper
