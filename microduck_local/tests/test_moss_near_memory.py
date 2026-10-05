"""The range floor is TWO floors: one for targeting, one for the memory.

`_toys_in_view` fed both `_see` and `_remember`, so `min_x` (0.30 m) did not
merely stop the rover driving at litter it could not reach — it stopped the
brain writing that litter down at all. Measured cost: 7.6-13.3% of every toy
detection, all of it ahead of the front bumper
(`scripts/probe_moss_dropped_dets.py`). These are the three things the split
has to keep true, and each one has been shown to fail without it.
"""
import dataclasses
import math

import pytest

from microduck_local.brain.runtime import Senses
from microduck_local.robots import moss
from microduck_local.sensors.detector import Detection, DetectionFrame

pytestmark = pytest.mark.skipif(
    not moss.moss_ready(),
    reason="MOSS assets missing — uv run fetch-robot moss")

NEAR = moss.FRONT_EXTENT_M          # the split's floor for the memory


def _frame(t, x, name="cap0", radius=0.02, bearing=0.0):
    """One toy detection at BASE-frame `x` straight ahead of the camera."""
    rng = (x - moss.CAMERA_POS[0]) / math.cos(bearing)
    assert rng > 0.0, "behind the camera"
    w = 2.0 * math.atan(radius / rng)
    return DetectionFrame(t=t, detections=[
        Detection("toy", name, bearing, -0.05, w, rng, 1.0)])


def _brain(**kw):
    from microduck_local.brain.tidy_moss import TidyMoss
    b = TidyMoss()
    b.p = dataclasses.replace(b.p, **kw)
    assert all(getattr(b.p, k) == v for k, v in kw.items())
    return b


def _remembered(b, x, *, state, t=0.02):
    """Does a detection at base x reach the memory from `state`?"""
    b._to(state, 0.0)
    s = Senses(t=t, odom=(0.0, 0.0, 0.0), speed=0.0, det=_frame(t - 0.01, x))
    b._carry_fix(s)
    b._remember(s, t)
    return [] if b._mem is None else list(b._mem.items)


def test_toys_in_view_honours_an_explicit_floor():
    b = _brain()
    frame = _frame(0.0, 0.25)
    assert b._toys_in_view(frame) == []                  # default floor, 0.30
    kept = b._toys_in_view(frame, NEAR)
    assert len(kept) == 1
    assert abs(kept[0][1] - 0.25) < 1e-6
    # ...and the floor is a floor, not a switch: below the bumper is still out.
    assert b._toys_in_view(_frame(0.0, 0.17), NEAR) == []


def test_a_close_detection_reaches_the_memory_only_when_the_floors_differ():
    split = _remembered(_brain(near_min_x=NEAR), 0.25, state="search")
    assert len(split) == 1, "the camera saw it and the brain must write it down"
    assert math.hypot(split[0].x - 0.25, split[0].y) < 0.05
    joined = _remembered(_brain(near_min_x=0.30), 0.25, state="search")
    assert joined == [], "near_min_x == min_x is the old behaviour"
    # A detection past BOTH floors is remembered either way — this test would
    # pass on a brain that simply remembers everything, without it.
    assert len(_remembered(_brain(near_min_x=0.30), 0.60, state="search")) == 1


def test_the_memory_floor_does_not_move_the_target_floor():
    """The failure this guards: lowering `min_x` itself doubled the drops,
    because the arm unfolds into anything nearer than it can work on."""
    b = _brain(near_min_x=NEAR, object_memory=False)
    b._to("search", 0.0)
    for k in range(1, 6):
        t = 0.02 * k
        b.step(Senses(t=t, odom=(0.0, 0.0, 0.0), speed=0.0,
                      det=_frame(t - 0.01, 0.25)))
    assert b.state == "search", "0.25 m is inside min_x: not a target"
    assert b._mem is not None and len(b._mem.items) == 1, "but it IS known"
    # The same detection past `min_x` is a target, so the state machine works.
    b2 = _brain(near_min_x=NEAR, object_memory=False)
    b2._to("search", 0.0)
    b2.step(Senses(t=0.02, odom=(0.0, 0.0, 0.0), speed=0.0,
                   det=_frame(0.01, 0.45)))
    assert b2.state == "approach"


def test_the_memory_floor_lifts_only_while_the_arm_is_stowed():
    """Deployed, the gripper and whatever is in it fill the near frame: a
    carried object at 0.25 m would be written down as litter lying wherever
    the rover happened to be standing."""
    for state in ("search", "approach"):
        assert _remembered(_brain(near_min_x=NEAR), 0.25, state=state), state
    for state in ("deploy", "creep", "pinch", "lift", "stow"):
        assert _remembered(_brain(near_min_x=NEAR), 0.25, state=state) == [], state
