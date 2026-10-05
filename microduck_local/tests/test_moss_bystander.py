"""Steering round the litter the rover is NOT going for.

`bystander_avoid_m` reads the object memory as obstacles while driving. The
measurement behind it: only 12% of what the driving surfaces touch was ever
inside the brain's target gate, and a quarter to two thirds of it is in the
memory at the moment of contact (`scripts/probe_moss_blindspot.py`). These
are the properties that make it a dodge rather than a flinch, and each one
has been shown to fail against a planted regression.
"""

import dataclasses
import math

import pytest

from microduck_local.brain.runtime import Senses
from microduck_local.robots import moss

pytestmark = pytest.mark.skipif(
    not moss.moss_ready(), reason="MOSS assets missing — uv run fetch-robot moss"
)

LOOK = 0.45
CORRIDOR = moss.HALF_WIDTH_M + 0.03


def _brain(**kw):
    from microduck_local.brain.tidy_moss import TidyMoss

    b = TidyMoss()
    b.p = dataclasses.replace(b.p, bystander_avoid_m=LOOK, **kw)
    b._odom = (0.0, 0.0, 0.0)  # at the origin, facing +x
    b._to("approach", 0.0)
    return b


def _remember(b, *pts, times=3):
    """Put confirmed entries at these WORLD (x, y) — the rover is at the
    origin facing +x, so world and base coincide."""
    from microduck_local.brain.moss_search import ObjectMemory

    if b._mem is None:
        b._mem = ObjectMemory()
    for k in range(times):
        b._mem.observe([(x, y, 0.04, math.hypot(x, y)) for x, y in pts], 0.1 * k)
    assert all(b._mem.confirmed(e) for e in b._mem.items) == (times > 1)
    return b._mem.items


def test_something_dead_ahead_slows_the_rover_and_turns_it_away():
    b = _brain()
    _remember(b, (0.25, 0.06))  # ahead and slightly LEFT
    (vx, _vy, wz), note = b._dodge((0.30, 0.0, 0.0), "approach")
    assert vx < 0.30, "a thing in the corridor has to cost speed"
    assert wz < 0.0, "+wz is left, so it must steer RIGHT away from it"
    assert "in the way" in note
    # ...and mirrored.
    b2 = _brain()
    _remember(b2, (0.25, -0.06))
    assert b2._dodge((0.30, 0.0, 0.0), "approach")[0][2] > 0.0


def test_the_thing_it_is_driving_at_is_never_a_bystander():
    """Steering away from the target would stall the mission — the failure
    mode the whole change is scored against."""
    b = _brain()
    _remember(b, (0.25, 0.06))
    b._fix = (0.25, 0.06)  # ...and it is the current target
    assert b._dodge((0.30, 0.0, 0.0), "approach") == ((0.30, 0.0, 0.0), "approach")


def test_it_ignores_what_is_out_of_the_corridor_or_out_of_range():
    b = _brain()
    _remember(b, (0.25, CORRIDOR + 0.02))  # beside the tracks
    assert b._dodge((0.30, 0.0, 0.0), "a")[0] == (0.30, 0.0, 0.0)
    b = _brain()
    _remember(b, (LOOK + 0.10, 0.0))  # too far ahead to matter
    assert b._dodge((0.30, 0.0, 0.0), "a")[0] == (0.30, 0.0, 0.0)
    b = _brain()
    _remember(b, (-0.25, 0.0))  # behind
    assert b._dodge((0.30, 0.0, 0.0), "a")[0] == (0.30, 0.0, 0.0)


def test_a_one_frame_phantom_does_not_move_the_wheels():
    b = _brain()
    _remember(b, (0.25, 0.0), times=1)  # seen once: not confirmed
    assert not any(b._mem.confirmed(e) for e in b._mem.items)
    assert b._dodge((0.30, 0.0, 0.0), "a")[0] == (0.30, 0.0, 0.0)


def test_a_cluster_steers_like_its_worst_member_not_like_a_crowd():
    """Litter lies together. Summing would swing the nose by the COUNT.

    The three have to sit further apart than `ObjectMemory`'s own association
    gate (0.10 m + 0.20 per m of range) or they fold into ONE entry and this
    passes without testing anything — which is what it did first, and the
    planted summing regression walked straight through it.
    """
    worst = (0.20, 0.00)  # nearest, and dead centre
    others = [(0.38, 0.17), (0.40, -0.17)]
    one = _brain()
    _remember(one, worst)
    many = _brain()
    _remember(many, worst, *others)
    assert len(many._mem.items) == 3, "the memory merged them: nothing tested"
    w1 = one._dodge((0.30, 0.0, 0.0), "a")[0][2]
    wn = many._dodge((0.30, 0.0, 0.0), "a")[0][2]
    assert abs(w1) > 0.0
    assert wn == w1, "three objects must not mean three nudges"


def test_it_does_not_fire_when_stopped_or_reversing():
    for vx in (0.0, -0.12):
        b = _brain()
        _remember(b, (0.25, 0.0))
        assert b._dodge((vx, 0.0, 0.0), "a")[0] == (vx, 0.0, 0.0), vx


def test_the_nudge_fades_out_instead_of_switching_off():
    """A step at the edge of the box would jerk the nose back."""
    b = _brain()
    _remember(b, (LOOK - 0.005, 0.0))
    assert abs(b._dodge((0.30, 0.0, 0.0), "a")[0][2]) < 0.02
    b = _brain()
    _remember(b, (0.25, CORRIDOR - 0.005))
    assert abs(b._dodge((0.30, 0.0, 0.0), "a")[0][2]) < 0.02


def test_the_off_switch_works_and_only_the_driving_states_dodge():
    from microduck_local.brain.tidy_moss import TidyMoss, TidyMossParams

    assert TidyMossParams().bystander_avoid_m > 0.0, "it SHIPS on (measured)"
    s = Senses(t=0.02, odom=(0.0, 0.0, 0.0), speed=0.0)
    off = TidyMoss()
    off.p = dataclasses.replace(off.p, bystander_avoid_m=0.0)
    off._odom = (0.0, 0.0, 0.0)
    off._to("approach", 0.0)
    _remember(off, (0.25, 0.05))
    off._fix, off._fix_t = (0.9, 0.0), 0.02
    assert off.step(s).twist[2] == 0.0, "0 must still be the off switch"
    # ...and with it on, a state that is deliberately closing the last few
    # centimetres on its own target is left alone. Asserted on whether `step`
    # CALLS the dodge, not on the twist: those states mostly command zero
    # anyway, so comparing twists would pass without the gate.
    for state in ("search", "approach", "deploy", "creep", "pinch", "lift", "stow"):
        b = _brain()
        b._to(state, 0.0)
        _remember(b, (0.25, 0.05))
        b._fix, b._fix_t = (0.9, 0.0), 0.02
        called = []
        real = b._dodge
        b._dodge = lambda tw, nt, _r=real, _c=called: (_c.append(1), _r(tw, nt))[1]
        b.step(Senses(t=0.02, odom=(0.0, 0.0, 0.0), speed=0.0, arm={}))
        assert bool(called) == (state in ("search", "approach")), state
