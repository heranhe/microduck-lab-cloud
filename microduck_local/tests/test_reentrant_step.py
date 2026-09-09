"""A probe that re-enters `Chase.step` measures neither variant.

The fourth shape of a confident zero, and the only one none of the other
three catch. On 2026-09-09 a probe wrapped `Chase._hold_target` and called
the real method a second time each tick to compare clamped against unclamped,
and reported a 7.56% firing rate for `post_margin`. A direct sweep of the same
knob showed bit-identical results at 0, 0.25 and 0.30 — the rate was zero and
the knob was reverted (roadmap 12o). Nothing detected the discrepancy except
an independent measurement, which is the durable lesson: **when a probe and a
direct sweep disagree the sweep wins, because the probe has more surface area
to be wrong on.** Sweep first; build a probe only when a sweep cannot answer.

`step` mutates from its opening lines — `gait.update`, the localiser, every
counter — so a second call for one tick double-advances all of it. These lock
a guard that says so from the inside, where it catches wrappers, decorators
and direct private-method calls alike rather than only what a reader of probe
source thought to look for.
"""

from __future__ import annotations

import warnings

import pytest

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.controllers import ReentrantStepWarning


def brain():
    return REGISTRY.make("chase", goal=(2.0, 0.0), bounds=(1.7, 1.4))


def senses(t: float) -> Senses:
    return Senses(t=t, tof=None, tof_age=None, det=None, det_age=None,
                  speed=0.0, odom=(0.0, 0.0, 0.0), skill=None, bumped=False)


def test_stepping_twice_for_one_tick_warns():
    b = brain()
    b.step(senses(1.0))
    with pytest.warns(ReentrantStepWarning, match="called twice"):
        b.step(senses(1.0))


def test_the_warning_names_the_remedy_not_just_the_fault():
    b = brain()
    b.step(senses(1.0))
    with pytest.warns(ReentrantStepWarning) as rec:
        b.step(senses(1.0))
    msg = str(rec[0].message)
    assert "Step once and read the flags afterwards" in msg
    assert "double-advances" in msg


def test_a_normal_drive_loop_never_warns():
    """The harnesses are correct and must stay silent — a check that cries
    wolf on the ordinary path is one that gets deleted."""
    b = brain()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReentrantStepWarning)
        for i in range(200):
            b.step(senses(i * 0.02))


def test_two_brains_on_the_same_tick_never_warn():
    """Every duck is stepped at the same `t` each tick; the state is per-brain
    and a shared clock is not re-entry."""
    a, c = brain(), brain()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReentrantStepWarning)
        for i in range(50):
            a.step(senses(i * 0.02))
            c.step(senses(i * 0.02))


def test_it_warns_once_per_brain_not_once_a_tick():
    """A per-tick warning would bury a battery's output and be filtered off."""
    b = brain()
    b.step(senses(1.0))
    with pytest.warns(ReentrantStepWarning) as rec:
        b.step(senses(1.0))
    assert len(rec) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReentrantStepWarning)
        for _ in range(5):
            b.step(senses(1.0))          # already reported; stays quiet


def test_reset_starts_the_comparison_over():
    """An episode boundary may legitimately restart the clock, and that is not
    a probe re-entering anything."""
    b = brain()
    b.step(senses(1.0))
    b.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReentrantStepWarning)
        b.step(senses(1.0))


def test_going_backwards_in_time_is_not_re_entry():
    """Only an exact repeat is re-entry; a rewound clock is a new run."""
    b = brain()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReentrantStepWarning)
        b.step(senses(5.0))
        b.step(senses(1.0))


def test_the_guard_sits_above_the_first_mutation():
    """It has to fire BEFORE `gait.update`, or the thing it warns about has
    already happened by the time it says so."""
    import inspect

    from microduck_local.brain.controllers import Chase
    src = inspect.getsource(Chase.step)
    assert src.index("_reentry_warned") < src.index("self.gait.update")


def test_the_warning_is_a_runtime_warning_so_a_battery_can_promote_it():
    assert issubclass(ReentrantStepWarning, RuntimeWarning)
