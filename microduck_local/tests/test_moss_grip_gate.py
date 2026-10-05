"""tidy_moss's lift gate on the wrist depth camera (`grip_depth_max_m`)."""
import dataclasses

import pytest

from microduck_local.brain.runtime import Senses
from microduck_local.brain.tidy_moss import TidyMoss
from microduck_local.robots import moss

pytestmark = pytest.mark.skipif(
    not moss.moss_ready(),
    reason="MOSS assets missing — uv run fetch-robot moss")


def _s(t, grip):
    return Senses(t=t, target_obs={"grip": grip})


def _ticks(b, grip, t0, seconds, dt=0.02):
    """Ask the gate every tick for `seconds`; the first time it says yes."""
    t = t0
    while t < t0 + seconds + 1e-9:
        if b._grip_is_deep(_s(t, grip)):
            return t
        t += dt
    return None


def test_deep_grip_lifts_at_once():
    b = TidyMoss()
    assert b._grip_is_deep(_s(10.0, (0.005, -0.006, 0.009)))


def test_no_reading_is_not_evidence():
    b = TidyMoss()
    assert b._grip_is_deep(_s(10.0, None))
    assert TidyMoss()._grip_is_deep(Senses(t=10.0))


def test_shallow_grip_waits_then_lifts_anyway():
    b = TidyMoss()
    shallow = (0.018, -0.017, 0.029)            # the lift-loss median, 4.2 cm
    assert not b._grip_is_deep(_s(10.0, shallow))
    yes = _ticks(b, shallow, 10.02, 3.0)
    assert yes is not None
    assert abs(yes - (10.0 + b.p.grip_depth_wait_s)) < 0.03


def test_reseated_grip_clears_the_block():
    b = TidyMoss()
    assert not b._grip_is_deep(_s(10.0, (0.0, 0.0, 0.04)))
    assert b._grip_is_deep(_s(10.02, (0.0, 0.0, 0.01)))


def test_a_new_grip_restarts_the_wait():
    """A gap between asks is a fresh grip: an old block must not carry over
    and let a shallow grip through early."""
    b = TidyMoss()
    shallow = (0.0, 0.0, 0.04)
    _ticks(b, shallow, 10.0, 1.5)               # blocked 1.5 s, then lost
    assert not b._grip_is_deep(_s(20.0, shallow))
    assert not b._grip_is_deep(_s(21.0, shallow))  # 1.0 s into the new wait


def test_small_objects_skip_the_gate():
    """A butt reads 'shallow' in a good grip; only big objects are gated."""
    b = TidyMoss()
    b._fix_size = 0.03                                   # a butt
    assert b._grip_is_deep(_s(10.0, (0.0, 0.0, 0.05)))
    b = TidyMoss()
    b._fix_size = 0.115                                  # a can
    assert not b._grip_is_deep(_s(10.0, (0.0, 0.0, 0.05)))


def test_gate_off():
    b = TidyMoss()
    b.p = dataclasses.replace(b.p, grip_depth_max_m=None)
    assert b._grip_is_deep(_s(10.0, (0.0, 0.0, 0.2)))


def test_handover_asks_the_gate():
    """The creep's handover condition must include the gate, or the
    parameter is decoration."""
    import inspect
    src = inspect.getsource(TidyMoss.step)
    i = src.index("self._grip_is_a_grasp(senses)")
    assert "self._grip_is_deep(senses)" in src[i:i + 200]
