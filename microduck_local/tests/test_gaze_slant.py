"""`gaze_slant` (roadmap 12ap): the gaze law treats its range as the slant it
is, asin(h / r), instead of ground distance, atan2(h, r). Off, the shipped law
to the bit; on, deeper at close range, the same far away."""

from __future__ import annotations

import math

from microduck_local.brain.controllers import Chase, ChaseParams


def test_the_default_is_off_and_the_law_is_unchanged():
    b = Chase(ChaseParams())
    assert b.p.gaze_slant is False
    h = b.p.cam_z - 0.035
    want = math.atan2(h, 0.27)
    assert abs(b._gaze(0.27) - (want - b.p.cam_level) / b.p.head_gain) < 1e-9


def test_on_it_is_deeper_close_and_converges_far():
    on = Chase(ChaseParams(gaze_slant=True, head_down=2.0))
    h = on.p.cam_z - 0.035
    for r in (0.20, 0.27):
        deeper = (math.asin(h / r) - math.atan2(h, r))
        assert deeper > math.radians(5)
        assert on._gaze(r) > Chase(ChaseParams(head_down=2.0))._gaze(r)
    assert abs(on._gaze(2.0) - Chase(ChaseParams(head_down=2.0))._gaze(2.0)) < 0.01   # far: the same
    assert on._gaze(0.05) <= 2.0                                                       # inside h: clipped, no NaN
