"""The head on the REMEMBERED ball in support and retreat (`head_memory_s`,
roadmap 12ag). The loss audit's trace put 74% of the long blind time in
support, retreat and avoid, states that consulted the ball in none of the
head laws; this one yaws the head toward the board's ball, else the duck's
own last sighting, for `head_memory_s`. Ships ON at 30 s in
support/wait/retreat; `avoid` is recorded off."""

from __future__ import annotations

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.runtime import Senses
from microduck_local.brain.team import Team
from microduck_local.sensors.detector import Detection, DetectionFrame


def _supporter(**knobs) -> tuple[Chase, Team]:
    """d0 at the origin facing +x, blind; teammate d1 is on the ball and
    reports it at (0, 1): straight off d0's LEFT shoulder."""
    tm = Team("cream", half_x=1.5)
    b = Chase(ChaseParams(**knobs), goal=(1.5, 0.0), team=tm, duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7)
    return b, tm


def _run(b: Chase, tm: Team, until: float = 6.0):
    t, out = 0.0, None
    while t <= until:
        tm.claim("d1", t, 0.2, (0.0, 1.0), (0.1, 0.9, 0.0), 0.05)   # d1 is right on it, every tick
        out = b.step(Senses(t=t, det=None, det_age=None, odom=(0.0, 0.0, 0.0), speed=0.0))
        t += 0.02
    return out


def test_the_defaults_are_the_measured_ones():
    p = ChaseParams()
    assert p.head_memory_s == 30.0 and p.head_memory_states == "support+wait+retreat"


def test_a_supporter_looks_at_the_boards_ball():
    b, tm = _supporter()
    out = _run(b, tm)
    assert b.state in ("support", "retreat"), b.state    # a still supporter unsticks into a retreat; both are the knob's states
    assert out.head[2] > 1.0, out.head                        # the ball is at +90 deg: the yaw goes to its clip


def test_off_the_supporter_looks_straight_ahead():
    b, tm = _supporter(head_memory_s=0.0)
    out = _run(b, tm)
    assert b.state in ("support", "retreat"), b.state    # a still supporter unsticks into a retreat; both are the knob's states
    assert out.head[2] == 0.0, out.head


def test_the_own_memory_outlives_the_loss_only_for_the_head():
    """A ball seen, then lost: with the knob the memory stays (the head's),
    but nothing walks to it (`seek_s` is the walk's own knob); without it
    the memory is gone a tick after the loss, as before."""
    for knob, kept in ((30.0, True), (0.0, False)):
        b = Chase(ChaseParams(head_memory_s=knob))
        det = DetectionFrame(t=1.0, detections=[Detection("ball", "", 0.0, -0.3, 0.1, 0.6, 0.9)])
        b.step(Senses(t=1.0, det=det, det_age=0.0, odom=(0.0, 0.0, 0.0), speed=0.0))
        assert b.memory is not None
        for k in range(1, 200):                                # 4 s blind
            b.step(Senses(t=1.0 + 0.02 * k, det=None, det_age=None, odom=(0.0, 0.0, 0.0), speed=0.0))
        assert (b.memory is not None) is kept
        assert b.state != "seek"
