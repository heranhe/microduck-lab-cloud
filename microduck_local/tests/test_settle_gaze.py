"""The settle looks down (`settle_gaze_neck` / `settle_head_down`, roadmap
12d, built after 12aj): standing on the kick spot, the gaze routes to the
neck and the head clip deepens, so the swing gate can read a sighting of a
ball the walking gaze is blind to (0.08-0.19 m from the root). Settle ONLY;
off, the head tuple is the shipped one to the bit."""

from __future__ import annotations

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.runtime import Senses


def _on_the_spot(state: str, spot_x: float = 0.0, **knobs) -> tuple[float, float, float, float]:
    """A duck at the origin facing +x with a kick spot planned `spot_x` ahead
    (line laid, clocks fresh), stepped once blind: the line-up branch aims
    the gaze at the spot's ball, and the head tuple it emits is what the
    servo gets this tick. A spot AT the feet is a settle; one 0.3 m out is
    a line-up still walking in (on the spot it would become a settle within
    the tick, and the head is emitted after that transition)."""
    b = Chase(ChaseParams(gaze_still=True, **knobs))
    b.state = state
    b.spot = (spot_x, 0.0, "kick_right", 0.0, "kick")
    b.lined = True
    b.t_state = 5.0
    out = b.step(Senses(t=5.0, det=None, det_age=None, odom=(0.0, 0.0, 0.0), speed=0.0))
    assert b.state == state, b.state                          # one tick, still there
    return out.head


def test_the_defaults_are_off():
    p = ChaseParams()
    assert p.settle_gaze_neck == 0.0 and p.settle_head_down == 0.0


def test_off_the_settle_gaze_is_the_shipped_one_to_the_bit():
    head = _on_the_spot("settle")
    assert head[0] == 0.0 and str(head[0]) == "0.0"           # the neck slot: plain 0.0, not -0.0
    assert 0.0 < head[1] <= ChaseParams().head_down            # the head clipped at the shipped cap


def test_on_the_settle_looks_deeper_through_the_neck():
    head = _on_the_spot("settle", settle_gaze_neck=1.0, settle_head_down=1.0)
    assert head[0] < 0.0                                       # the neck pulls the same way
    assert head[1] > ChaseParams().head_down                   # past the walking clip
    assert abs(head[0]) == head[1]                             # the whole command mirrored (fraction 1.0)


def test_the_knobs_do_nothing_outside_the_settle():
    head = _on_the_spot("lineup", spot_x=0.30, settle_gaze_neck=1.0, settle_head_down=1.0)
    assert head[0] == 0.0 and head[1] <= ChaseParams().head_down


def test_the_gaze_law_takes_its_overrides_only_when_given():
    b = Chase(ChaseParams())
    assert b._gaze(0.08) == b._gaze(0.08, None, None)
    assert b._gaze(0.08, 1.0, 1.0) > b._gaze(0.08)            # deeper cap, and it is used
    assert b._head_pose(0.5) == (0.0, 0.5, 0.0, 0.0)
    assert b._head_pose(0.5, 0.0, 1.0) == (-0.5, 0.5, 0.0, 0.0)
