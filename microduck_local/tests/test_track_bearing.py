"""A coasting track's bearing after the duck WALKS (roadmap 12af / 12ar).

`Track.bearing` is what the last hit measured, turned since only by the
body's yaw — so a duck that walks while coasting carries a bearing that
stopped meaning "bearing" when it moved (12af's trace: 112° off), while the
`xy` beside it is fine. `Track.bearing_from` / `range_from` read the pair off
`xy`, and `TrackerParams.coast_from_xy` wires them into the coast. The knob
ships OFF: with it off every number here is the old behaviour to the bit.
"""

import math

import pytest

from microduck_local.brain.tracker import Tracker, TrackerParams
from microduck_local.sensors.detector import Detection, DetectionFrame


def ball(bearing, rng):
    return DetectionFrame(0.0, [Detection("ball", "ball0", bearing, 0.0, 2 * math.atan(0.07 / rng), rng, 0.9)])


def frame(t, bearing, rng):
    return DetectionFrame(t, [Detection("ball", "ball0", bearing, 0.0, 2 * math.atan(0.07 / rng), rng, 0.9)])


def test_bearing_from_and_range_from_are_the_inverse_of_the_placement():
    """A hit places `xy` at pos + range·(cos, sin)(yaw + bearing); read back
    from the SAME pose the pair comes out unchanged, and from another pose it
    is the honest geometry to the remembered point."""
    tr = Tracker()
    tr.update(frame(0.0, 0.3, 1.2), 0.0, yaw=0.4, pos=(0.5, -0.2))
    t0 = tr.best("ball", 0.0, min_hits=1)
    assert t0.xy is not None
    assert abs(t0.bearing_from((0.5, -0.2), 0.4) - t0.bearing) < 1e-12
    assert abs(t0.range_from((0.5, -0.2)) - t0.range) < 1e-12
    # From somewhere else: plain geometry to `xy`, wrapped into (-pi, pi].
    pos, yaw = (-1.0, 0.8), -2.9
    want = math.atan2(t0.xy[1] - pos[1], t0.xy[0] - pos[0]) - yaw
    want = math.atan2(math.sin(want), math.cos(want))
    assert abs(t0.bearing_from(pos, yaw) - want) < 1e-12
    assert abs(t0.range_from(pos) - math.dist(t0.xy, pos)) < 1e-12
    # No position, no answer (a track from a frame with no odometry).
    plain = Tracker()
    plain.update(ball(0.0, 1.0), 0.0, yaw=0.0)
    p0 = plain.best("ball", 0.0, min_hits=1)
    assert p0.xy is None and p0.bearing_from((0.0, 0.0), 0.0) is None and p0.range_from((0.0, 0.0)) is None


@pytest.mark.parametrize("on", [False, True])
def test_a_turn_and_a_walk_with_no_new_hit(on):
    """The item's case: a ball seen dead ahead at 1.0 m, then the duck turns
    90° left and walks 0.3 m along its new heading and never sees it again.
    Knob ON, the bearing and range follow `xy`; knob OFF, they are the old
    numbers — the turn, and nothing for the walk."""
    tk = Tracker(TrackerParams(coast_from_xy=on))
    tk.update(frame(0.0, 0.0, 1.0), 0.0, yaw=0.0, pos=(0.0, 0.0))
    t0 = tk.best("ball", 0.0, min_hits=1)
    assert abs(t0.xy[0] - 1.0) < 1e-9 and abs(t0.xy[1]) < 1e-9
    # Turn to +90°, then walk 0.3 m along it, in ticks, with no frame at all.
    for k in range(1, 11):
        tk.update(None, 0.02 * k, yaw=math.pi / 2, pos=(0.0, 0.03 * k))
    tr = tk.best("ball", 0.2, min_hits=1)
    assert tr is not None and tr.age(0.2) == pytest.approx(0.2)      # never hit again
    assert tr.xy == t0.xy                                            # the memory itself never moves
    true_b = math.atan2(0.0 - 0.3, 1.0 - 0.0) - math.pi / 2
    true_r = math.hypot(1.0, 0.3)
    if on:
        assert tr.bearing == pytest.approx(true_b, abs=1e-9)         # -1.862 rad
        assert tr.range == pytest.approx(true_r, abs=1e-9)           # 1.044 m
    else:
        assert tr.bearing == pytest.approx(-math.pi / 2, abs=1e-9)   # the turn only
        assert tr.range == pytest.approx(1.0, abs=1e-9)              # the walk is invisible
        assert abs(tr.bearing - true_b) > math.radians(15)           # …and that is 16.7° of lie
        assert abs(tr.range - true_r) > 0.04


def test_the_knob_off_is_the_old_tracker_to_the_bit():
    """Every coast path with the knob off matches a tracker built before it
    existed — including one fed no odometry at all, where ON has nothing to
    read and must fall back to the same yaw rotation."""
    def run(p, with_pos):
        tk = Tracker(p)
        out = []
        for k in range(12):
            t = 0.1 * k
            fr = frame(t, 0.2, 1.0 + 0.05 * k) if k < 3 else None
            yaw = 0.15 * k
            tk.update(fr, t, yaw=yaw, pos=(0.04 * k, -0.02 * k) if with_pos else None)
            b = tk.best("ball", t, min_hits=1)
            out.append(None if b is None else (b.bearing, b.range, b.xy))
        return out

    base = run(TrackerParams(), True)
    assert run(TrackerParams(coast_from_xy=False), True) == base
    assert run(TrackerParams(coast_from_xy=True), True) != base          # the knob is LIVE
    # No odometry: nothing is placed, so ON and OFF are the same rows.
    assert run(TrackerParams(coast_from_xy=True), False) == run(TrackerParams(), False)


def test_the_re_derivation_does_not_drift_or_move_the_memory():
    """It is read off `xy` every tick, not integrated, so 200 ticks of walking
    in a circle leave the bearing exact and the position untouched."""
    tk = Tracker(TrackerParams(coast_from_xy=True))
    tk.update(frame(0.0, -0.4, 1.5), 0.0, yaw=0.0, pos=(0.0, 0.0))
    tr = tk.best("ball", 0.0, min_hits=1)
    xy = tr.xy
    for k in range(1, 201):
        a = 0.05 * k
        pos, yaw = (0.4 * math.cos(a), 0.4 * math.sin(a)), a
        tk.tracks[0].last_t = 0.01 * k                    # keep it alive past `coast_s`
        tk.update(None, 0.01 * k, yaw=yaw, pos=pos)
        want = math.atan2(xy[1] - pos[1], xy[0] - pos[0]) - yaw
        assert tk.tracks[0].bearing == pytest.approx(math.atan2(math.sin(want), math.cos(want)), abs=1e-12)
        assert tk.tracks[0].range == pytest.approx(math.dist(xy, pos), abs=1e-12)
    assert tk.tracks[0].xy == xy


def test_a_hit_still_wins_over_the_remembered_position():
    """The re-derivation happens before the frame is folded, so a fresh
    detection still sets the bearing — and re-places `xy` from it."""
    tk = Tracker(TrackerParams(coast_from_xy=True))
    tk.update(frame(0.0, 0.0, 1.0), 0.0, yaw=0.0, pos=(0.0, 0.0))
    tk.update(None, 0.1, yaw=0.0, pos=(0.3, 0.0))
    assert tk.tracks[0].range == pytest.approx(0.7, abs=1e-9)
    # The ball is really 0.9 m ahead now (it rolled): the hit says so.
    tk.update(frame(0.2, 0.0, 0.9), 0.2, yaw=0.0, pos=(0.3, 0.0))
    tr = tk.best("ball", 0.2, min_hits=1)
    assert tr.hits == 2 and tr.age(0.2) == 0.0
    assert tr.range == pytest.approx(0.7 + tk.p.smooth * (0.9 - 0.7), abs=1e-9)
    assert tr.xy[0] == pytest.approx(0.3 + tr.range, abs=1e-9)


def test_the_knob_is_set_from_the_environment_like_every_other_battery_knob():
    """`MICRODUCK_TRACKER` fills in what the caller did not name; an unknown
    name or an unreadable value raises rather than measuring the default."""
    assert TrackerParams().coast_from_xy is False
    assert TrackerParams.env_over(spec="") == {}
    assert TrackerParams.env_over(spec="coast_from_xy=1") == {"coast_from_xy": True}
    assert TrackerParams.env_over(spec="coast_from_xy=on,rest_coast_s=30") == {
        "coast_from_xy": True, "rest_coast_s": 30.0}
    # The caller wins: `Chase` hands `rest_coast_s` down from its own spec.
    assert TrackerParams.env_over({"rest_coast_s": 0.0}, spec="rest_coast_s=30") == {}
    with pytest.raises(ValueError):
        TrackerParams.env_over(spec="coats_from_xy=1")
    with pytest.raises(ValueError):
        TrackerParams.env_over(spec="coast_from_xy=yes-please")
    with pytest.raises(ValueError):
        TrackerParams.env_over(spec="ignore=post")
    with pytest.raises(ValueError):
        TrackerParams.env_over(spec="coast_from_xy")


def test_the_duck_s_own_tracker_reads_the_environment(monkeypatch):
    """`for_detector` is the constructor `Chase` uses; a bare
    `TrackerParams()` is NOT, so goldens and tests do not move under the var."""
    monkeypatch.setenv("MICRODUCK_TRACKER", "coast_from_xy=1")
    assert TrackerParams.for_detector("datasheet").coast_from_xy is True
    assert TrackerParams().coast_from_xy is False
    # …and it does not disturb the knobs the brain hands down.
    assert TrackerParams.for_detector("datasheet", rest_coast_s=30.0).rest_coast_s == 30.0
