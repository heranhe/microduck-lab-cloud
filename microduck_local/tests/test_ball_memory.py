"""The resting-ball memory (roadmap Track 4 item 12f).

The floor has had rolling resistance since 2026-09-06, so a ball that stops
STAYS stopped — but the tracker forgot it on the same 2.5 s clock as a ball
that might have rolled anywhere, while the kick plan is a median 3.6 s old
when the swing fires. So at the moment that decides the kick the brain was
reasoning about a track that had already expired, which is why the ahead gate
could only fire on 5% of swings on its own.

A memory is only worth having if it is honest about when it stops being true,
so most of this file is about the ways it is voided.
"""

import math

import pytest

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.tracker import Track, Tracker, TrackerParams
from microduck_local.sensors.detector import Detection, DetectionFrame


def _ball(bearing=0.0, rng=0.5, name="ball0"):
    return Detection("ball", name, bearing, 0.0, 0.05, rng, 0.9)


def _feed(tr: Tracker, t: float, *, bearing=0.0, rng=0.5, pos=(0.0, 0.0), yaw=0.0):
    tr.update(DetectionFrame(t=t, detections=[_ball(bearing, rng)]), t, yaw, pos)


def test_at_rest_needs_two_hits_agreeing_it_is_slow():
    """Not 'we have no velocity for it': a ball seen ONCE, rolling, also has
    no velocity, and calling that at rest is how a memory becomes a lie."""
    p = TrackerParams(rest_coast_s=20.0)
    t = Track(id=1, cls="ball", bearing=0.0, elevation=0.0, width=0.05, range=0.5,
              conf=0.9, born_t=0.0, last_t=0.0, xy=(0.5, 0.0), vel_hits=0)
    assert not t.at_rest(p.rest_vel)                     # one look: not yet
    t.vel_hits, t.vel = 2, (0.0, 0.0)
    assert t.at_rest(p.rest_vel)                         # two looks, standing still
    t.vel = (0.4, 0.0)
    assert not t.at_rest(p.rest_vel)                     # …and a rolling one never is
    t.vel, t.xy = (0.0, 0.0), None
    assert not t.at_rest(p.rest_vel)                     # no position, nothing to remember


def test_a_resting_ball_outlives_the_coast_clock_and_a_rolling_one_does_not():
    for vel, kept in (((0.0, 0.0), True), ((0.6, 0.0), False)):
        tr = Tracker(TrackerParams(rest_coast_s=20.0))
        for k in range(4):                                # enough hits for a velocity
            _feed(tr, 0.2 * k, pos=(0.0, 0.0))
        assert tr.tracks, "no track was formed"
        tr.tracks[0].vel, tr.tracks[0].vel_hits = vel, 2
        tr.update(None, 10.0, 0.0, (0.0, 0.0))            # 10 s with no sighting
        assert bool(tr.tracks) is kept, f"vel={vel}"


def test_the_memory_is_off_by_default():
    """Every number in the roadmap was measured without it."""
    assert TrackerParams().rest_coast_s == 0.0
    assert ChaseParams().rest_predict_s == 0.0 and ChaseParams().rest_coast_s == 0.0
    tr = Tracker(TrackerParams())                         # default: the 2.5 s clock for everything
    for k in range(4):
        _feed(tr, 0.2 * k)
    tr.tracks[0].vel, tr.tracks[0].vel_hits = (0.0, 0.0), 2
    tr.update(None, 10.0, 0.0, (0.0, 0.0))
    assert not tr.tracks


def test_disturb_voids_the_memory_and_a_fresh_look_restores_it():
    tr = Tracker(TrackerParams(rest_coast_s=20.0))
    for k in range(4):
        _feed(tr, 0.2 * k, pos=(0.0, 0.0))
    t0 = tr.tracks[0]
    t0.vel, t0.vel_hits = (0.0, 0.0), 2
    assert t0.at_rest(0.05)
    assert tr.disturb("ball") == 1                        # something may have moved it
    assert not t0.at_rest(0.05)
    tr.update(None, 10.0, 0.0, (0.0, 0.0))
    assert not tr.tracks, "a disturbed memory must expire on the ordinary clock"
    # …and a fresh sighting settles it either way.
    tr2 = Tracker(TrackerParams(rest_coast_s=20.0))
    for k in range(4):
        _feed(tr2, 0.2 * k, pos=(0.0, 0.0))
    tr2.tracks[0].vel, tr2.tracks[0].vel_hits = (0.0, 0.0), 2
    tr2.disturb("ball")
    _feed(tr2, 1.0, pos=(0.0, 0.0))
    assert tr2.tracks[0].rest_block is False


def test_disturb_can_be_aimed_at_one_place_and_leaves_a_ball_elsewhere_alone():
    tr = Tracker(TrackerParams(rest_coast_s=20.0))
    for k in range(4):
        _feed(tr, 0.2 * k, pos=(0.0, 0.0))
    t0 = tr.tracks[0]
    t0.vel, t0.vel_hits = (0.0, 0.0), 2
    assert t0.xy is not None
    far = (t0.xy[0] + 3.0, t0.xy[1])
    assert tr.disturb("ball", far, 0.30) == 0             # a body three metres away is not on it
    assert t0.at_rest(0.05)
    assert tr.disturb("ball", t0.xy, 0.30) == 1           # standing on it is
    assert not t0.at_rest(0.05)


def test_disturb_only_touches_its_own_class():
    tr = Tracker(TrackerParams(rest_coast_s=20.0))
    fr = DetectionFrame(t=0.0, detections=[_ball(), Detection("duck", "d1", 0.6, 0.0, 0.1, 0.8, 0.9)])
    for k in range(4):
        tr.update(DetectionFrame(t=0.2 * k, detections=fr.detections), 0.2 * k, 0.0, (0.0, 0.0))
    assert {t.cls for t in tr.tracks} == {"ball", "duck"}
    tr.disturb("ball")
    assert [t.rest_block for t in tr.tracks if t.cls == "duck"] == [False]


def test_the_brain_wires_both_halves_from_one_knob_string():
    """A battery sets `MICRODUCK_CHASE` once; without `rest_coast_s` the track
    is gone before `rest_predict_s` can act on it, so both travel together."""
    b = Chase(ChaseParams(rest_predict_s=6.0, rest_coast_s=20.0, rest_vel=0.05),
              goal=(1.5, 0.0), bounds=(1.5, 1.25), goal_w=0.7)
    assert b.tracker.p.rest_coast_s == 20.0 and b.tracker.p.rest_vel == 0.05
    off = Chase(ChaseParams(), goal=(1.5, 0.0), bounds=(1.5, 1.25), goal_w=0.7)
    assert off.tracker.p.rest_coast_s == 0.0


def test_a_resting_balls_uncertainty_stays_usable_without_any_special_case():
    """A resting ball needs no sigma special-case, and the first draft that
    gave it a smaller velocity prior was a no-op dressed as a fix.

    `sigma` already switches from the generic `vel_prior` to the track's OWN
    measured velocity scatter once the fix is older than `vel_sig_after_s`
    (1 s), and `at_rest` requires the two hits that produce that scatter — so
    every resting ball is already on the good branch. This pins that, so
    nobody re-adds the special case."""
    t = Track(id=1, cls="ball", bearing=0.0, elevation=0.0, width=0.05, range=0.5,
              conf=0.9, born_t=0.0, last_t=0.0, xy=(0.5, 0.0), xy_t=0.0,
              vel=(0.0, 0.0), vel_hits=2, sig_meas=0.03, vel_sig=0.004)
    assert t.at_rest(0.05)
    old_fix = t.sigma(4.0, 0.06, 1.0)
    assert old_fix == t.sigma(4.0, 0.05, 1.0), "the prior is not consulted past vel_sig_after_s"
    assert old_fix < 0.06, "a 4 s old fix of a still ball is still a good fix"
    # …and inside the first second the generic prior still applies, as it must:
    # that is where a ball may have been rolling and not yet measured.
    assert t.sigma(0.5, 0.06, 1.0) > t.sigma(0.5, 0.01, 1.0)
    assert t.sigma(0.0, 0.06, 1.0) == pytest.approx(0.03)


def test_predict_returns_the_remembered_place_for_a_still_ball_however_old():
    """The whole point: an old fix of a motionless ball is still a good fix."""
    t = Track(id=1, cls="ball", bearing=0.0, elevation=0.0, width=0.05, range=0.5,
              conf=0.9, born_t=0.0, last_t=0.0, xy=(0.5, 0.25), xy_t=0.0,
              vel=(0.0, 0.0), vel_hits=2)
    assert t.predict(9.0, 0.3) == (0.5, 0.25)
    assert math.dist(t.predict(0.0, 0.3), (0.5, 0.25)) == 0.0
