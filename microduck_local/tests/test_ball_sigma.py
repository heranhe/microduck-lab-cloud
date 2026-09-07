"""Locks for the ball's uncertainty (roadmap Track 4 §6 C.1): a track's
position sigma is the detector's declared noise at its range, settles
with the smoothing, grows with the age of the hit by the velocity's own
scatter (or a prior when there is none), reaches the chase brain as
`predicted_sigma`, and a duck's tracker is built for its detector preset."""

import math

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.runtime import Senses
from microduck_local.brain.team import brain_kwargs
from microduck_local.brain.tracker import Track, Tracker, TrackerParams
from microduck_local.sensors.detector import Detection, DetectionFrame


def _frame(t, bearing, rng):
    return DetectionFrame(t=t, detections=[Detection("ball", "", bearing, -0.3, 2 * math.atan(0.03 / rng), rng, 0.9)])


def test_the_tracker_is_built_for_the_detectors_datasheet():
    ideal, sheet, hostile = (TrackerParams.for_detector(k) for k in ("ideal", "datasheet", "hostile"))
    assert ideal.meas_bearing_sigma == 0.0 and ideal.meas_range_frac == 0.0
    assert abs(sheet.meas_bearing_sigma - math.radians(1.0)) < 1e-9 and sheet.meas_range_frac == 0.10
    assert hostile.meas_range_frac > sheet.meas_range_frac
    assert TrackerParams().meas_range_frac == sheet.meas_range_frac    # the default IS the datasheet
    assert TrackerParams.for_detector(None).meas_range_frac == 0.0     # no detector: only the floor
    assert TrackerParams.for_detector("hostile").ignore == ("post",)   # nothing else moves


def test_a_hit_carries_the_datasheets_error_at_its_range_and_settles_with_the_smoothing():
    tk = Tracker(TrackerParams.for_detector("datasheet"))
    tr = tk.update(_frame(0.0, 0.0, 0.6), 0.0, 0.0, (0.0, 0.0))[0]
    assert abs(tr.sig_meas - math.hypot(math.radians(1.0) * 0.6, 0.06)) < 1e-6   # 6.1 cm, range-dominated
    for k in range(1, 4):
        tk.update(_frame(0.1 * k, 0.0, 0.6), 0.1 * k, 0.0, (0.0, 0.0))
    settled = math.sqrt(0.6 / 1.4)
    assert abs(tr.sig_meas - math.hypot(math.radians(1.0) * 0.6, 0.06) * settled) < 1e-6
    # A perfect detector still carries the floor.
    tk0 = Tracker(TrackerParams.for_detector("ideal"))
    tr0 = tk0.update(_frame(0.0, 0.0, 0.6), 0.0, 0.0, (0.0, 0.0))[0]
    assert tr0.sig_meas == TrackerParams().meas_floor


def test_sigma_grows_with_the_age_of_the_hit_by_the_velocitys_scatter_or_a_prior():
    tr = Track(id=1, cls="ball", bearing=0.0, elevation=0.0, width=0.1, range=0.6, conf=0.9,
               born_t=0.0, last_t=0.0, xy=(0.6, 0.0), xy_t=0.0, sig_meas=0.05)
    assert tr.sigma(0.0) == 0.05
    assert abs(tr.sigma(1.0, vel_prior=0.15) - math.hypot(0.05, 0.15)) < 1e-9     # no velocity: the prior
    tr.vel, tr.vel_hits, tr.vel_sig = (0.5, 0.0), 3, 0.04
    assert abs(tr.sigma(1.0) - math.hypot(0.05, 0.04)) < 1e-9                     # a measured scatter instead
    # A rolling ball seen three times: the velocity's scatter is measured,
    # and forgotten with the velocity when the hits are too far apart.
    tk = Tracker(TrackerParams.for_detector("datasheet"))
    for k, (b, r) in enumerate(((0.0, 0.6), (0.0, 0.52), (0.0, 0.45))):
        t = 0.2 * k
        (tr2,) = tk.update(_frame(t, b, r), t, 0.0, (0.0, 0.0))
    assert tr2.vel_hits == 2 and tr2.vel_sig >= 0.0 and tr2.sigma(0.4) >= tr2.sig_meas
    (tr2,) = tk.update(_frame(3.0, 0.0, 0.45), 3.0, 0.0, (0.0, 0.0))
    assert tr2.vel_hits == 0 and tr2.vel_sig == 0.0


def test_the_chase_brain_reports_the_sigma_of_its_estimate_and_takes_its_detector_from_the_roster():
    b = Chase(ChaseParams(), goal=(1.5, 0.0), det_noise="hostile")
    assert b.tracker.p.meas_range_frac == 0.3
    b.step(Senses(t=1.0, det=_frame(1.0, 0.0, 0.6), det_age=0.0, odom=(0.0, 0.0, 0.0), speed=0.0))
    assert b.predicted is not None and b.predicted_sigma is not None
    assert abs(b.predicted_sigma - math.hypot(math.radians(3.0) * 0.6, 0.18)) < 1e-6
    b.step(Senses(t=1.5, det=_frame(1.0, 0.0, 0.6), det_age=0.5, odom=(0.0, 0.0, 0.0), speed=0.0))
    assert b.predicted_sigma > math.hypot(math.radians(3.0) * 0.6, 0.18)         # half a second older
    from microduck_local.world import World, make_pitch
    sc = make_pitch()
    w = World(sc, seed=1)
    kw = brain_kwargs(sc.ducks[0], w, {})
    assert kw["det_noise"] == "datasheet"
