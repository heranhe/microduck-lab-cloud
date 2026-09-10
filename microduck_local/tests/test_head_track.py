"""The head in two axes and in time (`track_pitch`, `look_hold_s`,
`head_lead_s`; roadmap Track 4 item 12ae). The pitch and the hold ship ON, the lead OFF; the first
test is that the pre-12ae head is bit-identical with every knob at its OFF value,
and that the defaults are the measured ones; the rest drive one brain with synthetic frames and read the
four-slot head command back."""

from __future__ import annotations

import math

import numpy as np
import pytest

from microduck_local.brain import REGISTRY
from microduck_local.brain.controllers import ChaseParams
from microduck_local.brain.gait import TURN_KICK
from microduck_local.brain.runtime import Senses
from microduck_local.sensors.detector import Detection, DetectionFrame

CAM_Z = 0.24
SHIPPED = (True, 0.45, 2.5, 0.0)           # track_pitch, track_pitch_max, look_hold_s, head_lead_s: what 12ae measured


def _frame(t: float, bearing: float, rng: float, cam_pitch: float = 0.117) -> DetectionFrame:
    """A ball at `bearing` (body frame; the head is level so camera = body)
    and slant range `rng`, with the elevation the geometry gives."""
    elev = -math.asin((CAM_Z - 0.035) / rng) + cam_pitch
    width = 2 * math.atan(0.035 / rng)
    return DetectionFrame(t, [Detection("ball", "ball0", bearing, elev, width, rng, 0.9)],
                          cam_z=CAM_Z, cam_pitch=cam_pitch)


def _drive(brain, frames, until: float, odom=(0.0, 0.0, 0.0), dt: float = 0.02):
    """Step the brain at 50 Hz; a frame arrives at its own time and stays the
    latest until the next. Returns every Intent, in order."""
    out = []
    fr_i, last = 0, None
    t = 0.0
    while t <= until + 1e-9:
        while fr_i < len(frames) and frames[fr_i].t <= t + 1e-9:
            last = frames[fr_i]
            fr_i += 1
        out.append(brain.step(Senses(t=t, det=last, det_age=None if last is None else t - last.t,
                                     speed=0.3, odom=odom)))
        t += dt
    return out


OFF = dict(track_pitch=False, track_pitch_turn=0.0, look_hold_s=0.0, head_lead_s=0.0)   # the pre-12ae head


def test_the_old_head_is_untouched_by_the_new_knobs_when_they_are_off():
    """Whatever ships, the knobs at their OFF values are the head as it was
    before them, to the bit - and the shipped values are what the fresh block
    measured (roadmap 12ae)."""
    p = ChaseParams()
    assert (p.track_pitch, p.track_pitch_max, p.look_hold_s, p.head_lead_s) == SHIPPED
    a = REGISTRY.make("chase", p=ChaseParams(**OFF))
    b = REGISTRY.make("chase", p=ChaseParams(**{**OFF, "track_pitch_margin": 0.15}))
    frames = [_frame(0.1 * k, 0.6, 0.28) for k in range(5)]
    ia = _drive(a, frames, 3.0)
    ib = _drive(b, frames, 3.0)
    assert [i.head for i in ia] == [i.head for i in ib]
    # The shipped law as it stands: the yaw follows the ball, and in `chase`
    # the walking gaze already centres a ball inside `head_range` - while a
    # sighting is FRESH (the last frame is at 0.4 s; fresh ends 0.4 s later).
    seen = [i for k, i in enumerate(ia) if i.note == "chase" and k * 0.02 < 0.75]
    assert seen and all(i.head[2] > 0.3 and i.head[1] > 0.3 for i in seen)


def test_track_pitch_takes_over_where_the_gaze_declines():
    """A ball 0.28 m out, 40 deg off: below a level walking camera's frame
    (36.7 deg). With the chase gaze declining it (`head_range` under the
    ball's range - in play it is `gaze_bearing_max`, `avoid`, `retreat`, the
    line-up's second stage), the shipped head yaws and never pitches; the
    tracking pitch asks for the smallest command that keeps it
    `track_pitch_margin` inside the frame, in the same tick."""
    frames = [_frame(0.1 * k, 0.55, 0.28) for k in range(4)]
    ship = _drive(REGISTRY.make("chase", p=ChaseParams(head_range=0.2, **OFF)), frames, 0.6)
    walking = [i for i in ship if i.twist[0] > TURN_KICK and i.note == "chase"]
    assert walking and all(i.head[1] == 0.0 and i.head[2] > 0.3 for i in walking), [i.note for i in ship]
    b = REGISTRY.make("chase", p=ChaseParams(head_range=0.2, track_pitch=True))
    intents = _drive(b, frames, 0.6)
    walking = [i for i in intents if i.twist[0] > TURN_KICK and i.note == "chase"]
    assert walking, [i.note for i in intents]
    last = walking[-1]
    assert last.head[2] > 0.3                                     # yaw toward the ball
    assert 0.0 < last.head[1] <= b.p.track_pitch_max                # a pitch, capped
    # The geometry: the commanded axis reaches the ball's depression minus (half_v - margin).
    p = b.p
    dep = math.asin((CAM_Z - 0.035) / 0.28)
    want = dep - (math.radians(30) - p.track_pitch_margin)
    assert last.head[1] == pytest.approx(min((want - p.cam_level_walk) / p.head_gain, p.track_pitch_max), abs=1e-6)
    # Where the gaze DOES fire it keeps the head: the two brains agree to the bit.
    a = _drive(REGISTRY.make("chase", p=ChaseParams(**OFF)), frames, 0.6)
    c = _drive(REGISTRY.make("chase", p=ChaseParams(**{**OFF, "track_pitch": True})), frames, 0.6)
    assert [i.head for i in a] == [i.head for i in c]
    # A far ball needs no pitch at all.
    far = _drive(REGISTRY.make("chase", p=ChaseParams(head_range=0.2, track_pitch=True)),
                 [_frame(0.1 * k, 0.55, 0.9) for k in range(4)], 0.6)
    assert all(i.head[1] == 0.0 for i in far if i.note == "chase")


def test_track_pitch_defers_to_a_turn_in_place_unless_allowed():
    """A ball 130 deg round: the brain turns in place, and the walker cannot
    turn with its head down, so the pitch is 0 - unless `track_pitch_turn`
    lends it the benched-free 0.10."""
    def turning(p):
        b = REGISTRY.make("chase", p=p)
        intents = _drive(b, [_frame(0.1 * k, 2.3, 0.30) for k in range(4)], 0.6)
        turns = [i for i in intents if i.note == "turn" and i.twist[0] <= TURN_KICK and i.twist[2] != 0.0]
        assert turns, [i.note for i in intents]
        return turns[-1].head
    off = turning(ChaseParams(track_pitch=True, track_pitch_turn=0.0))
    assert off[1] == 0.0 and abs(off[2]) > 0.5
    on = turning(ChaseParams(track_pitch=True, track_pitch_turn=0.10))
    assert on[1] == pytest.approx(0.10) and abs(on[2]) > 0.5


def test_look_hold_keeps_the_head_on_the_coasting_track_past_predict_s():
    """Sightings stop at 0.3 s. Shipped, the head returns to centre once the
    track is older than `predict_s` (1 s); with `look_hold_s` 2.5 it stays on
    the track's bearing until the tracker forgets it at `coast_s`."""
    def head_yaw_at(p, t):
        b = REGISTRY.make("chase", p=p)
        intents = _drive(b, [_frame(0.1 * k, 0.8, 0.6) for k in range(4)], t)
        return intents[-1].head[2], intents[-1].note
    y_ship, _ = head_yaw_at(ChaseParams(look_hold_s=0.0), 1.6)
    y_hold, note = head_yaw_at(ChaseParams(look_hold_s=2.5), 1.6)
    assert y_ship == 0.0
    assert y_hold > 0.4, (y_hold, note)
    y_gone, _ = head_yaw_at(ChaseParams(look_hold_s=2.5), 3.2)     # past coast_s: nothing to hold on to
    assert y_gone == 0.0


def test_head_lead_aims_ahead_of_a_rolling_ball():
    """A ball crossing left to right at 1 m/s, 1 m out: with `head_lead_s`
    the yaw command is on the RIGHT of where the ball is (less positive
    bearing) - where it will be 140 ms later."""
    def yaw(p):
        b = REGISTRY.make("chase", p=p)
        frames = []
        for k in range(6):
            t = 0.1 * k
            x, y = 1.0, 0.5 - 1.0 * t
            frames.append(DetectionFrame(t, [Detection("ball", "ball0", math.atan2(y, x), -0.2, 0.07,
                                                       math.hypot(x, y), 0.9)], cam_z=CAM_Z, cam_pitch=0.117))
        return _drive(b, frames, 0.52)[-1].head[2]
    plain, lead = yaw(ChaseParams(head_lead_s=0.0)), yaw(ChaseParams(head_lead_s=0.14))
    assert lead < plain - 0.05, (plain, lead)


def test_the_knobs_read_off_the_environment():
    p = ChaseParams.from_env("track_pitch=1,track_pitch_max=0.45,look_hold_s=2.5,head_lead_s=0.14")
    assert p.track_pitch is True and p.track_pitch_max == 0.45 and p.look_hold_s == 2.5 and p.head_lead_s == 0.14
    assert np.isclose(ChaseParams().cam_level_walk, 0.117)
