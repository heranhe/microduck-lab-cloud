"""The brain's freshness gate, read against the SENSOR'S OWN CADENCE.

Roadmap 12av follow-up (1) measured the shipped kick pair at the robot's
documented 2 Hz and found 41 pp of whiff. It also found the reason the number
could not be quoted as a fact about the hardware: `Chase.DET_MAX_AGE` is a
constant **0.4 s**, and one period at 2 Hz is **0.5 s**. The brain therefore
calls its own newest detection stale on a sixth of its ticks before anything
has been missed, and `scripts/probe_ball_loss.py` — whose loss EVENT rule used
the same constant — opened a loss 0.4 s after every frame and closed it at the
next, manufacturing ~5 events a second per duck and reporting the duck's own
blinding as "median loss better on 12/12 seeds".

Two locks here, and they are the same lock in two places:

* **the gate must be expressible in periods** (`ChaseParams.det_max_periods`),
  and the default must change NOTHING at any rate the lab has rows for — 10 Hz
  and 5 Hz, and 2 Hz too, because `runs/detrate/gym-ship2-*.jsonl` are on disk
  and must still reproduce;
* **a loss event is silence longer than the camera's own cadence**, so a ball
  present in every single frame is never a loss, at any rate.
"""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain.controllers import Chase, ChaseParams, det_gate
from microduck_local.brain.runtime import Senses
from microduck_local.sensors import (
    Detection,
    DetectionFrame,
    Detector,
    DetectorNoise,
    DetectorSpec,
    Target,
)
from microduck_local.world import Ball, Duck, Scenario, compose
from microduck_local.world.compose import DuckAddress, spawn_duck

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")

RATES = (10.0, 5.0, 2.0)


def brain(rate_hz: float, knobs: str = "", monkeypatch=None) -> Chase:
    """A `Chase` constructed under a camera of this rate — the CONSTRUCTED
    brain, which is the only thing an arm may be asserted on (playbook rule 0:
    `from_env` alone has been wrong here before)."""
    monkeypatch.setenv("MICRODUCK_CAMERA", f"rate_hz={rate_hz}")
    monkeypatch.setenv("MICRODUCK_CHASE", knobs)
    return Chase()


def test_det_gate_is_a_floor_and_never_shortens() -> None:
    assert det_gate(0.4, 0.0, 0.5) == 0.4          # off: the bare constant
    assert det_gate(0.4, 1.0, 0.1) == 0.4          # 10 Hz: one period is well inside it
    assert det_gate(0.4, 4.0, 0.1) == 0.4          # …and so are four
    assert det_gate(0.4, 1.0, 0.5) == 0.5          # 2 Hz: the period is LONGER than the gate
    assert det_gate(0.4, 1.5, 0.5) == 0.75
    # A floor, so no setting can make the brain call a fresh frame stale.
    for periods in (0.0, 0.5, 1.0, 1.5, 3.0):
        for period in (0.04, 0.1, 0.2, 0.5):
            assert det_gate(0.4, periods, period) >= 0.4


def test_the_shipped_default_is_byte_identical_at_every_rate(monkeypatch) -> None:
    """Every row on disk — 10 Hz, 5 Hz and 12av's 2 Hz blocks — must still
    reproduce. The knob ships OFF, so the gate is the class constant at every
    rate, and the class constant itself has not moved."""
    assert Chase.DET_MAX_AGE == 0.4
    assert ChaseParams().det_max_periods == 0.0
    for rate in RATES:
        b = brain(rate, "", monkeypatch)
        assert b.det_period == pytest.approx(1.0 / rate)
        assert b.DET_MAX_AGE == Chase.DET_MAX_AGE


def test_the_knob_lengthens_the_gate_only_where_the_period_is_long(monkeypatch) -> None:
    """And at 10 Hz it is a structural no-op — which is exactly why the 10 Hz
    arm is the control: it MUST reproduce the baseline episode for episode
    (`kick_gym.is_identical` reads BROKEN there, and that reading is the pass).
    Check a knob's reachable set before spending a battery on it."""
    for periods, want2 in ((1.0, 0.5), (1.5, 0.75)):
        knobs = f"det_max_periods={periods}"
        assert brain(10.0, knobs, monkeypatch).DET_MAX_AGE == 0.4      # no-op: the control arm
        assert brain(5.0, knobs, monkeypatch).DET_MAX_AGE == 0.4       # no-op
        assert brain(2.0, knobs, monkeypatch).DET_MAX_AGE == want2     # the only rate it reaches


def test_the_gate_the_brain_constructs_is_the_one_its_step_reads(monkeypatch) -> None:
    """Not an arithmetic identity: the same 0.45 s old detection — younger than
    a 2 Hz period, older than 0.4 s — must reach the tracker under the knob and
    be refused without it."""
    frame = DetectionFrame(t=0.0, detections=[Detection("ball", "", 0.1, -0.3, 0.2, 0.5, 0.9)])
    seen = {}
    for knobs in ("", "det_max_periods=1.0"):
        b = brain(2.0, knobs, monkeypatch)
        s = Senses(t=0.45, det=frame, det_age=0.45, tof=None, tof_age=None,
                   speed=0.0, odom=(0.0, 0.0, 0.0), skill=None, bumped=False)
        b.step(s)
        det = b.inputs()["det"]
        seen[knobs] = (det["max"], det["stale"], len(b.tracker.tracks))
    assert seen[""] == (0.4, True, 0)                       # shipped: stale, nothing tracked
    assert seen["det_max_periods=1.0"] == (0.5, False, 1)   # the knob: the frame is used


# --- the loss probe's event rule -------------------------------------------

def _static_world():
    """One duck with a ball parked in front of it: the ball is in EVERY frame
    the detector takes, and nothing ever moves."""
    sc = Scenario(name="detfresh", floor=(8, 8), walls=[], balls=[Ball((0.5, 0.0))],
                  ducks=[Duck("a", (0.0, 0.0, 0.0), None, None)])
    m = compose(sc)
    d = mujoco.MjData(m)
    spawn_duck(m, d, DuckAddress.resolve(m, "a"), 0.0, 0.0, 0.0)
    mujoco.mj_forward(m, d)
    tgt = Target("ball0", "ball", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball0"), 0.035)
    return m, d, tgt


def _events(rate_hz: float, gate: float, seconds: float = 120.0, dt: float = 0.02) -> int:
    """The probe's own rule, over the detector's REAL arrival schedule (rate,
    datasheet latency and its jitter), with the ball in every frame."""
    m, d, tgt = _static_world()
    det = Detector(m, site="a/head_camera", targets=[tgt], seed=3,
                   spec=DetectorSpec(rate_hz=rate_hz), noise=DetectorNoise.datasheet())
    # "Visible in EVERY frame", made exact: the datasheet's own size gate drops
    # a 3.5 cm ball at 0.5 m about 2 % of the time, and those are REAL misses
    # that a loss rule is entitled to see. This asks the narrower question the
    # fix is about — nothing missed, only the cadence — the way the probe's own
    # `_hook` wraps `capture`.
    _orig = det.capture

    def capture(data, t):
        fr = _orig(data, t)
        if not any(x.cls == "ball" for x in fr.detections):
            fr.detections.append(Detection("ball", "ball0", 0.0, -0.3, 0.2, 0.5, 0.9))
        return fr
    det.capture = capture
    last_ball_t, last_frame_t, opens, open_now = 0.0, -1.0, 0, False
    for k in range(int(seconds / dt)):
        t = k * dt
        det.sample(d, t)
        fr = det.last
        if fr is not None and fr.t > last_frame_t:
            last_frame_t = fr.t
            if any(x.cls == "ball" for x in fr.detections):
                last_ball_t, open_now = t, False
        if t - last_ball_t > gate and not open_now:       # probe_ball_loss.run's rule
            opens, open_now = opens + 1, True
    return opens


def test_a_ball_in_every_frame_is_never_a_loss_at_2hz() -> None:
    """The fix, and the bug it replaces, on the same schedule. The old flat
    0.4 s rule fires after EVERY frame at 2 Hz — that is the ~5 events a second
    per duck that made 12av follow-up (1)'s "median loss 1.02 -> 0.10 s, better
    on 12/12 seeds" an artifact of the instrument."""
    import probe_ball_loss as P

    period = 0.5
    gate = det_gate(Chase.DET_MAX_AGE, P.LOSS_PERIODS, period)
    assert gate == 0.75
    assert _events(2.0, gate) == 0                          # the fixed rule: not one loss
    manufactured = _events(2.0, Chase.DET_MAX_AGE)          # the old rule, same frames
    assert manufactured > 200                               # ~1 per frame over 120 s at 2 Hz


def test_at_10hz_and_5hz_the_gate_is_the_shipped_constant() -> None:
    """So no number any 10 Hz probe run has ever produced moves — 12af / 12ar /
    12ae / 12as included."""
    import probe_ball_loss as P

    assert det_gate(Chase.DET_MAX_AGE, P.LOSS_PERIODS, 0.1) == Chase.DET_MAX_AGE
    assert det_gate(Chase.DET_MAX_AGE, P.LOSS_PERIODS, 0.2) == Chase.DET_MAX_AGE
    assert "Chase.DET_MAX_AGE" not in P.run.__code__.co_names   # the rule asks the brain, not the class


@pytest.mark.slow
def test_the_loss_event_count_on_a_seeded_60s_run_is_unchanged_at_10hz(monkeypatch) -> None:
    """The arithmetic above, spent on a real 2v2-shaped run: same events, same
    start times, same viewFrac under the old rule and the new one."""
    import probe_ball_loss as P

    monkeypatch.delenv("MICRODUCK_CAMERA", raising=False)
    old = P.run(1, 60.0, 1, loss_periods=0.0)               # gate = the flat 0.4 s
    new = P.run(1, 60.0, 1)                                 # gate = max(0.4, 1.5 * 0.1)
    assert new["lossGate"] == old["lossGate"] == Chase.DET_MAX_AGE
    assert new["detHz"] == 10.0
    assert len(new["losses"]) == len(old["losses"]) > 0
    assert [e["t0"] for e in new["losses"]] == [e["t0"] for e in old["losses"]]
    assert new["viewFrac"] == old["viewFrac"]


def test_the_probe_row_records_the_rule_it_counted_under() -> None:
    """A rate arm's rows must carry the rule they were counted under, or the
    next reader pools two instruments (this item's whole lesson)."""
    import probe_ball_loss as P

    assert P.LOSS_PERIODS >= 1.0                            # at least one period of cadence
    # One period of silence is due; the rest is the datasheet's own latency
    # jitter (0.026 s + |N(0, 0.02)|) plus a 0.02 s tick.
    assert P.LOSS_PERIODS * 0.5 > 0.5 + 0.026 + 4 * 0.02 + 0.02
    consts = [c for c in P.run.__code__.co_consts if isinstance(c, str)]
    for k in ("detHz", "lossGate", "lossPeriods"):
        assert k in consts


def test_no_other_brain_moved() -> None:
    """`Follow` (people, at the lab rate) keeps the flat constant: this item
    measured the chase brain and nothing else, and a constant that moves
    without a measurement is how a camera number got wrong in eight places."""
    from microduck_local.brain.controllers import Follow

    assert Follow.DET_MAX_AGE == 0.4
    assert not hasattr(Follow, "det_period")
    assert math.isclose(Chase.TOF_MAX_AGE, 0.25) and np.isfinite(Chase.TOF_MAX_AGE)
