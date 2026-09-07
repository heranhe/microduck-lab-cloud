"""Locks for the shared ball (roadmap Track 4 §6 C.3): the board's ball is
the freshest sighting by default and, with `Team.fuse`, the inverse-
variance mean of every live sighting weighed by its sender's sigma grown
by its age; the chase brain sends its track's sigma with the claim."""

import math

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.runtime import Senses
from microduck_local.brain.team import Team, brain_kwargs
from microduck_local.sensors.detector import Detection, DetectionFrame


def test_the_board_ball_is_the_freshest_sighting_unless_it_fuses():
    tm = Team("cream")
    tm.claim("d0", 10.0, 0.5, (0.50, 0.00), ball_sigma=0.02)     # sure, a little older
    tm.claim("d1", 10.3, 1.5, (0.80, 0.30), ball_sigma=0.20)     # fresh, and loose
    assert tm.ball(10.3) == (0.80, 0.30)                          # freshest wins by default
    tm.fuse = True
    bx, by = tm.ball(10.3)
    assert 0.50 < bx < 0.55 and 0.0 < by < 0.05, (bx, by)         # near the sure one
    # Surer than either alone: the sure claim has grown to 0.049 in 0.3 s
    # (0.02 of its own, 0.045 of `vel_prior` x age), the loose one is 0.20.
    assert tm.ball_sigma(10.3) < math.hypot(0.02, 0.15 * 0.3) < 0.20
    # As the sure claim ages, its weight decays toward the fresh one.
    bx2, by2 = tm.ball(11.5)
    assert bx2 > bx and by2 > by
    # A sender that said nothing about its error gets the default sigma.
    tm.claim("d2", 11.5, 1.0, (0.60, 0.10))
    assert math.isnan(tm.claims["d2"].ball_sigma)
    assert tm.ball(11.5) is not None
    # A wipe forgets everything, including the sigmas.
    tm.reset()
    assert tm.ball(11.5) is None and tm.ball_sigma(11.5) is None


def test_a_single_sighting_is_returned_as_it_is_and_stale_ones_drop_off():
    tm = Team("cream", fuse=True)
    tm.claim("d0", 10.0, 0.5, (0.50, 0.00), ball_sigma=0.05)
    assert tm.ball(10.0) == (0.50, 0.00)
    assert tm.ball(10.0 + 3 * tm.stale_s + 0.1) is None


def test_the_roster_sets_the_boards_fusion_from_the_brains_knob(monkeypatch):
    from microduck_local.world import World, make_pitch
    sc = make_pitch(per_side=2)
    w = World(sc, seed=1)
    assert ChaseParams().fuse_ball is False                       # ships off until the ledger is read
    teams = {}
    for d in sc.ducks:
        brain_kwargs(d, w, teams)
    assert all(tm.fuse is False for tm in teams.values())
    monkeypatch.setenv("MICRODUCK_CHASE", "fuse_ball=1")
    teams = {}
    for d in sc.ducks:
        brain_kwargs(d, w, teams)
    assert all(tm.fuse is True for tm in teams.values())


def test_the_chase_brain_sends_its_tracks_sigma_with_the_claim():
    tm = Team("cream", half_x=1.5)
    b = Chase(ChaseParams(), goal=(1.5, 0.0), team=tm, duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7)
    det = DetectionFrame(t=1.0, detections=[Detection("ball", "", 0.0, -0.3, 0.1, 0.6, 0.9)])
    b.step(Senses(t=1.0, det=det, det_age=0.0, odom=(0.0, 0.0, 0.0), speed=0.0))
    c = tm.claims["d0"]
    assert c.ball is not None and math.isfinite(c.ball_sigma)
    assert abs(c.ball_sigma - b.predicted_sigma) < 1e-9
    b.step(Senses(t=3.0, det=None, det_age=None, odom=(0.0, 0.0, 0.0), speed=0.0))   # lost it
    assert tm.claims["d0"].ball is None and math.isnan(tm.claims["d0"].ball_sigma)


def test_the_fusion_window_drops_claims_much_older_than_the_freshest():
    tm = Team("cream", fuse=True, fuse_window=0.5)
    tm.claim("d0", 10.0, 0.5, (0.50, 0.00), ball_sigma=0.02)     # sure, but 1 s older than the freshest
    tm.claim("d1", 11.0, 1.5, (0.80, 0.30), ball_sigma=0.20)
    bx, by = tm.ball(11.0)
    assert abs(bx - 0.80) < 1e-9 and abs(by - 0.30) < 1e-9         # outside the window: the freshest alone
    assert abs(tm.ball_sigma(11.0) - 0.20) < 1e-9
    tm.fuse_window = 3.0
    bx, _ = tm.ball(11.0)
    assert bx < 0.80                                              # inside it: fused as before
