"""Locks for the goal-post particle filter (brain/localize.py, roadmap
Track 4 §6 C.2): tracks the odometry when nothing is seen, corrects a
drifting odometry from post sightings, weighs a frame once, re-seeds on a
respawn, and derives the posts from what the chase brain is built with."""

import math

import numpy as np

from microduck_local.brain.localize import Localizer, LocalizerParams, pitch_posts
from microduck_local.sensors.detector import Detection, DetectionFrame

POSTS = pitch_posts((1.5, 1.25), 0.7)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _frame(t, truth, posts=POSTS, cam_yaw=0.0, max_range=3.0, half_fov=math.radians(31)):
    """What a perfect detector on a duck at `truth` reports about the posts
    in front of it (camera frame, so the head's yaw is subtracted)."""
    x, y, yaw = truth
    dets = []
    for px, py in posts:
        rng = math.hypot(px - x, py - y)
        b = _wrap(math.atan2(py - y, px - x) - yaw - cam_yaw)
        if rng <= max_range and abs(b) <= half_fov:
            dets.append(Detection("post", "", b, 0.0, 2 * math.atan(0.05 / rng), rng, 1.0))
    return DetectionFrame(t=t, detections=dets, cam_yaw=cam_yaw)


def test_pitch_posts_sit_on_the_mouth_line_the_world_scores():
    assert set(POSTS) == {(1.5, 0.35), (1.5, -0.35), (-1.5, 0.35), (-1.5, -0.35)}


def test_with_nothing_seen_the_estimate_is_the_odometry():
    loc = Localizer(POSTS, seed=1)
    for k in range(60):
        odom = (0.01 * k, 0.0, 0.0)
        est = loc.update(odom, None)
    assert abs(est[0] - odom[0]) < 0.03 and abs(est[1]) < 0.03 and abs(est[2]) < 0.03
    assert loc.hits == 0 and loc.resets == 1


def test_post_sightings_correct_a_drifting_odometry():
    """A duck walks a loop while its odometry integrates a yaw bias and a
    distance scale error (`OdomNoise`'s two per-run terms). Dead reckoning
    ends up far off; the filter, fed the same odometry plus what a camera
    would see of the posts, ends up close."""
    loc = Localizer(POSTS, LocalizerParams(n=300), seed=3)
    truth = [0.0, -0.5, 0.0]
    odom = [0.0, -0.5, 0.0]
    scale, bias = 1.06, math.radians(1.0) * 0.02          # 6% scale, 1 deg/s of gyro bias at 50 Hz
    t = 0.0
    est = None
    turns = 0
    for k in range(1200):                                   # 24 s at 50 Hz
        v, wz = 0.3, (0.6 if (k // 200) % 2 else -0.2)
        t += 0.02
        turn = wz * 0.02
        # At the boards the robot turns round - and its gyro sees that turn
        # exactly as it sees any other, so the odometry gets it too (a
        # first draft bounced only the truth and measured its own bug).
        if abs(truth[0] + v * 0.02 * math.cos(truth[2])) > 1.3 or abs(truth[1] + v * 0.02 * math.sin(truth[2])) > 1.05:
            turn += math.pi
            turns += 1
        truth[2] = _wrap(truth[2] + turn)
        truth[0] += v * 0.02 * math.cos(truth[2])
        truth[1] += v * 0.02 * math.sin(truth[2])
        odom[2] = _wrap(odom[2] + turn + bias)
        odom[0] += scale * v * 0.02 * math.cos(odom[2])
        odom[1] += scale * v * 0.02 * math.sin(odom[2])
        frame = _frame(t, truth) if k % 5 == 0 else None    # a 10 Hz detector
        est = loc.update(tuple(odom), frame)
    raw_err = math.hypot(odom[0] - truth[0], odom[1] - truth[1])
    est_err = math.hypot(est[0] - truth[0], est[1] - truth[1])
    raw_yaw = abs(_wrap(odom[2] - truth[2]))
    est_yaw = abs(_wrap(est[2] - truth[2]))
    # Measured on this exact walk: raw 0.649 m / 24.0 deg, estimate 0.025 m
    # / 3.9 deg over 209 sightings and 4 turns at the boards. In play
    # (scripts/probe_odom_goal.py) the filter takes `datasheet` drift from
    # a 0.215 m median error to 0.072 m and `hostile` from 0.706 to 0.089.
    assert turns >= 2 and raw_err > 0.25, (turns, raw_err)  # dead reckoning really is lost by now
    assert est_err < 0.10 and est_err < 0.25 * raw_err, (est_err, raw_err)
    assert est_yaw < 0.10 and est_yaw < 0.25 * raw_yaw, (est_yaw, raw_yaw)
    assert loc.hits > 100


def test_a_frame_is_weighed_once_however_often_it_is_presented():
    loc = Localizer(POSTS, seed=5)
    odom = (0.0, 0.0, 0.0)
    loc.update(odom, None)
    f = _frame(0.1, odom)
    loc.update(odom, f)
    h = loc.hits
    for _ in range(10):                                     # the same frame, ten more ticks
        loc.update(odom, f)
    assert loc.hits == h


def test_a_respawn_reseeds_instead_of_scattering():
    loc = Localizer(POSTS, seed=7)
    loc.update((0.0, 0.0, 0.0), None)
    est = loc.update((1.2, -0.8, 2.0), None)               # a teleport: the arena reset the odometry
    assert loc.resets == 2
    assert abs(est[0] - 1.2) < 0.05 and abs(est[1] + 0.8) < 0.05 and abs(_wrap(est[2] - 2.0)) < 0.05


def test_the_cloud_reports_its_own_spread():
    loc = Localizer(POSTS, seed=9)
    loc.update((0.0, 0.0, 0.0), None)
    s0 = loc.spread()
    for k in range(200):                                    # a long blind walk: the cloud opens
        loc.update((0.005 * k, 0.0, 0.0), None)
    assert loc.spread() > s0
    assert np.isfinite(loc.spread())
