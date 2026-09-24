"""Self-localisation on the pitch from the goal posts (roadmap Track 4 §6
C.2, and the fix for item 10's shared frame).

The brain's (x, y, yaw) is dead reckoning: the truth at spawn, drifting with
every step (`world/arena.OdomNoise`). Measured at the `datasheet` preset two
teammates' frames wander 0.456 m apart over a run, so "the ball is at (x, y)"
on the team board stops being a place the other duck can act on, and the
goal — "where it was at spawn" — moves with the drift. RoboCup teams close
this with a particle filter over field landmarks; this is that, with the
four goal posts as the landmarks (a `post` detection class the sim's
detector reports; on the robot, a coloured-post class in duck_detect).

The filter is the textbook one and deliberately small:

- state: N particles of (x, y, yaw) in the WORLD frame, which is the frame
  the brain's odometry is defined in (the truth at spawn);
- motion: the odometry's own delta between updates, rotated into each
  particle's heading, plus noise proportional to the distance and the turn
  (so a scale error or a gyro bias is something the cloud can spread over);
- measurement: for each post sighting, bearing (camera frame plus the
  head's yaw, exactly as the tracker does it) and the width-ranged
  distance, scored against the NEAREST-fitting known post per particle — a
  max-mixture association, which is what four identical posts allow;
- resample (systematic) when the effective sample size halves;
- estimate: the weighted mean, with the yaw averaged on the circle.

Two guards that matter in this sim: a detection frame is presented on
every 50 Hz tick until the next one arrives, so a frame is weighed ONCE
(by its timestamp); and a respawn teleports the duck and resets its
odometry, which shows up as an impossible one-tick jump and re-seeds the
cloud rather than scattering it. `pose` is the current estimate for probes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sensors.detector import DetectionFrame


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class LocalizerParams:
    n: int = 200
    init_xy_sigma: float = 0.02        # the spawn is known to about this
    init_yaw_sigma: float = 0.02
    move_xy_sigma: float = 0.05        # per metre of odometry delta (covers a scale error)
    move_yaw_sigma: float = 0.05       # per radian of odometry turn (covers a gyro bias)
    jitter_xy: float = 0.003           # per update: the cloud never fully collapses
    jitter_yaw: float = 0.003
    bearing_sigma: float = 0.08        # rad: the detector's bearing noise plus a pitched head's azimuth slop
    range_sigma_frac: float = 0.25     # of the range: width-ranging a 5 cm post is coarse
    max_range: float = 3.0             # sightings beyond this carry no usable range
    resample_ess: float = 0.5          # resample when ESS < this fraction of n
    jump_m: float = 0.5                # an odometry jump this big in one tick is a respawn


class Localizer:
    def __init__(self, posts, params: LocalizerParams | None = None, seed: int = 0):
        self.posts = np.asarray(posts, float).reshape(-1, 2)
        self.p = params or LocalizerParams()
        self.rng = np.random.default_rng(seed)
        self.X: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self._prev_odom: tuple[float, float, float] | None = None
        self._last_frame_t: float | None = None
        self.pose: tuple[float, float, float] | None = None
        self.hits = 0          # post sightings weighed
        self.resets = 0        # re-seeds (the first one plus every respawn)

    # -- lifecycle ---------------------------------------------------------
    def reset(self, odom) -> None:
        n, p = self.p.n, self.p
        self.X = np.column_stack([
            odom[0] + self.rng.normal(0.0, p.init_xy_sigma, n),
            odom[1] + self.rng.normal(0.0, p.init_xy_sigma, n),
            odom[2] + self.rng.normal(0.0, p.init_yaw_sigma, n)])
        self.w = np.full(n, 1.0 / n)
        self._prev_odom = (float(odom[0]), float(odom[1]), float(odom[2]))
        self.pose = self._prev_odom
        self.resets += 1

    def update(self, odom, frame: DetectionFrame | None) -> tuple[float, float, float]:
        """Fold this tick's odometry (x, y, yaw) and the detector's current
        frame (or None); return the corrected (x, y, yaw)."""
        if self.X is None or self._prev_odom is None:
            self.reset(odom)
            assert self.pose is not None
            return self.pose
        ox, oy, oyaw = self._prev_odom
        dx, dy = float(odom[0]) - ox, float(odom[1]) - oy
        if math.hypot(dx, dy) > self.p.jump_m:          # a respawn: the odometry itself was re-seeded
            self.reset(odom)
            assert self.pose is not None
            return self.pose
        self._move(dx, dy, _wrap(float(odom[2]) - oyaw), oyaw)
        self._prev_odom = (float(odom[0]), float(odom[1]), float(odom[2]))
        if frame is not None and frame.t != self._last_frame_t:
            self._last_frame_t = frame.t
            cam_yaw = float(getattr(frame, "cam_yaw", 0.0))
            for d in frame.detections:
                if d.cls != "post" or d.range_est > self.p.max_range:
                    continue
                self._weigh(_wrap(d.bearing + cam_yaw), float(d.range_est))
                self.hits += 1
        self.pose = self._estimate()
        return self.pose

    # -- the filter --------------------------------------------------------
    def _move(self, dx: float, dy: float, dyaw: float, prev_yaw: float) -> None:
        assert self.X is not None
        p, n = self.p, len(self.X)
        # The odometry's world delta, expressed in the odometry's OWN previous
        # heading: what the legs reported moving, forward and leftward.
        c, s = math.cos(prev_yaw), math.sin(prev_yaw)
        fwd, left = dx * c + dy * s, -dx * s + dy * c
        dist = math.hypot(fwd, left)
        nf = fwd + self.rng.normal(0.0, p.move_xy_sigma * dist + p.jitter_xy, n)
        nl = left + self.rng.normal(0.0, p.move_xy_sigma * dist + p.jitter_xy, n)
        ny = dyaw + self.rng.normal(0.0, p.move_yaw_sigma * abs(dyaw) + p.jitter_yaw, n)
        yaw = self.X[:, 2]
        self.X[:, 0] += nf * np.cos(yaw) - nl * np.sin(yaw)
        self.X[:, 1] += nf * np.sin(yaw) + nl * np.cos(yaw)
        self.X[:, 2] = np.arctan2(np.sin(yaw + ny), np.cos(yaw + ny))

    def _weigh(self, bearing: float, rng: float) -> None:
        assert self.X is not None and self.w is not None
        p = self.p
        # Predicted bearing and range to every post from every particle: (n, k).
        dxp = self.posts[None, :, 0] - self.X[:, None, 0]
        dyp = self.posts[None, :, 1] - self.X[:, None, 1]
        pb = np.arctan2(dyp, dxp) - self.X[:, None, 2]
        db = np.arctan2(np.sin(pb - bearing), np.cos(pb - bearing))
        pr = np.hypot(dxp, dyp)
        sr = max(p.range_sigma_frac * rng, 0.05)
        ll = -0.5 * (db / p.bearing_sigma) ** 2 - 0.5 * ((pr - rng) / sr) ** 2
        best = ll.max(axis=1)                            # the post this sighting fits best, per particle
        self.w = self.w * np.exp(best - best.max())
        tot = float(self.w.sum())
        if not np.isfinite(tot) or tot <= 0.0:
            self.w = np.full(len(self.X), 1.0 / len(self.X))
            return
        self.w /= tot
        if 1.0 / float((self.w ** 2).sum()) < p.resample_ess * len(self.X):
            self._resample()

    def _resample(self) -> None:
        assert self.X is not None and self.w is not None
        n = len(self.X)
        u = (self.rng.random() + np.arange(n)) / n
        idx = np.searchsorted(np.cumsum(self.w), u)
        idx = np.clip(idx, 0, n - 1)
        self.X = self.X[idx].copy()
        self.w = np.full(n, 1.0 / n)

    def _estimate(self) -> tuple[float, float, float]:
        assert self.X is not None and self.w is not None
        x = float(self.w @ self.X[:, 0])
        y = float(self.w @ self.X[:, 1])
        yaw = float(math.atan2(self.w @ np.sin(self.X[:, 2]), self.w @ np.cos(self.X[:, 2])))
        return x, y, yaw

    def spread(self) -> float:
        """The cloud's position spread (m, RMS about the mean): how sure it is."""
        assert self.X is not None and self.w is not None
        x, y, _ = self._estimate()
        return float(math.sqrt(self.w @ ((self.X[:, 0] - x) ** 2 + (self.X[:, 1] - y) ** 2)))


def pitch_posts(bounds: tuple[float, float], goal_w: float) -> list[tuple[float, float]]:
    """The four goal posts of a pitch, in the world frame, from what a chase
    brain is already constructed with: `bounds` = (half_x, half_y) inside
    the boards, whose x IS the goal mouth line (`World.goal_for` puts the
    mouth at floor/2 - 0.25 = bounds[0]), and `goal_w` the mouth's width."""
    hx, hw = float(bounds[0]), float(goal_w) / 2.0
    return [(hx, hw), (hx, -hw), (-hx, hw), (-hx, -hw)]
