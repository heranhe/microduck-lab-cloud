"""Hand-written controllers over the ToF matrix (roadmap 2.3).

`wander_from_tof` is the first brain: cruise forward, slow down as the
middle of the depth matrix closes in, turn toward whichever side has more
room, and spin in place when nothing ahead is far enough. It reads only
what the sensor reports (dropped zones count as unknown, never as "far"),
and it emits only a twist, so it runs unchanged against the real robot's
`tof.stream` and `robot.move` once the bridge exists.

It is deliberately dumb: it is the baseline a learned brain has to beat,
and the lesson page shows exactly which zones it looked at.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields, replace

import numpy as np

from .gait import TURN_KICK, GaitWatch, back_up, clip_wz, max_wz, turn
from .intercept import Interceptor
from .runtime import REGISTRY, Intent, Senses, age_inputs
from .tracker import Tracker, TrackerParams

# The camera's horizontal HALF-field (62° full, sensors/detector.py) plus a
# little: past this a target is not in the picture at any head pitch.
GAZE_MAX_BEARING = 0.6


@dataclass(frozen=True)
class WanderParams:
    cruise: float = 0.3        # m/s when the way is clear
    slow_at: float = 0.7       # start slowing when the centre is nearer than this
    stop_at: float = 0.3       # stop and spin when nearer than this
    turn: float = 0.8          # rad/s while steering around something
    spin: float = 1.0          # rad/s when boxed in
    rows: tuple[int, int] = (2, 7)   # zone rows that count: skip the sky, keep the floor edge
    max_range_m: float = 4.0


def _column_clearance(depth_mm: np.ndarray, valid: np.ndarray | None,
                      p: WanderParams) -> np.ndarray:
    """Nearest reported target per column over the counted rows; +inf where
    no zone in the column reported anything."""
    d = depth_mm.astype(np.float64) / 1000.0
    ok = (depth_mm > 0) if valid is None else (valid & (depth_mm > 0))
    r0, r1 = p.rows
    d, ok = d[r0:r1], ok[r0:r1]
    d = np.where(ok, d, np.inf)
    return d.min(axis=0)


def tof_hits_3d(frame) -> tuple[np.ndarray, np.ndarray] | None:
    """Every zone's hit in the body's HEADING frame, relative to the trunk
    (the frame the mount pose is given in): (rows, cols, 3) points and the
    (rows, cols) depths; None for a frame without a mount pose."""
    if frame.mount_pos is None or frame.mount_rot is None or frame.dirs_local is None:
        return None
    d = frame.depth_mm.astype(np.float64) / 1000.0
    dirs = frame.dirs_local @ frame.mount_rot.T
    return frame.mount_pos[None, None, :] + dirs * d[..., None], d


TRUNK_Z = 0.117                    # the standing trunk height (the ToF mount pose is trunk-relative)


def tof_clearance_bearings(frame, ahead_half: float = 0.10, side_half: float = 0.40,
                           zmin: float = -0.03, zmax: float = 0.5) -> tuple[float, float, float]:
    """(ahead, left, right) body-height clearance selected by BEARING off the
    body's nose, not by sensor column - because the sensor is IN THE HEAD.
    A column is "ahead" only while the head looks along the walking line;
    yaw the head and the middle columns report whatever is off to the side,
    which the brain then stops for (measured: every head-gaze variant lost
    kicks that way). Placed in the heading frame, a hit's bearing says where
    it really is, so a head turned off the line simply returns +inf ahead -
    honestly blind, rather than confidently wrong. `ahead_half` (5.7 deg)
    matches the two middle columns of a 45 deg sensor, `side_half` (23 deg)
    the outer three. Ranges are horizontal, which is what the walking
    thresholds mean. Falls back to the level-head columns for a synthetic
    frame with no mount pose."""
    hits = tof_hits_3d(frame)
    if hits is None:
        cols = _column_clearance(frame.depth_mm, frame.valid, WanderParams(rows=(2, 5)))
        return float(cols[3:5].min()), float(cols[0:3].min()), float(cols[5:8].min())
    pts, _ = hits
    z = pts[..., 2]
    ok = frame.valid & (frame.depth_mm > 0) & (z > zmin) & (z < zmax)
    v = pts - frame.mount_pos[None, None, :]                 # from the APERTURE, as the column version measured
    bear = np.arctan2(v[..., 1], v[..., 0])                  # +left, the brain's convention
    rng = np.where(ok, np.hypot(v[..., 0], v[..., 1]), np.inf)
    ahead = rng[np.abs(bear) <= ahead_half]
    left = rng[(bear > ahead_half) & (bear <= side_half)]
    right = rng[(bear < -ahead_half) & (bear >= -side_half)]
    return (float(ahead.min()) if ahead.size else np.inf,
            float(left.min()) if left.size else np.inf,
            float(right.min()) if right.size else np.inf)


def tof_floor_ball(frame, r_max: float = 0.5, z_lo: float = -0.09, z_hi: float = -0.02) -> tuple[float, float] | None:
    """A ball-sized thing on the floor inside `r_max`, seen by the ToF:
    hits above the floor plane but below 10 cm (trunk-relative z between
    `z_lo`, 2.7 cm up - floor hits scatter to 1.8 cm with datasheet noise -
    and `z_hi`, 9.7 cm up; a ball is 7 cm tall), at least two adjacent
    zones of them, in columns with NOTHING taller near (a wall or a duck
    has hits above the band in the same columns; a ball has the floor
    behind it) - returned as (bearing, horizontal range) of the nearest
    such cluster, heading frame. The camera loses a floor ball inside
    0.3 m unless the head dips; the ToF at 45 deg shows one at 0.3 m as a
    3-6 zone blob (measured). None for a frame without a mount pose."""
    hits = tof_hits_3d(frame)
    if hits is None:
        return None
    pts, _ = hits
    z = pts[..., 2]
    rng = np.hypot(pts[..., 0], pts[..., 1])
    live = frame.valid & (frame.depth_mm > 0)
    ok = live & (z > z_lo) & (z < z_hi) & (rng < r_max)
    if ok.sum() < 2:
        return None
    r, c = np.unravel_index(np.argmin(np.where(ok, rng, np.inf)), ok.shape)
    win = np.zeros_like(ok)
    win[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
    sel = ok & win
    if sel.sum() < 2:
        return None
    cols = sel.any(axis=0)
    tall = live & (z >= z_hi) & (rng < r_max + 0.15) & cols[None, :]
    if tall.any():
        return None
    x, y = float(pts[..., 0][sel].mean()), float(pts[..., 1][sel].mean())
    return float(math.atan2(y, x)), float(math.hypot(x, y))


def tof_clearance_3d(frame, zmin: float = -0.03, zmax: float = 0.5) -> np.ndarray:
    """Nearest return per column that is a BODY-height thing — a wall, a
    duck, furniture — whatever the head is doing: each zone's hit is placed
    in the body's heading frame from the mount pose the frame carries
    (rotated by the head's pose: an unrotated placement read the FLOOR as a
    wall 0.35 m ahead whenever the head dipped 0.6 rad, measured), and hits
    on the floor (below `zmin`, trunk-relative: 8.7 cm above the floor, a
    ball is 7 cm tall) or above `zmax` do not count. Falls back to the
    level-head rows when a frame carries no mount pose (synthetic frames)."""
    hits = tof_hits_3d(frame)
    if hits is None:
        return _column_clearance(frame.depth_mm, frame.valid, WanderParams(rows=(2, 5)))
    pts, d = hits
    z = pts[..., 2]
    ok = frame.valid & (frame.depth_mm > 0) & (z > zmin) & (z < zmax)
    return np.where(ok, d, np.inf).min(axis=0)


@dataclass(frozen=True)
class ClosingParams:
    range_m: float = 1.2           # something inside this, ahead...
    rate: float = 0.12             # ...closing on me faster than my own walk by this (m/s)
    window_s: float = 0.4          # over this much ToF history (6 frames at 15 Hz)
    # The manoeuvre. Measured from a standstill: a pure sidestep moves the
    # walker 1 cm in its first second (6 cm in two); a turn toward the
    # freer side then a walk moves it off the line - and out of the
    # oncoming path, which a stop alone is not.
    turn_s: float = 1.0
    walk_s: float = 1.5
    speed: float = 0.3
    cooldown_s: float = 1.5        # after one, look again this much later


class ClosingWatch:
    """Something walking at me. The ToF's clearance ahead (the middle
    columns, body-height returns only) shrinks at my own speed when I walk
    at a wall and faster when a person or a duck comes at me; the
    difference, fitted over a short window, is the closing rate. Past
    `rate` inside `range_m`, the answer is a manoeuvre out of its path: a
    turn toward the freer side (the columns with more clearance), then a
    walk. Nothing here needs more than the robot's ToF and its own
    commanded speed."""

    def __init__(self, p: ClosingParams = ClosingParams()):
        self.p = p
        self.reset()

    def reset(self) -> None:
        self._hist: list[tuple[float, float]] = []      # (t, clearance ahead)
        self._last_t: float | None = None
        self.closing = 0.0                               # m/s toward me beyond my own walk (last estimate)
        self.side = 0.0
        self.t0 = -1e9
        self.until = -1e9
        self.count = 0

    def step(self, frame, t: float, speed: float, cold: bool = True) -> tuple[float, float, float] | None:
        """Fold the newest ToF frame; returns the twist to hold now, or
        None when there is nothing to get out of the way of."""
        p = self.p
        if frame is not None and frame.t != self._last_t:
            self._last_t = frame.t
            cols = tof_clearance_3d(frame)
            ahead = float(cols[3:5].min())
            if np.isfinite(ahead):
                self._hist.append((frame.t, ahead))
            else:
                self._hist.clear()
            self._hist = [h for h in self._hist if frame.t - h[0] <= p.window_s]
            if len(self._hist) >= 3 and self._hist[-1][0] > self._hist[0][0]:
                ts = np.array([h[0] for h in self._hist])
                ds = np.array([h[1] for h in self._hist])
                slope = float(np.polyfit(ts - ts[0], ds, 1)[0])          # m/s, negative = shrinking
                self.closing = -slope - max(float(speed), 0.0)
            else:
                self.closing = 0.0
            if ahead < p.range_m and self.closing > p.rate and t >= self.until + p.cooldown_s:
                left = float(np.mean(np.minimum(cols[0:3], 4.0)))
                right = float(np.mean(np.minimum(cols[5:8], 4.0)))
                self.side = 1.0 if left >= right else -1.0
                self.t0 = t
                self.until = t + p.turn_s + p.walk_s
                self.count += 1
        if t >= self.until:
            return None
        if t < self.t0 + p.turn_s:
            return turn(self.side, cold)
        return (p.speed, 0.0, 0.0)


def wander_from_tof(depth_mm: np.ndarray, valid: np.ndarray | None = None,
                    p: WanderParams = WanderParams(),
                    prefer_left: bool | None = None) -> tuple[float, float, float]:
    """One decision from one frame. Returns (vx, vy, wz).

    `prefer_left` breaks a tie (and keeps a turn going) — a stateless
    controller re-deciding every frame would dither between the two sides.
    """
    cols = _column_clearance(depth_mm, valid, p)
    centre = float(cols[2:6].min())
    left = float(np.mean(np.minimum(cols[:4], p.max_range_m)))
    right = float(np.mean(np.minimum(cols[4:], p.max_range_m)))
    if prefer_left is None:
        prefer_left = left >= right
    elif abs(left - right) > 0.15:
        prefer_left = left > right
    sign = 1.0 if prefer_left else -1.0        # +wz turns left (toward +y, column 0)
    if centre < p.stop_at:
        return 0.0, 0.0, sign * p.spin
    if centre < p.slow_at:
        frac = (centre - p.stop_at) / (p.slow_at - p.stop_at)
        return p.cruise * frac, 0.0, sign * p.turn
    return p.cruise, 0.0, 0.0


class Wander:
    """Stateful wrapper: remembers the turn direction and, when the duck has
    made no progress for a while under a forward command, spins to unstick.
    Also a `Brain` (runtime.py): `step(senses)` gates the ToF on age."""

    kind = "wander"
    TOF_MAX_AGE = 0.25       # ~3 frames at 15 Hz: older than that, stand

    def __init__(self, p: WanderParams = WanderParams(), stuck_s: float = 2.0,
                 unstick_s: float = 1.2):
        self.p = p
        self._senses: Senses | None = None
        self.prefer_left: bool | None = None
        self.stuck_s, self.unstick_s = stuck_s, unstick_s
        self._still_since: float | None = None
        self._unstick_until = -1.0
        self.last: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.state = "cruise"

    def reset(self) -> None:
        self.prefer_left = None
        self._still_since = None
        self._unstick_until = -1.0
        self.state = "cruise"

    def step(self, senses: Senses) -> Intent:          # the Brain interface
        self._senses = senses
        f = senses.fresh_tof(self.TOF_MAX_AGE)
        tw = self.decide(None if f is None else f.depth_mm, None if f is None else f.valid,
                         senses.t, senses.speed)
        return Intent(twist=tw, note=self.state)

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        return age_inputs(self._senses, self.TOF_MAX_AGE, 9e9)

    def decide(self, depth_mm: np.ndarray | None, valid: np.ndarray | None,
               t: float, speed: float | None = None) -> tuple[float, float, float]:
        if t < self._unstick_until:
            self.state = "unstick"
            self.last = (0.0, 0.0, (1.0 if self.prefer_left else -1.0) * self.p.spin)
            return self.last
        if depth_mm is None:
            self.state = "blind"
            self.last = (0.0, 0.0, 0.0)      # no frame yet: stand, do not guess
            return self.last
        vx, vy, wz = wander_from_tof(depth_mm, valid, self.p, self.prefer_left)
        self.prefer_left = wz > 0 if wz else self.prefer_left
        self.state = "spin" if vx == 0.0 and wz != 0.0 else ("steer" if wz else "cruise")
        # Stuck detector: asking for forward motion and getting none.
        if speed is not None and vx > 0.1 and abs(speed) < 0.03:
            if self._still_since is None:
                self._still_since = t
            elif t - self._still_since > self.stuck_s:
                self._still_since = None
                self._unstick_until = t + self.unstick_s
                if self.prefer_left is None:
                    self.prefer_left = True
                self.state = "unstick"
                self.last = (0.0, 0.0, (1.0 if self.prefer_left else -1.0) * self.p.spin)
                return self.last
        else:
            self._still_since = None
        self.last = (vx, vy, wz)
        return self.last


@dataclass(frozen=True)
class FollowParams:
    target_cls: str = "person"     # what to follow ("person" or "duck")
    distance: float = 0.7          # hold this far behind, m
    k_turn: float = 8.0            # wz per rad of bearing (swept on 8 episodes: 3.0 kept sight 0.36 of the time, 8.0 with the idle sidestep 0.51)
    k_speed: float = 1.2           # vx per m of distance error (0.6 could not close on a 0.35 m/s walker)
    max_speed: float = 0.5
    min_speed: float = 0.12        # alpha_walking treats slower asks as "stand"
    turn_first: float = 0.6        # rad off the nose beyond which it turns in place before walking
    k_lead: float = 0.0            # wz per rad/s of bearing RATE: turn toward where the target is going
    # Keep the gait WARM: standing still, the walker cannot start a right
    # turn and starts a left one slowly, so the person walks out of the
    # frame before the body follows (measured: the learned brain sidesteps
    # ±0.23 the whole time and holds the bearing at 0.13 rad; the scripted
    # one stood, went cold and averaged 0.82). A sidestep toward the
    # target's side whenever it would otherwise stand keeps the legs going.
    idle_vy: float = 0.25
    idle_coast: bool = True        # …also while coasting on a lost track
    coast_speed: float = 0.0       # walking speed on a coasted track (0: stand and turn, measured safer)
    lost_s: float = 2.0            # keep the last bearing this long, then search
    # DEAD KNOB - declared, never read. The search turn goes through
    # `gait.turn()`, which takes its magnitude from `gait.max_wz()`. Left in
    # place with this note rather than deleted because deleting it silently
    # would let the next person re-add it; sweeping it would measure exactly
    # nothing, which is the failure mode this file documents elsewhere.
    # (Below 1.0 a COLD walker does not turn at all - exactly 0, both ways.)
    search_wz: float = 1.0
    tof_stop: float = 0.35         # never walk into what the ToF says is right there
    head_yaw_gain: float = 0.8     # look toward the target (the robot's own gaze intent)
    # Get out of the path of whatever walks at me (ClosingWatch): OFF.
    # Measured over 12 episodes: the walker cannot clear a person walking
    # at it (it sidesteps 1 cm in its first second; a turn-and-walk moves
    # it 0.1 m off the line in 1.8 s, and the person arrives in 2-5 s), so
    # the charge case's contact time is the same either way (4.9 vs 5.3
    # s/ep) while in ordinary following the dodge fires on a person who is
    # merely walking toward the duck and loses it (in band 0.49 -> 0.39,
    # in sight 0.86 -> 0.57). It halves falls in the charge case (0.17 ->
    # 0.08/ep), within the noise. `eval-brain --avoid` measures it.
    avoid: bool = False


class Follow:
    """Keep the nearest target of a class ahead at a fixed distance, from a
    TRACK over the detector's frames (bearing, width-derived range; brain/
    tracker.py), with the ToF as a bumper. Loses it: the track coasts —
    its bearing turning with the body — for `lost_s`, then turn to search.

    Deliberately simple — it is the baseline the learned brain (3.1/3.2) is
    measured against, and every number it uses is one the real robot can
    produce today or after one detector retrain. Turns go through
    brain/gait.py: the walker does not start a right turn from a standstill.
    """

    kind = "follow"
    DET_MAX_AGE = 0.4
    TOF_MAX_AGE = 0.25

    def __init__(self, p: FollowParams = FollowParams(), tracker: TrackerParams = TrackerParams()):
        self.p = p
        self.tracker = Tracker(tracker)
        self.closing = ClosingWatch()
        self.gait = GaitWatch()
        self.reset()

    def reset(self) -> None:
        self.state = "search"
        self.last_bearing = 0.0
        self.last_seen_t: float | None = None
        self.last_range: float | None = None
        self.track_id: int | None = None
        self._prev_track: tuple[float, float] | None = None     # (t, bearing) for the lead term
        self._senses: Senses | None = None
        self.last = (0.0, 0.0, 0.0)
        self.tracker.reset()
        self.gait.reset()
        self.closing.reset()

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["target"] = None if self.last_seen_t is None else {
            "bearing": round(self.last_bearing, 3), "range": _r(self.last_range),
            "since": round(self._senses.t - self.last_seen_t, 2), "track": self.track_id}
        out["tracks"] = self.tracker.payload(self._senses.t)
        out["closing"] = round(self.closing.closing, 2)
        return out

    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        p = self.p
        cold = self.gait.update(senses)
        yaw = None if senses.odom is None else senses.odom[2]
        self.tracker.update(senses.fresh_det(self.DET_MAX_AGE), senses.t, yaw)
        # Stay on the track we have while it lives; otherwise the best one.
        target = None
        if self.track_id is not None:
            target = next((tr for tr in self.tracker.tracks if tr.id == self.track_id), None)
        if target is None:
            target = self.tracker.best(p.target_cls, senses.t)
            self.track_id = None if target is None else target.id
        # ToF bumper: the nearest thing in the middle columns.
        tof = senses.fresh_tof(self.TOF_MAX_AGE)
        ahead = np.inf
        if tof is not None:
            cols = _column_clearance(tof.depth_mm, tof.valid, WanderParams())
            ahead = float(cols[3:5].min())
        fresh = target is not None and target.age(senses.t) <= self.DET_MAX_AGE
        vy = 0.0
        if target is not None and (fresh or target.age(senses.t) < p.lost_s):
            self.last_bearing = target.bearing
            self.last_range = target.range
            if fresh:
                self.last_seen_t = senses.t
            rate = 0.0
            if p.k_lead and self._prev_track is not None and senses.t > self._prev_track[0]:
                rate = (target.bearing - self._prev_track[1]) / (senses.t - self._prev_track[0])
            self._prev_track = (senses.t, target.bearing)
            wz = clip_wz(p.k_turn * target.bearing + p.k_lead * float(np.clip(rate, -2.0, 2.0)))
            err = target.range - p.distance
            vx = float(np.clip(p.k_speed * err, 0.0, p.max_speed))
            if abs(target.bearing) > p.turn_first:
                vx, _, wz = turn(target.bearing, cold)      # turn first, walk after
            elif 0.0 < vx < p.min_speed:
                vx = 0.0 if err < 0.1 else p.min_speed
            if not fresh:
                # Coasting: face where the track says it went, but do not
                # walk at a range nobody has measured lately (measured:
                # walking on a coasted track bumped the person 50% more).
                vx = p.coast_speed if err > 0.2 else 0.0
                if abs(target.bearing) > 0.15:
                    vx, _, wz = turn(target.bearing, cold)
                else:
                    wz = 0.0
            self.state = ("hold" if vx == 0.0 and abs(wz) < 0.2 else "approach") if fresh else "coast"
            if p.idle_vy and vx == 0.0 and abs(wz) < 1.0 and (fresh or p.idle_coast):
                vy = p.idle_vy * (1.0 if target.bearing >= 0.0 else -1.0)
        else:
            self.track_id = None
            self._prev_track = None
            vx, _, wz = turn(1.0 if self.last_bearing >= 0 else -1.0, cold)
            self.state = "search"
        if ahead < p.tof_stop and (vx > 0 or vy != 0.0):
            # Never walk into what is right there — a cold-turn kick included,
            # the idle sidestep too: blocked, the search turns LEFT without
            # the kick (the turn that does start from a standstill, see
            # brain/gait.py).
            vx, vy = 0.0, 0.0
            if self.state == "search":
                wz = max_wz()
            else:
                self.state = "blocked"
        # Something walking at me - the person turning back, another duck -
        # gets a sidestep out of its path (after the bumper: this one is
        # meant to move with something right there).
        dodge = self.closing.step(tof, senses.t, senses.speed, cold) if p.avoid else None
        if dodge is not None:
            vx, vy, wz = dodge
            self.state = "dodge"
        head_yaw = float(np.clip(p.head_yaw_gain * self.last_bearing, -0.6, 0.6)) if self.last_seen_t else 0.0
        self.last = (vx, vy, wz)
        return Intent(twist=self.last, head=(0.0, 0.0, head_yaw, 0.0), note=self.state)


@dataclass(frozen=True)
class ChaseParams:
    """The chase brain's constants. Most were measured INTO their value; a
    good few were measured OFF and keep their number in the comment beside
    them, because "we tried it and it was worse" is the expensive part and
    deleting it invites the next person to spend the afternoon again.

    Shipping OFF (0 / False), with their measurements below: `two_stage`,
    `lineup_lat` (its speed-up) and `search_walk_after` (line-up
    precision), `kick_deflect_*` (the kick
    map in the stance), `kick_cone` (shoot only from close),
    `predict_steer` (a walked prediction line), `look_aim`, `search_sweep`,
    `gaze_still`/`gaze_neck`/`gaze_yaw` (what the
    head does about the ball), `tof_ball_m` (the ToF seeing a ball at the
    feet), `seek_s` (a ball memory), `push_beyond` (deliberate bumping),
    `search_sided`, `mate_keepout` (teammates' poses as obstacles),
    `support_turn_vx`, `support_mode="ahead"` (a poacher supporter, found
    then killed by fresh seeds). Read the numbers before re-trying one -
    and read the README's note on what 8 seeds can and cannot resolve
    first, because several of those "worse" verdicts are inside the noise
    and say only "not shown to help".
    """
    target_cls: str = "ball"
    speed: float = 0.45            # walk at the ball
    k_turn: float = 3.0
    turn_first: float = 0.6        # rad off the nose: turn in place before walking
    lost_s: float = 2.0
    tof_stop: float = 0.3          # walls and ducks (body-height ToF returns); the ball and the floor do not count
    side_stop: float = 0.22        # a wall this close in the side columns: no turn in place toward it
    # The shipped kicks (measured, `walker-facts`-style, on the walker): a
    # ball 0.08 m ahead of the trunk and 0.06 m to the kicking foot's side
    # flies 1.6 m; 0.10 m dead ahead barely moves; the other side, nothing.
    kick_ahead: float = 0.08
    kick_side: float = 0.06
    # The kick map (a standing duck, the ball swept over (ahead, side) of
    # the trunk, kick_left; the right kick checked mirrored): the ball
    # leaves at an angle to the BODY heading that depends on the side
    # offset - 15 deg/cm near 2 cm, 4.5 deg/cm around 4-8 cm, where the
    # shipped spot sits - and at the spot it is +21.6 deg for the left foot
    # (2.1 m) and -11 deg for the right (1.9 m), the same whichever way the
    # body is yawed. Set, the line-up stands the body rotated by this so a
    # kick from the sweet spot flies along the line to the goal. OFF
    # (measured): in play the ball is 2-3 cm off the sweet spot when the
    # kick fires - the line-up, not the map, is what scatters shots - and
    # the rotated stance scored 1.38 goals a run against 2.00 without it
    # (8 seeds x 300 s, 10.4 vs 8.4 kicks); on a 12-spot lone-shot probe
    # the direction error was 28 vs 35 deg mean absolute, noise-dominated
    # either way. The map's lesson that ships: line-up precision is the
    # next lever for goals, and the sweet spot is 6-10 cm ahead, 4-8 cm
    # to the side.
    #
    # RE-MEASURED 2026-09-06 on the quantity it moves, because the verdict
    # above was 8 seeds judged on GOALS (which need 136). It is refuted
    # again, far more strongly, and now with a mechanism. 24 seeds x 300 s
    # of 2v2 (`runs/deflect/`), paired, against 178 kicks / 32.0 deg mean
    # absolute error:
    #
    #   compensating by the measured error (+13.7 / -6.0 deg)
    #       120 kicks (-2.42 a seed, p=0.004)   |err| 44.7 (+13.1, p=0.0005)
    #   compensating by the in-play map (+23.6 / -28.7 deg)
    #        80 kicks (-3.82 a seed, p<1e-4)    |err| 43.0 (+11.4, p=0.037)
    #
    # It does not halve the error, it DOUBLES it, and it costs a third to a
    # half of the touches. WHY: the spot is laid out in the rotated heading,
    # so rotating the stance does not pre-aim the shot - it moves where the
    # duck stands. Measured: rotating the left stance +23.6 deg shifts the
    # ball's departure off the body by +24.0 deg, essentially 1:1, and the
    # ball's SIDE offset grows +8.7 cm (p=0.0006). At the measured
    # +1.90 deg of aim error per cm of side offset that predicts +16.5 deg
    # of extra error; +13.1 and +11.4 were observed. The coefficient
    # predicts its own failure.
    #
    # THE GENERAL LESSON, which kills a family of ideas and not just this
    # knob: you cannot fix this kick by ROTATING anything. The kick spot is
    # defined relative to the body heading, so every rotation moves the
    # ball's side offset, and the side offset IS the error. That includes
    # rotating the intended line by the PREDICTED offset, which this
    # roadmap proposed earlier the same day and this refutes. The only
    # levers left are the offset itself, or declining the shot when it is bad.
    kick_deflect_left: float = 0.0
    kick_deflect_right: float = 0.0
    # DECLINE the shot when the ball is too far to the side to be worth
    # swinging at. This is the only lever the coefficient leaves open: the
    # side offset IS the aim error (+1.90 deg/cm) and no rotation can remove
    # it, because every rotation moves the offset (see `kick_deflect_*`).
    # What is left is not to swing. Metres of |side| offset at the swing
    # above which the duck drops its spot and re-approaches; 0 = off.
    # The sweet spot is 0.04-0.08 m and the observed median is 0.133 m.
    #
    # It refuses only on a FRESH estimate (`Chase.predicted`, which needs
    # `predict_s` > 0 — on since 2026-09-06). A stale ball gets its swing:
    # that is what stops the gate turning into a duck that never kicks.
    #
    # MEASURED OFF the day it was built. 24 paired seeds x 300 s of 2v2
    # (`runs/decline/`), against 178 kicks / 137 effective / 23.0% whiffs:
    #
    #   decline > 0.12 m   152 kicks, 124 effective, 18.4% whiff
    #   decline > 0.09 m   133 kicks, 107 effective, 19.5% whiff
    #
    # The per-foot bias LOOKS like it is coming out (left +13.7 -> +7.1 ->
    # +4.1, right -6.0 -> -0.6 -> +1.6), and that is the trap: every one of
    # those intervals spans zero, the baseline's included. Paired per seed
    # nothing improves — whiff rate -4.8 points at 0.12 is p = 0.059, mean
    # absolute error moves the WRONG way (+2.7, +5.3), and kicks fall 1.88 a
    # seed at 0.09 (p = 0.002). Rule 6 again: a better rate on fewer touches.
    #
    # THE DIAGNOSTIC: the side offset of the kicks that survived did not
    # change (-0.002 m, p = 0.83; +0.004 m, p = 0.75). The obvious reading is
    # that the estimate is junk and the gate refuses at random. That reading
    # was written here first and it is WRONG — `scripts/probe_shot_gate.py`
    # measures the signal directly, over 162 kicks on 24 seeds:
    #
    #   |predicted side| vs |actual side|   r = +0.48 (p = 7e-5)
    #   the estimate's own error            median 0.023 m, 90th 0.083
    #   as a gate at 0.12 m                 refuses 31% of the swings it can
    #                                       see, and 88% of those really were
    #                                       wide
    #
    # The estimate is GOOD. What it is not is AVAILABLE: the brain has a
    # fresh fix on **34% of swings**. So the gate can only assess a third of
    # the population, refuses about a tenth of all swings, and cannot move a
    # pooled statistic that the other two thirds still dominate — while
    # paying the full price in touches for the ones it does refuse.
    #
    # Kept, off, with the numbers. The mechanism is sound and the precision
    # is there; COVERAGE is what is missing, and coverage is the camera's
    # blind radius (4c), not a threshold to retune. Re-run this the day the
    # brain can see the ball inside 0.35 m — and not before.
    kick_side_max: float = 0.0
    # Plan the kick spot for where the ball WILL be when the duck gets there,
    # not where it was last seen: at most this many seconds of lead, from the
    # track's own velocity and `ball_decel`. 0 = off.
    #
    # MEASURED WHY THIS EXISTS. Over 191 kicks the duck reaches its spot to
    # 1.4 cm — the walk-in is not the problem — and the SPOT is a median
    # 0.349 m from the ball when it should be `kick_ahead` = 0.08 m, because
    # the ball drifts 0.22-0.27 m during a line-up that is 3.3 s old by the
    # time the swing fires. Not one of those 191 kicks had the ball on the
    # sweet spot. So the plan is right when it is made and stale when it is
    # used, and a lead is the only one of the three ways out that addresses
    # the drift rather than avoiding it (`scripts/probe_kick_line.py`).
    #
    # SHIPS OFF: MEASURED AND IT DOES NOT WORK. Swept 0 / 0.5 / 1.0 / 2.0 s
    # over 24 seeds x 300 s of 2v2 each, judged on the on-spot fraction:
    #
    #     lead   kicks   on the sweet spot   whiffed   spot-to-ball   plan age
    #     0.0     191          0%              18%        0.285 m      3.26 s
    #     0.5     175          0%              22%        0.264 m      2.92 s
    #     1.0     125          0%              21%        0.261 m      2.98 s
    #     2.0     130          0%              24%        0.287 m      3.22 s
    #
    # Zero of every arm, and the whiff rate rises. The reason is the same
    # blindness that causes the staleness: the track's velocity is
    # differenced from SIGHTINGS, and the sightings stop at `refresh_min`
    # (0.35 m) — so the prediction is extrapolated from data that is exactly
    # as old as the plan it is meant to rescue. You cannot predict your way
    # out of not looking. Three aim-side fixes have now died on this
    # (`kick_deflect_*`, `two_stage`/`lineup_lat`, and this), which is what
    # points at the head and the blind radius instead.
    spot_lead: float = 0.0
    lineup_range: float = 0.6      # a ball seen inside this is worth lining up on
    # …and the spot is re-planned from sightings down to this range (the
    # detector's own slant `range_est`), then walked blind.
    #
    # 0.35 IS THE LEVEL CAMERA'S BLIND RADIUS, measured: a floor ball inside
    # 0.37 m of ground distance — 0.35 of slant range — is not reported at
    # all (`scripts/probe_head_pitch.py`). So the constant is honest about
    # what a level head can do, and the gaze can do better: pitched by
    # `_gaze` the same camera holds the ball to 0.18 m at the shipped clamp
    # and 0.08 m at the joint stop.
    #
    # MEASURED OFF ANYWAY, twice, for the same reason. The first time: at
    # 0.2 m the bearing noise is centimetres, the foot choice flipped, and
    # the spot dithered for 8 s. Re-measured over 24 seeds x 300 s of 2v2
    # with the probe that can see the placement (`scripts/probe_kick_line.py`),
    # `refresh_min` 0.35 -> 0.20 does exactly what it promises to the PLAN —
    # ball ahead of the trunk 0.238 -> 0.184 m, side 0.141 -> 0.083,
    # spot-to-ball 0.285 -> 0.220, plan age 3.26 -> 2.11 s, whiffs 18% ->
    # 10% — and costs 58% of the touches: 191 kicks -> 80 (p < 1e-11). The
    # whiff RATE improving while the kick COUNT halves is the `two_stage`
    # shape again (AGENTS.md rule 6): in absolute terms it is 156 effective
    # kicks against 72. Still 0 of 80 on the sweet spot.
    #
    # CONFIRMED ON 24 FRESH SEEDS, and this is the only arm of the whole
    # head/blindness investigation that survived one. Pooled over 48 paired
    # seeds, 150 kicks against the baseline's 360:
    #     ball ahead of the trunk   0.244 -> 0.182 m   (p = 7e-11)
    #     spot-to-ball              0.297 -> 0.215 m   (p = 1e-12)
    #     plan age                  3.32  -> 2.30 s    (p = 2e-19)
    #     ball drift since the plan 0.224 -> 0.151 m   (p = 7e-8)
    #     near the sweet spot       1/360 -> 9/150     (p = 0.0001)
    #     ON the sweet spot         0/360 -> 1/150     (p = 0.29)
    # and on the play ledger (24 seeds of 2v2) possession 21.6 -> 26.5 s/min
    # (p = 0.0003, better on 19 of 24 seeds) with goals 39 -> 47 (p = 0.34),
    # signed `ballProgress` FLAT (-0.050, p = 0.66) and the whiff rate flat
    # pooled (19.2% -> 17.3%).
    #
    # So it is a real fix to the STALENESS and it is not a win: the duck
    # keeps re-planning instead of swinging, which is possession bought with
    # touches, and the ball ends up no further forward. Ships at 0.35, with
    # the numbers, because the next person to reach for the blind radius
    # should start from here and not from the head.
    refresh_min: float = 0.35
    # The line-up is two stages (traced: with the spot 8 cm behind the
    # ball, the walk-in's last steering steps and the square-up's turn in
    # place pushed the ball 5-50 cm before the kick - the ball moved 15 cm
    # on average between the last sighting and the kick, more than the
    # 7 cm the sighting was off by). Stage one goes to a pre-spot
    # `approach_back` behind the kick spot on the kick line and squares up
    # THERE, the ball 30 cm from the feet; stage two walks straight in
    # along the line at `approach_speed` and stops on the spot.
    # OFF (measured over 8 seeds x 300 s of 1v1, goals attributed to a
    # kick within 4 s or to a bump): the two-stage line-up puts the ball
    # on the sweet spot (side error 3 cm, heading 4 deg, the ball moving
    # 2 cm before the kick, 4 goals from 11 lone shots against 1 from
    # 12) but kicks 3.5 times a run against 8.4, and those kicks scored
    # 0.00 against 0.75 kicked goals a run; bumped goals were 1.25 either
    # way. With this walker the kick that happens beats the kick that is
    # placed. `two_stage` switches it on.
    # AND READ ITS ADVANCE-PER-KICK WITH THE ADVANCE ITSELF. 0.163 against
    # the shipped 0.091 is a RATIO whose denominator is the thing this
    # arm changes. Re-measured over 12 paired seeds x 300 s (seeds
    # 100-111): shipped 87 kicks / advance +0.82 m/min, two_stage 35
    # kicks / +0.76, the `lineup_lat` variant 44 kicks / +0.71 - the
    # advance is FLAT to under 1 sigma across arms whose kick counts
    # differ by 5.8 - so advance-per-kick here is 1/kicks, and any change
    # that kicks LESS scores higher on it while moving the ball no
    # further. Measured directly instead, the ball's travel in the 2 s
    # after the swing: 17.7 +- 3.9 cm a kick shipped (34 kicks), 14.4 +-
    # 3.4 two-stage (17), 13.6 +- 3.0 with the faster line-up (20). The
    # placed kick is not worth more. Traced, it is not placed either: the
    # swing fires with the ball a median 21 cm (shipped) / 25 cm
    # (two_stage) ahead of the trunk where the sweet spot is 6-10, in
    # 3 of 17 two-stage kicks and 7 of 34 shipped ones inside a generous
    # box round it - the spot is planned at `refresh_min` or further and
    # the ball moves 20-28 cm during the attempt, so what scatters the
    # shot is the plan going stale, which a LONGER line-up makes worse.
    # The cost in the same paired block: goals 36 -> 21 (-2.3 sigma,
    # better on 11 of 12 seeds) and possession 18.6 -> 13.1 s/min
    # (-4.4 sigma).
    two_stage: bool = False
    approach_back: float = 0.22
    approach_speed: float = 0.25
    approach_tol: float = 0.04
    # Stage two follows the LINE, not the heading: on a pure forward
    # command the walker holds its yaw but crabs sideways (measured 9 cm
    # of side error over the 22 cm walk-in, and shots got worse), so a
    # gentle cross-track law steers it back onto the line - `k_lat` per m
    # off the line, `k_head` per rad off the heading, capped at
    # `approach_wz` so no step is a turn against the ball.
    k_lat: float = 4.0
    k_head: float = 1.5
    approach_wz: float = 0.5
    # Stage one's whole job is to put the duck ON the kick line, squared up,
    # behind the spot - so a duck that is already there has nothing to walk
    # back for. `lineup_lat` is how near the line counts as on it (stage
    # two's cross-track law closes the rest); 0 makes every line-up go via
    # the pre-spot, which is how the two-stage line-up was first measured.
    # Traced over 12 duck-runs of 300 s of 1v1 with `two_stage`: 27 s of
    # every 300 is the walk to the pre-spot and 4 s the square-up on it,
    # against 12 s of walk-in, and 55 attempts a run end in the back-off
    # below - the pre-spot BEHIND the duck. (That back-off turns away and
    # walks; `gait.back_up` would reach it without turning at all, and is
    # the obvious thing to measure here now that the walker is known to
    # reverse at 0.23 m/s. Untried.)
    # MEASURED at 0.06, `two_stage` on, against `two_stage` alone: the
    # line-up really does get faster - a kicking attempt 5.63 s -> 4.43 s
    # (its walk to the pre-spot 1.79 -> 1.19 s, the square-up on it 0.89 ->
    # 0.68), the pre-spot back-off 55 attempts -> 35 a run, 40% -> 42% of
    # attempts reaching the walk-in and 6.7% -> 8.7% of them firing - and
    # the play does not move: over 24 PAIRED seeds x 300 s of 1v1 (two
    # blocks of 12, the second fresh) kicks 72 -> 83, ballAdvance -0.036
    # +- 0.050, signed progress -0.105 +- 0.061, goals 45 -> 34 (sign
    # p = 0.24), falls 16 -> 11. Ships at 0: a real speed-up that buys
    # nothing the benchmark can see. Note the first 12 seeds promised +26%
    # kicks and a THIRD of the falls and the fresh 12 gave neither, which
    # is what 16 fall events and 35 kicks a block are worth.
    lineup_lat: float = 0.0

    # Traced: a re-plan with the ball already inside `backoff_range` puts
    # the pre-spot behind the duck, and the turn in place toward it is a
    # turn against the ball. Back off instead (turn away, walk clear - the
    # retreat manoeuvre) and line up again from further out. And a search
    # begun with the ball at the feet turns in place without ever seeing
    # it (a cold standing turn is exactly 0 rad/s): after `search_walk_after` with
    # no sighting, walk `search_walk_s` to change the view.
    backoff_range: float = 0.35
    search_walk_after: float = 0.0     # 0: off (measured with the two-stage line-up, see above)
    search_walk_s: float = 1.0
    lineup_tol: float = 0.03       # trunk within this of the kicking spot: kick
    lineup_s: float = 4.0          # give up a line-up after this long
    settle_s: float = 0.4          # stand this long on the spot before the kick (robotd kicks at standing tuning)
    kick_clear: float = 0.35       # no kick with anything closer than this ahead
    aim_tol: float = 0.25          # face the kick direction within this before kicking (rad)
    # Shoot only from inside the goal's cone. Measured on the shipped brain:
    # 7.4 kicks a run for 0.25 kicked goals - one shot in four - and a lone
    # shot's direction error is 28-35 deg, so a kick from far out is a
    # lottery whatever the line-up does. With `kick_cone` > 0 a ball whose
    # goal mouth subtends less than that half-angle is DRIBBLED instead
    # (the push spot, walked through toward the goal), which carries it
    # closer until the cone opens. 0.35 rad is about a metre out on a 0.7 m
    # goal. 0 = off (kick from anywhere).
    kick_cone: float = 0.0
    aim_max: float = 1.05          # aim at the goal only within this of the line of sight (rad)
    # What to do when the goal is FURTHER round the ball than `aim_max`, i.e.
    # when kicking at it means walking round to the far side. Three answers,
    # and the shipped one is the reason 30 of 53 kicks in the Track 4 baseline
    # sent the ball back toward the kicker's own goal:
    #   "los"    give up on the goal and kick along the line of sight — which
    #            is straight at our own goal whenever the duck reached the
    #            ball from the goal side, and the support geometry puts it
    #            there (a supporter stands `support_back` goal-side of the
    #            ball and walks in from there when it becomes the attacker).
    #   "clamp"  kick at the edge of the cone on the goal's side: the same
    #            walk-round `aim_max` already allows, and never worse than
    #            `aim_max` off the best available line.
    #   "goal"   always at the goal, whatever the walk-round costs.
    #
    # MEASURED, 24 paired seeds x 300 s of 2v2 and then 24 FRESH ones
    # (docs/roadmap.md Track 4.3.1). `clamp` ships: back-kicks 211 of 419
    # (50%) -> 109 of 324 (34%) pooled over the 48, p < 0.0001, and it
    # replicates on its own in each block (p = 0.011 then p = 0.0001) —
    # while goals, possession, advance, signed progress and crowd are all
    # flat over the 48. It costs 23% of the touches (419 -> 324): aiming
    # better means walking further round. Falls do not resolve pooled
    # (+0.29, p = 0.37) though the fresh block alone looked bad (p = 0.052),
    # which is what a block on its own is worth.
    # `goal` ships OFF with its numbers: it aims best (27% back, p = 0.0003)
    # and plays worst — half the kicks and signed progress -0.26 (p = 0.012,
    # worse on 17 of 24), the ball ending up nearer the ducks' own goals,
    # because a duck arcing round the ball is in possession the whole way
    # and shoves it backwards as it goes. That re-earns the first form's
    # verdict (4 kicks and 2 falls a run for 1.0 goals against 1.75 for the
    # cone rule) with an instrument that can see the mechanism.
    aim_mode: str = "clamp"
    # THE HEAD. `_gaze` is a law that puts a floor ball at range `rng` on the
    # camera's axis; `head_down` clamps the command it may ask for, and the
    # gaze is applied while WALKING at a ball inside `head_range`.
    #
    # MEASURED END TO END (`scripts/probe_head_pitch.py`: the command swept on
    # the shipped walker, and the blind radius read off the REAL `Detector` on
    # a real composed `World` at each pose). Three facts, none of which was
    # known when 0.6 was chosen:
    #
    # 1. WHERE THE COMMAND SATURATES. Standing, the head-pitch slot buys
    #    0.79 rad of camera depression per unit of command, linearly, until
    #    cmd 1.25 — where the `head_pitch` JOINT reaches its +1.571 MJCF
    #    limit and the camera stops at 1.17 rad (67°). Past that the command
    #    does nothing. So the limit is the JOINT, not the policy (the walker
    #    tracks the command all the way to it and stays upright), and 0.6
    #    (0.65 rad, 37°) was half of what is there.
    # 2. WHAT EACH POSE SEES. Nearest floor ball the detector still reports,
    #    standing, ground distance trunk→ball (and the far edge, which is the
    #    price — pitching down trades the horizon for the feet):
    #        level        0.37 m … 0.90+   ("the level camera loses a floor
    #                                        ball inside ~0.3 m" — measured
    #                                        at 0.37, or 0.35 of the slant
    #                                        `range_est` the brain compares
    #                                        `refresh_min` against)
    #        head 0.6     0.18 m … 0.77
    #        head 0.9     0.12 m … 0.36
    #        head 1.25    0.08 m … 0.21
    #    so the deep end is only worth asking for when the ball really is
    #    that close — which is exactly what `_gaze` decides, since the
    #    command it asks for is a function of the range.
    #
    #    AND THAT IS WHY RAISING `head_down` DOES NOTHING. Read off the
    #    brain that is running (3 seeds x 90 s of 2v2, 3656 gaze frames in
    #    `lineup`/`settle`), the gaze COMMAND is a median 0.259 and the
    #    camera depression it reaches a median 0.245 rad — 14°, with the
    #    90th percentile at 0.566. The clamp binds in under a tenth of the
    #    frames, because `_gaze` aims the axis AT the ball and most of a
    #    line-up happens at 0.3-0.6 m where that asks for a quarter of a
    #    radian. Raising the ceiling of a limit that is not being hit is
    #    not a change; measured, `head_down` 0.6 -> 1.0 moved the median
    #    depression only 0.245 -> 0.339 rad and no kick metric at all.
    # 3. THE NECK SLOT IS FREE AND THE HEAD SLOT IS NOT. `head_pose_cmd[0]`
    #    is `neck_pitch` and this brain never commanded it: `Chase.step`
    #    emitted `(0.0, gaze, 0.0, 0.0)`. Swept (walking at 0.30, 4 headings,
    #    steady over seconds 2-6), depression is additive and linear in the
    #    two slots — +head looks DOWN at 0.79 rad/unit standing, +neck looks
    #    UP, so a downward gaze is a NEGATIVE neck command, worth 0.43
    #    rad/unit — and the same depression costs completely different
    #    amounts of forward speed:
    #        59°  head +1.00 alone      -19.6 %      neck -0.30 head +0.60   +4.4 %
    #        69°  head +1.25 alone      -28.2 %      neck -0.60 head +0.60   -4.9 %
    #        72°  (not reachable)                    neck -0.40 head +0.80   -1.1 %
    #    `gaze_neck` is the fraction of the gaze command mirrored onto the
    #    neck slot, and `neck_gain` its rad-per-command, so `_gaze` divides
    #    the wanted depression by what the two slots together deliver. At 0
    #    it is bit-for-bit the old single-slot law.
    head_down: float = 0.6
    head_range: float = 0.9
    head_gain: float = 0.75        # camera rad per unit of head-pitch command (standing; measured 0.789)
    neck_gain: float = 0.43        # …and per unit of NECK command (measured 0.43, same sweep)
    # Fraction of the gaze routed to the neck slot (see 3 above). Ships at 0
    # because the gaze itself is off; if `gaze_still` is ever turned on it
    # must be 1, since the head slot alone raises the whiff rate 19.2% ->
    # 26.4% (p = 0.027, replicated) and the split does not.
    #
    # It also looks DEEPER for less command, because the neck's gain is much
    # closer to the head's while WALKING (0.78 against 0.93) than standing
    # (0.43 against 0.79), and the law above is calibrated standing. Read off
    # the running brain in `lineup`/`settle`: the shipped gaze reaches a
    # median 0.245 rad with a 90th percentile of 0.566; `head_down` 1.0
    # through the head slot reaches 0.339 / 0.637; the split at `head_down`
    # 0.64 reaches 0.309 / **0.812** on a median command of 0.199. The tail
    # is where a gaze earns its keep (it is the close-in frames), which is
    # why the split arm went blind for 0.14 s at the swing and the
    # head-slot arm for 1.20 s.
    gaze_neck: float = 0.0
    cam_level: float = 0.197
    cam_z: float = 0.21
    # Hold the gaze while the duck is STANDING STILL, instead of dropping it
    # the moment it stops. The gaze used to be gated on `vx > 0`, so the
    # settle in front of every swing was taken with the head level — and the
    # level camera cannot see a ball inside 0.37 m, which is precisely where
    # the ball is by then. Measured over 195 kicks (24 seeds x 300 s of 2v2,
    # `scripts/probe_gaze.py`): the head-pitch command at the swing is 0.000
    # at the median, the 3.5 s run-up is head-UP 55% of the time and standing
    # still 49%, the duck has not seen the ball for 1.48 s and 0.175 m of
    # walking when it fires, and 81% of the duck-steps spent with the ball
    # truly inside 0.40 m are steps in which the detector is reporting
    # nothing. That is the owner's "they walk around looking even when the
    # ball is under their feet", and "they keep scaling back up", as numbers.
    # STILL, NOT SLOW: a turn in place keeps the head level, because the
    # walker cannot turn in place with its head down (0.2 rad in 5 s against
    # 3.1 level — measured in tidy.py, and the reason the old gate existed).
    #
    # SHIPS OFF. It does exactly what it says and the kick does not care.
    # The mechanism, on the same 24 seeds x 300 s of 2v2 as the numbers
    # above (with `gaze_neck` = 1, `head_down` = 0.64): the run-up is
    # head-UP 55% -> 37% of the time, the duck has not seen the ball for
    # 1.48 s / 0.175 m at the swing -> **0.14 s / 0.006 m**, and kicks taken
    # having never seen the ball in the whole 3.5 s run-up fall from 16 of
    # 195 to 1 of 189. The blindness is real, it is a choice, and this
    # un-chooses it.
    #
    # And then it buys nothing. Over 48 PAIRED seeds (two blocks of 24, the
    # second fresh — `scripts/probe_kick_line.py`), against 360 baseline
    # kicks and 339 with the gaze held:
    #     on the sweet spot   0/360      ->  4/339 (1.2%, p = 0.055) — and
    #                                        all four are in the first block,
    #                                        none in the fresh one
    #     whiffed             69/360 19.2% ->  64/339 18.9%  (p = 1.00)
    #     ball ahead / spot-to-ball / plan age / drift: all flat
    # The play ledger over 24 seeds of 2v2 agrees: goals, falls, possession,
    # signed progress, spread, crowd and depth all flat, with `ballAdvance`
    # +0.115 (p = 0.048) while signed `ballProgress` is -0.003 (p = 0.98) —
    # churn, which is exactly what rule 5 says advance measures on its own.
    #
    # WORSE THROUGH THE HEAD SLOT ALONE, and this one replicates: at
    # `gaze_neck` = 0 the whiff rate goes 19.2% -> **26.4%** (84 of 318,
    # p = 0.027), in BOTH blocks (27.9%, 25.1%). With the neck carrying the
    # gaze the cost disappears (18.9%). That is the bench measurement
    # showing up in play — the head slot costs 12-28% of forward speed at
    # these depressions and the split costs about nothing — and it is the
    # reason `gaze_neck` exists at all.
    #
    # WHY IT CANNOT WIN, measured: on the kick spot the ball is 37° off the
    # nose and the camera's horizontal HALF-field is 31°. The last
    # centimetres of a line-up are unseeable at ANY pitch, and what a held
    # gaze recovers is the run-in, which the spot has already been planned
    # from. The lever that does move the placement is `refresh_min` — see
    # its note.
    gaze_still: bool = False
    # …and yaw the head at it too while standing. The pitch alone cannot
    # reach the endpoint: on the kick spot the ball is 0.08 m ahead and
    # 0.06 m to the kicking foot's side, which is 37° off the nose, and the
    # camera's horizontal HALF-field is 31°. A yaw would cover it (the walker
    # tracks a head-yaw command to 1.42 rad, and standing there is no forward
    # speed to lose) — but the ToF sits on the HEAD, so a yawed head points
    # the bumper sideways, which is exactly how `look_aim` was measured off
    # ("the brain stops for what it then sees").
    #
    # TWO THINGS TO KNOW BEFORE TRYING IT.
    #
    # 1. It is a DEAD KNOB ALONE, the same way `head_yaw_when="always"` was
    #    dead without `predict_s` (roadmap 4e). The yaw is only ever non-zero
    #    inside the `gaze_still` branch, and `_gaze_range`, the only thing
    #    this widens, is only called from there. Measured, not read: three
    #    seeds x 40 s of 2v2, 24 000 duck-ticks, 4 157 of them in
    #    lineup/settle where it would apply — **0 ticks differ** with it on.
    #    Turning it on with `gaze_still` off measures nothing. The arm is
    #    `gaze_still=1, gaze_neck=1, gaze_yaw=1`: held, carried on the neck
    #    so it does not cost forward speed, and able to reach the endpoint.
    #
    # 2. The ToF risk it was parked on now has a fix. `yaw_clear` gates this
    #    yaw too, on the same signal, and head-yaw tracking measured FREE
    #    once gated (roadmap 4e). That is what makes the arm worth running:
    #    `gaze_still` was judged with the hazard still in it.
    #
    # MEASURED OFF, 2026-09-06 — no longer "unknown". Three arms x 24 seeds
    # x 300 s of 2v2 on `scripts/probe_kick_line.py` (`runs/gazeyaw/`),
    # against the shipped brain's 178 kicks / 23.0% whiffs / 1.7% on-spot:
    #
    #   gaze_still+gaze_neck      204 kicks  23.5% whiff (p=0.91)  0.5% on-spot
    #   +gaze_yaw                 211 kicks  19.9% whiff (p=0.45)  0.5% on-spot
    #
    # It reaches the endpoint and it does not help. Nothing it was meant to
    # fix moves: on-spot does not rise, spot-to-ball gets WORSE (0.285 ->
    # 0.307 m), plan age is flat (3.03 -> 2.96 s), and paired per seed the
    # absolute aim error is flat too (p=0.93). What it does move is the
    # SYSTEMATIC aim: +12.4 deg, 95% CI [+6.2, +18.6], against a shipped
    # brain whose interval spans zero.
    #
    # WHY, and this is the useful part. The bias looks mechanical — pooled
    # over 462 kicks the aim error tracks the head yaw held at the swing
    # (r = +0.34, p = 5e-15; past +0.40 rad the mean error is +40.7 deg).
    # It is not. Control for where the ball actually was and the head-yaw
    # term collapses to +0.03 deg per deg (p = 0.66) while the ball's SIDE
    # offset carries everything: **+1.90 deg of aim error per cm**, t = 21,
    # R^2 = 0.56. The head yaw was a proxy for a ball off to the side.
    # `gaze_yaw` widens the bearing the gaze will accept, so it lets a duck
    # swing at balls it should not have swung at. Seeing the endpoint was
    # never the problem; standing in the right place is.
    gaze_yaw: bool = False
    # After a kick the ball is ahead and low: stand and look down `look_s`
    # before searching (measured: a 9 s search spin with the ball 0.17 m
    # ahead). A search dips the head every `search_dip_every`.
    look_s: float = 0.8
    look_range: float = 0.3
    # The kick map says where the ball goes BEFORE it moves: +21.6 deg for
    # the left foot, -11 for the right, off the body heading. `look_aim`
    # yaws the look after a kick to that angle, at `look_aim_range` (near
    # the horizon: the ball is a metre or two out by then) instead of the
    # 0.3 m dip - so the ball, which leaves the level camera at once
    # 30-55 deg off the nose, stays in the frustum long enough to track.
    # Measured OFF (8 seeds x 300 s of 1v1, against 2.38 goals / 7.4 kicks /
    # 0.38 falls a run): 2.25 / 6.4 / 0.50; with the gaze on the track while
    # searching (predict_s 3) 1.88 / 4.6 / 0.25; with the search sweep
    # 1.88 / 6.8 / 0.38. A head turned off the walking line leaves the ToF
    # bumper looking sideways, and the brain stops for what it then sees.
    look_aim: bool = False
    look_aim_range: float = 1.5
    # Where the ball ACTUALLY leaves, relative to the body heading — measured
    # in play rather than on a bench: 237 kicks over 24 seeds x 300 s of 1v1
    # and 2v2 (`scripts/probe_kick_line.py`), taking each kick's line from the
    # ball's travel over the next CARRY_S.
    #
    #     left foot    +23.6 deg   95% CI [+13.1, +34.1]   (bench said +21.6)
    #     right foot   -28.7 deg   95% CI [-33.8, -23.6]   (bench said -11.0)
    #
    # The bench was right about the left foot and 18 deg wrong about the
    # right, which is worth knowing: the bench swept a ball across a STANDING
    # duck's foot at the sweet spot, and in play the ball is 2-3 cm off it.
    # These are the in-play numbers.
    kick_exit_left: float = math.radians(23.6)
    kick_exit_right: float = math.radians(-28.7)
    # After a kick, hunt along where the ball REALLY went (`u` plus the exit
    # angle above) instead of along the line the kick was aimed at, and
    # publish THAT line to the team board. Pure knowledge: it changes where
    # the duck looks and what it tells its teammates, never how it stands.
    #
    # MEASURED NEUTRAL, and shipping on anyway — with the reason stated so
    # nobody mistakes it for a win. 24 paired seeds x 300 s of 2v2 and then
    # 24 fresh ones, exit line against aim line, nothing resolves on either
    # block or pooled over the 48: goals +0.125 (p = 0.49), falls -0.083
    # (p = 0.77), possession -0.514 (p = 0.32), ballAdvance +0.023
    # (p = 0.48), signed progress -0.026 (p = 0.54), crowd -0.003 (p = 0.64),
    # back-kicks 34% -> 37% (p = 0.45). It is a real null and not a dead
    # path — the arms differ seed by seed (68 falls against 57 on one block,
    # 57 against 64 on the other), which is what rule 0 asks you to check.
    #
    # It ships on because the alternative is knowingly hunting along a line
    # the ball does not take: the hunt is a fallback that fires only when a
    # kicked ball is lost, and the search behind it finds the ball anyway,
    # so being right costs nothing and buys nothing measurable. No
    # performance claim is made for it.
    #
    # The other way of using the same measurement — rotating the STANCE so the
    # kick flies along `u` (`kick_deflect_*`) — is refuted, and this time with
    # the mechanism. Set to the measured values, over 24 paired seeds of 2v2:
    # goals 2.08 -> 1.38 (p = 0.045), ballAdvance 0.839 -> 0.655 (p = 0.023),
    # kicks 151 -> 78, and the aim error it was supposed to remove got WORSE,
    # 45.4 -> 67.3 deg mean absolute. The reason is in the map's own shape:
    # the deflection is a function of where the ball sits relative to the
    # foot (15 deg/cm near 2 cm, 4.5 deg/cm at 4-8 cm), so rotating the stance
    # moves the ball to a different part of that function and produces a
    # DIFFERENT deflection — the right foot's went from -27.3 to -48.6 deg off
    # the body. A fixed rotation cannot cancel an offset that its own rotation
    # changes. The lever stays what the map said first: line-up precision.
    hunt_exit: bool = True
    # Measured: a kick leaves at ~1.4 m/s. Published to the team board as
    # the ball's velocity; below `Team.vel_use` (0.7) a coasting track is
    # ignored so this does not re-open intercept-on-claim.
    kick_speed: float = 1.4
    # A searching head sweeps +-`search_sweep` rad (period `search_sweep_s`)
    # while the body circles, when it has no track to look at.
    #
    # RE-SCREENED 2026-09-05 AND ITS OLD VERDICT WAS STALE. It shipped off on
    # "a searching head sweep makes the body turn MORE, 5/5 seeds", and the
    # mechanism recorded beside that was the ToF: the sensor is on the HEAD,
    # so a turned head reported walls that were not ahead and the brain
    # stopped for them. The clearance rule has since moved from sensor
    # COLUMNS to BEARINGS ("Chase: clearance by bearing, not by sensor
    # column"), and that coupling is now dead - measured by recomputing both
    # rules on the same 1 439 904 frames and binning by head yaw: past
    # 0.70 rad the old rule stops on 13.8% of frames and the shipped one on
    # 0.3%.
    #
    # So the arm was re-run on 24 seeds x 300 s of 2v2, and every field is
    # inside the noise: spinFrac +0.007 (p = 0.55), falls 57 -> 56
    # (p = 0.93), blocked seconds -0.18 (p = 0.78), possession +0.93
    # (p = 0.45), ball-in-view -0.005 (p = 0.71). It is a CLEAN NULL now,
    # not a knob with a reason - free to leave off, and free to turn on if
    # something else wants a swept head.
    #
    # (The bearing rule did not remove the coupling so much as make it
    # honest: past 0.70 rad the duck now has essentially no forward obstacle
    # sense at all rather than a wrong one. That is where the head-tracking
    # arm's falls come from - see `head_yaw_when`.)
    search_sweep: float = 0.0
    search_sweep_s: float = 4.0
    # Hunt: a ball lost right after a kick, or after being walked into
    # inside `hunt_lost_range` (the histograms: half a run is search, and
    # the ball leaves the view rolling off along a known line - the kick's,
    # or the duck's own heading), is looked for by WALKING that line for
    # `hunt_s` with the head level (a floor ball is in view from 0.3 m out
    # to the camera's range) before the standing search begins.
    # ON, with its stops (below). Without them it walked into things (1.50
    # falls a run); with them, over 8 seeds x 300 s of 1v1: 8.6 kicks a
    # run against 8.4, 0.12 falls against 0.50, goals within the noise of
    # eight seeds (1.50 against 2.00, 0.25 kicked against 0.75).
    hunt_s: float = 3.0
    hunt_lost_range: float = 0.6
    # The hunt's own stops (traced: it walked at 0.45 m/s turning at full
    # rate into the boards, where the ToF returns nothing inside 3 cm, and
    # into the other duck beside it, outside the camera's cone and the
    # ToF's middle columns). Slower, a capped turn, and it ends - not
    # alternates with "blocked" - when anything is inside `hunt_stop`
    # ahead, a duck track is beside it, or the boards are `hunt_margin`
    # ahead in odometry.
    hunt_speed: float = 0.3
    hunt_wz: float = 0.5
    hunt_stop: float = 0.45
    hunt_margin: float = 0.35
    # A ball memory in odometry: the last sighting, the end of a hunted
    # line, the centre spot at a kickoff. A search with a memory further
    # than `seek_min` away WALKS there first (the hunt's speed and stops)
    # instead of circling on the spot - with the hunt and the circle,
    # 107 s of a 300 s run were still search. Forgotten after `seek_s` or
    # once there with nothing seen. OFF (0): measured over 8 seeds x 300 s
    # of 1v1 at 2.38 goals, 7.8 kicks, 0.75 falls a run against 2.25 / 9.4
    # / 0.38 without - the goals did not move and the blind walks fell.
    seek_s: float = 0.0
    seek_min: float = 0.4
    seek_tol: float = 0.25
    # The ball's trajectory (tracker: an odometry-frame position and
    # velocity from consecutive hits). Measured: a kicked ball leaves at
    # 1.4 m/s and slows at 0.04 m/s^2 on this floor - it rolls to the
    # boards - and leaves the level camera at once, 30-55 deg off the
    # nose, so the track coasted with a stale range for two seconds and a
    # new track was born when it was found again. With `predict_s` > 0
    # the head YAWS toward the predicted bearing (`head_yaw_gain`, to
    # `head_yaw_max`; always, or only while searching / looking), and
    # with `predict_steer` the search opens toward the predicted side and
    # the hunt walks to the predicted point (clamped to the pitch).
    # The head half SHIPS ON (`predict_s` 1.0 + `head_yaw_when` "always" +
    # `yaw_clear` 0.45); the STEERING half stays off. That split is measured,
    # and it reverses an earlier reading taken on 8 seeds of 1v1 judged
    # mostly on goals - a metric that needs 136 seeds to move (Track 4.1.5),
    # so the old table below could not have seen this either way. Keep it as
    # the record of what the steering costs:
    #   8 seeds x 300 s of 1v1, against 2.25 goals / 9.4 kicks / 0.38 falls
    #   with everything off - yaw always + steer 1.12 / 10.5 / 1.62; yaw off
    #   + steer 2.12 / 9.1 / 1.12; yaw in search + steer 2.12 / 10.0 / 0.75;
    #   yaw in search, no steer 2.12 / 6.5 / 0.75.
    # Every arm there that walks a predicted line loses falls, and the
    # steering is what walks blind lines into things, so `predict_steer`
    # stays off. The head does not walk anywhere. Re-measured properly on
    # 48 paired seeds of 2v2 (24 discovery + 24 fresh, agreeing): the head
    # bundle buys +8.0 points of ball-in-view (p<0.0001, 42/48 seeds) and
    # cuts the median time the ball is lost by 0.27 s (p=0.0002), for no
    # measured cost in falls, kicks, possession, spread, crowd, depth or
    # ball progress. The gate is what makes it free - see `yaw_clear`.
    ball_decel: float = 0.04
    predict_s: float = 1.0         # how long a prediction is worth acting on after the last hit (0: off)
    head_yaw_gain: float = 0.9
    head_yaw_max: float = 1.4          # the walker's trained head-yaw range (upstream curriculum: +-1.40 rad)
    # Only let the head leave the walking line while the ToF says the line is
    # empty: a look target is dropped when the forward clearance is inside
    # this (`hunt_stop` = 0.45 is the natural scale; 0 = off).
    #
    # This exists because head-yaw ball tracking was a confirmed TRADE: on 48
    # paired seeds it bought +7.8 points of ball-in-view and cost +1.54 falls
    # a run (p<0.0001 both ways). The ToF is ON THE HEAD, so yawing it points
    # the bumper off the walking line, and since the clearance rule became
    # bearing-based it reports `+inf` honestly rather than a false wall - so a
    # yawed duck walks with no forward obstacle sense at all. The brain had
    # that signal and never consulted it before turning the head.
    #
    # Capping the yaw's MAGNITUDE was tried first and removes both halves
    # together, because the visibility and the falls live in the same frames
    # by magnitude. Clearance separates them. Measured over all 48 seeds,
    # counting frames where the head is past 0.35 rad (blind) and the old
    # column rule - which does not care where the head points - says
    # something really is ahead:
    #
    #                      blind frames   of which, obstacle ahead   per 1000
    #   head tracking off          0.0%                          -        0.0
    #   tracking, ungated         16.7%                      13.0%       21.8
    #   tracking, gated 0.45      14.1%                       8.2%       11.6
    #
    # The gate drops total blind frames by only 16% but the DANGEROUS ones by
    # 47%: it is selective, which is exactly why the cap was not the lever.
    # Against the same bundle ungated it removes 1.60 falls a run (p<0.0001,
    # worse on only 8 of 48 seeds) while giving up no visibility at all
    # (+0.002, p=0.80). Both blocks agree. What is LEFT: 8.2% of blind frames
    # still have something ahead, because the gate only consults the clearance
    # the head can currently see - it cannot know about what it has already
    # turned away from.
    yaw_clear: float = 0.45
    head_yaw_when: str = "always"  # or "search": yaw the head only while searching / looking
    predict_steer: bool = False    # the hunt bends and the search opens toward the prediction
    search_dip_every: float = 1.5
    search_dip_s: float = 0.6
    dip_range: float = 0.22
    # The search is a slow WALKING circle (`search_vx` forward with the
    # turn), not a turn in place: instrumented over 300 s, during search
    # the ball was inside the camera's frustum 1% of the time and detected
    # 0% - it sat 90-120 degrees off the nose, and a standing turn barely
    # turns the walker (the cold-turn kick fires once, the next dip stands
    # it still again). Walking, the body rotates and the camera sweeps.
    search_vx: float = TURN_KICK
    # Turning toward the side the ball was last on (a right turn when it
    # went right) probed 4 s against 10 s for a ball to the right, and
    # measured OFF over 8 seeds: 1.62 goals, 8.9 kicks, 0.62 falls a run
    # against 2.25 / 9.4 / 0.38 always turning left - within the noise of
    # eight seeds, with the falls on the walker's weak right turn.
    search_sided: bool = False
    # Dribbling: OFF (inf). Measured — a ball pushed at 0.3 m/s for half a
    # second rolls on at about the walking speed on this floor and the duck
    # walks behind it without ever lining up; the kick wins. Re-measured
    # at 1.4 with goals attributed (8 seeds x 300 s): 2.25 goals a run
    # either way (0.25 kicked, 2.00 bumped), 6.4 kicks and 1.8 pushes
    # against 9.4 kicks, and 0.75 falls against 0.38 - the deliberate
    # bump scores no more than the accidental one and falls twice as often.
    push_beyond: float = math.inf
    push_behind: float = 0.16
    push_speed: float = 0.3
    push_s: float = 0.5
    # The other duck's BODY (measured over 4 traced runs: 5 of 7 falls had the
    # other duck 3–9 cm away and this one turning in place — search, blocked
    # or lining up — the walker tips over when it turns against a body it
    # cannot see below its ToF rows). A tracked duck inside `duck_keepout`
    # and ahead: nothing walks or turns toward it; inside `duck_touch` it is
    # against us: stand until it moves.
    duck_keepout: float = 0.4
    duck_touch: float = 0.22
    duck_bearing: float = 1.2      # rad off the nose that counts as "ahead"
    # Standing against something (avoid, blocked) longer than `stuck_s`:
    # two ducks meeting at the ball otherwise stand and wait for each
    # other (traced: 8 s nose to nose). Retreat: turn toward the freer
    # side, then walk clear.
    stuck_s: float = 1.5
    retreat_turn_s: float = 1.0
    retreat_walk_s: float = 1.2
    # Team play (brain/team.py): a supporter stands `support_back` from the
    # ball toward its own goal, `support_side` to the side per rank, facing
    # the ball, and never inside `support_min` of it.
    support_back: float = 0.7
    # Where a supporter stands relative to the ball: "back" (toward our own
    # goal - it defends, and 3v3 scores 0.75 goals a run) or "ahead" (toward
    # the goal we attack: a poacher, in position to walk a loose ball in,
    # which is how most goals are actually scored here).
    support_mode: str = "back"
    support_side: float = 0.45
    support_min: float = 0.45
    # Traced over 3 seeds x 300 s of 3v3: 10 of 14 falls were supporters
    # turning in place with a teammate 5-28 cm away or against the boards
    # - a body beside the duck is outside the camera's 62 deg and the ToF's
    # 45 deg, so neither the avoid rule nor the wall rule saw it. Two
    # answers: the support spot stays `support_margin` inside the pitch
    # (`bounds`, from make_pitch), and a supporter with any duck track
    # inside `beside_m` (however stale within `beside_s`: the track's
    # bearing turns with the body, so a duck seen a second ago still says
    # where it is) stands instead of turning in place.
    support_margin: float = 0.35
    # Where a duck with a static ROLE stands when it is not the one on the
    # ball (roadmap Track 4.3). All three are a spot to hold, not a new state
    # machine: the same `_support` servo walks to them and faces the ball.
    #   defender  — on the line from the ball to its own goal, this far out
    #               from the goal line, and never over the halfway line. Its
    #               job is to be BETWEEN, which is the one thing the shipped
    #               roster never does: the deepest duck of a 2v2 averages
    #               1.35 m up a 1.7 m half.
    #   striker   — ahead of the ball toward the goal it attacks, offset to
    #               the side the ball is NOT on. The offset is the whole
    #               difference from the poacher that was measured off (that
    #               one stood ON the line and reversed on fresh seeds), and
    #               it is what keeps a striker from being a second duck on
    #               the ball.
    #   mid       — between the ball and the centre spot, on the ball's side,
    #               kept inside the middle third.
    defend_depth: float = 0.5
    strike_ahead: float = 0.8
    strike_side: float = 0.4
    mid_side: float = 0.5
    beside_m: float = 0.3
    beside_s: float = 1.5
    # Use the colour classifier to tell a teammate from an opponent
    # (roadmap Track 4.4.2): with it on, a duck gives a STRANGER
    # `opp_keepout` of room and keeps the standard `duck_keepout` for a
    # teammate, since the team board already coordinates teammates and
    # nothing coordinates an opponent.
    #
    # SHIPS OFF, MEASURED. At 0.55 m on 3v3 it looked like the crowding fix
    # the 13-fall trace asked for — crowd 26.5% -> 19.7%, p = 0.002 over 24
    # seeds — and then did NOT replicate: 21.4% -> 20.5%, p = 0.593 on 24
    # fresh ones, and better on 25 of 48 pooled, which is a coin. Falls
    # (181 -> 160, p = 0.27) and everything else are unresolved; the only
    # replicated effect is the cost, possession -2.16 s/min (p = 0.032).
    # The same shape as the poacher and the bump-stand rule, caught by the
    # same rule: confirm on seeds the effect was not found on.
    #
    # The SENSE is not what failed, and it stays: a colour-aware rule with a
    # better idea than "stand further off" can use it. For scale, the static
    # roles (Track 4.3) take the same metric from 22.4% to 3.5% and DID
    # replicate — standing somewhere useful beats standing further away.
    use_color: bool = False
    opp_keepout: float = 0.0       # an opponent this near and ahead: treat it as a duck to avoid
    # The ToF sees the ball at the feet (tof_floor_ball): inside `tof_ball_m`
    # with the head dipped, a floor blob feeds the tracker as a ball sighting
    # when the camera has none - the level camera loses a floor ball inside
    # 0.3 m, which is where the line-up and the kick live. Measured OFF at
    # 0.5 m (8 seeds x 300 s of 1v1): 1.62 goals, 8.1 kicks, 0.75 falls a
    # run against 2.38 / 7.4 / 0.38 - a blob at the feet is as often the
    # other duck's foot as the ball, and a line-up on a foot is a fall.
    #
    # RE-OPENED at the replacement module's real 60 deg vertical FOV (the
    # original was measured at 48) and it closes harder. 1v1 goals are a
    # weak instrument, so this counted the blob's own EVENTS instead - 6
    # seeds x 180 s, every tick the blob fired, against the ball's true
    # position:
    #
    #   V FOV   blob ticks   camera already had the ball   blind-case ticks   of those, the ball
    #    48 deg      4785                   87.5%                  599              30.1%
    #    60 deg      5809                   93.6%                  374              36.9%
    #
    # Two independent reasons it stays off, both worse at 60 than at 48:
    #  1. It is almost always REDUNDANT - the camera already has the ball on
    #     88-94% of the ticks the blob fires.
    #  2. In the case it exists for (camera blind) it is WRONG about two
    #     times in three, and that is the case that ends in a line-up on a
    #     foot.
    # The wider lens does raise blind-case precision (30 -> 37%) but cuts
    # the opportunity by 38% (599 -> 374 ticks), because it reaches 5.5 cm
    # further into the blind zone itself (blind radius 28.5 -> 23.0 cm,
    # docs/camera-hardware.md 3d). Better optics shrink this feature's job
    # faster than they improve it.
    tof_ball_m: float = 0.0
    # A bump (Senses.bumped: the body is touching another body - contacts in
    # the sim, the IMU / servo loads on the robot): no turn in place for
    # `bump_stand_s` after the contact STARTED. 12 of 13 traced 3v3 falls
    # were standing turns beside an unseen opponent. MEASURED against no
    # rule at all, and THE FALL REDUCTION DID NOT REPLICATE:
    #   seeds 24-35        falls 4.83 -> 3.25   -1.58 +/- 0.92  p = 0.14
    #   seeds 24-35 again  falls 6.17 -> 4.00   -2.17 +/- 1.01  p = 0.060
    #   seeds 200-211      falls 4.08 -> 4.33   +0.25 +/- 1.04  p = 0.88
    #   all 24 DISTINCT layouts             -0.81 +/- 0.69  p = 0.264
    # The first two are the same twelve layouts measured twice (per layout
    # -1.88 +/- 0.84, p = 0.055; pooling them as 24 says p = 0.012, which
    # is repeated measures, not replication, and is what this comment said
    # first). On twelve layouts nobody had run the effect is absent and
    # slightly reversed. "A third fewer falls" is WITHDRAWN - the
    # poacher's shape exactly, caught by AGENTS.md's third rule.
    #
    # It stays on for rosters anyway, as a default nobody has earned in
    # either direction: the pooled point estimate still favours it
    # (better on 15/24) and nothing it was suspected of costing moved
    # (kicks +0.83 p = 0.51, goals -0.58 p = 0.50, advance and progress
    # flat on the fresh block), so flipping it off would be reading noise
    # the other way. Falls want ~376 seeds for a 25% shift; this is 24.
    # `bump_back` below is the arm worth measuring against it next. Its first form
    # cost 1v1 1.50 goals and 1.00 falls against 2.38 / 0.38, and a trace
    # of 838 bumps said why, refuting the obvious guess on the way:
    #   * NOT possession. The feet meet a median 0.66 m from the ball; both
    #     ducks are inside 0.35 m of it in 18% of bumps; and two seconds
    #     later the ball is further from BOTH ducks by the same +0.074 m.
    #     Nobody is walked over - so the "exempt the duck at the ball"
    #     knob this once carried is GONE, not merely defaulted off: a knob
    #     on a premise the data refuted only invites someone to try it.
    #   * It cancelled the ESCAPE. 70% of its firing was in `blocked`, a
    #     state that is 12.6% of the run, where the walk is already zeroed
    #     and the turn is the only command left. 6 of 8 falls were a stand
    #     pressed against the other duck: the walker leans on it.
    #   * It fed itself. Standing on a body keeps touching it, which
    #     refreshed the timer: bumps went 44 -> 105 a run and one freeze
    #     ran 74 s. So the window is edge-triggered now (`bump_gap_s`).
    # 0 for a lone attacker; a roster with teammates gets
    # `team_bump_stand_s` through brain/team.py's brain_kwargs.
    bump_stand_s: float = 0.0
    team_bump_stand_s: float = 0.5
    # Only where a standing turn beside a body is the danger. Never in
    # `blocked` / `avoid` / `retreat` (the turn IS the escape) and never in
    # `search` (its circle WALKS at `search_vx`, and freezing that stops the
    # one behaviour that finds the ball - 5 of 8 traced falls were there).
    # ("block" is inert unless `intercept_eta` is on, and it belongs here for
    # the same reason `support` does: standing on the line is a turn in place
    # with a body possibly beside it.)
    bump_stand_states: tuple[str, ...] = ("support", "lineup", "settle", "turn", "block")
    # A contact episode ends after this long without one; the freeze runs
    # from its onset and is never extended by staying in contact.
    bump_gap_s: float = 1.0
    # Back up instead of standing, for this long. UNTRIED, ships at 0.
    #
    # Standing is what the rule does today, and standing does not END the
    # contact: measured from 0.10 m of separation, 16 trials, a standing
    # duck was still at 0.099 m four seconds later and cleared 0.30 m in
    # 0 of 16. A straight reverse cleared it in a median 1.6 s (14/16),
    # beating turn-90-and-walk (2.7 s) and turn-180-and-walk (3.2 s) - and
    # unlike either it keeps the ball in frame, so no `search` follows.
    # The walker reverses at 0.23 m/s, faster than it walks forwards; the
    # "it cannot" above this was a dead-band reading (see `gait.back_up`).
    #
    # Why it is the obvious next thing to measure here: the two failure
    # modes the trace found are both a duck that cannot separate. "6 of 8
    # falls were a stand pressed against the other duck: the walker leans
    # on it" is a stand that had somewhere to go. "Standing on a body keeps
    # touching it - 44 -> 105 bumps a run" is the same. And the states this
    # rule is kept OUT of are excluded because "the turn IS the escape" -
    # which was true only while a reverse was believed impossible.
    #
    # MEASURED, and it is the most closed null in this file. Three arms on
    # the same twelve fresh layouts (3v3, seeds 200-211, 300 s a seed):
    #
    #            falls  kicks  goals   possession  advance
    #   no rule   4.08   6.50   2.17     11.83      0.40
    #   stand     4.33   7.33   1.58     13.00      0.42
    #   back      4.33   6.67   2.00     11.97      0.44
    #
    # stand -> back on falls is EXACTLY 0.00 +/- 1.13, p = 1.000 (52 events
    # against 52); nothing else resolves either. And the knob is NOT inert:
    # instrumented over one 3v3 run it issues 690 reverse commands (13.8 s
    # a run) and cuts the ticks spent touching another body from 4473 to
    # 2529 of 90000 - a 43% drop, exactly the self-feeding the 838-bump
    # trace found ("standing on a body keeps touching it").
    #
    # So: the mechanism is real, it operates, and it does not matter. Time
    # in contact is not what makes a 3v3 duck fall. That agrees with the
    # probe it was built on - a turn beside a STATIC body fell 0 times in
    # 98 trials down to 8 cm; the falls need a duck that is MOVING into
    # you - and it means the remaining lever is the closing duck, not the
    # contact. Ships at 0, as a measured null rather than an untried idea.
    bump_back: float = 0.0
    # Teammates' poses off the team board (brain/team.py): a teammate
    # inside `mate_keepout` counts as a duck beside me (no turn in place,
    # no hunt) and, ahead, as a duck to avoid - the camera and the ToF
    # cannot see one beside or behind me. Measured OFF (3v3, 4 seeds x
    # 300 s: 1.50 goals, 4.5 kicks, 5.25 falls a run with it at 0.4
    # against 1.50 / 5.0 / 4.50 without): a fresh trace put 12 of 13
    # falls beside an OPPONENT, which no board carries, turning in place.
    mate_keepout: float = 0.0
    # A supporter at its spot turns to face the ball WALKING (`support_turn_vx`
    # > 0, like the search circle) instead of standing - the traced 3v3
    # falls were standing turns beside a body nothing had seen. Measured
    # OFF at 0.2 (4 seeds x 300 s): 2v2 1.50 goals / 8.8 kicks / 4.00
    # falls a run against 1.50 / 9.8 / 3.50; 3v3 1.00 / 4.2 / 5.25
    # against 1.50 / 5.0 / 4.50 - the walking turn bumps what it cannot see.
    support_turn_vx: float = 0.0
    # Yielding to a duck that clearly has the ball: OFF by default. Measured
    # over 8 seeds × 300 s: off 1.50 goals / 8.5 kicks / 2.12 falls a run,
    # on (0.5 m) 1.12 / 7.0 / 2.12 — it costs play and saves nothing.
    yield_range: float = 0.0
    yield_ratio: float = 0.7
    yield_s: float = 1.5
    yield_cooldown_s: float = 3.0
    # Blocking a ball that is rolling into our own goal (brain/intercept.py):
    # leave the play, walk onto the ball-to-goal line ahead of it, and let the
    # body stop it. The repo owner's idea, watching a 2v2 — "come in from the
    # side and deflect it" rather than line up a kick on a ball that is
    # already past.
    #
    # `intercept_eta` is the master knob and 0 is OFF: a ball predicted to
    # reach our own goal within this many seconds is worth leaving the play
    # for. THE BASELINE it is aimed at (`scripts/probe_threat.py`, 48 seeds
    # x 300 s of 2v2, `runs/thr-base48b.jsonl`): 54 of 70 threats conceded,
    # with the best-placed defender playing on (support 24%, line-up 21%,
    # chase 13%) or searching (18%) through them, and a defender 0.15-0.50 m
    # off the ball's path conceding 31 of 31.
    #
    # SHIPS OFF: IT DOES WHAT IT SAYS AND THE THING IT WAS BUILT TO MOVE DOES
    # NOT MOVE. Measured at 8.0 over 48 paired seeds x 300 s of 2v2 and then
    # 48 FRESH ones (seeds 100-147), plus a 24-seed ledger:
    #
    #   * It engages, and exactly as designed: through a threat the
    #     best-placed defender is in `block` 21% of the ticks, and `lineup`
    #     falls 21% -> 6% and `chase` 13% -> 5%. It is not a dead path.
    #   * The conceded fraction does NOT resolve: 108/137 (79%) -> 102/138
    #     (74%) pooled over the 96 seeds, z = -0.96, p = 0.34 (77->73% and
    #     81->75% on the two blocks; the direction is consistent and the
    #     size is not).
    #   * What DOES replicate is where the ball spends its time: seconds a
    #     minute inside 0.9 m of some mouth, 15.80 -> 14.26, -1.53 +/- 0.60,
    #     p = 0.010 pooled, better on 60 of 96 seeds, same sign on both
    #     blocks (p = 0.093 then 0.054). The 0.45 m clock did not
    #     (p = 0.016 -> 0.558).
    #   * The ledger is quiet: goals 39 -> 36 (p = 0.71), falls 57 -> 46
    #     (p = 0.19), possession +1.5 s/min (p = 0.16) - the feared cost of
    #     abandoning the attack does not appear - advance, crowd, spread,
    #     depth and back-kicks all flat. The one big number, signed
    #     ballProgress -0.369 (p < 0.001), is attribution and not harm: a
    #     duck standing in front of a goalward-rolling ball BECOMES the
    #     possessing duck, and over 6 seeds 3804 of those ticks carry the
    #     ball toward that duck's own mouth at 4.65 m/min against 0.75 in
    #     every other state. `ballAdvance` - the forward half of the same
    #     accumulator, credited by the same rule - is flat, which a duck
    #     genuinely shoving the ball goalward could not manage.
    #   * The cost that is real: kicks 193 -> 149, a fifth of the touches.
    #
    # The ceiling was in the baseline all along, and it is the reason to
    # leave this off rather than tune it: 40 of the 70 threats are declared
    # with the ball ALREADY inside 0.3 m of the goal line (38 of those 40
    # conceded), and a block was geometrically available - perfect knowledge,
    # a generous walk model - in only 20 of the 54 conceded ones. The lever
    # is earlier than the block.
    intercept_eta: float = 0.0
    intercept_vmin: float = 0.12   # …closing this fast on our goal (m/s, a scalar rate, not a velocity)
    intercept_dt: float = 0.6      # …differenced over this window of SIGHTINGS (never a coasted track)
    intercept_age: float = 0.8     # …the newest of which is no older than this
    intercept_ahead: float = 0.25  # stand at least this far goal-side of the ball
    intercept_keep: float = 0.45   # …and no nearer than this to our own goal (below ahead+keep there is no block: see intercept.py)
    intercept_tol: float = 0.12    # on the line within this: stand and face the ball
    intercept_hold: float = 1.0    # keep blocking this long after the trigger drops (the estimate is jittery)
    intercept_clear: float = 0.0   # > 0: with the ball this near, sweep ACROSS the line instead of standing in it

    @staticmethod
    def env_names(spec: str | None = None) -> set[str]:
        """The knob NAMES a battery set through `MICRODUCK_CHASE`.

        A default cannot be told from a caller's explicit value by comparing
        them — `bump_stand_s=0` on the command line and the shipped 0.0 are
        the same number — so the roster default in `brain/team.py` asks which
        names were spoken rather than guessing from the values."""
        if spec is None:
            spec = os.environ.get("MICRODUCK_CHASE", "")
        return {item.partition("=")[0].strip() for item in spec.split(",") if item.strip()}

    @staticmethod
    def from_env(spec: str | None = None) -> "ChaseParams":
        """The defaults with `MICRODUCK_CHASE` applied — how a battery says
        which variant it is measuring:

            MICRODUCK_CHASE="two_stage=1,approach_speed=0.4" uv run eval-pitch …

        Every knob above ships on a measurement, and until now the only way
        to measure one was to edit its default, run, and edit it back — a
        step that is invisible in the battery's own record and was done
        wrong at least once. `--tag` says which variant a row belongs to;
        this says what the variant IS, from the same command line.

        An unknown name or an unreadable value RAISES: a typo that silently
        measured the default would be the expensive kind of mistake here.
        Tuple-valued knobs are not settable this way."""
        p = ChaseParams()
        if spec is None:
            spec = os.environ.get("MICRODUCK_CHASE", "")
        if not spec.strip():
            return p
        kinds = {f.name: getattr(p, f.name) for f in fields(p)}
        over: dict = {}
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            k, sep, v = item.partition("=")
            k, v = k.strip(), v.strip()
            if not sep or k not in kinds:
                raise ValueError(f"MICRODUCK_CHASE: {item!r} is not <ChaseParams field>=<value>")
            cur = kinds[k]
            if isinstance(cur, bool):
                if v.lower() not in ("0", "1", "true", "false", "on", "off"):
                    raise ValueError(f"MICRODUCK_CHASE: {k}={v!r} is not a boolean")
                over[k] = v.lower() in ("1", "true", "on")
            elif isinstance(cur, str):
                over[k] = v
            elif isinstance(cur, tuple):
                raise ValueError(f"MICRODUCK_CHASE: {k} is a tuple; set it in code")
            else:
                over[k] = float(v)
        return replace(p, **over)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Chase:
    """Walk at the nearest ball, line up behind it on the line to the goal,
    and KICK it with the shipped kick policy (roadmap soccer). Tracks the
    ball (brain/tracker.py) with the head pitched down on the way in; a
    floor ball leaves the camera ~0.2 m out, so the last leg is dead
    reckoning in odometry to a spot `kick_ahead` behind the ball and
    `kick_side` to the foot's side, then a 0.5 s kick window. Keeps off
    the other ducks and the walls, retreats when stood against something,
    and in a team (brain/team.py) takes the attacker's or a supporter's
    role. Searches turning left, dipping the head for a near ball."""

    kind = "chase"
    wants_head = True
    DET_MAX_AGE = 0.4
    TOF_MAX_AGE = 0.25

    def __init__(self, p: ChaseParams | None = None, goal: tuple[float, float] | None = None,
                 team=None, duck_id: str = "", bounds: tuple[float, float] | None = None,
                 goal_w: float = 0.0, role: str | None = None):
        # No params given (the lab, the benchmark, the /sim page): the
        # shipped defaults, with `MICRODUCK_CHASE` applied so a battery can
        # name its variant on the command line. A caller that passes `p`
        # (brain/team.py's roster kwargs, a test) is never overridden.
        self.p = ChaseParams.from_env() if p is None else p
        self.goal = None if goal is None else (float(goal[0]), float(goal[1]))
        self.goal_w = float(goal_w)        # the mouth's width: how wide a target the ball has (kick_cone)
        self.bounds = None if bounds is None else (float(bounds[0]), float(bounds[1]))   # the pitch's half-extents inside the boards
        self.team = team
        self.duck_id = duck_id
        # The STATIC role off the scenario ("defender" / "midfielder" /
        # "striker" / None), which is a different thing from `self.role` — the
        # dynamic attack/support the board hands out every tick. This one says
        # which third of the pitch this duck may take the ball on and where it
        # stands when it does not have it; that one says whether it has it now.
        self.job = role
        self.tracker = Tracker()
        self.gait = GaitWatch()
        self.blocker = Interceptor()
        self.reset()

    def reset(self) -> None:
        self.kicks = 0
        self.pushes = 0
        self.declines = 0          # swings refused by `kick_side_max`
        self.attack: float | None = None                            # heading of the goal it attacks (first odom yaw)
        self.kickoff()

    def kickoff(self) -> None:
        """Play restarts (a goal; World.kickoff put the duck back on its
        spawn): forget the ball, the spot and whatever manoeuvre was under
        way; keep the tally and the goal. On a pitch the ball is on the
        centre spot: remember that."""
        self.state = "search"
        self.role = "attack"
        self.last_bearing = 0.0
        self.last_seen_t: float | None = None
        self._senses: Senses | None = None
        self._mates: list[tuple[float, float]] = []          # (range, bearing) of live teammates, off the team board
        self._last_foot: str | None = None                   # the foot of the last kick (the look aims by it)
        self._bump_t = -1e9                                  # last contact
        self._bump_t0 = -1e9                                 # onset of the current contact episode
        self.last = (0.0, 0.0, 0.0)
        self.spot: tuple[float, float, str | None, float, str] | None = None   # x, y, foot, heading, "kick"|"push"
        self.lined = False                      # stage two of the line-up: on the line, walking straight in
        self.t_state = 0.0
        self._yield_t0 = -9.0
        self._yield_end = -9.0
        self._poses: list[tuple[float, float, float, float]] = []      # (t, x, y, yaw) over the stuck window
        self._retreat_t0 = -9.0
        self._retreat_sign = 1.0
        self._look_t0 = -9.0
        self._hunt_t0 = -9.0
        self._hunt_u: float | None = None           # the line to walk (odometry heading)
        self.memory: tuple[float, float, float] | None = None   # (x, y, t) where the ball was, odometry frame
        if self.goal is not None:
            self.memory = (0.0, 0.0, 0.0)           # a pitch: play starts from the centre spot
        self._last_range: float | None = None
        self._search_t0: float | None = None
        self._prev_skill = None
        self.tracker.reset()
        self.gait.reset()
        self.blocker.reset()

    def _gaze(self, rng: float) -> float:
        """The gaze COMMAND that puts a floor ball at `rng` on the camera's
        axis. With `gaze_neck` > 0 the same command drives both slots, so the
        divisor is what the two of them deliver together (measured additive,
        `head_gain` + `neck_gain` per unit); at 0 this is the old law."""
        p = self.p
        want = math.atan2(p.cam_z - 0.035, max(rng, 0.05))
        gain = p.head_gain + p.neck_gain * p.gaze_neck
        return float(np.clip((want - p.cam_level) / max(gain, 1e-6), 0.0, p.head_down))

    def _head_pose(self, cmd: float, yaw: float = 0.0) -> tuple[float, float, float, float]:
        """The 4-slot head command for a gaze of `cmd`. The neck looks UP on a
        positive command, so a downward gaze mirrors it negative. With both
        extras off the slots are plain 0.0 and not -0.0 — the old tuple, to
        the bit."""
        k = self.p.gaze_neck
        return (-k * cmd if k else 0.0, cmd, yaw, 0.0)

    def _gaze_range(self, odom, ball) -> tuple[float, float] | None:
        """(range, bearing) to aim the gaze at during a line-up when the ball
        is not being seen right now: where the tracker last PLACED it
        (odometry frame, so it survives the duck walking on — `Track.range`
        does not, it only moves on a hit), else the ball this line-up was
        planned around. None when neither exists, or when the target is
        further off the nose than the camera's own horizontal half-field
        (31°, so `GAZE_MAX_BEARING` is already generous): pitching the head
        cannot bring in something the lens does not cover sideways, and
        `gaze_yaw` is the knob that can.

        Worth knowing about the endpoint: ON the kick spot the ball is
        `kick_ahead` 0.08 m forward and `kick_side` 0.06 m to the side, i.e.
        37° off the nose — OUTSIDE the 31° half-field. The last few
        centimetres of a line-up are unseeable with a fixed head at any
        pitch. What is inside is the run-in: at 0.20 m ahead the same side
        offset is 17°, at 0.15 m it is 22°."""
        p = self.p
        tgt = None
        if ball is not None and ball.xy is not None:
            tgt = ball.xy
        elif self.spot is not None:
            sx, sy, _, u, _ = self.spot
            tgt = (sx + p.kick_ahead * math.cos(u), sy + p.kick_ahead * math.sin(u))
        if tgt is None:
            return None
        dx, dy = tgt[0] - odom[0], tgt[1] - odom[1]
        bearing = _wrap(math.atan2(dy, dx) - odom[2])
        if abs(bearing) > (p.head_yaw_max if p.gaze_yaw else GAZE_MAX_BEARING):
            return None
        return math.hypot(dx, dy), bearing

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["target"] = None if self.last_seen_t is None else {
            "bearing": round(self.last_bearing, 3), "range": None,
            "since": round(self._senses.t - self.last_seen_t, 2)}
        out["tracks"] = self.tracker.payload(self._senses.t)
        out["chase"] = {"kicks": self.kicks, "pushes": self.pushes, "role": self.role,
                        **({"declines": self.declines} if self.declines else {}),
                        **({"job": self.job} if self.job else {}),
                        "bumped": round(max(0.0, self._senses.t - self._bump_t), 2) if self._bump_t > -1e8 else None,
                        "tofBall": None if getattr(self, "tof_ball", None) is None else
                        [round(self.tof_ball[0], 2), round(self.tof_ball[1], 2)],
                        "memory": None if self.memory is None else [round(self.memory[0], 2), round(self.memory[1], 2)],
                        "predicted": None if getattr(self, "predicted", None) is None else [round(self.predicted[0], 2), round(self.predicted[1], 2)],
                        "spot": None if self.spot is None else
                        [round(self.spot[0], 3), round(self.spot[1], 3), self.spot[2] or self.spot[4]]}
        if self.team is not None:
            out["team"] = self.team.payload(self._senses.t)
        return out

    # -- geometry -------------------------------------------------------------
    def _ball_xy(self, odom, ball) -> tuple[float, float]:
        x, y, yaw = odom
        a = yaw + ball.bearing
        return x + ball.range * math.cos(a), y + ball.range * math.sin(a)

    def _own_goal(self, odom) -> tuple[float, float]:
        if self.goal is not None:
            return -self.goal[0], self.goal[1]                  # the pitch is centred on the origin
        a = self.attack if self.attack is not None else odom[2]
        return odom[0] - 2.0 * math.cos(a), odom[1] - 2.0 * math.sin(a)

    def _plan(self, odom, ball) -> tuple[float, float, str | None, float, str]:
        """Where to stand to kick a ball seen at (bearing, range): behind it
        on the line the kick should go — toward the goal (`goal`, in the
        odometry frame; without one, the heading the duck was placed with)
        when that costs under `aim_max` of detour, else whatever `aim_mode`
        says (the shipped "los" kicks along the line of sight, because a full
        walk-round crossed walls and the other duck — measured) —
        offset sideways so the nearer foot meets it. The left foot kicks a
        ball to its LEFT. A far goal (`push_beyond`) makes it a push spot
        squarely behind the ball. Returns (x, y, foot, heading, mode)."""
        p = self.p
        x, y, yaw = odom
        bx, by = self._ball_xy(odom, ball)
        los = yaw + ball.bearing
        if p.spot_lead > 0 and self._senses is not None and ball.vel_hits >= 2:
            # Where it will be when we arrive: the walk at `speed`, capped so
            # a bad velocity cannot throw the spot across the pitch.
            eta = min(p.spot_lead, math.hypot(bx - x, by - y) / max(p.speed, 1e-3))
            pred = ball.predict(self._senses.t + eta, p.ball_decel)
            if pred is not None:
                bx, by = pred
        if self.goal is not None:
            u = math.atan2(self.goal[1] - by, self.goal[0] - bx)
            far = math.hypot(self.goal[0] - bx, self.goal[1] - by) > p.push_beyond
            if p.kick_cone > 0 and self.goal_cone(bx, by) < p.kick_cone:
                far = True                          # too fine a target from here: dribble it closer
        else:
            u, far = (self.attack if self.attack is not None else los), False
        # The detour: how far round the ball this duck must get to send it
        # along `u`. Past `aim_max`, `aim_mode` says what to do instead.
        detour = _wrap(u - los)
        if abs(detour) > p.aim_max:
            if p.aim_mode == "clamp":
                u = _wrap(los + math.copysign(p.aim_max, detour))
            elif p.aim_mode != "goal":
                u, far = los, False
        if far:
            return bx - p.push_behind * math.cos(u), by - p.push_behind * math.sin(u), None, u, "push"
        rel = _wrap(math.atan2(by - y, bx - x) - u)
        foot = "kick_left" if rel >= 0 else "kick_right"
        if self.spot is not None and self.spot[2] in ("kick_left", "kick_right") and abs(rel) < 0.3:
            foot = self.spot[2]                                   # hysteresis: nearly on the line, keep the foot
        side = -p.kick_side if foot == "kick_left" else p.kick_side     # stand to the ball's other side
        # The body heading that sends the kick along u (the map's deflection
        # is in the body frame, so the spot is laid out in that heading too).
        h = _wrap(u - (p.kick_deflect_left if foot == "kick_left" else p.kick_deflect_right))
        return (bx - p.kick_ahead * math.cos(h) - side * math.sin(h),
                by - p.kick_ahead * math.sin(h) + side * math.cos(h), foot, h, "kick")

    def _board_ball(self, t: float) -> tuple[tuple[float, float] | None, float]:
        """The freshest ball sighting on the team board, and its age. A duck
        that cannot see the ball itself is usually standing off while a
        teammate is on it, and the board is the only way it learns the ball
        is coming at all — the camera loses a floor ball at 0.3 m and, over
        the threat battery, the best-placed defender had a fresh detection on
        22% of the ticks and any live track on 43%."""
        if self.team is None:
            return None, math.inf
        best = None
        for c in self.team.claims.values():
            if c.ball is not None and (best is None or c.t > best.t):
                best = c
        return (None, math.inf) if best is None else (best.ball, t - best.t)

    def _block_target(self, odom, ball, seen: bool, fresh: bool, t: float) -> tuple[float, float] | None:
        """Where to stand to block a ball rolling at our own goal, or None to
        carry on playing. The decision is `brain/intercept.py`'s; this only
        assembles what it needs out of what this duck actually has."""
        p = self.p
        if p.intercept_eta <= 0 or self.goal is None:
            return None
        mine = self._ball_xy(odom, ball) if seen else None
        board, age = self._board_ball(t)
        # The trigger is differenced from SIGHTINGS: my own when it is fresh,
        # else the board's, which is a teammate's. (The board cannot say how
        # old the sighting UNDER a claim is — a claim is re-stamped every tick
        # — so a teammate coasting a track can feed one stale point in. The
        # 0.6 s difference window is what stops a single one from firing it.)
        sight = mine if fresh else (board if age <= p.intercept_age else None)
        return self.blocker.update(t, p, mine if mine is not None else board, sight,
                                   self._own_goal(odom), odom, self.duck_id,
                                   self.team.mates(self.duck_id, t) if self.team is not None else [])

    def _servo(self, odom, target, cold, stop: float, slow_in: float = 0.2) -> tuple[float, float, float, float]:
        """(vx, wz, dist, bearing) toward a point: turn in place first when
        it is well off the nose, walk with steering otherwise, stop inside."""
        p = self.p
        dx, dy = target[0] - odom[0], target[1] - odom[1]
        dist = math.hypot(dx, dy)
        bearing = _wrap(math.atan2(dy, dx) - odom[2])
        if dist <= stop:
            return 0.0, 0.0, dist, bearing
        if abs(bearing) > 0.5 and dist > 0.08:
            vx, _, wz = turn(bearing, cold)
            return vx, wz, dist, bearing
        return (0.25 if dist < slow_in else p.speed), clip_wz(p.k_turn * bearing), dist, bearing

    def _on_the_line(self, odom, spot, u: float, heading_err: float) -> bool:
        """Is the duck already where stage one is trying to put it — on the
        kick line, squared up, still short of the spot? Then stage two can
        start here (`lineup_lat` > 0; 0 sends every line-up via the pre-spot,
        which is how `two_stage` was first measured).

        Only INSIDE the pre-spot's own distance: further back the walk to the
        pre-spot runs at `speed`, which beats the walk-in's `approach_speed`,
        and the ball is far enough that the square-up there is free."""
        p = self.p
        if p.lineup_lat <= 0.0:
            return False
        along = (spot[0] - odom[0]) * math.cos(u) + (spot[1] - odom[1]) * math.sin(u)
        lat = -(odom[0] - spot[0]) * math.sin(u) + (odom[1] - spot[1]) * math.cos(u)
        return (p.lineup_tol < along <= p.approach_back and abs(lat) <= p.lineup_lat
                and abs(heading_err) <= p.aim_tol)

    # -- the machine ----------------------------------------------------------
    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        p = self.p
        t = senses.t
        cold = self.gait.update(senses)
        odom = senses.odom or (0.0, 0.0, 0.0)
        if self.attack is None and senses.odom is not None:
            self.attack = odom[2]                  # placed facing the goal it attacks (make_pitch does)
        det_in = senses.fresh_det(self.DET_MAX_AGE)
        self.tof_ball: tuple[float, float] | None = None
        if p.tof_ball_m > 0 and (det_in is None or not any(d.cls == "ball" for d in det_in.detections)):
            tof_fr = senses.fresh_tof(self.TOF_MAX_AGE)
            blob = None if tof_fr is None else tof_floor_ball(tof_fr, r_max=p.tof_ball_m)
            if blob is not None:
                self.tof_ball = blob
                from ..sensors.detector import Detection, DetectionFrame
                det_in = DetectionFrame(t=tof_fr.t, detections=[Detection("ball", "", blob[0], -0.6, 0.2, blob[1], 0.8)])
        self.tracker.update(det_in, t, odom[2], (odom[0], odom[1]) if senses.odom is not None else None)
        ball = self.tracker.best(p.target_cls, t, min_hits=1)
        fresh = ball is not None and ball.age(t) <= self.DET_MAX_AGE
        seen = ball is not None and ball.age(t) < p.lost_s
        # Where the ball is going: its predicted position, and the bearing
        # to it from here (the head looks there; the search opens there).
        self.predicted: tuple[float, float] | None = None
        pred_bearing: float | None = None
        if ball is not None and ball.xy is not None and ball.age(t) <= p.predict_s:
            px, py = ball.predict(t, p.ball_decel)
            if self.bounds is not None:
                px = float(np.clip(px, -self.bounds[0] + 0.1, self.bounds[0] - 0.1))
                py = float(np.clip(py, -self.bounds[1] + 0.1, self.bounds[1] - 0.1))
            self.predicted = (px, py)
            pred_bearing = _wrap(math.atan2(py - odom[1], px - odom[0]) - odom[2])
        self._mates = []
        if senses.bumped:
            if t - self._bump_t > p.bump_gap_s:
                self._bump_t0 = t                            # a NEW contact, not the same one continuing
            self._bump_t = t
        if self.team is not None:
            self.team.claim(self.duck_id, t, ball.range if seen else math.inf,
                            self._ball_xy(odom, ball) if seen else None, (odom[0], odom[1], odom[2]))
            self.role = self.team.role(self.duck_id, t)
            for _, (mx, my, _) in self.team.mates(self.duck_id, t):
                self._mates.append((math.hypot(mx - odom[0], my - odom[1]),
                                    _wrap(math.atan2(my - odom[1], mx - odom[0]) - odom[2])))
        other = self.tracker.best("duck", t, min_hits=1)
        # The nearest duck ahead to avoid: a seen one, or a teammate by the board.
        threats = [(r, b) for r, b in self._mates if r < p.mate_keepout and abs(b) < p.duck_bearing]
        if other is not None and other.age(t) <= 0.6 and abs(other.bearing) < p.duck_bearing:
            # With the colour sense on, an OPPONENT gets its own keep-out:
            # a teammate is on the board (which knows who is quicker) and a
            # stranger is not, so they are not the same obstacle.
            keep = (p.opp_keepout if (p.use_color and p.opp_keepout > 0 and not self._is_mate(other))
                    else p.duck_keepout)
            if other.range < keep:
                threats.append((other.range, other.bearing))
        duck_rb = min(threats) if threats else None
        near_duck = duck_rb is not None
        clearly_nearer = (other is not None and other.age(t) <= p.lost_s and other.range < p.yield_range
                          and ball is not None and other.range < p.yield_ratio * ball.range
                          and abs(_wrap(other.bearing - ball.bearing)) < 0.8)
        if clearly_nearer and self.state != "yield" and t - self._yield_end > p.yield_cooldown_s:
            self._yield_t0 = t
        yielding = clearly_nearer and t - self._yield_t0 < p.yield_s
        if self.state == "yield" and not yielding:
            self._yield_end = t
        tof = senses.fresh_tof(self.TOF_MAX_AGE)
        ahead = left_near = right_near = np.inf
        if tof is not None:
            # Body-height things only (not the floor the head looks at, not
            # the ball), selected by bearing so a turned head cannot report
            # a wall that is really off to the side.
            ahead, left_near, right_near = tof_clearance_bearings(tof)
        skill = None
        head = (0.0, 0.0, 0.0, 0.0)
        gaze_at: float | None = None
        gaze_yaw = 0.0
        retreating = t - self._retreat_t0 < p.retreat_turn_s + p.retreat_walk_s
        if self._prev_skill is not None and senses.skill is None:
            self._look_t0 = t                                   # the kick window just ended: look for the ball ahead
        self._prev_skill = senses.skill
        looking = t - self._look_t0 < p.look_s and not fresh
        if seen:
            self._last_range = ball.range
            self.last_bearing = ball.bearing
            if fresh:
                bx_, by_ = self._ball_xy(odom, ball)
                self.memory = (bx_, by_, t)
        elif self._last_range is not None and self._last_range < p.hunt_lost_range and self.state in ("chase", "lineup", "turn"):
            self._hunt_u = odom[2]                              # walked into it: it rolled off ahead
            self._last_range = None
        if self.state == "look" and not looking and not fresh and self._hunt_u is not None:
            self._hunt_t0 = t                                   # the look after the kick found nothing: hunt the line
        hunting = p.hunt_s > 0 and t - self._hunt_t0 < p.hunt_s and not seen and self._hunt_u is not None
        if p.hunt_s > 0 and self._hunt_u is not None and not seen and t - self._hunt_t0 >= p.hunt_s \
                and self.state == "hunt":
            # The hunt ran its course without a sighting: the ball is further
            # along the line - remember a point there and forget the line.
            self.memory = (odom[0] + 0.6 * math.cos(self._hunt_u), odom[1] + 0.6 * math.sin(self._hunt_u), t)
            self._hunt_u = None
        if hunting and (ahead < p.hunt_stop or self._beside(t) or self._boards_ahead(odom, p.hunt_margin)):
            hunting = False                                     # the hunt ends here; the search takes over
            self._hunt_t0 = -1e9
            self._hunt_u = None
        if (not seen and not hunting and self.memory is not None
                and (t - self.memory[2] > p.seek_s
                     or math.hypot(self.memory[0] - odom[0], self.memory[1] - odom[1]) <= p.seek_min)):
            self.memory = None                                  # stale, or here with nothing seen: forget it
        seeking = p.seek_s > 0 and not seen and not hunting and self.memory is not None
        if seeking and (ahead < p.hunt_stop or self._beside(t)):
            seeking = False                                     # something in the way: circle here instead
            self.memory = None
        # Is the ball rolling into OUR goal, and am I the one to stand in
        # front of it? (brain/intercept.py; `intercept_eta` = 0 ships it off.)
        block_at = self._block_target(odom, ball, seen, fresh, t)
        if senses.skill is not None:
            vx, wz = 0.0, 0.0                                   # the kick owns the reflex tier
            self.state = "kick"
        elif looking:
            vx, wz = 0.0, 0.0
            gaze_at = p.look_aim_range if p.look_aim else p.look_range
            self.state = "look"
        elif retreating:
            if t - self._retreat_t0 < p.retreat_turn_s:
                vx, _, wz = turn(self._retreat_sign, cold)
            else:
                vx, wz = p.speed, 0.0
            self.spot = None
            self.state = "retreat"
        elif near_duck:
            # Turn AWAY from it (the side that puts it behind us), never
            # into it, and not at all while it is touching: a stand is the
            # one thing the walker does safely against another body. The
            # cold-gait kick creeps forward, so no kick with it near the nose.
            self.spot = None
            if duck_rb[0] < p.duck_touch:
                vx, wz = 0.0, 0.0
            else:
                vx, _, wz = turn(-1.0 if duck_rb[1] >= 0.0 else 1.0, cold)
                if abs(duck_rb[1]) < 0.5:
                    vx = 0.0
            self.state = "avoid"
        elif block_at is not None:
            # Leave the play and get in the way. Not a line-up: the servo
            # faces where it WALKS, and only once it is on the line does the
            # duck square up on the ball — so the body ends across the path
            # with the camera on the thing it is stopping, and no square-up
            # ever happens next to the ball (which is what turns a line-up
            # into a shove).
            self.spot = None
            vx, wz, bdist, _ = self._servo(odom, block_at, cold, p.intercept_tol)
            if bdist <= p.intercept_tol:
                b = self.blocker.ball
                bb = 0.0 if b is None else _wrap(math.atan2(b[1] - odom[1], b[0] - odom[0]) - odom[2])
                vx, wz = (0.0, 0.0) if abs(bb) < 0.3 else turn(bb, cold)[::2]
                if wz != 0.0 and self._beside(t):
                    vx, wz = 0.0, 0.0                  # a body beside us: never a turn in place
            self.state = "block"
        elif self.role == "support":
            vx, wz = self._support(odom, ball, seen, cold)
        elif yielding and self.state not in ("settle",):
            vx, wz = 0.0, 0.0
            self.spot = None
            self.state = "yield"
        elif self.state == "push":
            _, _, _, u, _ = self.spot
            vx, wz = p.push_speed, clip_wz(p.k_turn * _wrap(u - odom[2]))
            if t - self.t_state >= p.push_s:
                self.spot = None
                self.state = "search"
                self._look_t0 = t
        elif self.state in ("lineup", "settle") and self.spot is not None:
            # Refresh the spot while the ball is in view and not too close
            # (see refresh_min), then walk the rest blind.
            if fresh and self.state != "settle" and p.refresh_min <= ball.range < p.head_range:
                new = self._plan(odom, ball)
                if self.lined and math.hypot(new[0] - self.spot[0], new[1] - self.spot[1]) > 0.05:
                    self.lined = False                          # the ball is not where the line was laid: lay it again
                self.spot = new
            sx, sy, foot, u, mode = self.spot
            heading_err = _wrap(u - odom[2])
            if mode == "kick" and p.two_stage and not self.lined \
                    and self._on_the_line(odom, (sx, sy), u, heading_err):
                self.lined = True                               # nothing to go back for
                self.t_state = t                                # stage two gets its own clock
            if mode == "kick" and p.two_stage and not self.lined:
                # Stage one: the pre-spot behind the kick spot on the line;
                # square up there, where a turn in place cannot touch the ball.
                px, py = sx - p.approach_back * math.cos(u), sy - p.approach_back * math.sin(u)
                vx, wz, pdist, bearing = self._servo(odom, (px, py), cold, p.approach_tol)
                ball_rng = math.hypot(sx + p.kick_ahead * math.cos(u) - odom[0], sy + p.kick_ahead * math.sin(u) - odom[1])
                if abs(bearing) > 1.8 and ball_rng < p.backoff_range and pdist > p.approach_tol + 0.02:
                    # The pre-spot is behind us with the ball at our feet:
                    # turning to it is a turn against the ball. Back off.
                    self.spot = None
                    self._retreat_t0 = t
                    self._retreat_sign = -1.0 if _wrap(math.atan2(sy - odom[1], sx - odom[0]) - odom[2]) >= 0 else 1.0
                    vx, _, wz = turn(self._retreat_sign, cold)
                    self.state = "retreat"
                    dist = 9.0
                elif pdist <= p.approach_tol + 0.02 and abs(heading_err) <= p.aim_tol:
                    self.lined = True
                    self.t_state = t                            # stage two gets its own clock
                elif pdist <= p.approach_tol + 0.02:
                    vx, _, wz = turn(heading_err, cold)
                dist = pdist + p.approach_back if self.spot is not None else 9.0   # nowhere near the spot yet
            else:
                vx, wz, dist, bearing = self._servo(odom, (sx, sy), cold, p.lineup_tol)
                if mode == "kick" and p.two_stage and self.state != "settle":
                    # Stage two: in along the line, steering onto it (the
                    # walker crabs on a pure forward command), stop on the
                    # spot by the distance left along the line.
                    along = (sx - odom[0]) * math.cos(u) + (sy - odom[1]) * math.sin(u)
                    lat = -(odom[0] - sx) * math.sin(u) + (odom[1] - sy) * math.cos(u)   # +: left of the line
                    if along > p.lineup_tol:
                        vx = p.approach_speed
                        wz = float(np.clip(-p.k_lat * lat + p.k_head * heading_err, -p.approach_wz, p.approach_wz))
                        dist = along
                    else:
                        vx, wz, dist = 0.0, 0.0, 0.0
            # Hysteresis on both: a settling duck wobbles a centimetre and a
            # few hundredths of a radian, which flipped it between the
            # square-up and the settle at the tolerance (measured: 22 s
            # standing at the spot, no kick).
            settling = self.state == "settle"
            on_spot = dist <= p.lineup_tol + (0.03 if settling else 0.0)
            squared = abs(heading_err) <= p.aim_tol + (0.15 if settling else 0.0)
            if self.spot is None:
                pass                                            # backing off (above)
            elif on_spot and not squared and mode == "kick" and p.two_stage:
                # On the spot but off the heading: a turn in place here is a
                # turn against the ball (traced: 14 s of it). Back off and
                # lay the line again from further out.
                self.spot = None
                self._retreat_t0 = t
                self._retreat_sign = -1.0 if heading_err >= 0 else 1.0
                vx, _, wz = turn(self._retreat_sign, cold)
                self.state = "retreat"
            elif on_spot and not squared:
                vx, _, wz = turn(heading_err, cold)            # a push spot: square up
                self.state = "lineup"
            elif on_spot:
                # Stand first: robotd runs a kick at the standing tuning, and
                # a kick started mid-stride fell 4 times in 7 here. Something
                # within `kick_clear` ahead (a wall, a duck) means the swing
                # lands on it: let it go and look again.
                vx, wz = 0.0, 0.0
                if ahead < p.kick_clear:
                    self.spot = None
                    self.state = "search"
                elif not settling:
                    self.state = "settle"
                    self.t_state = t
                elif t - self.t_state >= p.settle_s:
                    if mode == "push":
                        self.pushes += 1
                        self.state = "push"
                        self.t_state = t
                        vx, wz = p.push_speed, 0.0
                    elif self._too_wide(odom):
                        # The geometry says this one misses. Drop the spot
                        # and walk it again rather than spend a touch on a
                        # shot already 20-plus degrees wide.
                        self.declines += 1
                        self.spot = None
                        self.state = "chase"
                        self.t_state = t
                    else:
                        skill = foot
                        self._last_foot = foot
                        self.kicks += 1
                        self.spot = None
                        self.state = "kick"
                        # Where the ball is going — which is NOT `u`, the line
                        # it was aimed along: the kick leaves the foot at an
                        # angle to the body (`kick_exit_*`, measured in play).
                        heading = self._kick_heading(foot, u)
                        self._hunt_u = heading
                        if self.team is not None:
                            origin = (self._ball_xy(odom, ball) if seen
                                      else (sx + p.kick_ahead * math.cos(u),
                                            sy + p.kick_ahead * math.sin(u)))
                            self.team.publish_kick(t, origin, heading, p.kick_speed)
            elif self.state == "lineup" and t - self.t_state > p.lineup_s:
                self.spot = None
                self.state = "search"
                vx, _, wz = turn(1.0, cold)
            else:
                self.state = "lineup"
                if vx > 0 and fresh and ball.range < p.head_range and abs(ball.bearing) < 0.6:
                    gaze_at = ball.range
            # …and keep looking at it through the settle and the square-up,
            # which is where the swing is decided and where the old gate
            # (`vx > 0`, below) dropped the head. Aimed at the ball's last
            # PLACE rather than its last range, because the duck has walked
            # since. The application gate still refuses a turn in place.
            if p.gaze_still and gaze_at is None and self.state in ("lineup", "settle"):
                got = self._gaze_range(odom, ball)
                if got is not None:
                    gaze_at, gaze_yaw = got
        elif seen:
            self.last_bearing = ball.bearing
            if fresh:
                self.last_seen_t = t
            if fresh and ball.range < p.lineup_range and abs(ball.bearing) < 0.5:
                self.spot = self._plan(odom, ball)
                self.lined = False
                self.state = "lineup"
                self.t_state = t
                vx, wz = p.speed, clip_wz(p.k_turn * ball.bearing)
                gaze_at = ball.range
            elif abs(ball.bearing) > p.turn_first:
                vx, _, wz = turn(ball.bearing, cold)
                self.state = "turn"
            else:
                vx, wz = p.speed, clip_wz(p.k_turn * ball.bearing)
                self.state = "chase"
                if fresh and ball.range < p.head_range:
                    gaze_at = ball.range
        elif hunting:
            if self.predicted is not None and p.predict_steer:  # the line bends to where the ball is going
                self._hunt_u = math.atan2(self.predicted[1] - odom[1], self.predicted[0] - odom[0])
            vx, wz = p.hunt_speed, float(np.clip(p.k_turn * _wrap(self._hunt_u - odom[2]), -p.hunt_wz, p.hunt_wz))
            self.state = "hunt"
        elif seeking and self.state not in ("look",):
            # Walk to where the ball was, head level, at the hunt's pace.
            vx, wz, sdist, sbear = self._servo(odom, self.memory[:2], cold, p.seek_tol)
            vx = min(vx, p.hunt_speed) if abs(sbear) <= 0.5 else vx
            if sdist <= p.seek_tol:
                self.memory = None                              # here, and nothing seen: forget it
            self.state = "seek"
        else:
            if p.hunt_s > 0 and self._hunt_u is not None and self.state not in ("search", "hunt", "look") \
                    and self._last_range is None and t - self._hunt_t0 >= p.hunt_s \
                    and ahead >= p.hunt_stop and not self._beside(t) and not self._boards_ahead(odom, p.hunt_margin):
                self._hunt_t0 = t                               # lost while walking into it: hunt before searching
                vx, wz = p.hunt_speed, float(np.clip(p.k_turn * _wrap(self._hunt_u - odom[2]), -p.hunt_wz, p.hunt_wz))
                self.state = "hunt"
            else:
                if self.state != "search" or self._search_t0 is None:
                    self._search_t0 = t
                # Toward the side the ball was last on (probed: the sweep runs
                # at ~24 deg/s, so a ball to the right found by a left turn
                # takes 10 s, by a right turn 4); the cold-turn kick starts
                # the right turn the standing walker cannot.
                side = pred_bearing if pred_bearing is not None and p.predict_steer else (self.last_bearing if p.search_sided else 1.0)
                vx, _, wz = turn(1.0 if side >= 0.0 else -1.0, cold)
                vx = max(vx, p.search_vx)                       # a walking circle: the body actually turns
                self.state = "search"
                since = t - self._search_t0
                if since % p.search_dip_every < p.search_dip_s:
                    # A standing pause with the gaze down. The comment here
                    # used to say it was for seeing a near ball below the
                    # level camera; MEASURED, it does not do that. Over 24
                    # seeds x 300 s of 2v2, detector frames binned by the
                    # state that COMMANDED the head pose: frozen in the dip,
                    # 11 balls found in 15 646 frames (0.07%); walking the
                    # same search circle, 505 in 11 116 (4.54%). The dip
                    # takes 58% of search frames and returns 2.1% of the
                    # search's sightings - 65x worse than simply walking on
                    # with the head level (z = 26.2).
                    #
                    # It stays because REMOVING it is much worse, and that
                    # confirmed on fresh seeds: `search_dip_s` = 0 gives back
                    # 19 s a run of standing still and costs falls 121 -> 195
                    # over 48 seeds (+61%) and 30% of the kicks, for no
                    # visibility gain at all (+0.008, p = 0.60). `_search_t0`
                    # resets on every ENTRY, so this is not a duty cycle in a
                    # long hunt: the median search is 0.60 s, exactly the
                    # dip, and 48% of searches are nothing but this pause. It
                    # is a flinch every time the ball leaves view (~33 a run
                    # a duck) - and standing is the one thing this walker
                    # does safely against another body, which is why taking
                    # it away costs falls. Mis-commented, not mis-designed.
                    vx, wz = 0.0, 0.0
                    gaze_at = p.dip_range
                elif p.search_walk_after and since > p.search_walk_after and (since - p.search_walk_after) % (p.search_walk_after) < p.search_walk_s:
                    vx, wz = p.speed, 0.0                       # a cold standing turn is exactly 0 rad/s: move to see from elsewhere
        # A wall beside us: no turn in place toward it (measured: a line-up
        # turning against the boards tipped over). Turn toward the side
        # with more room — in a corner that is still a turn, the one move
        # that gets out of a corner (standing there measured as a deadlock).
        if vx <= TURN_KICK and wz != 0.0 and self.state != "retreat":
            if wz > 0 and left_near < p.side_stop and right_near > left_near:
                wz = -1.0
            elif wz < 0 and right_near < p.side_stop and left_near > right_near:
                wz = max_wz()
        if ahead < p.tof_stop and vx > 0 and self.state != "push":
            # A wall or the other duck right there: no walking, no cold-turn
            # creep (measured: every remaining fall was a line-up walking
            # into a wall or a kicked turn creeping into one). Turning still
            # happens — the left turn that starts from a standstill.
            vx = 0.0
            if self.state in ("lineup", "settle", "support"):
                wz = max_wz() if wz > 0 else -max_wz() if wz < 0 else 0.0
            else:
                wz = max_wz()
                self.state = "blocked"
        # Not moving while stood against something (avoid, blocked) for
        # `stuck_s`, whatever the state labels say frame to frame: retreat.
        self._poses.append((t, odom[0], odom[1], odom[2]))
        while self._poses and t - self._poses[0][0] > p.stuck_s:
            self._poses.pop(0)
        if self.state in ("avoid", "blocked", "yield") and len(self._poses) > 1 \
                and t - self._poses[0][0] >= p.stuck_s - 0.05:
            _, x0, y0, yaw0 = self._poses[0]
            if math.hypot(odom[0] - x0, odom[1] - y0) < 0.05 and abs(_wrap(odom[2] - yaw0)) < 0.3:
                self._poses = []
                self._retreat_t0 = t
                self._retreat_sign = 1.0 if left_near >= right_near else -1.0
        # A turn in place keeps the head level whatever the state asked for:
        # the walker cannot turn in place with its head down (0.2 rad in 5 s
        # against 3.1 level, measured in tidy.py). A COLD turn carries
        # `TURN_KICK` of forward command to start the gait, so "turning in
        # place" is not "vx == 0" — which is why the old `vx > 0` gate let
        # the head down during exactly the manoeuvre that cannot take it.
        turning = wz != 0.0 and vx <= TURN_KICK
        if gaze_at is not None and not (p.gaze_still and turning) \
                and (vx > 0 or self.state in ("look", "search")
                     or (p.gaze_still and wz == 0.0)):
            # The gaze YAW is gated on forward clearance exactly like the look
            # yaw below, and for the same measured reason: the ToF is on the
            # head, so a yawed head is honestly blind ahead. The gaze PITCH is
            # not gated — the dip re-screened as a clean null (roadmap 4e) and
            # gating it would change the shipped brain, which this does not:
            # with `gaze_yaw` off the yaw is 0.0 either way.
            gyaw = float(np.clip(p.head_yaw_gain * gaze_yaw,
                                 -p.head_yaw_max, p.head_yaw_max)) if p.gaze_yaw else 0.0
            if p.yaw_clear > 0.0 and ahead < p.yaw_clear:
                gyaw = 0.0
            head = self._head_pose(self._gaze(gaze_at), gyaw)
        look_at = pred_bearing if pred_bearing is not None else (
            ball.bearing if p.predict_s > 0 and ball is not None and ball.age(t) <= p.predict_s else None)
        if look_at is None and self.state == "look" and p.look_aim and self._last_foot is not None:
            look_at = p.kick_exit_left if self._last_foot == "kick_left" else p.kick_exit_right
        elif look_at is None and self.state == "search" and p.search_sweep > 0 and self._search_t0 is not None:
            look_at = p.search_sweep * math.sin(2.0 * math.pi * (t - self._search_t0) / p.search_sweep_s) / p.head_yaw_gain
        if look_at is not None and senses.skill is None and (p.head_yaw_when == "always" or self.state in ("search", "look")) \
                and not (p.yaw_clear > 0.0 and ahead < p.yaw_clear):
            # …unless the way ahead is not clear. The ToF is ON THE HEAD, so
            # yawing it points the bumper off the walking line: since the
            # clearance rule became bearing-based it reports `+inf` honestly
            # instead of a false wall, which means a yawed duck walks with no
            # forward obstacle sense at all (measured: past 0.70 rad it stops
            # on 0.3% of frames where the old column rule stopped on 13.8%).
            # That is where head-tracking's falls come from. The brain has the
            # signal and did not consult it; this consults it, keeping the
            # head on the line whenever something is inside `yaw_clear`.
            head = (head[0], head[1], float(np.clip(p.head_yaw_gain * look_at, -p.head_yaw_max, p.head_yaw_max)), head[3])
        # The two arms of the same rule, on the SAME gate - a turn in place,
        # beside a body, in a state where that turn is not itself the escape
        # - so an A/B between them measures the action and nothing else.
        if t - self._bump_t0 < max(p.bump_stand_s, p.bump_back) and vx <= TURN_KICK and wz != 0.0 \
                and self.state in p.bump_stand_states:
            if p.bump_back > 0 and t - self._bump_t0 < p.bump_back:
                vx, _, wz = back_up()                           # back out of it, which standing never does
            elif p.bump_stand_s > 0:
                vx, wz = 0.0, 0.0                               # touching a body: stand, do not turn in place
        self.last = (vx, 0.0, wz)
        return Intent(twist=self.last, head=head, note=self.role if self.role != "attack" else self.state, skill=skill)

    def _attack_x(self, x: float) -> float:
        """A point's position along the pitch in ATTACK coordinates: −1 at the
        goal we defend, +1 at the one we attack. Both teams read the same
        numbers, so a third is a third whichever way a duck is pointing."""
        if self.bounds is None or self.bounds[0] <= 0:
            return 0.0
        sign = 1.0 if (self.goal is None or self.goal[0] >= 0) else -1.0
        return sign * x / self.bounds[0]

    def _from_attack_x(self, a: float) -> float:
        sign = 1.0 if (self.goal is None or self.goal[0] >= 0) else -1.0
        return sign * a * (self.bounds[0] if self.bounds else 0.0)

    def _too_wide(self, odom) -> bool:
        """Is the ball too far to the SIDE for this swing to be worth a
        touch? (`kick_side_max`; False when the knob is off.)

        The side offset is the kick error — +1.90 deg of aim error per cm,
        measured over 462 kicks — and it cannot be aimed out, because the
        spot is laid out in the body heading so every rotation moves the
        offset itself. Declining is the only remaining lever.

        The estimate is `self.predicted`, the track's position propagated by
        its own velocity, which exists only while the sighting is inside
        `predict_s`. With no fresh estimate this returns False and the duck
        swings: the plan's own assumed ball position sits ON the sweet spot
        by construction, so gating on it would refuse nothing, and gating on
        nothing at all would be a duck that never kicks."""
        if self.p.kick_side_max <= 0.0 or self.predicted is None:
            return False
        dx, dy = self.predicted[0] - odom[0], self.predicted[1] - odom[1]
        side = -dx * math.sin(odom[2]) + dy * math.cos(odom[2])
        return abs(side) > self.p.kick_side_max

    def _kick_heading(self, foot: str, u: float) -> float:
        """The line the ball actually leaves on: aim heading plus the in-play
        foot exit angle when `hunt_exit` is on (the shipped default)."""
        p = self.p
        if not p.hunt_exit:
            return u
        return u + (p.kick_exit_left if foot == "kick_left" else p.kick_exit_right)

    def _hold_target(self, bxy, odom) -> tuple[float, float]:
        """Where this duck stands while a teammate has the ball.

        With no static role it is the shipped supporter's spot — back from the
        ball toward our own goal, spread sideways by rank. With one it is that
        role's post (`ChaseParams.defend_depth` / `strike_ahead` / `mid_side`),
        kept inside the third the role owns so that holding a post and being
        allowed to take the ball are the same geometry (`Team.zone_ok`)."""
        p = self.p
        t = self._senses.t
        og = self._own_goal(odom)
        if self.job == "defender":
            dx, dy = bxy[0] - og[0], bxy[1] - og[1]
            n = math.hypot(dx, dy)
            if n < 1e-6:
                target = og
            else:
                # On the line from our goal to the ball: between, which is the job.
                target = (og[0] + p.defend_depth * dx / n, og[1] + p.defend_depth * dy / n)
        elif self.job == "striker":
            g = self.goal if self.goal is not None else (og[0] + 2.0, og[1])
            gx, gy = g[0] - bxy[0], g[1] - bxy[1]
            n = math.hypot(gx, gy)
            ux, uy = (gx / n, gy / n) if n > 1e-6 else (1.0, 0.0)
            # Off the kick line, on the side the ball is NOT on: a striker
            # standing ON the line is the poacher that reversed on fresh
            # seeds, and it is a second duck on the ball.
            side = -p.strike_side if bxy[1] >= 0 else p.strike_side
            target = (bxy[0] + p.strike_ahead * ux - side * uy, bxy[1] + p.strike_ahead * uy + side * ux)
        elif self.job == "midfielder":
            a = float(np.clip(self._attack_x(bxy[0]) * 0.5, -1.0 / 3.0, 1.0 / 3.0))
            target = (self._from_attack_x(a), p.mid_side * (1.0 if bxy[1] >= 0 else -1.0))
        else:
            anchor = og if p.support_mode == "back" else (self.goal if self.goal is not None else og)
            gx, gy = anchor[0] - bxy[0], anchor[1] - bxy[1]
            gn = math.hypot(gx, gy)
            ux, uy = (gx / gn, gy / gn) if gn > 1e-6 else (-math.cos(odom[2]), -math.sin(odom[2]))
            rank = self.team.rank(self.duck_id, t) if self.team is not None else 0
            side = p.support_side * ((rank + 1) // 2) * (1 if rank % 2 == 0 else -1)
            return (bxy[0] + p.support_back * ux - side * uy, bxy[1] + p.support_back * uy + side * ux)
        # Holding a post and being allowed to take the ball are the same
        # geometry: clip to the zone the board uses (halfway without a mid,
        # thirds with one).
        z = self.team.zone_of(self.duck_id) if self.team is not None else None
        if z is not None:
            a = float(np.clip(self._attack_x(target[0]), z[0], z[1]))
            target = (self._from_attack_x(a), target[1])
        return target

    def _support(self, odom, ball, seen: bool, cold: bool) -> tuple[float, float]:
        """A supporter: hold the post its role gives it (`_hold_target`) —
        without a role, back from the ball toward our own goal, offset
        sideways by rank — facing the ball. The ball's position comes from
        my own track when I see it, else from a teammate's claim."""
        p = self.p
        t = self._senses.t
        bxy = self._ball_xy(odom, ball) if seen else (
            self.team.led_ball(t) if self.team is not None else None)
        self.spot = None
        if bxy is None:
            self.state = "support"
            vx, _, wz = turn(1.0, cold)                    # nobody has it: look for it
            return vx, wz
        target = self._hold_target(bxy, odom)
        if self.bounds is not None:                         # never a spot in the boards
            m = p.support_margin
            target = (float(np.clip(target[0], -self.bounds[0] + m, self.bounds[0] - m)),
                      float(np.clip(target[1], -self.bounds[1] + m, self.bounds[1] - m)))
        vx, wz, dist, _ = self._servo(odom, target, cold, 0.12)
        if math.hypot(bxy[0] - odom[0], bxy[1] - odom[1]) < p.support_min and vx > 0:
            vx = 0.0                                        # the attacker's room
        if dist <= 0.12:
            b = _wrap(math.atan2(bxy[1] - odom[1], bxy[0] - odom[0]) - odom[2])
            vx, wz = (0.0, 0.0) if abs(b) < 0.3 else turn(b, cold)[::2]
            if wz != 0.0:
                vx = max(vx, p.support_turn_vx)
        if vx <= TURN_KICK and wz != 0.0 and self._beside(t):
            vx, wz = 0.0, 0.0                               # a body beside us: no turning in place
        self.state = "support"
        return vx, wz

    def goal_cone(self, bx: float, by: float) -> float:
        """Half the angle the goal mouth subtends from a ball at (bx, by),
        in the odometry frame: how fine a target this shot is. +inf off a
        pitch or without a mouth width."""
        if self.goal is None or self.goal_w <= 0:
            return math.inf
        gx = self.goal[0]
        a1 = math.atan2(-self.goal_w / 2 - by, gx - bx)
        a2 = math.atan2(self.goal_w / 2 - by, gx - bx)
        return abs(_wrap(a2 - a1)) / 2.0

    def _boards_ahead(self, odom, margin: float) -> bool:
        """The point `margin` ahead in odometry lies outside the pitch's bounds (None off a pitch: never)."""
        if self.bounds is None:
            return False
        x, y = odom[0] + margin * math.cos(odom[2]), odom[1] + margin * math.sin(odom[2])
        return abs(x) > self.bounds[0] or abs(y) > self.bounds[1]

    def _is_mate(self, tr) -> bool:
        """Is this duck track one of ours? Only with the colour sense on, and
        only on the track's VOTE — one frame of the classifier is a coin at
        the hostile preset. Unknown counts as an opponent: the cost of
        treating a teammate as a stranger is a wasted metre, and the cost of
        the reverse is walking into one."""
        return bool(self.p.use_color and self.team is not None
                    and getattr(tr, "color", None) == self.team.name)

    def _beside(self, t: float) -> bool:
        """Any duck track inside `beside_m`, at any bearing, within `beside_s`;
        or a teammate inside `mate_keepout` by the team board."""
        p = self.p
        return any(tr.cls == "duck" and tr.range < p.beside_m and tr.age(t) <= p.beside_s
                   for tr in self.tracker.tracks) or any(r < p.mate_keepout for r, _ in self._mates)


def _r(v) -> float | None:
    return None if v is None else round(float(v), 3)


class Script:
    """No brain: the world's drive script / manual command steers."""

    kind = "script"

    def __init__(self):
        self.state = "script"

    def step(self, senses: Senses) -> Intent:
        return Intent()

    def reset(self) -> None:
        pass

    def inputs(self) -> dict:
        return {}


REGISTRY.register("wander", Wander)
REGISTRY.register("follow", Follow)
REGISTRY.register("chase", Chase)
REGISTRY.register("script", Script)
