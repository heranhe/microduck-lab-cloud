"""A tracker over the detector's frames (roadmap 1.3's "tracker gives ids").

The detector returns per-frame detections; a brain that acts on the raw
list re-picks its target every frame (a ghost or a second person steals
it) and loses it the first frame it is missed. `Tracker` associates each
frame's detections with the tracks it already has — same class, nearest
bearing inside a gate, range not wildly different — smooths bearing and
range, counts hits, and COASTS a track through misses: with odometry, the
remembered bearing turns with the body, so a person the duck turns away
from is still "at −1.2 rad", not gone. Ghosts (no consistent position)
never reach the hit count a brain asks for. (The bearing turns with the
body's YAW; `TrackerParams.coast_from_xy` is the knob that also turns it with
the body's TRANSLATION, by re-reading it off the remembered position.)

In the sim the detector also hands out the true object name; the tracker
keeps it as `name` for the tools and tests, but its `id` is its own —
what the real robot will have.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields

from ..sensors.detector import Detection, DetectionFrame


@dataclass
class Track:
    id: int
    cls: str
    bearing: float
    elevation: float
    width: float
    range: float
    conf: float
    born_t: float
    last_t: float               # last frame that HIT
    hits: int = 1
    misses: int = 0             # FRAMES since the last hit (no frame, no miss: use age() for staleness)
    name: str = ""
    names: dict = field(default_factory=dict)   # sim name → count, for `name`
    # The colour classifier's VOTES over this track's hits, and the winner.
    # A vote, not the last frame's answer: at the hostile preset the
    # classifier is right three times in four, so one frame of it is a coin
    # a brain must not act on — and a track that has been seen ten times has
    # a colour that is right far more often than any single look.
    colors: dict = field(default_factory=dict)
    color: str | None = None
    # Where it is and where it is going, in the ODOMETRY frame (only when
    # the brain passes its position in): the last hit's position, the time
    # of it, and a smoothed velocity from consecutive hits. A rolling ball
    # leaves the camera in a frame or two; this is what says where to look.
    xy: tuple[float, float] | None = None
    xy_t: float = 0.0
    vel: tuple[float, float] = (0.0, 0.0)
    vel_hits: int = 0                            # hits the velocity rests on (2+ before trusting it)
    # THE BALL'S UNCERTAINTY (roadmap Track 4 s6 C.1). A point estimate
    # with no covariance is why nothing downstream could say how much to
    # trust it: `sig_meas` is the position sigma of the last hit, built
    # from the detector's DECLARED noise at that range (`TrackerParams.
    # meas_*`, the datasheet numbers a real robot would be configured
    # with), and `vel_sig` the scatter of the velocity samples about the
    # smoothed velocity. `sigma(t)` grows them by the time since the hit:
    # a ball not seen for a while may have rolled. Calibrated against the
    # truth by scripts/probe_shot_gate.py.
    sig_meas: float = 0.0
    vel_sig: float = 0.0
    # THE RESTING BALL (roadmap Track 4 item 12f). Set when something may
    # have moved this track since its last hit - our own kick, a body next
    # to it - so it stops counting as at rest until a fresh sighting says
    # otherwise. Cleared by any hit.
    rest_block: bool = False

    def at_rest(self, rest_vel: float) -> bool:
        """Has this track been MEASURED to be standing still, and has
        nothing since been seen that might have moved it?

        Deliberately not "we have no velocity for it": a ball seen once,
        rolling, also has no velocity, and treating that as at rest is how
        a memory becomes a lie. It takes two hits agreeing it is slow."""
        return (self.xy is not None and not self.rest_block
                and self.vel_hits >= 2 and math.hypot(*self.vel) < rest_vel)

    def sigma(self, t: float, vel_prior: float = 0.06, vel_sig_after_s: float = 0.0) -> float:
        """The 1-sigma position uncertainty (m) of `predict(t)`: the last
        hit's own error, plus how far an uncertain velocity may have
        carried the ball since - the measured scatter of the velocity when
        the track has one and the hit is older than `vel_sig_after_s`,
        `vel_prior` otherwise."""
        dt = max(0.0, t - self.xy_t) if self.xy is not None else 0.0
        sv = self.vel_sig if (self.vel_hits >= 2 and dt > vel_sig_after_s) else vel_prior
        return math.hypot(self.sig_meas, sv * dt)

    def age(self, t: float) -> float:
        return t - self.last_t

    def bearing_from(self, pos: tuple[float, float], yaw: float) -> float | None:
        """The bearing (body frame) to this track's remembered POSITION from
        a body standing at `pos` and facing `yaw`. None without a position.

        `Track.bearing` is what the last HIT measured, turned since only by
        the body's yaw (`Tracker.update`) - never by its translation. So a
        duck that walks while coasting a track carries a bearing that stopped
        meaning "bearing" the moment it moved: measured at 112 degrees off
        after a walk (roadmap 12af). `xy` does not have that problem, and
        this reads the bearing off it.

        Exact inverse of `_place` at the moment of the hit: that writes
        `xy = pos + range * (cos, sin)(yaw + bearing)`, so called with the
        same pose this returns the same bearing to the float. The knob that
        wires it into the coast is `TrackerParams.coast_from_xy`."""
        if self.xy is None:
            return None
        a = math.atan2(self.xy[1] - pos[1], self.xy[0] - pos[0]) - yaw
        return math.atan2(math.sin(a), math.cos(a))

    def range_from(self, pos: tuple[float, float]) -> float | None:
        """The distance to the remembered position from `pos`, the same way
        `_place` used `range` on the way in (a ground distance in the
        odometry plane - `range` itself is the detector's 3-D SLANT range,
        and the ~2 cm the camera's height puts between them is the same
        approximation `_place` already makes). None without a position."""
        if self.xy is None:
            return None
        return math.hypot(self.xy[0] - pos[0], self.xy[1] - pos[1])

    def predict(self, t: float, decel: float = 0.0) -> tuple[float, float] | None:
        """Position at t from the last hit and the velocity (a constant
        deceleration `decel` along it, to a stop). None without a position."""
        if self.xy is None:
            return None
        dt = max(0.0, t - self.xy_t)
        vx, vy = self.vel
        speed = math.hypot(vx, vy)
        if speed < 1e-6 or self.vel_hits < 2:
            return self.xy
        if decel > 0:
            t_stop = speed / decel
            dt = min(dt, t_stop)
            dist = speed * dt - 0.5 * decel * dt * dt
        else:
            dist = speed * dt
        return (self.xy[0] + vx / speed * dist, self.xy[1] + vy / speed * dist)


@dataclass(frozen=True)
class TrackerParams:
    gate_rad: float = 0.35         # a detection this close in bearing can update a track
    gate_range_frac: float = 0.6   # …if its range is within this fraction, too
    # Weight of the new measurement. MEASURED as already optimal, and the
    # sweep is worth keeping because the obvious intuition is wrong: heavier
    # averaging makes a STILL ball worse, not better. Reading `Track.xy`
    # against truth over 3 seeds of 1v1 (still ball inside 0.6 m / a ball
    # rolling above 0.3 m/s): 1.00 -> 3.42 / 9.75 cm, 0.60 -> 2.70 / 8.90,
    # 0.30 -> 4.36 / 14.26, 0.15 -> 8.89 / 14.40.
    #
    # The mechanism is the FRAME. This smooths bearing and range, in the
    # BODY frame, and the body walks and turns - so a ball stationary in the
    # WORLD still sweeps quickly in bearing, and averaging it lags. Past
    # ~0.6 the lag costs more than the noise it removes. "Still ball"
    # describes the world, not the measurement.
    #
    # The version the frame argument does NOT kill - smooth `xy` itself, in
    # the odometry frame where a stationary ball genuinely is stationary -
    # was measured too, as an EMA of the raw per-frame xy (still / rolling):
    # shipped 2.70 / 8.90 cm, xy-EMA a=0.60 2.41 / 10.37, a=0.30 2.19 /
    # 17.01, a=0.15 3.24 / 31.39. It works as predicted (19% off a still
    # ball, and unlike polar smoothing it keeps improving past a=0.6) and it
    # is still not worth building: it costs heavily on a rolling ball so it
    # needs gating on `vel`, and 0.5 cm off a 5.5 cm placement error is
    # below what the soccer benchmark can resolve (~200 seeds for +0.3
    # goals). Recorded so nobody re-derives it. docs/camera-hardware.md 3c.
    smooth: float = 0.6
    coast_s: float = 2.5           # a track survives this long without a hit
    # …unless it has been measured AT REST, in which case it survives
    # `rest_coast_s` (roadmap Track 4 item 12f). The floor has had rolling
    # resistance since 2026-09-06 - a ball that stops STAYS stopped - but the
    # tracker still forgot it on the same 2.5 s clock as a ball that might
    # have rolled anywhere, and the kick plan is a median 3.6 s old when the
    # swing fires. So at the moment that decides the kick, the brain is
    # reasoning about a ball whose track expired a second earlier, which is
    # why the ahead gate could only fire on 5% of swings on its own.
    # `rest_vel` is what counts as still. 0 = off, the 2.5 s clock for
    # everything, which is every number measured before this.
    rest_vel: float = 0.05
    rest_coast_s: float = 0.0
    confirm_hits: int = 2          # hits before a brain should trust it
    vel_smooth: float = 0.5        # weight of a new velocity sample (hits 0.05-1 s apart)
    vel_min_dt: float = 0.05
    vel_max_dt: float = 1.0
    # The detector's declared noise (a `DetectorNoise` preset; `for_detector`
    # reads one), which is what a track's `sig_meas` is built from: bearing
    # sigma x range, and the width-ranged distance's own fraction (range
    # is radius / tan(width / 2), so a 10% width error is a 10% range
    # error). `meas_floor` is what a perfect detector still carries - the
    # frame's age and the body's motion between frame and pose, measured
    # 1.8 cm in `_place`'s note. `vel_prior` is how fast a ball with no
    # measured velocity may be moving: most sightings are of a still one.
    #
    # CALIBRATED against the truth (scripts/probe_shot_gate.py, 24 seeds x
    # 300 s of 2v2, 15 730 sampled estimates). The error of a 2-D estimate
    # is radial, so a calibrated per-axis sigma puts 39% of errors inside
    # 1 sigma and 86% inside 2 (not 68 / 95). First draft: the sensor
    # sigma shrunk by the polar smoothing's sqrt(k/(2-k)), and a 0.15 m/s
    # prior - fresh hits 30% / 77% (too small: the smoothing does NOT
    # reduce the placement error, the body frame lags, see `smooth`), and
    # hits 0.3-1.0 s old 61-79% inside 1 sigma (far too large: a ball not
    # seen for a second has mostly not moved). So: the raw sensor sigma,
    # and 0.06 m/s. Overall r(sigma, error) 0.41, 0.49 on fresh hits.
    meas_bearing_sigma: float = math.radians(1.0)
    meas_range_frac: float = 0.10
    meas_floor: float = 0.02
    vel_prior: float = 0.06
    # The two refinements the calibration named (roadmap C.1): the sensor
    # sigma times `meas_scale`, and the velocity's own scatter used to grow
    # the sigma only once the hit is older than `vel_sig_after_s` (before
    # that the prior: a track WITH a velocity was growing by its samples'
    # scatter while the ball stood still). MEASURED (each alone on the
    # discovery block, both together on the fresh block, 15 694 samples):
    # against the first model's 55% inside 1 sigma / 95% inside 2 (a
    # calibrated radial error: 39 / 86) and r(sigma, error) 0.41, the pair
    # gives 32% / 82% and r 0.57 - from a fifth too wide to a little too
    # tight, with the sigma tracking the error far better (r 0.70-0.75
    # on hits older than 0.3 s). What is left is the floor: a ball unseen
    # for 0.3-0.6 s is usually NEAR, its range-proportional sigma shrinks
    # to 0.042 while its error stays 0.057 - the near-ball error is the
    # frame's age and the body's motion, not the range. A 4 cm floor is
    # the next probe run.
    meas_scale: float = 0.8
    vel_sig_after_s: float = 1.0
    # THE COASTING TRACK'S BEARING (roadmap 12af's "recorded, not built").
    # On, a track that has a position and did not hit this frame has its
    # `bearing` AND `range` re-read off `xy` for the pose the body has now
    # (`Track.bearing_from` / `range_from`), instead of only being turned by
    # the body's yaw. Off (the default, and every number measured before
    # this), the yaw rotation alone - so a duck that WALKS while coasting
    # carries a bearing that is 112 degrees off after a walk (12af's per-tick
    # trace) and a range that is short or long by however far it walked,
    # while `xy` beside them is fine.
    #
    # It is one knob for both fields because they are one estimate: a brain
    # that reads `bearing` and `range` off the same track is asking where the
    # ball is, and answering half of that from the remembered position and
    # half from the pose at the last hit would be worse than either. At the
    # moment of a hit both are the exact inverse of `_place`, so turning this
    # on changes NOTHING until the body moves.
    coast_from_xy: bool = False
    # Detection classes the tracker never turns into tracks: landmarks. The
    # goal posts (`post`) are for the localiser (brain/localize.py), which
    # reads them off the frame; as tracks they would only cost association
    # time and shift every other track's id.
    ignore: tuple[str, ...] = ("post",)

    @classmethod
    def for_detector(cls, preset: str | None, **kw) -> "TrackerParams":
        """The tracker a duck with this detector preset should run: its
        uncertainty model is the detector's datasheet, nothing else moves.

        `MICRODUCK_TRACKER` (see `env_over`) fills in the knobs the CALLER
        did not name, so a battery can say which tracker variant it is
        measuring from the command line - the same job `MICRODUCK_CHASE`
        does for the brain. The caller always wins: `brain/controllers.py`
        hands `rest_coast_s` / `rest_vel` down from `ChaseParams`, and those
        stay that one spec's business."""
        from ..sensors.detector import DetectorNoise  # noqa: PLC0415
        nz = DetectorNoise.preset(preset or "ideal")
        kw = dict(meas_bearing_sigma=float(nz.bearing_sigma_rad),
                  meas_range_frac=float(nz.width_sigma_frac), **kw)
        return cls(**kw, **cls.env_over(kw))

    @classmethod
    def env_over(cls, explicit: dict | None = None, spec: str | None = None) -> dict:
        """The knobs `MICRODUCK_TRACKER` sets, minus any the caller already
        named in `explicit`:

            MICRODUCK_TRACKER="coast_from_xy=1" uv run eval-pitch ...

        Same contract as `ChaseParams.from_env`: an unknown name or an
        unreadable value RAISES rather than silently measuring the default,
        and tuple-valued knobs are not settable this way. Not applied by
        `TrackerParams()` itself - a bare constructor stays the shipped
        defaults, so tests and goldens do not move under an env var."""
        if spec is None:
            spec = os.environ.get("MICRODUCK_TRACKER", "")
        if not spec.strip():
            return {}
        base = cls()
        kinds = {f.name for f in fields(base)}
        over: dict = {}
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            k, sep, v = item.partition("=")
            k, v = k.strip(), v.strip()
            if not sep or k not in kinds:
                raise ValueError(f"MICRODUCK_TRACKER: {item!r} is not <TrackerParams field>=<value>")
            if explicit and k in explicit:
                continue                                   # the caller named it; it wins
            cur = getattr(base, k)
            if isinstance(cur, bool):
                if v.lower() not in ("0", "1", "true", "false", "on", "off"):
                    raise ValueError(f"MICRODUCK_TRACKER: {k}={v!r} is not a boolean")
                over[k] = v.lower() in ("1", "true", "on")
            elif isinstance(cur, int):
                over[k] = int(v)
            elif isinstance(cur, tuple):
                raise ValueError(f"MICRODUCK_TRACKER: {k} is a tuple; set it in code")
            else:
                over[k] = float(v)
        return over


class Tracker:
    def __init__(self, p: TrackerParams = TrackerParams()):
        self.p = p
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_frame_t: float | None = None
        self._prev_yaw: float | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self._last_frame_t = None
        self._prev_yaw = None

    def update(self, frame: DetectionFrame | None, t: float, yaw: float | None = None,
               pos: tuple[float, float] | None = None) -> list[Track]:
        """Fold one detection frame (or None) at time t. `yaw` is the body
        heading now: bearings of coasting tracks turn with the body. With
        `pos` (the body's odometry position) each hit also places the track
        in the odometry frame and feeds its velocity.

        With `coast_from_xy` and a `pos`, a track that HAS a position gets
        its bearing and range re-read off that position for the pose now
        (`Track.bearing_from` / `range_from`) instead of only turning with
        the yaw - so a coasting track's polar pair survives a walk, not just
        a turn. A hit below still replaces both with the measurement."""
        p = self.p
        from_xy = p.coast_from_xy and yaw is not None and pos is not None
        if from_xy:
            for tr in self.tracks:
                if tr.xy is None:
                    continue
                tr.bearing = tr.bearing_from(pos, yaw)
                tr.range = tr.range_from(pos)
        if yaw is not None and self._prev_yaw is not None:
            dyaw = math.atan2(math.sin(yaw - self._prev_yaw), math.cos(yaw - self._prev_yaw))
            if dyaw:
                for tr in self.tracks:                  # a hit below replaces this with the measurement
                    if from_xy and tr.xy is not None:
                        continue                        # already read off `xy`, yaw and all
                    tr.bearing = math.atan2(math.sin(tr.bearing - dyaw), math.cos(tr.bearing - dyaw))
        self._prev_yaw = yaw
        if frame is not None and frame.t != self._last_frame_t:
            self._last_frame_t = frame.t
            hit = self._associate(frame.detections, frame.t, getattr(frame, "cam_yaw", 0.0))
            if pos is not None and yaw is not None:
                for tr in hit:
                    self._place(tr, frame.t, pos, yaw)
        self.tracks = [tr for tr in self.tracks
                       if t - tr.last_t <= (p.rest_coast_s
                                            if (p.rest_coast_s > 0.0 and tr.at_rest(p.rest_vel))
                                            else p.coast_s)]
        return self.tracks

    def disturb(self, cls: str, xy: tuple[float, float] | None = None,
                radius: float = 0.0) -> int:
        """Something may have moved it: stop these tracks counting as at rest
        until a fresh sighting. With `xy` and `radius`, only the tracks whose
        remembered position is inside that circle; without, every track of
        the class. Returns how many were marked.

        This is the honest half of a resting-ball memory. Keeping a ball that
        nothing has touched is a good prior; keeping one that a foot just went
        through is a duck swinging at a place where the ball is not.
        """
        n = 0
        for tr in self.tracks:
            if tr.cls != cls or tr.rest_block:
                continue
            if xy is not None and radius > 0.0 and tr.xy is not None:
                if math.dist(tr.xy, xy) > radius:
                    continue
            tr.rest_block = True
            n += 1
        return n

    def _place(self, tr: Track, t: float, pos: tuple[float, float], yaw: float) -> None:
        """A hit: the track's odometry-frame position, and a velocity sample
        against the previous hit when the two are usefully apart in time.

        KNOWN BIAS, not yet fixed here. `pos`/`yaw` are the pose the caller
        has NOW, while `t` is the frame's timestamp - so a hit is anchored
        where the duck is rather than where it was when the picture was
        taken, and `xy` carries about (duck speed x detector period) of
        error in the direction of travel: ~3 cm at 10 Hz, ~6 cm at 5 Hz.
        `brain/tidy.py`'s `stale_fix` was the same bug in the same shape and
        has now been measured. THE RESULT IS NOT "FIX IT HERE TOO".
        Correcting the placement alone LOST 0.38 toys (p = 0.031) even
        though it cut the estimate's error from 5.4 cm to 3.7 cm, because
        the stop distance downstream had been hand-fitted against the
        biased estimate and absorbed it. Only correcting BOTH won
        (+0.44 toys, grasp 88% -> 93%). See AGENTS.md rule 7.

        MEASURED HERE, and the answer is DON'T. 12 348 ball sightings over
        2 seeds x 180 s of 1v1, each placement checked against truth and
        split into what moved between the frame and now:

            frame age          100 ms (median)
            placement error    7.7 cm   <- what the brain acts on
            duck moved since   1.8 cm   <- THIS bias
            ball moved since   0.7 cm   <- what predict() is for

        and in the line-up case that actually scores (ball inside 0.6 m and
        nearly still, 6 253 of those sightings): 5.5 cm of error, 1.7 cm of
        it from the stale pose. **The error is dominated by neither term -
        it is the detector's own bearing and range noise.** Fixing the pose
        removes at most a third of it, and by rule 7 it would ALSO move the
        line-up off the constants fitted around it. Bad trade; not done.

        (An earlier version of this note said the ball moves 1.4 m/s and so
        `predict()` dominates. That is the peak right after a kick, not the
        operating point: at the median the ball has moved 0.7 cm since the
        frame. Most sightings are of a nearly stationary ball.)

        VELOCITY is less affected - both samples carry a similar error, so
        it largely cancels in the difference - but the POSITION does not,
        and `Track.xy` is what a dead-reckoned approach steers by.
        """
        p = self.p
        a = yaw + tr.bearing
        xy = (pos[0] + tr.range * math.cos(a), pos[1] + tr.range * math.sin(a))
        if tr.xy is not None:
            dt = t - tr.xy_t
            if p.vel_min_dt <= dt <= p.vel_max_dt:
                sample = ((xy[0] - tr.xy[0]) / dt, (xy[1] - tr.xy[1]) / dt)
                k = p.vel_smooth if tr.vel_hits else 1.0
                tr.vel = (tr.vel[0] + k * (sample[0] - tr.vel[0]), tr.vel[1] + k * (sample[1] - tr.vel[1]))
                tr.vel_hits += 1
                # How much the samples disagree with the smoothed velocity:
                # the velocity's own uncertainty (roadmap C.1).
                dev = math.hypot(sample[0] - tr.vel[0], sample[1] - tr.vel[1])
                tr.vel_sig = dev if tr.vel_hits <= 1 else tr.vel_sig + 0.5 * (dev - tr.vel_sig)
            elif dt > p.vel_max_dt:
                tr.vel, tr.vel_hits, tr.vel_sig = (0.0, 0.0), 0, 0.0        # too long ago to say
        tr.xy, tr.xy_t = xy, t
        # This placement's own error (roadmap C.1): the detector's bearing
        # and range noise at this range, never under the floor. Not shrunk
        # by the smoothing: measured, it does not reduce the error.
        tr.sig_meas = max(p.meas_floor,
                          p.meas_scale * math.hypot(p.meas_bearing_sigma * tr.range, p.meas_range_frac * tr.range))

    def _associate(self, dets: list[Detection], t: float, cam_yaw: float = 0.0) -> list[Track]:
        """Detections come in the CAMERA's frame; tracks are kept in the
        BODY's (a brain steers the body), so `cam_yaw` — where the head
        was looking — is added on the way in."""
        p = self.p
        used: set[int] = set()
        hit: set[int] = set()
        dets = [d for d in dets if d.cls not in p.ignore]
        body = [math.atan2(math.sin(d.bearing + cam_yaw), math.cos(d.bearing + cam_yaw)) for d in dets]
        # Greedy nearest-first: best pairs first, one detection per track.
        pairs = []
        for i, d in enumerate(dets):
            for tr in self.tracks:
                if tr.cls != d.cls:
                    continue
                db = abs(math.atan2(math.sin(body[i] - tr.bearing), math.cos(body[i] - tr.bearing)))
                if db > p.gate_rad:
                    continue
                if abs(d.range_est - tr.range) > p.gate_range_frac * max(tr.range, 0.3):
                    continue
                pairs.append((db, i, tr.id))
        pairs.sort()
        for db, i, tid in pairs:
            if i in used or tid in hit:
                continue
            tr = next(x for x in self.tracks if x.id == tid)
            d = dets[i]
            k = p.smooth
            tr.bearing = tr.bearing + k * math.atan2(math.sin(body[i] - tr.bearing), math.cos(body[i] - tr.bearing))
            tr.elevation += k * (d.elevation - tr.elevation)
            tr.width += k * (d.width - tr.width)
            tr.range += k * (d.range_est - tr.range)
            tr.conf = max(d.conf, 0.7 * tr.conf)
            tr.hits += 1
            tr.misses = 0
            tr.last_t = t
            tr.rest_block = False          # a fresh look settles it either way
            if d.name:
                tr.names[d.name] = tr.names.get(d.name, 0) + 1
                tr.name = max(tr.names, key=tr.names.get)
            if d.color:
                tr.colors[d.color] = tr.colors.get(d.color, 0) + 1
                tr.color = max(tr.colors, key=tr.colors.get)
            used.add(i)
            hit.add(tid)
        for tr in self.tracks:
            if tr.id not in hit:
                tr.misses += 1
        born = []
        for i, d in enumerate(dets):
            if i in used:
                continue
            tr = Track(self._next_id, d.cls, body[i], d.elevation, d.width, d.range_est, d.conf,
                       t, t, name=d.name, names={d.name: 1} if d.name else {},
                       colors={d.color: 1} if d.color else {}, color=d.color)
            self.tracks.append(tr)
            born.append(tr)
            self._next_id += 1
        return [tr for tr in self.tracks if tr.id in hit] + born

    def best(self, cls: str, t: float, min_hits: int | None = None) -> Track | None:
        """The track of `cls` to act on: confirmed, freshest, then nearest."""
        need = self.p.confirm_hits if min_hits is None else min_hits
        cands = [tr for tr in self.tracks if tr.cls == cls and tr.hits >= need]
        if not cands:
            return None
        return min(cands, key=lambda tr: (round(tr.age(t), 1), tr.range))

    def payload(self, t: float) -> list[dict]:
        return [{"id": tr.id, "cls": tr.cls, "name": tr.name, "bearing": round(tr.bearing, 3),
                 "range": round(tr.range, 3), "hits": tr.hits, "age": round(tr.age(t), 2),
                 **({"xy": [round(tr.xy[0], 3), round(tr.xy[1], 3)],
                     "vel": [round(tr.vel[0], 2), round(tr.vel[1], 2)]} if tr.xy is not None else {})}
                for tr in self.tracks]


__all__ = ["Track", "Tracker", "TrackerParams"]
