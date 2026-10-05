"""MOSS's model of the room: what it has seen, where it has looked, and where
to go when nothing is in view (2026-09-28).

Asked on /sim with one bottle cap left in `moss-yard`: "didn't it see that
cap before? why doesn't it go over there? do we need a map?" MOSS had no
memory of objects — only the one it was going for — and when nothing was in
view it spun on the spot, forever, unless a one-frame phantom detection (the
detector's modelled false positive, ~1 per 20 s) sent it across the yard.
Measured over two 15-minute runs: once only small objects were left, 10 of
the approaches were at phantoms, and the only way it ever reached the cap was
by chasing them.

Three pieces, all in the brain's ODOMETRY frame and built only from what the
robot has (its detections and its own pose):

* `ObjectMemory` — every toy seen, kept after it leaves the view. A sighting
  is CONFIRMED on its second detection within `gate_m`, which is also what
  keeps a phantom out: it never recurs in the same place. An entry is
  forgotten when the camera looks right at where it should be, from close
  enough that it would be found on every frame, and finds nothing.
* `Coverage` — when each patch of floor was last in the camera's view,
  within `cover_m`. For the map on /sim, and for choosing where to look.
* `Patrol` — when a full turn on the spot shows nothing: drive to the
  nearest remembered object and look at it from close up; with nothing
  remembered, drive the rim of the work area, `inset_m` in from its walls,
  turning to face the middle at every waypoint.

The WORK AREA was declared at first — the scenario's wall rectangle handed to
the brain, because the RealSense's depth was not modelled. Asked on /sim the
same day ("is it drawing those walls from the actual environment?"), it now
comes from MOSS's own sensing:

* `RoomMap` — an occupancy grid built from the depth scan
  (`robots/moss.DEPTH_SCAN_RAYS`: the D455f's depth row at the lens's 75 mm),
  in the odometry frame: every return marks its cell occupied and the cells
  the ray passed through free. The work area is the rectangle round what is
  solid, once the scans have swept most of a turn. And what the depth cannot
  see — anything under its 75 mm slice or inside its 0.52 m near limit — is
  FELT: a patrol leg that stalls against something marks it.

The declared area survives only as the fallback for a MOSS with no depth
(`tof: null`), and the /sim map says which one it is drawing. The map is
built on odometry, which is ideal in `moss-yard`; with drift the walls would
smear, which is what SLAM would correct and this does not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Seen:
    """One remembered object, odometry frame."""
    x: float
    y: float
    size: float
    hits: int
    first: float
    last: float
    misses: int = 0
    id: int = 0
    #: Sum of the sightings' weights (1 / sigma^2): a close look outweighs
    #: many distant ones.
    wsum: float = 0.0


class ObjectMemory:
    """Toys seen, kept after they leave the view."""

    #: A sighting's position error, m, at range r: the detector ranges an
    #: object from its apparent width, which is ~10% noisy, so the error is
    #: mostly ALONG the line of sight and grows with range. The first cut used
    #: a flat 0.15 m gate and a plain mean, and one can at 1.5 m became a
    #: streak of five "remembered objects" on the /sim map.
    SIGMA0, SIGMA_PER_M = 0.02, 0.08
    #: Weight cap, so a pushed object's entry can still follow it.
    WSUM_CAP = 5.0 / (0.02 + 0.08 * 0.3) ** 2

    def __init__(self, gate_m: float = 0.10, confirm: int = 2,
                 tentative_s: float = 8.0, miss_frames: int = 5,
                 gate_per_m: float = 0.20):
        self.gate_m = gate_m
        self.gate_per_m = gate_per_m
        self.confirm = confirm
        self.tentative_s = tentative_s
        self.miss_frames = miss_frames
        self.items: list[Seen] = []
        self._next = 1

    def observe(self, pts: list[tuple[float, float, float, float]], t: float) -> list[Seen]:
        """Fold one frame's sightings (x, y, size, range) in; returns the
        entry each one matched or made, in order."""
        out: list[Seen] = []
        taken: set[int] = set()
        for x, y, size, rng in pts:
            wi = 1.0 / (self.SIGMA0 + self.SIGMA_PER_M * rng) ** 2
            best, bd = None, self.gate_m + self.gate_per_m * rng
            for e in self.items:
                if e.id in taken:
                    continue
                d = math.hypot(e.x - x, e.y - y)
                if d < bd:
                    best, bd = e, d
            if best is None:
                best = Seen(x, y, size, 0, t, t, id=self._next, wsum=wi)
                self._next += 1
                self.items.append(best)
            else:
                k = wi / (best.wsum + wi)
                best.x += k * (x - best.x)
                best.y += k * (y - best.y)
                best.size += k * (size - best.size)
                best.wsum = min(best.wsum + wi, self.WSUM_CAP)
            best.hits += 1
            best.last = t
            best.misses = 0
            taken.add(best.id)
            out.append(best)
        self._merge()
        return out

    def _merge(self) -> None:
        """Two entries that have converged on one spot are one object."""
        i = 0
        while i < len(self.items):
            a = self.items[i]
            j = i + 1
            while j < len(self.items):
                b = self.items[j]
                if math.hypot(a.x - b.x, a.y - b.y) < self.gate_m:
                    w = a.wsum + b.wsum
                    a.x = (a.x * a.wsum + b.x * b.wsum) / w
                    a.y = (a.y * a.wsum + b.y * b.wsum) / w
                    a.size = (a.size * a.wsum + b.size * b.wsum) / w
                    a.wsum = min(w, self.WSUM_CAP)
                    a.hits += b.hits
                    a.first, a.last = min(a.first, b.first), max(a.last, b.last)
                    del self.items[j]
                else:
                    j += 1
            i += 1

    def confirmed(self, e: Seen) -> bool:
        return e.hits >= self.confirm

    def near(self, x: float, y: float, gate: float | None = None) -> Seen | None:
        g = self.gate_m if gate is None else gate
        best, bd = None, g
        for e in self.items:
            d = math.hypot(e.x - x, e.y - y)
            if d < bd:
                best, bd = e, d
        return best

    def forget_near(self, x: float, y: float, r: float) -> None:
        self.items = [e for e in self.items if math.hypot(e.x - x, e.y - y) >= r]

    def looked(self, expected: list[Seen], seen_ids: set[int], t: float) -> None:
        """Entries the camera should have found this frame (in view and close
        enough for their size) and did not: after `miss_frames` of those in a
        row the thing is not there any more."""
        for e in expected:
            if e.id not in seen_ids:
                e.misses += 1
        self.items = [e for e in self.items if e.misses < self.miss_frames
                      and (self.confirmed(e) or t - e.last < self.tentative_s)]


@dataclass
class Coverage:
    """When each cell of the work area was last in the camera's view."""
    area: tuple[float, float, float, float]      # xmin, xmax, ymin, ymax
    cell_m: float = 0.2
    last: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.nx = max(1, int(round((self.area[1] - self.area[0]) / self.cell_m)))
        self.ny = max(1, int(round((self.area[3] - self.area[2]) / self.cell_m)))
        self.last = [-1e9] * (self.nx * self.ny)

    def centre(self, i: int, j: int) -> tuple[float, float]:
        return (self.area[0] + (i + 0.5) * (self.area[1] - self.area[0]) / self.nx,
                self.area[2] + (j + 0.5) * (self.area[3] - self.area[2]) / self.ny)

    def mark(self, cam_xy: tuple[float, float], yaw: float, half_fov: float,
             near_m: float, far_m: float, t: float) -> None:
        for j in range(self.ny):
            for i in range(self.nx):
                cx, cy = self.centre(i, j)
                dx, dy = cx - cam_xy[0], cy - cam_xy[1]
                r = math.hypot(dx, dy)
                if not near_m <= r <= far_m:
                    continue
                b = math.atan2(dy, dx) - yaw
                b = math.atan2(math.sin(b), math.cos(b))
                if abs(b) <= half_fov:
                    self.last[j * self.nx + i] = t

    def ages(self, t: float, cap_s: float = 99.0) -> list[int]:
        """Seconds since each cell was seen, capped, row-major from ymin."""
        return [int(min(t - v, cap_s)) for v in self.last]


class Patrol:
    """Where to go when a full turn shows nothing."""

    def __init__(self, area: tuple[float, float, float, float] | None,
                 inset_m: float = 0.45):
        self.area = area
        self.waypoints: list[tuple[float, float]] = []
        if area is not None:
            x0, x1, y0, y1 = area
            x0, x1, y0, y1 = x0 + inset_m, x1 - inset_m, y0 + inset_m, y1 - inset_m
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            # Around the rim, counter-clockwise: corners and edge midpoints.
            self.waypoints = [(x0, y0), (xm, y0), (x1, y0), (x1, ym),
                              (x1, y1), (xm, y1), (x0, y1), (x0, ym)]
        self.i: int | None = None

    @property
    def centre(self) -> tuple[float, float] | None:
        if self.area is None:
            return None
        return ((self.area[0] + self.area[1]) / 2, (self.area[2] + self.area[3]) / 2)

    def next_waypoint(self, x: float, y: float,
                      blocked=None) -> tuple[float, float] | None:
        """The next rim waypoint: the nearest one the first time, then on
        round the loop, skipping any `blocked(x, y)` says is in something."""
        if not self.waypoints:
            return None
        n = len(self.waypoints)
        if self.i is None:
            order = sorted(range(n), key=lambda k: math.hypot(
                self.waypoints[k][0] - x, self.waypoints[k][1] - y))
        else:
            order = [(self.i + k) % n for k in range(1, n + 1)]
        for k in order:
            if blocked is None or not blocked(*self.waypoints[k]):
                self.i = k
                return self.waypoints[k]
        return None


def area_from_world(world) -> tuple[float, float, float, float] | None:
    """The work area: the scenario's wall rectangle (their bounding box, less
    half a wall), or None when the room has no walls."""
    sc = getattr(world, "scenario", None)
    walls = getattr(sc, "walls", None) or []
    xs, ys = [], []
    for w in walls:
        a, b = getattr(w, "a", None), getattr(w, "b", None)
        if a is None:
            a, b = getattr(w, "start", None), getattr(w, "end", None)
        if a is None or b is None:
            continue
        xs += [float(a[0]), float(b[0])]
        ys += [float(a[1]), float(b[1])]
    if len(xs) < 4:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


class RoomMap:
    """What the depth has shown of the room: a log-odds occupancy grid in the
    odometry frame, `half_m` either side of where the robot started.

    A return adds `hit` to its cell, the cells the ray crossed lose `miss`,
    both clamped, so a thing that moves (an upright can, picked) is cleared by
    the rays that later pass where it stood. A cell is SOLID above `occ`: two
    returns, or one wall seen from two places. Returns the device marks
    invalid (inside 0.52 m, or dropped) are skipped whole — a ray with no
    range says nothing about the free space along it either.
    """

    def __init__(self, centre: tuple[float, float], cell_m: float = 0.05,
                 half_m: float = 4.0, hit: float = 0.85, miss: float = 0.4,
                 lo_min: float = -2.0, lo_max: float = 3.5, occ: float = 1.2,
                 sweep_bins: int = 36, sweep_need: float = 0.9):
        self.cell = cell_m
        self.n = int(round(2 * half_m / cell_m))
        self.x0 = centre[0] - half_m
        self.y0 = centre[1] - half_m
        self.lo = np.zeros((self.n, self.n), np.float32)      # [iy, ix]
        self.hit, self.miss = hit, miss
        self.lo_min, self.lo_max, self.occ = lo_min, lo_max, occ
        #: World bearings the scan has looked along, in `sweep_bins` bins:
        #: the area is not trusted until `sweep_need` of them are seen — a
        #: rectangle round three walls is the wrong room.
        self.swept = np.zeros(sweep_bins, bool)
        self.sweep_need = sweep_need
        #: Things FELT, not seen: (x, y) where a leg stalled. The depth does
        #: not clear these — what it could not see it cannot see past.
        self.felt: list[tuple[float, float]] = []
        self.scans = 0
        self._area: tuple[float, float, float, float] | None = None

    def _ij(self, x, y):
        return (np.floor((np.asarray(x) - self.x0) / self.cell).astype(int),
                np.floor((np.asarray(y) - self.y0) / self.cell).astype(int))

    def update(self, frame, odom: tuple[float, float, float],
               max_range: float | None = None) -> None:
        """Fold one scan (`sensors.lidar.LidarFrame`) in, taken at `odom`. A
        reading at `max_range` is "nothing there": free space, no wall."""
        ok = np.asarray(frame.valid, bool)
        if not ok.any():
            return
        x, y, yaw = odom
        c, s = math.cos(yaw), math.sin(yaw)
        mx, my = (0.0, 0.0) if frame.mount_pos is None else (
            float(frame.mount_pos[0]), float(frame.mount_pos[1]))
        ox, oy = x + c * mx - s * my, y + s * mx + c * my
        a = yaw + np.asarray(frame.angles, np.float64)[ok]
        r = np.asarray(frame.ranges, np.float64)[ok]
        ca, sa = np.cos(a), np.sin(a)
        b = np.floor(((a % (2 * math.pi)) / (2 * math.pi)) * len(self.swept)).astype(int)
        self.swept[np.clip(b, 0, len(self.swept) - 1)] = True
        step = self.cell * 0.5
        k = np.arange(0.0, float(r.max()), step)
        tt = k[None, :]
        free = tt < (r[:, None] - self.cell)
        fx = ox + ca[:, None] * tt
        fy = oy + sa[:, None] * tt
        fi, fj = self._ij(fx[free], fy[free])
        far = r < (1e9 if max_range is None else max_range - 1e-6)
        hi, hj = self._ij(ox + ca * r, oy + sa * r)
        def inside(i, j):
            return (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)

        m = inside(fi, fj)
        fflat = np.unique(fj[m] * self.n + fi[m])
        m = inside(hi, hj) & far
        hflat = np.unique(hj[m] * self.n + hi[m])
        fflat = np.setdiff1d(fflat, hflat, assume_unique=True)
        flat = self.lo.reshape(-1)
        flat[fflat] -= self.miss
        flat[hflat] += self.hit
        np.clip(self.lo, self.lo_min, self.lo_max, out=self.lo)
        self.scans += 1
        self._area = None

    def solid(self) -> np.ndarray:
        """Flat indices (iy * n + ix) of the solid cells."""
        return np.flatnonzero(self.lo.reshape(-1) > self.occ)

    def centre_of(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (self.x0 + (flat % self.n + 0.5) * self.cell,
                self.y0 + (flat // self.n + 0.5) * self.cell)

    @property
    def swept_frac(self) -> float:
        return float(self.swept.mean())

    def area(self) -> tuple[float, float, float, float] | None:
        """The work area MOSS has sensed: the rectangle round everything
        solid, once most of a turn has been scanned; None before."""
        if self._area is None and self.swept_frac >= self.sweep_need:
            xs, ys = self.centre_of(self.solid())
            if xs.size >= 8:
                h = self.cell / 2
                self._area = (float(xs.min()) - h, float(xs.max()) + h,
                              float(ys.min()) - h, float(ys.max()) + h)
        return self._area

    def feel(self, x: float, y: float) -> None:
        self.felt.append((x, y))

    def blocked(self, x: float, y: float, r: float) -> bool:
        """Is anything solid or felt within `r` of (x, y)?"""
        if any(math.hypot(fx - x, fy - y) < r for fx, fy in self.felt):
            return True
        n = int(math.ceil(r / self.cell))
        i, j = self._ij(x, y)
        i0, i1 = max(int(i) - n, 0), min(int(i) + n + 1, self.n)
        j0, j1 = max(int(j) - n, 0), min(int(j) + n + 1, self.n)
        if i0 >= i1 or j0 >= j1:
            return False
        win = self.lo[j0:j1, i0:i1] > self.occ
        if not win.any():
            return False
        jj, ii = np.nonzero(win)
        cx = self.x0 + (ii + i0 + 0.5) * self.cell
        cy = self.y0 + (jj + j0 + 0.5) * self.cell
        return bool((np.hypot(cx - x, cy - y) < r).any())

    def payload(self) -> dict:
        """For the /sim map: the grid's frame and its solid cells."""
        return {"x0": round(self.x0, 3), "y0": round(self.y0, 3),
                "cell": self.cell, "n": self.n,
                "solid": self.solid().tolist(),
                "felt": [[round(x, 3), round(y, 3)] for x, y in self.felt],
                "swept": round(self.swept_frac, 2)}
