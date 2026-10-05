"""MOSS's arm moves between poses by ROUTES that clear its own body, timed
minimum-jerk (2026-09-28).

Asked on /sim: the arm "clips through the box going into the rest position"
and "gets stuck as it sweeps out to grab something". MEASURED over eight
7-minute moss-yard runs (the arm's VISUAL meshes against the bin and hull,
plus physical contacts and stalled joints): the sweep out to the grasp pose
clipped 50% of its time and sat stalled 92 s; the learned approach, which
moves the arm while driving, clipped 50% and pressed the rover at up to 292 N;
and the rest pose itself had the gripper's mesh ON the bin's front wall, so
every move out of it began in contact. The stow, the lift, the release and
the planned fold were clean.

Three pieces:

* `ArmClearance` — how close the VISIBLE arm (its visual meshes' convex
  hulls) comes to the rover's hull, bin walls and bin floor, plus the
  collision proxies touching them and the jaws inside the bin's clutter layer.
  The visual meshes because the proxies are thinner than the arm (26 mm
  against a 70 mm forearm): a route that only clears the proxies lets the arm
  you see sink 2-5 cm into the bin. The robot's own kinematics, as
  `robots/moss_pinch.MossKinematics` — nothing reads the world.
* `route` — straight in joint space if every sample of the segment keeps
  `margin`; else through the first HUB that joins both ends clearly. A pose
  that starts below the margin (a jammed arm) may leave at its own clearance.
* `MinJerkRoute` — each segment a quintic (zero velocity and acceleration at
  both ends), timed so the busiest joint peaks at `vmax`, all joints arriving
  together; followed on a LEASH (the command never leads the arm by more than
  `leash`), so a joint that meets something cannot wind up and snap.
"""
from __future__ import annotations

import numpy as np

from ..robots import moss

ARM = list(moss.ARM_JOINTS)
_ARM_BODIES = ("upper_arm", "lower_arm", "wrist", "gripper", "finger")

#: THE FARTHEST `mj_geomDistance` MAY BE ASKED, m. Past this it answers 0.0 —
#: *touching* — for mesh/box pairs that are plainly apart, and the caller has
#: no way to tell that from a real contact. MEASURED 2026-09-28 over 4488
#: (arm geom, hull geom, pose) queries, counting a cutoff's zeros that some
#: SMALLER cutoff had already reported as "farther than me": none at 0.02,
#: 0.11% at 0.03, 0.56% at 0.10. The worst single case: `visual_upper_arm_link_0`
#: against `bin_x1` at the rest pose reads 0.0 at any cutoff >= 0.10 while its
#: nearest vertex is **85 mm** from that box and none is inside it.
#: It cost route availability, not safety — a false 0.0 fails `segment`, so a
#: clear path is refused and the old ramp runs. At the shipped 0.03 it refused
#: three of the pose pairs the brain routes between (`lift <-> fold waypoint`,
#: `retract waypoint -> grasp`) outright. Every margin here is <= 0.015, so
#: 0.02 loses nothing: `cap` is clamped to it rather than trusted.
CAP_MAX = 0.02


class ArmClearance:
    """Clearance of the visible arm from MOSS's own body, base frame."""

    def __init__(self, kin=None):
        import mujoco

        from ..robots.moss_pinch import MossKinematics
        self.mj = mujoco
        self.kin = kin or MossKinematics()
        m = self.m = self.kin.m
        self.d = mujoco.MjData(m)
        def name(i):
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""

        self.vis = [g for g in range(m.ngeom)
                    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2
                    and any(k in name(int(m.geom_bodyid[g])) for k in _ARM_BODIES)]
        self.obst = [m.geom(n).id for n in moss.HULL_GEOMS]
        self.arm_g = {i for i in range(m.ngeom) if m.geom_contype[i] == 2}
        self.env_g = {i for i in range(m.ngeom) if m.geom_contype[i] == 1}
        self.planes = {i for i in range(m.ngeom)
                       if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
        self.qa = [m.joint(j).qposadr[0] for j in ARM]
        self.fa = [m.joint(j).qposadr[0] for j in moss.FINGER_JOINTS]
        self.rover = m.body(moss.BASE_BODY).id
        self.pads = [m.geom(n).id for n in ("pad_left", "pad_right")]
        self.lid = moss.BIN_FLOOR_Z + 0.08          # clutter up to 8 cm deep

    def clearance(self, q, jaw: float = 0.041, cap: float = CAP_MAX) -> float:
        """Metres of air between the visible arm and the rover (capped at
        `cap`, itself capped at `CAP_MAX`); negative when inside it, -1 for a
        proxy contact, -0.2 for jaws down in the bin's clutter."""
        cap = min(float(cap), CAP_MAX)
        mj, m, d = self.mj, self.m, self.d
        d.qpos[self.qa] = q
        d.qpos[self.fa] = jaw
        mj.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if ((g1 in self.arm_g and g2 in self.env_g)
                    or (g2 in self.arm_g and g1 in self.env_g)) \
                    and g1 not in self.planes and g2 not in self.planes:
                return -1.0
        R = d.xmat[self.rover].reshape(3, 3)
        for g in self.pads:
            p = R.T @ (d.geom_xpos[g] - d.xpos[self.rover])
            if (moss.BIN_INTERIOR_X[0] < p[0] < moss.BIN_INTERIOR_X[1]
                    and moss.BIN_INTERIOR_Y[0] < p[1] < moss.BIN_INTERIOR_Y[1]
                    and p[2] < self.lid):
                return -0.2
        w = cap
        for g in self.vis:
            for o in self.obst:
                w = min(w, mj.mj_geomDistance(m, d, g, o, cap, None))
                if w < 0:
                    return w
        return w

    def segment(self, a, b, need: float, step: float = 0.04,
                start_at: float | None = None) -> bool:
        """Does every sample of a -> b keep `need`? Near `a` (0.15 rad) a
        start that is itself below `need` may keep only its own clearance."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.abs(b - a).max() / step) + 1)
        for k in range(n):
            q = a + (b - a) * k / (n - 1)
            want = need
            if start_at is not None and np.abs(q - a).max() < 0.15:
                want = min(need, start_at - 1e-4)
            if self.clearance(q) < want:
                return False
        return True

    def route(self, start, goal, hubs=(), margin: float = 0.01):
        """[start, (hub,) goal] clear by `margin`, or None."""
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        c0 = self.clearance(start)
        if self.segment(start, goal, margin, start_at=c0):
            return [start, goal]
        for h in hubs:
            h = np.asarray(h, float)
            if (self.segment(start, h, margin, start_at=c0)
                    and self.segment(h, goal, margin)):
                return [start, h, goal]
        return None


def quintic(k: float) -> float:
    """Minimum-jerk progress 0 -> 1: zero velocity and acceleration at both ends."""
    k = min(1.0, max(0.0, k))
    return k * k * k * (10.0 - 15.0 * k + 6.0 * k * k)


class MinJerkRoute:
    """Waypoints in joint space, each segment a quintic timed so the busiest
    joint peaks at `vmax` (a quintic's peak is 1.875x its mean)."""

    def __init__(self, waypoints, vmax: float = 1.5, min_seg_s: float = 0.3):
        self.w = [np.asarray(q, float) for q in waypoints]
        self.dur = [max(min_seg_s, 1.875 * float(np.abs(b - a).max()) / vmax)
                    for a, b in zip(self.w, self.w[1:])]
        self.total = float(sum(self.dur))

    def at(self, t: float) -> np.ndarray:
        for (a, b), T in zip(zip(self.w, self.w[1:]), self.dur):
            if t < T:
                return a + (b - a) * quintic(t / T)
            t -= T
        return self.w[-1].copy()

    def done(self, t: float) -> bool:
        return t >= self.total


class RouteFollower:
    """Follow a `MinJerkRoute` on the path it was CHECKED on.

    MEASURED in the room, the first cut led each joint by at most the leash
    independently: the base joint (which swings the whole arm, and started
    with a jaw on the bin's side wall) lagged while elbow and wrist ran on
    schedule, the arm left the planned line and clipped the bin 7 mm deep in
    a pose nobody had checked — and the fold timed out there. So the route's
    CLOCK advances only while every joint is within `leash` of the plan at
    the new time; otherwise it waits. All joints stay on the verified path,
    together, and a blocked arm holds a command within `leash` of itself
    instead of winding up.
    """

    def __init__(self, route: MinJerkRoute, leash: float):
        self.route, self.leash = route, leash
        self.tau = 0.0
        self._t = None

    def step(self, t: float, arm_now: dict | None) -> np.ndarray:
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        cand = self.route.at(self.tau + dt)
        if arm_now:
            q = np.array([float(arm_now.get(j, c)) for j, c in zip(ARM, cand)])
            if np.abs(cand - q).max() <= self.leash:
                self.tau += dt
            else:
                return self.route.at(self.tau)
        else:
            self.tau += dt
        return cand

    def done(self) -> bool:
        return self.route.done(self.tau)


def leashed(cmd: dict, arm_now: dict | None, leash: float) -> dict:
    """The command no more than `leash` from where each joint IS."""
    if not arm_now:
        return cmd
    out = dict(cmd)
    for j in ARM:
        if j in out and j in arm_now:
            q = float(arm_now[j])
            out[j] = float(np.clip(out[j], q - leash, q + leash))
    return out


__all__ = ["CAP_MAX", "ArmClearance", "MinJerkRoute", "RouteFollower",
           "leashed", "quintic"]
