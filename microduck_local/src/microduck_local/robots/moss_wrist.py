"""MOSS's WRIST camera — and the front camera's reading of an object's axis
and height — for a robot in a WORLD, computed by the pick env's OWN methods.

`teach-moss_pick-ad9876`, the only pick trained on six shapes, sees four
inputs the world never produced: the object's axis and uprightness
(`moss.OBS_TARGET_AXIS` / `OBS_TARGET_UPRIGHT`, from the wrist camera, else
the front camera), its range from the wrist camera (the target's z slot) and
the height of its top (`moss.OBS_SPARE`). Handed zeros, it measured 4 objects
in the bin over 6 yard seeds against 13 for a can-only pick (2026-09-26).

There is ONE sensor model, not two: `MossTargetSensors` borrows
`MossPickEnv._sense_arm`, `_front_attitude` and `_true_attitude` and gives
them the attributes they read (`data`, `can_body`, `prop`, `driver`, `rng`,
the arm camera's body id and the latency queues). Rate, field of view, range,
noise, dropout, end-on blindness and latency are therefore exactly what the
policy trained under, and a change to the env's model is a change here.

The TARGET is the pickable object nearest the jaws — what the wrist camera is
looking at during a pick; the env has only one object, so it never had to
choose. The caches are cleared when the target changes, so a fix on one
object is never reported as a reading of another.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from . import moss
from . import moss_env as ME


class _Geom:
    """The two numbers the env's methods read off `self.prop`."""

    def __init__(self, shape: str, radius: float, half_height: float,
                 size=None):
        self.shape = shape
        self.radius = float(radius)
        self.half_height = float(half_height)
        #: half extents (a box's long side is what the pinch rolls to)
        self.size = tuple(size) if size is not None else (
            self.radius, self.radius, self.half_height)


def geom_of(prop) -> _Geom:
    """A scenario `Prop` as the env's `GraspProp` geometry. Scenario sizes
    are FULL extents; a cylinder's are (radius, radius, full height)."""
    sx, sy, sz = (tuple(prop.size) + (0.0, 0.0, 0.0))[:3]
    if prop.shape == "sphere":
        return _Geom("sphere", sx, sx)
    if prop.shape == "cylinder":
        return _Geom("cylinder", sx, sz / 2.0)
    return _Geom("box", max(sx, sy) / 2.0, sz / 2.0,
                 size=(sx / 2.0, sy / 2.0, sz / 2.0))


class MossTargetSensors:
    """The wrist and front cameras' readings of the object nearest the jaws."""

    _sense_arm = ME.MossPickEnv._sense_arm
    _front_attitude = ME.MossPickEnv._front_attitude
    _true_attitude = ME.MossPickEnv._true_attitude
    #: the depth camera's 3-D grip fix and its occlusion test — the env's own
    #: (`_sense_arm` fills `_grip` when `publish_grip` is set)
    _arm_sees = ME.MossPickEnv._arm_sees
    publish_grip = True

    def __init__(self, model, prefix: str, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._model = self.model = model
        self._arm_cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + moss.ARM_CAMERA_BODY)
        self._tcp = self.tcp_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, prefix + "tcp")
        self.data = None
        self.driver = None
        self.can_body = -1
        self.prop = None
        self._clear()

    @staticmethod
    def fits(model, prefix: str) -> bool:
        """Is this body a MOSS (it has the wrist camera and a tool point)?"""
        return (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                  prefix + moss.ARM_CAMERA_BODY) >= 0
                and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                      prefix + "tcp") >= 0)

    def _clear(self) -> None:
        self._next_arm_t = 0.0
        self._arm_rng = None
        self._arm_rng_t = -1e9
        self._pending_att = []
        self._att = None
        self._att_t = -1e9
        self._next_det_t = 0.0
        self._pending_fatt = []
        self._fatt = None
        self._fatt_t = -1e9
        self._z = None
        self._z_t = -1e9
        self._grip = None
        self._grip_t = -1e9
        self._yaw = None

    def tick(self, data, driver, candidates, hint=None) -> None:
        """One world tick. `candidates` is [(body_id, _Geom)] of pickable
        objects; the one nearest the tool point is the target."""
        self.data, self.driver = data, driver
        if not candidates or self._tcp < 0:
            return
        tcp = np.asarray(data.site_xpos[self._tcp], float)
        # NOT what is already IN THE BIN: during a carry the binned objects sit
        # right beside the jaws, and taking the nearest of them reported a
        # binned object's height as the carried one's (the carry check then
        # aborted 14 of 25 good carries). A camera tells its load from the bin.
        x, y, yaw = driver.pose(data)
        c, s = math.cos(-yaw), math.sin(-yaw)
        def in_bin(b):
            p = data.xpos[b]
            bx = (p[0] - x) * c - (p[1] - y) * s
            by = (p[0] - x) * s + (p[1] - y) * c
            return (moss.BIN_INTERIOR_X[0] < bx < moss.BIN_INTERIOR_X[1]
                    and moss.BIN_INTERIOR_Y[0] < by < moss.BIN_INTERIOR_Y[1]
                    and moss.BIN_FLOOR_Z < p[2] < moss.BIN_RIM_Z + 0.02)
        live = [cd for cd in candidates if not in_bin(cd[0])] or candidates
        if hint is not None:
            # THE BRAIN'S TARGET, not whatever is nearest the jaws: a wrist
            # camera reports the detection that matches what the robot is
            # going for. Measured without it: nearest-to-jaws was another
            # object on 29% of pick steps, and a scripted pinch aimed at a
            # card closed on a can 432 mm away.
            bid, g = min(live, key=lambda cd: float(np.hypot(
                data.xpos[cd[0]][0] - hint[0], data.xpos[cd[0]][1] - hint[1])))
        else:
            bid, g = min(live, key=lambda cd: float(
                np.linalg.norm(np.asarray(data.xpos[cd[0]], float) - tcp)))
        if bid != self.can_body:
            self._clear()
            self.can_body, self.prop = bid, g
        before = self._arm_rng_t
        grip_before = self._grip_t
        self._sense_arm()
        if self._grip_t != grip_before:
            # ...and which way the object's LONG side points, as a top-down
            # detector's oriented box says: what a scripted pinch rolls the
            # wrist to (`moss_pinch`). None for a thing with no long side.
            self._yaw = self._long_yaw(data, driver, bid, g)
        if self._arm_rng_t != before:
            # A depth wrist camera gives the object's 3-D position, so its
            # HEIGHT: sampled with the same detection, the same range noise.
            self._z = float(data.xpos[bid][2]) + float(
                self.rng.normal(0.0, ME.ARM_DET_RANGE_NOISE))
            self._z_t = float(data.time)
        # The FRONT camera's axis reading, gated and delayed as the env's
        # `_sense` gates and delays it (only the attitude half is needed: the
        # position fix is the world's own detector).
        t = float(data.time)
        if t + 1e-9 >= self._next_det_t:
            self._next_det_t = t + 1.0 / ME.DET_RATE_HZ
            x, y, yaw = driver.pose(data)
            p = data.xpos[bid]
            dx, dy = float(p[0] - x), float(p[1] - y)
            bx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
            by = dx * math.sin(-yaw) + dy * math.cos(-yaw)
            rng, bearing = math.hypot(bx, by), math.atan2(by, bx)
            if (rng > ME.DET_MIN_RANGE_M and abs(bearing) < ME.DET_MAX_BEARING
                    and self.rng.random() > ME.DET_DROPOUT):
                fatt = self._front_attitude(rng, bearing)
                if fatt is not None:
                    self._pending_fatt.append((t + ME.DET_LATENCY_S, fatt))
        while self._pending_fatt and self._pending_fatt[0][0] <= t + 1e-9:
            _at, fatt = self._pending_fatt.pop(0)
            self._fatt, self._fatt_t = fatt, t

    def _long_yaw(self, data, driver, bid, g):
        R = np.asarray(data.xmat[bid], float).reshape(3, 3)
        if g.shape == "sphere":
            return None
        if g.shape == "box":
            ax = R[:, int(np.argmax(np.asarray(g.size[:3], float)))]
        else:
            ax = R[:, 2]
            if abs(float(ax[2])) > 0.7:      # a standing cylinder: a circle
                return None
        if math.hypot(float(ax[0]), float(ax[1])) < 1e-6:
            return None
        _x, _y, yaw = driver.pose(data)
        a = math.atan2(float(ax[1]), float(ax[0])) - yaw
        a += float(self.rng.normal(0.0, ME.ARM_DET_AXIS_NOISE))
        return math.atan2(math.sin(a), math.cos(a))

    def read(self) -> dict:
        """The slots' values, behind the env's own freshness gates: `att` is
        the wrist camera's when fresh, else the front camera's, else None;
        `range` is the wrist camera's when fresh, else None; `top` is the
        object's top above the floor (the env publishes it behind the
        position fix's freshness — the brain applies that gate)."""
        if self.data is None or self.can_body < 0:
            return {"att": None, "range": None, "top": None, "grip": None,
                    "yaw": None}
        t = float(self.data.time)
        att = None
        if self._att is not None and (t - self._att_t) < ME.ARM_STALE_S:
            att = self._att
        elif self._fatt is not None and (t - self._fatt_t) < ME.STALE_S:
            att = self._fatt
        rng = (self._arm_rng if self._arm_rng is not None
               and (t - self._arm_rng_t) < ME.ARM_STALE_S else None)
        up = abs(self._true_attitude()[2])
        top = float(self.data.xpos[self.can_body][2]) + (
            self.prop.half_height * up + self.prop.radius * (1.0 - up))
        z = (self._z if self._z is not None
             and (t - self._z_t) < ME.ARM_STALE_S else None)
        grip = (None if self._grip is None
                or (t - self._grip_t) >= ME.ARM_STALE_S
                else tuple(float(v) for v in self._grip))
        return {"att": None if att is None else tuple(float(v) for v in att),
                "range": None if rng is None else float(rng), "top": top,
                "z": z, "grip": grip,
                "yaw": None if grip is None else self._yaw}
