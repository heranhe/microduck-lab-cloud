"""MOSS's arm geometry for a SCRIPTED top-down pinch: forward kinematics and
a small inverse, in the robot's BASE frame (the `rover` body).

Why a script and not the learned pick, for small things (2026-09-28): the
learned pick comes down from above but does not line up, and at the scale of
a half-gram cigarette butt one pad brushing it first shoves it away — 22 of
30 attempts in the pick env ended with the butt more than 3 cm from where it
lay, and it picked 4 of 30. Hovering over the object, rolling the wrist to
it, descending straight and closing slowly picked 24/30 (jaws ALONG the
butt) and 11/30 (ACROSS it — the pads still stand 8 mm apart fully shut, and
a butt is 7-9 mm wide), with the wrist camera's noise.

Everything here is the robot's own kinematics — what a real MOSS computes
from its joint encoders — so nothing reads the world.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from . import moss

ARM = list(moss.ARM_JOINTS)
#: position and "jaws straight down" are solved on these four; the roll
#: turns the jaws about the vertical and is set on its own
_IK_N = 4


class MossKinematics:
    """Forward and inverse kinematics of MOSS's arm, base frame."""

    def __init__(self):
        self.m = moss.model()
        self.d = mujoco.MjData(self.m)
        self._qa = [self.m.joint(j).qposadr[0] for j in ARM]
        self._fa = [self.m.joint(j).qposadr[0] for j in moss.FINGER_JOINTS]
        self._lo = np.array([self.m.joint(j).range[0] for j in ARM])
        self._hi = np.array([self.m.joint(j).range[1] for j in ARM])
        self._rover = self.m.body(moss.BASE_BODY).id
        self._tcp = self.m.site("tcp").id
        self._pads = (self.m.geom("pad_left").id, self.m.geom("pad_right").id)
        self._finger = self.m.joint(moss.GRIPPER_JOINT)
        #: the pads' half-height: how far above the floor their centres must
        #: stay so their lower edges clear it
        self.pad_half = float(self.m.geom_size[self._pads[0]][2])

    def _pose(self, q, jaw: float) -> None:
        self.d.qpos[self._qa] = q
        self.d.qpos[self._fa] = jaw
        mujoco.mj_kinematics(self.m, self.d)

    def _to_base(self, p):
        R = self.d.xmat[self._rover].reshape(3, 3)
        return R.T @ (np.asarray(p, float) - self.d.xpos[self._rover])

    def _rot_base(self, R):
        Rb = self.d.xmat[self._rover].reshape(3, 3)
        return Rb.T @ np.asarray(R, float).reshape(3, 3)

    def tool(self, arm: dict, jaw: float):
        """(tool point, tool rotation), base frame, for an arm pose."""
        self._pose([float(arm[j]) for j in ARM], jaw)
        return (self._to_base(self.d.site_xpos[self._tcp]),
                self._rot_base(self.d.site_xmat[self._tcp]))

    def body_in_base(self, arm: dict, jaw: float, body: str):
        """(position, rotation) of a body of the arm — a camera — in the base
        frame, for an arm pose."""
        self._pose([float(arm[j]) for j in ARM], jaw)
        b = self.m.body(body).id
        return self._to_base(self.d.xpos[b]), self._rot_base(self.d.xmat[b])

    def object_in_base(self, arm: dict, jaw: float, grip) -> np.ndarray:
        """The wrist depth fix (object in the TOOL frame) in the base frame."""
        p, R = self.tool(arm, jaw)
        return p + R @ np.asarray(grip, float)

    def padmid(self, arm: dict, jaw: float) -> np.ndarray:
        """The midpoint between the pads, base frame, for an arm pose."""
        self._pose([float(arm[j]) for j in ARM], jaw)
        return self._padmid()

    def _padmid(self):
        a, b = self._pads
        return self._to_base((self.d.geom_xpos[a] + self.d.geom_xpos[b]) / 2.0)

    def _jaw_dir(self):
        """The direction the jaws close along, horizontal, base frame."""
        R = self.d.xmat[self.m.jnt_bodyid[self._finger.id]].reshape(3, 3)
        v = self._rot_base(R) @ np.asarray(self._finger.axis, float)
        v = v[:2]
        return v / (np.linalg.norm(v) + 1e-9)

    def roll_for(self, arm: dict, jaw_yaw: float) -> float:
        """The wrist roll that points the jaws' closing direction along
        `jaw_yaw` (base frame, modulo pi), from this pose."""
        q = np.array([float(arm[j]) for j in ARM], float)
        want = np.array([math.cos(jaw_yaw), math.sin(jaw_yaw)])
        best, best_r = -1.0, q[4]
        for r in np.linspace(self._lo[4] + 1e-3, self._hi[4] - 1e-3, 121):
            q[4] = r
            self._pose(q, 0.041)
            s = abs(float(self._jaw_dir() @ want))
            if s > best:
                best, best_r = s, float(r)
        return best_r

    def solve(self, target, roll: float, arm0: dict, jaw: float = 0.041,
              iters: int = 80) -> tuple[dict, float]:
        """The arm pose that puts the midpoint between the pads at `target`
        (base frame) with the tool pointing straight down, wrist roll fixed.
        Returns (pose, residual)."""
        q = np.array([float(arm0[j]) for j in ARM], float)
        q[4] = roll
        target = np.asarray(target, float)

        def f(q):
            self._pose(q, jaw)
            tz = float(self._rot_base(self.d.site_xmat[self._tcp])[2, 2])
            return np.concatenate([self._padmid() - target, [0.5 * (tz + 1.0)]])

        r = f(q)
        for _ in range(iters):
            if np.linalg.norm(r) < 5e-4:
                break
            J = np.zeros((4, _IK_N))
            for i in range(_IK_N):
                dq = np.zeros(5)
                dq[i] = 1e-4
                J[:, i] = (f(q + dq) - r) / 1e-4
            step = -J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(4), r)
            q[:_IK_N] = np.clip(q[:_IK_N] + step, self._lo[:_IK_N], self._hi[:_IK_N])
            r = f(q)
        return dict(zip(ARM, (float(v) for v in q))), float(np.linalg.norm(r))
