"""MOSS's base drive: his own track envelope around Innate's velocity PD.

`robots/mars_drive.py` ported Innate's planar-base controller and measured
the one thing that could not be ported verbatim — an `xfrc_applied` velocity
loop is EXPLICIT, so it is contractive only while `KP*dt/M < 2`. That maths
is a property of the integrator and the body, not of a manufacturer, so this
module reuses those gains and that clamp rather than keeping a second copy of
numbers that must agree.

**What is MOSS's, and it is the whole point of the file: the envelope.** A
tracked chassis cannot strafe and cannot turn faster than its tracks will
differ, so a twist is only meaningful after it has been through his own
kinematics:

    moss_dimos/diffdrive.twist_to_tracks   left  = vx - wz * w/2
                                           right = vx + wz * w/2
                                           then SCALE both if either exceeds
                                           `MAX_TRACK_SPEED_MPS`

Scaling and not clipping is his design and his docstring's reason: "a
full-speed forward plus a turn slows both tracks together instead of
flattening the turn". `clamp_cmd` below is that function and its inverse, so
a command this lab issues is a command his ESP32 could execute — the firmware
takes `drive LEFT RIGHT` and nothing else.

**The width is MEASURED, not taken from his file.** `diffdrive.py` carries
`track_width_m = 0.30`, which he flagged on 2026-09-24 as an uncalibrated
default; the collision track boxes in `moss_robot.xml` are centred at
y = +-0.122, so the real figure is `moss.TRACK_CENTRES_M` = 0.244. It matters
in exactly one direction: a yaw command divided by 0.30 when the robot is
0.244 wide comes out **23% slow**, every time, in a way no reward or brain
would ever attribute to the axle spacing.

**The arm half is one line here, unlike MARS's.** `mars.urdf` ships no
`<actuator>` block, so `mars_drive` runs a position PD into `qfrc_applied`
itself. Laurent's MJCF already has seven `position` servos with his own gains
(kp 70 / kv 3 on the arm, kp 700 / kv 8 on the fingers), so this driver only
writes `data.ctrl` and lets MuJoCo do what his own missions do.

**Untested on hardware.** Nothing here has driven a MOSS, the gains are sim
gains, and the base is his simplified one with a lab heading added
(`robots/moss.add_planar_base`). What IS his: the track width, the speed
limit, the scaling rule and the servos.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import mujoco
import numpy as np

from . import moss

# Innate's planar-base PD, ported and MEASURED in `mars_drive` — see this
# module's docstring for why they are imported rather than re-typed.
from .mars_drive import (
    CMD_VEL_TIMEOUT_S,
    GAIN_LIMIT,
    HOLD_SETTLE_S,
    KP_FORWARD,
    KP_HOLD_LINEAR,
    KP_HOLD_YAW,
    KP_LATERAL,
    KP_YAW,
    MAX_BASE_ANGULAR_SPEED,
    MAX_BASE_LINEAR_SPEED,
)

#: The fastest yaw his tracks can produce: both at the limit, opposed.
#: 2 * 0.6 / 0.244 = 4.92 rad/s. COMPUTED from his two numbers rather than
#: typed, so a corrected track width moves it.
MAX_SPIN_RAD_S = 2.0 * moss.MAX_TRACK_SPEED_MPS / moss.TRACK_CENTRES_M


def twist_to_tracks(vx: float, wz: float,
                    width: float = moss.TRACK_CENTRES_M) -> tuple[float, float]:
    """(left, right) track speeds in m/s — his `diffdrive.twist_to_tracks`."""
    half = 0.5 * width
    return vx - wz * half, vx + wz * half


def tracks_to_twist(left: float, right: float,
                    width: float = moss.TRACK_CENTRES_M) -> tuple[float, float]:
    """(vx, wz) from two track speeds — his `diffdrive.tracks_to_twist`."""
    return 0.5 * (left + right), (right - left) / width


def clamp_cmd(vx: float, wz: float) -> tuple[float, float]:
    """The twist his firmware would actually execute, given (vx, wz).

    Through his mapping and back: if either track would exceed
    `MAX_TRACK_SPEED_MPS`, BOTH are scaled by the same factor, which keeps
    the turn radius and slows the pair. A caller gets back the twist it will
    get, so an observation of "what was commanded" and the motion that
    follows cannot disagree — the failure `robots/mars_drive.clamp_cmd`
    avoids by clamping each component independently, which for a tracked base
    would quietly straighten a turn instead of slowing it.

    Non-finite input is a ValueError, as it is in his file: a NaN command is
    a bug upstream and a motor command is the wrong place to discover it.
    """
    for name, value in (("vx", vx), ("wz", wz)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    left, right = twist_to_tracks(vx, wz)
    peak = max(abs(left), abs(right))
    if peak > moss.MAX_TRACK_SPEED_MPS:
        k = moss.MAX_TRACK_SPEED_MPS / peak
        left, right = left * k, right * k
    return tracks_to_twist(left, right)


class MossDriver:
    """One attached MOSS: drive the base, hold the arm, read the odometry.

    `model` is the COMPILED model the robot lives in — MOSS's own scene
    (`moss.model()`) or a `/sim` world with N bodies in it — and `prefix` is
    what it was attached under (`""` standalone, `"m0/"` in a room). Every
    address is resolved once, by NAME: a 50 Hz world loop must not be doing
    string lookups, and a revision that renames a link fails at construction
    instead of servoing a joint that does not exist.

    The same six-method protocol `world/arena.WorldRobot` calls on a
    `MarsDriver` (`spawn`, `pose`, `velocity`, `set_cmd`, `set_arm`,
    `arm_targets`, `step`), because that protocol is the lab's and not MARS's.
    """

    def __init__(self, model: mujoco.MjModel, prefix: str = ""):
        self.model = model
        self.prefix = prefix
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                         prefix + moss.BASE_BODY)
        if self.base_id < 0:
            raise KeyError(f"no body {prefix + moss.BASE_BODY!r} in the model "
                           "— a MOSS must be attached under this prefix")
        self.base_qadr = tuple(int(model.joint(prefix + n).qposadr[0])
                               for n in moss.BASE_JOINTS)
        self.base_dadr = tuple(int(model.joint(prefix + n).dofadr[0])
                               for n in moss.BASE_JOINTS)
        #: {joint: ctrl index} for his seven position servos. Resolved by
        #: actuator name, which in his MJCF equals the joint name.
        # Five arm servos and ONE gripper — his follower finger has no
        # actuator any more (`moss.couple_fingers`: one servo, two jaws), so
        # a command for it would be an address that does not exist.
        self.act: dict[str, int] = {
            n: int(model.actuator(prefix + n).id)
            for n in (*moss.ARM_JOINTS, moss.GRIPPER_JOINT)}
        #: {joint: (qpos adr, dof adr)} for every joint the TELEMETRY
        #: reports — the five arm joints and BOTH jaws. `world/arena.py`'s
        #: generic `WorldRobot.arm_qpos` reads this attribute by name to fill
        #: `Senses.arm`, so a driver without it hands every brain `None` and
        #: a learned skill can never build its observation at all. That is
        #: exactly what happened: the trained pickup sat still through every
        #: 12 s creep window and the loop stowed 0/9.
        self.adr: dict[str, tuple[int, int]] = {
            n: (int(model.joint(prefix + n).qposadr[0]),
                int(model.joint(prefix + n).dofadr[0]))
            for n in moss.JOINT_NAMES}
        self._joint_qadr = {n: a[0] for n, a in self.adr.items()}
        self._my_dofs = np.array(sorted(
            {*self.base_dadr,
             *(int(model.joint(prefix + n).dofadr[0]) for n in moss.JOINT_NAMES)}))
        self._home_qpos = moss.home_qpos(model, prefix)
        #: The two pad geoms, for `held_body`.
        self._pad_geoms = {int(model.geom(prefix + g).id)
                           for g in moss.FINGER_PADS}
        self._home_adr = np.array(sorted(
            {*self.base_qadr, *self._joint_qadr.values()}))
        # The step-size guard: both are properties of THIS model — its
        # timestep and this robot's own inertia — so a room that compiled at
        # 5 ms gets different gains from one at 2 ms, and neither diverges.
        self.dt = float(model.opt.timestep)
        self.base_inertia = self._apparent_inertia(model)
        self.kp_forward = self._stable_gain(KP_FORWARD, self.base_inertia[0])
        self.kp_lateral = self._stable_gain(KP_LATERAL, self.base_inertia[1])
        self.kp_yaw = self._stable_gain(KP_YAW, self.base_inertia[2])
        if self._belts_carry_the_floor():
            # The belts are on the ground (his V0.4 hulls, marked by
            # `moss.tracks_as_support`), so the yaw loop is working against
            # their residual drag and its standing error is 36%. Read off the
            # MODEL rather than passed in, because whoever builds the spec is
            # the one who knows which geometry this is.
            self.kp_yaw = self._stable_gain(moss.TRACK_SUPPORT_KP_YAW,
                                            self.base_inertia[2])
        self._arm: dict[str, float] = {
            k: v for k, v in moss.ARM_HOME.items()
            if k != moss.FINGER_JOINTS[1]}
        self._cmd_vx = 0.0
        self._cmd_wz = 0.0
        # -inf, so a driver that was never commanded is already expired: an
        # un-driven MOSS holds still rather than waiting out the watchdog.
        self._cmd_t = -math.inf
        self._hold: tuple[float, float, float] | None = None
        self._still_since: float | None = None

    # -------------------------------------------------- the step-size guard

    def _apparent_inertia(self, model: mujoco.MjModel) -> tuple[float, float, float]:
        """(m_x, m_y, I_yaw) the base's three DoFs actually feel, at HOME.

        The Schur complement of the base 3x3 of the mass matrix, which
        `mars_drive` measured against the empirical stability boundary and
        found predicts it to two digits. Its own `MjData`, because a driver
        is handed a model and no state and the mass matrix needs a pose.

        On MOSS this reads the mass Laurent weighed rather than MuJoCo's
        density default — `moss.set_base_inertial` is upstream of every
        compile — which is the entire reason that number was worth asking him
        for: the gains below are picked from it.
        """
        probe = mujoco.MjData(model)
        probe.qpos[self._home_adr] = self._home_qpos[self._home_adr]
        mujoco.mj_forward(model, probe)
        full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, probe, full)
        block = full[np.ix_(self.base_dadr, self.base_dadr)]
        inv = np.linalg.inv(block)
        return tuple(float(1.0 / inv[i, i]) for i in range(3))

    def _belts_carry_the_floor(self) -> bool:
        """Do this model's tracks actually touch the ground?

        True only for a spec that ran `moss.tracks_as_support`, which marks
        the belt hulls with `moss.TRACK_CONTACT_PRIORITY` precisely so this
        can be asked. The legacy boxes sit 4 mm proud and make no contacts,
        so nothing about them needs compensating.
        """
        for name in moss.TRACK_GEOMS:
            try:
                g = self.model.geom(self.prefix + name)
            except KeyError:
                return False
            if int(g.priority[0]) != moss.TRACK_CONTACT_PRIORITY:
                return False
        return True

    def _stable_gain(self, kp: float, inertia: float) -> float:
        """The gain, or as much of it as this timestep can integrate:
        `min(kp, GAIN_LIMIT * inertia / dt)`."""
        return min(float(kp), GAIN_LIMIT * float(inertia) / self.dt)

    def gains(self) -> dict[str, float]:
        """What this driver ended up with, for a test or a panel to read."""
        return {"forward": self.kp_forward, "lateral": self.kp_lateral,
                "yaw": self.kp_yaw, "dt": self.dt,
                "mass_x": self.base_inertia[0], "mass_y": self.base_inertia[1],
                "inertia_yaw": self.base_inertia[2]}

    # -------------------------------------------------------- state and pose

    def spawn(self, data: mujoco.MjData, x: float = 0.0, y: float = 0.0,
              yaw: float = 0.0) -> None:
        """Put this MOSS at (x, y, yaw) in the HOME pose, at rest.

        Writes state, does not step. The applied-force row is cleared too: a
        spawn that left the last step's drive force in `xfrc_applied` would
        shove the robot on the next `mj_step` with no command in sight.
        """
        data.qpos[self._home_adr] = self._home_qpos[self._home_adr]
        qx, qy, qyaw = self.base_qadr
        data.qpos[qx], data.qpos[qy], data.qpos[qyaw] = float(x), float(y), float(yaw)
        data.qvel[self._my_dofs] = 0.0
        data.qfrc_applied[self._my_dofs] = 0.0
        data.xfrc_applied[self.base_id] = 0.0
        self._arm = {k: v for k, v in moss.ARM_HOME.items()
                     if k != moss.FINGER_JOINTS[1]}
        for name, value in self._arm.items():
            data.ctrl[self.act[name]] = value
        self._cmd_vx = self._cmd_wz = 0.0
        self._cmd_t = -math.inf
        self._hold = None
        self._still_since = None
        mujoco.mj_forward(self.model, data)

    def pose(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(x, y, yaw) of the base, straight off the planar joints. `yaw`
        accumulates; wrap it at the consumer."""
        qx, qy, qyaw = self.base_qadr
        return (float(data.qpos[qx]), float(data.qpos[qy]),
                float(data.qpos[qyaw]))

    def velocity(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(v_forward, v_lateral, wz) in the base frame."""
        dx, dy, dyaw = self.base_dadr
        yaw = float(data.qpos[self.base_qadr[2]])
        cos, sin = math.cos(yaw), math.sin(yaw)
        vx, vy = float(data.qvel[dx]), float(data.qvel[dy])
        return (vx * cos + vy * sin, -vx * sin + vy * cos,
                float(data.qvel[dyaw]))

    def tracks(self, data: mujoco.MjData) -> tuple[float, float]:
        """What the two tracks are doing, m/s — the pair his firmware
        commands and his encoders report. The measured twist through his own
        kinematics, so a telemetry panel and the ESP32 speak one language."""
        v_forward, _v_lateral, wz = self.velocity(data)
        return twist_to_tracks(v_forward, wz)

    # ------------------------------------------------------------ commanding

    def set_cmd(self, vx: float, wz: float, t: float) -> None:
        """Command a body-frame twist, stamped at sim time `t`.

        The twist is put through `clamp_cmd` — his scaling rule — so what is
        stored is what the tracks can deliver. `step()` compares `t` against
        `data.time`, so the watchdog measures SIM seconds: a slow host makes
        a robot late, never runaway.
        """
        self._cmd_vx, self._cmd_wz = clamp_cmd(vx, wz)
        self._cmd_t = float(t)

    def cmd(self) -> tuple[float, float]:
        return self._cmd_vx, self._cmd_wz

    def set_arm(self, targets: Mapping[str, float]) -> None:
        """Absolute joint targets, by name; unknown names are refused.

        Refused rather than ignored because a typo in a brain's arm intent
        would otherwise be a joint that silently never moves (AGENTS.md rule
        0: a knob that changes nothing is broken, not null).
        """
        # The follower finger is accepted and FOLDED ONTO the leader rather
        # than refused: a brain that says "open both jaws" is not wrong, it
        # is describing one servo in the vocabulary his MJCF used to have.
        # What it cannot do is give them different values.
        t = dict(targets)
        follower = t.pop(moss.FINGER_JOINTS[1], None)
        if follower is not None and moss.GRIPPER_JOINT not in t:
            t[moss.GRIPPER_JOINT] = float(follower)
        unknown = set(t) - set(self.act)
        if unknown:
            raise KeyError(f"{sorted(unknown)} are not MOSS commands — have "
                           f"{[*moss.ARM_JOINTS, moss.GRIPPER_JOINT]} "
                           "(one gripper: the real jaw has a single servo)")
        self._arm.update({k: float(v) for k, v in t.items()})

    def arm_targets(self) -> dict[str, float]:
        return dict(self._arm)

    # ------------------------------------------------------------- the loop

    def step(self, data: mujoco.MjData) -> None:
        """One control pass, before EVERY `mj_step` — not decimatable.

        `mars_drive` measured why: decimated to 50 Hz the base's velocity
        loop has gain 3.04, past the explicit loop's bound of 2, and a
        0.3 m/s run ends going backwards at 2.6 m/s.
        """
        self.drive(data)
        self.servo(data)

    #: THE TRACKS MUST NOT TOUCH THE GROUND, and that is a property of this
    #: drive rather than an oversight in the model. His collision boxes sit
    #: with their underside 4 mm proud of the floor, so MOSS makes ZERO floor
    #: contacts here and this force law pushes a body that nothing opposes.
    #:
    #: Put it on the ground — Laurent's V0.4 collision candidate replaces the
    #: boxes with belt hulls that do reach the floor — and the rover stops:
    #: commanded 0.2 m/s for 2 s it travels 3 mm instead of 400, because the
    #: contact answers the 40 N this law can apply (kp_forward 200 at a
    #: 0.2 m/s error) with 44 N of friction. MEASURED by giving the belts
    #: contact priority and sweeping: friction 0.9 -> 3 mm, 0.2 -> 33 mm,
    #: 0.08 -> 140 mm, 0.03 -> 314 mm. Reaching the commanded speed needs
    #: about 0.02, which is not a rubber track on a floor.
    #:
    #: The mismatch is structural, not a gain. A real tracked vehicle is
    #: PROPELLED by belt-to-ground friction: the belt surface moves backwards
    #: and the ground pushes the robot forwards. This model applies a force
    #: to the chassis and treats the same friction as pure drag, so it pays
    #: the resistance twice and collects the propulsion never. Raising
    #: `kp_forward` would paper over it and make every turn a fight between
    #: an arbitrary force and an arbitrary friction.
    #:
    #: **And the reason it is structural is the planar base, measured
    #: 2026-09-24.** His belt hulls are sized exactly right: spawned at the
    #: pinned ride height their lowest vertices sit at z = -0.00 mm, touching
    #: the floor with zero penetration. But `base_x/base_y/base_yaw` pin z,
    #: so the rover's 42.4 N of weight is carried by the JOINT and the tracks
    #: bear **0.00 N at rest**. A track with no load on it cannot generate
    #: traction; friction there is pure parasitic drag whose normal force
    #: appears only when something pushes the chassis sideways into the
    #: contact (summed normal force reaches 44-337 N during a 0.2 m/s run,
    #: on a robot that weighs 42). So the sweep above is not measuring
    #: traction at all — it is measuring how much unloaded drag the force law
    #: has to fight.
    #:
    #: That narrows the fix. Low-friction belts (mu ~ 0.01 gets 0.385 m of the
    #: commanded 0.400) would make the tracks an honest SUPPORT surface under
    #: a frankly kinematic drive, and it is at least coherent: the robot then
    #: stands on the floor, collides with furniture and cannot hover. A real
    #: friction model needs the weight to actually reach the tracks, which
    #: means a free root under gravity rather than a planar joint — and that
    #: is a different robot, not a different gain.
    #:
    #: So: either the tracks stay clear of the floor and this drive stays a
    #: kinematic abstraction whose numbers describe the CONTROLLER (which is
    #: what every drive figure in this repo currently is), or the belts get
    #: modelled as driven surfaces and this law is replaced. Worth settling
    #: before anyone calibrates skid-steer width against hardware, because
    #: there is no friction model here to calibrate.
    def drive(self, data: mujoco.MjData) -> None:
        """The base half. Owns `data.xfrc_applied[rover]` — all six
        components are assigned, including the three this drive never uses,
        so nothing else's leftover z-force can ride along on a body whose z
        is pinned by the planar joints."""
        dx, dy, dyaw = self.base_dadr

        # The governor, on the STATE, before anything reads it.
        lin = math.hypot(data.qvel[dx], data.qvel[dy])
        if lin > MAX_BASE_LINEAR_SPEED:
            data.qvel[dx] *= MAX_BASE_LINEAR_SPEED / lin
            data.qvel[dy] *= MAX_BASE_LINEAR_SPEED / lin
        if abs(data.qvel[dyaw]) > MAX_BASE_ANGULAR_SPEED:
            data.qvel[dyaw] = math.copysign(MAX_BASE_ANGULAR_SPEED,
                                            data.qvel[dyaw])

        expired = data.time - self._cmd_t > CMD_VEL_TIMEOUT_S
        vx = 0.0 if expired else self._cmd_vx
        wz = 0.0 if expired else self._cmd_wz

        yaw = float(data.qpos[self.base_qadr[2]])
        cos, sin = math.cos(yaw), math.sin(yaw)
        v_forward = data.qvel[dx] * cos + data.qvel[dy] * sin
        v_lateral = -data.qvel[dx] * sin + data.qvel[dy] * cos

        force_forward = self.kp_forward * (vx - v_forward)
        # A tracked chassis has no sideways DoF at all, so the lateral term
        # is not a tracking error but a non-holonomic constraint: drive the
        # drift to zero, never to a commanded value.
        force_lateral = -self.kp_lateral * v_lateral
        torque_yaw = self.kp_yaw * (wz - data.qvel[dyaw])
        hold_x, hold_y, hold_yaw = self._station_keeping(data, vx, wz)
        row = data.xfrc_applied[self.base_id]
        row[0] = force_forward * cos - force_lateral * sin + hold_x
        row[1] = force_forward * sin + force_lateral * cos + hold_y
        row[2] = row[3] = row[4] = 0.0
        row[5] = torque_yaw + hold_yaw

    def servo(self, data: mujoco.MjData) -> None:
        """The arm half: his own position actuators, commanded by name."""
        for name, value in self._arm.items():
            data.ctrl[self.act[name]] = value

    def _station_keeping(self, data: mujoco.MjData, vx: float,
                         wz: float) -> tuple[float, float, float]:
        """World-frame (fx, fy, tz) holding a stopped base.

        Zero while driving, and zero for the first `HOLD_SETTLE_S` of quiet;
        the pose latches once, on the first step past that, and is released
        the instant a non-zero command arrives. The latch is why a shove
        returns the robot to where it stopped rather than to where the shove
        left it.
        """
        if vx or wz:
            self._hold = self._still_since = None
            return 0.0, 0.0, 0.0
        qx, qy, qyaw = self.base_qadr
        if self._still_since is None:
            self._still_since = float(data.time)
        if data.time - self._still_since < HOLD_SETTLE_S:
            return 0.0, 0.0, 0.0
        if self._hold is None:
            self._hold = (float(data.qpos[qx]), float(data.qpos[qy]),
                          float(data.qpos[qyaw]))
        hx, hy, hyaw = self._hold
        d_yaw = hyaw - float(data.qpos[qyaw])
        yaw_err = math.atan2(math.sin(d_yaw), math.cos(d_yaw))
        return (KP_HOLD_LINEAR * (hx - float(data.qpos[qx])),
                KP_HOLD_LINEAR * (hy - float(data.qpos[qy])),
                KP_HOLD_YAW * yaw_err)



def _moss_held_body(self, data: mujoco.MjData) -> int:
    """Which body the jaws have hold of, or -1 — `WorldRobot.held_body`.

    The arena delegates this to the driver and answers -1 for a driver that
    does not implement it, which is what MOSS did: `Senses.holding` was
    always None, so the scripted loop had to guess a grip from the servo
    alone and could not tell a crush from a carry.

    The predicate is BOTH pads touching the same foreign body. Not the servo
    stall: a jaw squeezing a can flat against the floor stalls exactly like
    one carrying it, and the difference is whether the pads have it between
    them. The robot's own subtree is excluded by `body_rootid` rather than by
    a name list — `mars_drive.held_body`'s own bug was a prefix-blind scan
    counting the toy as part of the robot.
    """
    m = self.model
    mine = int(m.body_rootid[self.base_id])
    touched: dict[int, set[int]] = {}
    for i in range(data.ncon):
        con = data.contact[i]
        for g, other in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
            if int(g) not in self._pad_geoms:
                continue
            b = int(m.geom_bodyid[other])
            if int(m.body_rootid[b]) == mine:
                continue
            touched.setdefault(b, set()).add(int(g))
    for body, pads in touched.items():
        if len(pads) >= 2:                  # both pads on the same thing
            return body
    return -1


MossDriver.held_body = _moss_held_body


__all__ = ["MAX_SPIN_RAD_S", "MossDriver", "clamp_cmd", "tracks_to_twist",
           "twist_to_tracks"]
