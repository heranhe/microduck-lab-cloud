"""What MOSS can be TRAINED to do: drive a can into its jaws and lift it.

**One task, and it is deliberately the SEGMENT that fails.** The scripted
`brain/tidy_moss.py` drives, tucks, deploys, stows and folds back, and it
lands 2 cans in 9 over three 5-minute runs. Every failure is in the same
30 cm: the can is knocked aside or toppled out of line during the creep, so
the jaws close on air. So that is what this env trains — from the deployed
pose, with the arm already at `moss.GRASP_POSE` and the jaws open, bring the
can between the pads and hold it. Search, approach, stow and tuck stay code.

That split is this repo's own, on its other wheeled arm: `reach` and `pick`
are trained envs and `brain/tidy_arm.py` is a scripted brain that sequences
them (`docs/mars-roadmap.md` Phase 4/5). It is also what makes the result
exportable: the policy is an ONNX under `moss.CONTRACT_ID`, 32 floats in and
9 out at 25 Hz, which a code skill on his Jetson can run under onnxruntime
while DimOS sequences the behaviours around it.

**The physics this env inherits, all of it measured first** (`robots/moss.py`
carries the tables): the jaw holds his 66 mm can at 2-6 mm of interference
and ejects it at 10; the grasp point is fixed in the base frame so the BASE
is what has to move; the room's props carry `priority 1`, so without
`moss.tune_contacts` the litter's own frictionless-in-torsion model governs
every grip and nothing can be held at all. Training before those were known
would have optimised a reward against a grasp that physically cannot close.

**The ladder is in the SPAWN BOX, not in the weights** (AGENTS.md: if the
rollouts never contain the skill, fix the physics curriculum). A policy that
never once has a can between its pads never learns to close on one:

    rung 0   the can spawns BETWEEN the pads (0.25-0.27 m, |y| < 0.015)
    rung 1   just outside them               (0.30-0.36 m, |y| < 0.04)
    rung 2   the brain's own creep distance  (0.36-0.55 m, |y| < 0.12)

**Untested on hardware.** Nothing here has driven a MOSS.
"""

from __future__ import annotations

import json
import math
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import moss
from .moss_drive import MossDriver, clamp_cmd

TASKS: tuple[str, ...] = ("pick", "approach", "stow")

#: 25 Hz, the contract's rate, over his 2 ms step.
CTRL_DT = 1.0 / moss.CONTROL_HZ
DECIMATION = int(round(CTRL_DT / moss.GRASP_PHYSICS_DT))
EPISODE_S = 8.0
#: How long the arm is given to reach the deployed pose before the can is
#: placed. MEASURED: 9 mrad of joint error at 2 s, and a long way off at 0.8.
DEPLOY_SETTLE_S = 2.0

# ----------------------------------------------------------- the prop
#
# **What the robot is trained to pick up is DATA, not a constant block in
# this file.** A `GraspProp` is the thing a room's `world/scenario.Prop`
# already is — a shape, a size, a mass and a colour — plus the two numbers
# that are NOT derivable from geometry and have to be measured per object on
# this particular jaw: how far to close, and how high up the object to take
# it. A 40 mm cube and a 66 mm can want different values of both, and no
# amount of size arithmetic produces them.
#
# The contact parameters default to what `world/compose.py` gives a room's
# prop — `priority 1`, `condim 3`, friction 0.8/0.005/0.0001 — because
# training against a grippier object than the rooms serve is the harness bug
# this repo has hit most often (it is why the first pick brain stowed 5/5 in
# the robot's own scene and 0/3 in a room).

@dataclass(frozen=True)
class GraspProp:
    """An object a robot can be TRAINED to pick up."""

    id: str
    #: "cylinder" (size = radius, half-height), "box" (half extents),
    #: "sphere" (radius). The three `world/scenario.PROP_SHAPES` allows.
    shape: str
    size: tuple[float, ...]
    mass: float
    #: MEASURED on MOSS's jaw, per object: the finger command that grips it,
    #: and how far up it to take hold. See `robots/moss.GRASP_JAW_CTRL_M`'s
    #: table for how these are found — a sweep, not a calculation.
    jaw_ctrl_m: float
    grasp_height_m: float
    rgba: tuple[float, float, float, float] = (0.93, 0.45, 0.38, 1.0)
    condim: int = 3
    priority: int = 1
    friction: tuple[float, float, float] = (0.8, 0.005, 0.0001)
    #: contact SOFTNESS 0..1 (`physics_contact`; 0 = rigid, as before)
    soft: float = 0.0

    @property
    def radius(self) -> float:
        """The footprint radius — what the jaw has to span."""
        if self.shape == "cylinder":
            return float(self.size[0])
        if self.shape == "sphere":
            return float(self.size[0])
        return float(max(self.size[:2]))

    @property
    def half_height(self) -> float:
        if self.shape == "cylinder":
            return float(self.size[1])
        if self.shape == "sphere":
            return float(self.size[0])
        return float(self.size[2])

    @property
    def nominal_radius(self) -> float:
        """What a DETECTOR should invert this object's apparent width with.

        The largest half-extent, which is what `world/scenario.Prop.radius()`
        models a prop as. It is a separate number from `radius` on purpose:
        an upright can is 33 mm across and 57.5 mm tall, and a detector
        looking at its silhouette sees the taller one.
        """
        return float(max(self.radius, self.half_height))

    def add_to(self, spec, name: str = "can", parent=None, pose=None):
        """Put this prop in a scene spec, with its freejoint — or, given a
        `parent` body and a `(pos, quat)` in its frame, WELDED to it with no
        joint (static clutter: see `MossStowEnv._maybe_new_prop`)."""
        if parent is None:
            body = spec.worldbody.add_body(name=name, pos=[POCKET[0], 0.0,
                                                           self.half_height])
            body.add_freejoint(name=f"{name}_free")
        else:
            body = parent.add_body(name=name, pos=list(pose[0]),
                                   quat=list(pose[1]))
        from ..physics_contact import contact_softness
        _soft = contact_softness(self.soft)
        geom = dict(name=f"{name}_geom", mass=self.mass, condim=self.condim,
                    priority=self.priority, friction=list(self.friction),
                    **({} if _soft is None else
                       {"solref": _soft[0], "solimp": _soft[1]}),
                    rgba=list(self.rgba))
        if self.shape == "cylinder":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          size=[self.radius, self.half_height, 0], **geom)
        elif self.shape == "sphere":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                          size=[self.radius, 0, 0], **geom)
        else:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=list(self.size[:3]), **geom)
        return body


#: The objects this lab has MEASURED MOSS's jaw against. Adding one is an
#: entry plus a sweep (`scripts/probe_moss_grasp.py`), never a guess: the two
#: grasp numbers are the whole reason this is a table and not a size tuple.
GRASP_PROPS: dict[str, GraspProp] = {
    # His own: a 330 ml drinks can, the prop his missions use.
    # condim 6 and a real rolling coefficient, MATCHING what a room now gives
    # a cylinder (`world/compose.py`). Both used to declare a rolling number
    # on a condim-3 geom, where MuJoCo ignores it — so a knocked can rolled
    # until something stopped it, and what usually stopped it was the rover's
    # own tracks. The same bug the ball had until 2026-09-06, one shape over.
    # 0.002 is the measured short-carpet value from `world/scenario.Prop`.
    "can": GraspProp(id="can", shape="cylinder", size=(0.033, 0.0575),
                     mass=0.018, jaw_ctrl_m=moss.GRASP_JAW_CTRL_M,
                     grasp_height_m=moss.GRASP_HEIGHT_M,
                     condim=6, friction=(0.5, 0.005, 0.002)),
    # The playroom block (`world/scenario.PICKABLE_KINDS["block"]`), 40 mm and
    # 20 g — the object MARS's pick is measured on, here to show the table is
    # a table. MEASURED the same way (`scripts/probe_moss_prop.py`, drill
    # rung, close-and-lift, 4 seeds a row):
    #
    #     jaw ctrl   inner   lifted   mean lift
    #      0.010     28 mm    4/4      0.087 m
    #      0.014     36 mm    4/4      0.087 m
    #      0.018     44 mm    4/4      0.087 m
    #
    # A CUBE IS NOT A CAN, and that is the finding: the can holds only across
    # a 4 mm window (60-64 mm inner) and ejects outside it, while the block
    # survives everything from 12 mm of interference to a 4 mm gap. Flat pads
    # on flat faces, and — the part worth distrusting — a pass at a nominal
    # GAP means some of those lifts are the block riding the palm rather than
    # being pinched. 0.014 is the middle of the range and the value that also
    # obeys the can's own 2-6 mm interference rule.
    "block": GraspProp(id="block", shape="box", size=(0.02, 0.02, 0.02),
                       mass=0.02, jaw_ctrl_m=0.014, grasp_height_m=0.020,
                       rgba=(0.95, 0.75, 0.2, 1.0)),
}
DEFAULT_PROP = "can"

#: Kept as module names because the rest of this file reads them constantly;
#: they now come FROM the default prop rather than being declared beside it.
CAN_RADIUS_M = GRASP_PROPS[DEFAULT_PROP].radius
CAN_HALF_M = GRASP_PROPS[DEFAULT_PROP].half_height
CAN_MASS_KG = GRASP_PROPS[DEFAULT_PROP].mass

#: Where the pads meet: the grasp pocket, in the base frame.
POCKET = (moss.GRASP_STANDOFF_M, 0.0, moss.GRASP_HEIGHT_M)

RUNGS: tuple[int, ...] = (0, 1, 2)
DEFAULT_RUNG = 0
#: (x_lo, x_hi, |y| max) per rung — the ladder, in metres.
RUNG_BOX: dict[int, tuple[float, float, float]] = {
    0: (0.25, 0.27, 0.015),
    1: (0.30, 0.36, 0.040),
    #: The contract the mission loop delivers into — one number, two
    #: halves. See `moss.PICK_HANDOVER_BOX`.
    2: moss.PICK_HANDOVER_BOX,
}

# ------------------------------------------------------------------ rewards
#
# Progress-pay, which is the shape every working reward in this repo has: the
# policy earns for however much nearer the can got to the pocket and loses the
# same for pushing it away, so an episode's total IS the ground it made up and
# a robot that shoves the can around earns nothing.
W_PROGRESS = 60.0
#: ONE-SHOT, the first time the jaws close on the can in an episode — the
#: stepping stone between "drive it into the pocket" and "lift it".
#:
#: It has to be one-shot and it has to exist, and both halves were learned
#: the hard way. Paid PER STEP (at 0.3) a policy crushed the can and sat on
#: it for 199 of 200 steps; removed entirely, the next policy learned to
#: drive the can to within 5 mm of the pocket and then never close at all —
#: 0/12 grips — because closing earned nothing until the can was already
#: airborne, and it can never be airborne until something closes. An
#: unsampled state's value is never learned. A bonus that fires once cannot
#: be farmed and still marks the doorway.
W_CLOSE_BONUS = 6.0
#: Per step while the can is in the pocket AND the jaws are shut on it. SMALL,
#: and it was 2.0 for one afternoon: a crude close-and-hold script earned
#: +140 an episode on that alone, which is a policy that parks its jaws on a
#: can and collects for the rest of the run. `robots/mars_env.py` records the
#: same failure in the same words ("the policy parks the claw on the block").
#: It exists only to make CLOSING attractive before anything has ever lifted.
W_HELD = 0.3
#: Per metre GAINED in height while held — progress-pay, like the gap term,
#: so standing still with a can in the jaws earns nothing and the only way to
#: keep earning is to keep raising it.
W_LIFT = 120.0
#: Once, on a completed pick. Bounded, so it cannot be farmed.
SUCCESS_BONUS = 20.0
#: Once, when the can leaves the reachable band. Not a big number: a clumsy
#: approach that shoves the can 5 cm is a bad attempt, not a catastrophe, and
#: a large penalty here teaches the policy to stand still.
KNOCKED_PENALTY = -3.0
#: **Per metre the can is SHOVED while not yet held.** `KNOCKED_PENALTY`
#: alone was not a cost on shoving: it only fires when the can leaves a
#: 60 x 60 cm band, so a policy could push it 20 cm for free while earning
#: `W_PROGRESS` 60/m for closing. It did exactly that — 100% of creep ticks
#: command base motion, saturated, and the chassis is a third of all
#: can-contacts. Cans then accumulate at the walls, which is where shoved
#: things stop: 6 of 9 undelivered cans end within 0.25 m of one, and they
#: get there by being knocked rather than by spawning there.
#:
#: Set against `W_PROGRESS` deliberately: moving the can a centimetre now
#: costs about half what closing a centimetre earns, so approaching is still
#: worth it and barging is not. Only while NOT held — once the jaws have it,
#: moving the can IS the task.
#: -30 was MEASURED and it is too blunt: it charges for the can moving AT
#: ALL, so the policy learned to hesitate everywhere rather than to approach
#: cleanly. Cans shoved fell 20.9 cm -> 6.5 cm (the term works) but creep
#: conversion fell 67% -> 52% and deliveries 55/72 -> 43/72. Kept small, as a
#: nudge against barging, with the SPECIFIC failure priced below instead.
W_CAN_DISTURB = -5.0
#: **Per step the can is in contact with the ROBOT'S BODY while not held.**
#: The failure this names was described by a human watching the lab — "it's
#: kind of scooting it and then running into the body" — and it measures: 6
#: of 24 episodes drive the can into the hull, gripper mount or track before
#: gripping it, first contact happening while the grasp gap is still 81 mm.
#:
#: This is the difference between moving the can (sometimes necessary, and
#: what `W_CAN_DISTURB` over-charged for) and RAMMING it into the chassis,
#: which is never useful: the pocket is between the pads, so a can against
#: the hull is a can that has to be backed out of and re-approached, and on
#: the real robot it is a can wedged in the tracks.
W_CAN_BODY_HIT = -3.0
#: **Per step a PAD hits the can with its leading or trailing EDGE** rather
#: than with the face that grips. A pad grips on its inner face; a contact
#: whose normal runs along the pad's forward axis is the gripper barging the
#: can sideways, which is how a can ends up knocked over and out of reach.
#:
#: MEASURED on the shipped leg over 24 episodes at rung 2, classifying every
#: pad-vs-can contact normal in the PAD's own frame:
#:
#:     pad_right  front/back edge   1030
#:     pad_left   front/back edge    940
#:     pad_left   GRIP face          611
#:     pad_right  GRIP face           21
#:
#: **54% of all pad contact is edge-on**, against 17% on the faces that hold
#: it, in every one of 24 episodes at a median 97 ticks each. So the policy
#: spends most of its contact budget pushing the can with the wrong surface.
#: Priced per step, and swept through the environment because the right
#: weight is a question rather than a constant.
W_PAD_EDGE_HIT = float(os.environ.get("MICRODUCK_MOSS_PAD_EDGE", "-0.08"))
#: The pads, and how square a contact has to be to count as the grip face.
#: A normal is assigned to the axis it lies closest to, so this is simply
#: "which of the pad's three axes took the load".
PAD_GEOMS = ("pad_left", "pad_right")
#: **Turning the BASE, charged above turning the ARM.** `W_BASE_EFFORT`
#: prices forward and yaw the same and prices both at almost nothing, so the
#: policy solves a lateral offset by swinging the whole vehicle.
#:
#: MEASURED at rung 2: the can sits 7.8 degrees off centre at the median and
#: 15.5 at worst, and the base yaw travels 75 DEGREES per episode to reach
#: it — while `shoulder_pan`, which spans +-110 degrees and could cover that
#: bearing many times over, travels a near-identical 74. Two joints doing
#: the same job, one of which moves 4.3 kg of robot.
#:
#: This is not only wasteful, it is the difference between a manoeuvre that
#: is safe in a room and one that is not: an arm that pans sweeps its own
#: envelope, and a chassis that yaws sweeps the furniture. Raised by a human
#: watching it in the lab — "we don't need to rotate the whole robot, that
#: could have real world implications because it could be running into
#: stuff."
W_BASE_YAW = float(os.environ.get("MICRODUCK_MOSS_BASE_YAW", "-0.08"))
#: The geoms that count as "the body" for that: everything a can should
#: never meet on its way into the jaws. The PADS and the palm are excluded —
#: touching those is the grasp working.
CAN_BODY_GEOMS = ("hull", "gripper_mount", "track_1", "track_-1",
                  "bin_x1", "bin_x-1", "bin_floor", "bin_y1", "bin_y-1")
#: Contact noise below this per step is not a shove. The can jitters at rest
#: on a condim-6 contact and charging for that would tax standing still.
CAN_DISTURB_DEADBAND_M = 0.0005
#: Paid when a proved hold does NOT survive the ramp to the carry pose. It
#: has to cost more than the hold earned, or a policy banks the hold bonus
#: and lets the can go on the way up.
DROP_ON_LIFT_PENALTY = -30.0
#: **Driving costs something**, per control step, as a fraction of the
#: envelope. MEASURED on the live checkpoint at 100k steps, evaluated on the
#: rung it trained on: 50, 54 and 57 forward/back sign changes in 200 steps,
#: 0.61-0.76 m of path for 0.07-0.28 m of travel, and the can shoved from
#: 0.012 m to 0.105 m away. With a free base and a progress-paid gap, jiggling
#: nets zero and costs nothing, so the policy jiggles — and a tracked chassis
#: rocking on top of a can is how the can leaves.
#:
#: Small, and flat rather than ramped: at 0.20 it also suppressed the
#: APPROACH at rung 2, which is the thing the base is for. This is the term
#: `AGENTS.md` warns about most, so it is weighted to be felt when there is
#: nothing to do and ignored when there is somewhere to be.
#: **Raisable only now that driving is OPTIONAL.** Until 2026-09-25 the
#: hand-over contract reached past the arm (0.55 m against a 0.508 m reach),
#: so the base was doing necessary work on a third of episodes and no weight
#: here could have removed the driving without removing the grasp — which is
#: exactly what the -0.08 base-yaw and -0.04 effort terms failed to do, twice.
#: With the contract inside reach (`moss.PICK_HANDOVER_BOX`), the arm can take
#: every can from a standing start and this becomes a real lever.
W_BASE_EFFORT = float(os.environ.get("MICRODUCK_MOSS_BASE_EFFORT", "-0.04") or -0.04)
#: Action-rate, tiny. The duck's ramped version drove the G1's idle to -92
#: (AGENTS.md, "a ramped penalty can eat its task"), so this one is flat.
#: **Jerk, and it was priced at nothing.** MEASURED on the shipped leg: the
#: summed |delta action| is 1.127 per step at the median and reaches 8.000,
#: which is all eight channels flipping rail to rail inside one 40 ms tick.
#: Over an episode that totals 83, costing 0.83 against a success bonus of
#: 20 — four per cent of the prize for behaviour a human watching described
#: as erratic, and for base commands that sit saturated 100% of the time.
#:
#: Raised to 0.03, so the same jerk costs ~12% instead of 4%. NOT higher on
#: purpose: a blanket can-displacement penalty was set at -30 earlier the
#: same day, cut shoving 21 cm -> 6.5 cm exactly as intended, and cost 12
#: cans of 72 because the policy learned to hesitate rather than to move
#: cleanly. A smoothness term is the same kind of blunt instrument.
W_ACTION_RATE = float(os.environ.get("MICRODUCK_MOSS_ACTION_RATE", "-0.03"))

#: How far the arm may nudge each joint per control step. A NUDGE and not an
#: absolute target, for the reason `mars_env`'s `reach` measured: given an
#: angle to aim at, the arm reaches the target and can never sit still on it,
#: because no command means "stay". A nudge has a zero, and zero means hold.
ARM_DELTA_RAD = 0.03
#: The jaw's own action is a nudge too, in metres of finger travel.
JAW_DELTA_M = 0.004
#: The base's action, scaled into his own envelope by `clamp_cmd`.
MAX_VX, MAX_WZ = 0.20, 1.0

# --------------------------------------------------------------- the belief
#
# **The policy is shown what it would SEE, not where the can is.** Feeding it
# the truth every step trains a robot that cannot exist: on the rover the can
# arrives through a 10 Hz RealSense with latency, it leaves the frame at close
# range, and it MOVES when the tracks nudge it. A policy trained on truth has
# never had a stale fix and has no idea what to do with one.
#
# So the observation carries a FIX — the last thing the camera reported —
# carried forward by the base's own motion between frames, exactly as
# `brain/tidy_moss.py` carries it, and `OBS_TARGET_SEEN` goes to 0 when it is
# stale. The REWARD still reads the truth: the env knows where the can is, and
# paying a policy for its own belief would pay it for believing whatever is
# convenient.
DET_RATE_HZ = moss.CAMERA_RATE_HZ          # 10 Hz, his D455 as the lab models it
# ASSUMPTIONS, and he said so: none of the three has been measured on MOSS.
DET_LATENCY_S = 0.05
DET_BEARING_NOISE = 0.02                   # rad
DET_RANGE_NOISE = 0.015                    # m, 1 sigma
#: Past this the can is under the camera's near edge and simply not in
#: frame — a GEOMETRIC RGB limit (a 75 mm lens looking level through a 62 deg
#: vertical field first sees the floor at ~0.12 m), and explicitly NOT a
#: depth cutoff. Laurent's correction, 2026-09-24: valid depth on a D455f
#: starts around **0.52 m** and depends on configuration, so there is no
#: usable depth at this robot's 0.26 m grasp standoff at all. Everything the
#: policy is shown here is inverted from apparent WIDTH, which is an RGB
#: measurement; nothing in the pick may be built on depth at that range.
DET_MIN_RANGE_M = 0.12
#: Recorded so the distinction cannot quietly collapse again.
DEPTH_MIN_RANGE_M = moss.DEPTH_MIN_RANGE_M
DET_MAX_BEARING = 0.76                     # +-43.5 deg, half the 87 deg lens
#: A frame that just does not report it — the detector's own miss rate.
DET_DROPOUT = 0.08
#: A fix older than this reads as unseen.
STALE_S = 0.6

# ------------------------------------------------------- the WRIST aperture
#
# The second camera, and it answers a DIFFERENT question. The front D455
# gives range and bearing from the chassis — where the can is — and it is
# the only one that sees anything at standoff. The wrist camera looks down
# the approach from 12 cm behind the tool point and gives ATTITUDE: which way a lying cylinder
# points, which the front camera can only infer from a silhouette at 40 cm
# and which the observation did not carry at all until 2026-09-25.
#
# MEASURED on the side mount (2026-09-27), 3,127 frames of 478dad picks,
# the object inside each camera's modelled field and range (no occlusion,
# as the sensors here model it):
#
#     grasp gap      front      wrist
#     0.00-0.05 m     62%        96%
#     0.05-0.10 m     90%        93%
#     0.10-0.20 m     95%        84%
#     0.20-0.30 m     88%        45%
#
# The wrist owns the last centimetres, where the jaw has to commit to an
# orientation; the front holds it from standoff in. (The table this replaced,
# front 91/19/15% and wrist 40/62/45%, was the retired mount past the jaw
# tips, on 5e9df7's picks.)
ARM_DET_RATE_HZ = moss.ARM_CAMERA_RATE_HZ
#: A short USB pipeline rather than the RealSense's, so less latency.
ARM_DET_LATENCY_S = 0.03
#: OURS. An axis read off a near object is a good measurement, and a can
#: across a sixth of the frame at 30 cm is a lot of pixels — but a cylinder
#: seen end-on has no axis at all, so the error is not uniform. 0.10 rad is
#: a guess at the average and the first thing to replace with a measurement.
ARM_DET_AXIS_NOISE = 0.10                  # rad, on the axis angle
#: How often it reports nothing. Lower than the front camera's 8%: it is a
#: simpler stream and much closer to its subject.
ARM_DET_DROPOUT = 0.04
#: An attitude older than this is not published. Same reasoning as `STALE_S`
#: and deliberately tighter: the jaw is moving fast at this range, so a
#: stale axis is worse than none.
ARM_STALE_S = 0.4
#: **THE FRONT CAMERA'S OWN ATTITUDE — the half of "use both cameras" that
#: the approach needs.** The wrist looks down at the jaws, so it sees nothing
#: until the can is ~20 cm away, and MEASURED over 1,500 ticks per rung on
#: 2026-09-25 the axis slots were live on 67% of ticks at rung 0, 7% at
#: rung 1 and **0.00% at rung 2** — which is the rung `brain/tidy_moss.py`
#: actually hands over at. Training stage 3 that way would have fed the
#: policy three dead inputs and then handed the robot live ones.
#:
#: A silhouette DOES give an axis (the bounding box's long side, the same
#: quantity `_true_attitude` documents as deployable), just a worse one than
#: the wrist's: the can subtends fewer pixels the further away it is, so the
#: error grows with range rather than sitting at one number.
FRONT_AXIS_NOISE_BASE = 0.14               # rad at the lens
FRONT_AXIS_NOISE_PER_M = 0.26              # rad per metre of range
#: On top of the position stream's 8%: reading an axis is harder than
#: noticing a can, and the frames it fails on are not the same frames.
FRONT_AXIS_DROPOUT = 0.15
#: A lying cylinder pointing AT the camera is a CIRCLE — there is no axis in
#: that image, and a detector that reported one would be inventing it. Above
#: this |cos| between the can's axis and the view ray the front camera
#: reports nothing rather than a guess. Only applied to a can that is
#: substantially lying: an upright one's silhouette is never ambiguous.
FRONT_AXIS_ENDON_COS = 0.93
FRONT_AXIS_ENDON_UPRIGHT = 0.5
#: Uprightness off a box aspect ratio, coarser than the wrist's 0.05.
FRONT_UPRIGHT_NOISE = 0.12
#: **PUBLISHING THE AXIS IS OPT-IN, and this default is a measurement.**
#: Every MOSS policy exported before 2026-09-25 was trained with slots 28-30
#: permanently zero, so its baked-in normalizer carries sd 1.75e-05 there: a
#: real value of 0.9 arrives as z = 51,522 and clips to the +-100 bound, three
#: of thirty-two inputs pinned at a number the network has never seen.
#:
#: MEASURED on the shipped pick leg (`teach-moss_pick-5e9df7`), rung 2, the
#: same 12 seeds:
#:
#:     attitude slots zeroed (as it trained)   10/12 picked
#:     attitude slots filled                    0/12 picked
#:
#: So filling them is not a harder task, it is a BROKEN OBSERVATION for an old
#: policy — and the deployed mission runs three such legs. A policy trained
#: with the axis must therefore ask for it, and the ones that were not keep
#: the observation they were trained on.
ATTITUDE_DEFAULT = os.environ.get("MICRODUCK_MOSS_ATTITUDE", "0") not in ("", "0")

#: **THE WRIST HAS TO APPEAR ROLLED IN A ROLLOUT BEFORE IT CAN LEARN TO ROLL.**
#: MEASURED 2026-09-25: `wrist_roll` is -2.9 deg at EVERY reset, sd 0.00 over
#: 12 seeds with domain randomisation ON, and a trained policy leaves it at
#: sd 2.6-3.9 deg at the grasp against a can axis spanning 54 deg. Adding the
#: axis to the OBSERVATION did not change that (slope 0.021 deg/deg): an
#: unsampled state's value is never learned, which is this repo's oldest
#: lesson and why the headstand needed a drill rung rather than more pay.
#: +-90 deg covers every orientation a symmetric jaw can need.
WRIST_START_RAND = float(os.environ.get("MICRODUCK_MOSS_WRIST_START", "0") or 0.0)
#: **PAY FOR TURNING TOWARD ALIGNMENT, progress-style.** Legitimate to reward
#: only because the axis is now OBSERVABLE (slots 28-30, from the wrist camera
#: on 65% of ticks inside 10 cm); before that it would have been rewarding
#: what the robot cannot see.
W_JAW_ALIGN = float(os.environ.get("MICRODUCK_MOSS_JAW_ALIGN", "0") or 0.0)
#: The action-rate penalty charges for moving the wrist — the one motion we
#: are trying to buy. See the retarget at its use site.
WRIST_FREE = os.environ.get("MICRODUCK_MOSS_WRIST_FREE", "0") not in ("", "0")
#: **NOTHING CHARGED FOR KNOCKING THE CAN OVER until now.** `W_CAN_DISTURB`
#: prices SLIDING it and `W_CAN_BODY_HIT` prices hitting it with the chassis —
#: and MEASURED on 2026-09-25 the chassis is 0.9% of all can contacts, while
#: 30% are the PALM striking the can's top rim at 115 mm on a 115 mm can, the
#: highest-leverage point there is, and every pad contact lands at 90 mm, 32 mm
#: above the centre of mass. So the robot was tipping the can over for free.
#:
#: Charged on UPRIGHTNESS LOST while the jaws do not have it, which prices the
#: OUTCOME rather than a geom: seating a can against the palm is a legitimate
#: grasp and must stay free, while catching its rim and tipping it must not.
W_CAN_TOPPLE = float(os.environ.get("MICRODUCK_MOSS_TOPPLE", "0") or 0.0)
#: **HOW TALL THE THING IS** — the one question the cameras were never asked.
#: Slot 26 carries the can's CENTRE height and measures sd 5.6 mm across every
#: pose, i.e. a constant; nothing told the policy how high it must lift to
#: clear the object, which is exactly the "it never gets above it, so it keeps
#: bumping into it" a human watching the lab described. Published into the
#: SPARE slot 31 and deployable from the same box the range comes from: a
#: detector's bounding-box HEIGHT times range is the object's height, the same
#: arithmetic that already turns its WIDTH into `range_est`.
PUBLISH_SIZE = os.environ.get("MICRODUCK_MOSS_SIZE_OBS", "0") not in ("", "0")
#: **RESCALE WHEN THE MEASURE CHANGES.** `GAP_FROM_TCP` retargeted the progress
#: term from the chassis to the gripper without touching its weight, and the
#: gripper-to-can distance closes only a few cm net per episode where the
#: chassis-to-can distance closed ~30: MEASURED, the term fell from about +18
#: per episode to +1.47 and two runs trained with almost no approach shaping
#: at all, on a sparse success bonus. Changing WHAT a term measures changes its
#: MAGNITUDE, and nothing was reporting magnitudes — `scripts/reward_budget.py`
#: exists because of this.
GAP_TCP_SCALE = float(os.environ.get("MICRODUCK_MOSS_GAP_SCALE", "1") or 1.0)
#: **A DENSE aim, not a progress-paid one.** `W_JAW_ALIGN` pays the IMPROVEMENT
#: in alignment, which is worth nothing to a policy that reaches alignment by
#: ROLLING THE CAN instead of turning its wrist — and MEASURED over five runs
#: and ~10M steps the wrist-to-axis slope stayed at 0.046, 0.021, -0.074, i.e.
#: zero every time, while wrist MOTION grew to sd 32 deg. It moves; it does not
#: aim.
#:
#: The correct wrist angle is not a thing to search for: `wrist_roll` maps ~1:1
#: to jaw yaw (164.7 deg of yaw over a 180 deg sweep), so the jaw is square
#: when `sin^2(jaw - can) = 1`. This pays that error DIRECTLY, every step the
#: can is within reach, which is the dense signal a search can follow — the
#: same reason an IK-drawn reference found a G1 kick that reward search never
#: did. Gated on proximity so it cannot pay for posing at a distant can.
W_ALIGN_HOLD = float(os.environ.get("MICRODUCK_MOSS_ALIGN_HOLD", "0") or 0.0)
#: Within this gripper-to-can distance the aim is worth paying for.
ALIGN_HOLD_RANGE_M = 0.22
#: **THE ACTUATOR ENVELOPE — MOSS's missing BAM.** The duck trains against a
#: measured actuator model (firmware current limit, back-EMF, load-dependent
#: gearbox friction, bus lag) precisely so a policy cannot learn to rely on
#: torque the servo does not have. The arm has no such model: Laurent's MJCF
#: declares forcerange +-2.20 N·m and MuJoCo silently CLAMPS to it, which is a
#: limit the policy pays nothing to sit against.
#:
#: MEASURED 2026-09-25 on `teach-moss_pick-2df5ad-s3`, 1,700 control ticks:
#:
#:     joint            at the torque clamp    over 0.75 rad/s
#:     shoulder_pan            2.4%                 0.7%
#:     shoulder_lift          38.2%                 4.9%
#:     elbow_flex              1.6%                 0.8%
#:     wrist_flex               0.0%                0.1%
#:     wrist_roll               0.4%                0.6%
#:
#: `shoulder_lift` — the joint holding the arm up against gravity — is at
#: stall for more than a third of every episode. On hardware that is a servo
#: that heats, lags and winds up its PID, and when the load shifts it releases:
#: the same joint reaches 6.15 rad/s peak against a commanded rate of 0.75
#: (the action is an increment of +-0.03 rad at 25 Hz). That is the violence a
#: human watching the lab described, and nothing in the reward mentions it.
#:
#: NOT a substitute for a real envelope. A faithful BAM for these servos needs
#: the STS3215's stall torque and no-load speed MEASURED, which is Laurent's
#: to give — until then this charges for living at the clamp rather than
#: pretending the clamp is free.
W_TORQUE_SAT = float(os.environ.get("MICRODUCK_MOSS_TORQUE_SAT", "0") or 0.0)
#: Rated joint speed. The commanded rate is 0.03 rad/step at 25 Hz = 0.75
#: rad/s, so anything past this is momentum or gravity, not a command.
ARM_RATED_RAD_S = float(os.environ.get("MICRODUCK_MOSS_RATED_RAD_S", "1.5") or 1.5)
W_OVERSPEED = float(os.environ.get("MICRODUCK_MOSS_OVERSPEED", "0") or 0.0)
#: **DRAGGING THE GRIPPER ALONG THE FLOOR.** Nothing stopped it: MEASURED
#: 2026-09-25 on the best policy, an arm or gripper geom is on the floor for
#: 3.9% of control ticks and the base is DRIVING for 3.2% of them — i.e.
#: almost every floor contact is a scrape under power, and it is the pads
#: doing it (pad_left 52 contacts, pad_right 7). On the real robot that is the
#: fingers being dragged across a floor, taking whatever is on it with them,
#: and the servos loaded sideways in the one axis they are weakest.
#: Charged per step, and harder while the base moves, because resting a pad on
#: the floor is untidy and dragging it is damage.
W_ARM_FLOOR = float(os.environ.get("MICRODUCK_MOSS_ARM_FLOOR", "0") or 0.0)
#: How much worse the same contact is while driving.
ARM_FLOOR_DRAG_MULT = 4.0
#: |vx| above which the base counts as driving for that multiplier.
ARM_FLOOR_DRIVE_MPS = 0.1
#: **COME AT IT FROM ABOVE.** Charged per step the tool point is BELOW the
#: object's top while still outside grasp radius — an approach that arrives
#: low hits the can's side and knocks it over, which is what the palm-on-the-
#: rim measurement showed (30% of contacts at 115 mm on a 115 mm can). Today
#: that is 7.6% of ticks; the gripper is above the top 47.7% of the time, so
#: it already knows how, just not reliably.
W_LOW_APPROACH = float(os.environ.get("MICRODUCK_MOSS_LOW_APPROACH", "0") or 0.0)
#: **TAKE THE BASE AWAY FROM THE PICK.** Not a penalty — penalties failed at
#: this five times, because until the hand-over contract came inside the arm's
#: reach the base was doing NECESSARY work and no weight can price away a
#: motion the task requires.
#:
#: It no longer is. MEASURED 2026-09-25 over 60,000 arm configurations with
#: the base held still, the gripper covers the whole `PICK_HANDOVER_BOX`
#: (0.36-0.47 m, |y| <= 0.12) from a standing start. And MEASURED on the
#: policy trained right after that fix, it still drives 17.1 cm per episode
#: and turns 18.8 deg, because it was warm-started from one that had to and
#: 800k steps at -0.30 did not unlearn the habit.
#:
#: So the base is simply not this policy's to command: the APPROACH leg drives,
#: the PICK leg reaches. That is the split the scripted demo has and the one a
#: human watching the lab keeps asking for. `PICK_BASE_REACH_SCALE` damps the
#: base INSIDE a radius and its own notes record that damping at DEPLOY time
#: made things worse (9/14/8/15 cans of 36 against 22) because a policy that
#: steers with its base loses its manifold without it — which is the argument
#: for TRAINING without one, not for leaving it.
BASE_LOCK = os.environ.get("MICRODUCK_MOSS_BASE_LOCK", "0") not in ("", "0")
#: **THE WRIST CAMERA KNOWS HOW CLOSE THE CAN IS, AND WE THREW IT AWAY.**
#: `_sense_arm` computes the camera-to-can distance every tick, uses it only to
#: decide whether the can is in frame, and discards it — so the one sensor that
#: WORKS at grasp range reported orientation and nothing else.
#:
#: The front camera cannot fill this in: Laurent's correction of 2026-09-24 is
#: that a D455f's valid depth starts near 0.52 m, and the whole pick happens at
#: 0.36-0.47 m, entirely inside its blind zone (see `DET_MIN_RANGE_M`). So the
#: policy has had NO proximity signal during the grasp at all — it could not
#: know it was about to run into the can, which is exactly what a human
#: watching the lab kept describing.
#:
#: Published in place of slot 26, the can's HEIGHT off the front camera, which
#: MEASURED carries sd 0.0054 m across every pose — a constant — and is now
#: redundant with slot 30 (uprightness) and slot 31 (the object's top). The
#: wrist range carries sd 0.033 m over 0.09-0.27 m (1st-99th percentile, side
#: mount, 478dad under its trained settings, 2026-09-27), 6x the information
#: in the slot it replaces. It is the range to the lens, 12 cm behind the tool
#: point, so it never reads zero at contact; the retired mount read 0.04-0.18.
#:
#: Deployable: this is a monocular range off the SAME bounding box the front
#: camera's `range_est` already inverts from apparent width, on a camera
#: 9-27 cm from its subject rather than 40.
PUBLISH_PROXIMITY = os.environ.get("MICRODUCK_MOSS_PROXIMITY", "0") not in ("", "0")
#: **LITTER IS NOT ALL ONE CAN.** Every pick policy here trained on a single
#: 66x115 mm cylinder, so "it generalises to trash" is a claim nothing has
#: tested. A fresh SHAPE per episode costs 23 ms of recompile against a 355 ms
#: episode — MEASURED, 6.6% throughput — so the obvious way is affordable and
#: there is no need to mutate geom types inside a live model.
#:
#: Sizes are bounded by what the jaw can close on: it opens to 82 mm, so a
#: graspable width past ~70 mm is a shape this robot cannot pick and would
#: only teach failure. Mass stays >= 5 g because MuJoCo handles a 1 g object
#: against a 6 kg robot badly.
PROP_VARIETY = os.environ.get("MICRODUCK_MOSS_PROP_VARIETY", "0") not in ("", "0")
#: **THE WRIST DRILL — make the skill HAPPEN so it can be reinforced.**
#:
#: MEASURED across seven runs and ~12M steps, the wrist-to-axis slope was
#: +0.046, +0.021, -0.074, -0.200, -0.134, +0.010: noise about zero, no trend.
#: And it is NOT that the policy stopped exploring — log_std held at -0.51
#: (sigma 0.60) across all seven, so it tried exactly as hard on the last run
#: as the first. Undirected per-step noise explores ACTION space; rolling the
#: wrist to match an axis while descending and closing is a narrow manifold in
#: TRAJECTORY space, and sixty steps of independent noise never assembles it.
#: Turning sigma up would only flail harder.
#:
#: So instead of paying more or randomising more, remove everything that is
#: not the skill. The can spawns LYING, already between the pads, at a random
#: yaw, with the wrist at a random roll — no approach, no driving, no choice
#: of when to close. The ONLY thing that decides the grasp is whether the jaw
#: is square, so a rollout that succeeds is one that aimed, and the credit is
#: unambiguous. This is the same move as the headstand's drill rung and the
#: kick's termination ladder, both of which worked here after reward shaping
#: had stalled ("an unsampled state's value is never learned").
WRIST_DRILL = os.environ.get("MICRODUCK_MOSS_WRIST_DRILL", "0") not in ("", "0")
#: The drill is short: it is one motion, not an episode.
WRIST_DRILL_S = 3.0


#: THE LITTER SET (2026-09-27): what a litter-picking robot really meets,
#: as rigid lumps — a flex sheet costs more than the rest of the env. Off by
#: default so every earlier run draws the same six shapes it always did.
LITTER_KINDS = ("paper", "butt", "cap")
LITTER = os.environ.get("MICRODUCK_MOSS_LITTER", "0") not in ("", "0")


def sample_litter(rng, kind: str) -> GraspProp:
    """One of the litter lumps."""
    if kind == "paper":
        # crumpled paper: a light wad that SITS where it lands — the high
        # rolling coefficient is what makes it a wad and not a ball (condim 6
        # so MuJoCo honours it; condim 3 silently ignores rolling friction)
        return GraspProp(id="paper", shape="sphere",
                         size=(float(rng.uniform(0.015, 0.030)),),
                         mass=float(rng.uniform(0.002, 0.006)),
                         jaw_ctrl_m=moss.GRASP_JAW_CTRL_M,
                         grasp_height_m=moss.GRASP_HEIGHT_M,
                         condim=6, friction=(0.9, 0.01, 0.03),
                         rgba=(0.94, 0.93, 0.88, 1.0))
    if kind == "butt":
        # cigarette butt, lying: a BOX (the yard can yaw a prop but not tip
        # it, and a standing cylinder butt is not litter anyone meets)
        return GraspProp(id="butt", shape="box",
                         size=(float(rng.uniform(0.012, 0.017)),
                               float(rng.uniform(0.0035, 0.0045)),
                               float(rng.uniform(0.0035, 0.0045))),
                         mass=float(rng.uniform(0.0003, 0.0008)),
                         jaw_ctrl_m=moss.GRASP_JAW_CTRL_M,
                         grasp_height_m=moss.GRASP_HEIGHT_M,
                         condim=4, friction=(1.2, 0.005, 0.0001),
                         rgba=(0.95, 0.72, 0.42, 1.0))
    # bottle cap
    return GraspProp(id="cap", shape="cylinder",
                     size=(float(rng.uniform(0.014, 0.016)),
                           float(rng.uniform(0.005, 0.007))),
                     mass=float(rng.uniform(0.0015, 0.003)),
                     jaw_ctrl_m=moss.GRASP_JAW_CTRL_M,
                     grasp_height_m=moss.GRASP_HEIGHT_M,
                     condim=6, friction=(0.6, 0.005, 0.002),
                     rgba=(0.30, 0.55, 0.85, 1.0))


def handover_prop(hs: dict) -> "GraspProp":
    """A handover state's object as the pick env's prop. The state keeps the
    scenario's FULL extents (a cylinder's are radius, radius, full height)."""
    sx, sy, sz = (list(hs["size"]) + [0.0, 0.0, 0.0])[:3]
    size = {"cylinder": (sx, sz / 2.0), "sphere": (sx,),
            "box": (sx / 2.0, sy / 2.0, sz / 2.0)}[hs["shape"]]
    return GraspProp(id=str(hs["obj"]).rstrip("0123456789"), shape=hs["shape"],
                     size=tuple(float(v) for v in size), mass=float(hs["mass"]),
                     jaw_ctrl_m=moss.GRASP_JAW_CTRL_M,
                     grasp_height_m=moss.GRASP_HEIGHT_M)


def sample_prop(rng, litter: bool = False, ball_frac: float = 0.0) -> GraspProp:
    """One piece of litter: shape, size and mass drawn per episode.

    Deliberately a FAMILY OF RIGID PRIMITIVES rather than a deformable sheet.
    MuJoCo 3.10 has `flex` and could simulate paper, but a flex sheet inside a
    gripper at 50 Hz across 32 envs costs more than the whole rest of this env,
    and most real litter is crumpled rather than flat — a lump is the honest
    cheap model, and a thin box stands in for a flattened card.
    """
    kinds = ("can", "tall", "squat", "block", "card", "ball")
    # `ball_frac` draws nothing when 0, so every earlier run's episodes replay
    if ball_frac > 0.0 and rng.random() < ball_frac:
        kind = "ball"
    else:
        kind = str(rng.choice(kinds + LITTER_KINDS if litter else kinds))
    if kind in LITTER_KINDS:
        return sample_litter(rng, kind)
    if kind in ("can", "tall", "squat"):
        r = float(rng.uniform(0.022, 0.034))
        h = (float(rng.uniform(0.055, 0.090)) if kind == "tall"
             else float(rng.uniform(0.018, 0.035)) if kind == "squat"
             else float(rng.uniform(0.030, 0.075)))
        return GraspProp(id=kind, shape="cylinder", size=(r, h),
                         mass=float(rng.uniform(0.010, 0.060)),
                         jaw_ctrl_m=moss.GRASP_JAW_CTRL_M,
                         grasp_height_m=moss.GRASP_HEIGHT_M,
                         condim=6, friction=(0.5, 0.005, 0.002),
                         rgba=(0.93, 0.45, 0.38, 1.0))
    if kind == "ball":
        return GraspProp(id="ball", shape="sphere",
                         size=(float(rng.uniform(0.018, 0.032)),),
                         mass=float(rng.uniform(0.008, 0.040)),
                         jaw_ctrl_m=moss.GRASP_JAW_CTRL_M,
                         grasp_height_m=moss.GRASP_HEIGHT_M,
                         condim=6, friction=(0.5, 0.005, 0.002),
                         rgba=(0.45, 0.72, 0.55, 1.0))
    if kind == "card":
        return GraspProp(id="card", shape="box",
                         size=(float(rng.uniform(0.020, 0.035)),
                               float(rng.uniform(0.012, 0.022)),
                               float(rng.uniform(0.002, 0.006))),
                         mass=float(rng.uniform(0.005, 0.015)),
                         jaw_ctrl_m=moss.GRASP_JAW_CTRL_M,
                         grasp_height_m=moss.GRASP_HEIGHT_M,
                         condim=4, friction=(0.9, 0.005, 0.0001),
                         rgba=(0.85, 0.80, 0.62, 1.0))
    e = float(rng.uniform(0.014, 0.028))
    return GraspProp(id="block", shape="box", size=(e, e, e),
                     mass=float(rng.uniform(0.010, 0.050)),
                     jaw_ctrl_m=moss.GRASP_JAW_CTRL_M,
                     grasp_height_m=moss.GRASP_HEIGHT_M,
                     condim=3, friction=(0.8, 0.005, 0.0001),
                     rgba=(0.55, 0.60, 0.85, 1.0))
#: Range noise on it. Worse than the front camera's 0.015 m in absolute terms
#: would be wrong — it is much closer to its subject — but it is a monocular
#: estimate, so it is not free either. A guess until measured, like the rest.
ARM_DET_RANGE_NOISE = 0.008
#: Inside this horizontal radius the descent is the grasp, not an approach.
LOW_APPROACH_RADIUS_M = 0.08
#: Which action drives the roll — JOINT_NAMES index, inside the arm slice.
ACT_WRIST_ROLL = moss.ACT_ARM.start + moss.JOINT_NAMES.index("wrist_roll")
#: **The belief PREDICTS the can, not just the robot.** Between camera frames
#: the fix used to be carried by ego motion alone, which is right for a can
#: that is standing still and wrong for the case this env exists to train: a
#: can that has just been shoved. So successive fixes give a velocity, and the
#: fix is advanced along it — the constant-velocity step every real tracker
#: takes — with the same damping the floor applies, so the prediction stops
#: when the can would.
#:
#: **MEASURED, and it does not pay — so it ships OFF.** `VEL_EMA = 0` holds
#: the belief where it was last seen; any positive value extrapolates. Mean
#: |belief - truth| over 6 seeds, 120 control steps each, against the truth:
#:
#:     setting                     all frames    between camera frames
#:     hold where last seen          15.5 mm            16.4 mm
#:     predict above 0.05 m/s        17.4 mm            21.3 mm   <- worse
#:     predict above 0.15 m/s        15.6 mm            16.4 mm   <- no-op
#:     predict above 0.30 m/s        15.5 mm            16.4 mm   <- no-op
#:
#: The reason is arithmetic, not implementation: a velocity differenced from
#: two 10 Hz fixes with 15 mm of range noise carries ~0.2 m/s of noise, which
#: is the same size as the shoves it is meant to track. Gate it below that and
#: it extrapolates noise; gate it above and it never fires. A tracker cannot
#: separate a moving can from a noisy one at this frame rate and this sigma.
#:
#: The machinery stays because the measurement is the useful artifact and
#: because the conclusion is a property of THIS detector: at a higher frame
#: rate, or with the D455's depth stream instead of a width-inverted range,
#: the arithmetic changes and `VEL_EMA` is where it turns back on.
VEL_EMA = 0.0
VEL_DAMP = 0.88
VEL_FLOOR = 0.15          # m/s — above the 0.2 m/s noise floor, when enabled
#: No prediction may run further than this from the last real sighting. A
#: tracker that extrapolates without limit invents an object.
MAX_PREDICT_M = 0.12

#: **The can SHIFTS.** Per control step, the chance of a small random shove —
#: which is what happens on the real floor when a track clips it, and what
#: makes a stale fix wrong rather than merely old. Without this the belief
#: model teaches only latency; with it, it teaches recovery.
CAN_SHIFT_PROB = 0.02
#: How often the can spawns LYING DOWN rather than standing. Measured from
#: the scripted loop, where the approach topples it far more often than not;
#: 0.5 trains both and keeps the upright case, which is what a can that has
#: not been driven into yet looks like.
CAN_LYING_P = 0.5
#: **A can CAUGHT MID-TOPPLE**, at an angle that is not a resting pose. The
#: sampler only ever produced the two STABLE poses — exactly upright or
#: exactly on its side — so nothing in training was a can in the act of
#: falling, which is what the approach actually creates: the chassis clips
#: it, it starts over, and the policy meets it at 40 degrees with angular
#: velocity on it. Spotted by a human watching the lab.
CAN_TILTED_P = 0.2
#: The tilt band for that. Not near 0 or 90 — those are the stable cases
#: already covered, and a pose 2 degrees off upright teaches nothing new.
CAN_TILT_RANGE_DEG = (25.0, 70.0)
#: How fast it is falling when the episode starts, rad/s about the tip axis.
#: A tilted can with zero angular velocity is a statue for the first frames;
#: a real one is already going over.
CAN_TOPPLE_RATE = (0.5, 2.5)
#: **The can's POSE randomisation is not domain randomisation.** It was
#: gated on `domain_rand`, which the lab's preview pins False to protect the
#: shared mjModel from being written to — so the watched trainee got an
#: upright can EVERY episode while the trainer saw the 50/50 mix. A resting
#: pose is a qpos choice and writes no model field, so it is separated here
#: and defaults on for both.
CAN_POSE_RAND_DEFAULT = True
#: **A can LEANING against something**, which the sampler cannot produce on
#: its own: every pose it makes is free-standing, so the one case the room
#: actually creates — a can that rolled to the skirting and stopped against
#: it at an angle — was never trained. 6 of 9 undelivered cans in a room run
#: end within 0.25 m of a wall, so this is the state the loop most often
#: gives up on.
#:
#: The support is a low static kerb behind the can, standing in for a wall,
#: a skirting board or a chair leg. It is a REAL obstacle, not a pose trick:
#: the arm has to take the can without driving it into the kerb, which is
#: the thing the room asks for and the bare scene never did.
#: **0.0 — NOT USED, kept as the record of four failed attempts.** The idea
#: is right and the implementation is not: a can posed at a lean angle does
#: not stay there. Placing the kerb along the yaw direction, then along the
#: can's true tip axis, then at a corrected face distance, then letting
#: physics settle it — 0 of 11 cans ended up resting against the obstacle,
#: median 220 mm away from it, because a free cylinder tipping over slides
#: out from under itself rather than leaning.
#:
#: And it BROKE THE DRILL RUNG, which is the part worth keeping: rung 0
#: exists to spawn the can between the pads, and the settle moved it 103 mm
#: away — `test_the_pick_rungs_are_a_spawn_ladder` caught it. A spawn family
#: that silently rewrites another rung is worse than a missing one.
#:
#: What it would take, for whoever picks this up: the can has to be BUILT
#: resting — solve for the contact pose against the kerb rather than posing
#: an angle and hoping — or the kerb has to be a concave corner that a
#: toppling can cannot slide out of. The `lean_kerb` mocap body stays in the
#: scene, parked, so that work does not start from nothing.
CAN_LEANING_P = 0.0
#: The kerb: half-extents and how far behind the can it sits. Low enough
#: that the arm can come down over the can, high enough to hold it at an
#: angle rather than letting it lie flat.
LEAN_KERB_SIZE = (0.05, 0.12, 0.045)
#: Gap between the can's side and the kerb's FACE. Small: the can has to
#: be resting on it at t=0, not falling toward it from a distance.
LEAN_KERB_GAP = 0.004
CAN_SHIFT_IMPULSE = 0.02                   # m/s, applied to the can's own dof

#: How far a finger has to stall SHORT of its command to count as gripping
#: something. MEASURED: closing on air the servo reaches its target to well
#: under a millimetre, and closing on his 66 mm can at `GRASP_JAW_CTRL_M` it
#: sits ~6 mm short — which is the interference itself, read back through the
#: servo. 2 mm is comfortably between the two, and it is the same quantity an
#: Innate-style code skill reconstructs from `present_load` on the robot.
GRIP_STALL_M = 0.002
#: How far off the floor the can has to be before the jaws are CARRYING it
#: rather than squeezing it in place. Small — a can lifted by a centimetre is
#: unambiguously off the ground — but it is what separates a grip from a
#: crush, and without it a policy learned to crush.
AIRBORNE_M = 0.010
#: Success: the can is held this far above where it started.
SUCCESS_LIFT_M = 0.08
#: ...AND HELD DEEP: the object's centre within this of the tool point when
#: it is lifted (0 = off, as every earlier run trained). MEASURED in moss-yard
#: (2026-09-26, 34 carries): grips more than 45 mm from the tool point were
#: delivered 0/7 — every one slid out between the pads on the lift or the
#: swing — while `_held` accepts anything within 55 mm.
DEEP_GRIP_M = float(os.environ.get("MICRODUCK_MOSS_DEEP_GRIP", "0") or 0.0)
#: ...AND A SPHERE HELD THROUGH ITS CENTRE: its centre within this of the
#: midpoint between the pads (0 = off). The pads pinching a ball AHEAD of its
#: centre squeeze it out along the jaw, level or not. MEASURED on 478dad's
#: env ball picks (2026-09-27, 120 episodes, then the yard's lift ramp and
#: swing to STOW_HIGH): centre under 16 mm from the pad midpoint 33/33 kept,
#: 16-20 mm 9/12, over 20 mm 2/9. (CORRECTED 2026-09-28: the pad midpoint
#: IS the tool point at every jaw opening once both fingers move — they are
#: tied; measured 0.0 mm at 41/20/10/0 mm. An earlier "10-15 mm apart" came
#: from moving one finger alone. So this is also the distance to the tool
#: point.)
SPHERE_CENTRE_M = float(os.environ.get("MICRODUCK_MOSS_SPHERE_CENTRE", "0") or 0.0)
#: START WHERE THE YARD REALLY HANDS OVER — the whole state, not just where the
#: object is. `data/moss_pick_handover_states_yard.json`: every `creep` entry of
#: tidy_moss with 478dad in moss-yard, seeds 13-72 (2026-09-27): the arm's
#: joints, the target object's pose in the base frame and its shape, size and
#: mass. From those states the pick grips at the fingertips 9 times in 24,
#: from the env's own spawns 3 in 43 — and fingertip grips are most of the
#: yard's lift losses. `HANDOVER_BANK` holds only the object's x, y and
#: uprightness with the arm at GRASP_POSE. Seeds 1-12 are held out to score.
HANDOVER_STATES = os.environ.get("MICRODUCK_MOSS_HANDOVER_STATES", "0") == "1"
#: THE SHARE OF EPISODES THAT ARE A BALL (on top of the variety's own sixth).
#: A pinched ball is squeezed out of the jaws unless its centre sits between
#: the pads (`SPHERE_CENTRE_M`); a fine-tune asking for that with balls in a
#: sixth of its episodes did not learn it (0281e5). 0 = the variety as is.
BALL_FRAC = float(os.environ.get("MICRODUCK_MOSS_BALL_FRAC", "0") or 0.0)
HANDOVER_STATES_FILE = (Path(__file__).parent / "data"
                        / "moss_pick_handover_states_yard.json")
#: What `publish_grip` puts in `moss.OBS_GRIP` when the depth camera has no
#: fresh fix: a grip distance no jaw closes on.
GRIP_UNSEEN_M = 0.15
#: START WHERE THE YARD HANDS OVER. `data/moss_pick_handovers_yard.npy`: 310
#: real handovers — the object nearest the jaws the tick tidy_moss enters
#: `creep` (moss-yard seeds 10-41; objects already in the bin removed).
#: Columns: base-frame x, y, z, |cos tilt|, robot speed, shape (can, block,
#: squat, ball), seed. MEASURED: 81% are NEARER than the rung-2 box (median
#: x 0.269 against 0.36-0.47) and only 4-7% fall inside any rung's box, and
#: 42% are lying (knocked over on the way in). The six-shape pick ad9876,
#: trained in the box, put 0-5 objects in the yard's bin against 13 for a
#: can-only pick. The bank gives the position and how upright it stands;
#: shape and size still come from `sample_prop`.
HANDOVER_BANK = os.environ.get("MICRODUCK_MOSS_HANDOVER_BANK", "0") not in ("", "0")
HANDOVER_FILE = Path(__file__).parent / "data" / "moss_pick_handovers_yard.npy"
HANDOVER_XY_SD = 0.01
#: THE SPAWN BOX, overriding the rung's: "x_lo,x_hi,y_max" (base frame, m).
#: For teaching the grasp CLOSE IN, from above: MEASURED 2026-09-26, ad9876
#: (last trained at 0.36-0.47, arm stretched) makes deep picks 93% at
#: 0.36-0.42 m, 75% at 0.28-0.34 and 65% at 0.22-0.28 — it learned the lunge;
#: `GRASP_POSE` itself puts the jaws at ~0.26 m, reaching down.
PICK_BOX = os.environ.get("MICRODUCK_MOSS_PICK_BOX", "")
#: ...AND STILL HELD THIS LONG AFTERWARDS. Terminating the instant the can
#: crosses `SUCCESS_LIFT_M` pays for LIFTING, not for a grip, and the two
#: come apart: measured 2026-09-24 over 48 seeds, the shipped leg and a
#: fine-tune of it score 93% and 91% on the instant test — a tie — and 62%
#: against 25% on this one. The mission ranks them the same way this does
#: (27/72 cans against 16/72), and the OPPOSITE of what the lift rate alone
#: suggested: the fine-tune lifts MORE often (61% vs 53% of creep attempts)
#: and delivers fewer, because the extra lifts are marginal grips that clear
#: the grasp gate and are lost during the carry.
#:
#: One second at 25 Hz, which is long enough for a grip that is only wedged
#: to slip and short enough to leave the episode budget intact.
PICK_HOLD_STEPS = int(1.0 * moss.CONTROL_HZ)
#: **NOT USED.** Scoring the hold alone was
#: still the wrong horizon: trained against it (500k continuation) a leg went
#: from 62% to 81% sustained picks and DELIVERED FEWER cans, 19/72 against
#: the shipped 27/72, with an identical creep->lift rate of 53%. Equal lifts
#: and unequal deliveries puts the whole difference downstream of the lift —
#: in the pose and grip the carry inherits, which this env did not model at
#: all because it ended at the lift.
#:
#: So the horizon was extended: once the hold is proved, ramp to
#: `moss.LIFT_POSE` the way `brain/tidy_moss`'s `lift` state does and require
#: the can still held at the end (`_lift_handover`). **That failed the same
#: check, before anything was trained on it.** Scored over 24 seeds it ranks
#:
#:     leg                     handover criterion    cans delivered
#:     teach-moss_pick-5e9df7        79%                26%
#:     moss-pick-v1                  58%                38%
#:     teach-moss_pick-d879c3        21%                22%
#:
#: — it puts 5e9df7 top and the mission puts it second. FOUR pick criteria
#: have now been tried against delivery: the instant test (uncorrelated), the
#: sustained hold (correlated across legs, inverted under optimisation), the
#: lift rate (flat), and this one (mis-ranks without optimisation at all).
#:
#: The conclusion is structural rather than a search for a fifth: the losses
#: are spread across the WHOLE carry — lift, then the up-round-down stow,
#: where the shoulder swing alone takes the hold from 57% to 40% — so an env
#: that stops anywhere short of the bin cannot rank a pick leg by cans. The
#: seam needs a task that spans it, or the carry needs fixing directly.
#: Kept, uncalled, so the next person does not rebuild it.
LIFT_TEST_S = 1.8      # `TidyMossParams.lift_s`, the mission's own ramp
#: The band outside which the can is gone for this episode.
BAND_X = (0.10, 0.70)
BAND_Y = 0.30


#: EVERY ctor flag that changes what the policy SEES. This list being
#: incomplete is worse than not having it: `publish_size` was added after the
#: helper and not listed, so a probe built the env without it, fed a policy
#: trained on slot 31 a zero there, and measured 3 grips where the policy
#: really makes 65 — a "result" that would have been reported as a collapse.
#: A flag that changes the OBSERVATION belongs here; one that changes only the
#: reward (jaw_align, can_topple, gap_tcp) does not.
OBS_FLAGS = ("publish_attitude", "publish_size", "publish_proximity",
             "publish_grip", "grip_xyz")
#: ...and the flags that change the DYNAMICS a policy trained under. Not the
#: observation, so not in OBS_FLAGS, but just as fatal to get wrong: a policy
#: trained with `base_lock` learns its base outputs are no-ops and leaves them
#: unconstrained, so EVALUATING it without the lock lets that garbage drive the
#: robot — measured, it reported 23.7 deg of chassis turn for a policy that
#: physically could not turn while training. Third instance of the same
#: mistake in one day: a guard that covers some of the flags reads as covering
#: all of them.
DYN_FLAGS = ("base_lock",)
EVAL_FLAGS = OBS_FLAGS + DYN_FLAGS


def obs_env_kwargs(policy_or_run, flags=OBS_FLAGS) -> dict:
    """The env settings a POLICY must be evaluated under, read off its run.

    A policy trained with the can's axis in slots 28-30 and one trained
    without are not interchangeable, and nothing about the .onnx file says
    which it is. An axis-blind export's normalizer has var 3e-10 in those
    slots, so publishing a live axis to it clips three of thirty-two inputs
    at the bound: MEASURED, the shipped leg went 10/12 -> 0/12 and another
    went 59/60 -> 0/60 in a probe written to compare them — the same mistake
    twice, once in the code and once in the measurement of the code.

    So consumers ask the RUN, not the person running them. Accepts a run
    directory or any path inside one (`runs/x/policy.onnx`). Runs from before
    the variant was recorded are axis-blind, which is the safe default and
    also the truth.
    """
    p = Path(policy_or_run)
    for cand in (p, p.parent, p.parent.parent):
        meta = cand / "run.json"
        if meta.is_file():
            try:
                kw = (json.loads(meta.read_text()).get("env_kwargs") or {})
            except (OSError, ValueError):
                return {"publish_attitude": False}
            return {k: bool(kw.get(k, False)) for k in flags}
    return {k: False for k in flags}


def _run_env_kwargs(policy_or_run) -> dict | None:
    """A run's recorded `env_kwargs`, found as `obs_env_kwargs` finds them;
    None when there is no readable run.json."""
    p = Path(policy_or_run)
    for cand in (p, p.parent, p.parent.parent):
        meta = cand / "run.json"
        if meta.is_file():
            try:
                return json.loads(meta.read_text()).get("env_kwargs") or {}
            except (OSError, ValueError):
                return None
    return None


def arm_camera_mount_of(policy_or_run) -> int | None:
    """The wrist-camera mount a run TRAINED behind (`moss.ARM_CAMERA_MOUNT`).

    A run.json without the key predates it and trained on the retired mount
    past the jaw tips (`ARM_CAMERA_MOUNT_LEGACY`); no run.json at all is
    unknown (None).
    """
    kw = _run_env_kwargs(policy_or_run)
    if kw is None:
        return None
    return int(kw.get("arm_camera_mount", moss.ARM_CAMERA_MOUNT_LEGACY))


def wrist_mount_mismatch(policy_or_run, who: str) -> str | None:
    """Why a leg's wrist inputs are not what it learned, or None: it trained
    behind another mount AND the wrist reached its observation (attitude,
    proximity, or a stow's arm-camera drop point). A leg that never saw the
    wrist camera is not changed by moving it."""
    kw = _run_env_kwargs(policy_or_run)
    mount = arm_camera_mount_of(policy_or_run)
    if kw is None or mount == moss.ARM_CAMERA_MOUNT:
        return None
    if not any(kw.get(k) for k in ("publish_attitude", "publish_proximity",
                                   "drop_target")):
        return None
    return mount_mismatch_message(mount, who)


def mount_mismatch_message(trained: int, who: str = "this policy") -> str:
    return (f"{who} trained behind wrist-camera mount {trained}; the model "
            f"carries mount {moss.ARM_CAMERA_MOUNT} (moss.ARM_CAMERA_MOUNT), so "
            "its wrist slots 26 and 28-30 are not the readings it learned")


def episode_s_of(policy_or_run) -> float | None:
    """The pick episode a run trained in (`max_episode_s`), or None when its
    run.json does not record one (every run before 2026-09-27: 8 s)."""
    p = Path(policy_or_run)
    for cand in (p, p.parent, p.parent.parent):
        meta = cand / "run.json"
        if meta.is_file():
            try:
                kw = json.loads(meta.read_text()).get("env_kwargs") or {}
            except (OSError, ValueError):
                return None
            v = kw.get("max_episode_s")
            return float(v) if v else None
    return None


def eval_env_kwargs(policy_or_run) -> dict:
    """Every flag an EVALUATION must match the run's training on.

    `obs_env_kwargs` answers for what the policy SEES; this adds what it was
    ALLOWED TO DO. Both have to match or the measurement is of a different
    robot than the one that was trained.
    """
    out = obs_env_kwargs(policy_or_run, EVAL_FLAGS)
    mount = arm_camera_mount_of(policy_or_run)
    if mount is not None:
        out["arm_camera_mount"] = mount
    return out


def scene_spec(rung: int, prop: GraspProp | str = DEFAULT_PROP,
               clutter: tuple = (), clutter_poses=None):
    """MOSS's own scene with one PROP in it, at his 2 ms.

    `clutter` adds more props as free bodies AFTER everything else, so the
    prop's own qpos/qvel addresses do not move. They are parked off-scene in
    the keyframe; `MossStowEnv._place_clutter` drops them into the bin.
    """
    p = GRASP_PROPS[prop] if isinstance(prop, str) else prop
    spec = moss.scene_spec()
    p.add_to(spec, name="can")
    # The kerb a leaning can rests against. Always in the model, parked far
    # out of the way, and MOVED INTO PLACE on the episodes that lean —
    # adding or removing a geom would mean recompiling per reset.
    # MOCAP, so it moves through `data.mocap_pos` rather than `model.body_pos`:
    # a model write would bar this env from the lab's shared-model scope, and
    # the whole point is that the watched preview runs the same episodes the
    # trainer does.
    kerb = spec.worldbody.add_body(name="lean_kerb", pos=[6.0, 6.0, 0.045],
                                   mocap=True)
    kerb.add_geom(name="lean_kerb_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=list(LEAN_KERB_SIZE), rgba=[0.45, 0.45, 0.48, 1.0],
                  priority=1, friction=[0.8, 0.005, 0.0001])
    key = spec.key(moss.HOME_KEY)
    key.qpos = list(np.asarray(key.qpos, float)) + [
        POCKET[0], 0.0, p.half_height, 1.0, 0.0, 0.0, 0.0]
    if clutter and clutter_poses is not None:
        # STATIC: welded to the rover, so they ride with the bin and MuJoCo
        # never collides them with the bin or each other (one weld group) —
        # only with the arm and the object being dropped. As free bodies they
        # cost 7x per step (29 resting contacts against 6) and 10x per reset.
        rover = spec.body("rover")
        for k, (c, pose) in enumerate(zip(clutter, clutter_poses)):
            c.add_to(spec, name=f"clutter{k}", parent=rover, pose=pose)
        return spec
    for k, c in enumerate(clutter):
        c.add_to(spec, name=f"clutter{k}")
        key.qpos = list(np.asarray(key.qpos, float)) + [
            6.0 + 0.3 * k, -6.0, c.half_height, 1.0, 0.0, 0.0, 0.0]
    return spec


#: Compiled scenes for props that recur, keyed by (rung, prop, ...) — see
#: `MossPickEnv._bind_model`. Per process; a handful of entries.
#:
#: SHARED BETWEEN INSTANCES, and handed out at RESET time — so an env that is
#: stepping and a `vec_env._SpareReset` spare that is resetting on another
#: thread can hold the same `MjModel`. That is safe only because nothing here
#: writes into a compiled model, which is exactly the promise
#: `MossPickEnv.shares_compiled_model` makes to `vec_env._build_env`.
_RECURRING_SCENES: dict = {}
#: Guards compile-and-store: two spares missing the same key would otherwise
#: both compile, and the loser's model would be dropped from the cache while
#: the instance that took it was still stepping it.
_RECURRING_LOCK = threading.Lock()


def clear_scene_cache() -> None:
    """Drop every cached scene.

    `_scene_inputs` cannot see every input (read its docstring), so anything
    that monkeypatches a `moss` or `moss_env` constant and then builds a
    scene must call this. `tests/conftest.py` calls it around every test, so
    a test does not have to remember.
    """
    with _RECURRING_LOCK:
        _RECURRING_SCENES.clear()


def _scene_inputs() -> tuple:
    """The scene inputs outside `scene_spec`'s arguments that actually VARY
    at runtime, read now: where MOSS's files are, the V0.4 collision switch,
    the wrist camera's mount, the clock and the solver. A cached scene is
    reused only when all of it still matches — a test that points
    `MICRODUCK_MOSS_DIR` elsewhere or moves the camera must not get the last
    test's model.

    NOT EVERY INPUT, and deliberately so. `scene_spec` also reads this
    module's `GRASP_PROPS`, `POCKET` and `LEAN_KERB_SIZE`, and
    `moss.scene_spec()` reads many more `moss` constants than the six here —
    `BIN_VOID_Z`, `BIN_FLOOR_PRIORITY`, `FINGER_FRICTION` among them. Those
    are compile-time constants nothing varies in anger, and reading them all
    on every reset would cost more than the compile this saves. The hazard
    they leave is a TEST that patches one and then takes the handover path:
    it would be served the model compiled BEFORE the patch and would
    silently measure the unpatched physics, green. `clear_scene_cache()` is
    the answer to that one, not a wider key.
    """
    return (str(moss.CACHE_DIR), moss.collision_v04(),
            tuple(moss.ARM_CAMERA_POS), tuple(moss.ARM_CAMERA_AIM),
            float(moss.GRASP_PHYSICS_DT),
            tuple(sorted((k, repr(v)) for k, v in moss.SOLVER_OPTIONS.items())))


class MossPickEnv(gym.Env):
    """Drive the can into the jaws and lift it. 32 obs, 9 actions, 25 Hz."""

    metadata = {"render_modes": []}
    #: Reset on a spare instance while this one steps (`vec_env._SpareReset`):
    #: a reset here is ~50 steps of work — the 2 s deploy plus a recompile —
    #: and a pick episode is ~45 steps, so in lockstep it idled the fleet.
    background_reset = True
    #: ...and SEVERAL of those instances may end up holding ONE compiled
    #: model, because `_RECURRING_SCENES` caches a recurring prop's scene
    #: process-wide and hands it out at RESET time — long after
    #: `vec_env._build_env`'s construction-time check has passed, so that
    #: check cannot see it. This flag is the promise it asks for instead:
    #: nothing in this env writes into `self.model` once it is compiled
    #: (`_bind_model`, `MossDriver.__init__` and `mj_step` only read it, and
    #: `physics_contact` is applied to the SPEC before the compile), so a
    #: spare's reset thread cannot race the active instance's steps. An env
    #: that gains a per-step model write — the walk env's `_sync_model`
    #: domain randomization is the one in this repo — must NOT set this.
    #: `viz_server.Duck._make_env` makes the same call for the lab's preview
    #: envs under the name `writes_model`, and for the same reason.
    shares_compiled_model = True

    def __init__(
        self,
        task: str = "pick",
        *,
        max_episode_s: float = EPISODE_S,
        seed: int | None = None,
        # the trainer's shared knobs; the ones with no meaning here are
        # ignored rather than silently reinterpreted.
        domain_rand: bool = True,
        can_pose_rand: bool | None = None,
        obs_noise: bool = True,
        action_delay: bool = False,
        random_yaw: bool = False,
        push_robot: bool = False,
        actuator: str | None = None,
        actuator_force: str | None = None,
        # this env's own
        pick_rung: int = DEFAULT_RUNG,
        publish_attitude: bool = ATTITUDE_DEFAULT,
        wrist_start_rand: float = WRIST_START_RAND,
        jaw_align: float = W_JAW_ALIGN,
        align_hold: float = W_ALIGN_HOLD,
        torque_sat: float = W_TORQUE_SAT,
        overspeed: float = W_OVERSPEED,
        arm_floor: float = W_ARM_FLOOR,
        base_lock: bool = BASE_LOCK,
        publish_proximity: bool = PUBLISH_PROXIMITY,
        prop_variety: bool = PROP_VARIETY,
        wrist_drill: bool = WRIST_DRILL,
        low_approach: float = W_LOW_APPROACH,
        gap_scale: float = GAP_TCP_SCALE,
        wrist_free: bool = WRIST_FREE,
        can_topple: float = W_CAN_TOPPLE,
        publish_size: bool = PUBLISH_SIZE,
        prop: GraspProp | str = DEFAULT_PROP,
        handover_bank: bool = HANDOVER_BANK,
        deep_grip_m: float = DEEP_GRIP_M,
        pick_box: str = PICK_BOX,
        gap_from_tcp: bool | None = None,
        litter: bool | None = None,
        sphere_centre_m: float | None = None,
        arm_camera_mount: int | None = None,
        publish_grip: bool = False,
        handover_states: bool | None = None,
        ball_frac: float | None = None,
        grip_xyz: bool = False,
        handover_frac: float = 1.0,
    ):
        #: The wrist mount the policy behind this env TRAINED on, when the
        #: caller knows it (`arm_camera_mount_of`). The model always carries
        #: the current one; a mismatch is allowed — it is how an old leg is
        #: measured on the honest sensor — but never silent when the wrist
        #: reaches the observation (attitude, proximity, or a stow's drop
        #: point, which `MossStowEnv` sets before calling up here).
        self.arm_camera_mount = (moss.ARM_CAMERA_MOUNT if arm_camera_mount is None
                                 else int(arm_camera_mount))
        if (self.arm_camera_mount != moss.ARM_CAMERA_MOUNT
                and (publish_attitude or publish_proximity
                     or getattr(self, "drop_target", False))):
            warnings.warn(mount_mismatch_message(self.arm_camera_mount),
                          stacklevel=2)
        if task not in TASKS:
            raise SystemExit(f"unknown --task {task!r} for moss "
                             f"(have: {', '.join(TASKS)})")
        if actuator == "bam":
            raise SystemExit(
                "--actuator bam is the DUCK's XL330 fit; MOSS's arm is an "
                "SO-101 driven by the position servos in his own MJCF "
                "(kp 70 / kv 3), and pretending one switch covers it would "
                "put a servo model on this robot that nobody measured")
        if int(pick_rung) not in RUNGS:
            raise SystemExit(
                f"pick_rung={pick_rung!r} is not a rung of the ladder "
                f"({', '.join(str(r) for r in RUNGS)}) — a spawn box nobody "
                "measured is not a curriculum")
        self.rung = int(pick_rung)
        self.deep_grip_m = float(deep_grip_m)
        self.sphere_centre_m = float(SPHERE_CENTRE_M if sphere_centre_m is None
                                     else sphere_centre_m)
        #: The wrist depth camera's grip fix in `moss.OBS_GRIP` (see there).
        self.publish_grip = bool(publish_grip)
        self.ball_frac = float(BALL_FRAC if ball_frac is None else ball_frac)
        #: the grip IN FULL — tool-frame x, y, z and fresh (`moss.OBS_GRIP_XYZ`)
        self.grip_xyz = bool(grip_xyz)
        #: the share of episodes started from a yard handover state when
        #: `handover_states` is on; the rest spawn from the rung's box
        self.handover_frac = float(handover_frac)
        self._grip = None
        self._grip_t = -1e9
        self.handover_states = bool(HANDOVER_STATES if handover_states is None
                                    else handover_states)
        self._hstates = (json.loads(HANDOVER_STATES_FILE.read_text())
                         if self.handover_states else None)
        self._hs = None
        #: A CONSTRUCTOR ARGUMENT, not only the import-time constant, so it
        #: reaches run.json. MEASURED 2026-09-27: ad9876 replays at +86/ep
        #: with it on (its training reported +88) and -3 with it off; every
        #: fine-tune launched without it trained on the chassis-distance
        #: progress term and learned to DRAG objects toward the robot
        #: (+14..+17 cm, 55-60 of 60 episodes) instead of grasping them.
        self.gap_from_tcp = bool(GAP_FROM_TCP if gap_from_tcp is None
                                 else gap_from_tcp)
        self.pick_box = (tuple(float(v) for v in pick_box.split(","))
                         if pick_box else None)
        self._handovers = np.load(HANDOVER_FILE) if handover_bank else None
        #: Whether the can's AXIS reaches slots 28-30. Off unless asked for:
        #: see `ATTITUDE_DEFAULT` for the 10/12 -> 0/12 that decided it.
        self.publish_attitude = bool(publish_attitude)
        self.wrist_start_rand = float(wrist_start_rand)
        self.jaw_align = float(jaw_align)
        self.align_hold = float(align_hold)
        self.torque_sat = float(torque_sat)
        self.overspeed = float(overspeed)
        self.arm_floor = float(arm_floor)
        self.base_lock = bool(base_lock)
        self.publish_proximity = bool(publish_proximity)
        self.prop_variety = bool(prop_variety)
        self.litter = bool(LITTER if litter is None else litter)
        self.wrist_drill = bool(wrist_drill)
        if self.wrist_drill:
            # The drill is ABOUT the wrist, so it always starts somewhere in
            # its arc — set on the attribute, not the local, which is the
            # difference between doing this and appearing to.
            self.wrist_start_rand = max(self.wrist_start_rand, math.pi / 2)
        #: Last wrist-camera range, with its own freshness — the proximity the
        #: grasp is flown on. None until the wrist first sees the can.
        self._arm_rng = None
        self._arm_rng_t = -1e9
        self.low_approach = float(low_approach)
        self.gap_scale = float(gap_scale)
        self.wrist_free = bool(wrist_free)
        self.can_topple = float(can_topple)
        self.publish_size = bool(publish_size)
        if isinstance(prop, str) and prop not in GRASP_PROPS:
            raise SystemExit(
                f"unknown prop {prop!r} for moss (have: "
                f"{', '.join(GRASP_PROPS)}) — an object this jaw has never "
                "been measured against has no grasp numbers, and guessing "
                "them is how a policy trains against a grip that cannot close")
        self.prop = GRASP_PROPS[prop] if isinstance(prop, str) else prop
        self.domain_rand = bool(domain_rand)
        self.obs_noise = bool(obs_noise)
        self.max_steps = int(round(max_episode_s / CTRL_DT))
        self.rng = np.random.default_rng(seed)
        self.can_pose_rand = (CAN_POSE_RAND_DEFAULT
                              if can_pose_rand is None else bool(can_pose_rand))

        self._bind_model()
        self.arm_cmd[moss.GRIPPER_JOINT] = 0.041

        self.action_space = spaces.Box(-1.0, 1.0, (moss.NUM_ACTIONS,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            (moss.OBS_DIM,), np.float32)
        self.last_action = np.zeros(moss.NUM_ACTIONS, np.float32)
        self.prev_action = np.zeros(moss.NUM_ACTIONS, np.float32)
        #: The lab's trainee preview reads this off the env by name, as it does
        #: on every other body's (`walk_env`, `mars_env`). Called `steps` for one
        #: run, which killed the lab's whole duck loop the moment a MOSS
        #: trainee joined the roster.
        self.step_count = 0
        self._start_z = self.prop.half_height
        self._prev_gap = 0.0
        self._prev_lift = 0.0
        self._hold_streak = 0
        #: The wrist camera's last attitude report, and when it landed.
        self._att: tuple[float, float, float] | None = None
        self._att_t = -1e9
        # The FRONT camera's attitude belief, kept apart from the wrist's so
        # the fresher and better source can win per tick rather than the two
        # overwriting each other.
        self._fatt = None
        self._fatt_t = -1e9
        self._pending_fatt = []
        self._next_arm_t = 0.0
        self._pending_att: list = []
        self._prev_can_w = None
        self._closed_once = False
        #: Where the can STARTED, for the shove abort. Set in `reset` once the
        #: can has actually been placed.
        self._can_spawn_w = None
        #: Last step's jaw alignment, for the progress-paid term.
        self._prev_align = None
        #: Last step's uprightness, for the topple charge.
        self._prev_upright = None
        #: The belief, in the WORLD frame — which is where a velocity means
        #: something. The observation converts it into the base frame every
        #: tick, so ego motion needs no bookkeeping of its own.
        self._fix: np.ndarray | None = None          # world (x, y), PREDICTED
        self._seen_w: np.ndarray | None = None       # world (x, y), last SIGHTING
        self._vel: np.ndarray = np.zeros(2)          # world m/s, estimated
        self._fix_t = -1e9
        self._next_det_t = 0.0
        self._pending: list[tuple[float, np.ndarray]] = []

    # ----------------------------------------------------------- geometry

    def _can_base(self) -> np.ndarray:
        """The can in the rover's own frame — what the obs carries and what
        the reward is measured on."""
        x, y, yaw = self.driver.pose(self.data)
        p = self.data.xpos[self.can_body]
        dx, dy = p[0] - x, p[1] - y
        c, s = math.cos(-yaw), math.sin(-yaw)
        return np.array([dx * c - dy * s, dx * s + dy * c, p[2]])

    def _gap(self) -> float:
        """Distance from the can to the grasp pocket, in the base frame.
        Planar: the can's height is the grasp's business, not the base's."""
        cb = self._can_base()
        return float(math.hypot(cb[0] - POCKET[0], cb[1] - POCKET[1]))

    def _jaw_alignment(self) -> tuple[float, float]:
        """`(alignment, lying)` — how squarely the jaw meets a lying can.

        A parallel jaw grips a cylinder ACROSS its diameter, so the opening
        axis wants to be perpendicular to the can's; `sin^2` of the difference
        is 1 there and 0 with the jaw lined up along the can. Both angles are
        modulo pi because neither has a head.

        `lying` is 1 for a can on its side and 0 for one standing: an upright
        can is a circle from above and every jaw angle grips it equally, so
        this must not pay for aiming at one.
        """
        c, sn, upright = self._true_attitude()
        lying = float(np.clip(1.0 - abs(upright), 0.0, 1.0))
        can_yaw = 0.5 * math.atan2(sn, c)
        fl = self.model.joint("finger_left")
        R = self.data.xmat[self.model.jnt_bodyid[fl.id]].reshape(3, 3)
        ax = R @ np.asarray(fl.axis, float)
        _x, _y, byaw = self.driver.pose(self.data)
        cb, sb = math.cos(-byaw), math.sin(-byaw)
        jaw_yaw = math.atan2(sb * ax[0] + cb * ax[1], cb * ax[0] - sb * ax[1])
        return float(math.sin(jaw_yaw - can_yaw) ** 2), lying

    def _reward_gap(self) -> float:
        """The distance the PROGRESS term is paid on.

        Kept separate from `_gap()`, which stays the base-frame measure the
        rung ladder and the in-reach base gate are defined against — the
        spawn boxes mean "this far ahead of the chassis" and a test asserts
        it. Only what the reward PAYS FOR moves to the gripper.
        """
        if not self.gap_from_tcp:
            return self._gap()
        can = self.data.xpos[self.can_body]
        tcp = self.data.site_xpos[self.tcp_site]
        return float(math.hypot(float(can[0] - tcp[0]),
                                float(can[1] - tcp[1])))

    def _sphere_off_centre(self) -> float:
        """How far the object's centre is from the midpoint between the
        pads, m (`SPHERE_CENTRE_M`)."""
        mid = (self.data.geom_xpos[self.model.geom("pad_left").id]
               + self.data.geom_xpos[self.model.geom("pad_right").id]) / 2.0
        return float(np.linalg.norm(self.data.xpos[self.can_body] - mid))

    def _closed_on_can(self) -> bool:
        """Jaws shut on the can, whether or not it is off the ground yet.

        The doorway `W_CLOSE_BONUS` pays for once. Distinct from `_held`,
        which additionally requires the can to be CARRIED — see there for why
        stall alone cannot tell a grip from a crush.
        """
        can = self.data.xpos[self.can_body]
        tcp = self.data.site_xpos[self.tcp_site]
        if float(np.linalg.norm(can - tcp)) > 0.055:
            return False
        cmd = self.arm_cmd[moss.GRIPPER_JOINT]
        got = float(self.data.qpos[
            self.model.joint(moss.GRIPPER_JOINT).qposadr[0]])
        return (got - cmd) > GRIP_STALL_M

    def _deliver_handover(self) -> bool:
        """Run the MISSION'S WHOLE CARRY and say whether the can reaches the
        bin — lift ramp, up, round, down, open, settle.

        This is the one pick criterion that cannot be misaligned, because it
        IS the mission's metric rather than a proxy for it. Four proxies were
        tried and each failed its check against cans delivered (see
        `PICK_HOLD_STEPS` and `_lift_handover`); the losses are spread across
        the whole carry — the shoulder swing alone takes the hold from 57% to
        40% — so nothing that stops short of the bin can rank a pick leg.

        The policy does not act during any of it, on purpose: in the mission
        every one of these states is scripted, so what is being scored is the
        grip and pose the policy HANDED OVER, which is the only thing it
        controls once the lift begins.

        **AND IT IS STILL NOT USED, which is the end of this line of work.**
        Scored over 24 seeds it delivers 3/24, 4/24 and 0/24 for
        `moss-pick-v1`, `teach-moss_pick-5e9df7` and `teach-moss_pick-d879c3`
        — putting 5e9df7 top, where the room puts it second — and 3 against 4
        successes is noise at any reading. Two things defeat it. The signal is
        far too sparse to train on: one delivery attempt per episode at ~12%,
        against the instant test's ~94%. And the env's delivery is not the
        room's: here one attempt is scored from wherever the pick ended, while
        a room run makes about four attempts per can and re-picks what it
        drops, so the two are different denominators rather than the same
        measurement at different fidelity.

        FIVE criteria have now been measured against cans delivered — instant
        pick, sustained hold, lift rate, handover ramp, and this full carry —
        and none of them ranks pick legs the way the mission does. The
        conclusion is not that a sixth is needed. It is that **the pick leg is
        not where the cans are**: the carry loses 69% of what it is handed,
        and that is a property of the carry.
        """
        legs = ((moss.LIFT_POSE, LIFT_TEST_S), (STOW_HIGH, 2.0),
                (STOW_TURNED, 4.0), (STOW_INSIDE, 3.0))
        cur = np.array([self.arm_cmd[j] for j in moss.ARM_JOINTS], float)
        for pose, secs in legs:
            goal = np.asarray(pose, float)
            n = max(1, int(secs / moss.GRASP_PHYSICS_DT))
            for k in range(n):
                f = (k + 1) / n
                self.arm_cmd.update(
                    zip(moss.ARM_JOINTS, cur + f * (goal - cur)))
                self.driver.set_arm(self.arm_cmd)
                self.driver.step(self.data)
                mujoco.mj_step(self.model, self.data)
            cur = goal.copy()
        self.arm_cmd[moss.GRIPPER_JOINT] = moss.MISSION_OPEN_M
        self.driver.set_arm(self.arm_cmd)
        for _ in range(int(1.5 / moss.GRASP_PHYSICS_DT)):
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)
        cb = self._can_base()
        return bool(moss.BIN_INTERIOR_X[0] < cb[0] < moss.BIN_INTERIOR_X[1]
                    and moss.BIN_INTERIOR_Y[0] < cb[1] < moss.BIN_INTERIOR_Y[1]
                    and moss.BIN_FLOOR_Z < cb[2] < moss.BIN_RIM_Z)

    def _lift_handover(self) -> bool:
        """Ramp to `moss.LIFT_POSE` and say whether the can survived it.

        This is the mission's `lift` state, run inside the env so the grip is
        scored on the thing that actually decides delivery. Ramped, not
        commanded outright: a step input to a position servo shears the can
        out of the jaws, which is why the brain ramps too.
        """
        start = np.array([self.arm_cmd[j] for j in moss.ARM_JOINTS], float)
        goal = np.asarray(moss.LIFT_POSE, float)
        n = max(1, int(LIFT_TEST_S / moss.GRASP_PHYSICS_DT))
        for k in range(n):
            f = (k + 1) / n
            self.arm_cmd.update(
                zip(moss.ARM_JOINTS, start + f * (goal - start)))
            self.driver.set_arm(self.arm_cmd)
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)
        return bool(self._held())

    def _held(self) -> bool:
        """Is the can actually being CARRIED by the jaws?

        Three conditions, and the third is the one that took three training
        runs to arrive at:

        1. the can is at the jaws (within 55 mm of the `tcp` site),
        2. the gripper servo has STALLED short of its command — a position
           servo reaches its target on air and stops short on an object,
        3. **and the can is off the floor.**

        Without (3) the test is satisfied by CRUSHING. MEASURED on the second
        trained policy: it drove the gripper command to 0.0 m — an 8 mm jaw
        on a 66 mm can, 58 mm of interference — and collected the per-step
        hold bonus on 199 of 200 steps while squeezing a can that never left
        the ground (12/12 "grips", 0/12 lifts). Stall alone cannot tell a
        grip from a crush, because both stall. Weight on the jaws can.

        This is also the honest definition for the robot: a can that is still
        resting on the floor is not held, whatever the servo reads.
        """
        can = self.data.xpos[self.can_body]
        tcp = self.data.site_xpos[self.tcp_site]
        if float(np.linalg.norm(can - tcp)) > 0.055:
            return False
        cmd = self.arm_cmd[moss.GRIPPER_JOINT]
        got = float(self.data.qpos[
            self.model.joint(moss.GRIPPER_JOINT).qposadr[0]])
        if (got - cmd) <= GRIP_STALL_M:
            return False
        return float(can[2]) > self._start_z + AIRBORNE_M

    # -------------------------------------------------------------- gym API

    def _maybe_new_prop(self) -> None:
        """A FRESH SHAPE for this episode, when asked for.

        A METHOD rather than three lines inside `reset`, because the stow env
        overrides `reset` and so never ran them: `run.json` recorded
        `prop_variety: True` while every stow episode spawned the same can.
        A flag that is recorded and then ignored is worse than one that is
        missing — the provenance says the run tested something it did not.

        23 ms of recompile against a 355 ms episode (6.6%) is the price of
        training on litter instead of on one particular can.
        """
        self._hs = None
        if self._hstates is not None and (
                self.handover_frac >= 1.0
                or self.rng.random() < self.handover_frac):
            self._hs = self._hstates[int(self.rng.integers(len(self._hstates)))]
            self.prop = handover_prop(self._hs)
            self._bind_model(recurring=True)
            return
        if self.prop_variety:
            self.prop = sample_prop(self.rng, self.litter, self.ball_frac)
            self._bind_model()

    def _bind_model(self, recurring: bool = False) -> None:
        """Compile this episode's scene and cache everything hanging off
        the model. Called again when the PROP changes: a new shape is a new
        model, and every id cached here belongs to that model.

        `recurring`: the prop comes from a small fixed set (the handover
        states hold 9 distinct props across 771 states), so its compiled
        scene is kept and reused — the same bytes a fresh compile gives, as
        nothing here writes to a model, for ~10 ms less per such reset. A
        drawn prop never recurs and is never kept. The cached model is
        SHARED with every other instance that asks for the same scene; see
        `_RECURRING_SCENES` and `shares_compiled_model` for why that is safe
        and what would break it.
        """
        clutter = getattr(self, "_clutter", ())
        poses = getattr(self, "_clutter_poses", None)
        if recurring and not clutter and poses is None:
            key = (self.rung, self.prop, _scene_inputs())
            model = _RECURRING_SCENES.get(key)
            if model is None:
                # Under the lock, and looked up AGAIN inside it: two spares
                # resetting at once would otherwise both compile and the
                # second would evict the first from the cache.
                with _RECURRING_LOCK:
                    model = _RECURRING_SCENES.get(key)
                    if model is None:
                        model = _RECURRING_SCENES[key] = scene_spec(
                            self.rung, self.prop).compile()
            self.model = model
        else:
            self.model = scene_spec(self.rung, self.prop, clutter,
                                    poses).compile()
        #: Arm and gripper geoms — the ones that must not be dragged along
        #: the floor. Resolved from the model rather than guessed by name at
        #: every step.
        self._arm_geoms = set()
        for _g in range(self.model.ngeom):
            _n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, _g)
            if _n and any(_k in _n for _k in (
                    "pad", "finger", "jaw", "gripper", "palm", "wrist",
                    "elbow", "shoulder", "arm")):
                self._arm_geoms.add(_n)
        #: arm joint -> its actuator index, for the envelope charge.
        self._arm_act = {}
        for _i in range(self.model.nu):
            _n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, _i)
            if _n in moss.JOINT_NAMES[:5]:
                self._arm_act[_n] = _i
        self.data = mujoco.MjData(self.model)
        self.driver = MossDriver(self.model, "")
        m = self.model
        self.can_body = m.body("can").id
        self.tcp_site = m.site("tcp").id
        self.can_qadr = int(m.joint("can_free").qposadr[0])
        self.can_dadr = int(m.joint("can_free").dofadr[0])
        #: The WRIST camera's mount, resolved once. -1 on a model built
        #: without it, which simply means no attitude is ever published.
        self._arm_cam_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, moss.ARM_CAMERA_BODY)
        self.joint_q = np.array([int(m.joint(j).qposadr[0])
                                 for j in moss.JOINT_NAMES])
        self.joint_v = np.array([int(m.joint(j).dofadr[0])
                                 for j in moss.JOINT_NAMES])
        self.default_pose = np.array(moss.DEFAULT_POSE, np.float64)
        #: The arm command this env holds and nudges — it starts DEPLOYED,
        #: because that is where the brain hands over.
        self.arm_cmd = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
        self.arm_cmd = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
        self.arm_cmd[moss.GRIPPER_JOINT] = 0.041

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._maybe_new_prop()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.driver.spawn(self.data, 0.0, 0.0, 0.0)
        self.arm_cmd = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
        self.arm_cmd[moss.GRIPPER_JOINT] = 0.041
        # ...with the WRIST somewhere in its useful arc, so rolling it is a
        # state the policy has actually visited. COMMANDED, not merely posed,
        # or the controller drives it straight back to the deployed angle.
        if self.wrist_start_rand > 0.0:
            jr = self.model.joint("wrist_roll").range
            self.arm_cmd["wrist_roll"] = float(np.clip(
                self.arm_cmd["wrist_roll"]
                + self.rng.uniform(-self.wrist_start_rand,
                                   self.wrist_start_rand),
                float(jr[0]) + 1e-3, float(jr[1]) - 1e-3))
        self.driver.set_arm(self.arm_cmd)
        # Let the arm reach the deployed pose before the can is placed: this
        # env starts where the brain hands over, not where the arm folds.
        # MEASURED: the arm is within 9 mrad of the deployed pose by 2 s and
        # nowhere near it at 0.8 s — it travels ~1 rad at the shoulder.
        for _ in range(int(DEPLOY_SETTLE_S / moss.GRASP_PHYSICS_DT)):
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)
        lo, hi, ymax = RUNG_BOX[self.rung]
        if self.pick_box is not None:
            lo, hi, ymax = self.pick_box
        if self.wrist_drill:
            # between the pads, where closing is the only thing left to do
            lo, hi, ymax = 0.25, 0.27, 0.012
        x = float(self.rng.uniform(lo, hi))
        y = float(self.rng.uniform(-ymax, ymax))
        z = self.prop.half_height
        if self.domain_rand:
            z += float(self.rng.uniform(-0.002, 0.002))
        # The rung box is in the BASE frame and qpos is in the WORLD's. The
        # two are not the same by the time the arm has deployed: putting an
        # arm out shoves an undriven planar base, and station keeping latches
        # wherever it stopped. Written the other way round, rung 0's "can
        # between the pads" spawned it 68 mm off to one side, so the drill
        # rung was not a drill and nothing would ever have learned to close.
        bx, by, byaw = self.driver.pose(self.data)
        c, sn = math.cos(byaw), math.sin(byaw)
        # **UPRIGHT OR LYING DOWN**, and this is not decoration. MEASURED by
        # tracing the scripted loop: the can is knocked over during the
        # approach almost every time, so what the brain hands the policy is a
        # can on its SIDE — and an env that only ever spawned an upright one
        # trained a policy that drove the can to 12 mm, closed to 29 mm, then
        # oscillated open/shut without ever committing, because a cylinder
        # lying across the jaws is a different grasp from one standing in
        # them. 4/12 in its own env, 0/3 in the room.
        yaw = float(self.rng.uniform(-math.pi, math.pi))
        roll = 0.0
        topple = 0.0
        leaning = False
        if self.wrist_drill:
            roll = math.pi / 2               # on its side, always
        elif self.can_pose_rand:
            r = self.rng.random()
            if r < CAN_LEANING_P:
                # AGAINST A KERB: the can rests on it at an angle, the way a
                # can that rolled to the skirting does. The kerb goes just
                # BEYOND the can from the robot, so the arm meets both.
                leaning = True
                roll = math.radians(float(self.rng.uniform(35.0, 60.0)))
                z = (self.prop.half_height * math.cos(roll)
                     + self.prop.radius * math.sin(roll))
            elif r < CAN_LEANING_P + CAN_LYING_P:
                roll = math.pi / 2                   # resting on its side
                z = self.prop.radius
            elif r < CAN_LEANING_P + CAN_LYING_P + CAN_TILTED_P:
                # MID-TOPPLE: neither resting pose, and already going over.
                roll = math.radians(float(self.rng.uniform(*CAN_TILT_RANGE_DEG)))
                # Lift the centre so the rim rather than the base carries it.
                z = (self.prop.half_height * math.cos(roll)
                     + self.prop.radius * math.sin(roll))
                topple = float(self.rng.uniform(*CAN_TOPPLE_RATE))
        # roll about x by `roll`, then yaw about z — quaternion product.
        if self._handovers is not None:
            row = self._handovers[int(self.rng.integers(len(self._handovers)))]
            x = float(row[0] + self.rng.normal(0.0, HANDOVER_XY_SD))
            y = float(row[1] + self.rng.normal(0.0, HANDOVER_XY_SD))
            roll = math.acos(float(np.clip(row[3], 0.0, 1.0)))
            leaning, topple = False, 0.0
            z = (self.prop.radius if self.prop.shape == "sphere" else
                 self.prop.half_height * math.cos(roll)
                 + self.prop.radius * math.sin(roll))
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        quat = [cy * cr, cy * sr, sy * sr, sy * cr]
        self.data.qpos[self.can_qadr:self.can_qadr + 3] = [
            bx + x * c - y * sn, by + x * sn + y * c, z]
        self.data.qpos[self.can_qadr + 3:self.can_qadr + 7] = quat
        # Park or place the kerb. Parked far away on every other episode, so
        # it is present in the model and absent from the scene.
        if self.model.nmocap:
            if leaning:
                # WHERE IS IT ACTUALLY FALLING? Taken from the can's own
                # orientation rather than derived by hand: a roll about x
                # followed by a yaw about z does not tip the can along `yaw`,
                # and placing the kerb there let every leaning can fall flat.
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, np.asarray(quat, float))
                axis = R.reshape(3, 3)[:, 2]          # the can's own +z
                hx, hy = float(axis[0]), float(axis[1])
                n = math.hypot(hx, hy) or 1.0
                hx, hy = hx / n, hy / n               # unit, in the base frame
                # SIZE[0] — the kerb is yawed so its +x faces the can, so
                # the half-extent that matters is the short one. Using the
                # long one put the kerb 181 mm away and every can fell flat
                # before reaching it.
                d = self.prop.radius + LEAN_KERB_SIZE[0] + LEAN_KERB_GAP
                kx, ky = x + d * hx, y + d * hy
                self.data.mocap_pos[0] = [
                    bx + kx * c - ky * sn, by + kx * sn + ky * c,
                    LEAN_KERB_SIZE[2]]
                # Face the kerb's long side at the can.
                kyaw = math.atan2(hy, hx) + byaw
                self.data.mocap_quat[0] = [math.cos(kyaw / 2), 0.0, 0.0,
                                           math.sin(kyaw / 2)]
            else:
                self.data.mocap_pos[0] = [6.0, 6.0, LEAN_KERB_SIZE[2]]
        self.data.qvel[self.can_dadr:self.can_dadr + 6] = 0.0
        if leaning:
            # LET IT SETTLE AGAINST THE KERB rather than posing it there.
            # Placing a can at a computed lean angle and expecting it to stay
            # was three attempts of analytic geometry and it fell flat every
            # time; what matters for training is not the angle but that the
            # can is ARRIVED AT an obstacle it must not be shoved into. So
            # the physics decides the pose and the kerb decides the problem.
            mujoco.mj_forward(self.model, self.data)
            for _ in range(int(0.4 / self.model.opt.timestep)):
                mujoco.mj_step(self.model, self.data)
            self.data.qvel[self.can_dadr:self.can_dadr + 6] = 0.0
        if topple:
            # Falling about the axis it is already leaning over, in the
            # can's own yaw frame.
            self.data.qvel[self.can_dadr + 3] = topple * math.cos(yaw)
            self.data.qvel[self.can_dadr + 4] = topple * math.sin(yaw)
        mujoco.mj_forward(self.model, self.data)
        if self._hs is not None:
            self._pose_handover_state(self._hs)
        #: The lab's trainee preview reads this off the env by name, as it does
        #: on every other body's (`walk_env`, `mars_env`). Called `steps` for one
        #: run, which killed the lab's whole duck loop the moment a MOSS
        #: trainee joined the roster.
        self.step_count = 0
        self._start_z = float(self.data.xpos[self.can_body][2])
        self._can_spawn_w = np.array(self.data.xpos[self.can_body][:2], float)
        self._prev_align = None
        self._prev_upright = None
        self._prev_gap = self._reward_gap()
        self._prev_lift = 0.0
        self._hold_streak = 0
        #: The wrist camera's last attitude report, and when it landed.
        self._att: tuple[float, float, float] | None = None
        self._att_t = -1e9
        # The FRONT camera's attitude belief, kept apart from the wrist's so
        # the fresher and better source can win per tick rather than the two
        # overwriting each other.
        self._fatt = None
        self._fatt_t = -1e9
        self._pending_fatt = []
        self._next_arm_t = 0.0
        self._pending_att: list = []
        self._prev_can_w = None
        self._closed_once = False
        self._fix = None
        self._seen_w = None
        self._vel = np.zeros(2)
        self._fix_t = -1e9
        self._pending = []
        self._next_det_t = float(self.data.time)
        self._grip = None
        self._grip_t = -1e9
        if self._hs is not None:
            # The yard's brain enters `creep` holding a fresh fix of its
            # target; a replay that started blind did not reproduce the yard.
            self._fix = np.array(self.data.xpos[self.can_body][:2], float)
            self._seen_w = self._fix.copy()
            self._fix_t = float(self.data.time)
        self._sense()
        self.last_action[:] = 0.0
        self.prev_action[:] = 0.0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self.prev_action = self.last_action.copy()
        self.last_action = a
        # arm + jaw: NUDGES on the held command (see ARM_DELTA_RAD)
        for i, j in enumerate(moss.ARM_JOINTS):
            lo, hi = self.model.joint(j).range
            self.arm_cmd[j] = float(np.clip(
                self.arm_cmd[j] + a[i] * ARM_DELTA_RAD, lo, hi))
        # ONE gripper command for both jaws (`moss.ACT_GRIPPER`). This drove
        # the two fingers separately for one run, which was faithful to his
        # exported MJCF and wrong about his robot: the physical gripper has a
        # single servo, and a policy given two would learn a scissor the
        # hardware cannot perform. The follower is tied in the model now, so
        # writing the leader is the whole command.
        lo, hi = self.model.joint(moss.GRIPPER_JOINT).range
        self.arm_cmd[moss.GRIPPER_JOINT] = float(np.clip(
            self.arm_cmd[moss.GRIPPER_JOINT]
            + a[moss.ACT_GRIPPER.start] * JAW_DELTA_M, lo, hi))
        self.driver.set_arm(self.arm_cmd)
        # base: his own envelope decides what the twist becomes
        vx, wz = clamp_cmd(float(a[moss.ACT_BASE.start]) * MAX_VX,
                           float(a[moss.ACT_BASE.start + 1]) * MAX_WZ)
        # ...and inside the arm's reach, only part of it survives, so the
        # policy has to learn that the last stretch is the ARM's. At the
        # default 1.0 this is a no-op and the env is exactly what every
        # shipped leg trained in. See `PICK_BASE_REACH_SCALE`.
        if PICK_BASE_REACH_SCALE != 1.0 and self._gap() <= PICK_BASE_REACH_M:
            vx *= PICK_BASE_REACH_SCALE
            wz *= PICK_BASE_REACH_SCALE
        # ...or the base is not this policy's to command at all.
        if self.base_lock:
            vx = wz = 0.0

        self._shift_can()
        for _ in range(DECIMATION):
            self.driver.set_cmd(vx, wz, self.data.time)
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self._predict()
        self._sense()
        self._sense_arm()

        gap = self._reward_gap()
        cb = self._can_base()
        held = self._held()
        if not self._closed_once and self._closed_on_can():
            self._closed_once = True
            rew_close = W_CLOSE_BONUS
        else:
            rew_close = 0.0
        lift = max(0.0, float(self.data.xpos[self.can_body][2]) - self._start_z)
        rew = W_PROGRESS * self.gap_scale * (self._prev_gap - gap) + rew_close
        if held:
            rew += W_HELD
            rew += W_LIFT * max(0.0, lift - self._prev_lift)
        self._prev_lift = lift
        # WHAT THE SERVOS WERE ASKED FOR. Charged per arm joint per step for
        # sitting against the torque clamp, and for turning faster than the
        # servo is rated to — neither of which the clamp itself makes costly.
        if self.torque_sat > 0.0 or self.overspeed > 0.0:
            for _j, _ai in self._arm_act.items():
                if self.torque_sat > 0.0:
                    _lim = float(self.model.actuator_forcerange[_ai][1])
                    if _lim > 0 and abs(float(
                            self.data.actuator_force[_ai])) >= 0.995 * _lim:
                        rew -= self.torque_sat
                if self.overspeed > 0.0:
                    _v = abs(float(self.data.qvel[
                        self.model.joint(_j).dofadr[0]]))
                    if _v > ARM_RATED_RAD_S:
                        rew -= self.overspeed * (_v - ARM_RATED_RAD_S)
        # ON THE FLOOR, and worse under power.
        if self.arm_floor > 0.0:
            _driving = abs(float(a[moss.ACT_BASE.start])) > ARM_FLOOR_DRIVE_MPS
            _scrape = False
            for _c in range(self.data.ncon):
                _con = self.data.contact[_c]
                _n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                        _con.geom1) or ""
                _n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                        _con.geom2) or ""
                _pair = {_n1, _n2}
                if not (_pair & {"floor", "ground"}):
                    continue
                if _pair & self._arm_geoms:
                    _scrape = True
                    break
            if _scrape:
                rew -= self.arm_floor * (ARM_FLOOR_DRAG_MULT if _driving else 1.0)
        # ...AND COMING IN LOW. Below the object's top while still outside
        # grasp radius is an approach that arrives at the can's side.
        if self.low_approach > 0.0 and not held:
            _tcp = np.asarray(self.data.site_xpos[self.tcp_site], float)
            _can = np.asarray(self.data.xpos[self.can_body], float)
            _up = abs(self._true_attitude()[2])
            _top = float(_can[2]) + (self.prop.half_height * _up
                                     + self.prop.radius * (1.0 - _up))
            if (float(np.hypot(_tcp[0] - _can[0], _tcp[1] - _can[1]))
                    > LOW_APPROACH_RADIUS_M and float(_tcp[2]) < _top):
                rew -= self.low_approach
        jerk = np.abs(a - self.prev_action)
        if self.wrist_free:
            # The one motion we are trying to BUY should not be taxed.
            # Retargeted, not deleted: every other joint still pays.
            jerk[ACT_WRIST_ROLL] = 0.0
        rew += W_ACTION_RATE * float(jerk.sum())
        # TURNING THE JAW TOWARD SQUARE, paid as PROGRESS so that holding an
        # alignment earns nothing and only improving it does — a flat bonus
        # here is the parking trap `W_HELD` already fell into.
        if self.jaw_align > 0.0 or self.align_hold > 0.0:
            _align, _lying = self._jaw_alignment()
            if self.jaw_align > 0.0 and self._prev_align is not None:
                rew += self.jaw_align * _lying * (_align - self._prev_align)
            # ...and the DENSE half: paid every step the jaw is square to a
            # lying can and within reach of it, so turning the wrist is worth
            # something on its own rather than only as a step toward a grasp
            # the policy can reach by shoving instead.
            if self.align_hold > 0.0 and gap < ALIGN_HOLD_RANGE_M:
                rew += self.align_hold * _lying * (2.0 * _align - 1.0)
            self._prev_align = _align
        # ...unless the base is LOCKED, in which case these commands do
        # nothing and charging for them is a tax on a no-op: MEASURED at
        # -10.41 per episode, the single largest term in the reward, for an
        # action with no effect on the world. It taught the policy nothing and
        # ate 60% of the success bonus.
        if not self.base_lock:
            rew += W_BASE_EFFORT * (abs(float(a[moss.ACT_BASE.start]))
                                    + abs(float(a[moss.ACT_BASE.start + 1])))
        # ...and yaw again, on its own. The arm can cover the bearing; the
        # base turning to do the same work is the expensive way and the one
        # that hits things.
        if not self.base_lock:
            rew += W_BASE_YAW * abs(float(a[moss.ACT_BASE.start + 1]))
        self._prev_gap = gap

        # WHAT DID IT COST THE CAN TO GET HERE? Charged per metre the can
        # moves while the jaws do not have it, so a clean approach is worth
        # more than a barge. `_prev_can_w` is the world position, because a
        # base-frame delta would count the ROBOT's motion as the can's.
        # ...AND WHAT IT HIT ON THE WAY. Charged per step of contact with the
        # chassis while unheld, which is the failure a displacement penalty
        # cannot separate from a legitimate nudge.
        if not held:
            body_hit = edge_hit = False
            for c in range(self.data.ncon):
                con = self.data.contact[c]
                n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                       con.geom1) or ""
                n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                       con.geom2) or ""
                if "can_geom" not in (n1, n2):
                    continue
                other = n2 if n1 == "can_geom" else n1
                if other in CAN_BODY_GEOMS:
                    body_hit = True
                elif other in PAD_GEOMS and not edge_hit:
                    # WHICH FACE TOOK THE LOAD. The contact normal, rotated
                    # into the pad's own axes: x is the pad's leading and
                    # trailing edge, y the face that grips, z its top and
                    # bottom. Closest axis wins, which is all the question is.
                    nrm = np.asarray(con.frame[:3], float)
                    R = self.data.geom_xmat[self.model.geom(other).id]
                    local = R.reshape(3, 3).T @ nrm
                    if int(np.argmax(np.abs(local))) == 0:
                        edge_hit = True
            if body_hit:
                rew += W_CAN_BODY_HIT     # once per step, not per point
            if edge_hit:
                rew += W_PAD_EDGE_HIT

        can_w = np.array(self.data.xpos[self.can_body])
        if self._prev_can_w is not None and not held:
            shove = float(np.linalg.norm(can_w - self._prev_can_w))
            if shove > CAN_DISTURB_DEADBAND_M:
                rew += W_CAN_DISTURB * shove
        self._prev_can_w = can_w.copy()
        # ...AND FOR TIPPING IT. Uprightness lost while the jaws do not have
        # it, so a can caught by its rim costs something and a can seated
        # against the palm — which is a grasp, not a barge — does not.
        if self.can_topple > 0.0:
            up_now = abs(self._true_attitude()[2])
            if self._prev_upright is not None and not held:
                lost = max(0.0, self._prev_upright - up_now)
                rew -= self.can_topple * lost
            self._prev_upright = up_now

        knocked = not (BAND_X[0] < cb[0] < BAND_X[1] and abs(cb[1]) < BAND_Y)
        if knocked:
            rew += KNOCKED_PENALTY
        # A GRIP THAT SURVIVES, not a can that briefly left the floor. The
        # streak resets the moment either goes away, so a policy cannot bank
        # it by touching the line and letting go.
        if held and lift > SUCCESS_LIFT_M:
            self._hold_streak += 1
        else:
            self._hold_streak = 0
        # THE INSTANT TEST, KEPT — and not because it is good. Two stricter
        # criteria were built and measured against the mission, and both
        # FAILED the check that matters: does it rank policies the way cans
        # delivered ranks them? See `PICK_HOLD_STEPS` and `_lift_handover`,
        # which stay as the record of that rather than as live code.
        picked = held and lift > SUCCESS_LIFT_M
        if picked and self.deep_grip_m > 0.0:
            _d = float(np.linalg.norm(self.data.xpos[self.can_body]
                                      - self.data.site_xpos[self.tcp_site]))
            picked = _d <= self.deep_grip_m
        if (picked and self.sphere_centre_m > 0.0
                and self.prop.shape == "sphere"):
            picked = self._sphere_off_centre() <= self.sphere_centre_m
        if picked:
            rew += SUCCESS_BONUS
        # SHOVED IT INSTEAD OF REACHING FOR IT: over before the grip, so the
        # barge cannot be a route to the bonus. Only before the first close —
        # a can being carried is supposed to move.
        shoved_out = False
        if (SHOVE_LIMIT_M > 0.0 and not self._closed_once
                and self._can_spawn_w is not None):
            moved = float(np.linalg.norm(
                np.array(self.data.xpos[self.can_body][:2], float)
                - self._can_spawn_w))
            if moved > SHOVE_LIMIT_M:
                shoved_out = True
                rew += W_SHOVE_ABORT
        terminated = bool(knocked or picked or shoved_out)
        truncated = self.step_count >= self.max_steps
        return (self._obs(), float(rew), terminated, truncated,
                {"gap": gap, "held": held, "lift": lift, "picked": picked,
                 "success": picked, "shoved_out": shoved_out})

    def _predict(self) -> None:
        """Advance the belief along its estimated velocity, and damp it.

        This is the half that makes a stale fix USEFUL rather than merely
        old: a can that was shoved keeps moving after the last frame, and a
        belief pinned where it was last seen is wrong by exactly the distance
        it travelled. Bounded by `MAX_PREDICT_M` from the last real sighting,
        because a tracker that extrapolates without limit invents an object.
        """
        if self._fix is None:
            return
        self._vel *= VEL_DAMP
        if float(np.linalg.norm(self._vel)) < VEL_FLOOR:
            self._vel[:] = 0.0
        self._fix = self._fix + self._vel * CTRL_DT
        if self._seen_w is not None:
            off = self._fix - self._seen_w
            d = float(np.linalg.norm(off))
            if d > MAX_PREDICT_M:
                self._fix = self._seen_w + off * (MAX_PREDICT_M / d)

    def _to_world(self, base_xy) -> np.ndarray:
        x, y, yaw = self.driver.pose(self.data)
        c, sn = math.cos(yaw), math.sin(yaw)
        return np.array([x + base_xy[0] * c - base_xy[1] * sn,
                         y + base_xy[0] * sn + base_xy[1] * c])

    def _to_base(self, world_xy) -> np.ndarray:
        x, y, yaw = self.driver.pose(self.data)
        dx, dy = world_xy[0] - x, world_xy[1] - y
        c, sn = math.cos(-yaw), math.sin(-yaw)
        return np.array([dx * c - dy * sn, dx * sn + dy * c])

    def _sense(self) -> None:
        """One tick of the camera: sample at `DET_RATE_HZ`, deliver late."""
        t = float(self.data.time)
        if t + 1e-9 >= self._next_det_t:
            self._next_det_t = t + 1.0 / DET_RATE_HZ
            cb = self._can_base()
            rng = float(math.hypot(cb[0], cb[1]))
            bearing = math.atan2(cb[1], cb[0])
            in_frame = (rng > DET_MIN_RANGE_M
                        and abs(bearing) < DET_MAX_BEARING
                        and self.rng.random() > DET_DROPOUT)
            if in_frame:
                b = bearing + self.rng.normal(0.0, DET_BEARING_NOISE)
                r = rng + self.rng.normal(0.0, DET_RANGE_NOISE)
                self._pending.append(
                    (t + DET_LATENCY_S,
                     np.array([r * math.cos(b), r * math.sin(b)])))
                fatt = self._front_attitude(rng, bearing)
                if fatt is not None:
                    self._pending_fatt.append((t + DET_LATENCY_S, fatt))
        while self._pending and self._pending[0][0] <= t + 1e-9:
            _at, fix_base = self._pending.pop(0)
            fix_w = self._to_world(fix_base)
            if self._seen_w is not None and t > self._fix_t:
                raw = (fix_w - self._seen_w) / max(t - self._fix_t, 1e-3)
                self._vel = VEL_EMA * raw + (1.0 - VEL_EMA) * self._vel
            self._fix, self._seen_w, self._fix_t = fix_w, fix_w, t
        while self._pending_fatt and self._pending_fatt[0][0] <= t + 1e-9:
            _at, fatt = self._pending_fatt.pop(0)
            self._fatt, self._fatt_t = fatt, t

    def _front_attitude(self, rng: float, bearing: float):
        """The can's axis as the FRONT camera can read it, or None.

        Same triple as `_true_attitude`, degraded the way a silhouette
        degrades: noise that grows with range, an extra dropout, and nothing
        at all when the can points along the view ray, because then the image
        is a circle and the axis is not in it.
        """
        c, sn, upright = self._true_attitude()
        if self.rng.random() < FRONT_AXIS_DROPOUT:
            return None
        if upright < FRONT_AXIS_ENDON_UPRIGHT:
            # The axis, modulo pi, against the direction it is being viewed
            # from — both in the base frame, so the chassis yaw cancels.
            t = 0.5 * math.atan2(sn, c)
            if abs(math.cos(t - bearing)) > FRONT_AXIS_ENDON_COS:
                return None
        sigma = FRONT_AXIS_NOISE_BASE + FRONT_AXIS_NOISE_PER_M * max(rng, 0.0)
        ang = math.atan2(sn, c) + 2.0 * self.rng.normal(0.0, sigma)
        return (math.cos(ang), math.sin(ang),
                float(np.clip(upright + self.rng.normal(
                    0.0, FRONT_UPRIGHT_NOISE), 0.0, 1.0)))

    def _sense_arm(self) -> None:
        """One tick of the WRIST camera, which reports ATTITUDE.

        Modelled as a real aperture rather than as ground truth: the can has
        to be inside this camera's field and range, which is checked against
        the mount's ACTUAL pose in the model, and the axis arrives with
        noise, dropout and latency like any other measurement. Anything else
        would train the policy on a number the robot cannot produce — which
        is the mistake that cost the stow leg 9/12 in its env against 0/9 in
        a room.
        """
        t = float(self.data.time)
        if t + 1e-9 < self._next_arm_t:
            return
        self._next_arm_t = t + 1.0 / ARM_DET_RATE_HZ
        cam = self._arm_cam_id
        if cam < 0:
            return
        p = np.asarray(self.data.xpos[cam], float)
        R = np.asarray(self.data.xmat[cam], float).reshape(3, 3)
        v = R.T @ (np.asarray(self.data.xpos[self.can_body], float) - p)
        rng = float(np.linalg.norm(v))
        if rng > moss.ARM_CAMERA_MAX_RANGE_M or v[0] <= 1e-6:
            return
        az = abs(math.degrees(math.atan2(v[1], v[0])))
        el = abs(math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))))
        if (az > moss.ARM_CAMERA_HFOV_DEG / 2
                or el > moss.ARM_CAMERA_VFOV_DEG / 2):
            return
        if self.rng.random() < ARM_DET_DROPOUT:
            return
        # KEEP THE RANGE. It was computed above only to decide visibility.
        self._arm_rng = max(0.0, rng + self.rng.normal(0.0, ARM_DET_RANGE_NOISE))
        self._arm_rng_t = t
        if ((getattr(self, "publish_grip", False)
             or getattr(self, "grip_xyz", False)) and self._arm_sees()):
            # A DEPTH camera's fix is a 3-D point, so it says where the object
            # sits IN THE JAWS — the tool frame — which the range cannot: over
            # 122 yard lifts it read 120-124 mm (median) whether the grip was
            # deep or at the fingertips (corr -0.1). Same noise per axis.
            Rt = np.asarray(self.data.site_xmat[self.tcp_site], float).reshape(3, 3)
            rel = Rt.T @ (np.asarray(self.data.xpos[self.can_body], float)
                          - np.asarray(self.data.site_xpos[self.tcp_site], float))
            self._grip = rel + self.rng.normal(0.0, ARM_DET_RANGE_NOISE, 3)
            self._grip_t = t
        c, sn, upright = self._true_attitude()
        # Noise on the ANGLE, not on cos and sin separately — perturbing them
        # independently makes a vector that is not a unit vector, which is
        # not a thing a detector can report.
        ang = math.atan2(sn, c) + 2.0 * self.rng.normal(0.0, ARM_DET_AXIS_NOISE)
        self._pending_att.append(
            (t + ARM_DET_LATENCY_S,
             (math.cos(ang), math.sin(ang),
              float(np.clip(upright + self.rng.normal(0.0, 0.05), 0.0, 1.0)))))
        while self._pending_att and self._pending_att[0][0] <= t + 1e-9:
            _at, att = self._pending_att.pop(0)
            self._att, self._att_t = att, t

    def _arm_sees(self) -> bool:
        """Can the wrist lens see the object past the robot? Rays to its
        centre and four points on its rim; 3 of 5 must reach it first — the
        rule the mount sweep scored frames by (`moss.ARM_CAMERA_POS`)."""
        m, d, bid = self.model, self.data, self.can_body
        cam = np.asarray(d.xpos[self._arm_cam_id], float)
        c = np.asarray(d.xpos[bid], float)
        v = c - cam
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return False
        v /= n
        a = np.cross(v, [0.0, 0.0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(v, [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(v, a)
        r = 0.7 * float(self.prop.radius)
        hits = 0
        gid = np.zeros(1, np.int32)
        for pt in (c, c + r * a, c - r * a, c + r * b, c - r * b):
            dist = mujoco.mj_ray(m, d, cam, pt - cam, None, 1,
                                 self._arm_cam_id, gid)
            if gid[0] < 0 or m.geom_bodyid[gid[0]] == bid or dist >= 0.999:
                hits += 1
        return hits >= 3

    def _grip_obs(self) -> tuple[float, float]:
        """`moss.OBS_GRIP`: (distance of the object from the tool point, 1)
        from a fresh depth fix, else (`GRIP_UNSEEN_M`, 0)."""
        if (self._grip is not None
                and (float(self.data.time) - self._grip_t) < ARM_STALE_S):
            return float(np.linalg.norm(self._grip)), 1.0
        return GRIP_UNSEEN_M, 0.0

    def _grip_xyz_obs(self) -> tuple[float, float, float, float]:
        """`moss.OBS_GRIP_XYZ`: the depth fix in the tool frame and 1 when
        fresh, else (0, 0, `GRIP_UNSEEN_M`, 0)."""
        if (self._grip is not None
                and (float(self.data.time) - self._grip_t) < ARM_STALE_S):
            g = self._grip
            return float(g[0]), float(g[1]), float(g[2]), 1.0
        return 0.0, 0.0, GRIP_UNSEEN_M, 0.0

    def _pose_handover_state(self, hs: dict) -> None:
        """Put the arm and the object where the yard had them at `creep`:
        joints and jaw from the state, the object in the base frame, all at
        rest — then let physics have it."""
        m, d = self.model, self.data
        for j, v in zip(list(moss.ARM_JOINTS) + [moss.GRIPPER_JOINT], hs["arm"]):
            jt = m.joint(j)
            d.qpos[jt.qposadr[0]] = float(v)
            d.qvel[jt.dofadr[0]] = 0.0
            self.arm_cmd[j] = float(v)
        bx, by, byaw = self.driver.pose(d)
        c, s_ = math.cos(byaw), math.sin(byaw)
        x, y, z = hs["pos_b"]
        d.qpos[self.can_qadr:self.can_qadr + 3] = [bx + x * c - y * s_,
                                                   by + x * s_ + y * c, z]
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([math.cos(byaw / 2), 0.0, 0.0,
                                        math.sin(byaw / 2)]),
                           np.asarray(hs["quat_b"], float))
        d.qpos[self.can_qadr + 3:self.can_qadr + 7] = q
        d.qvel[self.can_dadr:self.can_dadr + 6] = 0.0
        if m.nmocap:
            d.mocap_pos[0] = [6.0, 6.0, LEAN_KERB_SIZE[2]]
        self.driver.set_arm(self.arm_cmd)
        mujoco.mj_forward(m, d)

    def _shift_can(self) -> None:
        """Shove the can now and then, so a stale fix goes WRONG and not just
        old. What a track clipping a can does on the real floor."""
        if not self.domain_rand or self.rng.random() > CAN_SHIFT_PROB:
            return
        d = self.can_dadr
        self.data.qvel[d:d + 2] += self.rng.normal(
            0.0, CAN_SHIFT_IMPULSE, 2)

    def marker_payload(self):
        """THE CAN, for the lab stage — `[x, y, z, radius]` in this env's own
        world frame, the shape `lab.ts` already draws a trainee's ball in.

        Emitted through the same field for exactly that reason: the viewer
        has drawn a trainee's object since the dribble work, and a MOSS
        practising on an invisible can is a stage nobody can read. Nothing in
        the browser had to learn what a can is.
        """
        p = self.data.xpos[self.can_body]
        # A FIFTH number when the prop is a cylinder: its half-height. The
        # viewer draws a cylinder then and a sphere otherwise, so a can reads
        # as a can — you can see it topple, which a sphere never shows.
        out = [float(p[0]), float(p[1]), float(p[2]), self.prop.radius]
        if self.prop.shape == "box":
            # A BOX, drawn as a box: `[x, y, z, rx, ry, rz, qw, qx, qy, qz]`,
            # ten numbers, which is how the viewer tells it from the nine a
            # cylinder sends and the four a sphere does. Without this every
            # square scrap of litter rendered as a SPHERE and a lab training
            # on six shapes looked like one training on one.
            out = [float(p[0]), float(p[1]), float(p[2]),
                   float(self.prop.size[0]), float(self.prop.size[1]),
                   float(self.prop.size[2])]
            out.extend(float(v) for v in self.data.xquat[self.can_body])
            return out
        if self.prop.shape == "cylinder":
            out.append(self.prop.half_height)
            # ...AND ITS ORIENTATION, which the line above claimed to show and
            # did not. Sending position and size only, the viewer drew every
            # can bolt upright whatever the physics was doing — so a lab
            # spawning half its cans on their side looked like a lab that
            # never varied, and a watching human reasonably said so twice.
            # `[x, y, z, r, halfH, qw, qx, qy, qz]`; a reader that stops at
            # five is unchanged.
            q = self.data.xquat[self.can_body]
            out.extend(float(v) for v in q)
        return out

    def ghost_payload(self):
        """WHERE THE POLICY THINKS THE CAN IS — `[x, y, z, conf, seen]`, the
        `ballGhost` shape, or None before the first frame arrives.

        The truth is `marker_payload`; this is the fix the observation
        actually carries, so the two can be drawn side by side while it
        trains. The GAP is the thing: at 10 Hz with 50 ms of latency, an 8%
        dropout and a can that gets shoved, a trainee driving confidently at
        nothing is explained by nothing else.

        `conf` decays with the fix's age over `STALE_S`, and `seen` is 1 while
        it is fresh — which is the same bit the policy reads in
        `OBS_TARGET_SEEN`.
        """
        if self._fix is None:
            return None
        age = float(self.data.time - self._fix_t)
        conf = float(max(0.0, 1.0 - age / STALE_S))
        # The same five numbers the ball ghost sends, plus the prop's own
        # half-height when it is a cylinder — so the belief is drawn as the
        # SHAPE the policy is chasing and not as a generic blob.
        out = [float(self._fix[0]), float(self._fix[1]),
               self.prop.half_height, conf, 1.0 if age < STALE_S else 0.0]
        if self.prop.shape == "cylinder":
            out.append(self.prop.half_height)
        return out

    def ghost(self) -> dict:
        """Belief beside truth, for the lab to draw — the same pairing the
        ball work added for the dribble trainee. `seen` is what the policy's
        own `OBS_TARGET_SEEN` slot carries this step."""
        cb = self._can_base()
        fresh = self._fix is not None and (self.data.time - self._fix_t) < STALE_S
        belief = None if self._fix is None else self._to_base(self._fix)
        return {"truth": [float(cb[0]), float(cb[1]), float(cb[2])],
                "belief": None if belief is None else
                          [float(belief[0]), float(belief[1])],
                "vel": [float(self._vel[0]), float(self._vel[1])],
                "seen": bool(fresh),
                "age": None if self._fix is None else
                       float(self.data.time - self._fix_t)}

    def _obs(self) -> np.ndarray:
        """`moss.CONTRACT_ID`'s 32 floats, in the order the body declares."""
        o = np.zeros(moss.OBS_DIM, np.float32)
        q = self.data.qpos[self.joint_q] - self.default_pose
        v = self.data.qvel[self.joint_v]
        vf, _vl, wz = self.driver.velocity(self.data)
        # THE BELIEF, not the truth — see the module's "the belief" block.
        fresh = self._fix is not None and (self.data.time - self._fix_t) < STALE_S
        if self._fix is None:
            target = np.zeros(3)
        else:
            base = self._to_base(self._fix)
            target = np.array([base[0], base[1], self.prop.half_height])
        if self.obs_noise:
            q = q + self.rng.normal(0, 0.002, q.shape)
            v = v + self.rng.normal(0, 0.05, v.shape)
        o[moss.OBS_JOINT_POS] = q
        o[moss.OBS_JOINT_VEL] = v
        o[moss.OBS_LAST_ACTION] = self.last_action
        o[moss.OBS_BASE_TWIST] = (vf, wz)
        o[moss.OBS_TARGET_BASE] = target
        o[moss.OBS_TARGET_SEEN] = 1.0 if fresh else 0.0
        # WHICH WAY IT IS LYING. Without this the jaw cannot be aligned to a
        # cylinder's axis, because the axis is not in the observation:
        # measured on the shipped leg, the correlation between the can's axis
        # yaw and `wrist_roll` at the grasp is -0.030 and the wrist moves
        # 3.2 degrees of standard deviation in total. It was not choosing the
        # base over the arm; it had nothing to choose with.
        #
        # Behind the same freshness gate as the position, because a belief
        # about attitude goes stale exactly as fast — and a stale attitude
        # published as fresh is the train/deploy mismatch that cost the stow
        # leg 9/12 in its env against 0/9 in the room.
        # ATTITUDE comes from the WRIST camera and carries its OWN freshness:
        # the front camera can be holding the can while the wrist sees
        # nothing, and publishing a stale axis as current is worse than
        # publishing none.
        # ...and the FRONT camera answers when the wrist cannot see, which
        # is the whole approach. The wrist wins whenever it is fresh because
        # it is the better measurement; the front one keeps the slots alive
        # from standoff in to the hand-over, instead of dead until 20 cm.
        att = None
        if not self.publish_attitude:
            att = None
        elif (self._att is not None
                and (self.data.time - self._att_t) < ARM_STALE_S):
            att = self._att
        elif (self._fatt is not None
                and (self.data.time - self._fatt_t) < STALE_S):
            att = self._fatt
        if att is not None:
            o[moss.OBS_TARGET_AXIS] = att[:2]
            o[moss.OBS_TARGET_UPRIGHT] = att[2]
        # HOW HIGH IT STANDS: the top of the can above the floor, behind the
        # same freshness gate as the position, because it comes off the same
        # detection. Without it the policy cannot know how far to lift to
        # clear the object it is reaching over.
        # PROXIMITY, from the camera that can actually see at this range,
        # replacing a slot that measured a constant. Behind the wrist's own
        # freshness gate: a stale distance to something you are about to hit is
        # worse than none.
        if (self.publish_proximity and self._arm_rng is not None
                and (self.data.time - self._arm_rng_t) < ARM_STALE_S):
            o[moss.OBS_TARGET_BASE.start + 2] = float(self._arm_rng)
        elif self.publish_proximity:
            o[moss.OBS_TARGET_BASE.start + 2] = 0.0
        if self.publish_size and fresh:
            half = self.prop.half_height
            up = abs(self._true_attitude()[2])
            top = float(self.data.xpos[self.can_body][2]) + (
                half * up + self.prop.radius * (1.0 - up))
            o[moss.OBS_SPARE] = top
        if self.publish_grip:
            o[list(moss.OBS_GRIP)] = self._grip_obs()
        if self.grip_xyz:
            o[list(moss.OBS_GRIP_XYZ)] = self._grip_xyz_obs()
        return o

    def _true_attitude(self) -> tuple[float, float, float]:
        """`(cos 2t, sin 2t, |cos tilt|)` for the can, in the BASE frame.

        DOUBLED angle because a cylinder has no head: an axis at 10 degrees
        and one at 190 are the same grasp, and a single angle would ask the
        policy to learn that wrap as if it were a real discontinuity.

        The third number is how upright it stands — 1 on its base, 0 flat —
        because the jaw closes across the diameter from ABOVE for one and
        from the SIDE for the other, and the bearing alone cannot say which.

        Deployable from the camera this robot actually has: a lying can's
        axis projects onto the detector's bounding box as the box's long
        axis, and its uprightness as the box's aspect. See
        `docs/moss-policy-schema.md` for how the robot fills these.

        BODY Z, for every shape — and for the lying cigarette butt that is a
        SHORT axis, 90 degrees off its long one. Do not "fix" it to the long
        axis: MEASURED 2026-09-27 on 478dad, 40 butts, every one of the 16
        picks had the jaw opening ALONG the long axis (an end-to-end grip,
        which this axis steers towards), all 7 grips across the 8 mm width
        failed, and reporting the long axis took picks 16/40 -> 0/40 (cards
        36 -> 31). The exact top height it would also fix measured neutral.
        """
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(self.data.xquat[self.can_body], float))
        axis = R.reshape(3, 3)[:, 2]
        _x, _y, byaw = self.driver.pose(self.data)
        upright = abs(float(axis[2]))
        horiz = math.hypot(float(axis[0]), float(axis[1]))
        if horiz < 1e-6:
            return 0.0, 0.0, upright          # dead upright: no axis to point
        t = math.atan2(float(axis[1]), float(axis[0])) - byaw
        return math.cos(2.0 * t), math.sin(2.0 * t), upright


__all__ = ["CTRL_DT", "DECIMATION", "EPISODE_S", "MossPickEnv", "POCKET",
           "CAN_LYING_P", "CAN_SHIFT_PROB", "DEFAULT_PROP", "DET_RATE_HZ", "GRASP_PROPS",
           "AIRBORNE_M", "GRIP_STALL_M", "W_CLOSE_BONUS",
           "GraspProp", "ARRIVE_X", "MossApproachEnv", "RUNGS", "RUNG_BOX",
           "MAX_PREDICT_M", "STALE_S", "SUCCESS_BONUS", "TASKS",
           "VEL_DAMP", "VEL_EMA", "W_BASE_EFFORT", "scene_spec"]


# ============================================================ the APPROACH
#
# The second trained part of the loop, and it exists because of a
# measurement rather than a plan: the scripted approach topples the can
# almost every time it drives up to one, so what the pickup skill is handed
# is a can lying on its side — which is the state that costs the loop most of
# its attempts. A proportional bearing controller has no reason not to bump
# it; a policy paid for arriving WITHOUT moving it does.
#
# Same contract as the pickup (`moss.CONTRACT_ID`), so it exports through the
# same path and a code skill on the Jetson runs both the same way. What
# differs is where it starts (the can a metre out, the arm folded) and what
# it is paid for (arrive, and leave the can where it was).

#: Where the approach hands over: the can this far ahead, centred, at rest.
ARRIVE_X = moss.DEPLOY_STANDOFF_M
ARRIVE_TOL_M = 0.06
ARRIVE_BEARING = 0.15                 # rad
#: The approach's own spawn box, in the base frame: far enough that driving
#: is the task, wide enough that turning is part of it.
APPROACH_BOX = (0.80, 1.50, 0.70)     # x_lo, x_hi, |bearing| max (rad)
#: The ladder, in the SPAWN again: rung 0 is a short straight run, rung 1 is
#: far enough and off-bearing enough that the can leaves the camera's frame
#: during the turn and the policy has to drive on its own belief of where it
#: was. `APPROACH_BOX` is rung 0 and stays the default.
APPROACH_RUNG_BOX: dict[int, tuple[float, float, float]] = {
    0: APPROACH_BOX,
    1: (1.20, 2.40, 1.40),
}
#: How far PAST its own spawn box a can may drift before the episode gives
#: up on it. Derived rather than typed, because a loss band that does not
#: contain the spawn box silently kills most of a rung.
LOST_MARGIN_M = 0.8
#: Slack on top of the time the drive itself needs, for the correction at the
#: end and for a policy that does not drive flat out.
APPROACH_SLACK_S = 3.0


def approach_seconds(rung: int) -> float:
    """How long an episode at `rung` has to be for arriving to be POSSIBLE.

    Derived from the rung's own box, like the loss band, and for the same
    reason. Fixed at the env's 8 s default, rung 1 could not be finished:
    its farthest spawn is 2.40 m, the handover pose is at 0.55 m, and
    `MAX_VX` is 0.20 m/s — 9.2 seconds of driving before a single radian of
    turning, against an 8.0 s episode. The policy arrived 5 times in 12 with
    every miss TRUNCATING rather than failing, which is what a 42% that is
    not about learning looks like.
    """
    _lo, hi, bmax = APPROACH_RUNG_BOX[rung]
    return ((hi - ARRIVE_X) / MAX_VX          # drive it
            + bmax / MAX_WZ                   # turn to face it
            + APPROACH_SLACK_S)

#: Progress toward the handover standoff.
W_APPROACH = 25.0
#: **Per metre the can MOVES**, and it is the whole point of this task. Large
#: relative to the progress term because arriving next to a toppled can is
#: worth less than arriving slowly next to an upright one.
W_DISTURB = -60.0
#: Once, for arriving in the band with the can still standing.
ARRIVE_BONUS = 25.0
#: Per step that any arm geom is outside the rover's own footprint. Laurent
#: asked for the arm to stay inside the shell while driving; this pays for it
#: rather than asserting it, so the policy owns the constraint it has to obey.
W_OUT_OF_SHELL = -0.5
#: The shell, from `robots/moss.py`'s own hull numbers.
SHELL_X = (-0.145, 0.115)
SHELL_Y = 0.141


class MossApproachEnv(MossPickEnv):
    """Drive up to a can and stop at the handover pose WITHOUT touching it.

    A subclass of the pickup env, not a copy: the model, the contract, the
    belief model, the can-shift disturbance and the observation are all the
    same, and what changes is the spawn, the reward and the terminal test.
    The arm starts TUCKED and is paid to stay inside the shell.
    """

    def __init__(self, task: str = "approach", **kw):
        kw.setdefault("pick_rung", 0)          # unused here; the box is its own
        self.approach_rung = int(os.environ.get(
            "MICRODUCK_MOSS_APPROACH_RUNG", kw.pop("approach_rung", 0)))
        if self.approach_rung not in APPROACH_RUNG_BOX:
            raise ValueError(f"approach rung {self.approach_rung} is not "
                             f"{tuple(APPROACH_RUNG_BOX)}")
        # The episode has to be long enough for the rung to be finishable;
        # a caller that asks for its own length still wins.
        kw.setdefault("max_episode_s", approach_seconds(self.approach_rung))
        super().__init__(task="pick", **kw)
        self._arm_geoms = [
            g for g in range(self.model.ngeom)
            if int(self.model.geom_group[g]) == moss.COLLISION_GROUP
            and int(self.model.geom_bodyid[g]) != self.model.body(moss.BASE_BODY).id
            and int(self.model.body_rootid[self.model.geom_bodyid[g]])
            == int(self.model.body_rootid[self.model.body(moss.BASE_BODY).id])
        ]

    # ------------------------------------------------------------- episode

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.driver.spawn(self.data, 0.0, 0.0, 0.0)
        # Commanded to the tuck, which is where the mission loop drives
        # from — and it does NOT get there. MEASURED: from the home pose the
        # arm jams on the hull and the track (`hull/palm`, `track_1/palm`)
        # 0.74 rad short and stays jammed, leaving a link 75 mm outside the
        # chassis; routed via `DROP_POSE` it jams 25 mm out instead, and no
        # path tried reaches the pose at all. So this spawns the arm where it
        # REALLY sits while driving, not at a configuration the robot has
        # never been in, and `W_OUT_OF_SHELL` pays the policy to fold it in
        # from there using the arm actions it already holds.
        self.arm_cmd = dict(zip(moss.ARM_JOINTS, moss.tuck_pose()))
        self.arm_cmd[moss.GRIPPER_JOINT] = 0.041
        self.driver.set_arm(self.arm_cmd)
        for _ in range(int(DEPLOY_SETTLE_S / moss.GRASP_PHYSICS_DT)):
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)
        lo, hi, bmax = APPROACH_RUNG_BOX[self.approach_rung]
        #: Far enough out that a can this rung can SPAWN at is never already
        #: lost, plus room to drift.
        self._lost_radius = hi + LOST_MARGIN_M
        rng = float(self.rng.uniform(lo, hi))
        bearing = float(self.rng.uniform(-bmax, bmax))
        x, y = rng * math.cos(bearing), rng * math.sin(bearing)
        bx, by, byaw = self.driver.pose(self.data)
        c, sn = math.cos(byaw), math.sin(byaw)
        lying = self.domain_rand and self.rng.random() < CAN_LYING_P
        yaw = float(self.rng.uniform(-math.pi, math.pi))
        if lying:
            z = self.prop.radius
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            cr, sr = math.cos(math.pi / 4), math.sin(math.pi / 4)
            quat = [cy * cr, cy * sr, sy * sr, sy * cr]
        else:
            z = self.prop.half_height
            quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.data.qpos[self.can_qadr:self.can_qadr + 3] = [
            bx + x * c - y * sn, by + x * sn + y * c, z]
        self.data.qpos[self.can_qadr + 3:self.can_qadr + 7] = quat
        self.data.qvel[self.can_dadr:self.can_dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self._start_z = float(self.data.xpos[self.can_body][2])
        #: Where the can STARTED, in world coordinates — the disturbance term
        #: is measured against this and not against the previous step, so a
        #: can nudged early and left alone keeps costing what it cost.
        self._can_home = np.array(self.data.xpos[self.can_body][:2], float)
        self._prev_reach = self._reach()
        self._arrived = False
        self._fix = None
        self._seen_w = None
        self._vel = np.zeros(2)
        self._fix_t = -1e9
        self._pending = []
        self._next_det_t = float(self.data.time)
        self._sense()
        self.last_action[:] = 0.0
        self.prev_action[:] = 0.0
        return self._obs(), {}

    # ------------------------------------------------------------- scoring

    def _reach(self) -> float:
        """How far the can is from the handover pose, in the base frame."""
        cb = self._can_base()
        return float(math.hypot(cb[0] - ARRIVE_X, cb[1]))

    def _disturbed(self) -> float:
        """How far the can has been moved from where it started, metres."""
        now = np.array(self.data.xpos[self.can_body][:2], float)
        return float(np.linalg.norm(now - self._can_home))

    def _out_of_shell(self) -> float:
        """How far any arm geom sticks out of the rover's footprint, metres."""
        x, y, yaw = self.driver.pose(self.data)
        c, sn = math.cos(-yaw), math.sin(-yaw)
        worst = 0.0
        for g in self._arm_geoms:
            p = self.data.geom_xpos[g]
            dx, dy = p[0] - x, p[1] - y
            bx, by = dx * c - dy * sn, dx * sn + dy * c
            worst = max(worst, bx - SHELL_X[1], SHELL_X[0] - bx,
                        abs(by) - SHELL_Y)
        return max(0.0, worst)

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self.prev_action = self.last_action.copy()
        self.last_action = a
        for i, j in enumerate(moss.ARM_JOINTS):
            lo, hi = self.model.joint(j).range
            self.arm_cmd[j] = float(np.clip(
                self.arm_cmd[j] + a[i] * ARM_DELTA_RAD, lo, hi))
        lo, hi = self.model.joint(moss.GRIPPER_JOINT).range
        self.arm_cmd[moss.GRIPPER_JOINT] = float(np.clip(
            self.arm_cmd[moss.GRIPPER_JOINT]
            + a[moss.ACT_GRIPPER.start] * JAW_DELTA_M, lo, hi))
        self.driver.set_arm(self.arm_cmd)
        vx, wz = clamp_cmd(float(a[moss.ACT_BASE.start]) * MAX_VX,
                           float(a[moss.ACT_BASE.start + 1]) * MAX_WZ)
        for _ in range(DECIMATION):
            self.driver.set_cmd(vx, wz, self.data.time)
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self._predict()
        self._sense()
        self._sense_arm()

        reach = self._reach()
        cb = self._can_base()
        rew = W_APPROACH * (self._prev_reach - reach)
        rew += W_DISTURB * self._disturbed() * CTRL_DT      # per second held
        rew += W_OUT_OF_SHELL * self._out_of_shell()
        rew += W_ACTION_RATE * float(np.abs(a - self.prev_action).sum())
        rew += W_BASE_EFFORT * (abs(float(a[moss.ACT_BASE.start]))
                                + abs(float(a[moss.ACT_BASE.start + 1])))
        self._prev_reach = reach

        bearing = abs(math.atan2(cb[1], cb[0]))
        speed = abs(self.driver.velocity(self.data)[0])
        arrived = (reach < ARRIVE_TOL_M and bearing < ARRIVE_BEARING
                   and speed < 0.05)
        if arrived and not self._arrived:
            self._arrived = True
            rew += ARRIVE_BONUS
        # The band has to come from the RUNG'S OWN SPAWN BOX. Hardcoded at
        # x<2.0, |y|<1.0 it was tighter than rung 1 spawns: that box reaches
        # 2.40 m at up to 1.4 rad, so 14 of 20 episodes began already outside
        # it and terminated as "lost" on their first step. The stage was
        # training on about six usable episodes in twenty, which is what its
        # ep_len of 41.6 was really reporting — shorter than rung 0's 77
        # while spawning twice as far away, a number that cannot mean what it
        # looks like.
        r = float(np.linalg.norm(cb[:2]))
        lost = not (0.10 < r < self._lost_radius)
        terminated = bool(arrived or lost)
        truncated = self.step_count >= self.max_steps
        return (self._obs(), float(rew), terminated, truncated,
                {"reach": reach, "disturbed": self._disturbed(),
                 "arrived": arrived, "out_of_shell": self._out_of_shell(),
                 "success": arrived})


# ===================================================================== stow
#
# Carry a gripped can to his own bin and let go of it INSIDE. One env for
# both halves of that, not two, and the reason is the failure it exists to
# fix: the scripted loop drops the can partway through the swing, and a
# carry task that ended when the arm arrived would be scored a success at
# exactly the moment the real loop fails. A handover between a "carry"
# policy and a "release" policy is one more seam for the can to fall
# through, and the seam is the bug. Success here is the can in the bin.

#: Where the can is let go: over the bin's interior, a clear 60 mm above the
#: rim it has to cross. Not the bin's centre — `DROP_POSE`'s docstring has
#: the measurement, the centre is 87 mm out of the arm's reach.
#:
#: This is a target the POLICY has to find its own way to, and that is the
#: point of the task. Ramping the arm straight from the lift pose to
#: `moss.DROP_POSE` — the obvious script — delivers the can 0 times in 10,
#: because `shoulder_lift` stalls 0.379 rad short against `bin_x1`, the bin's
#: own front wall. The jammed servo keeps pushing, so opening the jaws
#: releases the stored energy and the can is FLUNG: measured in the world
#: frame (the base does not move) it travels from x -0.034 to +0.266 while
#: falling 0.43 m, landing in front of the robot instead of in the bin.
#: Every straight ramp tried lands the can 0.45 m up, 0.19 m above the rim.
#:
#: The task is still feasible, which is why this is trained rather than
#: scripted: of 60,000 random arm poses, 179 put the gripper contact-free
#: just above the rim (0.26-0.34 m) and 93 put it contact-free INSIDE the
#: bin. The reachable set contains what the script cannot travel to.
#: The can's destination, and the WAYPOINT it has to be reached through.
#:
#: `STOW_TARGET` is inside the bin, a can's half-height above its floor —
#: where the can is meant to end up. But the distance to it cannot be
#: measured in a straight line, and that mistake is worth keeping written
#: down: the can starts at (+0.264, 0, 0.217) and a straight line to
#: (-0.087, 0, 0.168) passes through the bin's FRONT WALL (x -0.006, which
#: spans z 0.111 to 0.296) about 80% of the way along. It never goes over
#: the rim. Progress-pay on that line is an instruction to push the can into
#: the wall, and the policy did exactly as it was told — watched in the lab
#: it grinds the can against the frame, and ep_rew sat at -35 where the same
#: task aimed above the mouth had reached +1.2.
#:
#: So the distance is measured through `STOW_MOUTH`, over the rim: while the
#: can is not yet over the bin's footprint the gap is (can -> mouth) plus
#: (mouth -> target), and once it is over, it is simply (can -> target). The
#: two agree exactly when the can is AT the mouth, so the term stays
#: continuous across the handover and no reward appears out of nowhere.
STOW_TARGET = (-0.087, 0.0, moss.BIN_FLOOR_Z + 0.0575)
STOW_MOUTH = (-0.087, 0.0, moss.BIN_RIM_Z + 0.05)

#: Progress-pay again, on the distance from the can to that point.
W_STOW = 80.0
#: Keeping hold of it is worth the same 0.3 a step as in the pick env, but
#: as a COST OF LETTING GO rather than income for holding — and that sign is
#: the whole point. Written as income it was MEASURED at +59.5 for a policy
#: that pressed nothing for 200 steps: the pick env gets away with a holding
#: trickle because its episodes end within a second or two of the grip,
#: whereas a carry is long and 0.3 a step of free money for standing still
#: beats delivering. As a penalty for being empty-handed it says the same
#: thing about the can and pays exactly nothing for parking.
W_EMPTY = -0.3
#: And time costs, so dawdling with the can loses rather than breaks even.
#: Small next to the delivery bonus: the point is a gradient against
#: waiting, not a reason to fling the can.
W_TIME = -0.05
#: ONCE, the first time the GRIPPER is over the bin while still carrying.
#:
#: Progress-pay on the can alone cannot buy this, and the measurement is
#: unambiguous: ramping the arm along a route that genuinely works — from
#: the lift pose to a configuration verified to hold a can at z 0.182,
#: inside the bin — the can's distance to the mouth RISES for the first
#: third of the move (0.552 -> 0.581) before falling to 0.438. The working
#: route swings the shoulder round about 3.1 rad, from pan +1.35 to -1.78,
#: and the can goes further away while it does. A greedy progress term
#: punishes precisely the part of the manoeuvre that makes it possible, so
#: the policy contorts to shave millimetres off a straight line instead of
#: turning round, which is what it looks like on screen.
#:
#: One-shot rather than per-step, for the reason the pick env's close bonus
#: is: paid every step, a policy parks its hand over the bin and banks the
#: income instead of lowering the can. Gated on CARRYING so an empty
#: gripper waved over the bin earns nothing — the failure also has to not
#: satisfy the proxy.
W_OVER_BIN_BONUS = 15.0
#: Once, for a can at rest inside the bin.
STOW_BONUS = 40.0
#: **AND THEN GET THE ARM OUT.** The stow episode ENDED the moment the can
#: settled, so nothing in ~every stow policy trained here has ever been paid
#: to pull the arm back: it is left extended over the bin, which on the real
#: robot is the pose it would then try to DRIVE in. Nothing in the env even
#: mentions the tuck pose.
#:
#: Paid as progress toward `moss.tuck_pose()` in joint space — the folded pose
#: fitted against this collision model, which the repo's own note calls "a
#: statement about clearance" — so holding still earns nothing and only
#: folding does. The episode then ends when the arm is actually home, which
#: makes "delivered" mean delivered AND stowed rather than delivered and
#: abandoned mid-reach.
W_RETRACT = float(os.environ.get("MICRODUCK_MOSS_RETRACT", "0") or 0.0)
#: Within this much of every tuck joint, the arm is home.
RETRACT_TOL_RAD = 0.15
#: **WHICH joints the fold is actually about.** MEASURED by sweeping each
#: joint at the tuck pose and reading the arm's silhouette: `wrist_roll` over
#: its whole +-2.8 rad range leaves the arm top at 259.5 mm and its forward
#: reach at 114.2 mm — IDENTICAL, to a tenth of a millimetre. The tuck is a
#: statement about clearance (it passes the 261 mm bin rim by 1.5 mm), and
#: rolling the gripper about its own axis does not change the silhouette at
#: all. Requiring it cost 2.431 rad of a 5.20 rad residual — the largest
#: single component of the failure — for nothing.
RETRACT_JOINTS = tuple(j for j in moss.ARM_JOINTS[:5] if j != "wrist_roll")
#: How close counts as having reached the waypoint — looser than the tuck's
#: own tolerance, because it is a place to pass through, not to arrive at.
RETRACT_WP_TOL_RAD = 0.30
#: Paid once for clearing the bin, so the first leg is worth finishing on its
#: own rather than only as a step toward a pose the policy has never reached.
W_RETRACT_WP_BONUS = float(
    os.environ.get("MICRODUCK_MOSS_RETRACT_WP_BONUS", "0") or 0.0)
#: Route the retract through the waypoint at all.
RETRACT_STAGED = os.environ.get("MICRODUCK_MOSS_RETRACT_STAGED", "0") not in ("", "0")
#: **THE RETRACT DRILL — isolate the fold, the way every other stuck skill
#: here was isolated.**
#:
#: Seven training attempts failed and in every one the fold COMPETED with the
#: delivery: paying it more collapsed delivery 14 -> 9, DAgger collapsed it
#: 17 -> 6, and staging the reward collapsed it 14 -> 11 with episodes
#: dropping from 305 steps to 46. The policy kept trading the can for the
#: fold, because one episode contains both and only one of them pays early.
#:
#: So take the delivery away. The episode STARTS where a delivery ends — arm
#: over the bin, object already released into it — and the only thing left to
#: do is get home. MEASURED from 38 real post-delivery poses flown by
#: `teach-moss_stow-ccb305`:
#:
#:     joint            mean      sd
#:     shoulder_pan   -1.796   0.074
#:     shoulder_lift  -0.972   0.207
#:     elbow_flex     +0.081   0.111
#:     wrist_flex     +1.558   0.197
#:     wrist_roll     -1.360   0.691
#:
#: This is the same move as the pick's wrist drill and the headstand's drill
#: rung: a skill that never appears in a rollout cannot be reinforced, and a
#: skill that is always outbid by a competing objective never gets the chance.
RETRACT_DRILL = os.environ.get("MICRODUCK_MOSS_RETRACT_DRILL", "0") not in ("", "0")
POSTDELIVERY_MEAN = np.array([-1.7964, -0.9717, 0.0806, 1.5576, -1.3597])
POSTDELIVERY_SD = np.array([0.0743, 0.207, 0.1113, 0.1967, 0.6912])
#: The share of drill episodes that start AT `moss.RETRACT_WAYPOINT` instead
#: of the post-delivery pose. MEASURED (teach-moss_stow-bd60ff, 2026-09-26):
#: a drill from post-delivery alone learned leg 1 — waypoint reached 12/12,
#: where the shipped stow reaches it 0/12 — and never leg 2, ending 6.37 rad
#: from the tuck. Leg 2 only begins after leg 1 finishes, late in a short
#: episode, so it is barely sampled; starting some episodes on it is the
#: servo-ladder move, not a reward change.
#: The most OTHER pieces of litter already in the bin at reset; each episode
#: draws 0..BIN_CLUTTER of them from `sample_prop`. A real bin is not empty,
#: and an object that lands on another one can perch above the rim.
BIN_CLUTTER = int(os.environ.get("MICRODUCK_MOSS_BIN_CLUTTER", "0") or 0)
#: Clutter is dropped from staggered heights and allowed this long to settle.
CLUTTER_SETTLE_S = 0.6
#: NEVER START AN EPISODE WITH THE OBJECT ALREADY GONE. MEASURED
#: (2026-09-26 bisect): with six shapes a rung-0 reset — which carries the
#: object through seat, lift, high and turned during RESET — handed back an
#: object actually in the jaws 8 times in 60 (ball 0/15, card 0/11, tall
#: 0/3), and fell through silently. The failed aim chain trained on that.
#: With `valid_start`, a failed seat-and-carry draws a new shape, up to
#: VALID_START_REDRAWS times; `seat_redraws` / `start_held` record it.
VALID_START = os.environ.get("MICRODUCK_MOSS_VALID_START", "0") not in ("", "0")
VALID_START_REDRAWS = 6
#: Option A of the placement decision (2026-09-26): the arm camera looks into
#: the bin, `moss_bin.choose_drop_point` picks the clearest spot, the stow is
#: paid for progress to THAT spot and sees it in `moss.OBS_DROP`. Off = the
#: fixed `STOW_TARGET` and dead slots, exactly as before.
DROP_TARGET = os.environ.get("MICRODUCK_MOSS_DROP_TARGET", "0") not in ("", "0")
#: How much of STOW_BONUS depends on WHERE it lands. MEASURED (89fdb5,
#: 2026-09-26): with the drop point only in the progress term, landing
#: 5 cm closer was worth ~4 against a flat 40 for any delivery — perched on
#: clutter included — and the policy ignored the point (slope 0.0, zeroing
#: the slots changed nothing). The bonus is (1-a) + a*exp(-err/DROP_ACC_M)
#: of STOW_BONUS, times PERCHED_FRAC when it rests on clutter. 0 = as before.
DROP_ACCURACY = float(os.environ.get("MICRODUCK_MOSS_DROP_ACCURACY", "0") or 0.0)
DROP_ACC_M = 0.03
#: 0.25 with a=0.8 cut the median delivery to 4.3 of 40 (13 of 21 landings
#: perched) — enough to make delivering barely worth it; 0.5 with a=0.6
#: keeps a far perched delivery at ~10, well above DROP_PENALTY.
PERCHED_FRAC = 0.5
RETRACT_DRILL_WP = float(os.environ.get("MICRODUCK_MOSS_RETRACT_DRILL_WP", "0") or 0.0)
#: Joint noise around the waypoint start, rad.
RETRACT_DRILL_WP_SD = 0.05
#: REVERSE CURRICULUM: the share of drill episodes that start at a random
#: point ALONG the known collision-free path (post-delivery -> waypoint ->
#: tuck), weighted toward the tuck end of its joint-space arc length. MEASURED (teach-moss_fold-
#: 92301e-s1): 2M steps from scratch, and not one training episode reached
#: home — the finish was never sampled, so its value was never learned.
#: Starting some episodes a step from home samples it from the first update.
RETRACT_DRILL_PATH = float(os.environ.get("MICRODUCK_MOSS_RETRACT_DRILL_PATH", "0") or 0.0)
RETRACT_DRILL_PATH_SD = 0.03
#: Start the drill from REAL post-delivery arm poses instead of the Gaussian
#: POSTDELIVERY_MEAN/SD. MEASURED (teach-moss_fold-afd698-s1): 40/40 home from
#: the Gaussian and 0/38 from where the shipped stow actually leaves the arm
#: (elbow 2 sd off that mean, bin contact 73% of the fold). The bank:
#: `data/moss_handoff_poses.npy`, 174 poses — ccb305 at rung 2, six shapes,
#: seeds 0-300, clutter 0 and 0-6 (columns: 5 arm joints, jaw command).
#: `..._heldout.npy` is seeds 300-400, for evaluation only.
#: `moss_tuck_entry_poses_v2.npy`: those 174 + 92 poses where tidy_moss's
#: TUCK state really begins in the moss-yard (seeds 10-21; after release 41,
#: after a failed stow 30, after a failed pick 21), the world ones twice. The
#: first learned fold went home 4/4 after a release in the yard but ~half
#: from the failed-stow and failed-pick poses it had never seen.
RETRACT_DRILL_BANK = os.environ.get("MICRODUCK_MOSS_RETRACT_DRILL_BANK", "0") not in ("", "0")
HANDOFF_BANK = Path(__file__).parent / "data" / "moss_handoff_poses.npy"
HANDOFF_BANK_SD = 0.02
#: THE COMMAND LEASH, rad: each arm joint's command is kept within this of its
#: MEASURED position. MEASURED (seed 466, 0-6 clutter): a jaw pad pinned
#: against the dropped object and a clutter item for ~50 ticks while the
#: command kept advancing 0.03 rad/tick; when it slipped free the stored error
#: released at 10-14 rad/s with nothing in contact — the LEARNED fold and the
#: SCRIPTED one both did it. On the robot this is the brain writing goal =
#: present position +- the leash. 0 = off, as every earlier run trained.
CMD_LEASH_RAD = float(os.environ.get("MICRODUCK_MOSS_CMD_LEASH", "0") or 0.0)
#: The drill is one motion, not an episode.
RETRACT_DRILL_S = 8.0
#: Paid ONCE for finishing. Progress alone pays the same for the first radian
#: as the last, and the last one costs jerk with no bonus at the end, so the
#: fold stalled around the carry pose: 5.20 rad residual against a carry pose
#: that sits 5.19 rad from tuck, i.e. it came back and stopped.
W_HOME_BONUS = float(os.environ.get("MICRODUCK_MOSS_HOME_BONUS", "0") or 0.0)
#: How long the fold may take before the episode gives up. Without this the
#: retract phase never truncated — `truncated` is gated on nothing having been
#: released yet — so an arm that failed to fold ran forever.
#: MEASURED, not guessed: driving the arm to the tuck pose open-loop after a
#: delivery takes more than 150 steps — at 150 the fold truncated on 12 of 12
#: seeds with `shoulder_pan` 0.42 rad and `elbow_flex` 0.57 rad still moving,
#: so the terminal condition was unreachable and no amount of training could
#: have finished it. The arm moves 0.03 rad per step, so a 2.5 rad joint needs
#: ~84 steps on its own; 300 leaves room for the whole chain.
RETRACT_MAX_STEPS = 300
#: Touching the bin on the way out is how a gripper catches its own load and
#: drags it back out; charged per step, like the floor.
W_BIN_SCRAPE = float(os.environ.get("MICRODUCK_MOSS_BIN_SCRAPE", "0") or 0.0)
#: Once, for letting go of it anywhere else. Bigger than the pick env's
#: knock penalty because this is THE failure being trained out.
DROP_PENALTY = -15.0
#: A release is judged by where the can LANDS, not by where it was let go.
#: This is the single worst bug this task has had. Judged at the instant the
#: jaws opened, a can let go over the bin is not yet in it — it is 0.18 m up,
#: because the arm cannot descend to the rim — so the release scored
#: `dropped` and -15. MEASURED on the stage-1 policy: it parks the can
#: directly over the bin footprint 8 times out of 8, and simply opening the
#: jaws there lands it in the bin 8 times out of 8 — and it had correctly
#: learned never to open them, because the env punished the winning move.
#: The episode now keeps running after a release until the can comes to
#: rest, and asks then.
#: **How much of the policy's BASE command survives once the can is within
#: the arm's reach**, in `MossPickEnv`. 1.0 is the env every shipped leg
#: trained in, and it produces a policy that drives flat out for the whole
#: grab: 29,165 creep ticks in the room command base motion on 100% of them,
#: |vx| saturated at 0.200 m/s, |wz| a median 0.808 rad/s, 78% of ticks
#: turning harder than 0.3 rad/s. That is what a human watching the lab
#: described as "spinning around trying to hit it", and it is why 158 of 434
#: can-touches are the TRACK and the HULL rather than the gripper.
#:
#: Damping it at DEPLOY time only makes things worse — 9, 14, 8 and 15 cans
#: of 36 at scales 0.00, 0.25, 0.50 and 0.75, against 22 unchanged — because
#: the policy steers with the base and loses its manifold without it. So the
#: knob lives here, where a policy can be TRAINED to close the last stretch
#: with the arm. Set `MICRODUCK_MOSS_PICK_BASE_REACH` to sweep it.
PICK_BASE_REACH_SCALE = float(os.environ.get("MICRODUCK_MOSS_PICK_BASE_REACH", "1.0"))
#: The range inside which that scale applies — `moss.GRASP_STANDOFF_M` plus a
#: little, i.e. where the grasp pose can actually reach.
PICK_BASE_REACH_M = 0.30
#: **WHICH DISTANCE THE PROGRESS TERM PAYS FOR — and the reason the base
#: barges.** `POCKET` is a point fixed to the CHASSIS, so `_gap()` measures
#: can-to-chassis and the ARM CANNOT EARN THE DOMINANT TERM AT ALL. MEASURED
#: on 2026-09-25, from the deployed pose at rung 2, 60 steps of full command:
#:
#:     arm only    gripper moved 14.8 cm, the reward's gap moved  0.2 cm
#:     base only   gripper moved 41.1 cm, the reward's gap moved  4.8 cm
#:
#: A 60-weight term paying the base ~20x what it pays the arm is why a policy
#: drives at the can instead of reaching for it, why the penalties totalling
#: -12 against +18 of pay did not stop it, and why clamping the base in reach
#: (`PICK_BASE_REACH_SCALE`) on its own would leave the reward FLAT there —
#: nothing the arm does would earn. Retargeting the term to the GRIPPER makes
#: reaching pay; the base still earns, because driving moves the gripper too.
#:
#: Set `MICRODUCK_MOSS_GAP_TCP=1`. Off by default until a run measures it.
GAP_FROM_TCP = os.environ.get("MICRODUCK_MOSS_GAP_TCP", "0") not in ("", "0")
#: **A SHOVE ENDS THE EPISODE, instead of costing something affordable.**
#:
#: The reward already charges for disturbing the can (`W_CAN_DISTURB`) and for
#: hitting it with the chassis (`W_CAN_BODY_HIT`), and MEASURED on 2026-09-25
#: it does not matter: the shipped leg picks the can 93% of the time WHILE
#: moving it a median 6.3 cm (mean 17.6, max 108) and commanding both base
#: axes saturated in every condition, including with the can already between
#: its pads. Barging works, so barging is never unlearned. No weight fixes
#: that — the strategy has to stop succeeding.
#:
#: The precedent is this repo's own: reward shaping could not make the G1 kick
#: past 0.10 m, and a DeepMimic-style early termination on a 5 -> 15 -> 30 cm
#: ladder took it to 0.62 m. Same shape here, on the can's NET displacement
#: from where it spawned (path length accumulates the jitter of a can resting
#: against a pad and would abort on contact alone).
#:
#: CALIBRATED against how the policies actually fly today, so that no rung is
#: a structural no-op — the mistake `board_margin=0.25` already made here:
#:
#:     limit    aborts (shipped leg)   aborts (511e1f)
#:     15 cm           28%                   15%
#:      8 cm           45%                   28%
#:      4 cm           65%                   57%
#:
#: Only while the jaws have never closed on it: once it has been gripped, the
#: can is SUPPOSED to move. 0 disables, which is every leg trained before now.
SHOVE_LIMIT_M = float(os.environ.get("MICRODUCK_MOSS_SHOVE_LIMIT", "0") or 0.0)
#: Terminating already forfeits the pick bonus and every future step of pay,
#: so the sign is unambiguous without this; it is here so the abort cannot be
#: cheaper than a long episode of small penalties.
W_SHOVE_ABORT = -5.0

RELEASE_SETTLE_STEPS = 50
#: Below this the can has stopped moving (m/s).
CAN_AT_REST_MPS = 0.05
#: Contact between a pad and the can FLICKERS: a carried can breaks and
#: remakes contact with one pad as the arm accelerates, and read raw that
#: ended an episode as "dropped" on its first step. A hold has to be gone
#: for this many control steps, and the can has to have left the gripper's
#: reach, before it counts as let go.
DROP_DEBOUNCE = 4
#: How far the can has to be from the tcp before it is out of the jaws
#: whatever the contacts say — a released can falls clear of this in about a
#: tenth of a second.
CARRY_RADIUS_M = 0.08

#: How long a stow episode runs. NOT the env's 8 s default, and this is the
#: whole reason the task would not train. MEASURED: the policy carries the
#: can over the bin by about step 180, and the episode ended at 200 — so the
#: window in which letting go actually pays was 20 steps wide, a tenth of the
#: episode. Gaussian exploration opens the jaws at a uniformly random moment,
#: which is therefore almost always too early and scores -15, and "never
#: release" becomes locally optimal: all three stages of a chain trained this
#: way scored 0 of 12 into the bin with 10 of 12 STILL HOLDING at the end.
#: The +40 for a delivery was never sampled, so its value was never learned.
#:
#: This is the same mistake as the approach leg's 8.0 s episode against 9.2 s
#: of driving, and it has the same shape: a constant that does not follow
#: what the task actually takes. 20 s leaves ~300 steps of parked time in
#: which opening the jaws is the best thing the policy can do.
#:
#: (The park itself is a JAM, worth knowing: the policy holds a command the
#: arm cannot reach, pressed against the bin's front wall, which is how it
#: gets the can as low as 0.18 m above the rim. Replaying the commanded pose
#: from the lift reaches a DIFFERENT, contact-free configuration that puts
#: the can 0.35 m out in front instead — so a drill rung built by replaying
#: that pose does not reproduce the state at all.)
STOW_EPISODE_S = 20.0

#: Per rung: (how far off centre the can sits between the pads, how far it is
#: tilted). This is the answer to "the can can be in multiple positions" —
#: the pick policy does not hand over a centred can, it hands over whatever
#: it managed to close on.
#:
#: The values are read off the SEAT RATE, measured 20 resets per cell: a
#: square upright can seats 20/20, 4 mm off centre 11/20, 12 mm 5/20, 18 mm
#: 1/20, and any tilt costs about half. So the ladder runs to 12 mm and 0.4
#: rad rather than further: a grip the gripper can barely form is not a grip
#: the pick policy would ever hand over either.
#: The route to the bin, verified end to end: from the lift pose, fold the
#: arm UP and in (clear of the bin), swing the shoulder round, then descend
#: into the mouth. Ramped along these and opened at the end it puts the can
#: in the bin 6 times out of 6.
#:
#: The waypoints are the point. The final pose alone is reachable in free
#: space but NOT by a straight joint-space path from the lift: interpolated
#: directly the arm jams with `shoulder_pan` 0.851 rad short against
#: `bin_x1` (contacts `gripper_mount` and `wrist_link_proxy`) and strands
#: the can out to the side at y +0.215 — 0 of 6. The same trap as the tuck:
#: a pose that is reachable and a path that is not. The rotation has to
#: happen ABOVE the bin.
#:
#: It is NOT the can slipping, which was the first guess and was wrong:
#: measured in the gripper's own frame the can moves 1.9 mm over the whole
#: six-second rotation and the grip never lets go.
#: **The descent reaches the bin's mouth by LEANING ON IT, and that is load
#: bearing.** Ramped to `(-1.776, -0.699, 1.648, -0.722, 1.154)` the arm
#: never arrives: it stalls 0.44 rad short with 33 N against `bin_x1`, the
#: bin's own front wall, on every release of 51. The wall is a mechanical
#: stop, and the configuration it stops the arm in is exactly the one that
#: holds the can over the mouth — x = -0.128 m in the base frame against a
#: mouth of [-0.164, -0.010], z = 0.318 just clear of the 0.261 rim.
#:
#: That was worth an afternoon because it inverts. Searching for a pose the
#: arm could REACH with zero force on the wall found eleven, and the best of
#: them delivers 12/12 in `MossStowEnv` — and 5 of 16 in the room, releasing
#: the can at x = +0.018, z = 0.462: past the mouth and 20 cm too high. The
#: probe and the room disagreed completely, and the ROOM is the mission. The
#: contact was not the bug; it was the only thing putting the gripper in the
#: right place.
#:
#: So `STOW_INSIDE` is now the pose the jam SETTLES at, measured off 25 room
#: releases, rather than a pose commanded 0.44 rad through a wall. Same
#: delivery, a quarter of the force (12 seeds):
#:
#:     commanded past the wall   can x -0.128  z 0.318   12/16   32.8 N
#:     its own resting pose      can x -0.136  z 0.318   12/15    7.3 N
#:     contact-free (probe 12/12) can x +0.018 z 0.462    5/16      -
#:
#: The residual 7.3 N is the arm RESTING on the wall, not a servo driven
#: into it, and that is the part that matters off-sim: the old command held
#: five servos in a stall for the whole release. It is still a delivery that
#: depends on touching the bin, which is a fact about this geometry Laurent
#: should know before it is called a hardware routine.
#:
#: What this does NOT fix is the carry: only 29-33% of lifts still hold the
#: can when the release fires, and that loss dwarfs the drop's.
STOW_HIGH = (1.3534, -1.35, 0.10, 1.4345, -0.05)
STOW_TURNED = (-1.776, -1.35, 0.10, 1.4345, -0.05)
STOW_INSIDE = (-1.7706, -0.8998, 1.2069, -0.717, 1.154)
#: LET GO HIGH OVER THE MIDDLE, wrist as carried (2026-09-26). IK'd to put
#: the tool point at the bin centre (-0.087, 0) 11 cm above the rim — where
#: the learned stow lets go. `STOW_INSIDE` descends to 0.331 m with the wrist
#: turned, and there the object rests on the palm when the pads open: in the
#: stow env, low releases wedge on the pads or clutter (75% delivered) and a
#: high one over the centre delivers 91% and fills a bin fullest (6.08 items
#: in the fill test against 3.00 low). `STOW_TURNED`'s tool point sits over
#: the bin's FRONT WALL (x -0.012), which is why a last move is still needed.
STOW_RELEASE_HIGH = (-1.8745, -0.6995, -0.2721, 1.4345, -0.05)

#: WHERE EACH RUNG STARTS — one more piece of the route per rung, which is
#: the whole point of the ladder:
#:
#:   0  already swung round over the bin  -> learn the descent and the release
#:   1  folded up and clear, facing front -> learn the ROTATION as well
#:   2  the lift pose, can just picked up -> learn the fold-up too, i.e. all
#:
#: The first version of this went straight from rung 0 to the lift pose, so
#: rungs 1 and 2 differed only in how the can sat in the jaws and the whole
#: rotation arrived at once — the exact part that four chains of 2M steps
#: had already failed to learn, because turning the right way makes the
#: can's distance to the bin WORSE for a third of the move. A ladder whose
#: second rung contains the entire unsolved problem is not a ladder.
STOW_RUNG_START: dict[int, str] = {0: "turned", 1: "high", 2: "lift"}

#: NOT dead centre, and that is a measurement rather than taste. With the
#: pads' torsional friction at his 0.01 a perfectly centred upright can
#: seated 20 of 20 and was the easiest case; at the corrected 0.15 it seats
#: 0 of 20, while 4 mm of offset seats 10. A perfectly symmetric grasp is a
#: measure-zero configuration — both pads arrive at once on the axis of a
#: cylinder — and it is the artificial SEAT that is brittle there, not the
#: grip: the pick env, where a policy closes the jaws on a can standing on
#: the floor, went 9/12 to 12/12 under the same friction. So the drill rung
#: asks for a millimetre or two of offset, which is what a real handover
#: looks like anyway.
STOW_RUNG_GRIP: dict[int, tuple[float, float]] = {
    0: (0.004, 0.05),
    1: (0.008, 0.20),
    2: (0.012, 0.40),
}
#: Every rung seats the can at the GRASP pose, on the floor, and then RAISES
#: it to `moss.LIFT_POSE` before the episode starts. Both halves matter.
#:
#: Seating happens on the floor because that is where a grip that holds gets
#: made: seating in mid-air at the lift pose caught nothing at all (0 of 12),
#: the can falling through the closing jaws.
#:
#: The raise is what makes the task REAL, and leaving it out was a mistake
#: worth recording. Without it the episode began with the jaws merely around
#: a can still standing on the ground — MEASURED, the can sat at z 0.057
#: against a floor height of 0.058 for all 53 steps of a do-nothing episode,
#: then "dropped" when a slow drift broke contact. Nothing was ever carried,
#: so the policy was being trained to keep hold of something it had not
#: picked up, and every episode ended in 4 steps at full action scale. The
#: raise costs retries — it keeps the can 6-7 times in 9, and that band does
#: not move with ramp duration or jaw force — but `SEAT_TRIES` covers it, and
#: a rung that starts with the can in the air is the state the pick policy
#: actually hands over.
#: Tries to seat the can in the jaws before the episode gives up and takes
#: the nominal centred grip. Bounded because an unbounded retry loop in a
#: reset is a hang, and a reset that hangs takes the whole lab with it.
SEAT_TRIES = 8
SEAT_SETTLE_S = 0.6
#: How long the arm is given to raise the seated can clear of the floor.
RAISE_SETTLE_S = 1.2


class MossStowEnv(MossPickEnv):
    """Carry a held can to the bin and release it inside.

    Subclasses the pickup env for the model, the 32-slot contract, the belief
    model and the observation. What changes is where the episode STARTS — the
    can already in the jaws, seated at a random offset — and what it is paid
    for, which is the ground the can makes up toward the bin's mouth and then
    letting go of it there.
    """

    def __init__(self, task: str = "stow", **kw):
        self.retract = float(kw.pop("retract", W_RETRACT))
        self.home_bonus = float(kw.pop("home_bonus", W_HOME_BONUS))
        self.retract_staged = bool(kw.pop("retract_staged", RETRACT_STAGED))
        self.retract_drill = bool(kw.pop("retract_drill", RETRACT_DRILL))
        self.retract_drill_wp = float(kw.pop("retract_drill_wp",
                                             RETRACT_DRILL_WP))
        self.retract_drill_path = float(kw.pop("retract_drill_path",
                                               RETRACT_DRILL_PATH))
        bank = kw.pop("retract_drill_bank", RETRACT_DRILL_BANK)
        # True = the training bank; a path = that file (the held-out set)
        self._bank = None
        if bank:
            if bank is True:
                self._bank = np.load(HANDOFF_BANK)
            else:
                bp = Path(bank)
                if not bp.is_absolute() and not bp.is_file():
                    bp = HANDOFF_BANK.parent / bank
                self._bank = np.load(bp)
        self.bin_clutter = int(kw.pop("bin_clutter", BIN_CLUTTER))
        self.valid_start = bool(kw.pop("valid_start", VALID_START))
        self.seat_redraws = 0
        self.seat_fallback = False
        self.start_held = True
        self.cmd_leash = float(kw.pop("cmd_leash", CMD_LEASH_RAD))
        self._clutter = ()
        self._clutter_poses = None
        self.drop_target = bool(kw.pop("drop_target", DROP_TARGET))
        self.drop_accuracy = float(kw.pop("drop_accuracy", DROP_ACCURACY))
        self._drop = (float(STOW_TARGET[0]), float(STOW_TARGET[1]))
        self._drop_seen = 0
        if self.drop_target and kw.get("publish_attitude"):
            raise ValueError("drop_target and publish_attitude both want "
                             "obs slots 28-29 (moss.OBS_DROP / OBS_TARGET_AXIS)")
        if self.retract_drill:
            kw.setdefault("max_episode_s", RETRACT_DRILL_S)
        self.retract_wp_bonus = float(
            kw.pop("retract_wp_bonus", W_RETRACT_WP_BONUS))
        self.bin_scrape = float(kw.pop("bin_scrape", W_BIN_SCRAPE))
        kw.setdefault("pick_rung", 0)
        self.stow_rung = int(os.environ.get("MICRODUCK_MOSS_STOW_RUNG",
                                            kw.pop("stow_rung", 0)))
        if self.stow_rung not in STOW_RUNG_GRIP:
            raise ValueError(f"stow rung {self.stow_rung} is not "
                             f"{tuple(STOW_RUNG_GRIP)}")
        kw.setdefault("max_episode_s", STOW_EPISODE_S)
        super().__init__(task="pick", **kw)

    # ------------------------------------------------------------- episode

    def _target_base(self) -> np.ndarray:
        if self.drop_target:
            return np.array([self._drop[0], self._drop[1], STOW_TARGET[2]])
        return np.asarray(STOW_TARGET, float)

    def _look_into_bin(self) -> None:
        """Choose this episode's drop point from what the ARM camera sees.

        The look is taken from `moss_bin.BIN_LOOK_POSE` — set on the
        kinematic tree only and put back, no physics — the pose the brain
        tilts the wrist into after a delivery, before it folds.
        """
        if not self.drop_target:
            return
        from . import moss_bin
        dets = []
        self._drop_seen = 0
        cam = self._arm_cam_id
        if cam >= 0 and self._clutter:
            qsave = self.data.qpos.copy()
            for j, v in zip(moss.ARM_JOINTS[:5], moss_bin.BIN_LOOK_POSE):
                self.data.qpos[self.model.joint(j).qposadr[0]] = float(v)
            mujoco.mj_kinematics(self.model, self.data)
            cp = np.array(self.data.xpos[cam], float)
            cm = np.array(self.data.xmat[cam], float)
            ids = [self.model.body(f"clutter{k}").id
                   for k in range(len(self._clutter))]
            pts = [np.array(self.data.xpos[b], float) for b in ids]
            foot = [moss_bin.footprint_radius(c, self.data.xmat[b])
                    for c, b in zip(self._clutter, ids)]
            self.data.qpos[:] = qsave
            mujoco.mj_forward(self.model, self.data)
            for c, pw, fr in zip(self._clutter, pts, foot):
                if not (moss_bin.in_view(cp, cm, pw)
                        and moss_bin.clears_walls(
                            cp, pw, self.driver.pose(self.data))):
                    continue
                if self.rng.random() < moss_bin.DETECT_DROPOUT:
                    continue
                b = self._to_base(pw[:2])
                n = self.rng.normal(0.0, moss_bin.DETECT_POS_SD, 2)
                dets.append((float(b[0] + n[0]), float(b[1] + n[1]), fr))
                self._drop_seen += 1
        self._drop = moss_bin.choose_drop_point(dets, self.prop.radius)

    def _tcp_over_bin(self) -> bool:
        """Is the GRIPPER over the bin's mouth? (the can may lag behind it)"""
        t = self.data.site_xpos[self.tcp_site]
        x, y, yaw = self.driver.pose(self.data)
        c, sn = math.cos(-yaw), math.sin(-yaw)
        dx, dy = t[0] - x, t[1] - y
        bx, by = dx * c - dy * sn, dx * sn + dy * c
        return bool(moss.BIN_INTERIOR_X[0] < bx < moss.BIN_INTERIOR_X[1]
                    and moss.BIN_INTERIOR_Y[0] < by < moss.BIN_INTERIOR_Y[1])

    def _over_bin(self) -> bool:
        cb = self._can_base()
        return bool(moss.BIN_INTERIOR_X[0] < cb[0] < moss.BIN_INTERIOR_X[1]
                    and moss.BIN_INTERIOR_Y[0] < cb[1] < moss.BIN_INTERIOR_Y[1])

    def _stow_gap(self) -> float:
        """Distance to the bin's floor THROUGH its mouth, not through a wall.

        See `STOW_MOUTH`: the straight line from where the can starts to
        where it belongs goes through the bin's front wall, so paying for
        straight-line progress paid for grinding the can into the frame.
        """
        can = self._can_base()
        target = self._target_base()
        if self._over_bin():
            return float(np.linalg.norm(can - target))
        mouth = np.asarray(STOW_MOUTH, float)
        return float(np.linalg.norm(can - mouth)
                     + np.linalg.norm(mouth - target))

    def _carrying(self) -> bool:
        """Both pads in CONTACT with the can — `MossDriver.held_body`.

        Not the pick env's test, and not a servo-stall threshold either.
        `MossPickEnv._held` asks whether the can has come off the ground,
        which is what separates a grip from a crush THERE; here the can
        starts in the air, so it would answer yes to one resting on the bin's
        rim. The stall test is worse: MEASURED along a raise, a plainly
        carried can — 23 mm from the tcp and climbing from z 0.06 to 0.22 —
        reads a stall of 1.96 mm against `GRIP_STALL_M`'s 2.00, so the grip
        test said DROPPED for all twelve resets of a rung where nothing was
        ever dropped. A hold squeezes less once the can is off the floor and
        no longer pressed into the pads. Contact is the thing itself and has
        no threshold to sit the wrong side of.
        """
        if self._gripped_now():
            self._last_contact = self.step_count
            return True
        can = self.data.xpos[self.can_body]
        tcp = self.data.site_xpos[self.tcp_site]
        if float(np.linalg.norm(can - tcp)) > CARRY_RADIUS_M:
            return False
        return (self.step_count - self._last_contact) < DROP_DEBOUNCE

    def _in_bin(self) -> bool:
        """Is the can IN the bin — above its floor, below its rim?

        The floor bound is not decoration. Without it this box is open
        downwards, so a can lying on the GROUND anywhere under the bin's
        footprint scores as delivered, and `STOW_BONUS` pays for it. The bin
        is mounted on the rover with clearance beneath, so that region is
        reachable: a can dropped short can roll under the chassis into it.
        Found 2026-09-24 while auditing why a delivery metric disagreed with
        what the rollouts showed.
        """
        cb = self._can_base()
        return bool(moss.BIN_INTERIOR_X[0] < cb[0] < moss.BIN_INTERIOR_X[1]
                    and moss.BIN_INTERIOR_Y[0] < cb[1] < moss.BIN_INTERIOR_Y[1]
                    and moss.BIN_FLOOR_Z < cb[2] < moss.BIN_RIM_Z)

    def _raise_to(self, pose, seconds: float | None = None) -> bool:
        """RAMP the arm to `pose`, and say whether the can survived it.

        Not a step input to the target, which is what `brain/tidy_moss.py`
        used to do and what this task exists to fix: commanding `LIFT_POSE`
        outright from the grasp dropped the can 12 times out of 12, the
        servos all setting off at full rate and the jaws shearing out from
        under it. Ramped it keeps the can 6-7 times in 9 — a band that did
        not move across ramps of 1.2, 2.5 and 4.0 s or jaw commands of 27, 24
        and 21 mm, so the remainder is retried rather than tuned away.
        """
        start = np.array([self.arm_cmd[j] for j in moss.ARM_JOINTS], float)
        goal = np.asarray(pose, float)
        n = int((seconds or RAISE_SETTLE_S) / moss.GRASP_PHYSICS_DT)
        for k in range(n):
            f = (k + 1) / n
            self.arm_cmd.update(
                zip(moss.ARM_JOINTS, start + f * (goal - start)))
            self.driver.set_arm(self.arm_cmd)
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)
        return self._gripped_now()

    def _gripped_now(self) -> bool:
        """Both pads on the can THIS instant, with no debounce.

        The seat has to ask the raw question: a debounced answer reads True
        for the first few steps whatever the contacts say, which reported
        twelve seated resets out of twelve for a rung where the can was in
        fact standing on the floor untouched by `pad_left`.
        """
        return int(self.driver.held_body(self.data)) == int(self.can_body)

    def _maybe_new_prop(self) -> None:
        """The shape to carry, and this episode's clutter in the bin.

        No random draw at all when `bin_clutter` is 0, so every seed of a run
        without clutter replays the episodes it always did.
        """
        new = ()
        if self.bin_clutter > 0:
            n = int(self.rng.integers(0, self.bin_clutter + 1))
            new = tuple(sample_prop(self.rng, self.litter) for _ in range(n))
        if self.prop_variety:
            self.prop = sample_prop(self.rng, self.litter)
        if self.prop_variety or new or self._clutter:
            self._clutter, self._clutter_poses = new, None
            self._bind_model()
            if new:
                # settle ONCE as free bodies, then weld them where they lie
                self._clutter_poses = self._settle_clutter_once()
                self._bind_model()

    def _settle_clutter_once(self) -> list:
        """Drop the clutter into the bin as free bodies, let it come to rest,
        and return each item's pose in the ROVER's frame."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.driver.spawn(self.data, 0.0, 0.0, 0.0)
        self.arm_cmd = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
        self.arm_cmd[moss.GRIPPER_JOINT] = 0.041
        self.driver.set_arm(self.arm_cmd)
        keep = None
        if self.retract_drill:            # its object sits at STOW_TARGET
            keep = (float(STOW_TARGET[0]), float(STOW_TARGET[1]),
                    self.prop.radius + 0.005)
        self._place_clutter(keep_clear=keep)
        r = self.model.body("rover").id
        pr = np.array(self.data.xpos[r], float)
        Rr = np.array(self.data.xmat[r], float).reshape(3, 3)
        qr_inv = np.zeros(4)
        mujoco.mju_negQuat(qr_inv, np.array(self.data.xquat[r], float))
        out = []
        for k in range(len(self._clutter)):
            b = self.model.body(f"clutter{k}").id
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, qr_inv, np.array(self.data.xquat[b], float))
            out.append((Rr.T @ (np.array(self.data.xpos[b], float) - pr), q))
        return out

    def _place_clutter(self, keep_clear=None) -> None:
        """Drop this episode's clutter into the bin and let it settle.

        Staggered heights so no two start interpenetrating; `keep_clear` is an
        (x, y, r) disc left empty — the drill's already-delivered object.
        """
        if not self._clutter or self._clutter_poses is not None:
            return                       # static: already where it settled
        lo_x, hi_x = moss.BIN_INTERIOR_X
        lo_y, hi_y = moss.BIN_INTERIOR_Y
        for k, c in enumerate(self._clutter):
            m = c.radius + 0.004
            for _ in range(20):
                x = float(self.rng.uniform(lo_x + m, hi_x - m))
                y = float(self.rng.uniform(lo_y + m, hi_y - m))
                if keep_clear is None or math.hypot(
                        x - keep_clear[0], y - keep_clear[1]) > (
                        keep_clear[2] + c.radius):
                    break
            z = moss.BIN_FLOOR_Z + c.half_height + 0.003 + 0.045 * k
            yaw = float(self.rng.uniform(-math.pi, math.pi))
            adr = self.model.joint(f"clutter{k}_free").qposadr[0]
            self.data.qpos[adr:adr + 7] = [x, y, z, math.cos(yaw / 2),
                                           0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(CLUTTER_SETTLE_S / moss.GRASP_PHYSICS_DT)):
            self.driver.set_arm(self.arm_cmd)
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)

    def _clutter_in_bin(self) -> int:
        n = 0
        for k in range(len(self._clutter)):
            b = self.model.body(f"clutter{k}").id
            x, y, z = self.data.xpos[b]
            n += int(moss.BIN_INTERIOR_X[0] < x < moss.BIN_INTERIOR_X[1]
                     and moss.BIN_INTERIOR_Y[0] < y < moss.BIN_INTERIOR_Y[1]
                     and moss.BIN_FLOOR_Z < z < moss.BIN_RIM_Z)
        return n

    def _obs(self) -> np.ndarray:
        o = super()._obs()
        if self.drop_target:
            from . import moss_bin
            o[moss.OBS_DROP] = moss_bin.drop_obs(self._drop)
        return o

    def _stow_bonus(self) -> float:
        """STOW_BONUS, scaled by where it was LET GO when `drop_accuracy` > 0.

        The release point, not the resting point: MEASURED (ac5309), objects
        move a median 40 mm AFTER release (75th pct 72 mm; balls 43, squat
        cans 53, blocks 3) — scoring the rest position paid mostly for the
        bounce, and the policy never came to follow the spot (slope ~0). A
        perch is still the OUTCOME's discount.
        """
        a = self.drop_accuracy
        if a <= 0.0 or not self.drop_target:
            return STOW_BONUS
        _land, perched = self.landing()
        err = self._release_err if self._release_err is not None else _land
        f = (1.0 - a) + a * math.exp(-err / DROP_ACC_M)
        return STOW_BONUS * f * (PERCHED_FRAC if perched else 1.0)

    def landing(self) -> tuple[float, bool]:
        """Where the delivered object ended up: its distance from the drop
        point in the base frame, and whether it is PERCHED — resting more
        than 1 cm above where the bin floor would put it, i.e. on clutter."""
        cb = self._can_base()
        err = float(math.hypot(cb[0] - self._drop[0], cb[1] - self._drop[1]))
        # the object's LOWEST point, from its orientation — a centre-height
        # test calls a squat can lying on its side "perched" (centre = radius)
        R = np.asarray(self.data.xmat[self.can_body], float).reshape(3, 3)
        z = float(self.data.xpos[self.can_body][2])
        pr = self.prop
        if pr.shape == "sphere":
            low = z - pr.radius
        elif pr.shape == "cylinder":
            cz = abs(float(R[2, 2]))
            low = z - (pr.half_height * cz
                       + pr.radius * math.sqrt(max(0.0, 1.0 - cz * cz)))
        else:
            low = z - float(sum(abs(R[2, i]) * pr.size[i] for i in range(3)))
        perched = low > moss.BIN_FLOOR_Z + 0.01
        return err, bool(perched)

    def _reset_retract_drill(self):
        """Start where a delivery ENDS: arm over the bin, object already in it.

        The fold is then the only thing in the episode, so it cannot be
        outbid by the delivery — which is what happened in all seven previous
        attempts (delivery collapsed 14->9, 17->6, 14->11 as the fold was
        paid more).
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.driver.spawn(self.data, 0.0, 0.0, 0.0)
        at_wp = bool(self.rng.random() < self.retract_drill_wp)
        if at_wp:
            q = (np.asarray(moss.RETRACT_WAYPOINT, float)
                 + self.rng.normal(0.0, RETRACT_DRILL_WP_SD, 5))
        elif self._bank is not None:
            row = self._bank[int(self.rng.integers(len(self._bank)))]
            q = row[:5] + self.rng.normal(0.0, HANDOFF_BANK_SD, 5)
        else:
            q = POSTDELIVERY_MEAN + self.rng.normal(0.0, POSTDELIVERY_SD)
        if (self.retract_drill_path > 0.0
                and self.rng.random() < self.retract_drill_path):
            wp = np.asarray(moss.RETRACT_WAYPOINT, float)
            tk = np.asarray(moss.tuck_pose(), float)[:5]
            l1 = float(np.abs(wp - q).sum())
            l2 = float(np.abs(tk - wp).sum())
            # weighted toward HOME: uniform arc length AND a linear ramp both
            # put only 2 of 40 starts within 0.5 rad of it (measured) — the
            # elbow and wrist each swing ~3 rad on leg 2, so "near home" is
            # the last ~8% of the arc. A cube root puts ~a fifth there.
            u = (l1 + l2) * float(self.rng.uniform(0.0, 1.0)) ** (1.0 / 3.0)
            if u < l1:
                q = q + (wp - q) * (u / max(l1, 1e-9))
            else:
                q = wp + (tk - wp) * ((u - l1) / max(l2, 1e-9))
                at_wp = True                     # already on leg 2
            q = q + self.rng.normal(0.0, RETRACT_DRILL_PATH_SD, 5)
        for j, v in zip(moss.ARM_JOINTS[:5], q):
            lo, hi = self.model.joint(j).range
            v = float(np.clip(v, lo + 1e-3, hi - 1e-3))
            self.data.qpos[self.model.joint(j).qposadr[0]] = v
            self.arm_cmd[j] = v
        self.arm_cmd[moss.GRIPPER_JOINT] = moss.MISSION_OPEN_M
        self.driver.set_arm(self.arm_cmd)
        self._place_clutter(keep_clear=(float(STOW_TARGET[0]),
                                        float(STOW_TARGET[1]),
                                        self.prop.radius + 0.005))
        # the object is already IN the bin, which is what "delivered" means
        self.data.qpos[self.can_qadr:self.can_qadr + 3] = [
            float(STOW_TARGET[0]), float(STOW_TARGET[1]),
            moss.BIN_FLOOR_Z + self.prop.half_height]
        self.data.qpos[self.can_qadr + 3:self.can_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(0.2 / moss.GRASP_PHYSICS_DT)):
            self.driver.set_arm(self.arm_cmd)
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)
        self.step_count = 0
        self._released_at = 0
        self._release_err = None
        # Everything `reset` sets, set to what "already delivered" means.
        # Missing `_last_contact` killed every worker on its first step
        # (teach-moss_stow-71087f, 0 of 1.5M steps, hung 6 h). `_over_once`
        # True also stops a drill policy re-gripping the can in the bin to
        # collect W_OVER_BIN_BONUS; `_last_contact` in the past means it was
        # let go before tick 0, not "carried" for DROP_DEBOUNCE ticks.
        self._last_contact = -DROP_DEBOUNCE
        self._over_once = True
        self._released = True
        self._retracting = True          # the fold, from the first tick
        self._stowed_ok = True
        self._past_wp = at_wp            # a waypoint start is already on leg 2
        self._prev_tuck_err = None
        self._prev_can_w = None
        self._fix = self._seen_w = None
        self._fix_t = -1e9
        # WHAT THE STOW LEAVES IN THE TARGET SLOTS. At a real handoff the
        # front camera's last fix on the object is still fresh: MEASURED
        # (afd698), slots 24-26 arrived at |z| 2256/1817/232 under a fold
        # trained with them always zero, and it went home 0/38 — 36/38 with
        # them zeroed. So the drill starts with that fix, aged at random, and
        # it goes stale on its own (the bin is behind the front camera).
        _cw = np.array(self.data.xpos[self.can_body][:2], float)
        self._fix = _cw + self.rng.normal(0.0, 0.01, 2)
        self._fix_t = float(self.data.time) - float(
            self.rng.uniform(0.0, STALE_S))
        self._vel = np.zeros(2)
        self._pending = []
        self._att = self._fatt = None
        self._att_t = self._fatt_t = -1e9
        self._pending_att = []
        self._pending_fatt = []
        self._next_det_t = 0.0
        self._next_arm_t = 0.0
        self.last_action = np.zeros(moss.NUM_ACTIONS, np.float32)
        self.prev_action = np.zeros(moss.NUM_ACTIONS, np.float32)
        self._look_into_bin()
        self._prev_stow = self._stow_gap()
        self._prev_gap = self._reward_gap()
        self._prev_lift = 0.0
        self._start_z = float(self.data.xpos[self.can_body][2])
        self._can_spawn_w = np.array(self.data.xpos[self.can_body][:2], float)
        self._prev_align = None
        self._prev_upright = None
        self._closed_once = False
        self._hold_streak = 0
        self._sense()
        return self._obs(), {}

    def _seat_can(self, spread: float, tilt: float) -> bool:
        """Put the can between the open pads and close on it.

        On the FLOOR, at the grasp pose, which is where a grip that holds
        gets made. Two measurements shaped this. Stepping the jaw command
        straight to the grasp width shoved the can aside — the contacts at
        reset were `palm` and `pad_right` only, never `pad_left`, so the can
        was pinned against one side rather than held between the pads — so
        the command is ramped and the coupled fingers converge together,
        which seats it 11 times in 12. And seating it in MID-AIR does not
        work at all: at the lift pose the can falls through the closing jaws
        for 0 of 12, and holding it there while they shut, then letting go,
        made the floor rungs worse too (5 of 12), because a can pinned in
        place cannot settle into the pads. The grip the pick policy hands
        over is made against the floor, and so is this one.
        """
        tcp = np.array(self.data.site_xpos[self.tcp_site], float)
        off = self.rng.uniform(-spread, spread, 2)
        # x and y from the tcp; z from the FLOOR. Seating the can at the
        # tcp's own height buried it: the tcp sits at z 0.0485 and a can
        # resting on the floor has its centre at its half height, so a
        # dead-centre placement put it ~9 mm into the ground and the contact
        # spat it back out — 0 seated grips in 20, while a WIDER random
        # offset seated better because the z component of it was lifting the
        # can clear. Diagnosing that as "the centre grip is hard" would have
        # built the ladder upside down.
        # NOTE (2026-09-25): this places the object standing on the FLOOR at
        # its own half-height, which is a CAN-shaped convenience — a 330 ml
        # can's centre happens to sit near the grasp height. MEASURED across
        # six shapes it leaves a CARD 45 mm below the jaws and a TALL can
        # 385 mm away, not in the gripper at all, and delivery follows exactly
        # (card 0/11, tall 1/3). Placing it AT the tool point instead does not
        # help on its own: it then falls through the 0.6 s SEAT_SETTLE before
        # the pads close. A real fix seats it without gravity, or closes the
        # jaw first. Until then the stow leg's ceiling on varied litter is set
        # here, not by the policy.
        self.data.qpos[self.can_qadr:self.can_qadr + 3] = [
            tcp[0] + off[0], tcp[1] + off[1], self.prop.half_height]
        ax = self.rng.normal(size=3)
        ax /= float(np.linalg.norm(ax)) or 1.0
        ang = float(self.rng.uniform(-tilt, tilt))
        self.data.qpos[self.can_qadr + 3:self.can_qadr + 7] = [
            math.cos(ang / 2), *(ax * math.sin(ang / 2))]
        self.data.qvel[self.can_dadr:self.can_dadr + 6] = 0.0
        open_m = self.arm_cmd[moss.GRIPPER_JOINT]
        n = int(SEAT_SETTLE_S / moss.GRASP_PHYSICS_DT)
        for k in range(n):
            f = min(1.0, 2.0 * (k + 1) / n)     # shut by halfway, then settle
            self.arm_cmd[moss.GRIPPER_JOINT] = (
                open_m + f * (moss.GRASP_JAW_CTRL_M - open_m))
            self.driver.set_arm(self.arm_cmd)
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)
        return self._gripped_now()

    def _seat_and_carry(self, spread: float, tilt: float) -> bool:
        """Seat the object and carry it to this rung's start pose; True only
        if it is still in the jaws there. Up to SEAT_TRIES attempts."""
        for attempt in range(SEAT_TRIES):
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.driver.spawn(self.data, 0.0, 0.0, 0.0)
            self.arm_cmd = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
            self.arm_cmd[moss.GRIPPER_JOINT] = 0.041
            self.driver.set_arm(self.arm_cmd)
            self._place_clutter()
            for _ in range(int(DEPLOY_SETTLE_S / moss.GRASP_PHYSICS_DT)):
                self.driver.step(self.data)
                mujoco.mj_step(self.model, self.data)
            # Last try takes the nominal centred grip rather than hanging.
            s = 0.0 if attempt == SEAT_TRIES - 1 else spread
            t = 0.0 if attempt == SEAT_TRIES - 1 else tilt
            if not self._seat_can(s, t):
                continue
            if not self._raise_to(moss.LIFT_POSE):
                continue
            start = STOW_RUNG_START[self.stow_rung]
            if start == "lift":
                return True
            # Carry it as far along the route as this rung says, so that what
            # is left is the piece being learned.
            if not self._raise_to(STOW_HIGH, 2.0):
                continue
            if start == "high":
                return True
            if self._raise_to(STOW_TURNED, 4.0):
                return True
        return False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        spread, tilt = STOW_RUNG_GRIP[self.stow_rung]
        # A fresh shape BEFORE the seating attempts: the seat is what puts
        # this object in the jaws, so it has to know which object it is.
        self._maybe_new_prop()
        if self.retract_drill:
            return self._reset_retract_drill()
        self.seat_redraws = 0
        self.seat_fallback = False
        while True:
            ok = self._seat_and_carry(spread, tilt)
            if ok or not self.valid_start:
                break
            if self.seat_redraws >= VALID_START_REDRAWS:
                if self.seat_fallback:
                    break
                # Six shapes in a row would not seat: the reference can, which
                # seats 60/60, so the episode still starts with it IN the jaws.
                self.seat_fallback = True
                self.prop = GRASP_PROPS[DEFAULT_PROP]
                self._bind_model()
                if self._clutter:
                    self._clutter_poses = self._settle_clutter_once()
                    self._bind_model()
                continue
            self.seat_redraws += 1
            self._maybe_new_prop()
        #: False = this episode begins with the object NOT in the jaws.
        self.start_held = bool(ok)
        mujoco.mj_forward(self.model, self.data)
        self._look_into_bin()
        self.step_count = 0
        self._start_z = float(self.data.xpos[self.can_body][2])
        self._prev_gap = self._gap()
        self._last_contact = 0
        self._released_at: int | None = None
        self._release_err: float | None = None
        #: The retract phase: set once the can is delivered, so the episode
        #: continues until the arm is folded home rather than ending with it
        #: extended over the bin.
        self._retracting = False
        self._stowed_ok = False
        self._past_wp = False
        self._prev_tuck_err = None
        self._over_once = False
        self._prev_stow = self._stow_gap()
        self._prev_lift = 0.0
        self._closed_once = True          # it starts closed, by construction
        self._released = False
        self._fix = None
        self._seen_w = None
        self._vel = np.zeros(2)
        self._fix_t = -1e9
        self._pending = []
        self._next_det_t = float(self.data.time)
        self._sense()
        self.last_action[:] = 0.0
        self.prev_action[:] = 0.0
        return self._obs(), {}

    # ---------------------------------------------------------------- step

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self.prev_action = self.last_action.copy()
        self.last_action = a
        for i, j in enumerate(moss.ARM_JOINTS):
            lo, hi = self.model.joint(j).range
            if self.cmd_leash > 0.0:
                q = float(self.data.qpos[self.model.joint(j).qposadr[0]])
                lo = max(lo, q - self.cmd_leash)
                hi = min(hi, q + self.cmd_leash)
            self.arm_cmd[j] = float(np.clip(
                self.arm_cmd[j] + a[i] * ARM_DELTA_RAD, lo, hi))
        lo, hi = self.model.joint(moss.GRIPPER_JOINT).range
        self.arm_cmd[moss.GRIPPER_JOINT] = float(np.clip(
            self.arm_cmd[moss.GRIPPER_JOINT]
            + a[moss.ACT_GRIPPER.start] * JAW_DELTA_M, lo, hi))
        self.driver.set_arm(self.arm_cmd)
        # THE BASE STAYS PUT. The contract gives every MOSS policy (vx, wz)
        # and this task simply has no use for them: the bin is bolted to the
        # robot, so driving translates the can and the bin together and
        # cannot bring one closer to the other. All it can do is drag a
        # carried can into things — watched in the lab, the rover spends the
        # episode shuffling about with a can in its jaws, which is exploration
        # spent on a dimension that cannot pay. `W_BASE_EFFORT` taxed it but
        # did not forbid it. Zeroed here rather than removed from the action
        # vector, because the vector is the shared 8-action contract and the
        # deploy side must keep its shape; the policy simply learns these two
        # slots do nothing in this state, which is also true on the robot.
        vx = wz = 0.0
        for _ in range(DECIMATION):
            self.driver.set_cmd(vx, wz, self.data.time)
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self._predict()
        self._sense()
        self._sense_arm()

        gap = self._stow_gap()
        carrying = self._carrying()
        # Progress-pay on the can, not on the gripper: an arm that swings
        # over the bin having left the can behind has made no progress.
        rew = W_STOW * (self._prev_stow - gap)
        self._prev_stow = gap
        rew += W_TIME
        if not carrying:
            rew += W_EMPTY
        if carrying and not self._over_once and self._tcp_over_bin():
            self._over_once = True
            rew += W_OVER_BIN_BONUS
        rew += W_ACTION_RATE * float(np.abs(a - self.prev_action).sum())
        rew += W_BASE_EFFORT * (abs(float(a[moss.ACT_BASE.start]))
                                + abs(float(a[moss.ACT_BASE.start + 1])))

        # THE CAN HAS TO LAND BEFORE THIS IS JUDGED. Once the jaws open the
        # episode keeps running until the can stops moving, and only then is
        # it a delivery or a drop.
        if not carrying and self._released_at is None:
            self._released_at = self.step_count
            cb = self._can_base()
            self._release_err = float(math.hypot(cb[0] - self._drop[0],
                                                 cb[1] - self._drop[1]))
        speed = float(np.linalg.norm(
            self.data.qvel[self.can_dadr:self.can_dadr + 3]))
        settled = dropped = False
        # ...but once the arm is folding home the can is already delivered:
        # re-running the settle test paid STOW_BONUS again every tick and
        # re-terminated the episode, so the retract phase lasted exactly one
        # step and the arm never moved.
        if self._released_at is not None and not self._retracting:
            waited = self.step_count - self._released_at
            # A CAN LET GO OVER THE BIN IS STATIONARY BEFORE IT IS ANYWHERE.
            # The rest test alone fired as early as 6 ticks (0.24 s) after
            # release, while the can was still ABOVE the 0.261 m rim and had
            # barely begun to fall — it had been held still, so its speed was
            # near zero for reasons that have nothing to do with landing.
            # MEASURED: that scored a delivery in mid-air as a drop, which is
            # the lab's "it went in the basket and counted a fall". So the
            # rest test only counts once the can is BELOW the rim; a can
            # resting above it is either falling or perched, and either way
            # the settle clock decides it.
            at_rest = waited > 5 and speed < CAN_AT_REST_MPS
            below_rim = float(self._can_base()[2]) < moss.BIN_RIM_Z
            if (at_rest and below_rim) or waited >= RELEASE_SETTLE_STEPS:
                if self._in_bin():
                    settled = True
                    rew += self._stow_bonus()
                else:
                    dropped = True
                    rew += DROP_PENALTY
        # THE RETRACT PHASE. With `W_RETRACT` on, a settled can is not the
        # end of the episode: the arm still has to come home, and it is paid
        # per unit of joint angle folded toward the tuck pose.
        if self.retract > 0.0 and settled and not self._retracting:
            self._retracting = True
            self._stowed_ok = True
            settled = False                      # keep going: fold up first
        if self._retracting:
            # **A TWO-LEG OBJECTIVE, because a one-leg one was unearnable.**
            # Paying progress straight at the tuck pose asks the policy to
            # reduce a distance it cannot monotonically reduce: the bin sits
            # between the two poses, so the arm jams 0.65 rad short even with
            # the command held there for 600 steps. That is why 3M steps
            # plateaued at 2.66 rad and why MORE pay made it worse — the
            # reward was not weak, it was pointing through a wall.
            #
            # `moss.RETRACT_WAYPOINT` connects 14/14 real post-delivery poses
            # to the tuck by collision-free segments, so each LEG is a
            # distance the arm can actually close. Same move as the spawn
            # ladder: make the objective reachable, then let RL optimise it.
            _all = np.asarray(moss.tuck_pose(), float)
            _idx = [moss.ARM_JOINTS.index(j) for j in RETRACT_JOINTS]
            tuck = _all[_idx]
            cur = np.asarray([self.data.qpos[self.model.joint(j).qposadr[0]]
                              for j in RETRACT_JOINTS], float)
            if self.retract_staged and not self._past_wp:
                _wp = np.asarray(moss.RETRACT_WAYPOINT, float)[_idx]
                if float(np.abs(cur - _wp).max()) < RETRACT_WP_TOL_RAD:
                    self._past_wp = True
                    self._prev_tuck_err = None     # new leg, new baseline
                    rew += self.retract_wp_bonus
            _goal = (tuck if (self._past_wp or not self.retract_staged)
                     else np.asarray(moss.RETRACT_WAYPOINT, float)[_idx])
            err = float(np.abs(cur - _goal).sum())
            if self._prev_tuck_err is not None:
                rew += self.retract * (self._prev_tuck_err - err)
            self._prev_tuck_err = err
            if self.bin_scrape > 0.0:
                for _c in range(self.data.ncon):
                    _con = self.data.contact[_c]
                    _n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                            _con.geom1) or ""
                    _n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                            _con.geom2) or ""
                    if ({_n1, _n2} & self._arm_geoms) and any(
                            "bin" in _n for _n in (_n1, _n2)):
                        rew -= self.bin_scrape
                        break
            # NOT FAST: the command is capped at 0.75 rad/s, so anything past
            # the rated speed is momentum or gravity — charged per joint.
            if self.overspeed > 0.0:
                for _j in moss.ARM_JOINTS[:5]:
                    _v = abs(float(self.data.qvel[
                        self.model.joint(_j).dofadr[0]]))
                    if _v > ARM_RATED_RAD_S:
                        rew -= self.overspeed * (_v - ARM_RATED_RAD_S)
            if float(np.abs(cur - tuck).max()) < RETRACT_TOL_RAD:
                settled = True                   # delivered AND stowed
                rew += self.home_bonus
        terminated = bool(settled or dropped)
        # A release near the end of the episode must still be allowed to
        # LAND. Truncating on the step count alone cut the settle short and
        # scored a delivery as neither a success nor a drop, which teaches
        # the policy that letting go late is free. Bounded by
        # `RELEASE_SETTLE_STEPS`, so this can overrun by at most that.
        truncated = (self.step_count >= self.max_steps
                     and self._released_at is None)
        # ...and the FOLD gets its own clock. An arm that cannot get home is a
        # failed stow, not an infinite episode.
        if (self._retracting and self._released_at is not None
                and self.step_count - self._released_at > RETRACT_MAX_STEPS):
            truncated = True
        return (self._obs(), float(rew), terminated, truncated,
                {"stow_gap": gap, "carrying": carrying,
                 "stowed": settled, "dropped": dropped,
                 "success": settled})
