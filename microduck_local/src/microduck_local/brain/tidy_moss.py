"""The tidy brain for MOSS: drive to a can, scoop it, stow it in its own bin.

    search ──▶ approach ──▶ deploy ──▶ creep ──▶ close ──▶ lift ──▶ stow ──▶ release ──▶ tuck ─┐
      ▲        (tucked,     (jaws      (the      (grip)   (up     (over its   (open)    (fold) │
      │         arm inside   open at    base                0.25   own bin)                    │
      │         the shell)   can        drives                m)                               │
      └───────────────────── height)    the can ─────────────────────────────────────────────  ┘
                                        IN)

**A sibling of `brain/tidy_arm.py`, not a subclass** — the repo's own rule for
a whole brain whose vocabulary differs. Three things are structurally
different from MARS's, and each one REMOVES machinery rather than adding it:

* **The bin rides on the robot.** MARS drives to a basket, servos onto its
  rim and places over it; `tidy_arm` has `carry`, `deliver` and `place`
  states for that leg and reads the basket's rim height out of the scenario.
  MOSS carries its bin behind the arm, so the delivery target is a FIXED arm
  pose (`moss.DROP_POSE`) and the whole leg collapses into one swing. There
  is nothing to re-find, nothing to line up, and nothing to drop short of.
* **No run-time IK.** The grasp point is fixed in the robot's own frame, so
  the four poses are constants measured once (`robots/moss.py`) and what
  varies is where the BASE is — which this brain was steering anyway.
* **The base does the last centimetres, not the arm.** MEASURED, and it is
  the whole reason this brain is shaped the way it is: his palm sits ~60 mm
  up the approach axis and his can is 115 mm tall, so an arm descending onto
  a standing can lands on its lid and stops 35 mm short; and swinging the arm
  out of the tuck with a can in front sweeps it 20 cm away inside 0.4 s. So
  the jaws go to can height while the can is still 0.55 m off, and then the
  robot drives it in. A litter-picker's own order.

**What it is worth, MEASURED, and the two numbers disagree on purpose.**

* SCRIPTED, in MOSS's own scene with the can placed by hand
  (`scripts/probe_moss_grasp.py`'s sibling): **5/5 lifted, 5/5 stowed**.
* THIS BRAIN, in `moss-yard` off its own camera, 300 s, seeds 0/1/2:
  **2/9 cans stowed** (1, 1, 0), from 11 grasp attempts.

The gap is the brain's, not the arm's, and the failure is always the same:
the can is knocked aside or toppled out of line during the creep instead of
entering the jaws, so the machine closes on air and stows nothing. A 25 mm
belief error at the moment of closing (measured, below) against a grasp band
of +-30 mm is most of it. Worth quoting as 0.67 cans per 5 minutes, and worth
nobody's confidence until the creep is re-measured.

The can TOPPLES as it enters the jaws and is carried lying down; nothing here
grasps an upright can.

**Untested on hardware**, like everything else about this body.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..robots import moss
from ..robots import moss_env as ME
from .runtime import REGISTRY, Brain, Intent, Senses  # noqa: F401

#: The run whose `policy.onnx` is the shipped pickup skill.
#:
#: **A TIE BROKEN ON PROVENANCE, not a measured win** (2026-09-24). This was
#: `moss-pick-v1` — an RL leg whose chain ran through the lab but whose final
#: fine-tune was a CLI run. `teach-moss_pick-5e9df7` is the same leg
#: continued through `/teach` at rung 2 for 500k, so it is lab-produced end
#: to end, which is what this workspace asks of a trained leg.
#:
#: They cannot be told apart on the number that matters. Cans delivered over
#: 48 seeds of `moss-yard`, 144 cans an arm:
#:
#:     moss-pick-v1             46/144  (32%)
#:     teach-moss_pick-5e9df7   39/144  (27%)      0.93 se apart
#:
#: and the gap SHRANK as power went up — 1.6 se at 72 cans, 0.93 at 144 —
#: which is what noise does. At this n the smallest detectable difference is
#: about 15 points and the observed one is 5; separating them honestly needs
#: roughly 1300 cans an arm, which is not worth buying for a leg that is not
#: the bottleneck. On instant picks they are exactly tied, 45/48 each.
#:
#: The incumbent keeps the better point estimate, so this is reversible by
#: design: put `moss-pick-v1` back here and nothing else changes. What is NOT
#: a reason to prefer either is `5e9df7`'s better sustained-grip score (81%
#: against 62%) — that is the criterion it was trained against, and the same
#: criterion measured WORSE delivery under optimisation, so crediting it
#: would be crediting the proxy that failed.
#:
#: 2026-09-26: SWITCHED to `teach-moss_pick-ad9876`, the six-shape pick, once
#: the brain could run it as it trained (wrist-camera slots, base locked —
#: read off its run.json by `_flags`) and hand it objects where it trained
#: (per-object range, size-relative carry squeeze). moss-yard, 12 seeds x
#: 300 s, cans + block + squat + ball: ad9876 38/72, 5e9df7 37/72 (paired
#: +0.08 +- 0.42 per seed — a tie on the total); ad9876 delivers blocks 6/12
#: and balls 5/12 against 1/12 and 2/12, 5e9df7 squat cans 12/12 against 7/12.
#: Chosen for the shapes. Reversible: put "teach-moss_pick-5e9df7" back.
#: 2026-09-27: -> `teach-moss_pick-478dad`, ad9876 fine-tuned for objects
#: CLOSE IN (spawn 0.22-0.34 m) on the reward ad9876 actually trained under
#: (`gap_from_tcp` — unrecorded until now; three fine-tunes without it
#: learned to DRAG objects in). Deep picks in its env 81/78/90% at
#: 0.22-0.28/0.28-0.34/0.36-0.42 m against ad9876's 65/75/93%, card close in
#: 8/11 against 2/11, no dragging (-5.9 cm). moss-yard, 6 seeds x 300 s,
#: handover 0.41 m: 19/48 against 16/48. Reversible: "teach-moss_pick-ad9876".
#: 2026-09-28: -> the pick that sees its own grip (MOSS_PICK_GRIP, stage 6 of
#: 3d2aa6: from scratch, the depth fix in the tool frame, base free, 15 s to
#: re-grasp, yard handover starts, deep grips). moss-yard 48 seeds with the
#: small-object grasp fix: 358 v 281 for 478dad (paired +1.60 +- 0.27 per
#: seed, +1.67 / +1.54 per half); held-out yard states in the env 121/150 v
#: 86. Reversible: "teach-moss_pick-478dad".
SHIPPED_RUN = "teach-moss_pick_grip-3d2aa6-s6"
#: One trained policy per LEG of the loop, each under `moss.CONTRACT_ID` —
#: the three envs share one 32-slot observation and one 8-action layout, so a
#: session is interchangeable and only the state that drives it differs. A leg
#: with no run on disk keeps its scripted behaviour, which is what every one
#: of them had before there was anything trained to replace it.
SHIPPED_RUNS: dict[str, str] = {
    "pick": SHIPPED_RUN,
    "approach": "moss-approach-v1",
    "stow": "moss-stow-v1",
    # The learned FOLD (2026-09-26): out of the bin and home, from wherever
    # this brain's tuck state really begins — after a delivery, a failed
    # stow or a failed pick (trained on 92 such poses from moss-yard seeds
    # 10-21). MEASURED in the yard, seeds 0-2, 22 tucks: home 19/22, arm-bin
    # contact 1.2%, the fold's own worst joint speed 1.56 rad/s — against the
    # scripted tuck's 21/22, 4.0% and 11.5-11.9 rad/s on EVERY tuck.
    "fold": "teach-moss_fold-81875e",
}
#: Per-leg overrides, for A/Bing a fresh run against the shipped one.
POLICY_ENV_VARS: dict[str, str] = {
    "pick": "MICRODUCK_MOSS_POLICY",
    "approach": "MICRODUCK_MOSS_APPROACH_POLICY",
    "stow": "MICRODUCK_MOSS_STOW_POLICY",
    "fold": "MICRODUCK_MOSS_FOLD_POLICY",
}
#: Where each leg's policy expects the arm to BE when it takes over — the
#: pose its env resets to. Getting this wrong is not a detail: the pickup
#: policy handed control in the wrong pose scored 0/3 in the room against
#: 4/12 in its own env, which is what `deploy`'s arrival test exists for.
POLICY_START_POSE: dict[str, tuple[float, ...]] = {
    "pick": moss.GRASP_POSE,
    "approach": moss.tuck_pose(),
    "stow": moss.GRASP_POSE,
}


@dataclass(frozen=True)
class TidyMossParams:
    #: Where the can must be, in the base frame, before the arm comes out.
    #: Far enough that the deploy sweep misses it (`moss.DEPLOY_STANDOFF_M`).
    deploy_at: float = moss.DEPLOY_STANDOFF_M
    #: Where it must be for the jaws to be around it.
    grasp_at: float = moss.GRASP_STANDOFF_M
    #: Approach and creep speeds, m/s. The creep is slow because it is the
    #: motion that puts a 18 g can between two pads.
    approach_mps: float = 0.30
    creep_mps: float = 0.08
    #: Yaw gain on the bearing, per state. The creep's is lower: a hard turn
    #: at 0.08 m/s sweeps the jaws sideways THROUGH the can.
    approach_kp: float = 2.0
    creep_kp: float = 1.2
    #: Turn rate while searching, rad/s.
    search_wz: float = 0.8
    #: STEER ROUND THE LITTER IT IS NOT GOING FOR: how far ahead to look, m.
    #: 0 = off.
    #:
    #: MEASURED (2026-09-29, `scripts/probe_moss_blindspot.py`, 178 contacts
    #: over eight 300 s seeds): of everything the rover's driving surfaces
    #: touch, only **12% was ever inside its own target gate** — it is hitting
    #: things it is not going for, which `probe_moss_whom.py` had already put
    #: at 95% of contact time. And it is not blind to them: between **26% and
    #: 60% are in `ObjectMemory` AT THE MOMENT OF CONTACT** (the range is the
    #: match radius, 0.10 to 0.20 m), 54-86% at some point in the three
    #: seconds before. It knows and drives through them anyway, because
    #: nothing in the drive path reads the memory as an OBSTACLE —
    #: `_patrol_step` reads it only to choose a goal, and `approach` is an
    #: open-loop heading at `approach_kp` toward the fix.
    #:
    #: So this needs no new sensing, which is the whole argument for it: the
    #: two camera changes that suggest themselves were both measured first and
    #: neither reaches the geometry. Pitching the head down takes "in frame
    #: before contact" from 20% to 25% on the driving contacts, because the
    #: binding limit is LATERAL — 0.156 m of setback and an 87 deg lens cover
    #: a floor strip 0.09 m wide at x = 0.25 against tracks 0.42 m wide
    #: (`scripts/probe_moss_camera_cover.py`). The wrist camera at rest aims
    #: out to the robot's left and covers none of the zone. The position this
    #: acts on was recorded when the object WAS comfortably in view.
    #:
    #: **ON, and it is a measured YES — the first one in this line of work.**
    #: 64 paired 300 s seeds, 0.45 against off (2026-09-29), 56 of them after
    #: setting aside the runs where a prop wedges against the hull (below):
    #:
    #:     contact s, DRIVING states  5.30 -> 1.11 median   -2.86 +- 0.89
    #:                                36 seeds down, 20 up, sign p = 0.044
    #:     contact s, all states      4.99 -> 2.89 median   -2.23 +- 1.05
    #:     distinct props touched                           -0.3  +- 0.2
    #:     binned @180 s                                    +0.14 +- 0.15
    #:     binned @300 s                                    +0.09 +- 0.10
    #:
    #: Contact time while driving is roughly halved and **the mission is not
    #: paid for it** — binned moves slightly UP at both horizons, including
    #: the 180 s one that caught the arm rate cap costing -1.50/seed where
    #: 300 s read free. 0.45 m was chosen before the battery and never tuned
    #: against it.
    #:
    #: **What it does NOT fix: the pin.** In 4 of 64 seeds a prop ends up
    #: wedged against the hull and stays there for 200 s of a 300 s run —
    #: 4 of 64 in BOTH arms, the same seeds' worth either way, so this is a
    #: separate failure and steering earlier neither causes nor prevents it.
    #: The longest unbroken contact is likewise unmoved (-0.18 +- 0.12). What
    #: this buys is fewer and shorter glancing contacts, not a rescue.
    bystander_avoid_m: float = 0.45
    #: The corridor is the track width (`moss.HALF_WIDTH_M`, 0.212) plus this.
    bystander_margin_m: float = 0.03
    #: Peak yaw bias away from it, rad/s — full scale is `search_wz`, 0.8.
    bystander_wz: float = 0.8
    #: ...and how much of the forward speed to give up at full urgency. A
    #: diff-drive cannot sidestep: the nose has to come round, and at
    #: `approach_mps` 0.30 there are two seconds of corridor to do it in.
    bystander_slow: float = 0.5
    #: How long each arm move is given before the next state, in seconds.
    #: The servo is his (kp 70 / kv 3), and these are what the scripted probe
    #: used; a shorter deploy drops the can, a shorter stow drops it outside.
    deploy_s: float = 2.0
    #: How close every arm joint has to be to `moss.GRASP_POSE` before the
    #: learned skill takes over, and how long to wait before giving up on it.
    deploy_tol_rad: float = 0.05
    deploy_timeout_s: float = 6.0
    #: Hand the can to the pickup policy only once it is inside the box that
    #: policy trained in (`moss.PICK_HANDOVER_BOX`), nudging the base until
    #: it is. Set False to hand over on arm arrival alone, which is what this
    #: did before 2026-09-24 and what put 72% of handovers out of band.
    band_handover: bool = True
    #: WITH THE BASE LOCKED, hand over nearer: close in until the object is
    #: at most this far (base frame), not merely inside the box's far edge
    #: (0.47). The arm reaches 0.457 m at floor height (measured, joint-space
    #: sweep), so a pick that cannot drive (ad9876 trained base-locked) was
    #: handed objects at the edge of its reach and nudged them along instead
    #: of grasping. 0.41 is the middle of the box it trained on (0.36-0.47).
    band_far_locked_m: float = 0.41
    #: SMALL THINGS, STRAIGHT DOWN (2026-09-27). The jaws can point straight
    #: down at floor height from 0.10 to 0.38 m (joint sweep; 9 deg at 0.40,
    #: 18 at 0.42), and the pick USES that close in: at 0.24-0.30 m it shuts
    #: on a card at 4 deg from vertical (7/7 picked), a block at 5 (12/12) —
    #: at 0.38-0.42 m everything is taken 20-27 deg off vertical, card 5/7.
    #: Big objects stay at `band_far_locked_m`: from a close handover a can or
    #: tall can was lost on the lift. An object smaller than
    #: `small_object_m` (detected size) is handed over at `band_far_small_m`.
    size_aware_handover: bool = True
    small_object_m: float = 0.08
    #: A SCRIPTED TOP-DOWN PINCH for small objects (`robots/moss_pinch`):
    #: look with the wrist depth camera, hover over the object, roll the
    #: wrist to it, descend straight, close slowly, then the usual lift. The
    #: learned pick shoves a half-gram butt away before both pads close (22
    #: of 30 attempts in the pick env; 4/30 picked); this pinch picked 24/30
    #: with the jaws along the butt and 11/30 across it (the pads stand 8 mm
    #: apart fully shut), with the camera's noise. Objects detected under
    #: `pinch_size_m`; anything it cannot reach jaws-down falls back to the
    #: learned pick. `pinch_grip`: "along" | "across" the object's long side.
    #: ON since 2026-09-28, for objects under 3.5 cm (butts, caps), fast:
    #: 96 moss-yard seeds 838 v 784 in the bin (paired +0.56 +- 0.15 per seed,
    #: +0.71 / +0.33 / +0.50 / +0.71 per quarter); butts kept 63 v 16, caps
    #: 82 v 60. Under 8 cm it also took the cans (345 v 390 on 48); under 4.5
    #: cm +0.26 +- 0.17 (blocks and paper do as well with the learned pick,
    #: and a pinch costs seconds); at the first, slower timings (0.5 / 1.2 /
    #: 1.0 s) 369 v 390; faster still (0.15 / 0.5 / 0.4 s) no better.
    pinch_small: bool = True
    pinch_size_m: float = 0.035
    #: ...OR lying FLATTER than this, however wide it is (0 = off). The width
    #: gate alone is the wrong dimension for a card: `moss-yard`'s `card0` is
    #: 60 x 40 x 4 mm — FLATTER than the butt the pinch was built for (8 mm)
    #: and lower (centre 2 mm v 4 mm) — but reads 0.030-0.060 wide, so it fell
    #: to the learned pick, which grips at `GRASP_HEIGHT_M` = 50 mm with a
    #: 62 mm gap: a can's numbers, 38 mm above it. MEASURED on one seed: 19 of
    #: 21 carries never lifted it past 13 mm; the jaws brushed it along the
    #: floor once a second. Height is recoverable from the head camera, which
    #: is level with the base: `CAMERA_POS[2] + range * sin(elevation)`,
    #: median error <= 5 mm over 11 props, separating card/butt/cap at
    #: 0.004-0.007 from block/paper/ball/squat at 0.020+ and the cans at 0.058.
    pinch_flat_m: float = 0.015
    #: A flat object WIDER than this is a CARD rather than a butt: it wants
    #: the jaws on its long side and the pads pressed below the clearance.
    #: MEASURED within pick range, where each prop's apparent size is all but
    #: exact (p25 = median = p75): cap 0.015, butt 0.030, block 0.040, paper
    #: 0.044, ball/squat 0.050, **card 0.060**, cans 0.115. 0.050 sits in the
    #: gap. Read as a MEDIAN (`_wide_target`), never one tick.
    #: MEASURED by IK'ing onto `card0` at 30 poses: with its short side across
    #: the jaws it never lifted (0/15); with the long side across them and the
    #: pads low it lifted 9/9 when centred. The butt is the other way round —
    #: `pinch_grip` stays "along" for it — which is why this is per object and
    #: not a new global.
    pinch_wide_m: float = 0.050
    pinch_grip: str = "along"
    pinch_look_s: float = 0.25
    pinch_hover_m: float = 0.05
    pinch_move_s: float = 0.7
    pinch_close_s: float = 0.6
    pinch_floor_m: float = 0.0015
    #: ...and for a FLAT object, which the usual clearance cannot reach: it
    #: puts the pads 1.5 mm off the floor, leaving an 8 mm butt 6.5 mm of pad
    #: overlap (the cigarette works) and a 4 mm card 2.5 mm. Negative, so the
    #: pads press down: the grasp window measured on `card0` is a pad midpoint
    #: near 12 mm, and 18 mm and above never held it.
    pinch_flat_floor_m: float = -0.014
    #: ...from THIS close. The handover leaves small objects 0.43-0.52 m off
    #: (measured), where the wrist camera cannot see them and the jaws cannot
    #: point down at them (reach jaws-down ends ~0.38 m); the pinch first
    #: drives in on the head camera's fix, arm deployed, to here.
    pinch_range_m: float = 0.27
    #: after the close, straight up this far before the lift swings away
    pinch_rise_m: float = 0.06
    #: the stow's swing round to the bin (`stow_turn_s`) for something the
    #: PINCH picked up: most stow losses happen on that leg, and a pinched
    #: butt was lost there 58 times in 96 runs. 3.2 s (the usual is 2.4):
    #: 96 moss-yard seeds 878 v 838 in the bin (paired +0.42 +- 0.15, +0.46 /
    #: +0.38 per half); butts kept 43 v 26 on the first 48. Slowing the swing
    #: for EVERYTHING instead cut stow losses as much but cost cans (+0.25 +-
    #: 0.25). 4.0 s: +0.38 +- 0.25. None = the usual.
    pinch_turn_s: float | None = 3.2
    #: a head-camera fix further than this from the pinch's locked spot is
    #: another object
    pinch_lock_m: float = 0.06
    pinch_approach_s: float = 5.0
    band_far_small_m: float = 0.28
    #: How long to spend nudging into the band before handing over anyway. A
    #: cap, not a target: a can the base cannot line up is still worth an
    #: out-of-band attempt, because the alternative is no attempt at all.
    band_timeout_s: float = 3.0
    #: Closing speed while lining up, m/s. Forward only — arriving NEAR
    #: converts at 73% against 38% inside the trained box, so there is
    #: nothing here worth backing away from.
    #: 0.12, not 0.06 (2026-09-27). A time budget of moss-yard runs put
    #: deploy at 26% of every 300 s, and the arm was not what took it: it
    #: reaches the grasp pose in 1.2 s; the base then crept into the band at
    #: 0.06 m/s and 7 deploys in 20 ran to the 9 s cap. 48 seeds x 300 s with
    #: 478dad: 225 v 190 in the bin, paired +0.73 +- 0.29 per seed (+0.96 /
    #: +0.50 per half). 0.18 is no better (-0.12 +- 0.30 v 0.12), and cutting
    #: the band short instead (band_timeout_s 1.0) is worse (77 v 94).
    band_mps: float = 0.12
    close_s: float = 1.0
    lift_s: float = 1.8
    #: A GENTLER CARRY FOR THIN THINGS. A 4 mm card lying flat can only be
    #: pinched by its edges, and in moss-yard (2026-09-28) it sat still in the
    #: jaws, touching nothing but the pads, then left them in one tick at a
    #: jerk of the carry — the lift ending, a change of leg in the swing: 19
    #: card carries, 12 lost in the stow, 6 in the lift, 1 kept.
    #: `carry_ease`: every lift and stow leg eases in and out (smoothstep)
    #: instead of starting and stopping at full speed. ON since 2026-09-28:
    #: 48 seeds 382 v 358 in the bin (paired +0.50 +- 0.22, +0.54 / +0.46 per
    #: half); paper kept 37 v 26, cards lost in the lift 1 v 6. The card is
    #: still not delivered. Carrying small objects twice as SLOWLY instead
    #: lost (333 v 358; 337 with easing): the time costs more than it saves.
    carry_ease: bool = True
    stow_s: float = 3.4
    #: How long the LEARNED stow is given before the loop falls back to
    #: opening the jaws. Its env runs 20 s episodes and it delivers in a
    #: median of ~120 control steps, so this is generous on purpose.
    stow_policy_s: float = 20.0
    #: The three legs of the scripted route, in seconds. 0.6x the original
    #: 2.0 / 4.0 / 3.0 (2026-09-28): the 9 s stow was a clock, not the arm,
    #: and with the depth pick it was 24% of every run. moss-yard, 48 seeds x
    #: 300 s: 359 v 334 in the bin, paired +0.52 +- 0.21 per seed (+0.67 /
    #: +0.38 per half); objects lost during the stow 25% v 24%. 0.4x (3.6 s)
    #: loses more (31%) and gains less (+0.42 +- 0.31, 24 seeds).
    stow_high_s: float = 1.2
    stow_turn_s: float = 2.4
    stow_down_s: float = 1.8
    #: The stow releases when the arm has ARRIVED over the bin, not when its
    #: ramp's clock runs out — the same distinction `deploy` already makes,
    #: and for the same reason. MEASURED 2026-09-24: releasing on the clock
    #: opened the jaws with the carried can still at x = +0.151 m in the base
    #: frame, out in FRONT, when the bin's mouth is x = -0.087. The ramp
    #: commands the pose; the arm is still travelling to it.
    #: COMMAND THE JAW EVERY TICK OF THE CARRY, at the measured hold rather
    #: than at whatever the pickup policy last said. `WorldRobot.set_arm` is
    #: absolute — "a brain that wants two joints held says both every tick" —
    #: and `lift`/`stow` send only the five arm joints, so the gripper keeps
    #: the creep's final command for the whole carry. MEASURED 2026-09-24:
    #: that command is 0.0 m, a full squeeze, because the learned pickup
    #: closes hard to grip and the scripted `close` state (the only place
    #: `GRASP_JAW_CTRL_M` is ever commanded) is BYPASSED when a pick policy
    #: is loaded. Traced through a carry the jaws crush from 36.8 mm to
    #: 28.8 mm on a 66 mm can and it squirts out: 47 of 79 grip losses happen
    #: in `lift`, with the can still touching the floor.
    hold_jaw: bool = True
    #: Start the lift's ramp from where the arm IS, not from the pickup
    #: policy's last COMMAND. `_ramp`'s own docstring promises the former and
    #: the code did the latter; a servo under load lags, so the first tick of
    #: the lift was a step of that lag. MEASURED over 63 handovers: worst
    #: joint 0.062 rad median, but `shoulder_lift` reaches 0.38 rad at p95 and
    #: 0.60 rad at worst, and a step input is what the same docstring records
    #: as shearing the can out of the jaws.
    ramp_from_achieved: bool = True
    #: What to hold it at — `moss.GRASP_JAW_CTRL_M`, the gentlest setting the
    #: grasp sweep measured that still holds (6 mm of interference).
    carry_jaw_m: float = moss.GRASP_JAW_CTRL_M
    #: ...as INTERFERENCE, not a position: carry at the jaw's ACHIEVED
    #: position when the lift starts minus this. 27 mm is 6 mm of squeeze on
    #: a 66 mm can but WIDER than a 40 mm block or a 50 mm ball, so the lift
    #: opened the jaws on them — MEASURED in moss-yard (2026-09-26), blocks
    #: lifted 21 times and balls 16 and not one survived the lift.
    #: 10 mm, not 6 (2026-09-27): the carry's first swing (up to STOW_HIGH)
    #: rotates the wrist a long way and a close-in grip slid out there — at
    #: a 0.30 m handover 9 losses on that leg in 4 yard seeds, 4 at 10 mm, and
    #: objects in the bin 6 -> 13/32. At the default 0.41 m handover, 8 seeds:
    #: 33/64 against 28/64 (paired +0.62 +- 0.68 per seed — not resolved alone).
    carry_interference_m: float = 0.010
    carry_jaw_relative: bool = True
    #: ABANDON THE CARRY WHEN THE CAN IS GONE. The scripted stow never asked
    #: whether it was still holding anything: it ran all three ramps and
    #: opened the jaws over the bin regardless. MEASURED, only 31% of lifts
    #: still hold the can when the release fires, so about four stows in ten
    #: are nine seconds of theatre — in a 180 s run that is most of a minute
    #: spent posting nothing, while the can it dropped sits on the floor
    #: behind it. Watched in the lab, which is where this was spotted.
    #: STOP THE TRACKS ONCE THE ARM CAN REACH IT. The pickup policy drives
    #: the base flat out for the whole grab: measured over 29,165 creep ticks,
    #: it commands motion on 100% of them, |vx| saturated at the contract's
    #: 0.200 m/s and |wz| at a median 0.808 rad/s with 78% of ticks turning
    #: harder than 0.3 rad/s. That is not a policy converging on a can, it is
    #: a policy that was paid to close distance and never learned that the
    #: last 30 cm belong to the arm.
    #:
    #: It is also where the cans go: 158 of 434 can-touches in a run are the
    #: TRACK and the HULL, and a can knocked under the chassis is a can this
    #: loop then chases in circles. Spotted by a human watching the lab —
    #: "it gets stuck in the track and then it's spinning around trying to
    #: hit it... a lot of movement we don't really need from the tracks."
    #: RAMP THE TRANSITIONS, don't step them. `lift`, `stow` and `release`
    #: already ramp and are calm — max 2.00, 2.37 and 0.67 rad/s. `deploy`
    #: and `tuck` commanded their pose outright and are not: p95 9.47 and
    #: 7.94 rad/s, peaks of 12.9 and 13.5. An XL330's practical no-load speed
    #: is about 5.6 rad/s, so those peaks are not merely harsh, they are
    #: unreachable on the real arm — the sim only gets there because a
    #: position servo with no rate limit will slam. Spotted by a human
    #: watching the lab: "it's going really fast and kind of violent, I'm
    #: worried it might break the robot."
    deploy_ramp_s: float = 1.2
    #: The retract out of the bin, which is the one that can sweep delivered
    #: trash back out on its way past the rim.
    tuck_ramp_s: float = 1.5
    still_when_in_reach: bool = False
    #: Inside this range the base holds still and the ARM closes the gap.
    #: `moss.GRASP_STANDOFF_M` is 0.26 — where the grasp pose actually
    #: reaches — so this is that plus a little, not a guess at the reach.
    reach_stop_m: float = 0.30
    #: How much of the policy's commanded twist survives inside that range.
    #: MEASURED, and the answer is that ALL of it is needed by this policy:
    #:
    #:     scale   0.00   0.25   0.50   0.75   1.00
    #:     cans    9/36  14/36   8/36  15/36  22/36
    #:
    #: So the thrashing is not excess the deploy side can trim. The policy
    #: was trained with the base in its action vector and steers with it;
    #: taking it away puts it off its own manifold. `still_when_in_reach` is
    #: therefore OFF, and the fix belongs in the ENV
    #: (`moss_env.PICK_BASE_REACH_SCALE`), where a policy can learn that the
    #: last 30 cm are the arm's job instead of being told so at deploy time.
    reach_twist_scale: float = 1.0
    stow_abort_on_drop: bool = True
    #: Check the grip during LIFT as well. MEASURED (moss-yard, seeds 0-2):
    #: 8 of 10 drops happen at the very start of the carry — the grip never
    #: really held — and the check only ran in `stow`, so the brain noticed
    #: 2.8 s late. `lift_drop_grace_s` lets the jaw settle first.
    lift_abort_on_drop: bool = True
    #: LIFT ONLY ON A CONTACT THAT IS HAPPENING NOW. The creep->lift gate
    #: reads the DEBOUNCED grip (held within `grip_debounce_s`), so a contact
    #: that had just ended still read as held and the lift began with the
    #: pads touching nothing: MEASURED in moss-yard, 8 of 10 empty-handed
    #: carries had the pads on the object 0% of ticks from the first.
    lift_needs_live_contact: bool = True
    #: THE WRIST CAMERA SAYS WHETHER IT IS IN THE HAND, not just the pads.
    #: Both pads can touch an object still on the floor (pinched at an edge,
    #: pushed), and `Senses.holding` then says "held": MEASURED in moss-yard,
    #: a tall can carried 7.2 s with the pads on it 89% of ticks and in the
    #: hand 40%. The depth wrist camera gives the object's HEIGHT
    #: (`Senses.target_obs["z"]`). Calibrated on 6 yard seeds: in LIFT from
    #: 0.6 s a held object has risen (p5 -1.2 cm) and one left behind has not
    #: (median -14 cm); in STOW a held object never reads below 0.238 m.
    #: OFF: MEASURED three ways (6 yard seeds each) it cut short MORE real
    #: carries than it saved (21/30, 14/23 against 8/20 without) and put
    #: fewer objects in the bin (12/48 against 16/48) — first because the
    #: wrist target was a binned object, then with that fixed still.
    #: Kept, with `target_obs["z"]`, for a better-calibrated attempt.
    camera_hold_check: bool = False
    lift_rise_after_s: float = 0.6
    lift_rise_until_s: float = 0.9
    lift_min_rise_m: float = -0.02
    stow_min_z_m: float = 0.15
    lift_drop_grace_s: float = 0.3
    #: AFTER A DROP, straight home at this joint rate (rad/s), not the
    #: learned fold: the arm is not over the bin, so there is no bin to route
    #: around, and the fold's waypoint route took 7-12 s every time. 0.75 is
    #: the rate a command moves a joint at (0.03 rad per 25 Hz tick) — the
    #: same cap the fold obeys, so nothing about this is fast.
    drop_tuck_rate: float = 0.75
    drop_tuck_max_s: float = 6.0
    #: BACK OFF AFTER A DROP so the thing it dropped can be SEEN again, m.
    #:
    #: What a drop leaves behind, MEASURED over four 300 s moss-yard seeds
    #: (2026-09-29): **20 drops, one about every 60 s**, and every one lands
    #: directly ahead — base-frame x from 0.001 to 0.479 m, median 0.293, never
    #: more than 0.27 m off the centre line. That is the problem, because
    #: `min_x` (0.30 m) drops any detection nearer than itself: it exists to
    #: reject the robot's own bin and things under the chassis, and it cannot
    #: tell those from litter the robot has just put there. **11 of the 20 —
    #: 55% — land inside that gate**, so the brain does not merely
    #: deprioritise the object it dropped, it cannot see it at all. It then
    #: searches, drives forward, and the tracks are what finds it.
    #:
    #: Reversing is the cheapest fix because the gate is a RANGE test: the
    #: object does not have to move, the robot does. Clearing `min_x` + 5 cm
    #: needs a median of 0.112 m and at worst 0.349 m; 0.18 m covers the median
    #: and most of the spread without a long blind reverse.
    #:
    #: **OFF, and it is a measured NO.** Eight paired 300 s seeds, back-off at
    #: 0.18 m against none (2026-09-29): binned 88 -> 87 (-0.12 +- 0.12),
    #: drops 39 -> 49, and the thing it was built for — run-overs — 16 -> 25
    #: at +1.12 +- 2.52, which is an instrument that cannot resolve the change
    #: rather than an improvement. More drops is the mechanism working as
    #: designed (the robot gets another go instead of abandoning) and it buys
    #: nothing, because of the measurement below.
    #:
    #: **WHY IT CANNOT WORK: 13 of 16 run-overs are on objects the robot had
    #: NOT just dropped** (8 seeds; the 3 that follow a drop do so a median
    #: 18 s later). The robot mostly drives into litter it never picked up —
    #: it approached, the object passed inside `min_x`, and from there the
    #: brain is blind to it. Backing off after a DROP addresses a fifth of the
    #: problem at best, and this is the whole reason the number above is 0:
    #: the story "it drops it, then runs it over" is intuitive, was mine as
    #: well as the reporter's, and the drop-to-run-over link is not there.
    #:
    #: 0 = off (the old behaviour: fold, then search straight over it).
    drop_back_m: float = 0.0
    #: Gently — this is a blind move. `approach_mps` is 0.30.
    drop_back_mps: float = 0.12
    #: ...and only if the MAP says the space behind is clear, within this
    #: radius of the point the back bumper would reach. The rover's half width
    #: is 0.212 m (`moss.REAR_EXTENT_M`'s note), so this covers it. MOSS's
    #: scanner looks FORWARD: reversing is blind, and `RoomMap.blocked` — which
    #: counts both mapped cells and `felt` bumps, so a wall it has only ever
    #: touched still stops it — is the only thing that knows. UNKNOWN reads as
    #: clear, because the map is sparse early on; that is exactly why the
    #: reverse is bounded by `drop_back_m` instead of trusting the map alone.
    drop_back_clear_m: float = 0.25
    #: A reverse that has not covered its distance by now gives up and looks
    #: from where it is (a stall against something the map never had).
    drop_back_max_s: float = 5.0
    #: How long the grip has to read EMPTY before believing it. The pads lose
    #: and regain contact during the swing, so a single tick means nothing;
    #: `_gripped` is already debounced and this is on top of it.
    stow_drop_grace_s: float = 0.5
    stow_tol_rad: float = 0.12
    #: How long to keep holding the final pose while it settles before
    #: letting go anyway. A cap: a jam that never arrives still has to end,
    #: and dropping the can beside the bin beats carrying it forever.
    stow_settle_s: float = 3.0
    #: Let go from `moss_env.STOW_RELEASE_HIGH` (over the bin centre, 11 cm
    #: above the rim) instead of descending to `STOW_INSIDE`. ON, on the
    #: MECHANISM: at the end of each release in moss-yard (6 seeds), the
    #: descent left the can wedged in the open jaws 3 times in 12 — then the
    #: fold carried it off — and the high release 0 in 13. The end-of-run
    #: count moved 11 -> 13 (paired +1 0 -1 +1 +2 -1: inside the noise on its
    #: own); the fold still gets home (60/72 against 66/78) and over-speeds
    #: less (2 against 6). A first A/B that only counted objects at the end
    #: (14 v 13) could not see the wedge and was read as a null.
    release_high: bool = True
    #: How long a grip may READ as lost before the loop believes it. The
    #: pads break and remake contact constantly while the arm accelerates:
    #: traced in the room, `holding` flickered holds/released/holds/released
    #: inside 0.12 s right after a good grip, and read raw that ended the
    #: stow leg ONE TICK after it began (t=29.82 -> 29.84). The env has the
    #: same debounce on its own carry test for the same reason.
    grip_debounce_s: float = 0.25
    #: How long a grip has to HOLD before the pickup hands over. The loop
    #: used to leave on the first contact frame, and traced in the room that
    #: frame is routinely a flicker: `holds` at t=27.98, handed over at
    #: 28.00, `released` at 28.00. The scripted lift then raised an empty
    #: gripper and the stow policy — 9/12 into the bin in its own env — was
    #: handed nothing, 0 of 9 times. The pickup's OWN success test is
    #: stronger than one contact (held AND lifted clear), and this is the
    #: cheapest way to ask for the same thing from the room.
    grip_settle_s: float = 0.35
    #: DON'T LIFT A SHALLOW GRIP (2026-09-28). The wrist depth camera's grip
    #: fix (`Senses.target_obs["grip"]`, tool frame) says how deep the object
    #: sits in the jaws, and it tracks the truth to a few mm. Traced over 303
    #: carries in moss-yard: objects lost in the LIFT sat 4.2 cm from the tcp
    #: at lift start (median), kept ones 1.6 cm — the jaws had closed on an
    #: edge, and 12 cm cans fell 52 times in 119. |grip| > 3 cm flags 66/85
    #: lift losses and 28/163 good carries. While it reads shallow the pick
    #: policy keeps acting (it trained to re-grasp) instead of handing over;
    #: after `grip_depth_wait_s` it lifts anyway, so the gate cannot stall.
    #: It is also the pick env's OWN success test (`deep_grip_m` 0.035, which
    #: the shipped pick trained under) — the brain used to lift on any grip.
    #: moss-yard, 48 seeds x 300 s: 385 v 354 in the bin, paired +0.65 +-
    #: 0.17 (seeds 0-23 +0.71 +- 0.21; fresh 24-47 +0.58 +- 0.27, halves
    #: +0.08 / +1.08). 3.5 cm +0.38, 3 cm with no cap +0.67 (24 seeds).
    #: None turns it off.
    #: ONLY FOR OBJECTS BIG ENOUGH TO TWIST OUT (detected size >=
    #: `grip_depth_min_size_m`). A 0.6 g butt in a 7 N grip cannot slip under
    #: its own weight wherever it is held, and small objects READ shallow —
    #: |grip| median 3.5 cm, 58% over 3 cm — so the gate only made them wait
    #: out the timeout while the pick nudged them (butts kept 21 -> 4, found
    #: by the MOSS pick session). 48 seeds: large-only 392 v 385 for gating
    #: everything (+0.15 +- 0.13), v 354 ungated (+0.79 +- 0.15, halves
    #: +0.67 / +0.92).
    grip_depth_max_m: float | None = 0.030
    grip_depth_wait_s: float = 2.0
    grip_depth_min_size_m: float = 0.08
    #: How far apart the jaws must be for the grip to be a GRASP rather than
    #: a pinch. His can is 66 mm across, so jaws round it rest near 29 mm;
    #: MEASURED in the room, the pickup routinely ends with `holding` true at
    #: 7 mm, which is the jaws shut PAST the can with it wedged against the
    #: palm and the pads grazing it. That is a real contact and a useless
    #: grasp: handed on, the stow policy — which only ever trained on jaws at
    #: 27-29 mm — sees a finger position it has never seen and opens within
    #: 0.1 s, which is 0 of 9 cans in the room against 9/12 in its own env.
    min_grasp_m: float = 0.020
    #: ...but that is a CAN's number. A cap, block or card is thin enough
    #: that jaws really round it sit below 20 mm, so the rule read every
    #: grasp of one as a pinch and never lifted: in moss-yard (2026-09-27)
    #: pick attempts reached a lift for 0% of caps, paper and butts, 6% of
    #: cards and 10% of blocks, while the pick env, from the SAME handover
    #: states, picked caps 88%, blocks 79%, cards 43%. For an object the
    #: detector sizes under `small_object_m`, this is the floor instead.
    #: 0 since 2026-09-28: 48 moss-yard seeds, 281 v 225 in the bin (paired
    #: +1.17 +- 0.23 per seed, +0.96 / +1.38 per half); paper 0 -> 28 kept,
    #: blocks 28, caps 5, cards 2. Both pads on it is still required.
    min_grasp_small_m: float = 0.0
    #: How far a detection may sit from the can being tracked and still be
    #: believed to BE it. Wide enough for a fix that has drifted while the
    #: robot drove, narrow enough to reject a different can — the ones that
    #: caused this are 0.6 m and more away.
    retarget_gate_m: float = 0.30
    release_s: float = 1.8
    #: THE POP (asked on /sim 2026-09-28: "a jerky motion when it places the
    #: object ... that little pop"). The stow's first two waypoints hold
    #: `shoulder_lift` at -1.35, which is BEHIND the bin: 0.8 s into the
    #: "up" leg the upper arm lands on `bin_x1` at 17 N and stalls 0.22 rad
    #: short while its command keeps climbing, scrapes the wall at 18-26 N
    #: through the first half of the turn, then slides off the edge and
    #: snaps through the 0.22 rad at 2.3-2.8 rad/s (70 rad/s^2) — in every
    #: stow. A kinematic sweep of the path finds -1.10 contact-free end to
    #: end, the tool still 7 cm over the rim through the turn (0.335 m;
    #: 0.362 at -1.35). None keeps the old waypoints.
    #: MEASURED, moss-yard, 48 seeds x 15 min (bin floor fixed in both):
    #: stow peak joint acceleration 63.3 -> 2.8 rad/s^2 (median; p90 71.6 ->
    #: 2.8), arm on the bin 53.6 -> 0 N; objects truly in the bin 484 v 483
    #: (+0.02 +- 0.06), 449 v 447 at 5 min; stow time unchanged (5.40 s).
    stow_clear_lift: float | None = -1.10
    #: Open the jaws over this long at release instead of in one step (the
    #: step flicks the object sideways at 0.3-0.5 m/s as it goes). 0 = step.
    #: MEASURED OFF: 0.4 s left the flick where it was (0.34 v 0.36 m/s) and
    #: knocked 14 objects out of the bin against 6 over the same 48 seeds.
    release_open_s: float = 0.0
    #: THE REST POSE, AND HOW THE ARM GETS TO AND FROM IT (2026-09-28,
    #: `brain/moss_motion.py`). Asked on /sim: the arm "clips through the box
    #: going into the rest position" and "gets stuck as it sweeps out"; and
    #: "the wrist camera just points down at the track — point it outwards
    #: to scan". MEASURED, eight 7-minute yard runs, the arm's visible meshes
    #: against the bin and hull: the old rest pose (`moss.tuck_pose()`) has
    #: the gripper's mesh ON the bin's front wall, the sweep out (a straight
    #: ramp) clipped 50% of its time and stalled 92 s, and the learned
    #: approach pressed the rover at up to 292 N.
    #: `rest_pose` was searched (320k poses) for: >= 15 mm between the
    #: visible arm and the rover, nothing in the front camera's view, the
    #: whole arm INSIDE THE CIRCLE THE CHASSIS SWEEPS TURNING IN PLACE (0.21
    #: of the tracks' 0.222 m), high in front, and the most floor in the
    #: WRIST camera's view that the front camera cannot see. The circle is
    #: measured, not tidiness: the first pick reached 0.3 m out, and turning
    #: to search beside a wall it caught the wall, was dragged to the pan
    #: limit and pinned the robot there for two minutes. This one holds the
    #: wrist camera 0.30 m up looking out to the LEFT (bearing +100 deg),
    #: 24 deg below the horizon, >= 30 mm clear, and joins the grasp, release
    #: and lift poses by STRAIGHT clear segments. None keeps the old tuck.
    #: WHAT IT CAN ACTUALLY SEE, re-measured honestly 2026-09-28: 0.168 m^2
    #: of floor — in the cone, inside the detector's 0.60 m range, and past
    #: the 0.25 m self-reject — and ALL of it is floor the front camera cannot
    #: see (that camera covers 4.65 m^2 and none of this strip). The cone
    #: itself reaches 0.69 m^2, so three quarters of what it looks at is
    #: beyond the range the detector reports; the original search scored floor
    #: out to 1.5 m and paid 0.25 for the unreportable part, so it aimed too
    #: flat. Re-searching on reportable floor alone (40k poses + refinement)
    #: tops out at 0.279 m^2 — but every such pose aims 65 deg down, which is
    #: the "just pointed at the track" this pose was asked to stop doing, and
    #: the extra area buys nothing measurable (see `wrist_scan`). So the
    #: outward aim is kept deliberately, not by oversight.
    #: AND IT IS NOT WORTH RE-AIMING AT THE BLIND SPOT EITHER (2026-09-29).
    #: Asked whether off-to-the-side is really the best place: sideways is
    #: FORCED — in the front 60 deg only 25-74 poses of ~3600 are legal, killed
    #: by the turning circle (47-60%) and by the arm standing in the front
    #: camera's view (16-25%) — and front-left, at bearing +26..+60 deg, is
    #: exactly where the misses are (26,893 samples, 6 seeds). This pose
    #: catches 11.5% of them; a legal pose at +113 deg catches 17.0%. On 48
    #: paired seeds that pose put **454** in the bin at 5 min against this
    #: pose's **464**, and took 24-28% longer to clear the room (8 binned at
    #: 196 s against 158 s), because it reaches the release and lift poses only
    #: through a HUB and stalled 10.8 s a run against 5.2. A rest pose is a
    #: MOTION decision first and a sensing one second: route directness is the
    #: binding constraint, not blind-spot coverage.
    rest_pose: tuple[float, ...] | None = (-0.4472, -0.8663, 0.1402, 1.4805, -1.1788)
    #: Every move to and from rest (sweep out, fold home, straight home after
    #: a drop) by a route clear of the visible arm (`route_margin_m`), timed
    #: minimum-jerk so the busiest joint peaks at `motion_vmax`, on a leash
    #: of `motion_leash_rad`. Off: the old ramps and folds.
    plan_routes: bool = True
    route_margin_m: float = 0.010
    motion_vmax: float = 1.2
    motion_leash_rad: float = 0.10
    #: NO SCRIPTED BLEND MAY DEMAND MORE THAN THE JOINT'S RATED SPEED, rad/s.
    #:
    #: `MinJerkRoute` times its segments so the busiest joint peaks at
    #: `motion_vmax`; the pinch's blends and the carry's ramps did not — they
    #: ran a FIXED duration whatever distance the arm had to cover, so their
    #: peak rate was whatever the pose delta happened to be. MEASURED over
    #: four 300 s moss-yard seeds (`scripts/probe_moss_safety.py`), sampling
    #: every 2 ms physics step: `wrist_roll` reaches **8.4-10.0 rad/s**
    #: (480-574 deg/s) in `pinch/align`, from one-tick command steps of
    #: 0.166-0.200 rad, and the lift/stow ramps step 0.047-0.049 rad a tick.
    #: `align` is where the roll turns the jaws onto a card's long axis, which
    #: can be a quarter turn delivered in 0.42 s.
    #:
    #: Applied by TIMING the blend (duration = peak factor x delta / cap), not
    #: by clipping the command: these phases advance on their clock, so a
    #: clipped command would hand `descend` a roll that never arrived and
    #: grasp the card across its short axis again. 0 = off, for the A/B.
    #:
    #: **3.0, and the number was MEASURED, not reasoned.** The obvious choice
    #: was 1.5 — `robots/moss_env.ARM_RATED_RAD_S`, twice the policy's own
    #: 0.03 rad at 25 Hz — and at the 300 s horizon it looks free (binned
    #: -0.12 +- 0.23 per seed). It is not: at 180 s, where the room is still
    #: being cleared rather than already clear, 1.5 costs **-1.50 +- 0.60
    #: binned per seed** (71 -> 59 over eight paired seeds). A 5-minute score
    #: cannot see a slower robot, because it finishes either way. Swept at
    #: 180 s against no cap at all:
    #:
    #:     cap     binned (8 seeds)   worst step   worst stall   arm-on-bin
    #:     off             71            2.750 rad     3.68 s       211 N
    #:     6.0    -0.25 +- 0.45          1.660         3.00         205
    #:     3.0    -0.12 +- 0.30          0.281         1.57         127
    #:     1.5    -1.50 +- 0.60          0.449         0.62         118
    #:
    #: 3.0 takes almost all of the safety and none of the speed; 6.0 is barely
    #: a cap (1.66 rad in one tick is still a step input). 3.0 rad/s is 172
    #: deg/s, inside what an STS3215-class servo can turn unloaded, so it is
    #: not a rate the hardware would simply fail to follow — but nothing here
    #: has measured that servo, and that is the whole reason this is a knob.
    arm_rate_cap: float = 3.0
    #: ...AND A SLEW LIMITER ON THE EMITTED COMMAND, rad/s. 0 = off.
    #:
    #: Pacing the blends fixes the blends. It cannot fix a STATE TRANSITION,
    #: which is a step input by construction: `ramp_from_achieved` starts each
    #: leg at where the arm IS, and a servo under load lags its command, so
    #: the first tick of `lift` jumps by exactly that lag. MEASURED over eight
    #: 300 s moss-yard seeds with the blends already paced: the worst one-tick
    #: command step is still 0.122-0.449 rad, and every one of the eight is in
    #: `lift`. 0.449 rad in 20 ms is a 22 rad/s demand.
    #:
    #: This clips the command to the LAST COMMAND +- cap*dt, which is what a
    #: real servo bus would do anyway. It cannot silently stall a plan: every
    #: blend is paced under the same cap, so the limiter only ever has a
    #: transition's lag to pay off, and it pays it off in ceil(lag/cap*dt)
    #: ticks. The GRIPPER is deliberately outside it — its travel is 41 mm of
    #: slide on its own servo, and nothing here has measured that servo's
    #: rate, so capping it would be a number I made up.
    #:
    #: **OFF, and it is a measured NO — it trades one hazard for another.**
    #: Eight paired 300 s seeds against the paced arm (2026-09-29), mission
    #: score null (+0.12 +- 0.23 binned per seed). It does what it was written
    #: to do: the worst one-tick step falls 0.449 -> 0.060 rad, the arm's time
    #: on the floor 6661 -> 1979 substeps and the hardest floor contact
    #: 151 -> 118 N. But delaying the command leaves the arm arriving late,
    #: and the numbers that decide whether a servo survives got WORSE: the
    #: longest unbroken stall at the 2.2 N m clamp 0.99 -> 3.88 s, arm-on-bin
    #: contact 3310 -> 6243 substeps at 118 -> 190 N, and peak joint speed
    #: 6.9 -> 10.4 rad/s (gravity, not a command — the limiter does not hold
    #: the arm up). A 3.9 s stall is the exact failure this pass exists to
    #: remove, so this stays off and the transition step stays open: the fix
    #: belongs in where the lift's ramp STARTS, not in a clamp after it.
    arm_slew_cap: float = 0.0
    #: Drive up to a can with the arm HELD at rest (the scripted approach)
    #: instead of the learned approach, which drives AND moves the arm — it
    #: was adopted because the arm could not reach the old tuck, and in the
    #: room it pressed the arm into the hull and bin 72% of its time.
    approach_arm_rest: bool = True
    #: SCAN WITH THE WRIST CAMERA while the arm rests (search and approach):
    #: what it sees goes into the same object memory as the front camera's
    #: sightings, placed by the arm's own kinematics. The rest pose points it
    #: out to the left at floor the front camera cannot see — beside the
    #: robot, where small things drop out of the front camera's view as it
    #: drives past.
    #:
    #: ON, AND A NULL ON THE MISSION — both, honestly. 48 moss-yard seeds of
    #: 15 min, paired against the same arm motion without it: -0.04 ± 0.10
    #: objects in the bin at 5 min, -0.04 ± 0.07 at 15 min. (The first such
    #: battery came out IDENTICAL seed for seed, which is how we found that
    #: the world never sampled this detector at all — the /sim overlay drew
    #: its cone from the spec while `senses.arm_det` stayed None. `world/arena`
    #: samples it now and `tests/test_moss_motion` asks the world for a frame.)
    #: What it does deliver, instrumented over 3 seeds of 300 s: 92% of its
    #: detections survive the filters, it CREATES 1-5 memory entries per run
    #: against the front camera's 32-39, it is first to 1-5 of them by 2-94 s,
    #: and 0-3 per run are objects the front camera never sees at all. It
    #: cannot do more than that here because the arm only rests in
    #: search/approach for ~10% of a run, and because the yard saturates
    #: anyway (483-485 of 490 objects binned by 15 min either way). It is kept
    #: on because the information is real and free, not because it scored.
    #:
    #: MEASURED OFF — scanning from ANY arm pose (a joint-speed gate instead
    #: of the rest gate) trebles the points accepted (161-286 v 21-95) and
    #: LOSES picks (10/10/8 v 9/12/10). The extra entries are phantoms: 5-8
    #: per run are created during `lift`, where the camera is staring at the
    #: object IN THE JAWS and the kinematics place it on the floor 0.3 m away.
    #: The rest gate is what keeps the camera pointed at floor; it is not
    #: conservatism.
    wrist_scan: bool = True
    #: How close to `rest_pose` (every joint, rad) counts as resting.
    rest_tol_rad: float = 0.08
    tuck_s: float = 2.4
    #: THE LEARNED FOLD's budget. It takes ~190 control ticks (7.6 s) in its
    #: env; past this it hands on to search wherever the arm is.
    fold_policy_s: float = 12.0
    #: THE COMMAND LEASH for the fold, rad: each arm joint's goal is kept
    #: within this of its MEASURED position (goal = present +- leash, which
    #: is also what the brain writes on the robot). Without it a jaw pinned
    #: by the dropped object stores the advancing command and releases it at
    #: 10-14 rad/s when it slips free — learned fold and scripted alike. The
    #: fold trained under exactly this value (`moss_env.CMD_LEASH_RAD`).
    fold_leash_rad: float = 0.08
    #: THE FOLD ROUTE (2026-09-28). The learned fold runs at its 0.75 rad/s
    #: command cap the WHOLE way (7.8 s on every delivery in the yard, 29% of
    #: a 300 s run spent in `tuck`) — it is not slow, its route is long: it
    #: was paid to pass `moss.RETRACT_WAYPOINT`, 3.0 rad out and 3.2 rad
    #: back, from a release pose 2.0 rad from home. That waypoint was chosen
    #: to connect 14 DIFFERENT delivery poses; this brain releases from one
    #: (`fold_route_from`, 12 of 15 folds in four yard runs). A planner over
    #: the robot's own collision model found a waypoint from THAT pose whose
    #: route costs 2.0 rad of the slowest joint (the roll), 2.7 s at the same
    #: cap. Flown here as a straight, rate-capped, leashed script — the drop
    #: path's motion — when the arm starts within `fold_route_from_tol` of
    #: the release pose; any other start still gets the learned fold.
    #: MEASURED, moss-yard, 96 seeds: 876 v 848 in the bin (+0.29 +- 0.16),
    #: the fold 7.8 -> 2.5 s, 0 bin contacts on the route; at 1.2 rad/s
    #: +0.15 +- 0.15 and more contact, so the cap stays at 0.75. With
    #: `missed_pick_straight` 893 v 848 (+0.47 +- 0.15, halves +0.46/+0.48).
    fold_route: bool = True
    fold_route_from: tuple[float, ...] = (-1.875, -0.695, -0.264, 1.438, -0.05)
    fold_route_wp: tuple[float, ...] = (-0.5735, -1.138, 0.8424, 0.7053, 1.267)
    fold_route_from_tol: float = 0.25
    fold_route_rate: float = 0.75
    #: A MISSED PICK GOES STRAIGHT HOME, by the drop path. The learned fold
    #: trained from over-the-bin starts only; handed an arm out in FRONT (a
    #: creep or pinch that timed out) it swings back over the bin and scrapes
    #: it — 7 of 15 such folds in six yard runs, for up to the 12 s budget,
    #: against 0 of 107 folds from the release pose. With `fold_route`, 96
    #: yard seeds: 893 v 876 (+0.18 +- 0.10). Straight home still brushes the
    #: hull (~26 ticks a fold) — a planned route from the front is next.
    missed_pick_straight: bool = True
    #: THE ROOM MODEL (`brain/moss_search.py`). `object_memory`: remember
    #: every toy seen, and approach only a CONFIRMED one (two detections in
    #: the same place) — a phantom detection never recurs where it was.
    #: `patrol`: when a full turn shows nothing, drive to the nearest
    #: remembered object and look from close up, else drive the rim of the
    #: work area (`patrol_inset_m` in from its walls — sensed, below) turning
    #: to face the middle at each waypoint.
    #: MEASURED, moss-yard, 48 seeds x 15 min, objects truly in the bin:
    #: 486 v 469 (+0.35 +- 0.10, halves +0.46/+0.25); every seed cleared all
    #: ten grippable objects (39/48 before — the cap was left 11 times); 188
    #: phantom approaches -> 0; search 464 -> 258 s a run. At 5 min, the
    #: benchmark, 444 v 439 (+0.10 +- 0.17): the end game rarely starts by
    #: then, so nothing is lost and nothing gained there.
    object_memory: bool = True
    patrol: bool = True
    patrol_inset_m: float = 0.45
    patrol_mps: float = 0.15
    #: A leg that has not arrived in this long is abandoned.
    patrol_leg_s: float = 20.0
    #: How long to hold still and look on arrival.
    patrol_look_s: float = 1.0
    #: The detector finds a thing on EVERY frame out to this many times its
    #: size (its `w_full`, 2.81 deg at 640 px over 87 deg): the range a
    #: remembered object is looked at from, and inside which not seeing it
    #: means it is gone.
    see_range_per_size: float = 20.0
    #: How far out a look counts as having covered the floor, m.
    cover_m: float = 1.0
    #: THE WALLS FROM ITS OWN DEPTH (`moss_search.RoomMap`): the work area is
    #: the rectangle round what the RealSense's depth scan has found solid,
    #: once it has swept most of a turn. Off — or on a MOSS with no depth
    #: (`tof: null`) — the scenario's walls are handed to it instead, and
    #: the /sim map says "given".
    #: MEASURED, moss-yard, 48 seeds x 15 min against the given walls: the
    #: sensed rectangle matched the real walls in every run (worst error
    #: 0.000 m), ready at 128 s median (159 max) — before any rim leg is
    #: wanted; objects truly in the bin 482 v 486 (-0.08 +- 0.05, halves
    #: -0.08/-0.08), every grippable object binned in 47/48 v 48/48 and the
    #: difference is the card (delivered by luck 3 v 6 times) and one butt
    #: left by a corner jam; at 5 min 443 v 444. So: no measurable cost, and
    #: the walls are its own. Touch fired once (a corner inside the depth's
    #: 0.52 m floor).
    sense_walls: bool = True
    #: TOUCH, for what the depth cannot see (under its 75 mm slice, inside
    #: its 0.52 m floor): a patrol leg commanded forward at `stall_cmd_mps`
    #: or more that moves slower than `stall_mps` for `stall_s` has hit
    #: something. It is marked on the map `felt_ahead_m` in front of the
    #: base and the leg ends. [sim] `speed` is the body's true speed here;
    #: on the rover the tracks' encoders keep counting when they slip, so
    #: the stall has to come from motor current or the IMU.
    stall_s: float = 1.0
    stall_mps: float = 0.03
    stall_cmd_mps: float = 0.10
    felt_ahead_m: float = 0.22
    #: A rim waypoint within this of anything solid or felt is skipped.
    waypoint_clear_m: float = 0.25
    #: A detection older than this is not a fix any more.
    stale_s: float = 1.0
    #: Only cans IN FRONT are targets: a bearing past this is something the
    #: robot would have to turn for, and something nearer than `min_x` is
    #: too close for the arm to work on — it would unfold into the object.
    #: (This note used to say such a detection "is either under the chassis
    #: or the one already in the bin". MEASURED false: all of them are real
    #: litter ahead of the bumper, and the bin has its own test. See
    #: `near_min_x`, which is what that mistake cost.)
    #: What the detector models an upright can as: `Prop.radius()` is the
    #: largest half-extent, so for a 66 x 115 mm can it is 0.0575 m.
    can_radius_m: float = 0.0575
    #: Take range from the DETECTOR's per-object estimate (`range_est`), not
    #: from a can's radius. MEASURED in moss-yard (2026-09-26): with the can's
    #: radius, every non-can object read ~0.2 m FARTHER than it stood — block
    #: 0.52 m v 0.26 true, ball 0.53 v 0.30, squat 0.51 v 0.31, cans +-0.01 —
    #: so the robot drove 20 cm too close (the arm unfolded into the object:
    #: 16-30% of deploys timed out) and the pick was told to reach 20 cm past
    #: it (0 ball lifts, block never kept). `range_est` has been per-object
    #: since the arena passed each prop's own size to the detector; the can
    #: radius was the workaround for the class-constant bug that preceded it.
    range_from_detector: bool = True
    max_bearing: float = 1.05          # ~60 deg
    min_x: float = 0.30
    #: THE SAME FLOOR, FOR THE MEMORY ONLY, m — the one that is not a target
    #: test. `min_x` above gates what the robot may DRIVE AT; this gates what
    #: it is allowed to KNOW. They were one number, and that was the bug.
    #:
    #: MEASURED (2026-09-29, `scripts/probe_moss_dropped_dets.py`, three 300 s
    #: moss-yard seeds): of 38 686 `toy` detections the colour camera makes,
    #: `min_x` throws away **7.6%, 10.9% and 13.3%** — between one and two
    #: thousand sightings per five minutes, and the verdict below is about
    #: which objects those turn out to be. Every one sits at x 0.187 to
    #: 0.300, so **100% are ahead of the front bumper**
    #: (`moss.FRONT_EXTENT_M`, 0.186): not one is the chassis, and the bin —
    #: the other thing the comment on `min_x` claims to be rejecting — has its
    #: own test on the next line which fired **zero** times in all three runs.
    #: Nor is any of this a depth problem: `range_from_detector` ranges each
    #: object by its apparent width in the COLOUR image, and the depth row
    #: only builds the obstacle map. The camera sees them. The brain binned
    #: the detection.
    #:
    #: And `_toys_in_view` feeds `_remember` as well as `_see`, so such an
    #: object was not merely passed over as a target — it never entered
    #: `ObjectMemory` at all. This splits the two floors so it can: the
    #: targeting floor is untouched (lowering `min_x` itself was tried and
    #: doubled the drops — the jaws cannot work at 0.2 m), while the memory
    #: keeps the sighting, and the patrol's `choose` phase can then route
    #: back to a viewing distance for it, which is the "pull back and
    #: re-prioritise" behaviour already written but with nothing to act on.
    #:
    #: **OFF, and it is a measured NO — the percentages above are real and
    #: they are not what they look like.** Eight paired 300 s seeds, 0.186
    #: against 0.30 (2026-09-29): contact seconds, longest unbroken contact,
    #: distinct props touched, peak force, binned at 180 s and binned at
    #: 300 s all came back **BYTE-IDENTICAL on every seed**. Not a dead hook
    #: this time — the reason is in what those thousand rejected detections
    #: are OF. Broken down by brain state, they land in `creep`, `pinch`,
    #: `lift`, `stow`, `tuck`, `deploy` and `release`: the arm is out and the
    #: close thing in frame is the object in the jaws or the one being
    #: reached for. Only **15/26/0 per run happen in `search`** and 4/16/13
    #: in `approach`, and `_remember` (which runs per detector FRAME, not per
    #: 50 Hz tick) sees a fifth of those — so the split admits **2, 7 and 2
    #: detections per 300 s run**. Counting what a filter discards is not
    #: counting what it costs.
    #:
    #: The complement says the same thing from the other side: over 300 s of
    #: driving, litter is inside the 0.19-0.30 m band at all for **41, 62 and
    #: 55 ticks — about one second**, and every instance is an upright CAN.
    #: Never a cap or a card, because at `CAMERA_VFOV_DEG` 62 deg and
    #: `CAMERA_POS` 0.075 m up, a flat object clears the bottom of the frame
    #: only past **x = 0.281 m**. `min_x` at 0.30 was already sitting on the
    #: camera's own floor horizon; there is no blind band worth opening.
    #:
    #: Kept rather than deleted because the number is the finding and this is
    #: where the next person looks for it — the same reason `drop_back_m` and
    #: `arm_slew_cap` are still here at 0. `moss.FRONT_EXTENT_M` (0.186) is
    #: the value to set for the split; 0.30 is the old behaviour and, on this
    #: measurement, identical behaviour.
    #:
    #: Either way it lifts only while the arm is stowed (`search`/`approach`).
    #: Deployed, the gripper and whatever is in it fill the near frame, and a
    #: carried object at 0.2 m would be written down as litter lying wherever
    #: the rover happened to be standing — the same reason `_remember`
    #: already gates its `looked()` pruning on those two states.
    near_min_x: float = 0.30
    #: How many times a can may be attempted before it is written off, and
    #: how close two fixes have to be to count as the same can. A can against
    #: a wall cannot be approached — the robot would have to get behind it —
    #: and without this the machine retries it for the rest of the run.
    #: `brain/tidy_arm.py` carries the same pair for the same reason.
    max_retries: int = 3
    same_can_m: float = 0.22
    #: How long a written-off can stays written off. Not forever: the robot
    #: moves, and a can that was unreachable from one side may not be from
    #: another — which is the one advantage a mobile base has over an arm.
    give_up_s: float = 45.0
    #: Give up creeping after this long and go back to searching — a can that
    #: never arrives is a can that was knocked away, and standing still with
    #: the jaws open is the one state this machine could sit in forever.
    creep_timeout_s: float = 12.0
    #: How long the LEARNED skill gets before the machine takes back over and
    #: re-deploys. It is the trained episode length (`moss_env.EPISODE_S`),
    #: not a bigger number: past its own horizon the policy is extrapolating,
    #: and a fresh deploy puts it back at the state its episodes start from.
    #: Measured over a 12 s window the gripper action drifted OPEN (+0.129
    #: against +0.015 in the env at matched can distance) and the jaw
    #: oscillated 29 -> 34 -> 29 mm without ever committing.
    learned_window_s: float = 8.0
    #: HAND THE PICK A CLEAN START (2026-09-26). Diffing the pick's first
    #: observation in moss-yard against its env's showed the last-action slots
    #: carrying the previous leg's action (up to 680 sd off the env's zeros)
    #: and a stale pick command carried between attempts. Entering `creep`
    #: now zeroes the last action and re-seeds the command from `GRASP_POSE`,
    #: as the pick env's reset does — the same fix the stow and fold legs got.
    pick_clean_start: bool = True
    #: Run the pick at its trained 25 Hz, holding the command between control
    #: ticks (per world tick it integrates at twice its trained rate — arm
    #: and jaw at double speed). An earlier measurement (8 -> 4 of 18 cans,
    #: ad9876) could not separate the two. ON since 2026-09-27, found by
    #: diffing the yard's pick against the env's from one snapshot (the yard
    #: stepped every 0.02 s): 478dad in moss-yard, 24 seeds x 300 s, 99 v 71
    #: in the bin (paired +1.17 +- 0.36 per seed, +1.17 in each half), lift
    #: losses 39 -> 24, cans kept 20 -> 45.
    pick_at_control_hz: bool = True


def _shipped_policy() -> str | None:
    """The trained pickup this repo ships, if it is on disk.

    Found by PATH rather than by an environment variable, so the skill is
    live the moment the lab starts: a variable set in one shell does not
    survive `restart.sh` launching the backend in its own session, and a
    brain that silently falls back to the scripted pickup looks identical
    from the outside. `MICRODUCK_MOSS_POLICY` still overrides, for A/Bing a
    new run against this one.
    """
    here = Path(__file__).resolve().parents[3] / "runs" / SHIPPED_RUN / "policy.onnx"
    return str(here) if here.is_file() else None


def _shipped_policies() -> dict[str, str]:
    """Every leg's trained policy that is actually on disk, by task.

    Same PATH-not-variable rule as `_shipped_policy`, for the same reason: a
    variable exported in one shell does not survive `restart.sh` launching
    the backend in its own session, and a brain that quietly falls back to
    the scripted leg looks identical from the outside.
    """
    runs = Path(__file__).resolve().parents[3] / "runs"
    found: dict[str, str] = {}
    for task, run in SHIPPED_RUNS.items():
        override = os.environ.get(POLICY_ENV_VARS[task])
        cand = Path(override) if override else runs / run / "policy.onnx"
        if cand.is_file():
            found[task] = str(cand)
    return found


def _wrist_floor_band(pc, Rc, z: float = 0.03) -> tuple[float, float]:
    """The near and far radius, m, of the floor the WRIST camera can actually
    report, measured from the camera's own ground point.

    The /sim map used to draw this as a full wedge from the camera out to the
    detector's 0.60 m — which claims the floor right beside the robot, where
    nothing is in the camera's VERTICAL view: aimed 24 deg down from 0.30 m
    up, its nearest floor is 0.2-0.3 m away. Overstating a sensor's reach on
    the map is how the wrist camera went 48 seeds looking like it worked while
    the world never sampled it at all, so the map takes these from the pose.

    The near edge is where the bottom of the vertical fan meets the floor, the
    far edge the nearer of the top of the fan and the range sphere's cut.
    """
    vf = math.radians(moss.ARM_CAMERA_VFOV_DEG / 2)
    R = float(moss.ARM_CAMERA_MAX_RANGE_M)
    h = float(pc[2]) - z
    out = []
    for b in (-vf, vf):
        v = np.array([math.cos(b), 0.0, math.sin(b)])
        w = Rc @ v
        if w[2] >= -1e-6 or h <= 0.0:       # that edge never reaches the floor
            out.append(math.sqrt(max(R * R - h * h, 0.0)))
            continue
        s = h / -float(w[2])
        out.append(float(np.hypot(*(w[:2] * s))) if s <= R
                   else math.sqrt(max(R * R - h * h, 0.0)))
    near, far = min(out), max(out)
    far = min(far, math.sqrt(max(R * R - h * h, 0.0)))
    return round(max(0.0, near), 3), round(max(near, far), 3)


class TidyMoss:
    """MOSS's litter loop. `brain/runtime.Brain`'s protocol."""

    kind = "tidy_moss"

    def __init__(self, params: TidyMossParams | None = None,
                 policy: str | None = None, **_ignored):
        self.p = params or TidyMossParams()
        #: The LEARNED half. `policy` is a path to an ONNX exported by
        #: `export-walk` under `moss.CONTRACT_ID`; given one, the `creep` and
        #: `close` states hand control to it and this class keeps doing what
        #: a state machine is good at — searching, approaching, stowing,
        #: folding. That is the split this whole brain exists to make: the
        #: contact-rich 30 cm is learned, the mission logic is code.
        self._sess = None
        self._in = None
        paths = _shipped_policies()
        if policy:                      # an explicit path is the PICKUP skill
            paths["pick"] = policy
        self._policy_paths = paths
        self._sessions: dict[str, tuple[object, str]] = {}
        if paths:
            import onnxruntime as ort
            for task, path in paths.items():
                sess = ort.InferenceSession(
                    path, providers=["CPUExecutionProvider"])
                self._sessions[task] = (sess, sess.get_inputs()[0].name)
        #: What each loaded leg TRAINED with, read off its own run.json: the
        #: extra observation slots it expects filled, and whether its base was
        #: locked. `teach-moss_pick-ad9876` trained with attitude, proximity,
        #: size AND a locked base; handed zeros and a live base it put 4
        #: objects in the bin over 6 yard seeds (2026-09-26).
        self._flags: dict[str, dict] = {}
        for task, path in paths.items():
            fl = dict(ME.obs_env_kwargs(path))
            fl["base_lock"] = bool(ME.eval_env_kwargs(path).get("base_lock"))
            fl["episode_s"] = ME.episode_s_of(path)
            msg = ME.wrist_mount_mismatch(path, f"the {task} leg ({path})")
            if msg:
                warnings.warn(msg)
            self._flags[task] = fl
        # `_sess`/`_in` stay the PICKUP session: the creep and close states
        # read them directly and every existing test names them.
        if "pick" in self._sessions:
            self._sess, self._in = self._sessions["pick"]
        #: THE LEARNED STOW IS OFF BY DEFAULT, and the reason is a number.
        #: It scores 9/12 into the bin in its own env and 0 of 9 cans in the
        #: room, against 2 of 9 for the scripted path it would replace — so
        #: wiring it in makes the mission loop WORSE, and shipping it because
        #: it looks finished would be shipping a regression.
        #:
        #: What is known about the gap, all measured: the observation now
        #: matches its env at the handover (last_action, target_base and
        #: target_seen all zero-diff, joint_pos within 0.03; only joint_vel
        #: still differs, 0.65 max, because the env reads `qvel` and a brain
        #: finite-differences). The physics match (identical can, identical
        #: pad priority and friction, the room carries the corrected bin and
        #: arm geometry). What does NOT match is the GRASP: the pickup leaves
        #: the jaws at 7 mm in the room — shut past a 66 mm can, wedged
        #: against the palm — where the stow env always begins at 27-29 mm.
        #: Six separate fixes narrowed that without moving the room number,
        #: which says the remaining cause is not yet understood rather than
        #: nearly fixed.
        #:
        #: Set MICRODUCK_MOSS_LEARNED_STOW=1 to use it anyway, which is what
        #: the next person debugging this will want.
        self._use_learned_stow = (
            "stow" in self._sessions
            and os.environ.get("MICRODUCK_MOSS_LEARNED_STOW") == "1")
        self._policy_path = paths.get("pick")
        self.reset()

    def reset(self) -> None:
        self._act = np.zeros(moss.NUM_ACTIONS, np.float32)
        self._prev_arm: dict[str, float] | None = None
        self._prev_t = 0.0
        self._prev_yaw: float | None = None
        self._policy_cmd: dict[str, float] | None = None
        self._carry_from: dict[str, float] | None = None
        self._track_fix = None
        self._track_t = -1e9
        self._prev_grip_cmd = 0.041
        self._grip_seen_t = -1e9
        self._last_obs: np.ndarray | None = None
        self._last_policy_t = -1e9
        self._held_twist = (0.0, 0.0, 0.0)
        self._leg_from: dict[str, float] | None = None
        #: When the carry first read empty, for `stow_drop_grace_s`.
        self._empty_since: float | None = None
        #: Snapshot for `_ramp_from_here`, cleared on every transition.
        self._ramp_from: dict[str, float] | None = None
        self._leg_i: int = -1
        self._fold_seeded = False
        self._dropped = False
        self._drop_from: dict[str, float] | None = None
        #: The room model (`object_memory` / `patrol`): built on first use,
        #: the work area from the world's walls (`runtime.attach_world`).
        self.world = None
        self._mem = None
        self._cov = None
        self._patrol = None
        self._room = None
        self._lidar_t: float | None = None
        #: "sensed" (the depth's walls), "given" (the scenario's), or None.
        self._area_src: str | None = None
        self._srch: dict | None = None
        self._det_t: float | None = None
        self._mem_target: tuple[float, float] | None = None
        self._mem_picked = 0
        #: The scripted fold's route (poses), seeded on its first tick.
        self._route: list | None = None
        self._route_tried = False
        self._carry_jaw: float | None = None
        self._fix_size: float | None = None
        #: The head camera's height readings for the CURRENT target, and where
        #: that target is — see `_see`. `_fix_height` is the latest, the gate
        #: uses the median.
        self._fix_height: float | None = None
        self._fix_heights: list[float] = []
        self._fix_sizes: list[float] = []
        self._fix_heights_at: tuple[float, float] | None = None
        #: Was this target gated as FLAT? The pinch descends deeper for it and
        #: grips across its long side.
        self._target_flat = False
        self._kin = None
        #: `moss_motion`: the clearance model (built on first use) and the
        #: current state's planned move (None: not planned yet; False: no
        #: clear route this time, the old branch runs).
        self._clr = None
        self._mv = None
        self._adet_t: float | None = None
        #: The wrist camera in the ODOM frame WHILE IT SCANS, for the map:
        #: (x, y, heading of its view, near radius, far radius), or None. The
        #: radii are the band of floor it can really report
        #: (`_wrist_floor_band`) — the map must not draw reach the sensor
        #: hasn't got.
        self._wcam: tuple[float, float, float, float, float] | None = None
        #: Odometry when a `back_off` began — `drop_back_m` is measured from it.
        self._back_from: tuple[float, float] | None = None
        #: Something was dropped and the tuck that follows owes a back-off.
        #: A LATCH rather than a branch, because `_dropped` is consumed by the
        #: fold it selects and there are four ways out of `tuck`.
        self._back_due: bool = False
        #: The last arm command this brain actually EMITTED, and when
        #: (`_remember_arm`). Kept unconditionally — `_slew` is off by
        #: default and `_ramp_from_here` needs the same answer.
        self._last_arm: dict[str, float] | None = None
        self._last_arm_t: float = -1.0
        self._pinch: dict | None = None
        self._release_from: float | None = None
        self._pinch_twist = (0.0, 0.0, 0.0)
        self._pinched = False
        self._lift_z0: float | None = None
        self._low_since: float | None = None
        self._grip_from_t = 1e9
        self._depth_block_t: float | None = None
        self._depth_last_t = -1e9
        self._grip_ticks = 0
        self.state = "search"
        self._t0 = 0.0                      # when the current state began
        self._fix: tuple[float, float] | None = None   # last can fix, base frame
        self._fix_t = -1e9
        self._odom: tuple[float, float, float] | None = None
        self.picked = 0
        #: Cans that have been tried and not collected: world-frame (x, y, t)
        #: of the attempt. World frame, because the robot moves and the point
        #: of writing one off is that it is still there when we come back.
        self._written_off: list[tuple[float, float, float]] = []
        self._attempts = 0
        self._target_world: tuple[float, float] | None = None

    # ------------------------------------------------------------- sensing

    def _see(self, senses: Senses) -> tuple[float, float] | None:
        """The nearest toy, as (x, y) in the BASE frame.

        The detector reports a bearing and a range from the CAMERA, which sits
        `moss.CAMERA_POS` ahead of the base origin — his CAD number, so the
        offset is added here rather than absorbed into the standoff.
        """
        out = []
        for r, x, y, size, z in self._toys_in_view(senses.det):
            if self._is_written_off(x, y):
                continue                       # tried, could not collect
            if self.p.object_memory and self._mem is not None:
                w = self._world((x, y))
                e = (None if w is None else
                     self._mem.near(*w, self._mem.gate_m + self._mem.gate_per_m * r))
                if e is None or not self._mem.confirmed(e):
                    continue                   # seen once: maybe a phantom
            out.append((r, x, y, size, z))
        if not out:
            return None
        _r, x, y, size, z = min(out)
        self._fix_size = float(size)
        # PER TARGET. A single list across every object mixed a 22 mm cube's
        # readings into a 4 mm card's and pinched the cube 8 times, missing
        # every one: one tick's height is noisy (card0 reads -0.016..0.020
        # over a run) while its MEDIAN is 0.004.
        w_now = self._world((x, y))
        if w_now is not None:
            w_was = self._fix_heights_at
            if w_was is None or math.hypot(w_now[0] - w_was[0],
                                           w_now[1] - w_was[1]) > 0.08:
                self._fix_heights = []
                self._fix_sizes = []
            self._fix_heights_at = w_now
        self._fix_height = float(z)
        self._fix_heights.append(float(z))
        self._fix_sizes.append(float(size))
        if len(self._fix_heights) > 25:
            del self._fix_heights[0]
        if len(self._fix_sizes) > 25:
            del self._fix_sizes[0]
        return (x, y)

    def _toys_in_view(self, frame, min_x: float | None = None
                      ) -> list[tuple[float, float, float, float, float]]:
        """(range, x, y, size) of each toy detection, BASE frame, that could
        be a target: in front, not under the robot, not in its own bin.

        `min_x` overrides the range floor. It is a parameter because the two
        callers want different ones: `_see` picks something to drive at and
        must not pick what it cannot reach, while `_remember` only writes down
        what is there — see `near_min_x`, which is the whole of that argument.
        """
        if frame is None or not frame.detections:
            return []
        floor = self.p.min_x if min_x is None else float(min_x)
        out = []
        for det in frame.detections:
            if det.cls != "toy":
                continue
            r = self._range(det)
            x = moss.CAMERA_POS[0] + r * math.cos(det.bearing)
            y = moss.CAMERA_POS[1] + r * math.sin(det.bearing)
            # Three filters, and every one of them is a bug this brain had.
            # A can already IN THE BIN is still a can to a detector, and it
            # sits behind the camera, so its bearing folds back to a small
            # or negative x and the machine went straight from `search` to
            # `close` on it, over and over (seen on the 90 s contact sheet:
            # one can picked, then 80 s of cycling on nothing).
            if abs(det.bearing) > self.p.max_bearing:
                continue                       # not in front
            if x < floor:
                continue                       # under the robot, or behind
            if (moss.BIN_INTERIOR_X[0] - 0.05 < x < moss.BIN_INTERIOR_X[1] + 0.05
                    and moss.BIN_INTERIOR_Y[0] < y < moss.BIN_INTERIOR_Y[1]):
                continue                       # its own bin's contents
            # its physical SIZE, from the same detection: the detector ranges
            # each object by its own size, so range x angular width is it
            size = 2.0 * r * math.tan(max(float(det.width), 1e-4) / 2.0)
            # ...and how HIGH it sits. The head camera is level with the base
            # (measured: pitch -0.0 deg), so this is camera height plus
            # range x sin(elevation). `pinch_flat_m` gates on it.
            z = moss.CAMERA_POS[2] + r * math.sin(float(det.elevation))
            out.append((r, x, y, size, z))
        return out

    # -- the room model (brain/moss_search.py) ------------------------------
    def _to_base(self, wx: float, wy: float) -> tuple[float, float] | None:
        od = self._odom
        if od is None:
            return None
        dx, dy = wx - od[0], wy - od[1]
        c, s_ = math.cos(od[2]), math.sin(od[2])
        return (dx * c + dy * s_, -dx * s_ + dy * c)

    def _remember(self, senses: Senses, t: float) -> None:
        """Fold this camera frame into the memory and the coverage map."""
        from .moss_search import ObjectMemory, Patrol
        p = self.p
        if self._mem is None:
            self._mem = ObjectMemory()
            self._patrol = Patrol(None, p.patrol_inset_m)
        self._map_room(senses)
        self._scan_wrist(senses, t)
        # A delivered object leaves the memory with the delivery.
        if self._target_world is not None:
            self._mem_target = self._target_world
        if self.picked != self._mem_picked:
            self._mem_picked = self.picked
            if self._mem_target is not None:
                self._mem.forget_near(*self._mem_target, p.same_can_m)
                self._mem_target = None
        frame = senses.det
        if frame is None or frame.t == self._det_t or self._odom is None:
            return
        self._det_t = frame.t
        pts = []
        # The arm is only out of the way in these two; `near_min_x` says why.
        floor = (p.near_min_x if self.state in ("search", "approach")
                 else p.min_x)
        for rng, x, y, size, _z in self._toys_in_view(frame, floor):
            w = self._world((x, y))
            if w is not None:
                pts.append((w[0], w[1], size, rng))
        matched = self._mem.observe(pts, t)
        # Only with the arm tucked is the camera's view its own: deployed,
        # the gripper fills the top of the frame.
        if self.state not in ("search", "approach"):
            return
        half = 0.70                      # inside the 43.5 deg half field
        expected = []
        for e in self._mem.items:
            b = self._to_base(e.x, e.y)
            if b is None:
                continue
            cx = b[0] - moss.CAMERA_POS[0]
            r = math.hypot(cx, b[1])
            if (cx > 0 and abs(math.atan2(b[1], cx)) < half
                    and 0.2 < r < p.see_range_per_size * e.size):
                expected.append(e)
        self._mem.looked(expected, {e.id for e in matched}, t)
        if self._cov is not None:
            od = self._odom
            cam = self._world((moss.CAMERA_POS[0], 0.0))
            if cam is not None:
                self._cov.mark(cam, od[2], half, 0.15, p.cover_m, t)

    def _flat_target(self) -> bool:
        """Is the thing being deployed on LYING FLAT — flatter than
        `pinch_flat_m`? The MEDIAN of this target's recent height readings,
        never one tick: a single reading is noisy (card0 spans -0.016..0.020
        across a run while its median is 0.004), and gating on one pinched
        `paper0`, a 22 mm cube, 8 times in a run, missing every one."""
        if self.p.pinch_flat_m <= 0.0:
            return False
        hs = sorted(self._fix_heights[-15:])
        return len(hs) >= 5 and hs[len(hs) // 2] < self.p.pinch_flat_m

    def _grip_axis(self) -> str:
        """"along" or "across" — which way the jaws close on this target.

        THE NAMING IS A TRAP, and it cost a battery. "across" reads like the
        right choice for a card ("jaws across the card"), and it is the wrong
        one: MEASURED in the room it put the jaw axis 85-89 deg from the
        card's long axis on EVERY close, squeezing its 40 mm short side, which
        lifted it 0/15 on the bench. "along" puts the jaw axis ALONG the long
        axis — squeezing the 60 mm side — and the error falls to 1.4 deg.
        The butt keeps whatever `pinch_grip` says, which is "along" for its
        8 mm diameter; flipping that global to "across" measured 13 v 22
        binned and cost the cigarette.
        """
        if self._target_flat and self._wide_target():
            return "along"
        return self.p.pinch_grip

    def _wide_target(self) -> bool:
        """Is this flat target WIDE — a card rather than a butt? The MEDIAN of
        its apparent sizes: one tick overlaps (card 0.030-0.060, butt
        0.030-0.044), and reading one flipped the BUTT into the card's deep
        descend and left it on the floor in 11 of 20 runs against 0."""
        ss = sorted(self._fix_sizes[-15:])
        return len(ss) >= 5 and ss[len(ss) // 2] > self.p.pinch_wide_m

    def _at_rest(self, senses: Senses) -> bool:
        arm = senses.arm or {}
        return all(j in arm and abs(float(arm[j]) - v) < self.p.rest_tol_rad
                   for j, v in zip(moss.ARM_JOINTS, self._rest()))

    def _scan_wrist(self, senses: Senses, t: float) -> None:
        """The wrist camera's sightings into the memory, while the arm rests
        (`wrist_scan`)."""
        self._wcam = None
        frame = senses.arm_det
        if (not self.p.wrist_scan or self.state not in ("search", "approach")
                or self._odom is None or not self._at_rest(senses)):
            return
        if self._kin is None:
            from ..robots.moss_pinch import MossKinematics
            self._kin = MossKinematics()
        pc, Rc = self._kin.body_in_base(senses.arm, 0.041, moss.ARM_CAMERA_BODY)
        ax = Rc[:, 0]
        near, far = _wrist_floor_band(pc, Rc)
        self._wcam = (*(self._world((float(pc[0]), float(pc[1]))) or (0.0, 0.0)),
                      self._odom[2] + math.atan2(float(ax[1]), float(ax[0])),
                      near, far)
        if frame is None or frame.t == self._adet_t:
            return
        self._adet_t = frame.t
        pts = []
        for det in frame.detections:
            if det.cls != "toy":
                continue
            r = self._range(det)
            ce = math.cos(det.elevation)
            q = pc + Rc @ np.array([r * ce * math.cos(det.bearing),
                                    r * ce * math.sin(det.bearing),
                                    r * math.sin(det.elevation)])
            # on the floor, and not on (or in) the robot itself
            if q[2] > 0.20 or math.hypot(q[0], q[1]) < 0.25:
                continue
            w = self._world((float(q[0]), float(q[1])))
            if w is not None:
                size = 2.0 * r * math.tan(max(float(det.width), 1e-4) / 2.0)
                pts.append((w[0], w[1], size, r))
        if pts:
            self._mem.observe(pts, t)

    def _map_room(self, senses: Senses) -> None:
        """The depth into the room map, and the work area it gives: the
        sensed one when this MOSS has depth, the scenario's when it has not."""
        from .moss_search import Coverage, Patrol, RoomMap, area_from_world
        p = self.p
        lf = senses.lidar
        if p.sense_walls and lf is not None and self._odom is not None:
            if self._room is None:
                self._room = RoomMap(self._odom[:2])
            if lf.t != self._lidar_t:
                self._lidar_t = lf.t
                self._room.update(lf, self._odom, moss.DEPTH_MAX_RANGE_M)
        if self._room is not None:
            area = self._room.area()
            src = "sensed" if area is not None else None
        elif self._area_src == "given" or senses.t < 1.0:
            # A MOSS whose depth has said nothing in its first second has
            # none; before that, the first scan may simply not be in yet.
            return
        else:
            area = area_from_world(self.world)
            src = "given" if area is not None else None
        old = self._patrol.area
        if area is None or (src == self._area_src and old is not None and max(
                abs(a - b) for a, b in zip(area, old)) < 0.1):
            return
        self._patrol = Patrol(area, p.patrol_inset_m)
        self._cov = Coverage(area)
        self._area_src = src

    def _patrol_step(self, senses: Senses, t: float):
        """The search when `patrol` is on -> (twist, note)."""
        p = self.p
        od = self._odom
        spin = ((0.0, 0.0, p.search_wz), "search: turning for a can")
        if od is None or self._mem is None:
            return spin
        x, y, yaw = od

        def wrap(a):
            return math.atan2(math.sin(a), math.cos(a))

        st = self._srch
        if st is None:
            st = self._srch = {"phase": "spin", "turned": 0.0, "yaw": yaw,
                               "since": t, "goal": None, "look": None, "id": None}
        if st["phase"] == "spin":
            st["turned"] += abs(wrap(yaw - st["yaw"]))
            st["yaw"] = yaw
            if st["turned"] < 2.0 * math.pi:
                return spin
            st["phase"] = "choose"
        if st["phase"] == "choose":
            cands = [e for e in self._mem.items if self._mem.confirmed(e)
                     and not any(math.hypot(e.x - wx, e.y - wy) < p.same_can_m
                                 for wx, wy, _t in self._written_off)]
            if cands:
                e = min(cands, key=lambda e: math.hypot(e.x - x, e.y - y))
                view = min(max(e.size * p.see_range_per_size
                               + moss.CAMERA_POS[0], 0.45), 0.8)
                d = math.hypot(e.x - x, e.y - y)
                k = max(d - view, 0.0) / max(d, 1e-6)
                st.update(phase="go", kind="memory", id=e.id, since=t,
                          goal=(x + (e.x - x) * k, y + (e.y - y) * k),
                          look=(e.x, e.y))
            else:
                room = self._room
                clear = (None if room is None else
                         (lambda wx, wy: room.blocked(wx, wy, p.waypoint_clear_m)))
                wp = self._patrol.next_waypoint(x, y, clear) if self._patrol else None
                if wp is None:
                    st.update(phase="spin", turned=0.0, yaw=yaw)
                    return spin
                st.update(phase="go", kind="rim", id=None, since=t, goal=wp,
                          look=self._patrol.centre)
        if st["phase"] == "go":
            gx, gy = st["goal"]
            d = math.hypot(gx - x, gy - y)
            if t - st["since"] > p.patrol_leg_s:
                if st["kind"] == "memory":
                    e = next((e for e in self._mem.items if e.id == st["id"]), None)
                    if e is not None:
                        self._mem.items.remove(e)
                st["phase"] = "choose"
                return (0.0, 0.0, 0.0), "search: gave up on that leg"
            if d > 0.08:
                err = wrap(math.atan2(gy - y, gx - x) - yaw)
                wz = max(-p.search_wz, min(p.search_wz, 2.0 * err))
                vx = p.patrol_mps * max(0.0, math.cos(err)) ** 3 if abs(err) < 1.0 else 0.0
                # Touch: driving and not moving is something in the way.
                if vx >= p.stall_cmd_mps and abs(senses.speed or 0.0) < p.stall_mps:
                    st.setdefault("stall", t)
                    if t - st["stall"] >= p.stall_s:
                        if self._room is not None:
                            self._room.feel(x + p.felt_ahead_m * math.cos(yaw),
                                            y + p.felt_ahead_m * math.sin(yaw))
                        if st["kind"] == "memory":
                            e = next((e for e in self._mem.items if e.id == st["id"]), None)
                            if e is not None:
                                self._mem.items.remove(e)
                        st.pop("stall", None)
                        st["phase"] = "choose"
                        return (0.0, 0.0, 0.0), "search: bumped into something, marked it"
                else:
                    st.pop("stall", None)
                what = ("a remembered object" if st["kind"] == "memory"
                        else "the next spot on the rim")
                return (vx, 0.0, wz), f"search: driving to {what}"
            st["phase"] = "face"
        if st["phase"] == "face":
            lx, ly = st["look"] or (x + math.cos(yaw), y + math.sin(yaw))
            err = wrap(math.atan2(ly - y, lx - x) - yaw)
            if abs(err) > 0.12:
                return ((0.0, 0.0, max(-p.search_wz, min(p.search_wz, 2.0 * err))),
                        "search: turning to look")
            st.update(phase="look", since=t)
        if st["phase"] == "look":
            if t - st["since"] < p.patrol_look_s:
                return (0.0, 0.0, 0.0), "search: looking"
            if st["kind"] == "memory":
                # Looked right at it from close up and did not see it.
                e = next((e for e in self._mem.items if e.id == st["id"]), None)
                if e is not None:
                    self._mem.items.remove(e)
            st["phase"] = "choose"
        return (0.0, 0.0, 0.0), "search: choosing where to look"

    def map_payload(self, t: float) -> dict | None:
        """The room model, for the /sim map: the work area, the rim route,
        every remembered object, the current leg, and the coverage ages."""
        if self._mem is None:
            return None
        cov = None
        if self._cov is not None:
            ages = self._cov.ages(t)
            cov = {"n": [self._cov.nx, self._cov.ny],
                   "age": "".join("-" if a >= 99 else str(min(a // 5, 9)) for a in ages)}
        st = self._srch if self.state == "search" else None
        return {
            "area": list(self._patrol.area) if self._patrol and self._patrol.area else None,
            "areaSrc": self._area_src,
            "wcam": None if self._wcam is None else [round(v, 3) for v in self._wcam],
            "room": None if self._room is None else self._room.payload(),
            "rim": [list(w) for w in (self._patrol.waypoints if self._patrol else [])],
            "mem": [[round(e.x, 3), round(e.y, 3), round(e.size, 3), e.hits,
                     int(self._mem.confirmed(e)), round(t - e.last, 1),
                     int(any(math.hypot(e.x - wx, e.y - wy) < self.p.same_can_m
                             for wx, wy, _t in self._written_off))]
                    for e in self._mem.items],
            "target": list(self._target_world) if self._target_world else None,
            "goal": list(st["goal"]) if st and st.get("goal") else None,
            "look": list(st["look"]) if st and st.get("look") else None,
            "leg": (st.get("kind") if st and st["phase"] in ("go", "face", "look") else
                    (st["phase"] if st else None)),
            "cov": cov,
        }

    def _range(self, det) -> float:
        """Range from the APPARENT WIDTH and the can's own radius.

        NOT `Detection.range_est`, and this is the single most expensive bug
        in this brain's history: `range_est` is derived from a per-CLASS
        nominal radius, and a 66 x 115 mm drinks can is not the playroom
        block that class was sized on. MEASURED in the room — the brain shut
        its jaws believing the can was 0.259 m ahead while it truly stood at
        **0.491 m**, and did that seven times in 180 s while never touching
        a can (0/3 stowed).

        A sphere of radius R at range r subtends `2*atan(R/r)`, so the range
        is `R / tan(width/2)`. `can_radius_m` is what the DETECTOR models the
        prop as — `world/scenario.Prop.radius()` takes the largest half
        extent, which for an upright can is half its HEIGHT, not its own
        radius. On hardware this number is the detector's own calibration for
        the class it is reporting, which is why it is a parameter and not a
        constant taken from the MJCF.
        """
        if self.p.range_from_detector:
            return float(det.range_est)
        half = max(float(det.width) / 2.0, 1e-4)
        return self.p.can_radius_m / math.tan(half)

    def _world(self, base_xy) -> tuple[float, float] | None:
        """A base-frame point in world coordinates, using `odom`."""
        od = self._odom
        if od is None:
            return None
        c, s_ = math.cos(od[2]), math.sin(od[2])
        return (od[0] + base_xy[0] * c - base_xy[1] * s_,
                od[1] + base_xy[0] * s_ + base_xy[1] * c)

    def _is_written_off(self, x: float, y: float) -> bool:
        w = self._world((x, y))
        if w is None:
            return False
        return any(math.hypot(w[0] - wx, w[1] - wy) < self.p.same_can_m
                   for wx, wy, _t in self._written_off)

    def _pinch_step(self, senses: Senses, t: float, since: float):
        """One tick of the scripted pinch (`pinch_small`) -> (arm, note)."""
        from ..robots.moss_pinch import MossKinematics
        p = self.p
        if self._kin is None:
            self._kin = MossKinematics()
        K = self._kin
        here = {j: float((senses.arm or {}).get(j, v))
                for j, v in zip(moss.ARM_JOINTS, moss.GRASP_POSE)}
        jaw_now = float((senses.arm or {}).get(moss.GRIPPER_JOINT, 0.041))
        pc = self._pinch
        if pc is None or pc.get("t0") != self._t0:
            pc = self._pinch = {"t0": self._t0, "phase": "approach", "since": since,
                                "fixes": [], "yaws": [], "from": dict(here)}
        tob = senses.target_obs or {}

        def look():
            g = tob.get("grip")
            if g is not None:
                pc["fixes"].append(K.object_in_base(here, jaw_now, g))
                if tob.get("yaw") is not None:
                    pc["yaws"].append(float(tob["yaw"]))

        def ease(a):
            a = min(1.0, max(0.0, a))
            return a * a * (3.0 - 2.0 * a)

        def blend(a_pose, b_pose, a):
            k = ease(a)
            return {j: (1.0 - k) * a_pose[j] + k * b_pose[j] for j in moss.ARM_JOINTS}

        dt = since - pc["since"]
        ph = pc["phase"]

        def target():
            """The target in the base frame: the head camera's fix while it
            has one, then dead-reckoned by odometry — a flat thing slides
            under the camera's near edge as the robot closes in, and 12 of 21
            butt pinches stopped right there when this read the fix alone."""
            # LOCKED to the object the pinch started on: as a small thing
            # drops out of the head camera's view its fix jumps to the next
            # toy it can see — 9 of 12 butt pinches drove toward an object
            # 0.4-1.0 m away and gave up out of reach. A fix counts only
            # within `pinch_lock_m` of the locked spot.
            if self._fix is not None:
                wf = self._world(self._fix)
                w0 = pc.get("world")
                if w0 is None or (wf is not None and math.hypot(
                        wf[0] - w0[0], wf[1] - w0[1]) < p.pinch_lock_m):
                    pc["world"] = wf or w0
                    return self._fix
            w, od = pc.get("world"), self._odom
            if w is None or od is None:
                return None
            c, s_ = math.cos(-od[2]), math.sin(-od[2])
            dx, dy = w[0] - od[0], w[1] - od[1]
            return (dx * c - dy * s_, dx * s_ + dy * c)

        if ph == "approach":
            fx = target()
            if fx is not None and float(fx[0]) > p.pinch_range_m and dt < p.pinch_approach_s:
                self._pinch_twist = (p.band_mps, 0.0,
                                     p.creep_kp * math.atan2(float(fx[1]), max(float(fx[0]), 1e-3)))
                return {**pc["from"], moss.GRIPPER_JOINT: 0.041}, "pinch: closing in"
            self._pinch_twist = (0.0, 0.0, 0.0)
            pc.update(phase="look", since=since, from_=dict(here))
            pc["from"] = dict(here)
            return {**pc["from"], moss.GRIPPER_JOINT: 0.041}, "pinch: looking"
        if ph == "look":
            # HOVER FIRST, over the head camera's fix: from the deploy pose a
            # flat thing 0.27 m off is outside the wrist camera's view (every
            # butt in a 6-seed trace read nothing); from above it is not.
            fx = target()
            if fx is None:
                self._to("creep", t)
                return None, "pinch: lost it, learned pick"
            obj = np.array([float(fx[0]), float(fx[1]), 0.0])
            hover = obj + np.array([0.0, 0.0, p.pinch_hover_m + K.pad_half])
            pose, res = K.solve(hover, here["wrist_roll"], here)
            if res > 0.005:
                self._to("creep", t)                      # out of jaws-down reach
                return None, "pinch: out of reach, learned pick"
            pc.update(phase="hover", since=since, obj=obj, roll=here["wrist_roll"],
                      hover=hover, hover_pose=pose, fixes=[], yaws=[])
            return {**pc["from"], moss.GRIPPER_JOINT: 0.041}, "pinch: hovering"
        if ph == "hover":
            a = dt / self._paced(p.pinch_move_s, pc["from"], pc["hover_pose"])
            arm = blend(pc["from"], pc["hover_pose"], a)
            if a >= 1.0:
                pc.update(phase="settle", since=since, fixes=[], yaws=[])
            return {**arm, moss.GRIPPER_JOINT: 0.041}, "pinch: over it"
        if ph == "settle":
            look()
            if dt < p.pinch_look_s:
                return {**pc["hover_pose"], moss.GRIPPER_JOINT: 0.041}, "pinch: looking down"
            if not pc["fixes"]:
                self._to("creep", t)                      # the wrist camera saw nothing
                return None, "pinch: no fix, learned pick"
            obj = np.mean(pc["fixes"], axis=0)
            roll = pc["roll"]
            if pc["yaws"]:
                c = float(np.mean(np.cos(2 * np.array(pc["yaws"]))))
                s = float(np.mean(np.sin(2 * np.array(pc["yaws"]))))
                grip = self._grip_axis()
                jaw_yaw = 0.5 * math.atan2(s, c) + (
                    0.0 if grip == "along" else math.pi / 2)
                roll = K.roll_for(pc["hover_pose"], jaw_yaw)
            # the arm sags under its own weight: aim by where the pads REALLY
            # are over where they were sent
            sag = pc["hover"] - K.padmid(here, jaw_now)
            over = np.array([obj[0], obj[1], pc["hover"][2]]) + sag
            over_pose, _ = K.solve(over, roll, pc["hover_pose"])
            # Only a WIDE flat thing (a card) needs the pads pressed below
            # the clearance: the 8 mm butt and the 12 mm cap already get
            # enough pad overlap, and giving them the deep descend cost cap0
            # (left on the floor in 7 of 11 runs against ~0 before).
            deep = self._target_flat and self._wide_target()
            floor_m = p.pinch_flat_floor_m if deep else p.pinch_floor_m
            grasp = np.array([obj[0], obj[1], K.pad_half + floor_m]) + sag
            pose, res = K.solve(grasp, roll, over_pose)
            if res > 0.005:
                self._to("creep", t)
                return None, "pinch: out of reach, learned pick"
            pc.update(phase="align", since=since, over_pose=over_pose, grasp_pose=pose)
            return {**pc["hover_pose"], moss.GRIPPER_JOINT: 0.041}, "pinch: lining up"
        if ph == "align":
            a = dt / self._paced(0.6 * p.pinch_move_s,
                                 pc["hover_pose"], pc["over_pose"])
            arm = blend(pc["hover_pose"], pc["over_pose"], a)
            if a >= 1.0:
                pc.update(phase="descend", since=since, hover_pose=pc["over_pose"])
            return {**arm, moss.GRIPPER_JOINT: 0.041}, "pinch: lining up"
        if ph == "descend":
            a = dt / self._paced(p.pinch_move_s, pc["hover_pose"], pc["grasp_pose"])
            arm = blend(pc["hover_pose"], pc["grasp_pose"], a)
            if a >= 1.0:
                pc.update(phase="close", since=since)
            return {**arm, moss.GRIPPER_JOINT: 0.041}, "pinch: going down"
        if ph == "close":
            a = ease(dt / p.pinch_close_s)
            if dt < p.pinch_close_s:
                return ({**pc["grasp_pose"], moss.GRIPPER_JOINT: 0.041 * (1.0 - a)},
                        "pinch: closing")
            if self._gripped_raw(senses):
                # STRAIGHT UP FIRST. The lift swings toward LIFT_POSE at once,
                # and a card pinched on its 4 mm edges was pulled out sideways:
                # 72 card carries, 60 lost in the lift.
                up = K.padmid(pc["grasp_pose"], 0.0) + np.array([0.0, 0.0, p.pinch_rise_m])
                pose, _ = K.solve(up, pc["roll"], pc["grasp_pose"], jaw=0.0)
                pc.update(phase="rise", since=since, rise_pose=pose)
                return {**pc["grasp_pose"], moss.GRIPPER_JOINT: 0.0}, "pinch: got it, rising"
            self._attempts += 1
            if self._attempts >= p.max_retries:
                self._give_up(t)
            self._to("tuck", t)
            self._dropped = self.p.missed_pick_straight
            self._back_due = True
            return None, "pinch: missed"
        if ph == "rise":
            a = dt / self._paced(p.pinch_move_s, pc["grasp_pose"], pc["rise_pose"])
            arm = blend(pc["grasp_pose"], pc["rise_pose"], a)
            if a < 1.0:
                return {**arm, moss.GRIPPER_JOINT: 0.0}, "pinch: rising"
            if self._gripped_raw(senses):
                self._carry_from = dict(here)
                self._policy_cmd = {**here, moss.GRIPPER_JOINT: 0.0}
                self._pinched = True            # before `_to`: it reads it
                self._to("lift", t)
                return None, "pinch: got it"
            self._attempts += 1
            if self._attempts >= p.max_retries:
                self._give_up(t)
            self._to("tuck", t)
            self._dropped = self.p.missed_pick_straight
            self._back_due = True
            return None, "pinch: dropped it rising"
        return None, "pinch"

    @property
    def pinch_target_world(self):
        """The pinch's locked target, odometry frame (x, y, 0), or None —
        what the wrist camera should be reporting on during a pinch."""
        if self.state != "pinch" or not self._pinch:
            return None
        w = self._pinch.get("world")
        return None if w is None else (float(w[0]), float(w[1]), 0.0)

    def _give_up(self, t: float) -> None:
        """Write the current target off, so `_see` stops offering it."""
        if self._target_world is not None:
            self._written_off.append((*self._target_world, t))
        self._attempts = 0
        self._target_world = None
        self._fix = None

    def _expire(self, t: float) -> None:
        self._written_off = [w for w in self._written_off
                             if t - w[2] < self.p.give_up_s]

    def _carry_fix(self, senses: Senses) -> None:
        """Move the remembered fix with the robot.

        The can is INVISIBLE for the last few centimetres — it goes under the
        camera's near edge — so the creep runs on dead reckoning. `odom` is
        the body's own (x, y, yaw), and the fix is rotated and translated by
        whatever moved since the last tick.
        """
        od = senses.odom
        if od is None or self._fix is None:
            self._odom = od
            return
        if self._odom is not None:
            dx, dy = od[0] - self._odom[0], od[1] - self._odom[1]
            dyaw = od[2] - self._odom[2]
            c, s = math.cos(-self._odom[2]), math.sin(-self._odom[2])
            fx, fy = self._fix
            # into the previous base frame's world offset, then re-rotate
            bx, by = fx - (dx * c - dy * s), fy - (dx * s + dy * c)
            cy_, sy = math.cos(-dyaw), math.sin(-dyaw)
            self._fix = (bx * cy_ - by * sy, bx * sy + by * cy_)
        self._odom = od

    # ------------------------------------------------------------ the machine

    def _track(self, senses: Senses) -> None:
        """The fix the POLICY is shown — kept separately from the one the
        state machine steers by.

        `_see` filters for TARGET SELECTION: it drops anything behind, under
        the chassis, or already in the bin, so the machine cannot chase its
        own load. Those filters are wrong for tracking a can the robot has
        already committed to: they blank it at exactly grasp range, and the
        policy then reads `target_seen` 0 on 55% of creep steps against the
        2% it saw in training. So this keeps the nearest toy with no filter
        but the field of view.
        """
        frame = senses.det
        if frame is None:
            return
        toys = [d for d in frame.detections if d.cls == "toy"]
        if not toys:
            return
        # ONCE COMMITTED, STAY ON THE SAME CAN. Taking the nearest DETECTED
        # toy is right while hunting and wrong the moment the robot has
        # chosen one: the committed can drops under the camera's near edge at
        # almost exactly grasp range, and the nearest thing still visible is
        # then a DIFFERENT can, further away. MEASURED at a creep handover —
        # can1 truly at 0.504 m, can2 at 1.174, can0 at 1.502 — the brain
        # handed the pickup policy a target at 1.117 m, i.e. can2, twice the
        # furthest distance that policy ever trained on (its rung 2 tops out
        # at 0.55). It converted 3 of 9 attempts in the room against 12/12 in
        # its own env.
        #
        # KEPT ON A NULL. A/B'd over 6 seeds x 180 s: 8 cans of 18 into the
        # bin with this gate and 8 of 18 without it. So the retarget is a
        # real defect and demonstrably not what limits the loop — at this
        # sample size, which for 18 events is a weak instrument. It stays
        # because pointing a policy at an object it never trained on is
        # wrong whether or not this particular room punishes it, and because
        # the next person measuring the pickup should not have to rediscover
        # that the target can change under it mid-grasp.
        #
        # So while the loop has committed, the fix is the detection nearest
        # the one being tracked, and only if it is plausibly the same object.
        # Nothing within the gate means the can is out of sight, which the
        # policy is trained to handle — `target_seen` goes to 0 and it acts
        # on the last fix. Being blind is recoverable; being confidently
        # pointed at the wrong can is not.
        committed = self.state in ("deploy", "creep", "close")
        if committed and self._track_fix is not None:
            def gap(det):
                rr = self._range(det)
                px = moss.CAMERA_POS[0] + rr * math.cos(det.bearing)
                py = moss.CAMERA_POS[1] + rr * math.sin(det.bearing)
                return math.hypot(px - self._track_fix[0],
                                  py - self._track_fix[1])
            best = min(toys, key=gap)
            if gap(best) > self.p.retarget_gate_m:
                return                      # none of these is our can
        else:
            best = min(toys, key=lambda d: d.range_est)
        r = self._range(best)
        self._track_fix = np.array([
            moss.CAMERA_POS[0] + r * math.cos(best.bearing),
            moss.CAMERA_POS[1] + r * math.sin(best.bearing)])
        self._track_t = senses.t

    def _policy_obs(self, senses: Senses, fix,
                    task: str = "pick") -> np.ndarray | None:
        """`moss.CONTRACT_ID`'s 32 floats, built from what a BRAIN is handed.

        The env builds this from its own `MjData`; here it comes off
        `Senses`, which is the same information a code skill would have on
        the robot — achieved joint positions, odometry, the detector's fix.
        Joint velocities are finite-differenced because `Senses` carries no
        velocity channel, which is also what the Jetson would do.
        """
        arm = senses.arm
        if arm is None:
            return None
        dt = max(senses.t - self._prev_t, 1e-3)
        pos = np.array([float(arm.get(j, moss.ARM_HOME[j]))
                        for j in moss.JOINT_NAMES], np.float32)
        if self._prev_arm is None:
            vel = np.zeros_like(pos)
        else:
            prev = np.array([float(self._prev_arm.get(j, moss.ARM_HOME[j]))
                             for j in moss.JOINT_NAMES], np.float32)
            vel = (pos - prev) / dt
        wz = 0.0
        if senses.odom is not None and self._prev_yaw is not None:
            d = senses.odom[2] - self._prev_yaw
            wz = math.atan2(math.sin(d), math.cos(d)) / dt
        o = np.zeros(moss.OBS_DIM, np.float32)
        o[moss.OBS_JOINT_POS] = pos - np.asarray(moss.DEFAULT_POSE)
        o[moss.OBS_JOINT_VEL] = vel
        o[moss.OBS_LAST_ACTION] = self._act
        o[moss.OBS_BASE_TWIST] = (float(senses.speed or 0.0), wz)
        # NO FIX MEANS ZEROS, exactly as the env writes it ("if self._fix is
        # None: target = np.zeros(3)"). This used to fall back to the last
        # known position and publish it regardless, which is a train/deploy
        # mismatch for any leg whose env runs without a fix — and the STOW
        # leg is precisely that: the can is inside the jaws, occluded and far
        # nearer than the depth minimum, so its env publishes zeros for the
        # whole carry. Handed a real (x, y) there instead, a policy measured
        # at 9/12 into the bin in its own env delivered 0 of 9 in the room.
        tf = self._track_fix if self._track_fix is not None else fix
        if tf is None:
            o[moss.OBS_TARGET_BASE] = (0.0, 0.0, 0.0)
        else:
            o[moss.OBS_TARGET_BASE] = (tf[0], tf[1], 0.0575)
        # FRESHNESS, not a constant 1.0 — which is what this read for one
        # run. The can drops under the camera's near edge at almost exactly
        # grasp range, so `target_seen` going to 0 is not an edge case here,
        # it is the last second of every pickup. The env trains through it
        # (its own slot flips) and the policy learns to act blind; telling it
        # "I can see it" in the one phase it was trained to handle without
        # sight is a train/deploy mismatch in the only 30 cm that matter.
        o[moss.OBS_TARGET_SEEN] = (
            1.0 if (senses.t - self._track_t) < 0.6 else 0.0)
        # THE SLOTS THIS LEG TRAINED WITH, from the wrist/front cameras
        # (`Senses.target_obs`), exactly as its env fills them — and left at
        # zero for a leg that trained with them dead (a live value there
        # would clip against a normaliser whose var is 3e-10).
        fl = self._flags.get(task, {})
        tob = senses.target_obs or {}
        if fl.get("publish_attitude") and tob.get("att") is not None:
            o[moss.OBS_TARGET_AXIS] = tob["att"][:2]
            o[moss.OBS_TARGET_UPRIGHT] = tob["att"][2]
        if fl.get("publish_proximity"):
            o[moss.OBS_TARGET_BASE.start + 2] = float(tob.get("range") or 0.0)
        if (fl.get("publish_size") and o[moss.OBS_TARGET_SEEN] > 0
                and tob.get("top") is not None):
            o[moss.OBS_SPARE] = float(tob["top"])
        if fl.get("grip_xyz"):
            # the depth fix in the TOOL frame, as `MossPickEnv._grip_xyz_obs`
            g = tob.get("grip")
            o[list(moss.OBS_GRIP_XYZ)] = (
                (float(g[0]), float(g[1]), float(g[2]), 1.0) if g is not None
                else (0.0, 0.0, ME.GRIP_UNSEEN_M, 0.0))
        if fl.get("publish_grip"):
            # the wrist depth camera's grip fix, as `MossPickEnv._grip_obs`
            # publishes it (`moss.OBS_GRIP`); `read()` already applied the
            # freshness gate
            g = tob.get("grip")
            o[list(moss.OBS_GRIP)] = (
                (float(np.linalg.norm(g)), 1.0) if g is not None
                else (ME.GRIP_UNSEEN_M, 0.0))
        #: The last vector actually published to a policy. Kept because the
        #: only way to find a train/deploy gap is to diff this against what
        #: the env hands the same policy at its reset, slot by slot — every
        #: integration bug in this loop has been found that way and none of
        #: them by reasoning about it.
        self._last_obs = o
        return o

    def _band_error(self, fix: tuple[float, float] | None) -> float:
        """How far out of the pickup policy's reach is the can, to close in?

        Returns 0.0 when the handover should go ahead, and a positive metre
        count when the base should still close in. It never returns negative:
        backing off is not something this should do, and that is MEASURED
        rather than assumed.

        The first version of this drove the can to the middle of the box the
        policy trains in (`moss.PICK_HANDOVER_BOX`), on the argument that a
        policy is competent where it was trained. It doubled the in-band
        share, 28% -> 67%, and delivered one FEWER can. Tagging every
        handover with where it started and whether it reached a lift says
        why — the box is not the ranking:

            too NEAR  x < 0.36      8/11   73%
            IN BAND   0.36-0.55     5/13   38%
            too FAR   x > 0.55     10/18   56%
            too WIDE  |y| > 0.12    1/5    20%

        Nearer is better, not worse, so parking at the band's middle threw
        away the best cell it had. What actually loses attempts is the
        LATERAL offset: the policy's env spawns |y| <= 0.12 and the room
        hands it cans up to 0.297 m off to one side, and those convert at
        20%. So the gate closes in when the can is beyond the box and turns
        to centre it when it is off to the side, and otherwise gets out of
        the way. (11, 13, 18 and 5 attempts: the wide and near cells are
        thin, and it is the DIRECTION that is being acted on here.)
        """
        if fix is None:
            return 0.0
        lo, hi, ymax = moss.PICK_HANDOVER_BOX
        if self._flags.get("pick", {}).get("base_lock"):
            hi = min(hi, self.p.band_far_locked_m)
            if (self.p.size_aware_handover and self._fix_size is not None
                    and self._fix_size < self.p.small_object_m):
                hi = min(hi, self.p.band_far_small_m)
        x, y = float(fix[0]), float(fix[1])
        if abs(y) > ymax:
            # Turning is the caller's yaw term; keep a little forward creep
            # on so a can that is wide AND far still gets closed down.
            return max(x - hi, 0.02)
        return max(x - hi, 0.0)

    def _gripped_raw(self, senses: Senses) -> bool:
        """Is the can in the jaws RIGHT NOW? `Senses.holding`, undebounced.

        The world fills that from `WorldRobot.held_body`, which the driver
        answers by asking whether BOTH pads are touching the same foreign
        body (`robots/moss_drive.held_body`). That is the only test here that
        distinguishes a carry from a crush: a jaw squeezing a can flat
        against the floor stalls its servo exactly like one holding it, and
        reading the servo alone had this loop lift, stow and release nothing
        24 times in 300 s.

        The servo stall stays as a FALLBACK for a world that answers nothing
        — a bare env, or a body whose driver has no `held_body` — with the
        settle guard that keeps a closing transient from reading as a stall.
        """
        # ONLY the physical signal. There was a servo-stall fallback here
        # for one run and it fired constantly: the policy drives the gripper
        # command to 0, the jaw reaches 0 on air, and any transient reads as
        # a stall — 24 lift/stow/release cycles in 300 s, every one of them
        # carrying nothing. A brain runs in a WORLD, and a world always
        # answers this channel; a body whose driver cannot say, says no.
        return bool(senses.holding)

    def _gripped(self, senses: Senses) -> bool:
        """`_gripped_raw`, held for `grip_debounce_s` after the last yes.

        The pads break and remake contact constantly while the arm moves, so
        the raw answer stutters. Traced in the room it went holds/released/
        holds/released inside 0.12 s of a good grip, and the stow leg — which
        exits when the can is gone — ended one tick after it started.
        """
        if self._gripped_raw(senses):
            if (senses.t - self._grip_seen_t) > self.p.grip_debounce_s:
                self._grip_from_t = senses.t     # a fresh grip starts here
            self._grip_seen_t = senses.t
            return True
        return (senses.t - self._grip_seen_t) < self.p.grip_debounce_s

    def _grip_held_for(self, senses: Senses) -> bool:
        """Has the current grip lasted `grip_settle_s`? See that parameter."""
        return (senses.t - self._grip_from_t) >= self.p.grip_settle_s

    def _pick_window_s(self) -> float:
        """How long the learned pick gets: the episode it TRAINED in when that
        was longer than `learned_window_s` (a leg trained to recover and
        re-grasp in 15 s must not be cut off at 8 and sent to tuck)."""
        ep = (self._flags.get("pick") or {}).get("episode_s")
        return max(self.p.learned_window_s, float(ep)) if ep else self.p.learned_window_s

    def _ease(self, k: float) -> float:
        """`carry_ease`: smoothstep — zero speed at both ends of a leg."""
        return k * k * (3.0 - 2.0 * k) if self.p.carry_ease else k

    def _remember_arm(self, arm, t: float) -> None:
        """The last arm command EMITTED, and when.

        Kept on every tick, including the many that command no arm at all
        (every `_to` handoff, `pinch: missed`, `_abort_drop`): `set_arm`
        replaces the whole mapping and a state that names no arm leaves the
        driver holding the last one, so across such a tick the COMMAND did not
        move and only the clock did. Recording it the other way round — not
        advancing the clock until the next dict — is what let `_slew`'s budget
        grow across a gap: `room = cap * (t - t0)` over a 0.5 s run of
        arm-less ticks is 1.5 rad at the shipped cap, which is no limit at
        all, and those gaps are exactly the state transitions the limiter
        exists for.
        """
        if isinstance(arm, dict):
            self._last_arm = {j: float(arm[j]) for j in moss.ARM_JOINTS if j in arm}
        self._last_arm_t = t

    def _slew(self, arm, t: float):
        """`arm_slew_cap`: the arm command no further from the LAST EMITTED
        command than the cap allows in the time since it was emitted. The
        gripper passes through (see the param)."""
        cap = float(self.p.arm_slew_cap)
        if cap <= 0.0 or not isinstance(arm, dict):
            return arm
        last, t0 = self._last_arm, self._last_arm_t
        if last is not None and t > t0:
            room = cap * (t - t0)
            arm = {**arm, **{j: float(np.clip(arm[j], last[j] - room, last[j] + room))
                             for j in moss.ARM_JOINTS if j in arm and j in last}}
        return arm

    def _paced(self, seconds: float, a, b, eased: bool = True) -> float:
        """`seconds`, stretched until no joint exceeds `arm_rate_cap`.

        `a` and `b` are the two ends of a blend, either as dicts keyed by
        `moss.ARM_JOINTS` or as sequences in that order. A smoothstep's peak
        rate is 1.5x its mean and a straight ramp's is 1.0x, so `eased` picks
        the factor — getting that wrong would cap the wrong number.
        """
        cap = float(self.p.arm_rate_cap)
        if cap <= 0.0:
            return seconds
        def val(pose, j, i):
            return float(pose[j] if isinstance(pose, dict) else pose[i])
        span = max(abs(val(b, j, i) - val(a, j, i))
                   for i, j in enumerate(moss.ARM_JOINTS))
        return max(float(seconds), (1.5 if eased else 1.0) * span / cap)

    def _grip_is_a_grasp(self, senses: Senses) -> bool:
        """Are the jaws AROUND the can, or shut past it? See `min_grasp_m`."""
        arm = senses.arm or {}
        jaw = arm.get(moss.GRIPPER_JOINT)
        if jaw is None:
            return True                  # no channel to judge by; allow it
        small = (self._fix_size is not None
                 and self._fix_size < self.p.small_object_m)
        return float(jaw) >= (self.p.min_grasp_small_m if small
                              else self.p.min_grasp_m)

    def _grip_is_deep(self, senses: Senses) -> bool:
        """Is the object deep in the jaws, by the wrist depth camera? See
        `grip_depth_max_m`. Asked only while the handover's other tests pass;
        a gap of more than a tick between asks is a new grip, so the wait
        restarts. No reading (occluded, stale) is not evidence: allow it."""
        lim, t = self.p.grip_depth_max_m, senses.t
        if t - self._depth_last_t > 0.1:
            self._depth_block_t = None
        self._depth_last_t = t
        g = (senses.target_obs or {}).get("grip")
        small = (self._fix_size is not None
                 and self._fix_size < self.p.grip_depth_min_size_m)
        if lim is None or small or g is None or float(np.linalg.norm(
                np.asarray(g, float)[:3])) <= lim:
            self._depth_block_t = None
            return True
        if self._depth_block_t is None:
            self._depth_block_t = t
        return t - self._depth_block_t >= self.p.grip_depth_wait_s

    def _policy_step(self, senses: Senses, fix, task: str = "pick"):
        """Run one leg's learned skill, AT ITS OWN CONTROL RATE."""
        # THE POLICY RUNS AT 25 Hz, the world ticks at 50. Every MOSS policy
        # is trained inside an env that decimates to `moss.CONTROL_HZ` and
        # applies one action per control period; called every world tick it
        # gets TWICE the actions per second, so its integrated commands move
        # at twice the rate it ever experienced — the arm at 1.5 rad/s
        # instead of 0.75, the jaw at 0.2 m/s instead of 0.1.
        #
        # MEASURED before this: the brain's observation dt was 0.0200 s
        # against a 0.0400 s control tick, its |joint_vel| ran to 4.96, and a
        # traced pickup sat at 0.28-0.36 m oscillating the jaw
        # 0 -> 35 -> 41 -> 0 mm without ever closing, then gave up. The
        # pickup converted about a third of its attempts in the room against
        # 12/12 in its own env, and this is the difference between the two.
        #
        # Between control ticks the last command is HELD, which is what the
        # env does during its decimation.
        # ...AND YET RUNNING IT AT 25 Hz MEASURED WORSE. Holding the command
        # between control ticks, the way the env does, took the room from 8
        # cans of 18 to 4. So the mismatch is real and correcting it alone is
        # not the fix — 18 events cannot separate 4 from 8 with any
        # confidence either, so read this as "tried, did not help" rather
        # than "harmful". THE PICK is now held to 25 Hz by its caller
        # (`pick_at_control_hz`, on since 2026-09-27: 99 v 71 in 24 yard
        # seeds with 478dad); this method itself still runs per call.
        return self._policy_tick(senses, fix, task)

    def _policy_tick(self, senses: Senses, fix, task: str = "pick"):
        """Run one leg's learned skill for one tick -> (twist, arm).

        The deltas below are the ENV's, not this file's opinion:
        `ARM_DELTA_RAD` 0.03, `JAW_DELTA_M` 0.004, `MAX_VX` 0.20,
        `MAX_WZ` 1.0. A policy is a function of the action scaling it was
        trained under, so these two have to agree or the same network does
        something else on the robot than it did in the env.
        """
        obs = self._policy_obs(senses, fix, task)
        if obs is None:
            return (0.0, 0.0, 0.0), None
        sess, in_name = self._sessions[task]
        a = sess.run(None, {in_name: obs.reshape(1, -1)})[0][0]
        self._act = np.clip(np.asarray(a, np.float32), -1.0, 1.0)
        if self._policy_cmd is None:
            self._policy_cmd = {**_pose(POLICY_START_POSE[task]),
                                moss.GRIPPER_JOINT: 0.041}
        cmd = self._policy_cmd
        for i, j in enumerate(moss.ARM_JOINTS):
            cmd[j] = float(cmd[j] + self._act[i] * 0.03)
        cmd[moss.GRIPPER_JOINT] = float(np.clip(
            cmd[moss.GRIPPER_JOINT] + self._act[5] * 0.004, 0.0, 0.041))
        twist = (float(self._act[6]) * 0.20, 0.0, float(self._act[7]) * 1.0)
        if self._flags.get(task, {}).get("base_lock"):
            # it trained with the base LOCKED: its base outputs were no-ops
            # and are unconstrained, so they must not drive the robot
            twist = (0.0, 0.0, 0.0)
        self._held_twist = twist
        return twist, dict(cmd)

    def _ramp_from_here(self, target, since: float, seconds: float,
                        senses: Senses) -> dict[str, float]:
        """Ramp to `target` from wherever the arm WAS when this state began.

        `_ramp` interpolates from `_carry_from`, which the pickup handover
        seeds; this is for the transitions that have no handover to inherit —
        `deploy` and `tuck` — and it snapshots the achieved pose on the first
        tick of the state rather than trusting a command.
        """
        if self._ramp_from is None:
            if senses.arm is not None:
                self._ramp_from = {j: float(senses.arm.get(j, v))
                                   for j, v in zip(moss.ARM_JOINTS, target)}
            elif self._last_arm is not None and all(
                    j in self._last_arm for j in moss.ARM_JOINTS):
                # NO ARM READINGS THIS TICK, so there is nothing to snapshot —
                # and the old fallback commanded `target` OUTRIGHT, which is
                # the unbounded step input `arm_rate_cap` exists to remove
                # (`_paced` below is reached only past this branch). The last
                # command we EMITTED is where the driver is still holding the
                # arm, so it is the honest start: ramp from there and the
                # pacing applies as it does on every other tick.
                self._ramp_from = dict(self._last_arm)
            else:
                # Genuinely cold: no readings and nothing emitted yet, which
                # is the first tick of a run. Nothing can be ramped from an
                # unknown pose; the arm is at its keyframe and this is the one
                # case where the endpoint is all there is.
                return _pose(target)
        seconds = self._paced(seconds, self._ramp_from, target, eased=False)
        k = min(1.0, max(0.0, since / max(seconds, 1e-3)))
        return {j: (1.0 - k) * self._ramp_from[j] + k * v
                for j, v in zip(moss.ARM_JOINTS, target)}

    def _ramp(self, target, since: float, seconds: float) -> dict[str, float]:
        """Move the arm from where it IS to `target`, over `seconds`.

        The states after a grip used to command the endpoint outright, which
        is a step input to a position servo: MEASURED, the brain reported
        `holding=can1` at the grip and `holding=None` 1.9 s later, having
        dropped the can during the lift. It matters more with a LEARNED
        pickup than with the scripted one, because the policy finishes in
        whatever pose it likes and the jump from there is arbitrary.

        The gripper is deliberately NOT in the ramp: whatever closed on the
        can stays closed until `release` opens it.
        """
        if self._carry_from is None:
            self._carry_from = {j: float((self._policy_cmd or {}).get(j, v))
                                for j, v in zip(moss.ARM_JOINTS, moss.GRASP_POSE)}
        eased = self.state in ("lift", "stow") and self.p.carry_ease
        seconds = self._paced(seconds, self._carry_from, target, eased=eased)
        k = min(1.0, max(0.0, since / max(seconds, 1e-3)))
        if self.state in ("lift", "stow"):
            k = self._ease(k)
        return {j: (1.0 - k) * self._carry_from[j] + k * v
                for j, v in zip(moss.ARM_JOINTS, target)}

    def _abort_drop(self, t: float, where: str, direct: bool) -> Intent:
        """The jaws are empty mid-carry: count the attempt and go home for
        another go — STRAIGHT (see `drop_tuck_rate`) when the arm is not yet
        over the bin, by the fold when it is."""
        self._empty_since = None
        self._carry_from = None
        self._leg_from = None
        self._policy_cmd = None
        self._attempts += 1
        if self._attempts >= self.p.max_retries:
            self._give_up(t)
        self._to("tuck", t)
        self._dropped = bool(direct)
        self._back_due = True          # whichever fold runs, back off after it
        self._drop_from = None
        return Intent(twist=(0.0, 0.0, 0.0), arm=None,
                      note=f"{where}: dropped it — going back for it")

    def _after_tuck(self, t: float) -> None:
        """Tuck is finished: BACK OFF if something was just dropped, else look.

        Read here rather than in one fold's branch because there are four ways
        out of `tuck` — home by a clear route, the short route, the learned
        fold and the scripted fold — and a drop's tuck does not reliably take
        the one you would expect. MEASURED: the first cut hooked only the
        straight-home branch, and the eight-seed A/B came back BYTE-IDENTICAL
        (run-overs 16 -> 16, drops inside the gate 18 -> 18) because with
        `plan_routes` on the drop's tuck takes the ROUTE branch, which clears
        the flag on its way to `search`. A dead hook and a working one look
        the same in the diff and different only in the numbers.
        """
        due, self._back_due = self._back_due, False
        self._to("back_off" if due and self.p.drop_back_m > 0.0 else "search", t)

    def _dodge_keep(self) -> list[tuple[float, float]]:
        """World (x, y) of everything the rover is deliberately driving AT.

        The target is not a bystander — steering away from the thing being
        collected would stall the mission, which is the failure mode this
        whole change has to be scored against. Two of them, because the
        state machine has two notions of "what I am going for": the live
        fix in `approach`, and the memory entry a patrol leg was sent to.
        """
        keep = []
        # `self._fix` is used RAW, not through `step`'s `stale_s` test, so a
        # fix the state machine has already given up on goes on exempting a
        # 0.22 m patch of floor until something replaces it. Conservative —
        # it can only suppress a dodge, never invent one — and it is what the
        # 64-seed battery in `bystander_avoid_m` measured, so it stays as
        # measured rather than being tidied afterwards.
        if self._fix is not None:
            w = self._world(self._fix)
            if w is not None:
                keep.append(w)
        st = self._srch
        if (st is not None and st.get("kind") == "memory"
                and st.get("id") is not None and self._mem is not None):
            e = next((e for e in self._mem.items if e.id == st["id"]), None)
            if e is not None:
                keep.append((e.x, e.y))
        return keep

    def _dodge(self, twist, note: str):
        """Bias the drive away from remembered litter in the track corridor.

        `bystander_avoid_m` carries the measurement. Only the WORST offender
        moves the wheels: summing over a cluster would swing the nose by the
        number of objects that happen to be lying together, and they lie
        together often.
        """
        p = self.p
        vx, vy, wz = twist
        if vx <= 0.0 or self._mem is None or self._odom is None:
            return twist, note                 # stopped or reversing: nothing to dodge
        corridor = moss.HALF_WIDTH_M + p.bystander_margin_m
        keep = self._dodge_keep()
        worst = None
        for e in self._mem.items:
            if not self._mem.confirmed(e):
                continue                       # seen once: may be a phantom
            if any(math.hypot(e.x - kx, e.y - ky) < p.same_can_m
                   for kx, ky in keep):
                continue                       # this is the thing we want
            b = self._to_base(e.x, e.y)
            if b is None:
                continue
            bx, by = b
            if not (0.0 < bx < p.bystander_avoid_m) or abs(by) >= corridor:
                continue
            # Nearer is more urgent, and dead ahead is more urgent than
            # grazing the edge of the corridor: both terms are 1 at the worst
            # and 0 where the object leaves the box, so the nudge fades out
            # rather than switching off and jerking the nose back.
            urg = ((p.bystander_avoid_m - bx) / p.bystander_avoid_m
                   * (corridor - abs(by)) / corridor)
            if worst is None or urg > worst[0]:
                worst = (urg, by)
        if worst is None:
            return twist, note
        urg, by = worst
        # +wz is LEFT, so an object on the left (by > 0) subtracts.
        wz -= math.copysign(p.bystander_wz * urg, by if by != 0.0 else 1.0)
        vx *= max(0.0, 1.0 - p.bystander_slow * urg)
        return (vx, vy, wz), note + " (easing round something in the way)"

    def _rear_blocked(self, gap_m: float) -> bool:
        """Is anything solid behind the back bumper, as far as the MAP knows?

        MOSS's depth camera looks FORWARD, so a reverse is blind and this is
        the only check there is. `RoomMap.blocked` counts mapped cells AND
        `felt` bumps, so a wall the robot has only ever run into still stops
        it. An UNMAPPED rear reads clear — early in a run most of the room is —
        which is why the caller also bounds the move.
        """
        if self._room is None or self._odom is None:
            return False
        x, y, yaw = self._odom
        d = moss.REAR_EXTENT_M + max(0.0, float(gap_m))
        return bool(self._room.blocked(x - math.cos(yaw) * d,
                                       y - math.sin(yaw) * d,
                                       self.p.drop_back_clear_m))

    def _rest(self) -> tuple[float, ...]:
        """Where the arm waits: `rest_pose`, or the old tuck without one."""
        return (tuple(self.p.rest_pose) if self.p.rest_pose is not None
                else moss.tuck_pose())

    def _clearance(self):
        if self._clr is None:
            from .moss_motion import ArmClearance
            if self._kin is None:
                from ..robots.moss_pinch import MossKinematics
                self._kin = MossKinematics()
            self._clr = ArmClearance(self._kin)
        return self._clr

    def _move_to(self, goal, senses: Senses, since: float, hubs=()):
        """A planned move from where the arm IS on this state's first tick to
        `goal`: `(arm command, arrived)`, or None when no clear route exists
        (then the state's old branch runs)."""
        from .moss_motion import MinJerkRoute, RouteFollower
        if self._mv is None:
            if senses.arm is None:
                return _pose(goal), False
            q0 = [float(senses.arm.get(j, v)) for j, v in zip(moss.ARM_JOINTS, goal)]
            r = self._clearance().route(q0, goal, hubs, self.p.route_margin_m)
            self._mv = (False if r is None else RouteFollower(
                MinJerkRoute(r, self.p.motion_vmax), self.p.motion_leash_rad))
        if self._mv is False:
            return None
        q = self._mv.step(since, senses.arm)
        return dict(zip(moss.ARM_JOINTS, (float(x) for x in q))), self._mv.done()

    def _fold_hubs(self):
        return (moss.LIFT_POSE, tuple(self.p.fold_route_wp), moss.RETRACT_WAYPOINT)

    def _fold_route(self, senses: Senses):
        """The scripted fold's route if this fold starts from the release
        pose it was planned for, else None (seeded once per tuck)."""
        if not self._route_tried:
            self._route_tried = True
            arm = senses.arm or {}
            if (self.p.fold_route and not self._dropped
                    and all(j in arm for j in moss.ARM_JOINTS)):
                q0 = [float(arm[j]) for j in moss.ARM_JOINTS]
                if max(abs(a - b) for a, b in zip(q0, self.p.fold_route_from)
                       ) < self.p.fold_route_from_tol:
                    self._route = [q0, list(self.p.fold_route_wp),
                                   list(moss.tuck_pose())]
        return self._route

    def _folded(self, senses: Senses) -> bool:
        """Is the arm home — within the fold's own tolerance of the tuck on
        every joint that shapes the arm (`moss_env.RETRACT_JOINTS`)?"""
        from ..robots.moss_env import RETRACT_JOINTS, RETRACT_TOL_RAD
        arm = senses.arm or {}
        tuck = dict(zip(moss.ARM_JOINTS, self._rest()))
        return all(j in arm and abs(float(arm[j]) - tuck[j]) < RETRACT_TOL_RAD
                   for j in RETRACT_JOINTS)

    def _to(self, state: str, t: float) -> None:
        if state in ("creep", "pinch", "search"):
            self._pinched = False
        if (state == "creep" and self.state != "creep"
                and self.p.pick_clean_start):
            self._act = np.zeros(moss.NUM_ACTIONS, np.float32)
            self._policy_cmd = None
            self._last_policy_t = -1e9
        if state not in ("lift", "stow", "release"):
            self._carry_jaw = None
        if state == "lift":
            self._lift_z0 = None
        self._low_since = None
        self._release_from = None
        self._mv = None
        self.state, self._t0 = state, t
        self._fold_seeded = False
        self._route, self._route_tried = None, False
        if state == "search":
            self._srch = None
        if state == "lift" and self._mem is not None:
            # IT IS IN THE JAWS, not on the floor: forget that spot now, not
            # on delivery. The pinch picks without a creep target, so waiting
            # for the delivery left its spot in the memory and the patrol drove
            # back to look at empty floor (two trips in one traced run). A
            # drop is simply seen again.
            pw = (self._pinch or {}).get("world") if self._pinched else None
            jaw = (float(pw[0]), float(pw[1])) if pw is not None else self._world((0.26, 0.0))
            if jaw is not None:
                self._mem.forget_near(jaw[0], jaw[1], 0.15)
        # A new state ramps from where the arm IS now, not from the last
        # state's starting pose. Cleared here so `_ramp_from_here` snapshots
        # again on its first tick.
        self._ramp_from = None

    def step(self, senses: Senses) -> Intent:
        p, t = self.p, senses.t
        self._expire(t)
        self._carry_fix(senses)
        self._track(senses)
        if self.p.object_memory or self.p.patrol:
            self._remember(senses, t)
        seen = self._see(senses)
        if seen is not None:
            self._fix, self._fix_t = seen, t
        # STALE means stale. This read `1e9` for one run, so a fix never
        # expired and the machine ran the whole loop on a memory of a can it
        # had already binned.
        fix = self._fix if (t - self._fix_t) < self.p.stale_s else None
        since = t - self._t0
        arm: dict[str, float] | None = None
        twist = (0.0, 0.0, 0.0)
        note = self.state

        if self.state == "search":
            mv = (self._move_to(self._rest(), senses, since, self._fold_hubs())
                  if p.plan_routes else None)
            arm = (mv[0] if mv is not None else
                   self._ramp_from_here(self._rest(), since, p.tuck_ramp_s, senses))
            if fix is not None and seen is not None:
                self._policy_cmd = None     # the next leg seeds its own pose
                self._to("approach", t)
            elif p.patrol:
                twist, note = self._patrol_step(senses, t)
            else:
                twist = (0.0, 0.0, p.search_wz)
                note = "search: turning for a can"

        elif self.state == "approach":
            if fix is None:
                self._to("search", t)
            else:
                x, y = fix
                if x <= p.deploy_at:
                    self._to("deploy", t)
                elif p.approach_arm_rest:
                    twist = (p.approach_mps, 0.0,
                             p.approach_kp * math.atan2(y, x))
                    mv = (self._move_to(self._rest(), senses, since, self._fold_hubs())
                          if p.plan_routes else None)
                    # NO CLEAR ROUTE: the rest pose is commanded OUTRIGHT,
                    # and that is a step input — MEASURED the largest single
                    # thing the whole mission does. Entering `approach` with
                    # the wrist still rolled where a pinch left it,
                    # `wrist_roll` goes +1.904 -> -1.179 in one 20 ms tick:
                    # **3.083 rad, 154 rad/s**. No blend-pacing can help a
                    # command that is not a blend.
                    #
                    # Ramping it here (as the `tuck` branch above does) was
                    # tried and MEASURED WORSE, twice over: eight paired 300 s
                    # seeds, the step only fell 3.08 -> 2.50 rad while the
                    # longest stall at the torque clamp rose 1.57 -> 4.18 s
                    # and arm-on-bin contact 1766 -> 5986 substeps at
                    # 127 -> 165 N. Same result as `arm_slew_cap`: the stalls
                    # here are CONTACT stalls, so delaying the arm's arrival
                    # lengthens the window it spends pressed on the bin. The
                    # step stays until something gets the arm out of the
                    # rolled pose BEFORE this state, which is where it belongs.
                    arm = mv[0] if mv is not None else _pose(self._rest())
                    note = f"approach: can {x:.2f} m ahead, arm at rest"
                elif "approach" in self._sessions:
                    # THE LEARNED DRIVE. It holds the arm as well as the
                    # base, deliberately: the arm does not reach `TUCK_POSE`
                    # — it jams on the hull 0.74 rad short and rides 75 mm
                    # proud of the chassis — so folding it in is something
                    # the policy is paid for rather than a pose asserted here.
                    twist, arm = self._policy_step(senses, fix, "approach")
                    note = f"approach: can {x:.2f} m ahead, learned"
                else:
                    twist = (p.approach_mps, 0.0,
                             p.approach_kp * math.atan2(y, x))
                    note = f"approach: can {x:.2f} m ahead, arm in"

        elif self.state == "deploy":
            mv = (self._move_to(moss.GRASP_POSE, senses, since,
                                (self._rest(), moss.LIFT_POSE))
                  if p.plan_routes else None)
            arm = {**(mv[0] if mv is not None else
                      self._ramp_from_here(moss.GRASP_POSE, since,
                                           p.deploy_ramp_s, senses)),
                   moss.GRIPPER_JOINT: 0.041}
            note = "deploy: jaws open at can height"
            self._policy_cmd = None
            # NOT the last action, though the symmetry with the stow
            # handover says it should be: the stow leg gains from starting
            # with `o[OBS_LAST_ACTION]` at zero the way its env does, and
            # the pickup leg LOSES by it. MEASURED over the same three
            # seeds: clearing it here took the room from 4 cans in the bin
            # to 0. `deploy` holds for several ticks while the arm travels,
            # so zeroing here also zeroes it on every one of them, and the
            # pickup evidently reads its own recent history through that
            # slot in a way the stow policy does not. Left alone on the
            # evidence rather than on the argument.
            # Hand over when the arm has ARRIVED, not after a fixed wait.
            # MEASURED at the handover tick: on a 2 s timer the wrist was
            # still 0.708 rad from the grasp pose — most of the way back at
            # HOME — because the travel out of the TUCK is longer than the
            # travel the env settles from. The policy then opened every
            # episode in a pose it had never trained in, and the loop stowed
            # 0/3 while the same policy scored 4/12 in its own env.
            arrived = False
            if senses.arm is not None:
                arrived = max(
                    abs(float(senses.arm.get(j, 0.0)) - v)
                    for j, v in zip(moss.ARM_JOINTS, moss.GRASP_POSE)
                ) < p.deploy_tol_rad
            if arrived or since >= p.deploy_timeout_s:
                # ARRIVING IS NOT THE SAME AS BEING IN RANGE. This tested
                # the arm's pose alone and handed the policy whatever range
                # the can had drifted to while the arm travelled — 13 of 47
                # handovers (28%) inside the box it trains in, 38% too far,
                # 23% too near. The policy is 12/12 when its env starts it
                # in the box, so most of the room's failures were states it
                # had never been shown rather than a skill it lacked.
                band = self._band_error(fix) if p.band_handover else 0.0
                if band == 0.0 or since >= p.deploy_timeout_s + p.band_timeout_s:
                    small = (self._fix_size is not None
                             and self._fix_size < p.pinch_size_m)
                    flat = self._flat_target()
                    self._target_flat = flat
                    small = small or flat
                    self._to("pinch" if (p.pinch_small and small) else "creep", t)
                else:
                    # Close the loop on the ONE variable the next leg's
                    # competence depends on. Backing off is as necessary as
                    # closing in: a quarter of these are already too near.
                    x, y = fix if fix is not None else (0.0, 0.0)
                    twist = (p.band_mps, 0.0,
                             p.creep_kp * math.atan2(y, max(x, 1e-3)))
                    note = (f"deploy: lining up, can {x:.2f} m "
                            f"-> [{moss.PICK_HANDOVER_BOX[0]:.2f}, "
                            f"{moss.PICK_HANDOVER_BOX[1]:.2f}]")

        elif self.state == "pinch":
            self._pinch_twist = (0.0, 0.0, 0.0)
            arm, note = self._pinch_step(senses, t, since)
            twist = self._pinch_twist

        elif self.state == "creep" and self._sess is not None:
            # THE LEARNED SKILL drives here: base and arm together, until it
            # reports a grip or the window runs out. `fix` going stale is NOT
            # a failure here the way it is for the scripted creep — the can
            # drops under the camera's near edge at exactly this range, and
            # the policy was trained through that (its `target_seen` slot
            # goes to zero and it keeps acting). So the learned branch runs
            # on the last fix and only gives up on the clock.
            if self._fix is None or since > self._pick_window_s():
                self._attempts += 1
                if self._attempts >= p.max_retries:
                    self._give_up(t)
                self._to("tuck", t)
                self._dropped = self.p.missed_pick_straight
                self._back_due = True
            elif (self._gripped(senses) and self._grip_held_for(senses)
                  and self._grip_is_a_grasp(senses)
                  and self._grip_is_deep(senses)
                  and (not p.lift_needs_live_contact
                       or self._gripped_raw(senses))):
                self._carry_from = (
                    {j: float(senses.arm.get(j, 0.0)) for j in moss.ARM_JOINTS}
                    if (p.ramp_from_achieved and senses.arm is not None)
                    else dict(self._policy_cmd or {}))
                self._to("lift", t)
            else:
                if (p.pick_at_control_hz and t - self._last_policy_t
                        < 1.0 / moss.CONTROL_HZ - 1e-6
                        and self._policy_cmd is not None):
                    twist, arm = self._held_twist, dict(self._policy_cmd)
                else:
                    self._last_policy_t = t
                    twist, arm = self._policy_step(senses, self._fix)
                note = f"creep: learned pickup, can {self._fix[0]:.2f} m"
                # THE LAST 30 CM BELONG TO THE ARM. The policy keeps its arm
                # channels either way — only the tracks are silenced, and
                # only once the can is inside the reach the grasp pose was
                # measured at.
                if (p.still_when_in_reach
                        and float(self._fix[0]) <= p.reach_stop_m):
                    k = p.reach_twist_scale
                    twist = (twist[0] * k, twist[1] * k, twist[2] * k)
                    note = (f"creep: in reach at {self._fix[0]:.2f} m — "
                            f"tracks at {k:.0%}, arm closing")

        elif self.state == "creep":
            if fix is None or since > p.creep_timeout_s:
                # The can did not arrive: it was knocked aside, or it is
                # somewhere this base cannot drive to. Count the attempt.
                self._attempts += 1
                if self._attempts >= p.max_retries:
                    self._give_up(t)
                    note = "gave up on this one"
                self._to("tuck", t)
                self._dropped = self.p.missed_pick_straight
                self._back_due = True
            else:
                x, y = fix
                self._target_world = self._world((x, y)) or self._target_world
                if x <= p.grasp_at:
                    self._to("close", t)
                else:
                    twist = (p.creep_mps, 0.0, p.creep_kp * math.atan2(y, x))
                    note = f"creep: driving the can in, {x:.2f} m"

        elif self.state == "close":
            arm = {moss.GRIPPER_JOINT: moss.GRASP_JAW_CTRL_M}
            note = "close: gripping"
            if since >= p.close_s:
                self._to("lift", t)

        elif self.state == "lift":
            arm = self._ramp(moss.LIFT_POSE, since, p.lift_s)
            note = "lift"
            z = (senses.target_obs or {}).get("z") if p.camera_hold_check else None
            if z is not None and self._lift_z0 is None:
                self._lift_z0 = float(z)
            if (z is not None and self._lift_z0 is not None
                    and p.lift_rise_after_s <= since <= p.lift_rise_until_s
                    and float(z) - self._lift_z0 < p.lift_min_rise_m):
                return self._abort_drop(t, "lift (camera: not rising)",
                                        direct=True)
            if p.lift_abort_on_drop and since > p.lift_drop_grace_s:
                if not self._gripped(senses):
                    self._empty_since = self._empty_since or t
                    if t - self._empty_since >= p.stow_drop_grace_s:
                        return self._abort_drop(t, "lift", direct=True)
                else:
                    self._empty_since = None
            if since >= p.lift_s:
                self._carry_from = dict(zip(moss.ARM_JOINTS, moss.LIFT_POSE))
                # Hand the learned stow the pose its env starts from: the
                # lift pose with the jaws shut on the can. `_policy_step`
                # only seeds `_policy_cmd` when it is None, and coming out of
                # the pickup it still holds whatever THAT policy ended on —
                # which is not where the stow policy was trained to begin.
                if self._use_learned_stow:
                    # KEEP THE PICKUP'S JAW COMMAND. Seeding this from
                    # `_prev_grip_cmd` handed the stow policy 41 mm — the
                    # initial wide-open value, never updated — while the
                    # fingers were physically at 8 mm, shut on the can. The
                    # policy inherits the command, so its first tick drove
                    # the jaws open and the can fell: measured, the stow leg
                    # lasted 0.28 s and the loop delivered 0 of 9 while the
                    # same policy scored 9/12 in its own env, whose reset
                    # hands it 27 mm. Only the ARM moves to the lift pose;
                    # whatever closed on the can stays closed.
                    # AND CLEAR THE LAST ACTION. `o[OBS_LAST_ACTION]` is
                    # eight of the thirty-two slots, and every stow episode
                    # its env ever ran began with them at ZERO. Handing the
                    # stow policy the PICKUP's last action — a saturated
                    # vector from a different task — is eight slots of
                    # out-of-distribution input on its very first tick, and
                    # it answers the way a policy does off its manifold:
                    # measured in the room, every channel saturated at +/-1
                    # and the jaws went 0 -> 20 -> 40 mm in 0.2 s, dropping
                    # the can before the leg had begun.
                    self._act = np.zeros(moss.NUM_ACTIONS, np.float32)
                    held_jaw = (self._policy_cmd or {}).get(
                        moss.GRIPPER_JOINT, moss.GRASP_JAW_CTRL_M)
                    self._policy_cmd = {
                        **_pose(moss.LIFT_POSE),
                        moss.GRIPPER_JOINT: held_jaw}
                    # And FORGET the can. Its env resets the tracker at the
                    # start of every stow episode and the detector never
                    # re-acquires a can that is inside the jaws, so the
                    # policy learned the whole carry with those slots at
                    # zero. Carrying the approach's fix into the stow leg
                    # hands it an observation it has never seen.
                    self._track_fix = None
                    self._track_t = -1e9
                    self._fix = None
                self._to("stow", t)

        elif self.state == "stow" and self._use_learned_stow:
            # THE LEARNED STOW drives here, arm and jaws together, until it
            # lets go. The scripted version below ramps to `moss.DROP_POSE`,
            # which is a pose the arm CANNOT REACH while carrying: it stalls
            # 0.379 rad short against the bin's own front wall, and opening
            # the jaws from that jam flings the can clear of the bin. That
            # branch stays only for a lab with no stow policy on disk.
            twist, arm = self._policy_step(senses, fix, "stow")
            note = "stow: learned"
            if not self._gripped(senses):
                # It opened the jaws — the delivery is its own business now.
                self._carry_from = None
                self._policy_cmd = None
                self.picked += 1
                self._attempts = 0
                self._target_world = None
                self._fix = None
                self._to("tuck", t)
            elif since >= p.stow_policy_s:
                self._to("release", t)

        elif self.state == "stow":
            # UP, ROUND, DOWN — not a straight ramp to `moss.DROP_POSE`.
            # That pose cannot be reached while carrying: the arm stalls
            # 0.379 rad short against the bin's own front wall and opening
            # the jaws from that jam flings the can clear of the bin. The
            # rotation has to happen ABOVE the bin. Ramped along these three
            # waypoints and opened at the end, this delivers 6 of 6 in the
            # stow env, where the straight ramp delivers 0 of 10.
            drop_pose = ME.STOW_RELEASE_HIGH if p.release_high else ME.STOW_INSIDE
            turn_s = (p.pinch_turn_s if (self._pinched and p.pinch_turn_s)
                      else p.stow_turn_s)
            high, turned = ME.STOW_HIGH, ME.STOW_TURNED
            if p.stow_clear_lift is not None:
                high = (high[0], p.stow_clear_lift) + tuple(high[2:])
                turned = (turned[0], p.stow_clear_lift) + tuple(turned[2:])
            legs = ((high, p.stow_high_s),
                    (turned, turn_s),
                    (drop_pose, p.stow_down_s))
            t_end = 0.0
            arm = None
            for pose, dur in legs:
                if since < t_end + dur:
                    if self._leg_from is None or self._leg_i != legs.index(
                            (pose, dur)):
                        self._leg_i = legs.index((pose, dur))
                        self._leg_from = dict(self._carry_from or {})
                    k = self._ease(min(1.0, (since - t_end) / dur))
                    arm = {j: (1.0 - k) * self._leg_from.get(j, v) + k * v
                           for j, v in zip(moss.ARM_JOINTS, pose)}
                    break
                t_end += dur
                self._carry_from = dict(zip(moss.ARM_JOINTS, pose))
                self._leg_from = None
            note = "stow: up, round, down (scripted)"
            # THE CAMERA'S OPINION: an object the wrist camera sees down near
            # the floor for `stow_drop_grace_s` is not in the hand, whatever
            # the pads report.
            zc = (senses.target_obs or {}).get("z") if p.camera_hold_check else None
            if zc is not None and float(zc) < p.stow_min_z_m and since > p.stow_drop_grace_s:
                self._low_since = self._low_since if self._low_since is not None else t
                if t - self._low_since >= 0.3:
                    self._low_since = None
                    return self._abort_drop(t, "stow (camera: on the floor)",
                                            direct=self._leg_i in (-1, 0))
            else:
                self._low_since = None
            # IS THERE STILL A CAN? Asked every tick of the carry, because
            # the alternative is finishing the delivery with empty jaws and
            # then driving off as though it had worked.
            if p.stow_abort_on_drop and since > p.stow_drop_grace_s:
                if not self._gripped(senses):
                    self._empty_since = getattr(self, "_empty_since", None) or t
                    if t - self._empty_since >= p.stow_drop_grace_s:
                        self._empty_since = None
                        self._carry_from = None
                        self._leg_from = None
                        self._policy_cmd = None
                        # Past the first leg the arm is turning over (or into)
                        # the bin: the straight path home runs through it —
                        # MEASURED, 305 of 504 ticks against the bin and a
                        # 5.8 rad/s whip. The learned fold routes around it.
                        return self._abort_drop(
                            t, "stow", direct=self._leg_i in (-1, 0))
                else:
                    self._empty_since = None
            if arm is None:
                # THE CLOCK IS NOT THE ARM. This released as soon as the
                # three ramps had run, which commands the pose but does not
                # wait for it: following the carried can showed it at
                # x = +0.151 +- 0.319 m when the jaws opened, still out in
                # front, with the bin's mouth behind at x = -0.087. Hold the
                # final pose and let go once the arm is actually there.
                arm = _pose(drop_pose)
                reached = False
                if senses.arm is not None:
                    reached = max(
                        abs(float(senses.arm.get(j, 0.0)) - v)
                        for j, v in zip(moss.ARM_JOINTS, drop_pose)
                    ) < p.stow_tol_rad
                note = "stow: holding over the bin"
                if reached or since >= t_end + p.stow_settle_s:
                    self._to("release", t)

        elif self.state == "release":
            arm = {moss.GRIPPER_JOINT: 0.041}
            if p.release_open_s > 0.0:
                if self._release_from is None:
                    self._release_from = float((senses.arm or {}).get(
                        moss.GRIPPER_JOINT, moss.GRASP_JAW_CTRL_M))
                k = min(1.0, since / p.release_open_s)
                arm = {moss.GRIPPER_JOINT: self._release_from
                       + k * (0.041 - self._release_from)}
            note = "release: into the bin"
            if since >= p.release_s:
                self._carry_from = None
                self.picked += 1
                self._attempts = 0
                self._target_world = None
                self._fix = None
                self._to("tuck", t)

        elif (self.state == "tuck" and p.plan_routes and self._mv is not False
              and (mv := self._move_to(self._rest(), senses, since,
                                       self._fold_hubs())) is not None):
            # HOME BY A CLEAR ROUTE (`moss_motion`): from wherever the arm is —
            # after a delivery, a drop or a missed pick — straight to rest if
            # that clears the visible arm, else through a hub; minimum-jerk,
            # leashed. The folds below run only when no clear route exists.
            arm, arrived = mv
            arm[moss.GRIPPER_JOINT] = moss.MISSION_OPEN_M
            note = "tuck: home by a clear route"
            if (arrived and self._folded(senses)) or since >= p.fold_policy_s:
                self._dropped = False
                self._drop_from = None
                self._route = None
                self._policy_cmd = None
                self._after_tuck(t)

        elif self.state == "back_off":
            # THE THING IT JUST DROPPED IS IN FRONT AND TOO CLOSE TO SEE.
            # `drop_back_m` has the measurement; the short version is that a
            # drop lands a median 0.293 m ahead and `min_x` blinds the brain to
            # anything nearer than 0.30, so 55% of drops become invisible and
            # the robot drives over what it just put down. Reversing restores
            # the RANGE the detector needs, and the object is then the nearest
            # thing in front — no re-prioritising needed, the ordinary search
            # picks it up first.
            if self._back_from is None and self._odom is not None:
                self._back_from = (self._odom[0], self._odom[1])
            gone = (0.0 if self._back_from is None or self._odom is None else
                    math.hypot(self._odom[0] - self._back_from[0],
                               self._odom[1] - self._back_from[1]))
            left = p.drop_back_m - gone
            wall = self._rear_blocked(left)
            # The arm is already folded (this state is entered from the drop's
            # tuck); hold it there on the pacing every other leg uses.
            arm = self._ramp_from_here(self._rest(), since, p.tuck_ramp_s, senses)
            if wall or left <= 0.0 or since >= p.drop_back_max_s:
                self._back_from = None
                note = ("back off: something solid behind — looking from here"
                        if wall else f"back off: {gone:.2f} m, looking again")
                twist = (0.0, 0.0, 0.0)
                self._to("search", t)
            else:
                twist = (-p.drop_back_mps, 0.0, 0.0)
                note = (f"back off: {gone:.2f}/{p.drop_back_m:.2f} m so it can "
                        "be seen again")

        elif self.state == "tuck" and self._dropped:
            # STRAIGHT HOME after a drop, at the commanded joint rate, from
            # wherever the arm IS (snapshotted on the first tick).
            if self._drop_from is None:
                arm_now = senses.arm or {}
                self._drop_from = {j: float(arm_now.get(j, v)) for j, v in
                                   zip(moss.ARM_JOINTS, moss.tuck_pose())}
            tuck = dict(zip(moss.ARM_JOINTS, moss.tuck_pose()))
            step = p.drop_tuck_rate * since
            arm = {j: (tuck[j] if abs(tuck[j] - q0) <= step
                       else q0 + math.copysign(step, tuck[j] - q0))
                   for j, q0 in self._drop_from.items()}
            # The same leash as the fold: a blocked joint must not store the
            # ramp's advance and release it as a whip.
            arm_now = senses.arm or {}
            for j in list(arm):
                if j in arm_now:
                    q = float(arm_now[j])
                    arm[j] = float(np.clip(arm[j], q - p.fold_leash_rad,
                                           q + p.fold_leash_rad))
            arm[moss.GRIPPER_JOINT] = 0.041
            note = "tuck: dropped it — straight home"
            if self._folded(senses) or since >= p.drop_tuck_max_s:
                self._dropped = False
                self._drop_from = None
                self._after_tuck(t)

        elif self.state == "tuck" and self._fold_route(senses) is not None:
            # THE SHORT ROUTE (see `fold_route`): release pose -> waypoint ->
            # tuck, straight segments, every joint arriving together, the
            # slowest at `fold_route_rate`.
            route = self._route
            s_goal = p.fold_route_rate * since
            arm = dict(zip(moss.ARM_JOINTS, route[-1]))
            for a, b in zip(route, route[1:]):
                L = float(np.abs(np.subtract(b, a)).max())
                if s_goal < L:
                    k = s_goal / max(L, 1e-9)
                    arm = {j: float(qa + (qb - qa) * k)
                           for j, qa, qb in zip(moss.ARM_JOINTS, a, b)}
                    break
                s_goal -= L
            arm_now = senses.arm or {}
            for j in list(arm):
                if j in arm_now:
                    q = float(arm_now[j])
                    arm[j] = float(np.clip(arm[j], q - p.fold_leash_rad,
                                           q + p.fold_leash_rad))
            arm[moss.GRIPPER_JOINT] = moss.MISSION_OPEN_M
            note = "tuck: folding home by the short route"
            if self._folded(senses) or since >= p.fold_policy_s:
                self._route = None
                self._after_tuck(t)

        elif self.state == "tuck" and "fold" in self._sessions:
            # THE LEARNED FOLD. Seeded the way its env starts a fold: command
            # = the MEASURED arm pose (not a nominal start pose — it takes
            # over from wherever the delivery left the arm), jaws open, last
            # action zero, no target fix.
            if not self._fold_seeded:
                self._fold_seeded = True
                arm_now = senses.arm or {}
                self._policy_cmd = {
                    **{j: float(arm_now.get(j, moss.ARM_HOME[j]))
                       for j in moss.ARM_JOINTS},
                    moss.GRIPPER_JOINT: moss.MISSION_OPEN_M}
                self._act = np.zeros(moss.NUM_ACTIONS, np.float32)
                self._track_fix = None
                self._track_t = -1e9
                self._last_policy_t = -1e9
            # AT ITS OWN 25 Hz. Run every 50 Hz world tick, its integrated
            # command would move at 1.5 rad/s instead of the 0.75 it trained
            # at — the fast motion this leg exists to avoid.
            if t - self._last_policy_t >= 1.0 / moss.CONTROL_HZ - 1e-6:
                self._last_policy_t = t
                _tw, arm = self._policy_step(senses, fix, "fold")
                arm_now = senses.arm or {}
                for j in moss.ARM_JOINTS:
                    if j in arm_now:
                        q = float(arm_now[j])
                        arm[j] = float(np.clip(arm[j], q - p.fold_leash_rad,
                                               q + p.fold_leash_rad))
                self._policy_cmd.update(
                    {j: arm[j] for j in moss.ARM_JOINTS})
            arm = dict(self._policy_cmd)
            twist = (0.0, 0.0, 0.0)          # the fold does not drive
            note = "tuck: learned fold"
            if self._folded(senses) or since >= p.fold_policy_s:
                self._policy_cmd = None
                self._after_tuck(t)

        elif self.state == "tuck":
            # VIA THE WAYPOINT. Folding straight to the tuck pose jams the arm
            # on the bin — this state's own note has recorded that for months
            # ("0.74 rad short and riding 75 mm proud"), and it is not a
            # tuning failure: the bin sits between the two poses, so no
            # monotone path exists. `moss.RETRACT_WAYPOINT` is a planner's
            # answer that connects 14/14 real post-delivery poses to the tuck,
            # and routing through it folds the arm home on 14/24 against 0/24
            # direct.
            _half = max(p.tuck_s * 0.45, 0.6)
            _goal = (moss.RETRACT_WAYPOINT if since < _half
                     else moss.tuck_pose())
            arm = self._ramp_from_here(_goal, since, p.tuck_ramp_s, senses)
            # NOT "arm inside the shell", which is what this said and what
            # the contact sheets then reported: MEASURED, the arm jams on the
            # hull 0.74 rad short of `TUCK_POSE` and rides ~75 mm proud of
            # the chassis. A note is read as an observation, so it may not
            # claim a pose the robot never reaches.
            note = ("tuck: swinging clear of the bin" if since < _half
                    else "tuck: folding the arm home")
            if since >= p.tuck_s:
                self._after_tuck(t)

        # ROUND THE BYSTANDERS, and only while DRIVING. Not in `creep`,
        # `deploy` or `pinch`: there the rover is deliberately closing the
        # last few centimetres on its own target and contact is the job, not
        # a fault. One place rather than in each branch, because `search`
        # alone produces a twist from three of them (the spin, the patrol
        # leg and the rim walk) — the last hook here that lived in one branch
        # was dead and the A/B came back byte-identical (`_after_tuck`).
        if p.bystander_avoid_m > 0.0 and self.state in ("search", "approach"):
            twist, note = self._dodge(twist, note)

        # The PREVIOUS values update LAST, after the observation has used
        # them. Updating them at the top — which this did for one run — makes
        # `prev` equal to `now` by the time `_policy_obs` differences them,
        # so all seven joint velocities and the yaw rate were identically
        # zero in the brain while the env fed the policy real ones (std 0.11
        # to 0.28). Measured by diffing the two published observations slot
        # by slot; nothing in the code reads wrong until you do that.
        if senses.arm is not None:
            self._prev_arm = dict(senses.arm)
        if senses.odom is not None:
            self._prev_yaw = senses.odom[2]
        self._prev_t = t
        # THE CARRY HAS TO KEEP SAYING IT. `WorldRobot.set_arm` replaces the
        # whole mapping every tick, so a state that names only the arm hands
        # the gripper back to whatever the driver last stored — which coming
        # out of a learned creep is a full squeeze. A learned stow commands
        # the jaw itself and is left alone.
        if (p.hold_jaw and self.state in ("lift", "stow")
                and isinstance(arm, dict)
                and moss.GRIPPER_JOINT not in arm):
            jaw = p.carry_jaw_m
            if p.carry_jaw_relative:
                if self._carry_jaw is None and senses.arm is not None:
                    got = float(senses.arm.get(moss.GRIPPER_JOINT,
                                               p.carry_jaw_m))
                    self._carry_jaw = max(0.0, got - p.carry_interference_m)
                if self._carry_jaw is not None:
                    jaw = self._carry_jaw
            arm = {**arm, moss.GRIPPER_JOINT: jaw}
        arm = self._slew(arm, t)
        self._remember_arm(arm, t)
        return Intent(twist=twist, arm=arm, note=note)

    def inputs(self) -> dict:
        """What the inspector shows: the fix this brain is acting on."""
        return {
            "can": {"value": list(self._fix) if self._fix else None,
                    "age": None if self._fix is None else 0.0},
            "picked": {"value": self.picked, "age": 0.0},
            "written_off": {"value": len(self._written_off), "age": 0.0},
            "attempts": {"value": self._attempts, "age": 0.0},
            **({"map": self.map_payload(self._prev_t)}
               if self._mem is not None else {}),
        }


def _pose(q) -> dict[str, float]:
    """A joint-name mapping from one of `robots/moss.py`'s measured poses."""
    return dict(zip(moss.ARM_JOINTS, q))


REGISTRY.register("tidy_moss", TidyMoss)
