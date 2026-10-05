"""DRIBBLE: walk the ball where you are told, instead of striking it away.

**Why this recipe exists.** Roadmap A.4 measured the brain's `push` mode — walk
THROUGH the ball rather than kick it — against the shipped kick over 48 paired
seeds of 2v2, and it won on every ball measure: possession +3.15 s/min
(p<0.001, better on 38/48), ball advance +0.080 (p=0.003), signed progress
+0.073 (p=0.011). It cannot ship for one reason, in A.4's own words: **"a push
has no aim."** The duck walks through the ball wherever it happens to stand,
and near its own mouth that is into its own net — ten own goals on the fresh
block against none. This is the aim, trained.

**And it exists because the match brain is the wrong harness.** Measured on
`pitch-solo` (120 s, one duck, one ball, no opponents), the shipped chase brain
spends 29.5 % of the run FINDING the ball, 49.2 % going to it, 19.5 % retreating
or blocked — and **1.8 % actually hitting it**. Fixing the gaze
(`gaze_neck=1.0`, roadmap 12aw) more than doubles possession (12.9 → 27.7
s/min) and all but removes the hunting (turn 7.9 → 0.9 %), and the share that
touches the ball still HALVES, to 0.9 %: the time moves into `lineup`. A skill
that is 1-2 % of its own test cannot be measured there. Here the ball starts at
the feet and the clip ends when it is lost, so the whole episode is the skill.

**The aim is in the twist command, and that is not a detail.** AGENTS.md's
first reward rule is never to pay for what the policy cannot observe, and the
61-obs contract carries no yaw and no world position — so "dribble toward the
goal" is unlearnable here, and a heading anchored to the odometry frame is the
term that once produced 30 M steps of circling. What IS observable is
obs[48:50], the body-frame twist command the walker already reads. So the aim
is a DIRECTION IN THE DUCK'S OWN FRAME, and a brain that wants the goal turns
the body and commands forward — which is how every walker here is already
driven.

**The pay is the kick's five terms, RETARGETED — none deleted.** This is the
most repeated mistake in the repo (AGENTS.md: five retrains in one day), so
each one is written down with what it used to buy and what it buys now:

  * `ball_forward` -> `ball_along`. The kick pays for the ball leaving along a
    line latched at reset; a dribble pays for it moving along the COMMANDED
    direction, and at a dribble's speed (0.35 m/s, not the strike's 1.0).
  * `ball_overshoot` kept and retargeted to the new, lower target. A dribble
    that becomes a kick has lost the ball, so the dock is doing MORE work here
    than it did for the strike, not less.
  * `support_foot` -> `step_dont_skid` (the catalog's). The kick pays for the
    OTHER foot staying planted because it is a one-shot strike from standing.
    A dribble walks, so that target is simply wrong — but what the term was
    protecting (do not hop, do not skid-steer) is exactly the catalog term,
    which is the retarget rather than a deletion.
  * `legs_home` -> `no_limit_parking`. "Legs near the standing pose" pays MOST
    for standing still, which is this task's cheapest failure, so leaving it
    on would price the failure as success. What it really bought on the kick
    was a sane configuration, and that is the catalog's end-stop dock.
    (`_dribble_term_audit` in tests prints both terms at the spawn pose and at
    a walking pose; the rule is to print the numbers, not to argue.)
  * `head_home` -> `gaze_ball`. The kick pays for the head coming HOME. A
    dribble must be looking at the ball, and 12aw measured that the gaze is
    the binding constraint on ever seeing it — so this term pays for the ball
    sitting near the optical axis. The reward reads the true geometry
    (privileged, as every reward here does); the obs never does.

...plus `ball_close`, which has no kick analogue because the kick WANTS the
ball gone. It peaks at 0.13 m rather than 0 on purpose: a ball directly under
the trunk is one the feet trip over, not one under control.

**The terminal is the point.** `terminate_fn` ends the clip when the ball is
further than `DRIBBLE_LOST`. Without it the cheapest policy is to shove the
ball away on step one and bank the pose terms for the rest of the clip — the
same failure DeepMimic early termination fixed for the G1 kick (0.10 -> 0.62 m).
The ladder is SPAWNS AND COMMANDS ONLY, never the pay (AGENTS.md).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .. import contract as C
from .ball import _ball_camera, _ball_knob
from .core import (
    CATALOG,
    Behavior,
    CurriculumStage,
    RewardTerm,
    _register,
    _spawn_knob,
    _trunk_yaw,
)
from .kick import BALL_Z, _box_knob, _kick_ball_ids
from .lastmetre import LM_DETECT_EVERY, LM_MAX_RANGE, LM_MEM_TAU, _lm_sense  # noqa: F401
from .locomotion import _run_pose, _run_speed, _run_upright

DRIBBLE_TARGET = 0.35        # m/s along the command: a dribble, not a strike
DRIBBLE_KEEP = 0.13          # m — where `ball_close` peaks: just ahead of the feet
DRIBBLE_KEEP_STD = 0.10
DRIBBLE_LOST = 0.45          # m — past this the ball is out of the dribble band
# ...and past THIS it is unrecoverable and the clip ends. The gap between the
# two is the RECOVERY CORRIDOR, and it exists because the first version had
# none: `terminate_fn` was `_dribble_lost`, so the ball leaving 0.45 m ended
# the episode and handed the policy a fresh one. That deleted the only state
# in which SIGHT PAYS — the duck never had to re-find anything.
#
# The measurement that forced this (12ax F, `ed5413` at 2.93 M): blinding the
# detector completely (VFOV 2 deg, sighting rate 0.0 %) changed nothing —
# 7/20 full clips either way, 0.70 targets a clip either way, ep_len 250.9
# blind against 231.2 sighted. The perception path was UNUSED, so no weight on
# `gaze_ball` could have mattered: a task solvable blind can only ever BUY a
# head tilt, never need one. And the arithmetic said looking was a mistake —
# a deep gaze paid `gaze_ball` +5.4 over an episode and cost `stay_upright`
# -269.2, because the shorter clip forfeits every ungated per-step term.
# Sight was worth 2 % of what it cost, and the policy was right to refuse it.
#
# Ending the clip later is what removes that cost. It is the WORLD changing,
# not the pay — the rule this repo has already paid for twice.
DRIBBLE_GONE = 1.20          # m — past this the ball is gone and the clip ends
DRIBBLE_SEEK_SPEED = 0.25    # m/s toward a lost ball that counts as a full pace
# Width of the `gaze_ball` bell about the optical axis. MEASURED, not picked:
# a ball at `DRIBBLE_KEEP` (0.13 m) sits atan2(0.21 - 0.035, 0.13) = 53 deg =
# 0.93 rad BELOW a level camera, so a bell at 0.45 paid e^-4.3 = 0.014 at the
# spawn and 0.000 at the target state — flat exactly where the policy is, which
# is the "term left ON at a width that is flat where the policy starts" bug
# AGENTS.md names. Worse, it paid 0.944 for a ball KICKED AWAY, because a
# distant ball IS near a level axis: the term was pricing the failure as
# success. At 1.0 the spawn pays ~0.42 and pitching the head down to where the
# brain can actually see the ball climbs it toward 1.0, which is a gradient.
DRIBBLE_GAZE_STD = 1.00      # rad
# m at obs[52] = 1.0. KEPT at 0.50 now that the ball CAN be far (the recovery
# corridor runs to 1.20 m), and the old comment here — "the ball is never far
# in this task" — is no longer true. Deliberate, for two reasons: the slot is
# clipped into [0, 1] by `_lm_sense`, so nothing out there is out of the
# donor's distribution and no normalizer repair is needed; and a slot pegged
# at 1.0 IS the mode indicator ("outside the dribble band"), while the
# quantity a recovery actually steers on is the BEARING in obs[51]. The cost
# is that 0.5 m and 1.2 m read alike. If recovery stalls, this is the first
# knob to try — raising it to 1.20 resolves the corridor and coarsens the
# near field, and that trade has not been measured.
DRIBBLE_RANGE_SCALE = 0.50

# Spawn and command windows. Every one is a stage knob; none of them is pay.
DRIBBLE_AHEAD = (0.10, 0.18)
DRIBBLE_SIDE = (-0.09, 0.09)
DRIBBLE_SPEED = (0.12, 0.25)     # commanded speed, m/s
DRIBBLE_BEARING = 0.50           # rad — how far off the nose the command may point
# THE BALL'S ROLLING FRICTION — the physics rung, and on measurement the one
# that decides whether this task is a dribble at all.
#
# Upstream's ball.xml sets rolling friction to 0.0001 with the comment "low
# rolling resistance so a kicked ball actually rolls away". That is the right
# ball for a STRIKE and the wrong one for a dribble, and the cost was not
# visible until it was counted. On `runs/dribble-nobackward`, deterministic,
# BAM, 12-16 clips, only this coefficient changed (touches counted on the
# RISING edge of a real ball-to-duck contact, so a contact held over four
# steps is one touch):
#
#     rolling    touches/clip   steps in contact   travel per touch
#     0.002 (scene)     1.33               1.0%           0.069 m
#     0.002 (forced)    1.33               1.0%           0.069 m   <- the CONTROL
#     0.01              3.75               2.2%           0.079 m
#     0.05             12.00               8.5%           0.044 m
#
# and the shipped ball gives 1.33 touches per TARGET reached (targets sit
# 0.30-0.55 m out with a 0.15 m reach; one touch rolls the ball ~0.25 m), and
# 1.12 touches in a 20 s clip against 1.25 in an 8 s one — about one contact
# per episode however long the clip. So a multi-touch sequence was never in
# ANY rollout, which is why eight rounds of reward surgery could not teach
# one: an unsampled state's value is never learned. Ladder the ball, not the
# pay.
#
# READ THIS BEFORE CHANGING THE DEFAULT. Upstream's ball.xml says 0.0001, but
# the COMPOSED scene this repo trains on says 0.002 — the 2026-09-06 rolling
# friction fix raised it, and `contract.scene_walk_ball_xml` is where the ball
# actually comes from. The default here must be the COMPOSED value, not the
# upstream file's, or this hook silently makes the ball roll FREER than the
# lab's own ball: with 0.0001 the same policy's ball travelled 1.44 m a clip
# against 0.83 m, and `teach-dribble-f14433` trained its last rung on a ball
# no other part of this repo uses. `test_the_default_matches_the_composed_scene`
# is the guard; it is the test that caught it.
#
# The two forced rows above (scene vs forced 0.002) coming back bit-identical
# is the probe's own CONTROL, not a solver dead zone — it is the same number
# twice. An earlier draft of this comment read it as a dead band and argued
# that nothing below 0.002 can bite; that claim is unmeasured, so do not rely
# on it.
#
# Requires condim 6 on the ball geom, which `contract.scene_walk_ball_xml`
# patches in; under condim 3 MuJoCo ignores the coefficient entirely and this
# whole knob is a no-op (it has been missed twice in this repo).
DRIBBLE_BALL_ROLLING = 0.002     # the COMPOSED scene's ball (not ball.xml's 0.0001)
DRIBBLE_ROLL_LADDER = (0.05, 0.01, DRIBBLE_BALL_ROLLING)

DRIBBLE_EPISODE_S = 8.0          # a dribble needs strides; the kick's 2 s cannot hold one

# THE SPAWN GAZE — it decides whether this recipe has an observation at all,
# and it has been set three different ways for three different reasons. All
# three are kept because each one is right for a different donor.
#
# MEASURED (120 spawns a window, the real camera through `_ball_camera`):
#
#     neck    head    ball in frame at spawn
#    +0.00   +0.00            0 %
#    -0.20   +0.55           49 %
#    -0.30   +0.70           98 %
#    -0.38   +0.97          100 %
#    MIX (below)             40 %
#
# 1. HOME-only (0 %) was the first draft and is a TRAP: obs[51:55] carried
#    nothing for a whole 6M-step run, `_lm_world` was never set, and no reward
#    weight reaches a state the rollouts never contain.
# 2. DEEP-only (100 %) fixes that and is what a from-scratch run should use —
#    but it breaks a COLD warm start, because a distilled walker reads
#    `head_pitch` through a VecNormalize that has never seen it 0.65-1.10 rad
#    off HOME. Measured: fell 20/20 by 550k steps from a donor that falls
#    0/20, in all four combinations of this knob and the spawn velocity.
# 3. HOME-only again was the cold-warm-start workaround, and it cost the
#    ghost entirely: `seen` measured 0 % across a finished 6M run.
#
# The MIX below is for the case that actually applies now — a fine-tune from a
# DRIBBLE policy, whose normalizer has already seen these poses, so the
# constraint in (2) has lifted. Roughly 40 % of episodes start with a sighting
# to reinforce and the rest start blind, which samples both instead of betting
# the run on one. Ladder the world, not the pay.
# The GENTLE mix, not the full one. Same measurement, same lesson: with the
# full window (-0.45..0 / 0..0.95) the incumbent fell 3 of 14 even at the gentle
# turn gain, because a head pitched most of a radian is a pose it has never
# balanced from. The gentle mix keeps 0 falls over 14 clips and still puts the
# ball in frame on some spawns, which is what `gaze_ball` needs to reinforce.
# Widen it once the policy has learned to look — the full window is
# DRIBBLE_GAZE_DEEP, one knob away.
# MEASURED, and moved on 2026-09-24 because the command now comes from the
# BELIEF (`_ball_believed`): a spawn with the ball out of frame is a spawn with
# NO steering command at all, so this window sets whether the task is
# attemptable. Spawn sighting rate against the window, 60 resets each:
#
#     neck          head          ball in frame at spawn
#     -0.45,-0.30   0.90,1.05     100%   <- DRILL, stage 1
#     -0.45,-0.15   0.65,1.10      98%   <- DEEP, stage 2
#     -0.40,-0.10   0.50,1.10      88%
#     -0.35,-0.05   0.40,1.10      67%   <- the default below, stage 3
#     -0.30, 0.0    0.25,1.10      60%
#     -0.25, 0.0    0.0 ,0.60       2%   <- the OLD default
#
# The old default put the ball in frame on 2 of 100 spawns. Under a truth-fed
# command that cost nothing (the oracle steered anyway, which is exactly the
# defect); under a believed one it would hand 98 % of episodes no command at
# all — the unsampled-state trap this repo has paid for twice, where the
# rollouts never contain the thing being paid for and no weight can fix it.
#
# The default is the TOP rung rather than the easiest, because `--init-from`
# disables the curriculum (`stages = curriculum if init_from is None else ()`),
# so a fine-tune — which is how this recipe is actually trained — sees these
# numbers and never sees a stage knob.
DRIBBLE_GAZE_NECK = (-0.35, -0.05)
DRIBBLE_GAZE_HEAD = (0.40, 1.10)
#: The 100 %-in-frame window, for a from-scratch run (case 2 above).
DRIBBLE_GAZE_DEEP = ((-0.45, -0.15), (0.65, 1.10))
#: Stage 1's DRILL window — `lastmetre.LM_GAZE_STAGE1`'s own, measured at 100 %.
DRIBBLE_GAZE_DRILL = ("-0.45,-0.30", "0.90,1.05")

# SPAWN ALREADY MOVING, as a fraction of the commanded velocity.
#
# The live checkpoint measured `ball_with_me` at **exactly 0.000 across 4800
# steps**: the duck never once moved the ball along the command, so the term
# carrying 12 of the 24 available points had no gradient at all. AGENTS.md's
# rule is that an unsampled state's value is never learned and the fix is the
# WORLD, not the pay — so the episode starts with the duck already at walking
# speed, and the very first rollouts contain a touched ball.
#
# It is also the honest handover: a dribble is entered from a walk, never from
# a standstill, so a policy that has only ever started from rest has a
# precondition nothing in play provides. (This is 12aw's own idea, applied
# where it has something to bite on.)
DRIBBLE_SPAWN_VEL = (0.6, 1.0)

# --- THE TARGET (2026-09-24) ------------------------------------------------
# A DESTINATION, not just a direction. The recipe shipped with a body-frame
# velocity command, which has no place it is trying to get to — so there was
# nothing for the page to draw a goal for, the episode ended on a timer rather
# than an outcome, and the only score was a ratio. A target fixes all three.
#
# **Where it rides, and why it can.** A world position is UNOBSERVABLE here
# (the 61-obs layout carries no yaw and no world frame; AGENTS.md's first
# reward rule), so the target is presented as a BODY-FRAME vector from the
# BALL to the target plus the remaining distance, in `body_pose_cmd`
# (obs[55:58]) — three of six slots that carried keep-alive noise. Putting a
# task's sensing in the command slots is this package's established pattern:
# `find_ball` and `lastmetre` do exactly that with the head slots. The layout
# is untouched and the other three slots keep their noise.
#
# **It updates every step**, unlike a spawn knob: the vector to a fixed point
# rotates as the duck walks, and a value written once at reset would be a lie
# from the second step. `_dribble_command` rewrites it, which is also what
# guards it against `walk_env`'s mid-episode command resample.
#
# **Reaching one RE-SAMPLES rather than ending the clip.** Ending on success
# would forfeit the rest of the episode's income, which AGENTS.md calls the
# biggest possible attempt tax (the headstand's terminal cost it the unfold).
# So the task is a COURSE: reach a waypoint, get another. That keeps episodes
# full length and makes "targets reached per clip" the headline number, which
# is a count of outcomes rather than a ratio of distances.
# 0.30-0.55 m, NOT the 0.8-1.5 this shipped with. That first range made the
# task unreachable and therefore ungradeable: at a commanded 0.12-0.25 m/s an
# 8 s clip is 0.96-2.00 m of PERFECT travel, the measured ball travel was
# 0.41 m, and so the count sat at exactly 0 for every clip of a 6M-step run.
# A score that never leaves zero carries no gradient — the re-sample never
# fires, the course never becomes a course, and the only thing left driving
# the policy is the dense terms. Closer targets mean 2-4 reached in a good
# clip and partial credit in the COUNT, not just in the shaping.
DRIBBLE_TARGET_DIST = (0.30, 0.55)
DRIBBLE_TARGET_BEARING = 1.0       # rad either side of the duck's nose
DRIBBLE_TARGET_REACH = 0.15        # m — the ball this close has arrived
DRIBBLE_TARGET_SCALE = 2.0         # m at obs[57] = 1.0
# The steering law's proportional gain and yaw-rate ceiling, swept below.
# 0.4 / 0.15, not the 1.5 / 0.8 first guessed. MEASURED against the incumbent
# that has to follow it: at 1.5/0.8 the 0-fall walking policy fell 6 of 14 and
# episodes dropped to 70/400; at 0.4/0.15 it falls 0 of 14 and runs 400/400.
# A steering law the donor cannot follow measures the law, not the policy —
# and the aggressive one oscillated about the line anyway (median |wz| 0.42
# commanded for a target DEAD AHEAD).
#: How far BEHIND the ball, along the ball->target line, the duck is steered.
#: Roughly `DRIBBLE_KEEP` so that standing at the spot puts the ball at the
#: distance `ball_close` pays for.
# --- THE NUDGE: something knocks the ball off-line mid-clip ---------------
#
# WHY IT EXISTS, and it is not difficulty for its own sake. The dribble works
# BLIND: measured on the gaze fine-tune, ball/duck ratio 0.92 and a third of
# clips losing the ball, with the optical axis 11 deg down and the ball in
# frame on 2 % of steps. That is not a policy that ignores its eyes by
# accident — it is a task that never punishes blindness, because the ball
# starts at the feet and stays roughly where it is put. No weight on
# `gaze_ball` fixes that honestly: paying a policy to look at something it
# does not need to see buys the LOOKING, not the seeing.
#
# So the world gets the thing sight is for. At a Poisson rate something
# knocks the ball sideways off the line it was travelling. A blind policy
# keeps pushing where the ball used to be and loses it; a policy that looks
# sees the new position in obs[51:55] and can go and get it. The ghost then
# draws a belief that is actually load-bearing rather than decorative.
#
# Sized to be RECOVERABLE, not fatal: the impulse moves the ball a couple of
# tenths of a metre, well inside `DRIBBLE_LOST`, so the clip continues and
# the recovery is what is being scored. A nudge that instantly ended the
# episode would measure the nudge, not the skill.
DRIBBLE_NUDGE_RATE = 0.6         # expected knocks a second (0 = off)
DRIBBLE_NUDGE_SPEED = (0.25, 0.5)  # m/s imparted to the ball
DRIBBLE_NUDGE_SIDE = (0.7, 1.9)    # rad off the ball->target line; mostly sideways

DRIBBLE_BEHIND = 0.14
# How far to the SIDE the come-round spot sits, so an overrun duck arcs around
# the ball instead of walking across it. See the long note at the `else` branch
# in `_dribble_command`: the un-offset version pushed the ball backward at
# -0.043 m/s for 53 % of every clip, exactly cancelling the +0.040 the
# push-through branch earned. Sized just over the ball's 0.035 m radius plus a
# sole's half-width, so the commanded path clears it rather than grazing it.
DRIBBLE_ROUND_SIDE = 0.18
# The slowest ball a duck standing still is allowed to have sent away, m/s.
# Only a floor: `_ball_overshoot`'s real threshold is the duck's own speed, and
# this stops a stationary duck earning a free nudge. Deliberately well UNDER
# the 0.193 m/s the policy walks at — a floor at `DRIBBLE_TARGET` (0.35) made
# the whole term a no-op, since the duck never reaches 0.35.
DRIBBLE_KEEPABLE = 0.10
#: Neck and head offsets off `DEFAULT_POSE` a stage asks the duck to HOLD, and
#: (0, 0) means "no hold" so every run before this one is unchanged.
#:
#: This is the gaze ladder moved INSIDE the recipe, which is what should have
#: been done first. `behaviors/gazewalk.py` proved the gait exists — 62.4 deg of
#: depression at 1/20 falls where the donor falls 20/20 — and proved it does not
#: arrive by transfer: a dribble fine-tuned off it holds the pose (56 deg, ball
#: at touch range in frame 20.5 % against 2.3 %) and falls **40/40**, with
#: targets 0.550 -> 0.375. The dribble env asks for a head-down gait AND a
#: continuously varying steering command AND a ball underfoot, and 1.2 M steps
#: of fine-tuning will not fuse three things learned apart.
#:
#: So the policy never leaves the task: it starts from the best dribbler (39/40
#: full clips, 0.550 targets) and the HOLD ANGLE is laddered while everything
#: else stays where it already works. Rungs read off the clamp table in
#: `gazewalk`: 0.30 buys almost no sight and is where balancing a forward head
#: is learned, 0.60 puts the ball at the feet in frame ~65 %, 0.95 ~91 %.
DRIBBLE_HOLD = (0.0, 0.0)
#: Width of the hold bell, radians. 0.45, the width `gazewalk` measured as
#: giving gradient from a LEVEL head rather than being flat there — the bug
#: this package has now been bitten by three times (`gaze_ball`'s two bells,
#: `turn_track`'s Gaussian, and nearly this).
DRIBBLE_HOLD_STD = 0.45
#: How near a foot must be to the ball for the touch terms to act, m. Measured
#: for density before it was chosen: a foot is within 0.10 m of the ball on
#: 11.8 % of steps and within 0.15 m on 29.1 %, so shaping here is a dense
#: signal rather than the 1.4-touches-a-clip event it would be at contact.
DRIBBLE_NEAR_FOOT = 0.15
#: Foot speed at which a touch stops being a push and becomes a strike, m/s.
#: MEASURED, not picked: ball departure speed equals the foot's speed at
#: contact (ratio 0.98, r=+0.69), the duck walks at 0.193, and contact happens
#: at a median 0.412 — the NINETIETH PERCENTILE of the foot's own speed over a
#: stride, whose median is 0.083. The duck hits the ball with the fastest part
#: of its swing while spending 64 % of the stride under 0.20 m/s. There is a
#: factor of five of room here and nothing was asking for it.
DRIBBLE_SOFT_REF = 0.35
#: Width of the stance bells, m. The duck's dribble stance is BEHIND the ball
#: on the ball->target line, and until now nothing in this recipe paid for it:
#: `_near` is `hypot(ahead, left)`, so the ball beside the duck, behind it, or
#: in front of it on the line all scored identically. Measured consequence —
#: the duck is on the correct side of the ball **48 % of the time, which is
#: chance** — and the user watching the lab put it exactly right: "it just
#: starts to walk forwards and doesn't care about anything else."
DRIBBLE_STANCE_STD = 0.12
#: How much of `_near` survives the ball being on the WRONG side. A floor, not
#: a gate: `_near` feeds `_doing`, which gates six terms, and zeroing it
#: outright would take the whole income to nothing the moment the ball drifts
#: off-line — the over-gating that made falling free in D.
DRIBBLE_SIDE_FLOOR = 0.35
DRIBBLE_TURN_GAIN = 0.4
DRIBBLE_TURN_MAX = 0.15


def _dribble_cmd(env) -> tuple[float, float]:
    """The commanded direction as a BODY-frame unit vector, read off the twist
    command the policy can actually see (obs[48:50]). Falls back to straight
    ahead if the command is zero, so the terms are never divided by nothing."""
    vx, vy = float(env.twist_cmd[0]), float(env.twist_cmd[1])
    n = math.hypot(vx, vy)
    return (1.0, 0.0) if n < 1e-6 else (vx / n, vy / n)


def _ball_vel_body(env) -> tuple[float, float]:
    """The ball's ground velocity in the duck's own yaw frame (+x ahead, +y
    left) — the frame the command is in, so the two can be dotted. Written out
    rather than reused from the kick, whose `ball_speed_along` projects on a
    WORLD line latched at reset (AGENTS.md: a difference between two
    quantities is only an error when they are the same quantity)."""
    _, _, dadr = _kick_ball_ids(env)
    v = env.data.qvel[dadr:dadr + 3]
    yaw = _trunk_yaw(env)
    c, s = math.cos(yaw), math.sin(yaw)
    return (float(v[0]) * c + float(v[1]) * s, -float(v[0]) * s + float(v[1]) * c)


def _ball_offset(env) -> tuple[float, float]:
    """(ahead, left) of the ball from the trunk, in the duck's yaw frame."""
    _, qadr, _ = _kick_ball_ids(env)
    t = env._trunk_xpos
    dx = float(env.data.qpos[qadr]) - float(t[0])
    dy = float(env.data.qpos[qadr + 1]) - float(t[1])
    yaw = _trunk_yaw(env)
    c, s = math.cos(yaw), math.sin(yaw)
    return (dx * c + dy * s, -dx * s + dy * c)


def _duck_vel_body(env) -> tuple[float, float]:
    v = env.body_lin_vel()
    return (float(v[0]), float(v[1]))


def _new_target(env, r=None) -> None:
    """Draw a fresh target for the ball, `DRIBBLE_TARGET_DIST` away and within
    `DRIBBLE_TARGET_BEARING` of the duck's current heading."""
    r = r if r is not None else env._rng
    yaw = _trunk_yaw(env)
    _, q, _ = _kick_ball_ids(env)
    bx, by = float(env.data.qpos[q]), float(env.data.qpos[q + 1])
    d = r.uniform(*_box_knob(env, "MICRODUCK_DRIBBLE_TARGET_DIST", DRIBBLE_TARGET_DIST))
    bmax = _box_knob(env, "MICRODUCK_DRIBBLE_TARGET_BEARING",
                     (DRIBBLE_TARGET_BEARING, DRIBBLE_TARGET_BEARING))[0]
    a = yaw + float(r.uniform(-bmax, bmax))
    env._dribble_target = (bx + d * math.cos(a), by + d * math.sin(a))


def _target_dir_from(env, bx: float, by: float) -> tuple[float, float, float]:
    """(ux, uy, dist) from a GIVEN ball position to the target, world frame."""
    t = getattr(env, "_dribble_target", None)
    if t is None:
        return (1.0, 0.0, 0.0)
    dx, dy = t[0] - bx, t[1] - by
    n = math.hypot(dx, dy)
    return (1.0, 0.0, 0.0) if n < 1e-6 else (dx / n, dy / n, n)


def _target_dir(env) -> tuple[float, float, float]:
    """(ux, uy, dist) from the TRUE ball to the target, in the WORLD frame. The
    ball is the origin and not the duck: what has to arrive is the ball.

    TRUTH, and only the REWARD may read it. The reward is allowed to know where
    the ball really is — AGENTS.md's rule is that a policy must be able to
    OBSERVE what it is paid for, and the ball's position is observable through
    the camera. What may not read truth is anything that ends up in the
    observation: see `_ball_believed`."""
    _, q, _ = _kick_ball_ids(env)
    return _target_dir_from(env, float(env.data.qpos[q]), float(env.data.qpos[q + 1]))


def _ball_believed(env) -> tuple[float, float] | None:
    """WHERE THE BRAIN THINKS THE BALL IS — the tracker's own estimate, held
    and carried by odometry (`_lm_world`), or None if it has not seen it yet
    this episode. This is what the steering command and the published slots
    must be built from, and the reason is a measured defect, not tidiness.

    `_dribble_command` used to read `env.data.qpos[ball]` — the TRUTH — to
    place the dribble spot, then published the resulting heading in
    obs[48:51]. So an oracle told the policy where the ball was, every step,
    through the command channel. Measured: the commanded yaw correlates
    **+0.642** with the true ball bearing, and a permutation ablation (each
    channel's information destroyed, its marginal preserved, 30 clips) says
    that channel is the one carrying the task:

    | channel destroyed | full clips | ball lost |
    |---|---|---|
    | nothing | 28/30 | 0/30 |
    | CAMERA obs[51:55] | 28/30 | 0/30 |
    | ball->target obs[55:58] | 28/30 | 0/30 |
    | **commanded twist obs[48:51]** | **20/30** | **8/30** |

    That is the whole answer to why blinding the camera cost nothing across
    12ax F and G, and why no reward gate, terminal or displacement could ever
    have fixed it: the answer was already in the command. A harder knock just
    moves a ball the oracle keeps pointing at.

    It is also a sim2real defect and not only an experiment-design one. On the
    robot `Chase` steers on the DETECTOR's estimate; a policy trained against
    truth has learned to trust a command that is wrong exactly when the
    detector is stale, which is the moment it matters."""
    if _spawn_knob(env, "MICRODUCK_DRIBBLE_ORACLE") == "1":
        _, q, _ = _kick_ball_ids(env)               # the old truth-fed command
        return (float(env.data.qpos[q]), float(env.data.qpos[q + 1]))
    w = getattr(env, "_lm_world", None)
    return None if w is None else (float(w[0]), float(w[1]))


def _ball_with_me(env) -> float:
    """THE DRIBBLE, as one number: the ball advancing along the command AND
    the duck advancing along it, multiplied.

    A product, not a sum, and that is the whole correction. The first version
    of this recipe paid `ball_along` on its own and nothing at all for the
    duck's own travel — so its optimum was to stand still with the ball 13 cm
    away and collect `ball_close` + `gaze_ball` + `flat_feet` forever, which is
    precisely what it looked like: a duck staring at a ball. Measured on the
    shipped walker over 50 clips, the failure this has to price is not losing
    the ball, it is LEAVING it: the duck travels 0.60 m of a commanded 1.53
    while the ball manages 0.29 — a ball/duck ratio of 0.55, keeping up on
    32 % of episodes. A sum cannot tell "both moved together" from "one moved
    twice as much"; a product is zero unless both are happening.

    `_run_speed` beside it pays for the duck matching the command at all, and
    its own docstring records the same trap from the locomotion side ("a
    from-scratch policy parked itself and collected rent")."""
    # Toward the TARGET now, in the world frame, because that is where the
    # ball has to end up. The product still stands: a ball moving toward the
    # target while the duck stays put is a kick, and a duck walking toward it
    # without the ball is having left it behind.
    ux, uy, _ = _target_dir(env)
    _, _, dadr = _kick_ball_ids(env)
    bvx, bvy = float(env.data.qvel[dadr]), float(env.data.qvel[dadr + 1])
    dv = env.data.qvel[env._root_qvel:env._root_qvel + 2]
    ball = float(np.clip(bvx * ux + bvy * uy, 0.0, DRIBBLE_TARGET)) / DRIBBLE_TARGET
    duck = float(np.clip(float(dv[0]) * ux + float(dv[1]) * uy, 0.0, DRIBBLE_TARGET)) / DRIBBLE_TARGET
    return ball * duck


def _ball_progress(env) -> float:
    """GROUND MADE UP by the ball toward its target this step, normalised.

    THE SHAPE FIX, and the reason every weight change before it failed.
    `_ball_with_me` is a product of INSTANTANEOUS velocities, so it can be
    raised without the ball going anywhere: raising its weight 12 -> 30 took
    the term's own value 0.036 -> 0.094 (x2.6) while the ball's advance toward
    the target stayed at +0.013 m/s and targets a clip went 0.550 -> 0.475. A
    kick-and-chase policy satisfies a velocity product in bursts — one strike
    while it happens to be moving — and the product cannot tell that from a
    dribble.

    This is displacement, not velocity, so an episode's SUM telescopes to the
    distance the ball actually covered. Coincidence-timing earns nothing; only
    moving the ball does. It is the pattern MOSS's pickup already uses ("paid
    for GROUND MADE UP, not for being near the can") and the one this recipe
    should have had from the start.

    A REDRAWN target jumps the distance, so a step whose change exceeds what a
    ball can physically travel (a metre a second is already generous at this
    scale) is a redraw and scores zero rather than a windfall or a phantom
    loss."""
    _, _, dist = _target_dir(env)
    prev = getattr(env, "_dribble_tprev", None)
    env._dribble_tprev = dist
    if prev is None:
        return 0.0
    gain = prev - dist
    if abs(gain) > 0.05:                      # a target redraw, not ball travel
        return 0.0
    # GATED ON STILL HAVING THE BALL, and this gate is the whole term. Ungated
    # displacement is MAXIMISED BY KICKING: one strike banks 0.35 m of travel
    # in a single touch, and the clip ending afterwards forfeits nothing
    # because the reward is already banked. Measured — a 4-rung contact drill
    # whose whole design was to end the clip on a hard touch trained the
    # departure speed UP, 0.400 -> 0.499 m/s inside the drill and 0.460 ->
    # 0.620 on the full task, and took full clips from 29/30 to 16/30. The
    # drill amplified the term instead of correcting it.
    #
    # So I had added a term that rewards the very behaviour it was meant to
    # remove, which is why attempt 2 measured null and the drill measured
    # backwards. With the gate, the ONLY way to bank displacement is to move
    # the ball while it is still within reach — which is the definition of a
    # dribble, and a kick earns only the centimetres it travels before it
    # leaves.
    return float(np.clip(gain / (DRIBBLE_TARGET * C.CTRL_DT), -1.0, 1.0)) * _have(env)


def _ball_along(env) -> float:
    ux, uy = _dribble_cmd(env)
    vx, vy = _ball_vel_body(env)
    return float(np.clip(vx * ux + vy * uy, 0.0, DRIBBLE_TARGET)) / DRIBBLE_TARGET


def _ball_overshoot(env) -> float:
    """<= 0: the ball leaving FASTER THAN THE DUCK CAN FOLLOW.

    RETARGETED, not rescaled, and the old target is why this recipe learned
    kick-and-chase instead of dribbling. Against a fixed 0.35 m/s the term
    fired on 1.4 % of steps for a mean excess of 0.066 m/s — **0.0037 a step**
    against `ball_close`'s 6.0, about a 1600th of it. So the one hard strike
    per clip was free. Measured on the trained policy: **1.3 foot touches per
    8 s clip**, each sending the ball off at **0.436 m/s** while the duck walks
    at **0.193** — 2.3x its own pace, so the ball is gone by construction and
    the rest of the clip is pursuit. The touches are NOT random (median 30 deg
    off the target line, 70 % inside 45 deg); they are too hard and too rare,
    and the ball sits parked at a median 0.009 m/s in between.

    The honest threshold is the duck's OWN ground speed: a ball moving faster
    than the duck cannot be kept, whatever the reward says, so that is the line
    between a dribble touch and a kick. `DRIBBLE_TARGET` stays as the floor, so
    a stationary duck is not handed a free licence to nudge."""
    _, _, dadr = _kick_ball_ids(env)
    v = env.data.qvel[dadr:dadr + 2]
    dv = env.data.qvel[env._root_qvel:env._root_qvel + 2]
    # SPEED, not the component along the target line. The first retarget used
    # the projection and still read -0.0035 a step, because after a strike the
    # target is redrawn and the ball's velocity is no longer along the new
    # line — so the very rolls this is meant to price went unseen. An
    # uncontrollable ball is uncontrollable in every direction.
    #
    # And the floor is DRIBBLE_KEEPABLE, not `DRIBBLE_TARGET`: `max(0.35, duck)`
    # pinned the threshold at 0.35 for a duck that walks at 0.193, which made
    # the whole retarget a no-op. The floor only exists so a STATIONARY duck is
    # not free to nudge the ball away.
    keepable = max(DRIBBLE_KEEPABLE, math.hypot(float(dv[0]), float(dv[1])))
    return -max(0.0, math.hypot(float(v[0]), float(v[1])) - keepable)                 # <= 0


def _going(env) -> float:
    """How much of the commanded speed the duck is actually making, 0..1.

    The gate on every term that a STANDING duck could otherwise collect. Its
    reason is measured, not theoretical: with `ball_close` paying 6.0 flat,
    the live checkpoint at 1.66 M steps had learned to stand perfectly still
    beside the ball — `ball_with_me` averaged **0.000** over 4800 steps, the
    duck travelled 0.093 m in 8 s and the ball 0.000 m, and every episode ran
    the full 400 steps. 6.8 a step for zero risk beats reaching for 20 a step
    with a loss terminal attached, so the policy took it, and `ep_rew` climbed
    142 -> 1230 entirely through episode LENGTH while reward-per-step sat flat
    at 6.2 -> 6.8 (the sum-versus-rate trap this repo already has a name for).

    The earlier audit missed it by testing the wrong failure: "stood still"
    under ZERO actions makes the duck collapse, which scores 1.57 and looks
    priced out. A duck that stands COMPETENTLY holds `ball_close` at 0.96 and
    `flat_feet` at 0.98 and collects ~7. The failure state to audit is the one
    a competent policy can reach, not the one a dead one falls into."""
    ux, uy, _ = _target_dir(env)
    dv = env.data.qvel[env._root_qvel:env._root_qvel + 2]
    cmd = math.hypot(float(env.twist_cmd[0]), float(env.twist_cmd[1]))
    if cmd < 1e-6:
        return 0.0
    return float(np.clip((float(dv[0]) * ux + float(dv[1]) * uy) / cmd, 0.0, 1.0))


def _near(env) -> float:
    """How close the ball is to where a dribble wants it — the raw bell, with
    no gate. Peaks at `DRIBBLE_KEEP` rather than 0 on purpose: a ball directly
    under the trunk is one the feet trip over, not one under control."""
    ax, ay = _ball_offset(env)
    bell = math.exp(-(((math.hypot(ax, ay) - DRIBBLE_KEEP) / DRIBBLE_KEEP_STD) ** 2))
    # DIRECTION, added 2026-09-24. The distance bell alone is blind to WHERE
    # the ball is: 13 cm behind the duck scored the same as 13 cm in front of
    # it on the target line, so "get into position" was worth nothing. The
    # factor is the alignment between duck->ball and duck->target: 1 when the
    # ball lies toward the target, falling to the floor when it is beside or
    # behind. Floored rather than gated, because six terms hang off `_doing`.
    return float(bell) * (DRIBBLE_SIDE_FLOOR
                          + (1.0 - DRIBBLE_SIDE_FLOOR) * _ball_on_target_side(env))


def _ball_on_target_side(env) -> float:
    """0..1: is the ball between the duck and where the ball must go?

    `clip(cos, 0, 1)` of the angle between duck->ball and duck->target. 1 when
    the duck is lined up behind the ball, 0 once the ball is 90 deg off that
    line or behind the duck."""
    ax, ay = _ball_offset(env)
    n = math.hypot(ax, ay)
    t = getattr(env, "_dribble_target", None)
    if t is None or n < 1e-6:
        return 1.0
    yaw = _trunk_yaw(env)
    c, sn = math.cos(yaw), math.sin(yaw)
    tx, ty = t[0] - float(env.data.qpos[0]), t[1] - float(env.data.qpos[1])
    # the target in the duck's own yaw frame, so both vectors share a frame
    fx, fy = tx * c + ty * sn, -tx * sn + ty * c
    m = math.hypot(fx, fy)
    if m < 1e-6:
        return 1.0
    return max(0.0, min(1.0, (ax * fx + ay * fy) / (n * m)))


def _doing(env) -> float:
    """THE ONE GATE: going somewhere AND still holding the ball, 0..1.

    Every positive term here is multiplied by this, and that is the invariant
    to keep. The recipe has now been beaten THREE times by the same shape — a
    term satisfiable without doing the task gets satisfied without doing the
    task:

      1. `ball_close` paid 6.0 flat -> the policy stood perfectly still beside
         the ball (ball_with_me exactly 0.000 over 4800 steps, 0.09 m
         travelled, full-length episodes).
      2. Gating that, the GAIT PACKAGE added to stop a hop paid a still duck
         stay_upright 1.95 + pose 0.83 + flat_feet 2.49 = 5.28 a step -> it
         stood still again, reaching 0 targets in 16 clips.
      3. Gating those on `_going`, `keep_pace` at 8.0 paid +5.94 against
         ball_with_me's +1.00 -> the policy walked beautifully and left the
         ball (0 % airborne, 0.85 m travelled, ball 0.41 m, ratio 0.48 — the
         untrained walker's).

    Each fix moved the income to whatever was still ungated. So the gate is
    now ONE quantity meaning "doing the task", and a new term earns by
    multiplying it or it does not earn at all.

    `_engaged` below is the ONLY widening of this, and it is a widening of the
    same idea rather than a hole in it: with a recovery corridor the task has
    a second half — going and GETTING the ball — that `_near` reads as zero."""
    return _near(env) * _going(env)


def _have(env) -> float:
    """1.0 while the ball is inside the dribble band, 0.0 once it is out.

    The old `_dribble_lost` boundary, kept as a QUANTITY now that it no longer
    ends the episode."""
    ax, ay = _ball_offset(env)
    lost = _box_knob(env, "MICRODUCK_DRIBBLE_LOST", (DRIBBLE_LOST, DRIBBLE_LOST))[0]
    return 0.0 if math.hypot(ax, ay) > lost else 1.0


def _closing(env) -> float:
    """How much of a walking pace the duck is making TOWARD a lost ball, 0..1.

    Zero while the ball is still in the band: in there, closing IS dribbling
    and `_doing` already prices it — paying both would let a policy earn twice
    for one metre. Zero also when the duck is stationary or going the other
    way, which is what closes the exploit `_gaze_ball` records below: shoving
    the ball away and ADMIRING it earns nothing, because nothing here pays for
    the ball being far, only for the gap shrinking."""
    if _have(env) > 0.0:
        return 0.0
    ax, ay = _ball_offset(env)
    n = math.hypot(ax, ay)
    if n < 1e-6:
        return 0.0
    vx, vy = _duck_vel_body(env)
    return float(np.clip((vx * ax + vy * ay) / n / DRIBBLE_SEEK_SPEED, 0.0, 1.0))


def _engaged(env) -> float:
    """THE GATE, widened to the task's second half: dribbling the ball, OR
    going to get it back. `max`, not a sum — they are exclusive by
    construction (`_closing` is zero whenever `_have` is 1.0), so this is a
    selector and cannot pay for both at once.

    Only `gaze_ball` and `ball_seek` use it. Everything else still multiplies
    `_doing`, because everything else describes dribbling, and a term that
    pays while the ball is 0.8 m away is a term that pays for not having it."""
    return max(_doing(env), _closing(env))


def _ball_close(env) -> float:
    """Keeping the ball at the feet, while going somewhere. Retargeted, not
    deleted: it is still the only term that stops the ball being shoved away,
    and zeroing a load-bearing term because its weight was wrong is this
    repo's most repeated mistake."""
    return _doing(env)


def _gaze_ball(env) -> float:
    """How near the ball sits to the camera's optical axis. The head_home
    retarget: the kick wants the head home, a dribble wants it ON the ball."""
    cam, fwd, _, _ = _ball_camera(env)
    _, qadr, _ = _kick_ball_ids(env)
    d = np.array([float(env.data.qpos[qadr]) - float(cam[0]),
                  float(env.data.qpos[qadr + 1]) - float(cam[1]),
                  BALL_Z - float(cam[2])], float)
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return 1.0
    f = np.asarray(fwd, float)
    cosang = float(np.clip(d @ f / (n * float(np.linalg.norm(f))), -1.0, 1.0))
    off = math.acos(cosang)
    # TWO bells, and the split is the fix for a measured stall. One wide bell
    # at std 1.0 gives gradient from HOME (where the ball is 63 deg off axis)
    # but SATURATES long before the ball is in frame: a policy trained on it
    # settled at an optical axis 22 deg down, collecting 70 % of the term,
    # with the ball in frame on 4 % of steps and still needing ~56 deg. The
    # last 34 degrees were worth almost nothing, so it never paid for them.
    #
    # The narrow bell's width is the detector's own HALF-VFOV, so it only
    # rises as the ball actually enters the frame — which is the thing worth
    # paying for. Alone it would be flat at HOME (e^-4.5) and unlearnable,
    # which is the trap the wide one was introduced to escape. Together: the
    # wide bell gets the head moving, the narrow one makes finishing the
    # movement worth it.
    half_v = float(getattr(env, "_lm_half_v", math.radians(30.0)))
    wide = math.exp(-((off / DRIBBLE_GAZE_STD) ** 2))
    narrow = math.exp(-((off / max(half_v, 1e-3)) ** 2))
    bell = float(0.4 * wide + 0.6 * narrow)
    # GATED ON BEING ENGAGED — dribbling it, or going to get it back — and NOT
    # simply ungated, which is what the failure in 12ax F superficially argues
    # for. Ungating it outright would restore a perversion this recipe has
    # already measured: the bell alone paid **0.988 for a ball kicked away
    # against 0.655 for the head pitched down onto a ball at the feet**,
    # because a DISTANT ball is near a level axis. Paying that is paying for
    # the failure, and no amount of it would ever buy a head-down policy.
    #
    # `_engaged` is the narrowest widening that fixes the real complaint. The
    # complaint was never that the gate existed; it was that `_doing` is zero
    # for the entire recovery — the one stretch where finding the ball with
    # the camera is the whole job — so looking was unpaid exactly when it was
    # useful. Now it is paid while CLOSING, and closing on a distant ball
    # requires knowing where it is.
    #
    # The distant-ball payout is still there in the bell, and it is still
    # wrong, but it is now multiplied by a quantity that is zero unless the
    # gap is shrinking. Admiring a ball you kicked away pays nothing.
    return bell * _engaged(env)


def _turn_track(env) -> float:
    """Following the YAW the steering asks for, scaled by how much it asks.

    THE MISSING SUB-SKILL, priced. Nothing in this recipe paid for tracking
    `twist_cmd[2]`, and the measurement says that is what caps the whole task
    (bam, no knock, 20 clips): the duck OBEYS its forward command (0.189 m/s
    against 0.185 commanded) and the ball still advances **+0.002 m/s** toward
    the target, because the duck travels a median **73 deg off** the
    ball->target line and the commanded yaw sits **at its clip 58 % of the
    time**. At `DRIBBLE_TURN_MAX = 0.15` rad/s — 8.6 deg/s — turning 73 deg
    takes 8.5 s and the clip is 8. It cannot get behind the ball, so it walks
    alongside it for the whole episode.

    Raising the clip alone does not fix that, it just falls over: 0.8/0.5 buys
    20x the ball progress (0.002 -> 0.023 m/s) and **16/20 falls** against
    2/20. The turn has to be LEARNED while the clip is laddered, which is why
    this term exists and why `MICRODUCK_DRIBBLE_TURN` is now a stage knob.

    `_run_track_ang` is the locomotion catalog's own GPU
    `track_angular_velocity` — yaw rate against the command with the roll/pitch
    rates inside the same Gaussian, so thrashing the body to fake a yaw does
    not pay. Reused, not reinvented.

    SCALED BY THE DEMAND rather than gated on `_doing`, and both halves of that
    matter:

    * `_doing` would be zero exactly when the duck needs to turn — a duck
      pivoting to get behind the ball is not yet advancing along the line, so
      `_going` is low. That is the same trap `gaze_ball` was in.
    * ungated, the Gaussian pays **1.0 for a duck standing perfectly still
      with a zero command** (exp(-0/0.5) = 1), which is 2.0 a step of free
      standing income and the fourth rebuild of the optimum this recipe has
      already been beaten by three times.

    Scaling by |commanded yaw| / the stage's own clip solves both: a command
    that asks for nothing pays nothing, and a command that asks for a hard turn
    pays for delivering it. It cannot be farmed by staying mis-aimed to keep
    the demand high — that forfeits `ball_with_me` 12 + `ball_close` 6 to
    collect 2.

    AND IT IS THE FRACTION DELIVERED, not `_run_track_ang`'s Gaussian, which
    was the first version here and was FLAT where this recipe lives. That bell
    has std^2 = 0.5 while these commands are 0.15-0.50 rad/s, so a duck
    standing perfectly still scores exp(-0.15^2/0.5) = **0.956** against a
    perfect turn's 1.000 — 1.9 a step for refusing to turn, and the whole range
    of commands inside the noise of the bell. `tests/test_dribble.py` caught it
    before it trained, which is the fourth time the standing optimum has been
    rebuilt in this recipe out of a term that looked reasonable.

    So the shape is `_going`'s, the one this recipe already trusts: the signed
    fraction of the ASKED-FOR yaw actually delivered, clipped to [0, 1]. A
    still duck delivers nothing and is paid nothing; turning the wrong way is
    paid nothing rather than nearly full marks."""
    clip = _box_knob(env, "MICRODUCK_DRIBBLE_TURN",
                     (DRIBBLE_TURN_GAIN, DRIBBLE_TURN_MAX))[1]
    cmd = float(env.twist_cmd[2])
    demand = min(1.0, abs(cmd) / max(clip, 1e-6))
    if demand < 1e-3:
        return 0.0
    delivered = float(np.clip(float(env._gyro[2]) / cmd, 0.0, 1.0))
    return delivered * demand


def _ball_backward(env) -> float:
    """<= 0: the ball being driven AWAY from its target, priced at last.

    THE DEFECT THIS EXISTS FOR. `_ball_with_me` reads
    `clip(ball.u, 0.0, DRIBBLE_TARGET)`, so a ball shoved BACKWARD scores
    exactly what an untouched ball scores: zero. Half of every clip was
    invisible to the reward — the come-round branch retreating at -0.043 m/s
    against the push-through branch's +0.040 — and an optimiser cannot fix a
    failure it is never shown. 2.4 M steps of turn ladder moved the falls and
    not the ball for precisely this reason.

    ATTRIBUTION, measured rather than assumed, because pricing something the
    duck cannot control is just variance: with the nudge OFF the ball still
    retreats at -0.0324 m/s, and with it ON at -0.0367 — so the adversary
    accounts for about **12 %** and the duck for the rest. Gating this on
    foot-ball CONTACT would be perfectly attributable (backward speed there is
    -0.114 m/s, 3.5x the average) but contact is only **0.96 % of steps**, far
    too sparse to learn from. All steps, with 12 % of noise, is the honest
    trade, and it is written down so the next person need not rediscover which
    way it went.

    Units are the paid speed, so -1.0 is a full-speed retreat."""
    ux, uy, _ = _target_dir(env)
    _, _, dadr = _kick_ball_ids(env)
    v = env.data.qvel[dadr:dadr + 2]
    return min(0.0, float(v[0]) * ux + float(v[1]) * uy) / DRIBBLE_TARGET


def _gaze_hold(env) -> float:
    """Holding the head at the stage's commanded depth, WHILE WALKING, 0..1.

    Gated on `_run_speed` and not left ungated the way `gazewalk`'s own hold
    term is, because the two recipes have different failure modes. There the
    hold IS the task and a still duck collects 8.0 against ~16 for walking with
    it. Here the dribble's whole income is already only ~1.5 a step of ungated
    `stay_upright`, so an ungated 4.0 hold would hand a motionless duck 5.5 —
    within reach of the ~8-10 that dribbling pays, and this recipe has been
    beaten by the standing optimum five times. Gated on the walk it is worth
    nothing to a duck that stops.

    Returns 0 when no hold is asked for, so a stage that does not set the knob
    is exactly the recipe as it was."""
    nk, hd = _box_knob(env, "MICRODUCK_DRIBBLE_HOLD", DRIBBLE_HOLD)
    if abs(nk) < 1e-6 and abs(hd) < 1e-6:
        return 0.0
    q = env.data.qpos
    dn = float(q[env.joint_qpos_adr[5]]) - (C.DEFAULT_POSE[5] + nk)
    dh = float(q[env.joint_qpos_adr[6]]) - (C.DEFAULT_POSE[6] + hd)
    bell = math.exp(-((dn / DRIBBLE_HOLD_STD) ** 2 + (dh / DRIBBLE_HOLD_STD) ** 2))
    return float(bell) * _run_speed(env)


def _foot_near_ball(env) -> tuple[float, int, float]:
    """(proximity 0..1, foot body id, that foot's ground speed) for whichever
    foot is nearest the ball. Proximity is 1 at contact range and tapers to 0
    at `DRIBBLE_NEAR_FOOT`, so the touch terms act over an approach rather than
    an instant."""
    m = env.model
    cache = getattr(env, "_dribble_feet", None)
    if cache is None:
        cache = env._dribble_feet = [
            (g, int(m.geom_bodyid[g])) for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("foot_collision")]
    _, qadr, _ = _kick_ball_ids(env)
    bx, by = float(env.data.qpos[qadr]), float(env.data.qpos[qadr + 1])
    best, bid, bspd = 9.9, -1, 0.0
    v = np.zeros(6)
    for g, body in cache:
        fp = env.data.geom_xpos[g]
        d = math.hypot(float(fp[0]) - bx, float(fp[1]) - by)
        if d < best:
            mujoco.mj_objectVelocity(m, env.data, mujoco.mjtObj.mjOBJ_BODY, body, v, 0)
            best, bid, bspd = d, body, math.hypot(float(v[3]), float(v[4]))
    near = max(0.0, 1.0 - best / DRIBBLE_NEAR_FOOT)
    return near, bid, bspd


def _soft_approach(env) -> float:
    """Bringing the foot to the ball SLOWLY — the cause, not the outcome.

    Everything in this recipe before it priced what the BALL did afterwards:
    `ball_overshoot` docks a ball that outruns the duck, `ball_backward` docks
    one driven the wrong way. Both are outcomes of a quantity nothing was
    shaping. Ball departure speed equals the FOOT's speed at contact (0.98,
    r=+0.69), and the duck touches the ball at a median 0.412 m/s — the 90th
    percentile of its own foot speed — while 64 % of the stride is under
    0.20 m/s. It hits the ball with the fastest part of its swing because
    nothing ever asked it not to.

    Dense by construction: a foot is within `DRIBBLE_NEAR_FOOT` on 29 % of
    steps, so this shapes an APPROACH rather than the 1.4 contact events a
    clip that an at-touch reward would have to live on.

    Gated on walking, like `gaze_hold` and for the same measured reason: a duck
    standing still with a foot parked by the ball would otherwise collect the
    whole thing, and the standing optimum has taken this recipe five times."""
    near, _, spd = _foot_near_ball(env)
    if near <= 0.0:
        return 0.0
    soft = 1.0 - min(1.0, spd / DRIBBLE_SOFT_REF)
    return near * soft * _run_speed(env)


def _foot_aim(env) -> float:
    """Pointing the foot along the ball's line to the target while near it.

    `hip_yaw` is the DOF that swings a foot's heading, range about +-0.5 rad,
    and the policy uses a median of **0.081 rad** of it at contact with a
    correlation to where the ball actually goes of **+0.16**. It is not aiming;
    the ball leaves a median 35 deg off the target line. This prices the thing
    the joint is for.

    Deliberately a POSE quantity and not a velocity one, so it composes with
    `_soft_approach` instead of fighting it — a foot can be aimed and slow at
    the same time, and a foot's velocity direction is meaningless once it is."""
    near, body, _ = _foot_near_ball(env)
    if near <= 0.0 or body < 0:
        return 0.0
    # ON THE JOINT, not on a body axis. The first version read the foot body's
    # local X as "forward" and it is nothing of the kind: on a WALKING duck
    # that axis has a mean cosine of **-0.001** with the heading, because it
    # rotates through the swing. The term was noise, and it measured as noise —
    # aim got WORSE (30 deg -> 35 deg) and hip_yaw usage went DOWN. (A first
    # check on a duck driven by zero actions read -1.000 and looked like a
    # clean sign error; that duck had collapsed, which is the trap this repo
    # has a scar for.)
    #
    # `hip_yaw` is the DOF that actually swings a leg's heading, +-0.5 rad, and
    # the policy uses a median 0.081 of it at contact with a +0.16 correlation
    # to where the ball goes. Desired yaw is the bearing of the ball->target
    # line in the duck's own frame, clamped to what the joint can reach.
    side = 0 if body == _dribble_leg_bodies(env)[0] else 9
    ux, uy, _d = _target_dir(env)
    yaw = _trunk_yaw(env)
    c, sn = math.cos(yaw), math.sin(yaw)
    psi = math.atan2(-ux * sn + uy * c, ux * c + uy * sn)
    lo, hi = _dribble_hip_range(env, side)
    want = max(lo, min(hi, psi))
    have = float(env.data.qpos[env.joint_qpos_adr[side]]) - C.DEFAULT_POSE[side]
    bell = math.exp(-(((have - want) / 0.30) ** 2))
    return near * float(bell) * _run_speed(env)


def _dribble_hip_range(env, side: int) -> tuple[float, float]:
    """That hip_yaw joint's own limits, as offsets off `DEFAULT_POSE`.

    Read off the model rather than written down: the two hips are not
    symmetric (left is -0.436..+0.524) and a hardcoded pair would silently
    ask for a yaw one leg cannot reach."""
    key = f"_dribble_hiprange{side}"
    got = getattr(env, key, None)
    if got is None:
        rng = env.model.joint(C.JOINT_NAMES[side]).range
        got = (float(rng[0]) - C.DEFAULT_POSE[side], float(rng[1]) - C.DEFAULT_POSE[side])
        setattr(env, key, got)
    return got


def _dribble_leg_bodies(env) -> tuple[int, int]:
    """(left foot body, right foot body), cached."""
    got = getattr(env, "_dribble_legbodies", None)
    if got is None:
        m = env.model
        left = right = -1
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if nm.endswith("foot_collision"):
                if "left" in nm:
                    left = int(m.geom_bodyid[g])
                else:
                    right = int(m.geom_bodyid[g])
        got = env._dribble_legbodies = (left, right)
    return got


def _in_stance(env) -> float:
    """BEING BEHIND THE BALL, on its line to the target — the missing objective.

    Every term in this recipe priced the BALL (where it is, how fast, which
    way) or the GAIT (upright, flat, on pace). None priced where the DUCK
    stands, and the steering's own dribble spot — `ball - u * DRIBBLE_BEHIND` —
    was computed every step and never paid for. Measured: the duck is on the
    good side of the ball on **48 % of steps**, which is chance, and it
    delivers 56 % of the yaw it is commanded. It is not trying to get into
    position because position was worth nothing.

    Two bells: how far the duck is from the spot ALONG the ball->target line,
    and how far it is OFF that line. A duck level with the ball but a foot to
    the side is not in a dribbling stance, and one bell on straight-line
    distance could not tell the two apart.

    Gated on walking like the rest: a duck parked in a perfect stance is not
    dribbling, and the standing optimum has taken this recipe five times."""
    t = getattr(env, "_dribble_target", None)
    if t is None:
        return 0.0
    ux, uy, _d = _target_dir(env)
    _, qadr, _ = _kick_ball_ids(env)
    bx, by = float(env.data.qpos[qadr]), float(env.data.qpos[qadr + 1])
    behind = _box_knob(env, "MICRODUCK_DRIBBLE_BEHIND", (DRIBBLE_BEHIND, DRIBBLE_BEHIND))[0]
    # the spot: `behind` metres back down the line from the ball
    sx, sy = bx - ux * behind, by - uy * behind
    dx, dy = float(env.data.qpos[0]) - sx, float(env.data.qpos[1]) - sy
    along = dx * ux + dy * uy          # + means PAST the spot, toward the target
    lat = -dx * uy + dy * ux           # signed distance off the line
    bell = math.exp(-((along / DRIBBLE_STANCE_STD) ** 2
                      + (lat / DRIBBLE_STANCE_STD) ** 2))
    return float(bell) * _run_speed(env)


def _no_flight(env) -> float:
    """-1 each step neither foot touches the floor, else 0 (a penalty, <= 0)."""
    return 0.0 if (env.foot_contact_state["left"] or env.foot_contact_state["right"]) else -1.0


def _dribble_lost(env) -> bool:
    """Ball out of the dribble band. NO LONGER the terminal — `_dribble_gone`
    is. Kept because it is what `_have` means and what the caption reports."""
    ax, ay = _ball_offset(env)
    return math.hypot(ax, ay) > _box_knob(env, "MICRODUCK_DRIBBLE_LOST",
                                          (DRIBBLE_LOST, DRIBBLE_LOST))[0]


def _dribble_gone(env) -> bool:
    """THE TERMINAL: the ball is far enough that the clip is over.

    Set well outside `_dribble_lost` on purpose. Everything between the two is
    the corridor in which the duck has lost the ball and still has the episode
    — which is the only place a sighting is worth anything, and the reason
    this recipe's perception slots were dead code before (12ax F)."""
    ax, ay = _ball_offset(env)
    return math.hypot(ax, ay) > _box_knob(env, "MICRODUCK_DRIBBLE_GONE",
                                          (DRIBBLE_GONE, DRIBBLE_GONE))[0]


def _dribble_ball_geom(env) -> int:
    """Geom id of the ball, cached on the env (the BODY id is not enough: the
    rolling coefficient lives on the geom's friction triple)."""
    g = getattr(env, "_dribble_ball_geom", None)
    if g is None:
        g = env._dribble_ball_geom = int(env.model.geom("ball_geom").id)
    return g


def _dribble_reset(env) -> None:
    r = env._rng
    _, qadr, dadr = _kick_ball_ids(env)
    yaw = _trunk_yaw(env)
    ox = r.uniform(*_box_knob(env, "MICRODUCK_DRIBBLE_AHEAD", DRIBBLE_AHEAD))
    oy = r.uniform(*_box_knob(env, "MICRODUCK_DRIBBLE_SIDE", DRIBBLE_SIDE))
    env.data.qpos[qadr:qadr + 7] = [
        float(env.data.qpos[0]) + math.cos(yaw) * ox - math.sin(yaw) * oy,
        float(env.data.qpos[1]) + math.sin(yaw) * ox + math.cos(yaw) * oy,
        BALL_Z, 1.0, 0.0, 0.0, 0.0]
    env.data.qvel[dadr:dadr + 6] = 0.0
    # THE BALL. Set on the model, so it survives for the whole episode; the
    # write is idempotent and costs one float a reset. See
    # `DRIBBLE_BALL_ROLLING` for the measurement that put a ladder here.
    roll = _box_knob(env, "MICRODUCK_DRIBBLE_BALL_ROLLING",
                     (DRIBBLE_BALL_ROLLING, DRIBBLE_BALL_ROLLING))[0]
    env.model.geom_friction[_dribble_ball_geom(env)][2] = roll
    # THE COMMAND. Set here, after the base reset's `_sample_commands` has run,
    # so this recipe steers the twist slots without touching the sampler every
    # other behavior shares.
    spd = r.uniform(*_box_knob(env, "MICRODUCK_DRIBBLE_SPEED", DRIBBLE_SPEED))
    bmax = _box_knob(env, "MICRODUCK_DRIBBLE_BEARING", (DRIBBLE_BEARING, DRIBBLE_BEARING))[0]
    b = float(r.uniform(-bmax, bmax))
    env._dribble_speed = spd
    env._dribble_reached = 0
    env._dribble_tprev = None
    env._dribble_nudges = 0
    env._dribble_target = None
    env._dribble_cmd = (spd * math.cos(b), spd * math.sin(b))
    env.twist_cmd[0], env.twist_cmd[1] = env._dribble_cmd
    env.twist_cmd[2] = 0.0
    # THE GAZE. An offset off HOME, drawn per episode, so the policy is handed
    # a sighting instead of having to discover that looking down exists.
    nk = _box_knob(env, "MICRODUCK_DRIBBLE_GAZE_NECK", DRIBBLE_GAZE_NECK)
    hd = _box_knob(env, "MICRODUCK_DRIBBLE_GAZE_HEAD", DRIBBLE_GAZE_HEAD)
    env.data.qpos[env.joint_qpos_adr[5]] = C.DEFAULT_POSE[5] + r.uniform(*nk)
    env.data.qpos[env.joint_qpos_adr[6]] = C.DEFAULT_POSE[6] + r.uniform(*hd)
    # ...and give it the velocity to match, so the clip starts mid-walk.
    fv = r.uniform(*_box_knob(env, "MICRODUCK_DRIBBLE_SPAWN_VEL", DRIBBLE_SPAWN_VEL))
    if fv > 0.0:
        cy, sy = math.cos(yaw), math.sin(yaw)
        vx, vy = env._dribble_cmd
        env.data.qvel[env._root_qvel + 0] = fv * (vx * cy - vy * sy)
        env.data.qvel[env._root_qvel + 1] = fv * (vx * sy + vy * cy)
    env.data.ctrl[:] = env.data.qpos[env.joint_qpos_adr]
    if getattr(env, "bam", None) is not None:
        env.bam.reset(env.data.qpos[env.joint_qpos_adr])
    mujoco.mj_forward(env.model, env.data)
    env.last_spawn = (f"ball {ox:.2f} m ahead {abs(oy):.2f} m "
                      f"{'left' if oy >= 0 else 'right'}, command {spd:.2f} m/s at {math.degrees(b):+.0f}°")
    # Sensing state, fresh per episode — the same four head slots `lastmetre`
    # writes, through the same detector, so a dribble and a sensed kick read
    # obs[51:55] the same way and can be chained.
    env._lm_episode = env.episode_id
    env._lm_step_done = -1
    env._lm_det_step = -10 ** 9
    env._lm_det_seen = False
    env._lm_world = None
    env._lm_conf = 0.0
    env._lm_seen_steps = 0
    env._lm_half_h = math.radians(_ball_knob(env, "MICRODUCK_BALL_HFOV_DEG")) / 2
    env._lm_half_v = math.radians(_ball_knob(env, "MICRODUCK_BALL_VFOV_DEG")) / 2
    env._lm_range_scale = DRIBBLE_RANGE_SCALE
    # The first target, drawn AFTER the ball is placed (it is measured from
    # the ball) and after `_lm_episode` is set (the command hook checks it).
    _new_target(env, r)
    _dribble_command(env)
    _lm_sense(env, force=True)


def _dribble_obs(env) -> None:
    if getattr(env, "_lm_episode", None) != env.episode_id:
        return
    if env._lm_step_done == env.step_count:
        return
    env._lm_step_done = env.step_count
    _dribble_command(env)
    _lm_sense(env)


def _dribble_command(env) -> None:
    """Steer toward the target and republish it, every step.

    Three jobs, all of which must happen per-step rather than at reset:

    1. **The twist command points at the target.** The env plays the BRAIN
       here, which is the deployment shape (`chase` steers, the skill handles
       the ball) and is also why `keep_pace` is meaningful: the policy is
       being asked for a velocity and scored on delivering it.
    2. **obs[55:58] carries the BALL-to-target vector and the distance left.**
       It has to be rewritten every step because the body-frame direction to a
       fixed point rotates as the duck walks — a value written at reset is
       wrong by the second step. It is the ball's vector, not the duck's: the
       duck and the ball want different things the moment the ball is off the
       line, and that difference is the whole skill.
    3. **A reached target is replaced, not an ending.** See the note by
       `DRIBBLE_TARGET_DIST`.

    Also the guard against `walk_env`'s mid-episode `_sample_commands`, which
    would otherwise hand a dribble a fresh random direction mid-clip."""
    if getattr(env, "_lm_episode", None) != env.episode_id:
        return
    if getattr(env, "_dribble_target", None) is None:
        return
    ux, uy, dist = _target_dir(env)
    # ...reached? draw the next one and count it.
    reach = _box_knob(env, "MICRODUCK_DRIBBLE_TARGET_REACH",
                      (DRIBBLE_TARGET_REACH, DRIBBLE_TARGET_REACH))[0]
    if dist <= reach:
        env._dribble_reached = getattr(env, "_dribble_reached", 0) + 1
        _new_target(env)
        ux, uy, dist = _target_dir(env)
    # THE NUDGE. Before the steering reads the ball, so the spot and the slots
    # this step already reflect the new position — a knock the policy is told
    # about a step late is a latency test, not a sight test.
    rate = _box_knob(env, "MICRODUCK_DRIBBLE_NUDGE_RATE",
                     (DRIBBLE_NUDGE_RATE, DRIBBLE_NUDGE_RATE))[0]
    if rate > 0.0 and float(env._rng.uniform()) < rate * C.CTRL_DT:
        _, _, dadr = _kick_ball_ids(env)
        tux0, tuy0, _ = _target_dir(env)
        ang = math.atan2(tuy0, tux0) + float(env._rng.choice((-1.0, 1.0))) * float(
            env._rng.uniform(*_box_knob(env, "MICRODUCK_DRIBBLE_NUDGE_SIDE", DRIBBLE_NUDGE_SIDE)))
        sp = float(env._rng.uniform(*_box_knob(env, "MICRODUCK_DRIBBLE_NUDGE_SPEED", DRIBBLE_NUDGE_SPEED)))
        env.data.qvel[dadr] += sp * math.cos(ang)
        env.data.qvel[dadr + 1] += sp * math.sin(ang)
        env._dribble_nudges = getattr(env, "_dribble_nudges", 0) + 1
    yaw = _trunk_yaw(env)
    c, sn = math.cos(yaw), math.sin(yaw)
    # (1) the twist: WALK FORWARD AND TURN, never strafe.
    #
    # The first version commanded the body-frame vector to the target
    # directly, which for an off-axis target is a large sustained `vy` — and
    # the walker folded under it, trunk 0.118 -> 0.064 m in EIGHT steps,
    # tripping the env's own height terminal. That is not a policy failure: no
    # brain in this repo drives the walker that way. `Chase` emits forward
    # speed and a YAW RATE and turns the body to face where it is going, and
    # the walker's own training mix is mostly-forward for the same reason.
    # So this steers the way the deployment does.
    # STEER AT THE DRIBBLE SPOT, NOT AT THE TARGET.
    #
    # Steering straight at the destination was the single biggest defect in
    # this recipe and it is worth stating plainly: a dribbler has to be BEHIND
    # the ball. Walking at the target means that the moment the ball is off
    # the line, the duck walks AWAY from it. Measured over 30 clips of the
    # policy trained that way: 23 ended in a lost ball, and at the loss the
    # ball was a median 0.39 m BEHIND the duck (96 % of losses) and almost
    # stationary at 0.03 m/s while the duck was still walking at 0.20. It was
    # not losing the ball to a heavy touch — it was leaving it.
    #
    # The spot is `DRIBBLE_BEHIND` back along the ball->target line, which is
    # the same construction the kick already uses (`Chase._kick_spot`): stand
    # behind the thing, facing where it should go, and moving forward does the
    # work. When the duck is AT the spot the ball is between it and the
    # target, so the next step pushes the ball down the line.
    # THE BALL AS THE BRAIN BELIEVES IT, never `qpos`. See `_ball_believed` for
    # the ablation that forced this. No sighting yet this episode means the
    # brain genuinely does not know where the ball is: it holds the episode's
    # drawn heading, publishes zeros in the ball-derived slots — the same
    # "I don't know" the camera slots already use when `_lm_world` is None —
    # and does NOT quietly fall back on the truth.
    believed = _ball_believed(env)
    if believed is None:
        # NO COMMAND AT ALL, and specifically NOT "walk forward at the drawn
        # speed" — which was this block's first version and re-created the very
        # leak the change exists to close. 12ax G's finding was that the ball is
        # never far and the command points forward, so WALKING FINDS IT; a
        # forward fallback hands that back every time the duck stops looking.
        #
        # A zero twist is the honest thing for a brain with no fix on the ball,
        # and it prices itself without a new term: `_going` divides by the
        # commanded speed and returns 0 when there is none, so `_doing` is 0,
        # so every dribble term is 0. Not knowing where the ball is earns
        # `stay_upright` and nothing else. It is also IN DISTRIBUTION for the
        # donor — a zero command is what the walker was trained to stand on.
        env.twist_cmd[0] = env.twist_cmd[1] = env.twist_cmd[2] = 0.0
        env.body_cmd[0] = env.body_cmd[1] = env.body_cmd[2] = 0.0
        return
    bx, by = believed
    # ...and the ball->target line is measured from the BELIEVED ball too, so
    # the spot and the published slots are one consistent picture. The reward's
    # `_target_dir` keeps reading truth; these two must not.
    tux, tuy, _tdist = _target_dir_from(env, bx, by)
    ux, uy, dist = tux, tuy, _tdist
    behind = _box_knob(env, "MICRODUCK_DRIBBLE_BEHIND", (DRIBBLE_BEHIND, DRIBBLE_BEHIND))[0]
    # ONLY GO AROUND WHEN YOU ARE ON THE WRONG SIDE.
    #
    # Steering at the spot unconditionally was a bug with a measurable
    # signature: at spawn the duck is ALREADY behind the ball, so a spot
    # 0.14 m further back lands at or behind its own feet on 51 % of resets,
    # `|wz|` pins at its clip, and the duck spins on the first step. A
    # fine-tune from a donor that runs 400/400 collapsed to ep_len 16.6
    # within 900k steps on this.
    #
    # `rel` is how far the duck is PAST the ball along the ball->target line.
    # Negative means it is behind the ball with the ball between it and the
    # target — the good side — and then the thing to do is walk at the target
    # THROUGH the ball, which is a dribble. Only when it has overrun the ball
    # (rel >= 0) does it need to loop back to the spot.
    rel = ((float(env.data.qpos[0]) - bx) * tux + (float(env.data.qpos[1]) - by) * tuy)
    if rel < 0.0:
        spot = (env._dribble_target[0], env._dribble_target[1])                       # already behind it: push through
    else:
        # COME ROUND THE SIDE, NOT THROUGH THE BALL — and this offset is the
        # fix for a measured cancellation, not a refinement. Steering at the
        # bare point 0.14 m behind the ball sends an overrun duck on a path
        # straight across the ball's own position, so it walks INTO the ball on
        # the way to standing behind it. Measured on the turn ladder's top rung
        # (bam, no knock, 20 clips), split by these two branches:
        #
        #     PUSH THROUGH  47 % of steps   ball->target  +0.040 m/s
        #     COME ROUND    53 % of steps   ball->target  -0.043 m/s
        #
        # 0.47 x 0.040 + 0.53 x -0.043 = -0.004, which is the net ball progress
        # of the whole recipe to three decimals. The dribble WORKED and the
        # come-round undid it at the same rate. That is also why 2.4 M steps of
        # turn ladder moved the falls (16/20 -> 2/20) and not the ball: more
        # turning authority buys a faster loop, and the loop was the problem.
        #
        # The offset goes to WHICHEVER SIDE THE DUCK IS ALREADY ON, so the
        # shorter arc is the one commanded and the path curves around the ball
        # instead of over it. Picking a fixed side would send half the
        # approaches the long way round, through the ball again.
        pxx, pxy = -tuy, tux                     # left-hand normal to the ball->target line
        beside = ((float(env.data.qpos[0]) - bx) * pxx + (float(env.data.qpos[1]) - by) * pxy)
        side = 1.0 if beside >= 0.0 else -1.0
        r = _box_knob(env, "MICRODUCK_DRIBBLE_ROUND",
                      (DRIBBLE_ROUND_SIDE, DRIBBLE_ROUND_SIDE))[0]
        spot = (bx - tux * behind + pxx * side * r,
                by - tuy * behind + pxy * side * r)
    env._dribble_spot = spot
    dx, dy = spot[0] - float(env.data.qpos[0]), spot[1] - float(env.data.qpos[1])
    bearing = math.atan2(-dx * sn + dy * c, dx * c + dy * sn)
    spd = getattr(env, "_dribble_speed", 0.18)
    env.twist_cmd[0] = spd
    env.twist_cmd[1] = 0.0
    # Gain and clip are knobs because the first guess (1.5, +-0.8) commanded a
    # median |wz| of 0.42 even for a target DEAD AHEAD — a proportional law
    # with no damping oscillating about the line — and the shipped walker,
    # which survives this env 400/400 with no turn command at all, died at 28
    # steps under it. A steering law the incumbent cannot follow measures the
    # law, not the policy.
    kz, wzmax = _box_knob(env, "MICRODUCK_DRIBBLE_TURN", (DRIBBLE_TURN_GAIN, DRIBBLE_TURN_MAX))
    env.twist_cmd[2] = float(np.clip(kz * bearing, -wzmax, wzmax))
    # (2) the slots: from the BALL to the target, body frame, plus range left
    env.body_cmd[0] = ux * c + uy * sn
    env.body_cmd[1] = -ux * sn + uy * c
    env.body_cmd[2] = float(np.clip(dist / DRIBBLE_TARGET_SCALE, 0.0, 1.0))


def _dribble_caption(env) -> str:
    ax, ay = _ball_offset(env)
    _, _, dist = _target_dir(env)
    return (f"ball {math.hypot(ax, ay):.2f} m{'' if _have(env) else ' RECOVERING'}  "
            f"target {dist:.2f} m  "
            f"reached {getattr(env, '_dribble_reached', 0)}  seen {env._lm_det_seen}")


def dribble_target_payload(env):
    """The TARGET the ball is being taken to — [x, y, z, reach, reached] — so
    the lab can draw the long goal beside the ball and its ghost.

    `reach` is the radius that counts as arrived (the ring to draw) and
    `reached` the count so far this episode, which is the task's real score.
    None for any env without a target, which is every other recipe — and
    read through getattr for the same reason `ball_ghost_payload` is: the
    lab's roster can hold a plain walk env with no episodes at all."""
    t = getattr(env, "_dribble_target", None)
    if t is None or getattr(env, "_lm_episode", None) != getattr(env, "episode_id", None):
        return None
    return [round(float(t[0]), 4), round(float(t[1]), 4), 0.01,
            DRIBBLE_TARGET_REACH, int(getattr(env, "_dribble_reached", 0))]


_register(Behavior(
    id="dribble",
    emoji="🏃",
    title="Dribble the ball where it is told",
    description=(
        "Walk the ball along a commanded direction while keeping it at its feet, "
        "with the ball's position in its observation — the aim that the brain's "
        "push mode measured better than kicking but could not point."
    ),
    how_it_learns=(
        "The ball starts at its feet and a direction is commanded in the duck's own "
        "frame — the same twist slots the walker already reads, because the robot "
        "cannot observe its heading in the world. It is paid every step the ball "
        "moves along that direction up to 0.35 m/s and docked past it, for keeping "
        "the ball about 13 cm off its trunk, for keeping the ball near what its "
        "camera is pointed at, for stepping rather than skidding, and it is docked "
        "for cranking joints to their stops. The clip ends the moment the ball gets "
        "further than 45 cm: without that, the cheapest policy is to shove it away "
        "on the first step and collect the rest of the pay for standing still."
    ),
    keywords=("dribble", "dribbling", "walk the ball", "carry the ball",
              "ball control", "push the ball", "take the ball"),
    terms=(
        RewardTerm("ball_with_me", "Big points for the ball and the duck BOTH advancing along the command",
                   8.0, _ball_with_me),
        # THE TASK, as displacement. Sized to dominate: the audit that found
        # the task earning 8.3 % against standing's 36.1 % is why this is 30
        # and not 12, and the shape is why the weight alone was not enough.
        RewardTerm("ball_progress", "Big points for GROUND the ball makes up toward its target",
                   30.0, _ball_progress),
        # The term the first draft was missing entirely, and the reason it
        # would have stood still: nothing paid the duck to travel.
        # THE GAIT PACKAGE, and it is here because warm-starting from the
        # walker was NOT enough on its own. MEASURED: a run warm-started from
        # a distilled `alpha_walking` had, by 647 k steps, gone from the
        # donor's 0 % airborne / 0.119 m trunk to **14 % airborne / 0.106 m**
        # — further from a walk than the from-scratch policy it was meant to
        # replace — and was ending 20 of 20 episodes early. A prior is a
        # STARTING POINT, not a constraint: if nothing prices the gait, PPO
        # walks away from it, and the first 6 M-step attempt that invented a
        # crouched hop was not an accident of initialisation.
        #
        # So the three terms that make the locomotion recipe a WALK come in,
        # at the weights that recipe uses, and `keep_pace` doubles: on the
        # hopping policy it scored 0.042, which is a term that was being
        # ignored because it was cheap to ignore.
        # 8.0 -> 3.0 AND gated on still having the ball. At 8.0 ungated it paid
        # +5.94 a step against `ball_with_me`'s +1.00 on the finished policy —
        # a 6:1 ratio in favour of walking well, and the policy took it: 0 %
        # airborne, a clean gait, 0.85 m travelled, and the ball left behind
        # at 0.41 m (ratio 0.48, which is the UNTRAINED walker's 0.55).
        #
        # This is the third time in this recipe that the term added to fix a
        # problem became the thing optimised instead of the task — ball_close
        # bought standing still, the gait package bought standing still again,
        # and keep_pace bought walking away. The pattern is the same each
        # time: a term that can be satisfied WITHOUT doing the task will be.
        # So every term here is now conditioned on the task actually being
        # performed, and that is the invariant to preserve if anything is
        # added later.
        RewardTerm("keep_pace", "Points for matching the commanded speed while still holding the ball",
                   3.0, lambda env: _run_speed(env) * _doing(env)),
        # GATED ON GOING, like the ball terms — and this is the SECOND time the
        # standing optimum has been rebuilt here out of whatever was left
        # ungated. The first version gave a still duck `ball_close` 6.0 flat;
        # gating that fixed it, and then this gait package, added to stop a
        # HOP, handed a still duck stay_upright 1.95 + pose 0.83 + flat_feet
        # 2.49 = 5.28 a step for standing perfectly still — measured off
        # the live checkpoint at 1.4M, which reached 0 targets in 16 clips and
        # travelled 0.02 m. A gait you hold while stationary is not a gait:
        # these terms describe how the duck WALKS, so they earn while it walks.
        #
        # The general rule, paid for twice: after gating a term, re-audit the
        # SUM of everything still ungated. Standing income does not have to
        # come from the term you were looking at.
        # UNGATED, deliberately, and it is the one exception to the gate above.
        #
        # Gating it was over-correction and it MEASURED as one: the finished
        # run fell 7/20 where the previous one fell 0/20, and episodes
        # collapsed from 346 steps to 52. The mechanism is that the gate makes
        # a clip worth ~0 the moment the ball is lost, and a `terminate_on_fall`
        # that forfeits ~0 is not a deterrent — falling became free.
        #
        # Staying upright is not part of the task, it is a precondition for
        # attempting it, so it is priced whether or not the task is going
        # well. At 2.0 a perfectly still duck collects 2.0 a step against ~20
        # for real dribbling — a 10:1 gradient, where the 5.28 of standing
        # income that trapped this recipe twice was only ~4:1.
        RewardTerm("stay_upright", "Points for keeping the body upright (lean is cheap)",
                   1.5, _run_upright),
        RewardTerm("pose", "Points for a speed-appropriate leg pose while going",
                   1.0, lambda env: _run_pose(env) * _doing(env)),
        RewardTerm("ball_close", "Points for keeping the ball about 13 cm in front of the trunk",
                   3.0, _ball_close),
        RewardTerm("ball_overshoot", "Docked for knocking the ball past 0.35 m/s — that is a kick, not a dribble",
                   4.0, _ball_overshoot, is_penalty=True),
        # RETARGETED DOWN, 2.0 -> 0.5, and not deleted. The sighting it exists
        # for is already guaranteed by the spawn gaze (measured 100 % in frame)
        # and carried in obs[51:55]; at 2.0 it made LOOKING competitive with
        # WALKING, which is the "just staring at the ball" this recipe was
        # rebuilt to stop. It stays because something must still price the
        # head, and zeroing a term because its weight was wrong is this repo's
        # most repeated mistake.
        # 0.5 -> 3.0. At 0.5 it was never worth looking down: the finished
        # policy dribbled with `seen` at 0 % and the ghost could not be drawn
        # at all. It stays WELL under `ball_with_me`'s 12 and is gated on
        # `_doing` like everything else, so it cannot be farmed by staring at
        # a ball while standing — which is the failure this recipe has already
        # been beaten by three times in other clothes.
        RewardTerm("gaze_ball", "Points for keeping the ball IN FRAME, not merely near the axis",
                   4.0, _gaze_ball),
        # THE RECOVERY, priced. Without it the corridor above is dead time: a
        # lost ball puts `_near` at ~0 (the bell is e^-10 by 0.45 m), so every
        # other positive term is already zero out there, and a duck that
        # wandered would collect `stay_upright` alone for the rest of the clip.
        # Not ending the episode is what makes recovery POSSIBLE; this is what
        # makes it WORTH DOING.
        #
        # Sized deliberately under the dribble. Seeking pays at most
        # ball_seek 3.0 + gaze_ball 4.0 = 7.0 a step; dribbling pays
        # ball_with_me 12 + ball_close 6 + keep_pace 3 + gait ~3 = ~24 at full
        # value. That ~3.4:1 is the same shape as the 10:1 that prices
        # standing against dribbling, and it is what stops the obvious
        # exploit: shove the ball out, walk after it, repeat. You cannot farm
        # 7 a step by giving up 24.
        RewardTerm("ball_seek", "Points for closing on a ball that has got away",
                   3.0, _closing),
        # Same 2.0 the `run` recipe gives this term. Sized against the thing it
        # must not outbid: a duck that stays mis-aimed to keep the demand high
        # collects 2 and forfeits ball_with_me's 12 plus ball_close's 6.
        RewardTerm("turn_track", "Points for actually making the turn the steering asks for",
                   2.0, _turn_track),
        # THE GAZE LADDER'S PAY. 4.0 against `ball_progress`'s 30 and
        # `stay_upright`'s 1.5: enough to be worth buying now that the gait can
        # afford the pose, and nowhere near enough to outbid moving the ball.
        # 12ax F priced looking at +5.4 against -238.6 under the OLD gait; what
        # changed is not this weight, it is that a head-down duck can now walk.
        RewardTerm("gaze_hold", "Points for holding the head where it can see its own feet",
                   4.0, _gaze_hold),
        # THE TOUCH, shaped where it is caused. Sized against `ball_progress`'s
        # 30: together these two average ~2 a step (they act on 29 % of steps),
        # comparable to `stay_upright`'s 1.5 and nowhere near the task. They
        # are MEANS — a soft aimed foot that never moves the ball earns them
        # and nothing else.
        # THE STANCE. The missing objective: 8.0, below `ball_progress`'s 30
        # because getting into position is a MEANS, and above the gait terms
        # because it is the means the whole task was failing on (behind the
        # ball 48 % of steps, which is chance).
        RewardTerm("in_stance", "Points for standing behind the ball, on its line to the target",
                   8.0, _in_stance),
        RewardTerm("soft_approach", "Points for bringing the foot to the ball slowly, not at full swing",
                   6.0, _soft_approach),
        RewardTerm("foot_aim", "Points for pointing the foot down the ball's line to the target",
                   4.0, _foot_aim),
        # 3.0, from the arithmetic it has to beat rather than a guess. The
        # push-through branch earns `ball_with_me` ~0.125 a step on 47 % of
        # steps; at 3.0 the retreat costs ~0.37 a step on 53 %, about 3:1
        # against looping. Below ~0.9 the two merely cancel again, which is the
        # state this term was added to end.
        RewardTerm("ball_backward", "Docked for driving the ball AWAY from its target",
                   3.0, _ball_backward, is_penalty=True),
        # NOT `step_dont_skid`, which was the first choice and is DECORATION
        # here: it gates on |wz| >= 0.5 and this recipe commands wz = 0, so it
        # measured exactly 0.000 at the spawn, while walking and at the target
        # (AGENTS.md: a term that pays ~0 at both is decoration). `flat_feet`
        # is what that slot was really buying — feet under the body rather
        # than hopping or toeing at the ball.
        # 0.8 -> 2.5. NOT because it prices the flight phase — it does NOT,
        # and an earlier comment here claimed it did. Measured: the hopping
        # policy (10 % airborne) scores `flat_feet` 0.91 against the walker's
        # 0.93, because the term reads how FLAT a foot is while it is in
        # contact, not how much of the time there is a foot in contact at all.
        # It is raised because sole-flatness is worth more to a walk than to a
        # trick; the flight phase needed its own term, below.
        RewardTerm("flat_feet", "Points for keeping the feet flat on the floor while going",
                   2.5, lambda env: CATALOG["flat_feet"].fn(env) * _doing(env)),
        # THE FLIGHT PHASE, priced directly, because nothing else here reads
        # it. The shipped walker is airborne 0 % of the time; both trained
        # attempts invented a hop at 10-14 %, which is what "it's just
        # shuffling around" looks like as a number. A dock and not a bonus
        # (the `run` recipe pays for air time; a walk is the opposite task),
        # <= 0 by construction, and free for a policy that keeps a foot down.
        #
        # Sized honestly: at 10 % airborne this costs 0.3 a step, which will
        # NOT on its own outweigh the 8.65 a step the hopping policy earns
        # over a walker that barely dribbles. That gap is between two
        # different policies, not two behaviours of one, so no weight here can
        # settle it — the bet is that a walker PRIOR plus a standing dock on
        # flight lands on walk-and-dribble, and that has to be measured, not
        # argued.
        RewardTerm("no_flight", "Docked every step with neither foot on the floor",
                   3.0, _no_flight, is_penalty=True),
        CATALOG["no_limit_parking"],
    ),
    # SYMMETRIC: the command is drawn either side of the nose, so a left-going
    # dribble is the mirror of a right-going one and the mirror prior is free
    # data. (The kicks opt out because each one names a FOOT; nothing here
    # does.) `tests/test_behaviors.py` locks the opt-out list, which is how
    # this was caught — the first draft copied `lastmetre`'s False.
    symmetric=True,
    # This recipe legitimately carries a command in obs[48:50] — the direction
    # it is being asked to take the ball — so it declares the ceiling the way
    # a locomotion recipe does, and `tests/test_behaviors.py` checks the slots
    # against it instead of requiring a trick's zeros. The value is the
    # command's own maximum speed; `_dribble_command` then holds the episode's
    # draw against the mid-episode resample.
    forward_cmd=DRIBBLE_SPEED[1],
    episode_s=DRIBBLE_EPISODE_S,
    scene="ball",
    terminate_on_fall=True,
    terminate_fn=_dribble_gone,
    reset_fn=_dribble_reset,
    obs_fn=_dribble_obs,
    caption_fn=_dribble_caption,
    default_steps=6_000_000,
    curriculum=(
        CurriculumStage("straight ahead, slowly", 2_000_000,
                        {"MICRODUCK_DRIBBLE_BEARING": "0.0,0.0",
                         "MICRODUCK_DRIBBLE_SPEED": "0.12,0.16",
                         "MICRODUCK_DRIBBLE_SIDE": "-0.04,0.04",
                         "MICRODUCK_DRIBBLE_LOST": "0.60,0.60",
                         # No corridor and no nudge: GONE == LOST is exactly
                         # the pre-recovery recipe, so `_closing` is 0 by
                         # construction and `_engaged` collapses to `_doing`.
                         # Learn the dribble before learning to rescue it.
                         "MICRODUCK_DRIBBLE_GONE": "0.60,0.60",
                         "MICRODUCK_DRIBBLE_NUDGE_RATE": "0.0,0.0",
                         # THE TURN LADDER, rung 1: the clip the warm-start
                         # donor already survives (2/20 falls). Measured
                         # ceiling of this rung is the problem, not its safety
                         # — ball progress +0.002 m/s, because 8.6 deg/s
                         # cannot turn the median 73 deg inside an 8 s clip.
                         "MICRODUCK_DRIBBLE_TURN": "0.4,0.15",
                         # THE BALL, rung 1: 0.05 — a ball that stops 4.4 cm after a
                         # touch, so the 0.30-0.55 m target needs 7-12 of
                         # them and CANNOT be won with a kick. The old
                         # policy already scores 12 touches a clip here
                         # untrained, so this rung is about the state
                         # EXISTING in the rollouts, not about difficulty.
                         "MICRODUCK_DRIBBLE_BALL_ROLLING": f"{DRIBBLE_ROLL_LADDER[0]},{DRIBBLE_ROLL_LADDER[0]}",
                         "MICRODUCK_DRIBBLE_GAZE_NECK": DRIBBLE_GAZE_DRILL[0],
                         "MICRODUCK_DRIBBLE_GAZE_HEAD": DRIBBLE_GAZE_DRILL[1]},
                        detail=("The ball dead ahead, the command dead ahead, the slowest "
                                "speed and a forgiving loss radius. Nothing but touching "
                                "the ball forward pays, and the clip is long enough to "
                                "touch it more than once — which is the whole difference "
                                "from a strike. Nothing knocks the ball off-line yet.")),
        CurriculumStage("off the nose", 2_000_000,
                        {"MICRODUCK_DRIBBLE_BEARING": "0.30,0.30",
                         "MICRODUCK_DRIBBLE_SPEED": "0.12,0.22",
                         "MICRODUCK_DRIBBLE_LOST": "0.50,0.50",
                         # The corridor opens: 0.40 m of ground in which the
                         # ball is lost and the episode is not. Half-rate
                         # nudges so the state is SAMPLED before it is common
                         # — an unsampled state's value is never learned, and
                         # that is the whole reason this stage exists rather
                         # than turning the corridor on at full rate in one
                         # step.
                         "MICRODUCK_DRIBBLE_GONE": "0.90,0.90",
                         "MICRODUCK_DRIBBLE_NUDGE_RATE": "0.3,0.3",
                         # rung 2: 17 deg/s. Cold, this cost 4/20 falls; the
                         # rung exists so it is not met cold.
                         "MICRODUCK_DRIBBLE_TURN": "0.6,0.30",
                         # rung 2: 0.01 — 7.9 cm a touch, 4-7 touches a target.
                         # Measured at 3.75 touches a clip cold.
                         "MICRODUCK_DRIBBLE_BALL_ROLLING": f"{DRIBBLE_ROLL_LADDER[1]},{DRIBBLE_ROLL_LADDER[1]}",
                         # The DEEP window: 98 % of spawns have the ball in
                         # frame, so the believed command exists from step one
                         # and what this rung teaches is the RECOVERY, not the
                         # acquisition.
                         "MICRODUCK_DRIBBLE_GAZE_NECK": f"{DRIBBLE_GAZE_DEEP[0][0]},{DRIBBLE_GAZE_DEEP[0][1]}",
                         "MICRODUCK_DRIBBLE_GAZE_HEAD": f"{DRIBBLE_GAZE_DEEP[1][0]},{DRIBBLE_GAZE_DEEP[1][1]}"},
                        detail=("The command now points up to 17 degrees either side, so "
                                "the ball has to be moved somewhere other than straight "
                                "on — which is the aim the brain's push mode has never "
                                "had. The loss radius tightens.")),
        CurriculumStage("the full cone, at pace", 2_000_000,
                        {"MICRODUCK_DRIBBLE_BEARING": f"{DRIBBLE_BEARING},{DRIBBLE_BEARING}",
                         "MICRODUCK_DRIBBLE_SPEED": f"{DRIBBLE_SPEED[0]},{DRIBBLE_SPEED[1]}",
                         "MICRODUCK_DRIBBLE_LOST": f"{DRIBBLE_LOST},{DRIBBLE_LOST}",
                         "MICRODUCK_DRIBBLE_GONE": f"{DRIBBLE_GONE},{DRIBBLE_GONE}",
                         "MICRODUCK_DRIBBLE_NUDGE_RATE": f"{DRIBBLE_NUDGE_RATE},{DRIBBLE_NUDGE_RATE}",
                         # rung 3: 29 deg/s — the authority the task actually
                         # needs. Cold this is 16/20 falls and 0.023 m/s of
                         # ball progress; the ladder's whole job is to arrive
                         # here with the falls gone and the progress kept.
                         "MICRODUCK_DRIBBLE_TURN": "0.8,0.50",
                         # rung 3: upstream's own 0.0001. The honest one, and the
                         # one that decides the question: a touch sends
                         # this ball ~0.6 m, so if multi-touch survives
                         # the descent the duck is controlling it, and if
                         # it collapses back to 1.3 touches a target then
                         # a 15 g ball at this friction can only be
                         # KICKED by a 737 g duck — which is an answer.
                         "MICRODUCK_DRIBBLE_BALL_ROLLING": f"{DRIBBLE_ROLL_LADDER[2]},{DRIBBLE_ROLL_LADDER[2]}",
                         # ...and now it has to FIND it: 67 % of spawns start
                         # with the ball in frame, so a third of them begin
                         # with no command until the duck looks down.
                         "MICRODUCK_DRIBBLE_GAZE_NECK": f"{DRIBBLE_GAZE_NECK[0]},{DRIBBLE_GAZE_NECK[1]}",
                         "MICRODUCK_DRIBBLE_GAZE_HEAD": f"{DRIBBLE_GAZE_HEAD[0]},{DRIBBLE_GAZE_HEAD[1]}"},
                        detail=("The whole 29-degree cone and the whole speed range, with "
                                "the ball spawned anywhere in the box the feet can reach, "
                                "45 cm to lose it in and 1.2 m to get it back from. "
                                "Something knocks it off-line twice a second on average, "
                                "so the recovery is the task as much as the dribble is.")),
    ),
))
