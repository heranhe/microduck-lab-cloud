from __future__ import annotations

import math

import mujoco
import numpy as np

from .. import contract as C
from .ball import _ball_camera
from .core import (
    CATALOG,
    Behavior,
    CurriculumStage,
    RewardTerm,
    _register,
    _trunk_yaw,
)
from .dribble import _no_flight
from .kick import BALL_Z, _box_knob, _kick_ball_ids
from .locomotion import _run_cmd_norm, _run_pose, _run_speed, _run_upright

"""WALK WITH YOUR HEAD DOWN — the sub-skill the dribble is blocked on.

Not a ball task. This recipe exists because 12ax H ran out of reward ideas and
the measurement said why: **the duck cannot see the ball at the distance it
touches it, and it cannot stay upright while looking there.**

In-frame rate against ball distance for the best dribbler (20 clips):

    ball <= 0.15 m (touch range)    2.0 %
    ball 0.15 - 0.25 m              1.6 %
    ball 0.25 - 0.45 m             24.3 %
    ball > 0.45 m                  36.4 %

It sees the ball while it is far and is blind to it exactly when it is close
enough to kick. A soft touch is a TIMING skill — ball departure speed equals
the FOOT's speed at contact (ratio 0.98, r=+0.69), and the foot passes through
the body's own 0.19 m/s twice a stride — so the timing is available but its
control input is not. No reward can teach a policy to time something it cannot
observe, which is why four reward designs and a physics ladder measured null or
backwards.

THE INFORMATION IS BUYABLE. Clamping the head every step on the best dribbler,
20 clips each — `action` is an offset off `DEFAULT_POSE`, so this is exactly
what a policy would command:

| neck / head held | ep_len | fell | axis depression | in frame <= 0.15 m |
|---|---|---|---|---|
| the policy's own | 389 | 1/20 | 17.6 deg | **1.9 %** |
| -0.15 / +0.30 | 280 | 7/20 | 34.8 deg | 4.6 % |
| -0.30 / +0.60 | 123 | 19/20 | 56.2 deg | **65.5 %** |
| -0.45 / +0.95 | 56 | 20/20 | 70.6 deg | **91.1 %** |

So the camera CAN see a ball at the feet — 1.9 % to 91 % — and the duck falls
19-20 times in 20 trying. That is a GAIT deficiency, not a sensor limit: the
shipped walker was trained with its head level and has never balanced a head
pitched 60 deg down, which moves the heaviest thing on it forward of the hips.

So this recipe trains one thing: hold the head deep AND keep walking. The
ladder is read straight off that table — 0.30 survives partially (7/20 falls is
a gradient, not a wall), 0.60 is the rung that buys the sight, 0.95 is the
prize. A dribble can then warm-start off a body that can afford to look down.

WHY THE GAZE IS NOT COMMANDED THROUGH obs[51:55]. Those are the contract's
`head_pose_cmd` slots and this would be their proper use — but `lastmetre` and
`dribble` hijack them to carry the ball sighting, and a gait trained to track a
head command there would be conditioned on a channel the dribble fills with
something else entirely. The pose target is a STAGE CONSTANT instead, so what
transfers is an unconditional competence: this body can walk with its head
down. Nothing about the observation changes, which is the whole point of
warm-starting a dribble off it.
"""

#: Neck and head offsets off `DEFAULT_POSE` this stage asks the duck to hold.
#: Read off the clamp table above: the 0.60 rung is where the ball at the feet
#: comes into frame (65.5 %), and 0.95 is where it is properly centred (91 %).
GAZEWALK_HOLD = (-0.30, 0.60)
#: How wide a miss still pays, radians. Generous on purpose: at std 0.10 the
#: term is flat at the level head the donor starts from — the "term flat where
#: the policy starts" bug this package has been bitten by twice (`gaze_ball`'s
#: two bells, `turn_track`'s Gaussian). 0.45 gives gradient from 0.0 to 0.95.
GAZEWALK_STD = 0.45
#: Commanded forward speed. The dribble's own range, so the gait transfers.
GAZEWALK_SPEED = (0.12, 0.25)
#: Commanded yaw rate window, rad/s. ADDED after the first ladder shipped a
#: gait that walks STRAIGHT with its head down and falls 40/40 the moment the
#: dribble asks it to turn. The gaze transferred (depression 17.3 -> 52.3 deg in
#: the dribble env, in frame at touch range 2.3 % -> 11.2 %) and the turn did
#: not exist, so the two capabilities were trained separately and each was lost
#: when the other was. The dribble's own clip is 0.50 rad/s, so the gait has to
#: cover it.
GAZEWALK_YAW = 0.50
#: Where the ball sits ahead of the trunk, m — the dribble's own spawn band,
#: so the gait meets the obstacle it will actually have to walk around.
GAZEWALK_BALL = (0.09, 0.20)
GAZEWALK_EPISODE_S = 8.0


def _gazewalk_target(env) -> tuple[float, float]:
    """(neck, head) offsets this stage wants held."""
    return _box_knob(env, "MICRODUCK_GAZEWALK_HOLD", GAZEWALK_HOLD)


def _gaze_deep(env) -> float:
    """Holding the head where the stage asked, 0..1.

    UNGATED on purpose, and this is the one exception the recipe needs. Every
    other positive term here is gated on actually walking (`_run_cmd_norm`),
    because a duck standing still with a tidy head is not the skill. But the
    HOLD is the thing being taught, and gating it on locomotion would pay zero
    for the pose exactly while the duck is learning to survive it — the shape
    that made `gaze_ball` unlearnable in 12ax F. A still duck can collect this
    term and nothing else, which is 2.0 a step against the ~8 of walking with
    it held, the same 4:1 that prices standing elsewhere in this package.
    """
    nk, hd = _gazewalk_target(env)
    q = env.data.qpos
    dn = float(q[env.joint_qpos_adr[5]]) - (C.DEFAULT_POSE[5] + nk)
    dh = float(q[env.joint_qpos_adr[6]]) - (C.DEFAULT_POSE[6] + hd)
    return float(math.exp(-((dn / GAZEWALK_STD) ** 2 + (dh / GAZEWALK_STD) ** 2)))


def _gaze_depression(env) -> float:
    """The optical axis's depression in radians — what the pose is FOR.

    Reported rather than paid: the pose target is what the policy can control,
    and the depression follows from it. Kept so `report_fn` can say whether the
    rung actually bought the angle the clamp table promised."""
    _, fwd, _, _ = _ball_camera(env)
    return float(math.asin(max(-1.0, min(1.0, -float(fwd[2])))))


def _gaze_turn(env) -> float:
    """The signed fraction of the commanded yaw actually delivered, 0..1.

    The dribble's `turn_track` shape, for the dribble's reason: locomotion's
    `_run_track_ang` Gaussian has std^2 = 0.5 while these commands are
    0.0-0.5 rad/s, so a duck standing perfectly still scores 0.956 of it — a
    term flat across the entire command range. Scaled by the demand so a
    straight-ahead command pays nothing rather than paying for free."""
    clip = _box_knob(env, "MICRODUCK_GAZEWALK_YAW", (GAZEWALK_YAW, GAZEWALK_YAW))[1]
    cmd = float(env.twist_cmd[2])
    demand = min(1.0, abs(cmd) / max(clip, 1e-6))
    if demand < 1e-3:
        return 0.0
    return float(np.clip(float(env._gyro[2]) / cmd, 0.0, 1.0)) * demand


def _gazewalk_reset(env) -> None:
    """Spawn mid-walk with the head ALREADY at the stage's hold.

    Handed rather than discovered, for the reason the dribble's own spawn gaze
    exists: a rung whose rollouts never contain the pose cannot teach it. The
    duck still has to KEEP it — the measured drift is 72 % of the offset given
    back inside 40 steps — so the spawn is a starting point and not the task.
    """
    r = env._rng
    nk, hd = _gazewalk_target(env)
    env.data.qpos[env.joint_qpos_adr[5]] = C.DEFAULT_POSE[5] + nk
    env.data.qpos[env.joint_qpos_adr[6]] = C.DEFAULT_POSE[6] + hd
    spd = r.uniform(*_box_knob(env, "MICRODUCK_GAZEWALK_SPEED", GAZEWALK_SPEED))
    wzmax = _box_knob(env, "MICRODUCK_GAZEWALK_YAW", (GAZEWALK_YAW, GAZEWALK_YAW))[1]
    wz = float(r.uniform(-wzmax, wzmax))
    env.twist_cmd[0], env.twist_cmd[1], env.twist_cmd[2] = spd, 0.0, wz
    env._gazewalk_speed = spd
    env._gazewalk_yaw = wz
    yaw = _trunk_yaw(env)
    env.data.qvel[env._root_qvel + 0] = spd * math.cos(yaw)
    env.data.qvel[env._root_qvel + 1] = spd * math.sin(yaw)
    # THE BALL AT ITS FEET, and leaving it out was the defect that cost this
    # whole transfer. The first version left the ball where the scene put it —
    # inert, metres away — so the gait learned to walk head-down over BARE
    # FLOOR. Measured on the policy that came out of it: in the dribble env it
    # falls 20/20 with the ball present and **0/20 with the ball teleported
    # away**, same env, same steering, same observations. It walks perfectly at
    # 55 deg of depression and trips over the ball, every time.
    #
    # A duck looking at its own feet is a duck about to walk into whatever is
    # there, so the obstacle belongs in the rung that teaches the pose.
    _, qadr, dadr = _kick_ball_ids(env)
    yaw = _trunk_yaw(env)
    ox = float(r.uniform(*_box_knob(env, "MICRODUCK_GAZEWALK_BALL", GAZEWALK_BALL)))
    oy = float(r.uniform(-0.06, 0.06))
    env.data.qpos[qadr:qadr + 7] = [
        float(env.data.qpos[0]) + math.cos(yaw) * ox - math.sin(yaw) * oy,
        float(env.data.qpos[1]) + math.sin(yaw) * ox + math.cos(yaw) * oy,
        BALL_Z, 1.0, 0.0, 0.0, 0.0]
    env.data.qvel[dadr:dadr + 6] = 0.0
    env.data.ctrl[:] = env.data.qpos[env.joint_qpos_adr]
    if getattr(env, "bam", None) is not None:
        env.bam.reset(env.data.qpos[env.joint_qpos_adr])
    mujoco.mj_forward(env.model, env.data)
    env.last_spawn = (f"head held at neck {nk:+.2f} / head {hd:+.2f}, "
                      f"walking {spd:.2f} m/s")


def _gazewalk_obs(env) -> None:
    """Hold the commanded speed against `walk_env`'s mid-episode resample —
    the same guard the dribble needs, and for the same reason."""
    env.twist_cmd[0] = getattr(env, "_gazewalk_speed", 0.18)
    env.twist_cmd[1] = 0.0
    env.twist_cmd[2] = getattr(env, "_gazewalk_yaw", 0.0)


def _gazewalk_caption(env) -> str:
    nk, hd = _gazewalk_target(env)
    return (f"hold {nk:+.2f}/{hd:+.2f}  now "
            f"{float(env.data.qpos[env.joint_qpos_adr[5]]) - C.DEFAULT_POSE[5]:+.2f}/"
            f"{float(env.data.qpos[env.joint_qpos_adr[6]]) - C.DEFAULT_POSE[6]:+.2f}  "
            f"axis {math.degrees(_gaze_depression(env)):.0f}deg down")


_register(Behavior(
    id="gazewalk",
    emoji="👀",
    title="Walk with your head down",
    description=(
        "Hold the head pitched far enough down to see a ball at its own feet, "
        "and keep walking — the sub-skill the dribble is blocked on, because "
        "the duck is in frame on 2 % of touch-range steps and falls 19 times "
        "in 20 when the head is held where it could see."),
    how_it_learns=(
        "It is paid for two things at once and neither alone: holding the neck "
        "and head where the stage asks, and walking at the commanded speed "
        "without falling. The hold is spawned rather than discovered, because "
        "a rung whose rollouts never contain the pose cannot teach it, and the "
        "duck gives back 72 % of a handed gaze inside 40 steps — so keeping it "
        "IS the task. The ladder is read off a clamp measurement rather than "
        "guessed: holding neck -0.15 / head +0.30 costs 7 falls in 20 and buys "
        "almost no sight, -0.30/+0.60 puts the ball at the feet in frame on "
        "65 % of steps but costs 19 falls in 20, and -0.45/+0.95 reaches 91 % "
        "at 20 falls in 20. Each rung is the previous one's falls removed."),
    keywords=("gaze walk", "head down", "walk head down", "look down",
              "deep gaze", "gazewalk", "see its feet", "watch the ball"),
    terms=(
        # THE HOLD. The thing being taught, and the biggest income.
        RewardTerm("gaze_deep", "Points for holding the head where it can see its own feet",
                   8.0, _gaze_deep),
        # ...AND STILL WALKING. Without these the optimum is to stand with a
        # tidy head, which is the standing optimum this package has rebuilt
        # five times. `_run_speed` is gated on the commanded speed inside
        # locomotion, so it pays for delivering the walk and not for having
        # been asked.
        RewardTerm("keep_pace", "Points for walking at the commanded speed",
                   6.0, _run_speed),
        # TURNING WHILE HEAD-DOWN, which the first version of this recipe left
        # out entirely — and that omission cost the whole transfer: the gait it
        # produced held a 52 deg gaze in the dribble env and fell 40/40 there,
        # because the dribble commands up to 0.50 rad/s and this gait had only
        # ever walked straight. Same fraction-delivered shape as the dribble's
        # `turn_track` (NOT locomotion's Gaussian, which pays 0.956 to a duck
        # that refuses to turn at these command sizes).
        RewardTerm("turn_track", "Points for making the turn it is asked for, head down",
                   4.0, _gaze_turn),
        # Ungated, like the dribble's: falling is a precondition failure, not a
        # task failure, so it is priced whether or not the hold is going well.
        # At 2.0 a still duck collects gaze_deep 8.0 + this 2.0 = 10 against
        # ~16 for walking with the hold, which is the 1.6:1 that keeps standing
        # unattractive without making a fall free.
        RewardTerm("stay_upright", "Points for keeping the body upright",
                   2.0, _run_upright),
        RewardTerm("pose", "Points for a speed-appropriate leg pose while walking",
                   1.0, lambda env: _run_pose(env) * _run_cmd_norm(env)),
        RewardTerm("flat_feet", "Points for keeping the feet flat while walking",
                   2.0, lambda env: CATALOG["flat_feet"].fn(env) * _run_cmd_norm(env)),
        # The duck's gait has 0 % flight; a head-down policy inventing a hop to
        # keep its balance is the failure mode this docks directly, and the
        # dribble needed the same term for the same reason.
        RewardTerm("no_flight", "Docked every step with neither foot on the floor",
                   3.0, _no_flight, is_penalty=True),
        CATALOG["no_limit_parking"],
    ),
    symmetric=True,
    forward_cmd=GAZEWALK_SPEED[1],
    episode_s=GAZEWALK_EPISODE_S,
    scene="ball",          # the ball is INERT here; it is only what the in-frame metric measures against
    terminate_on_fall=True,
    reset_fn=_gazewalk_reset,
    obs_fn=_gazewalk_obs,
    caption_fn=_gazewalk_caption,
    default_steps=2_400_000,
    curriculum=(
        CurriculumStage("a glance down", 800_000,
                        {"MICRODUCK_GAZEWALK_HOLD": "-0.15,0.30"},
                        detail=("Neck -0.15, head +0.30: a 35 deg axis, which the "
                                "shipped gait already half-survives at 7 falls in 20. "
                                "It buys almost no sight (4.6 % at touch range) and is "
                                "not meant to — it is the rung where balancing a "
                                "forward head is learned at all.")),
        CurriculumStage("far enough to see its feet", 800_000,
                        {"MICRODUCK_GAZEWALK_HOLD": "-0.30,0.60"},
                        detail=("Neck -0.30, head +0.60: a 56 deg axis, and the rung "
                                "that matters — the ball at touch range is in frame on "
                                "65 % of steps here against 1.9 % at the policy's own "
                                "gaze. Cold it costs 19 falls in 20, which is exactly "
                                "what the previous rung exists to remove.")),
        CurriculumStage("the whole way down", 800_000,
                        {"MICRODUCK_GAZEWALK_HOLD": "-0.45,0.95"},
                        detail=("Neck -0.45, head +0.95: a 71 deg axis and 91 % in "
                                "frame at touch range, the pose a close dribble would "
                                "actually want. Cold it is 20 falls in 20 and an "
                                "ep_len of 56, so it is the top of the ladder and not "
                                "a starting point.")),
    ),
))
