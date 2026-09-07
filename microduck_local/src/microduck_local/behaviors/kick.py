"""The kick, trained HERE, with the head down (roadmap item 7 / 4c revisit,
2026-09-07).

The shipped kicks (upstream `Mjlab-BallKick-Flat-MicroDuck`) were trained
from HOME +-0.05 rad and whiff 12 of 12 from the pose a duck is in when it
is LOOKING at the ball (head +0.95 / neck -0.05 against HOME +0.39 /
+0.21) - so in play the gaze is dropped before the swing and the swing
stands on a plan that is 3 s old. The upstream fix is a patch that waits
on a GPU; this is the same recipe on this Mac: the walk scene plus
upstream's ball (`contract.scene_walk_ball_xml`), the ball spawned on the
kicking foot's sweet spot with the same placement noise, and the head and
neck spawned ACROSS THE GAZE RANGE, so a swing that works head-down is in
the rollouts from the first minute (AGENTS.md: put the state in the world,
not in the pay).

The pay mirrors upstream's: ball speed along the kick line, linear up to
the target and capped there; a penalty for overshooting it; the SUPPORT
foot on the floor (anti-hop); legs near HOME loosely; neck and head
pulled home tighter, which is what brings the head up as it kicks. The
policy is ball-blind, as the shipped ones are: the brain aims it.

Two recipes, one per foot, as upstream trains them. The export is a
61-obs policy the arena runs as a skill: `MICRODUCK_SKILL_KICK_RIGHT=
runs/<run>/policy.onnx` (world/arena.py `start_skill`) puts it under the
scripted brain in place of the shipped one, so the kick can be measured
in play with the gaze HELD through the swing (`gaze_still=1,gaze_neck=1`).
"""

import math

import mujoco
import numpy as np

from .. import contract as C
from .core import Behavior, RewardTerm, _register

BALL_TARGET_SPEED = 1.0          # upstream BALL_TARGET_SPEED (m/s along the kick line)
BALL_OFFSET = (0.09, 0.042)      # ahead, |side| of the kicking foot at HOME (upstream reset_ball_in_front_of_foot)
BALL_NOISE = 0.015               # uniform +- per axis: the placement DR a blind kick must cover
BALL_Z = 0.035
HEAD_DOWN = (0.0, 0.60)          # head_pitch offset above HOME: level .. gazing at the feet
NECK_DOWN = (-0.30, 0.0)         # neck_pitch offset below HOME (the two go opposite ways)
LEGS_STD = 0.5
HEAD_STD = 0.3


def _kick_ball_ids(env) -> tuple[int, int, int]:
    """(body id, qpos address, dof address) of the ball, cached on the env."""
    b = getattr(env, "_kick_ball", None)
    if b is None:
        m = env.model
        bid = m.body("ball").id
        jid = int(m.body_jntadr[bid])
        b = env._kick_ball = (bid, int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]))
    return b


def _yaw(env) -> float:
    qw, qx, qy, qz = (float(v) for v in env.data.qpos[3:7])
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _kick_reset_for(side: str):
    sgn = -1.0 if side == "right" else 1.0

    def _kick_reset(env) -> None:
        _, qadr, dadr = _kick_ball_ids(env)
        r = env._rng
        yaw = _yaw(env)
        ox = BALL_OFFSET[0] + r.uniform(-BALL_NOISE, BALL_NOISE)
        oy = sgn * BALL_OFFSET[1] + r.uniform(-BALL_NOISE, BALL_NOISE)
        x = float(env.data.qpos[0]) + math.cos(yaw) * ox - math.sin(yaw) * oy
        y = float(env.data.qpos[1]) + math.sin(yaw) * ox + math.cos(yaw) * oy
        env.data.qpos[qadr:qadr + 7] = [x, y, BALL_Z, 1.0, 0.0, 0.0, 0.0]
        env.data.qvel[dadr:dadr + 6] = 0.0
        env._kick_dir = (math.cos(yaw), math.sin(yaw))
        # The head and neck across the gaze range: the state the shipped
        # kick never saw, sampled from the first episode.
        env.data.qpos[env.joint_qpos_adr[5]] = C.DEFAULT_POSE[5] + r.uniform(*NECK_DOWN)
        env.data.qpos[env.joint_qpos_adr[6]] = C.DEFAULT_POSE[6] + r.uniform(*HEAD_DOWN)
        env.data.ctrl[:] = env.data.qpos[env.joint_qpos_adr]
        if getattr(env, "bam", None) is not None:
            env.bam.reset(env.data.qpos[env.joint_qpos_adr])
        mujoco.mj_forward(env.model, env.data)

    _kick_reset.__name__ = f"_kick_reset_{side}"
    return _kick_reset


def ball_speed_along(env) -> float:
    """The ball's speed along the kick line set at reset (m/s, signed)."""
    _, _, dadr = _kick_ball_ids(env)
    v = env.data.qvel[dadr:dadr + 3]
    d = getattr(env, "_kick_dir", (1.0, 0.0))
    return float(v[0] * d[0] + v[1] * d[1])


def _ball_forward(env) -> float:
    return float(np.clip(ball_speed_along(env), 0.0, BALL_TARGET_SPEED)) / BALL_TARGET_SPEED


def _ball_overshoot(env) -> float:
    return -max(0.0, ball_speed_along(env) - BALL_TARGET_SPEED)          # <= 0, a self-negating penalty


def _support_for(side: str):
    stance = "left" if side == "right" else "right"

    def _support_foot(env) -> float:
        return 1.0 if env.foot_contact_state[stance] else 0.0

    _support_foot.__name__ = f"_support_foot_{stance}"
    return _support_foot


def _legs_home(env) -> float:
    q = env._joint_pos_rel()[C.LEG_JOINT_IDS]
    return float(np.exp(-((q / LEGS_STD) ** 2)).mean())


def _head_home(env) -> float:
    q = env._joint_pos_rel()[C.HEAD_JOINT_IDS]
    return float(np.exp(-((q / HEAD_STD) ** 2)).mean())


for _side in ("right", "left"):
    _register(Behavior(
        id=f"kick_{_side}",
        emoji="⚽",
        title=f"Kick the ball ({_side} foot)",
        description=(
            f"Kick a ball off the {_side} foot down the line it is facing, "
            "from standing - with the head anywhere from level to looking at its feet."
        ),
        how_it_learns=(
            "The ball starts on the foot's sweet spot, a centimetre either way, "
            "and the head starts anywhere between level and looking straight "
            "down at it - the pose a duck is really in when it has just been "
            "watching the ball. It is paid every step the ball rolls away "
            "along its line, up to a metre a second and docked past it, for "
            "keeping the other foot planted, for legs near the standing pose, "
            "and for bringing its head home. The swing itself it has to find; "
            "the ball on the foot from the first minute is what makes that "
            "findable."
        ),
        keywords=(f"kick {_side}", f"kick_{_side}", f"{_side} foot kick", "kick the ball", "kick", "shoot"),
        terms=(
            RewardTerm("ball_forward", "Points every step the ball rolls away along the kick line, up to 1 m/s",
                       12.0, _ball_forward),
            RewardTerm("ball_overshoot", "Docked for ball speed past the 1 m/s target",
                       4.0, _ball_overshoot, is_penalty=True),
            RewardTerm("support_foot", "Points every step the other foot stays on the floor",
                       2.0, _support_for(_side)),
            RewardTerm("legs_home", "Points for legs near the standing pose (loosely)",
                       2.0, _legs_home),
            RewardTerm("head_home", "Points for bringing the neck and head home",
                       1.0, _head_home),
        ),
        default_steps=2_000_000,
        success_metric="ball speed along the kick line at 0.5 s, from every head pose",
        symmetric=False,
        episode_s=2.0,
        scene="ball",
        terminate_on_fall=True,
        reset_fn=_kick_reset_for(_side),
    ))
