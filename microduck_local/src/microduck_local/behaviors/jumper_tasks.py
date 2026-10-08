"""Env-owned Jumper prototype recipes, available through the teach panel."""

from .core import Behavior, RewardTerm, _register


def _jumper_env_owned(env):
    return 0.0


_JUMPER_TERMS = tuple(
    RewardTerm(key, friendly, weight, _jumper_env_owned, is_penalty=penalty)
    for key, friendly, weight, penalty in (
        ("track_linear_velocity", "tracks the requested speed", 2.0, False),
        ("track_angular_velocity", "tracks the requested turn", 1.0, False),
        ("upright", "holds the chassis level", 2.0, False),
        ("height", "keeps the body above the ground", 1.0, False),
        ("pose", "stays near its home joint configuration", 0.5, False),
        ("tripod_gait", "matches all six feet to the gait clock", 1.0, False),
        ("action_rate_penalty", "penalty for abrupt commands", 0.01, True),
        ("joint_vel_penalty", "penalty for excessive joint speed", 0.0001, True),
    )
)

JUMPER_STAND = Behavior(
    id="jumper_stand",
    suggest="stand still",
    emoji="🦀",
    title="Stand still (Jumper)",
    description="Practise a stable six-foot stance in the lab.",
    how_it_learns="The robot is scored for holding its home height and level chassis with all six feet planted.",
    keywords=("stand", "stand still", "站立", "站稳"),
    robot="jumper",
    terms=_JUMPER_TERMS,
    trainer=(
        "-m",
        "microduck_local.train",
        "--robot",
        "jumper",
        "--task",
        "stand",
        "--actuator",
        "xml",
    ),
)
JUMPER_TRIPOD = Behavior(
    id="jumper_tripod",
    suggest="walk with a tripod gait",
    emoji="🦀",
    title="Tripod gait (Jumper)",
    description="Prototype alternating LF/RM/LR and RF/LM/RR support groups.",
    how_it_learns="A 3.125 Hz clock is observable to the policy; six individual foot contacts are scored against the same clock. This recipe is a prototype, not a trained gait.",
    keywords=("tripod", "walk", "三足", "步态", "行走"),
    robot="jumper",
    terms=_JUMPER_TERMS,
    trainer=(
        "-m",
        "microduck_local.train",
        "--robot",
        "jumper",
        "--task",
        "tripod",
        "--actuator",
        "xml",
    ),
)

_register(JUMPER_STAND)
_register(JUMPER_TRIPOD)
