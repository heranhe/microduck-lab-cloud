from .airflip import *  # noqa: F401,F403 — cascades the full upstream namespace,

# mirroring the flat file's definition order exactly (each module sees
# everything defined before it, helpers included).

# ------------------------------------------------ imitation (authored motion)
# Instead of discovering a maneuver, TRACK one an animator keyframed. The
# reward stops being a landscape to explore and becomes "be where the clip
# says you should be right now" — the standard fix when a trick is too hard
# to stumble into, and the reason most published robot flips exist.

def _mi_pose_match(env) -> float:
    """Joints matching the reference pose for this instant. Two-layer: the
    wide layer keeps a gradient when the body is nowhere near the clip
    (which is where every run starts), the tight layer pays for precision."""
    ref, _ = env.clip.at(env.step_count)
    q = env._joint_qpos()
    d2 = float(((q - ref) ** 2).sum())
    return (0.5 * float(np.exp(-d2 / 6.0 ** 2))
            + 0.5 * float(np.exp(-d2 / 2.0 ** 2)))


def _mi_rotation_match(env) -> float:
    """Body rotation matching the clip's timeline.

    Two regimes, because one measure cannot serve both. A one-shot FLIP needs
    the accumulator (rotation past 180° must keep counting). A LOOPING gait
    needs the instantaneous attitude — and using the accumulator there was a
    silent disaster: it only counts BACKWARD rotation and clips at zero, so a
    duck face-planting FORWARD scored a constant ~80% of this term while
    lying on the floor. Nothing in the recipe noticed it had fallen."""
    _, ref_pitch = env.clip.at(env.step_count)
    if env.clip.loop:
        g = env._projected_gravity()
        pitch_now = float(np.arctan2(g[0], -g[2]))   # 0 upright, + = nose-down
        d = pitch_now - ref_pitch
    else:
        # Clip pitch is sim/editor convention (negative = lean back); the
        # accumulator counts backward rotation as positive. Negate to compare.
        d = float(env._bf_rot) - (-ref_pitch)
    return (0.5 * float(np.exp(-(d ** 2) / 2.0 ** 2))
            + 0.5 * float(np.exp(-(d ** 2) / 0.7 ** 2)))


def _mi_on_feet(env) -> float:
    """Still standing on your feet, for a LOOPING gait: upright AND at
    standing height AND actually touching the floor with a foot. A pose-match
    reward alone is blind to this — the joint angles of a run cycle are just
    as matchable lying face-down on the floor, which is exactly what the
    first run policy learned to do."""
    if not env.clip.loop:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] or c["right"]):
        return 0.0
    z = float(env._trunk_xpos[2])
    tall = float(np.exp(-((z - env.stand_z) ** 2) / 0.05 ** 2))
    return float(_upright(env)) * tall


def _mi_no_slip(env) -> float:
    """Penalty for a planted foot SLIDING (<= 0). Upstream's walking task
    prices this at -0.1 for a reason: without it the cheapest way to satisfy
    a gait's poses is to skate — feet moving through the cycle while stuck to
    the floor — which looks like walking and travels nowhere."""
    if not env.clip.loop:
        return 0.0
    total = 0.0
    for side in ("left", "right"):
        if not env.foot_contact_state[side]:
            continue
        v6 = _v6_buf(env)
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_GEOM,
                                 env.foot_geoms[side], v6, 0)
        total += float(v6[3] ** 2 + v6[4] ** 2)
    return -total


def _mi_no_spin(env) -> float:
    """Penalty for yawing (<= 0). Upstream commands zero turn rate and pays
    for tracking it; we have no command channel here, so price the spin
    directly — the duck was corkscrewing instead of running straight."""
    if not env.clip.loop:
        return 0.0
    return -float(env._gyro[2] ** 2)


def _mi_travel(env) -> float:
    """For a LOOPING clip (a gait): actually go somewhere. Matching a run
    cycle's poses is satisfied just as well by running on the spot, which is
    the classic way an imitation reward gets gamed — so forward travel is
    priced on its own, and only for loops."""
    if not env.clip.loop:
        return 0.0
    fwd, _, _ = _base_vel(env)
    return float(np.clip(fwd / 0.3, 0.0, 1.0)) * float(_upright(env))


def _mi_end_upright(env) -> float:
    """Once a ONE-SHOT clip is over, be standing — the landing is the part a
    reference clip cannot specify, because it depends on how the physics
    actually went. A loop has no end, so this stays out of its way."""
    if env.clip.loop or env.step_count < env.clip.steps:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] and c["right"]):
        return 0.0
    z = float(env._trunk_xpos[2])
    tall = float(np.exp(-((z - env.stand_z) ** 2) / 0.06 ** 2))
    return float(_upright(env)) * tall


_register(Behavior(
    id="imitate",
    emoji="🎬",
    title="Copy the animation",
    description=(
        "Physically perform a keyframed motion clip: match the authored pose "
        "and rotation at every instant, then land and stand."
    ),
    how_it_learns=(
        "This one is not asked to invent anything. An animation clip says "
        "where every joint should be at each moment, and the duck earns for "
        "being there — so the hard search for 'how do I flip' becomes the far "
        "easier problem of 'follow this'. What it still has to solve is the "
        "physics: momentum, contact and balance are not in the clip, and no "
        "keyframe can fake them."
    ),
    keywords=("copy the animation", "imitate", "follow the clip",
              "animation clip", "motion clip", "do the animation"),
    terms=(
        RewardTerm("pose_match", "Points for matching the animation's pose right now",
                   4.0, _mi_pose_match),
        RewardTerm("rotation_match", "Points for matching the animation's rotation timing",
                   4.0, _mi_rotation_match),
        RewardTerm("stick_it", "Big points for standing on both feet when a trick clip ends",
                   5.0, _mi_end_upright),
        RewardTerm("on_feet", "Points for staying on its feet at full height (looping clips)",
                   5.0, _mi_on_feet),
        RewardTerm("no_slip", "Penalty for planted feet skating along the floor", 0.5,
                   _mi_no_slip, is_penalty=True),
        RewardTerm("no_spin", "Penalty for turning instead of going straight", 0.3,
                   _mi_no_spin, is_penalty=True),
        RewardTerm("travel", "Points for actually moving forward (looping clips like a run)",
                   3.0, _mi_travel),
        RewardTerm("gentle_head", "Penalty each time the head hits the floor, scaled by impact",
                   1.0, _gentle_plant_pen, is_penalty=True),
        RewardTerm("soft_landings", "Penalty for slamming down hard", 0.75,
                   _soft_landing_pen, is_penalty=True),
        RewardTerm("no_limit_parking", "Penalty for grinding joints against their end stops", 1.0,
                   _limit_parking_pen, is_penalty=True),
        RewardTerm("save_energy", "Penalty for straining the motors", 0.5,
                   _torque_pen, is_penalty=True),
    ),
    default_steps=3_000_000,
    success_metric="how closely the authored motion is reproduced, and whether it lands",
    episode_s=4.0,        # the clip plus time to settle on the landing
    scene="all",
    terminate_on_fall=False,
    state_fn=_bf_update,  # the rotation accumulator the clip is compared against
    clip_name="backflip",
    # Asymmetric because it carries a CLIP, not because of any clip's own
    # left/right content. The clip phase rides in body_cmd[4:6] (obs 59/60,
    # see _get_obs) and the mirror map negates the cos slot, so M sends phase
    # p to pi - p: it TIME-REFLECTS the clip. The prior then asks for the
    # mirrored action at an unrelated instant of the motion, which no clip
    # satisfies — the shipped backflip clip is exactly mirror-symmetric
    # frame by frame and still fails it. That is why this is a flat False and
    # not derived from the clip; tests/test_symmetry.py carries the numbers.
    symmetric=False,
))




ONE_LEG_SUCCESS_STEPS = round(5.0 / C.CTRL_DT)


def _one_leg_5s_update(env):
    """Count consecutive valid control steps, not cumulative episode uptime.

    This timer is evaluation metadata, NOT a hidden-state reward. Learning
    still uses the existing dense one-leg/pose terms at every control step.
    """
    contacts = env.foot_contact_state
    fwd, lat, _ = _base_vel(env)
    stable = (
        contacts["left"] and not contacts["right"]
        and _foot_z(env, "right") - _foot_z(env, "left") >= 0.03
        and env._projected_gravity()[2] <= -np.cos(np.deg2rad(20.0))
        and env._trunk_xpos[2] >= 0.8 * env.stand_z
        and np.hypot(fwd, lat) <= 0.1
        and np.linalg.norm(env._gyro) <= 1.0
    )
    env._one_leg_hold_steps = env._one_leg_hold_steps + 1 if stable else 0
    env._one_leg_best_steps = max(env._one_leg_best_steps, env._one_leg_hold_steps)


_register(replace(
    BEHAVIORS["one_leg"],
    id="one_leg_5s",
    title="单脚站立 · 连续 5 秒",
    description="左脚支撑、右脚离地至少 3 厘米，连续稳定站立 5 秒才算成功。",
    how_it_learns=(
        "每一步奖励单脚支撑、直立、目标姿势与平稳动作；抬起的脚触地、"
        "支撑脚离地或姿态不稳定时，连续计时清零。每回合 20 秒，"
        "不能把多段短暂站立累计成 5 秒。训练与预览都使用 BAM 舵机。"
    ),
    keywords=("单脚站立5秒", "one leg 5 seconds", "single leg 5 seconds"),
    # Position/yaw are not in the shared observation contract. Keep the
    # velocity-based stay_put term, not world-anchored home penalties.
    terms=tuple(t for t in BEHAVIORS["one_leg"].terms
                if t.key not in {"stay_home", "face_home"}) + (
        RewardTerm("pose_match", "接近照片中的抬腿目标姿势", 4.0, _mi_pose_match),
    ),
    clip_name="photo_one_leg",
    state_fn=_one_leg_5s_update,
    episode_s=20.0,
    default_steps=6_000_000,
    success_metric="连续单脚稳定站立 ≥ 5.00 秒（250 个控制步），中断清零",
))


# Star-export EVERYTHING (helpers included) so downstream modules and the
# package __init__ can reassemble the old flat-module surface exactly.
__all__ = [n for n in dir() if not n.startswith("__")]
