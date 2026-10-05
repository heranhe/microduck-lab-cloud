"""A SCRIPTED grasp — the TEACHER, and the test of whether one can exist.

    uv run python scripts/scripted_pick.py [seeds] [variety]

Eight RL runs and ~13M steps never learned to aim the jaw at a lying object
(slope +0.046, +0.021, -0.074, -0.200, -0.134, +0.010, -0.053, -0.186), and a
drill that isolated the skill moved crossed-start grasps 22/36 -> 22/36. The
remaining route is a teacher: demonstrate it, distil it into the weights, then
let RL fine-tune past it. Nothing can be distilled from a teacher that does not
work, so this measures the teacher FIRST.

It uses the policy's own action space (five arm increments + a gripper
increment), the base locked, and drives the tool point with the damped-least-
squares Jacobian step this repo already uses in `pose.solve_ik`. The wrist
angle comes from the SENSED axis in slots 28-29 — the observation the robot
really has — not from simulator truth, so anything learned from it is
deployable.
"""
import collections
import math
import sys

import mujoco
import numpy as np

from microduck_local.robots import moss
from microduck_local.robots.moss_env import ARM_DELTA_RAD, MossPickEnv

ARM = moss.ARM_JOINTS
ACT_GRIP = moss.ACT_ARM.stop
PREGRASP_M = 0.055          # how high above the object to stage
#: MEASURED from a working policy's grasps (40 seeds): where the tool
#: point sits, relative to the object's centre, when the jaw closes.
GRASP_DZ_UP = 0.032         # upright: gripped near the top
GRASP_DZ_LIE = 0.012        # lying/tilted: just above the centre
LAMBDA = 0.08               # DLS damping


def _jaw_yaw(env) -> float:
    m, d = env.model, env.data
    fl = m.joint("finger_left")
    ax = d.xmat[m.jnt_bodyid[fl.id]].reshape(3, 3) @ np.asarray(fl.axis, float)
    _x, _y, byaw = env.driver.pose(d)
    c, s = math.cos(-byaw), math.sin(-byaw)
    return math.atan2(s * ax[0] + c * ax[1], c * ax[0] - s * ax[1])


def act(env, obs) -> np.ndarray:
    """One control step: stage above, descend, close, lift."""
    m, d = env.model, env.data
    tcp = np.asarray(d.site_xpos[env.tcp_site], float)
    can = np.asarray(d.xpos[env.can_body], float)
    a = np.zeros(env.action_space.shape[0], np.float32)

    cx = float(obs[moss.OBS_TARGET_AXIS][0])
    sy = float(obs[moss.OBS_TARGET_AXIS][1])
    upright = float(obs[moss.OBS_TARGET_UPRIGHT][0])
    seen = abs(cx) + abs(sy) > 1e-6
    half = env.prop.half_height if upright > 0.5 else env.prop.radius
    top = float(can[2]) + half

    horiz = float(np.hypot(tcp[0] - can[0], tcp[1] - can[1]))
    st = env._tstate
    if st == "over" and horiz < 0.015 and tcp[2] > top + 0.02:
        env._tstate = st = "down"
    elif st == "down" and tcp[2] <= float(can[2]) + (
            GRASP_DZ_UP if upright > 0.6 else GRASP_DZ_LIE) + 0.006:
        env._tstate = st = "close"
    elif st == "close" and env._closed_on_can():
        env._tstate = st = "lift"

    if st == "over":
        tgt = np.array([can[0], can[1], top + PREGRASP_M])
    elif st == "down":
        # MEASURED off a working policy at the moment it closes, rather than
        # guessed: the tool point sits +12.4 mm above a LYING object's centre
        # and +31.6 mm above an UPRIGHT one's — an upright can is gripped near
        # its top, where the pads come down over it, not at its waist. Aiming
        # at the top stalled 24 of 40 seeds 13 mm short; aiming at the waist
        # drove the palm into the can and scored 4/60.
        tgt = np.array([can[0], can[1], float(can[2]) + GRASP_DZ_UP
                        if upright > 0.6 else float(can[2]) + GRASP_DZ_LIE])
    elif st == "close":
        tgt = np.array([can[0], can[1], tcp[2]])
    else:
        tgt = np.array([can[0], can[1], top + 0.14])

    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, env.tcp_site)
    cols = [int(m.joint(j).dofadr[0]) for j in ARM]
    J = jacp[:, cols]
    e = tgt - tcp
    dq = J.T @ np.linalg.solve(J @ J.T + LAMBDA * LAMBDA * np.eye(3), e)
    a[moss.ACT_ARM] = np.clip(dq / ARM_DELTA_RAD, -1, 1)

    # THE WRIST: square to the object's axis. One line of arithmetic on a
    # SENSED quantity — the thing eight RL runs never found.
    if seen and upright < 0.6 and st in ("over", "down"):
        err = (0.5 * math.atan2(sy, cx) + math.pi / 2) - _jaw_yaw(env)
        err = math.atan2(math.sin(err), math.cos(err))
        a[moss.ACT_ARM.start + 4] = float(np.clip(err / ARM_DELTA_RAD, -1, 1))

    a[ACT_GRIP] = -1.0 if st in ("close", "lift") else 1.0
    return a


def main() -> None:
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    variety = len(sys.argv) > 2 and sys.argv[2] == "variety"
    by_shape = collections.defaultdict(lambda: [0, 0])
    by_pose = collections.defaultdict(lambda: [0, 0])
    for seed in range(seeds):
        env = MossPickEnv(seed=seed, pick_rung=2, base_lock=True,
                          publish_attitude=True, publish_size=True,
                          prop_variety=variety)
        obs, _ = env.reset()
        env._tstate = "over"
        up = abs(env._true_attitude()[2])
        pose = "upright" if up > 0.85 else ("lying" if up < 0.35 else "tilted")
        info = {}
        n = 0
        while n < int(env.max_steps):
            obs, _r, t, tr, info = env.step(act(env, obs))
            n += 1
            if t or tr:
                break
        ok = int(bool(info.get("picked")))
        by_shape[env.prop.id][0] += ok
        by_shape[env.prop.id][1] += 1
        by_pose[pose][0] += ok
        by_pose[pose][1] += 1
    tot = sum(v[0] for v in by_shape.values())
    n = sum(v[1] for v in by_shape.values())
    print(f"  SCRIPTED TEACHER: {tot}/{n} ({tot/n:.0%})")
    print("     by shape: " + "  ".join(f"{k} {v[0]}/{v[1]}"
                                        for k, v in sorted(by_shape.items())))
    print("     by pose : " + "  ".join(f"{k} {v[0]}/{v[1]}"
                                        for k, v in sorted(by_pose.items())))


if __name__ == "__main__":
    main()
