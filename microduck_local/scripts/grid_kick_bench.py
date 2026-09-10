"""Whiff over the BOX: how much of the ball box play produces does a kick
policy connect on? (roadmap 12b / 12ah, 2026-09-10)

    cd microduck_local
    uv run python scripts/grid_kick_bench.py --foot right vendored=policies/kick/kick_right.onnx wide=runs/<run>/policy.onnx
    uv run python scripts/grid_kick_bench.py --foot left --seeds 2 vendored=policies/kick/kick_left.onnx

`bench_kick_headdown.py` places the ball on the sweet spot only, so it could
never say what a kick does with the ball 5 cm off it - which is where play
puts it (item 12a). This places the ball on a grid of (ahead, side) offsets
from the root (0.02-0.20 m x -0.02..0.14 m, to the kicking foot's side), per
head pose, runs the policy from standing under the bench's protocol
(obs_noise off, the honest condim-6 ball, 1.2 s) and prints a hit map per
pose plus: coverage of the 4-16 x 1-13 cm box, the sweet-spot rate, falls,
the |body turn| over the window and the exit angle. A hit is 0.10 m of
travel, the play probe's whiff line.

Read with 12ab in mind: the turn here spans the whole 1.2 s window (it
includes post-swing drift); the in-play turn is `probe_kick_recover.py`'s
first-look-frame number. Two seeds a cell by default - a half-filled cell is
one hit. Measured 2026-09-10: the vendored right strike connects on 70-80 %
of the box already (roadmap 12b).
"""
import argparse, math, sys
import mujoco, numpy as np
from microduck_local import contract as C
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.behaviors.kick import _kick_ball_ids, BALL_Z
from microduck_local.brain.brain_env import onnx_infer

ap = argparse.ArgumentParser()
ap.add_argument("--foot", default="right")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("policies", nargs="+")
a = ap.parse_args()
sgn = -1.0 if a.foot == "right" else 1.0
AHEAD = [round(0.02 + 0.02 * i, 2) for i in range(10)]     # 0.02 .. 0.20
SIDE = [round(-0.02 + 0.02 * i, 2) for i in range(9)]      # -0.02 .. 0.14 (to the kicking foot's side)
POSES = (("level", 0.0, 0.0), ("line-up gaze (head +0.60)", 0.0, 0.60), ("neck -0.30 / head +0.60", -0.30, 0.60))
BOX = (0.04, 0.16, 0.01, 0.13)

def yaw_of(env):
    qw, qx, qy, qz = (float(v) for v in env.data.qpos[3:7])
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

def run_one(env, infer, ahead, side, neck, head, seed):
    obs, _ = env.reset(seed=seed)
    _, qadr, dadr = _kick_ball_ids(env)
    yaw = yaw_of(env)
    ox, oy = ahead, sgn * side
    x = float(env.data.qpos[0]) + math.cos(yaw) * ox - math.sin(yaw) * oy
    y = float(env.data.qpos[1]) + math.sin(yaw) * ox + math.cos(yaw) * oy
    env.data.qpos[qadr:qadr + 7] = [x, y, BALL_Z, 1, 0, 0, 0]
    env.data.qvel[dadr:dadr + 6] = 0.0
    env.data.qpos[env.joint_qpos_adr[5]] = C.DEFAULT_POSE[5] + neck
    env.data.qpos[env.joint_qpos_adr[6]] = C.DEFAULT_POSE[6] + head
    env.data.ctrl[:] = env.data.qpos[env.joint_qpos_adr]
    if getattr(env, "bam", None) is not None:
        env.bam.reset(env.data.qpos[env.joint_qpos_adr])
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    d = env._kick_dir
    yaw0 = yaw
    fell = False
    for k in range(a.steps):
        obs, r, term, trunc, info = env.step(infer(obs))
        if term:
            fell = True
            break
    dx, dy = float(env.data.qpos[qadr]) - x, float(env.data.qpos[qadr + 1]) - y
    along = dx * d[0] + dy * d[1]
    travel = math.hypot(dx, dy)
    turn = math.degrees(math.atan2(math.sin(yaw_of(env) - yaw0), math.cos(yaw_of(env) - yaw0)))
    exit_deg = math.degrees(math.atan2(-dx * d[1] + dy * d[0], along)) if travel >= 0.10 else float("nan")
    return travel, along, fell, turn, exit_deg

for spec in a.policies:
    label, path = spec.split("=", 1)
    infer = onnx_infer(path)
    env = BehaviorEnv(f"kick_{a.foot}", seed=5, max_episode_s=a.steps * C.CTRL_DT + 0.5,
                      domain_rand=False, random_yaw=False, obs_noise=False, action_delay=False)
    print(f"\n===== {label}: {path} (foot {a.foot}, {a.steps * C.CTRL_DT:.1f} s, {a.seeds} seed(s) a cell) =====")
    for plabel, neck, head in POSES:
        hits = np.zeros((len(AHEAD), len(SIDE)))
        turns, exits, falls, in_box, in_box_hit, spot_hit, n_spot = [], [], 0, 0, 0, 0, 0
        for i, ah in enumerate(AHEAD):
            for j, sd in enumerate(SIDE):
                h = 0
                for s in range(a.seeds):
                    travel, along, fell, turn, ex = run_one(env, infer, ah, sd, neck, head, 500 + s)
                    hit = travel >= 0.10
                    h += hit; falls += fell; turns.append(abs(turn))
                    if not math.isnan(ex): exits.append(ex)
                    if BOX[0] <= ah <= BOX[1] and BOX[2] <= sd <= BOX[3]:
                        in_box += 1; in_box_hit += hit
                    if 0.06 <= ah <= 0.10 and 0.04 <= sd <= 0.08:
                        n_spot += 1; spot_hit += hit
                hits[i, j] = h / a.seeds
        print(f"\n-- {plabel}: connects on {100 * in_box_hit / max(in_box, 1):.0f}% of the box "
              f"({in_box_hit}/{in_box}), {100 * spot_hit / max(n_spot, 1):.0f}% of the sweet spot, "
              f"falls {falls}, |body turn| median {np.median(turns):.0f} deg, "
              f"exit median {np.median(exits) if exits else float('nan'):+.0f} deg (sd {np.std(exits) if len(exits) > 1 else float('nan'):.0f})")
        print("ahead\\side " + " ".join(f"{sd:>5.2f}" for sd in SIDE))
        for i, ah in enumerate(AHEAD):
            print(f"{ah:>10.2f} " + " ".join(("  ### " if v >= 0.99 else "  ##  " if v >= 0.5 else "  .   " if v > 0 else "      ") for v in hits[i]))
