"""The gaze-pose bench and the box grid, for a kick recipe that SEES the ball
(roadmap 12h / E.1, 2026-09-10).

    cd microduck_local
    uv run python scripts/grid_kick_bench_sensed.py --foot right \
        shipped=policies/kick/kick_right.onnx sensed=runs/lastmetre-right-v1/policy.onnx
    uv run python scripts/grid_kick_bench_sensed.py --foot right --mode poses \
        sensed=runs/lastmetre-right-v1/policy.onnx

`scripts/grid_kick_bench.py` and `scripts/bench_kick_headdown.py` both drive
the policy through `BehaviorEnv(f"kick_{foot}", ...)` — the BLIND recipe's
env, whose four head command slots carry keep-alive noise and nothing else.
A sensed policy run there is a policy with its eyes taped shut, and the
number that comes back looks like a result. This is the same two benches
with the ENV built from a named recipe (`--recipe-a/--recipe-b`, or the
`<label>:<recipe>=<path>` form), so each arm is measured in the world its
own observation describes, and it says which env every row came from.

The arithmetic is the two originals', deliberately duplicated so the numbers
are comparable row for row: a hit is 0.10 m of ball travel, 60 control steps
(1.2 s) from standing, `obs_noise` and `domain_rand` off, the honest condim-6
ball. The grid's AHEAD/SIDE/POSES/BOX below are copied from
`grid_kick_bench.py` and must stay equal to it.

**The positive control is printed, not assumed** (AGENTS.md: before believing
a null, check the thing you measured could have moved). Every run reports
`seen@0` — the share of cells whose head slots actually carried the ball at
the first step. A sensed policy accidentally driven through the blind env
reads 0 % there, and so does a blind policy: the column is what tells you
which happened.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from microduck_local import contract as C
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.behaviors.kick import BALL_Z, _kick_ball_ids
from microduck_local.brain.brain_env import onnx_infer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_kick_headdown import POSES as BENCH_POSES  # noqa: E402 — one source for the gaze poses

# Copied from grid_kick_bench.py — keep equal to it or the two tables stop
# being comparable.
AHEAD = [round(0.02 + 0.02 * i, 2) for i in range(10)]     # 0.02 .. 0.20
SIDE = [round(-0.02 + 0.02 * i, 2) for i in range(9)]      # -0.02 .. 0.14 (to the kicking foot's side)
GRID_POSES = (("level", 0.0, 0.0), ("line-up gaze (head +0.60)", 0.0, 0.60), ("neck -0.30 / head +0.60", -0.30, 0.60))
BOX = (0.04, 0.16, 0.01, 0.13)
SPOT = (0.06, 0.10, 0.04, 0.08)


def yaw_of(env) -> float:
    qw, qx, qy, qz = (float(v) for v in env.data.qpos[3:7])
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def reseed_task_state(env) -> None:
    """Re-run the recipe's sensing on the ball WHERE THE BENCH JUST PUT IT.

    `lastmetre._lm_obs` senses once per control step and the reset has already
    spent step 0's report on the recipe's own spawn, so without this the slots
    would describe a ball that is no longer there — a bench measuring its own
    reset. Blind recipes have none of this state and are left alone.
    """
    if not hasattr(env, "_lm_world"):
        return
    from microduck_local.behaviors.lastmetre import _lm_sense
    env._lm_world, env._lm_conf, env._lm_det_seen = None, 0.0, False
    env._lm_det_step = -10 ** 9
    env._lm_seen_steps = 0
    _lm_sense(env, force=True)
    env._lm_step_done = env.step_count


def make_env(recipe: str, steps: int, seed: int = 5) -> BehaviorEnv:
    return BehaviorEnv(recipe, seed=seed, max_episode_s=steps * C.CTRL_DT + 0.5,
                       domain_rand=False, random_yaw=False, obs_noise=False, action_delay=False)


def place(env, ahead: float, side: float, neck: float, head: float, sgn: float, seed: int):
    """The bench protocol, `grid_kick_bench.py`'s to the line: reset, ball at
    (ahead, sgn*side) off the root, the gaze set, and everything else left as
    the recipe's own reset produced it — head yaw included (pinning it here
    made the line-up row read 62 % against the original's 57 %)."""
    env.reset(seed=seed)
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
    env._kick_dir = (math.cos(yaw), math.sin(yaw))
    reseed_task_state(env)
    return env._get_obs(), (x, y), yaw


def roll(env, infer, obs, x0: float, y0: float, yaw0: float, steps: int):
    # The positive control, as a FLAG: in a blind env obs[53] is keep-alive
    # noise, and its MEAN reads as a plausible-looking 4 %.
    seen0 = 1.0 if float(obs[53]) > 0.5 else 0.0
    seen_steps, n_steps = seen0, 1.0
    fell = False
    peak = 0.0
    _, qadr, dadr = _kick_ball_ids(env)
    d = env._kick_dir
    for _ in range(steps):
        obs, _, term, _, _ = env.step(infer(obs))
        seen_steps += 1.0 if float(obs[53]) > 0.5 else 0.0
        n_steps += 1.0
        v = env.data.qvel[dadr:dadr + 3]
        peak = max(peak, float(v[0] * d[0] + v[1] * d[1]))
        if term:
            fell = True
            break
    dx = float(env.data.qpos[qadr]) - x0
    dy = float(env.data.qpos[qadr + 1]) - y0
    along = dx * d[0] + dy * d[1]
    travel = math.hypot(dx, dy)
    turn = math.degrees(math.atan2(math.sin(yaw_of(env) - yaw0), math.cos(yaw_of(env) - yaw0)))
    exit_deg = math.degrees(math.atan2(-dx * d[1] + dy * d[0], along)) if travel >= 0.10 else float("nan")
    return travel, along, peak, fell, turn, exit_deg, seen0, seen_steps / n_steps


def grid(label: str, path: str, recipe: str, foot: str, steps: int, seeds: int) -> None:
    sgn = -1.0 if foot == "right" else 1.0
    infer = onnx_infer(path)
    env = make_env(recipe, steps)
    print(f"\n===== {label}: {path}\n      env {recipe}  (foot {foot}, {steps * C.CTRL_DT:.1f} s, {seeds} seed(s) a cell) =====")
    for plabel, neck, head in GRID_POSES:
        hits = np.zeros((len(AHEAD), len(SIDE)))
        turns, exits, seens, seen_shares = [], [], [], []
        falls, in_box, in_box_hit, spot_hit, n_spot = 0, 0, 0, 0, 0
        for i, ah in enumerate(AHEAD):
            for j, sd in enumerate(SIDE):
                h = 0
                for s in range(seeds):
                    obs, (x0, y0), yaw0 = place(env, ah, sd, neck, head, sgn, 500 + s)
                    travel, along, _, fell, turn, ex, seen0, seen_s = roll(env, infer, obs, x0, y0, yaw0, steps)
                    hit = travel >= 0.10
                    h += hit
                    falls += fell
                    turns.append(abs(turn))
                    seens.append(seen0)
                    seen_shares.append(seen_s)
                    if not math.isnan(ex):
                        exits.append(ex)
                    if BOX[0] <= ah <= BOX[1] and BOX[2] <= sd <= BOX[3]:
                        in_box += 1
                        in_box_hit += hit
                    if SPOT[0] <= ah <= SPOT[1] and SPOT[2] <= sd <= SPOT[3]:
                        n_spot += 1
                        spot_hit += hit
                hits[i, j] = h / seeds
        print(f"\n-- {plabel}: connects on {100 * in_box_hit / max(in_box, 1):.0f}% of the box "
              f"({in_box_hit}/{in_box}), {100 * spot_hit / max(n_spot, 1):.0f}% of the sweet spot, "
              f"falls {falls}, |body turn| median {np.median(turns):.0f} deg, "
              f"exit median {np.median(exits) if exits else float('nan'):+.0f} deg "
              f"(sd {np.std(exits) if len(exits) > 1 else float('nan'):.0f}), "
              f"seen@0 {100 * np.mean(seens):.0f}%, seen over the window {100 * np.mean(seen_shares):.0f}% "
              f"of steps / {100 * np.mean([x > 0 for x in seen_shares]):.0f}% of cells ever")
        print("ahead\\side " + " ".join(f"{sd:>5.2f}" for sd in SIDE))
        for i, ah in enumerate(AHEAD):
            print(f"{ah:>10.2f} " + " ".join(("  ### " if v >= 0.99 else "  ##  " if v >= 0.5
                                              else "  .   " if v > 0 else "      ") for v in hits[i]))


def poses(label: str, path: str, recipe: str, foot: str, steps: int, seeds: int,
          ahead: float, side: float) -> None:
    """bench_kick_headdown's table: the ball on the recipe's own sweet spot,
    the gaze set to each pose of the range, 1.2 s from standing."""
    sgn = -1.0 if foot == "right" else 1.0
    infer = onnx_infer(path)
    env = make_env(recipe, steps)
    print(f"\n===== {label}: {path}\n      env {recipe}  (foot {foot}, ball {ahead:.3f} m ahead / "
          f"{side:.3f} m to the {foot}, {seeds} seeds a pose, {steps * C.CTRL_DT:.1f} s) =====")
    print(f"  {'head pose':<44}{'whiff':>7}{'travel med':>12}{'peak m/s':>10}{'exit deg':>10}{'sd':>6}"
          f"{'turn':>7}{'falls':>7}{'seen@0':>8}")
    for plabel, neck, head in BENCH_POSES:
        travel, peak, exits, turns, seens, whiffs, falls = [], [], [], [], [], 0, 0
        for s in range(seeds):
            obs, (x0, y0), yaw0 = place(env, ahead, side, neck, head, sgn, 1000 + s)
            _, along, pk, fell, turn, ex, seen0, _ = roll(env, infer, obs, x0, y0, yaw0, steps)
            travel.append(along)
            peak.append(pk)
            turns.append(abs(turn))
            seens.append(seen0)
            if not math.isnan(ex):
                exits.append(ex)
            whiffs += along < 0.10
            falls += fell
        print(f"  {plabel:<44}{whiffs / seeds:>7.0%}{np.median(travel):>11.2f}m{np.median(peak):>10.2f}"
              f"{(np.median(exits) if exits else float('nan')):>+10.1f}"
              f"{(np.std(exits) if len(exits) > 1 else float('nan')):>6.1f}"
              f"{np.median(turns):>7.0f}{falls:>7}{100 * np.mean(seens):>7.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--foot", default="right", choices=("right", "left"))
    ap.add_argument("--mode", default="grid", choices=("grid", "poses"))
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=2, help="grid: seeds a cell (poses mode defaults to 12)")
    ap.add_argument("--recipe", default=None,
                    help="behavior id whose env every arm is driven through; default: "
                         "kick_<foot> for a path under policies/ or ../microduck, "
                         "kick_<foot>_sensed for a run under runs/ (override per arm with label:recipe=path)")
    ap.add_argument("--ahead", type=float, default=0.09, help="poses mode: ball ahead of the root (m)")
    ap.add_argument("--side", type=float, default=0.042, help="poses mode: ball to the kicking foot's side (m)")
    ap.add_argument("policies", nargs="+", metavar="LABEL[:RECIPE]=PATH")
    a = ap.parse_args()
    seeds = a.seeds if (a.mode == "grid" or a.seeds != 2) else 12
    for spec in a.policies:
        label, path = spec.split("=", 1)
        recipe = a.recipe
        if ":" in label:
            label, recipe = label.split(":", 1)
        if recipe is None:
            # A run directory is this recipe's own export; a vendored/shipped
            # file is blind. Named explicitly rather than guessed silently.
            recipe = f"kick_{a.foot}_sensed" if "runs/" in path else f"kick_{a.foot}"
        if a.mode == "grid":
            grid(label, path, recipe, a.foot, a.steps, seeds)
        else:
            poses(label, path, recipe, a.foot, a.steps, seeds, a.ahead, a.side)


if __name__ == "__main__":
    main()
