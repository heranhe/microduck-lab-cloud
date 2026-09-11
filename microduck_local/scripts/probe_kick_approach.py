"""Does the kick WALK to a ball it cannot reach? (roadmap 12as follow-up).

    cd microduck_local
    uv run python scripts/probe_kick_approach.py --foot right \
        v1=runs/lastmetre-right-v1/policy.onnx \
        approach=runs/lastmetre-right-v1-approach/policy.onnx

12as measured the sensed kick's box coverage and found the other half of
12h still open: "with the ball out of the swing's reach (0.22-0.45 m ahead)
the trunk advances 1-11 cm and the ball never moves". That was a scratch
probe. This is it, written down, so the approach rung can be judged against
the same number and by the same protocol.

**What it reports, per ball distance:** how far the TRUNK got along the kick
line, how far the BALL travelled, and the share of seeds where the ball moved
at all. Two horizons on every row - 2.0 s (the clip 12as measured in, and the
clip `kick_<side>_sensed` ships with) and the full `--seconds` - because a
policy trained on a 4 s clip that only starts walking at 2.5 s is a different
finding from one that never walks, and a single horizon cannot tell them
apart.

The protocol is `grid_kick_bench_sensed.py`'s, imported rather than copied:
the env is built from a NAMED recipe, so a sensed policy is not silently
driven through the blind recipe's env with its eyes taped shut. `seen@0` is
the positive control - a far ball is IN FRAME from a level head (the camera
sits 0.21 m up, so 0.40 m ahead is 28 deg below the axis, inside the 29 deg
half-VFOV) and OUT of frame from a head pitched at its own feet, which is why
both gazes are on the table.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from microduck_local import contract as C
from microduck_local.behaviors.kick import _kick_ball_ids
from microduck_local.brain.brain_env import onnx_infer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_kick_bench_sensed import make_env, place, yaw_of  # noqa: E402 — one protocol, one source

# 12as's own cells: inside the swing's reach (0.10, 0.16) and past it.
DISTANCES = (0.10, 0.16, 0.22, 0.30, 0.45)
POSES = (("level", 0.0, 0.0), ("neck -0.30 / head +0.60", -0.30, 0.60))
HIT = 0.10          # m of ball travel that counts as "the ball moved" (the benches')


def track(env, infer, obs, x0: float, y0: float, yaw0: float, steps: int, mark: int):
    """Roll, recording the trunk's advance along the kick line and the ball's
    travel at `mark` steps and at the end."""
    _, qadr, dadr = _kick_ball_ids(env)
    d = env._kick_dir
    tx0, ty0 = float(env.data.qpos[0]), float(env.data.qpos[1])

    def adv() -> float:
        return (float(env.data.qpos[0]) - tx0) * d[0] + (float(env.data.qpos[1]) - ty0) * d[1]

    def ball() -> float:
        return math.hypot(float(env.data.qpos[qadr]) - x0, float(env.data.qpos[qadr + 1]) - y0)

    seen0 = 1.0 if float(obs[53]) > 0.5 else 0.0
    at_mark = None
    fell, peak_adv = False, 0.0
    for k in range(steps):
        obs, _, term, _, _ = env.step(infer(obs))
        peak_adv = max(peak_adv, adv())
        if k + 1 == mark:
            at_mark = (adv(), ball())
        if term:
            fell = True
            break
    if at_mark is None:
        at_mark = (adv(), ball())
    turn = math.degrees(math.atan2(math.sin(yaw_of(env) - yaw0), math.cos(yaw_of(env) - yaw0)))
    return at_mark, (adv(), ball()), peak_adv, fell, abs(turn), seen0


def run(label: str, path: str, recipe: str, foot: str, seconds: float, seeds: int,
        side: float) -> None:
    sgn = -1.0 if foot == "right" else 1.0
    steps = int(round(seconds / C.CTRL_DT))
    mark = min(steps, int(round(2.0 / C.CTRL_DT)))
    infer = onnx_infer(path)
    env = make_env(recipe, steps)
    print(f"\n===== {label}: {path}\n      env {recipe}  (foot {foot}, ball {side:.3f} m to the "
          f"{foot}, {seeds} seeds a cell, horizon {seconds:.1f} s)")
    for plabel, neck, head in POSES:
        print(f"\n-- gaze {plabel}")
        print(f"  {'ball ahead':>11}{'advance@2s':>12}{'ball@2s':>9}{'moved@2s':>10}"
              f"{'advance@T':>11}{'ball@T':>9}{'moved@T':>9}{'peak adv':>10}"
              f"{'turn':>7}{'falls':>7}{'seen@0':>8}")
        for ahead in DISTANCES:
            a2, b2, aT, bT, pk, turns = [], [], [], [], [], []
            moved2 = movedT = falls = 0
            seens = []
            for s in range(seeds):
                obs, (x0, y0), yaw0 = place(env, ahead, side, neck, head, sgn, 300 + s)
                (am, bm), (ae, be), peak, fell, turn, seen0 = track(
                    env, infer, obs, x0, y0, yaw0, steps, mark)
                a2.append(am); b2.append(bm); aT.append(ae); bT.append(be)
                pk.append(peak); turns.append(turn); seens.append(seen0)
                moved2 += bm >= HIT
                movedT += be >= HIT
                falls += fell
            print(f"  {ahead:>10.2f}m{np.median(a2):>11.3f}m{np.median(b2):>8.2f}m"
                  f"{moved2 / seeds:>10.0%}{np.median(aT):>10.3f}m{np.median(bT):>8.2f}m"
                  f"{movedT / seeds:>9.0%}{np.median(pk):>9.3f}m{np.median(turns):>7.0f}"
                  f"{falls:>7}{100 * np.mean(seens):>7.0f}%")


def main() -> None:
    global DISTANCES
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--foot", default="right", choices=("right", "left"))
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="rollout horizon; the 2.0 s column is always reported too")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--distances", default=None, metavar="M,M,...",
                    help="ball distances ahead of the trunk, m (default: "
                         + ",".join(f"{d:g}" for d in DISTANCES) + "). The obs's range slot "
                         "saturates at lastmetre.LM_RANGE_SCALE, so a sweep ACROSS that value "
                         "is how you tell a curriculum ceiling from an observation ceiling")
    ap.add_argument("--side", type=float, default=0.042,
                    help="ball to the kicking foot's side (m) — the recipe's sweet-spot side")
    ap.add_argument("--recipe", default=None,
                    help="behavior id whose env every arm is driven through; default: "
                         "kick_<foot>_sensed for a run under runs/, kick_<foot> otherwise "
                         "(override per arm with label:recipe=path)")
    ap.add_argument("policies", nargs="+", metavar="LABEL[:RECIPE]=PATH")
    a = ap.parse_args()
    if a.distances:
        DISTANCES = tuple(float(v) for v in a.distances.split(","))
    for spec in a.policies:
        label, path = spec.split("=", 1)
        recipe = a.recipe
        if ":" in label:
            label, recipe = label.split(":", 1)
        if recipe is None:
            recipe = f"kick_{a.foot}_sensed" if "runs/" in path else f"kick_{a.foot}"
        run(label, path, recipe, a.foot, a.seconds, a.seeds, a.side)


if __name__ == "__main__":
    main()
