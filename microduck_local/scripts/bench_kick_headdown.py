"""Does a kick policy swing from the pose a duck is in when it has been
LOOKING at the ball? (roadmap item 7 / 4c revisit)

    cd microduck_local
    uv run python scripts/bench_kick_headdown.py --policy ../microduck/policies/ball_kick_right.onnx --foot right
    uv run python scripts/bench_kick_headdown.py --policy runs/kick-right-headdown-v1/policy.onnx --foot right

The kick recipe's own env (behaviors/kick.py: the walk scene plus the ball,
the ball on the foot's sweet spot with the placement noise) with the head
and neck SET to each pose of the gaze range - level (HOME), the shipped
gaze (head +0.60), the line-up gaze that puts the ball on the spot (head
+0.95 / neck -0.05 against HOME +0.39 / +0.21), and the neck-split gaze
(neck -0.30 / head +0.60) - then the policy runs from standing for 1.2 s.
A WHIFF is the ball moving less than 10 cm along the kick line, the same
line the kick probe draws. Per pose: whiff rate, the ball's travel and
peak speed, and falls. The shipped right kick measured 12/12 whiffs from
head +0.97 (item 7); a kick trained with the head down should not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from microduck_local import contract as C
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.behaviors.kick import _kick_ball_ids, ball_speed_along
from microduck_local.brain.brain_env import onnx_infer

POSES = (                              # (label, neck offset, head offset) off HOME
    ("level", 0.0, 0.0),
    ("head +0.30", 0.0, 0.30),
    ("head +0.60 (shipped gaze clamp)", 0.0, 0.60),
    ("neck -0.30 / head +0.60 (the split)", -0.30, 0.60),
    ("head +0.95 / neck -0.25 (the line-up gaze)", -0.25, 0.95),
)


def run(policy: str, foot: str, seeds: int, steps: int) -> list[dict]:
    infer = onnx_infer(Path(policy))
    rows = []
    for label, neck, head in POSES:
        travel, peak, peaks_t, exits, whiffs, falls = [], [], [], [], 0, 0
        for s in range(seeds):
            env = BehaviorEnv(f"kick_{foot}", seed=1000 + s, max_episode_s=steps * C.CTRL_DT + 0.5,
                              domain_rand=False, random_yaw=False)
            obs, _ = env.reset(seed=1000 + s)
            env.data.qpos[env.joint_qpos_adr[5]] = C.DEFAULT_POSE[5] + neck
            env.data.qpos[env.joint_qpos_adr[6]] = C.DEFAULT_POSE[6] + head
            env.data.ctrl[:] = env.data.qpos[env.joint_qpos_adr]
            if getattr(env, "bam", None) is not None:
                env.bam.reset(env.data.qpos[env.joint_qpos_adr])
            mujoco.mj_forward(env.model, env.data)
            obs = env._get_obs()
            _, qadr, _ = _kick_ball_ids(env)
            x0, y0 = float(env.data.qpos[qadr]), float(env.data.qpos[qadr + 1])
            d = env._kick_dir
            v, t_peak = 0.0, 0
            fell = False
            for k in range(steps):
                obs, r, term, trunc, info = env.step(infer(obs))
                sp = ball_speed_along(env)
                if sp > v:
                    v, t_peak = sp, k + 1
                if term:
                    fell = True
                    break
            dx, dy = float(env.data.qpos[qadr]) - x0, float(env.data.qpos[qadr + 1]) - y0
            along = dx * d[0] + dy * d[1]
            travel.append(along)
            peak.append(v)
            peaks_t.append(t_peak * C.CTRL_DT)
            # The exit line: where the ball went, off the body heading (+ = left).
            if along >= 0.10:
                exits.append(np.degrees(np.arctan2(-dx * d[1] + dy * d[0], along)))
            whiffs += along < 0.10
            falls += fell
        rows.append({"pose": label, "neck": neck, "head": head, "n": seeds, "whiff": whiffs / seeds,
                     "travel_med": float(np.median(travel)), "peak_med": float(np.median(peak)), "falls": falls,
                     "peak_t_med": float(np.median(peaks_t)),
                     "exit_med": float(np.median(exits)) if exits else float("nan"),
                     "exit_sd": float(np.std(exits)) if len(exits) > 1 else float("nan")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--foot", default="right", choices=("right", "left"))
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=60, help="control steps the policy runs (60 = 1.2 s)")
    args = ap.parse_args()
    rows = run(args.policy, args.foot, args.seeds, args.steps)
    print(f"{args.policy}  ({args.foot} foot, {args.seeds} seeds a pose, {args.steps * C.CTRL_DT:.1f} s)")
    print(f"  {'head pose':<44}{'whiff':>7}{'travel med':>12}{'peak m/s':>10}{'at':>7}{'exit deg':>10}{'sd':>6}{'falls':>7}")
    for r in rows:
        print(f"  {r['pose']:<44}{r['whiff']:>7.0%}{r['travel_med']:>11.2f}m{r['peak_med']:>10.2f}{r['peak_t_med']:>6.2f}s"
              f"{r['exit_med']:>+10.1f}{r['exit_sd']:>6.1f}{r['falls']:>7}")


if __name__ == "__main__":
    main()
