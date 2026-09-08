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

# (label, neck offset, head offset) off HOME - offsets, added to
# C.DEFAULT_POSE (head pitch 0.349 rad at home), so "+0.60" is 0.95 rad
# absolute: the line-up gaze in play (`head_down` 0.6 is a command offset,
# controllers.py). The last row was labelled "the line-up gaze" until
# 2026-09-08; it is 0.35 rad PAST it (a code review caught the label).
POSES = (
    ("level", 0.0, 0.0),
    ("head +0.30", 0.0, 0.30),
    ("head +0.60 = 0.95 rad abs (the line-up gaze: shipped head_down 0.6)", 0.0, 0.60),
    ("neck -0.30 / head +0.60 (the split)", -0.30, 0.60),
    ("neck -0.25 / head +0.95 = 1.30 rad abs (past the line-up gaze)", -0.25, 0.95),
)


def run(policy: str, foot: str, seeds: int, steps: int, gain_ratio: float = 1.0) -> list[dict]:
    infer = onnx_infer(Path(policy))
    rows = []
    for label, neck, head in POSES:
        travel, peak, peaks_t, exits, whiffs, falls = [], [], [], [], 0, 0
        for s in range(seeds):
            env = BehaviorEnv(f"kick_{foot}", seed=1000 + s, max_episode_s=steps * C.CTRL_DT + 0.5,
                              domain_rand=False, random_yaw=False)
            obs, _ = env.reset(seed=1000 + s)
            if gain_ratio != 1.0:
                # What the arena does for a kick window (`_set_gain_ratio`,
                # STANDING_GAIN_RATIO 0.8): Kp and its matching bias term.
                env.model.actuator_gainprm[:, 0] *= gain_ratio
                env.model.actuator_biasprm[:, 1] *= gain_ratio
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


VARIANTS = ("exact", "ball_rest", "home", "home_rest")


def replay(rows: list[dict], steps: int, gain_ratio: float, variants=VARIANTS) -> dict:
    """Roadmap item 12a: each recorded PLAY swing (probe_kick_line --dump-state)
    replayed in the kick recipe's own env from that exact state, under four
    variants that separate the candidates: `exact` (the duck's pose,
    velocities, last action and the ball's motion as they were); `ball_rest`
    (the ball's velocity zeroed); `home` (the duck put back in the standing
    HOME pose the recipe trains from, the ball left where it was);
    `home_rest` (both). A whiff is the ball moving < 10 cm along the body
    heading, as in play."""
    infer = {}
    out = {v: [] for v in variants}
    for i, row in enumerate(rows):
        st = row.get("state")
        if not st or row["foot"] not in ("kick_left", "kick_right"):
            continue
        foot = row["foot"].split("_")[1]
        if foot not in infer:
            infer[foot] = onnx_infer(Path(f"policies/kick/kick_{foot}.onnx"))
        for variant in variants:
            env = BehaviorEnv(f"kick_{foot}", seed=1000 + i, max_episode_s=steps * C.CTRL_DT + 0.5,
                              domain_rand=False, random_yaw=False)
            env.reset(seed=1000 + i)
            if gain_ratio != 1.0:
                env.model.actuator_gainprm[:, 0] *= gain_ratio
                env.model.actuator_biasprm[:, 1] *= gain_ratio
            d = env.data
            d.qpos[0:7] = st["root_qpos"]
            d.qvel[0:6] = st["root_qvel"]
            d.qpos[env.joint_qpos_adr] = st["joint_qpos"]
            d.qvel[env.joint_qvel_adr] = st["joint_qvel"]
            env.last_action[:] = np.asarray(st["last_action"], np.float32)
            env.prev_action = env.last_action.copy()
            env.prev_joint_vel[:] = np.asarray(st["prev_joint_vel"], np.float32)
            d.ctrl[:] = st["ctrl"]
            _, qadr, dadr = _kick_ball_ids(env)
            d.qpos[qadr:qadr + 7] = st["ball_qpos"]
            d.qvel[dadr:dadr + 6] = st["ball_qvel"]
            if variant.startswith("home"):
                d.qpos[env.joint_qpos_adr] = C.DEFAULT_POSE
                d.qvel[env.joint_qvel_adr] = 0.0
                d.qvel[0:6] = 0.0
                env.last_action[:] = 0.0
                env.prev_action = env.last_action.copy()
                env.prev_joint_vel[:] = 0.0
                d.ctrl[:] = C.DEFAULT_POSE
            if variant.endswith("rest"):
                d.qvel[dadr:dadr + 6] = 0.0
            env.twist_cmd[:] = 0.0
            env.head_cmd[:] = 0.0
            env.body_cmd[:] = 0.0
            if getattr(env, "bam", None) is not None:
                env.bam.reset(d.qpos[env.joint_qpos_adr])
            mujoco.mj_forward(env.model, d)
            qw, qx, qy, qz = (float(x) for x in d.qpos[3:7])
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            env._kick_dir = (float(np.cos(yaw)), float(np.sin(yaw)))
            obs = env._get_obs()
            x0, y0 = float(d.qpos[qadr]), float(d.qpos[qadr + 1])
            fell = False
            for _ in range(steps):
                obs, r, term, trunc, info = env.step(infer[foot](obs))
                if term:
                    fell = True
                    break
            dx, dy = float(d.qpos[qadr]) - x0, float(d.qpos[qadr + 1]) - y0
            along = dx * env._kick_dir[0] + dy * env._kick_dir[1]
            out[variant].append({"along": along, "whiff": along < 0.10, "fell": fell,
                                 "ahead": row.get("ahead"), "side": row.get("side"),
                                 "play_whiff": None if row.get("moved") is None else row["dist"] < 0.10,
                                 "ball_speed": float(np.hypot(st["ball_qvel"][0], st["ball_qvel"][1]))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=None, help="the kick ONNX to bench (not needed with --from-swings: the vendored local kicks)")
    ap.add_argument("--foot", default="right", choices=("right", "left"))
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=60, help="control steps the policy runs (60 = 1.2 s; play gives a kick 25 = 0.5 s)")
    ap.add_argument("--from-swings", default=None, metavar="ROWS.jsonl",
                    help="replay the play swings recorded by probe_kick_line --dump-state --out ROWS.jsonl (item 12a)")
    ap.add_argument("--gain-ratio", type=float, default=1.0,
                    help="scale the actuators' Kp as the arena does for a kick window (STANDING_GAIN_RATIO 0.8)")
    args = ap.parse_args()
    if args.from_swings:
        import json
        rows = [json.loads(line) for line in open(args.from_swings) if line.strip()]
        res = replay(rows, args.steps, args.gain_ratio)
        n_play = sum(1 for r in rows if r.get("state"))
        print(f"{args.from_swings}: {n_play} play swings replayed on the bench, {args.steps * C.CTRL_DT:.1f} s window, gain x{args.gain_ratio:g}")
        print(f"  play itself: whiff {np.mean([r['dist'] < 0.10 for r in rows if r.get('state')]):.0%}")
        print(f"  {'variant':<11}{'n':>4}{'whiff':>7}{'travel med':>12}{'falls':>7}")
        for v, rs in res.items():
            print(f"  {v:<11}{len(rs):>4}{np.mean([r['whiff'] for r in rs]):>7.0%}{np.median([r['along'] for r in rs]):>11.2f}m{sum(r['fell'] for r in rs):>7}")
        ex = res["exact"]
        print("  exact, by where the ball was (ahead of the root):")
        for lo, hi in ((0.0, 0.08), (0.08, 0.11), (0.11, 0.15), (0.15, 0.20), (0.20, 9.0)):
            g = [r for r in ex if r["ahead"] is not None and lo <= r["ahead"] < hi]
            if g:
                pw = [r["play_whiff"] for r in g if r["play_whiff"] is not None]
                print(f"    {lo:.2f}-{hi:.2f} m  n {len(g):3d}  bench whiff {np.mean([r['whiff'] for r in g]):4.0%}   play whiff {np.mean(pw) if pw else float('nan'):4.0%}")
        return
    rows = run(args.policy, args.foot, args.seeds, args.steps, args.gain_ratio)
    print(f"{args.policy}  ({args.foot} foot, {args.seeds} seeds a pose, {args.steps * C.CTRL_DT:.1f} s, gain x{args.gain_ratio:g})")
    print(f"  {'head pose':<44}{'whiff':>7}{'travel med':>12}{'peak m/s':>10}{'at':>7}{'exit deg':>10}{'sd':>6}{'falls':>7}")
    for r in rows:
        print(f"  {r['pose']:<44}{r['whiff']:>7.0%}{r['travel_med']:>11.2f}m{r['peak_med']:>10.2f}{r['peak_t_med']:>6.2f}s"
              f"{r['exit_med']:>+10.1f}{r['exit_sd']:>6.1f}{r['falls']:>7}")


if __name__ == "__main__":
    main()
