"""Does a get-up policy actually get up, from each way of lying down?
(roadmap B.1 / bead mdl-0ad)

    cd microduck_local
    uv run python scripts/bench_getup.py --policy runs/getup-l5/policy.onnx
    uv run python scripts/bench_getup.py --policy limp          # the null control
    uv run python scripts/bench_getup.py --policy runs/getup-l5/policy.onnx \\
        --actuator xml --tilt 45 75                             # one rung of the ladder

The `getup` recipe's own env, spawned flat on the BACK, the FRONT and the
SIDE in turn, driven by the DETERMINISTIC exported ONNX (never a training
curve — AGENTS.md verification discipline #1). Per pose:

  recovered   the fraction of episodes that reach a stand AND still hold it
              at the end of the episode. "A stand" is the recipe's own
              `_getup_hold_raw` gate — both feet down, nothing else on the
              floor, trunk upright, trunk tall — so the bench and the reward
              cannot drift apart about what standing means.
  t_stand     seconds from the spawn to the first qualifying stand (median
              over the episodes that reached one).
  held        seconds of qualifying stand in the episode, and the longest
              unbroken stretch: a policy that pops up and falls over scores
              a high `recovered` only if it is still standing at the buzzer,
              and the streak is what separates a stand from a bounce.
  full        the same, but also with the HEAD up where a standing duck's
              head is (>= 80% of the STAND keyframe's jaw height). The
              recipe's gate does not price head height, and the no-ladder
              control exploits exactly that: it recovers from a face-down
              start into a head-down half-crouch, trunk 0.099 of 0.120 and
              jaw at 0.055 of 0.233, which clears every gate by a few
              millimetres. `alpha_stand` finishes the same episodes with
              its jaw at 0.234. Read the two columns together — the gap
              between them is how much of a "recovery" is posture.

`--policy limp` and `--policy zero` are the null controls the discipline
asks for: a duck lying on the floor under a limp command stays there, so
any recovery the bench reports is the policy's and not gravity's.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from microduck_local import contract as C
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.behaviors.getup import _getup_hold_raw

# Fraction of the STAND keyframe's jaw height a duck must reach for its
# stand to count as a FULL one rather than a head-down half-crouch.
HEAD_UP_FRAC = 0.80

POSES = ("back", "front", "side")
# Positional spawn-family probabilities, in the recipe's own order.
_FAMILY_PROBS = {"back": "1.0,0.0,0.0", "front": "0.0,1.0,0.0",
                 "side": "0.0,0.0,1.0"}


def _stand_head_z(env, jaw_body_id: int) -> float:
    """The jaw height at the STAND keyframe of THIS env — the reference a
    full stand is measured against. Read off the model rather than written
    down, so a model change cannot leave the bench quoting a stale number."""
    import mujoco

    data = mujoco.MjData(env.model)
    mujoco.mj_resetDataKeyframe(env.model, data, env.key_stand)
    mujoco.mj_forward(env.model, data)
    return float(data.xpos[jaw_body_id][2])


def _driver(policy: str):
    """(name, fn(obs) -> action). `limp`/`zero` are the null controls."""
    if policy in ("limp", "zero"):
        z = np.zeros(C.NUM_JOINTS, np.float32)
        return policy, (lambda obs: z)
    from microduck_local.brain.brain_env import onnx_infer
    infer = onnx_infer(Path(policy))
    return policy, (lambda obs: np.asarray(infer(obs), np.float32))


def run_pose(policy: str, pose: str, seeds: int, seconds: float,
             actuator: str, tilt: tuple[float, float] | None,
             noise: bool = False) -> dict:
    import os

    name, act = _driver(policy)
    os.environ["MICRODUCK_SPAWN_FAMILY_PROBS"] = _FAMILY_PROBS[pose]
    if tilt is not None:
        os.environ["MICRODUCK_GETUP_TILT_LO"] = str(tilt[0])
        os.environ["MICRODUCK_GETUP_TILT_HI"] = str(tilt[1])
    steps = int(round(seconds / C.CTRL_DT))
    recovered, full, t_stand, held_s, streak_s = 0, 0, [], [], []
    for s in range(seeds):
        env = BehaviorEnv("getup", seed=1000 + s, max_episode_s=seconds + 1.0,
                          random_yaw=False, actuator=actuator,
                          obs_noise=noise, domain_rand=noise,
                          action_delay=noise)
        jaw = env.model.body("jaw_soft").id
        # The head height a STAND looks like, measured off this env's own
        # keyframe rather than hard-coded (render_rollout prints the same
        # 0.233 m reference).
        head_ref = _stand_head_z(env, jaw)
        obs, _ = env.reset(seed=1000 + s)
        first, held, streak, best = None, 0, 0, 0
        standing = head_high = False
        for i in range(steps):
            obs, _, terminated, truncated, _ = env.step(act(obs))
            standing = _getup_hold_raw(env) > 0.0
            head_high = float(env.data.xpos[jaw][2]) >= HEAD_UP_FRAC * head_ref
            if standing:
                held += 1
                streak += 1
                best = max(best, streak)
                if first is None:
                    first = i
            else:
                streak = 0
            if terminated or truncated:
                break
        # "Recovered" needs BOTH: it stood at some point, and it is still
        # standing at the buzzer. Either half alone has been mistaken for a
        # get-up here — the first by a policy that bounces upright and falls
        # back, the second by one that is merely lucky on the last frame.
        if first is not None and standing:
            recovered += 1
            if head_high:
                full += 1
        if first is not None:
            t_stand.append(first * C.CTRL_DT)
        held_s.append(held * C.CTRL_DT)
        streak_s.append(best * C.CTRL_DT)
        env.close()
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    return {"policy": name, "pose": pose, "seeds": seeds,
            "recovered": recovered / seeds, "full": full / seeds,
            "reached_stand": len(t_stand) / seeds,
            "t_stand_med": med(t_stand), "held_med": med(held_s),
            "streak_med": med(streak_s), "streak_max": max(streak_s or [0.0])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True,
                    help="policy.onnx, or 'limp'/'zero' for the null control")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--actuator", default="bam", choices=("bam", "xml"),
                    help="bam = the honest XL330 model (the default, and what "
                         "any claim about the robot must be made under)")
    ap.add_argument("--tilt", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="spawn tilt window in degrees off upright (default: "
                         "the recipe's own 80-115, i.e. flat on the floor)")
    ap.add_argument("--noise", action="store_true",
                    help="observation noise + domain randomization + the action "
                         "delay, i.e. the conditions a policy meets on the robot "
                         "rather than the clean ones a bench flatters it with")
    ap.add_argument("--poses", default=",".join(POSES))
    ap.add_argument("--out", default=None, help="write the rows as JSON here")
    args = ap.parse_args()

    rows = [run_pose(args.policy, p, args.seeds, args.seconds, args.actuator,
                     tuple(args.tilt) if args.tilt else None, args.noise)
            for p in args.poses.split(",") if p.strip()]
    tilt = f"{args.tilt[0]:.0f}-{args.tilt[1]:.0f}" if args.tilt else "80-115 (flat)"
    print(f"\n{args.policy}   actuator={args.actuator}  tilt={tilt}  "
          f"{'noisy' if args.noise else 'clean'}  "
          f"{args.seeds} seeds x {args.seconds:.0f}s")
    print(f"{'pose':<6} {'recovered':>10} {'full':>6} {'reached':>8} {'t_stand':>9} "
          f"{'held s':>8} {'streak s':>9} {'max':>6}")
    for r in rows:
        print(f"{r['pose']:<6} {r['recovered']:>9.0%} {r['full']:>6.0%} "
              f"{r['reached_stand']:>8.0%} "
              f"{r['t_stand_med']:>9.2f} {r['held_med']:>8.2f} "
              f"{r['streak_med']:>9.2f} {r['streak_max']:>6.2f}")
    n = max(len(rows), 1)
    print(f"{'ALL':<6} {sum(r['recovered'] for r in rows) / n:>9.0%} "
          f"{sum(r['full'] for r in rows) / n:>6.0%}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
