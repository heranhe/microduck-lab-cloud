"""Does a get-up policy recover from the poses a duck ACTUALLY falls into?
(roadmap B.1)

    cd microduck_local
    uv run python scripts/probe_fall_poses.py --seeds 24 --dump /tmp/falls.json
    uv run python scripts/probe_getup.py --falls /tmp/falls.json \\
        --policy runs/getup-l2/policy.onnx --noise
    uv run python scripts/probe_getup.py --falls /tmp/falls.json --policy limp

`scripts/bench_getup.py` scores a policy against the get-up recipe's own
SYNTHETIC lie — a tilt drawn from a window, a declared back/front/side mix,
both legs folded by one shared scalar. This script scores it against the
poses `scripts/probe_fall_poses.py` recorded from ducks that actually fell
over: the full `qpos` is replayed into the recipe's own env, so the
population under test is the real one and the two benches differ in
nothing else.

It is the denominator-side companion to the bench, in the sense of
AGENTS.md's "a rate's denominator must come from the same population as its
numerator": the bench's 36-of-36 is a real measurement over a population
that was designed rather than observed, and the only way to find out
whether that mattered is to run the same policy over the observed one.

Scoring is the bench's, term for term, so the two tables can be read
against each other:

  stood       reached the recipe's own four-gate stand (`_getup_hold_raw`:
              both feet down, nothing else on the floor, trunk upright,
              trunk tall) at some point AND still standing at the buzzer.
  t_stand     seconds from the replay to the first qualifying stand.
  held3       the longest unbroken stand was at least 3 s — the brief's
              "does it stay up" bar, separate from "did it get up".
  full        `stood`, and with the head where a standing duck's head is
              (>= 80% of the STAND keyframe's jaw height). The recipe
              prices head POSE and not head HEIGHT, and rungs 3-5 of the
              shipped ladder exploit exactly that.

`--policy limp` / `zero` are the null controls: a duck lying on the floor
under a released servo stays there, so any recovery reported is the
policy's and not the replay's.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from microduck_local import contract as C
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.behaviors.getup import _getup_hold_raw

# Reuse the bench's definitions rather than restating them: a second copy of
# "what a full stand is" is a second thing to drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_getup import HEAD_UP_FRAC, _driver, _stand_head_z  # noqa: E402

HELD_BAR_S = 3.0


def _replay(env, qpos: list[float]) -> None:
    """Put the duck back in a recorded resting pose.

    No drop and no settle, unlike `_getup_place`: the pose came off a duck
    that had already stopped moving at its own resting height, so re-dropping
    it would hand the policy unspent potential energy — the very artefact the
    recipe's spawn settles away. The x/y translation is zeroed (flat floor,
    and the obs contract carries no world position) so a long walk before the
    fall cannot walk the replay off the scene.
    """
    import mujoco

    d, m = env.data, env.model
    d.qpos[:] = np.asarray(qpos, float)
    d.qpos[0:2] = 0.0
    d.qvel[:] = 0.0
    # The servos hold what they fell in — a real robot's controller is still
    # running when it lands, and the recipe's spawn primes the same way.
    d.ctrl[:] = d.qpos[env.joint_qpos_adr]
    mujoco.mj_forward(m, d)
    if getattr(env, "bam", None) is not None:
        env.bam.reset(d.qpos[env.joint_qpos_adr])
    env.prev_joint_vel = env._joint_vel().copy()


def run_fall(policy_act, row: dict, seconds: float, actuator: str,
             noise: bool, seed: int) -> dict:
    env = BehaviorEnv("getup", seed=seed, max_episode_s=seconds + 1.0,
                      random_yaw=False, actuator=actuator, obs_noise=noise,
                      domain_rand=noise, action_delay=noise,
                      spawn_overrides={"MICRODUCK_SPAWN_FAMILY_PROBS": "0,0,0"})
    jaw = env.model.body("jaw_soft").id
    head_ref = _stand_head_z(env, jaw)
    env.reset(seed=seed)
    _replay(env, row["qpos"])
    obs = env._get_obs()
    steps = int(round(seconds / C.CTRL_DT))
    first, streak, best = None, 0, 0
    standing = head_high = False
    for i in range(steps):
        obs, _, terminated, truncated, _ = env.step(policy_act(obs))
        standing = _getup_hold_raw(env) > 0.0
        head_high = float(env.data.xpos[jaw][2]) >= HEAD_UP_FRAC * head_ref
        if standing:
            streak += 1
            best = max(best, streak)
            if first is None:
                first = i
        else:
            streak = 0
        if terminated or truncated:
            break
    env.close()
    return {"mode": row["mode"], "cls": row["cls"], "seed": row["seed"],
            "tilt_deg": row["tilt_deg"],
            # "Stood" needs both halves, as the bench insists: it got up, and
            # it is still up at the buzzer. Either alone has been mistaken
            # for a get-up here.
            "stood": bool(first is not None and standing),
            "full": bool(first is not None and standing and head_high),
            "reached": first is not None,
            "t_stand": (first * C.CTRL_DT) if first is not None else float("nan"),
            "streak_s": best * C.CTRL_DT,
            "held3": bool(best * C.CTRL_DT >= HELD_BAR_S)}


def render_sheet(policy_act, name: str, falls: list[dict], out: Path,
                 seconds: float, actuator: str, noise: bool,
                 per_row: int = 4) -> None:
    """A contact sheet of the replayed recoveries — one measured fall per
    resting pose, `per_row` frames across the episode.

    AGENTS.md verification discipline #2: look before you conclude. A table
    of `stood` fractions has been wrong here before, and the caption is the
    recipe's OWN `caption_fn`, so what the sheet says and what the reward
    sees cannot drift.
    """
    import mujoco

    from microduck_local.render_rollout import build_sheet, make_camera

    by_cls: dict[str, dict] = {}
    for r in falls:
        by_cls.setdefault(r["cls"], r)
    tiles, caps, hi = [], [], []
    for cls, row in sorted(by_cls.items()):
        env = BehaviorEnv("getup", seed=7000, max_episode_s=seconds + 1.0,
                          random_yaw=False, actuator=actuator, obs_noise=noise,
                          domain_rand=noise, action_delay=noise,
                          spawn_overrides={"MICRODUCK_SPAWN_FAMILY_PROBS": "0,0,0"})
        env.reset(seed=7000)
        _replay(env, row["qpos"])
        renderer = mujoco.Renderer(env.model, 480, 640)
        cam = make_camera("side", 0.9)
        steps = int(round(seconds / C.CTRL_DT))
        # Frames at t = 0 and then spread over the episode, so the first tile
        # is always the pose it actually started from — the check that the
        # replay put a FALLEN duck there and not a standing one.
        marks = [0] + [round((k + 1) * steps / (per_row - 1))
                       for k in range(per_row - 1)]
        obs = env._get_obs()
        for i in range(steps + 1):
            if i in marks:
                cam.lookat[:] = (float(env.data.xpos[env.trunk_body_id][0]),
                                 float(env.data.xpos[env.trunk_body_id][1]), 0.1)
                renderer.update_scene(env.data, camera=cam)
                tiles.append(renderer.render().copy())
                caps.append([f"{cls} ({row['mode']}) t={i * C.CTRL_DT:.2f}s",
                             env.behavior.caption_fn(env)])
                hi.append(_getup_hold_raw(env) > 0.0)
            if i < steps:
                obs, _, term, trunc, _ = env.step(policy_act(obs))
                if term or trunc:
                    break
        renderer.close()
        env.close()
    build_sheet(tiles, caps,
                [f"{name} — replayed REAL fall poses "
                 f"(actuator={actuator}, {'noisy' if noise else 'clean'})"],
                ["highlighted = the recipe's four-gate stand "
                 "(feet down, nothing else on the floor, upright, tall)"],
                hi, out)
    print(f"wrote {out}")


def _table(rows: list[dict], key: str) -> None:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    print(f"  {key:<10} {'n':>3} {'stood':>7} {'held 3s':>8} {'full':>6} "
          f"{'t_stand':>9} {'streak s':>9}")
    for name, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ts = [r["t_stand"] for r in g if r["reached"]]
        print(f"  {name:<10} {len(g):>3} "
              f"{sum(r['stood'] for r in g) / len(g):>7.0%} "
              f"{sum(r['held3'] for r in g) / len(g):>8.0%} "
              f"{sum(r['full'] for r in g) / len(g):>6.0%} "
              f"{(statistics.median(ts) if ts else float('nan')):>9.2f} "
              f"{statistics.median([r['streak_s'] for r in g]):>9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--falls", required=True,
                    help="the JSON written by scripts/probe_fall_poses.py --dump")
    ap.add_argument("--policy", required=True,
                    help="policy.onnx, or 'limp'/'zero' for the null control")
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="5 s to stand plus headroom to prove it holds")
    ap.add_argument("--actuator", default="bam", choices=("bam", "xml"))
    ap.add_argument("--noise", action="store_true",
                    help="observation noise + domain randomization + the "
                         "action delay — the conditions the robot meets")
    ap.add_argument("--modes", default=None,
                    help="only replay falls from these probe modes")
    ap.add_argument("--out", default=None, help="write the rows as JSON here")
    ap.add_argument("--sheet", default=None,
                    help="also render a contact sheet (one measured fall per "
                         "resting pose) to this PNG, and LOOK at it")
    args = ap.parse_args()

    falls = json.loads(Path(args.falls).read_text())
    # A duck that never fell is not a fall pose.
    falls = [r for r in falls if r["cls"] != "upright" and r["mode"] != "assumed"]
    if args.modes:
        keep = {m.strip() for m in args.modes.split(",")}
        falls = [r for r in falls if r["mode"] in keep]
    if not falls:
        raise SystemExit("no fallen poses in that dump")

    name, act = _driver(args.policy)
    rows = [run_fall(act, r, args.seconds, args.actuator, args.noise,
                     3000 + i)
            for i, r in enumerate(falls)]
    print(f"\n{name}   actuator={args.actuator}  "
          f"{'noisy' if args.noise else 'clean'}  "
          f"{len(rows)} measured falls x {args.seconds:.0f}s")
    print("\nby measured resting pose:")
    _table(rows, "cls")
    print("\nby how it fell:")
    _table(rows, "mode")
    print(f"\nALL  stood {sum(r['stood'] for r in rows) / len(rows):.0%}  "
          f"held3 {sum(r['held3'] for r in rows) / len(rows):.0%}  "
          f"full {sum(r['full'] for r in rows) / len(rows):.0%}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")
    if args.sheet:
        render_sheet(act, name, falls, Path(args.sheet), args.seconds,
                     args.actuator, args.noise)


if __name__ == "__main__":
    main()
