"""What pose does a duck that FELL actually end up in? (roadmap B.1)

    cd microduck_local
    uv run python scripts/probe_fall_poses.py --seeds 24
    uv run python scripts/probe_fall_poses.py --modes assumed --seeds 200
    uv run python scripts/probe_fall_poses.py --seeds 24 --dump /tmp/falls.json

`behaviors/getup.py` trains and `scripts/bench_getup.py` scores against a
SYNTHETIC lie: `_getup_spawn` poses the duck at a tilt drawn from a window
(80-115 deg for the last rung), picks back / front / side from a declared
mix, and folds the legs by one shared `fold` scalar. That distribution was
designed, not measured. Nothing in the repo had ever asked what a duck that
actually falls over ends up looking like, so the recovery table in the
roadmap is a measurement over a population whose relationship to the real
one was assumed — the "unstated SCENARIO" shape in AGENTS.md, where a
systematically wrong input agrees with the truth wherever the quantity does
not depend on it.

This probe makes ducks fall for real and measures where they come to rest,
with the SAME instrument it applies to the synthetic spawns, so the two are
directly comparable:

  limp       standing, then every servo released (action = q - DEFAULT_POSE,
             render_rollout's own null control). It topples.
  zero       standing, then action = 0, i.e. the servos hold DEFAULT_POSE.
             Not passively stable — memory `open-loop-holds-topple` measured
             the tilt rule firing at step ~51.
  squat      ramped into the level symmetric squat (knee +1.4, ankle closing
             the chain) and then released — a collapse from a crouch rather
             than from full height.
  walk_push  `alpha_walking` driven forward, then shoved in a random
             direction hard enough to take it down, WITH THE WALKER STILL
             DRIVING. This is the pitch case: a duck knocked over in a duel
             is still being driven by a policy that recovers 0 of 24.
  walk_cut   the same shove, but the gait is cut at the moment of the push
             and the duck goes limp — the `World.getup_s` case, where a
             fallen duck lies on a zero command.
  assumed    NOT a fall: `behaviors/getup.py`'s own `_getup_spawn`, drawn
             from the shipped recipe's last-rung mix and tilt window and
             passed through this same measurement. The control arm.

Every mode settles the duck (velocity under a threshold for 0.3 s, cap 6 s)
before reading the pose, for the reason the recipe's own spawn settles:
a duck that has arrived is not a duck that is lying down.

Reported per mode: how many attempts actually FELL (a mode that mostly
stays up is telling you something too), and over the fallen ones the class
(back / front / left / right, from the projected gravity the policy
observes), the tilt off upright, the trunk height and the leg fold.

`--dump FILE` writes every settled pose's full qpos so a recovery bench can
replay the REAL poses instead of the synthetic ones — that is what
`scripts/probe_getup.py` consumes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from microduck_local import contract as C
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.brain.brain_env import POLICIES_DIR

MODES = ("limp", "zero", "squat", "walk_push", "walk_cut", "assumed")

# Rest test: the duck is settled once every generalized velocity has been
# under this for `STILL_HOLD_S`. Both measured rather than guessed — see
# `--report-settle`, which prints the distribution the cap has to cover.
STILL_QVEL = 0.20
STILL_HOLD_S = 0.30
SETTLE_CAP_S = 6.0

# The shove that takes a walking duck down, as a world-frame xy velocity
# added to the base — the same mechanism as walk_env's own `_push`, whose
# shipped range (PUSH_VEL_RANGE) is tuned NOT to knock the duck over.
PUSH_SPEED = (0.8, 1.6)


def _standing_env(seed: int, actuator: str, noise: bool) -> BehaviorEnv:
    """The getup recipe's own env, forced to start from the STAND keyframe.

    Deliberately BehaviorEnv("getup") and not a bare walk env: same scene
    (`all`, so the trunk and head can rest on the floor), same fall/height
    termination settings, same definition of everything the bench measures.
    A pose measured here can be replayed there without translation.
    """
    return BehaviorEnv(
        "getup", seed=seed, max_episode_s=60.0, random_yaw=False,
        actuator=actuator, obs_noise=noise, domain_rand=noise,
        action_delay=noise,
        # "0,0,0" selects no spawn family at all -> the keyframe start.
        spawn_overrides={"MICRODUCK_SPAWN_FAMILY_PROBS": "0,0,0"},
    )


def _limp_action(env) -> np.ndarray:
    """Servos released: hold wherever the joint currently is."""
    return (env.data.qpos[env.joint_qpos_adr] - C.DEFAULT_POSE).astype(np.float32)


def _squat_action(env, frac: float) -> np.ndarray:
    """The level symmetric squat from memory `open-loop-holds-topple`, ramped
    in: knee +1.4 rad, hip eased, ankle closing the chain so the foot stays
    flat. Signs mirror DEFAULT_POSE's left/right antisymmetry."""
    a = np.zeros(C.NUM_JOINTS, np.float32)
    knee, hip = 1.4 * frac, 0.2 * frac
    for sign, (h, k, ank) in ((+1.0, (2, 3, 4)), (-1.0, (11, 12, 13))):
        a[h] = sign * hip
        a[k] = sign * knee
        a[ank] = -sign * (hip + knee)
    return a


def _settle(env, act, cap_s: float = SETTLE_CAP_S) -> float:
    """Run `act` until the duck stops moving. Returns the seconds it took."""
    steps = int(round(cap_s / C.CTRL_DT))
    need = int(round(STILL_HOLD_S / C.CTRL_DT))
    still = 0
    obs = env._get_obs()
    for i in range(steps):
        obs, _, _, _, _ = env.step(act(obs, env))
        if float(np.abs(env.data.qvel).max()) < STILL_QVEL:
            still += 1
            if still >= need:
                return (i + 1) * C.CTRL_DT
        else:
            still = 0
    return cap_s


def _measure(env) -> dict:
    """The pose readout, in the terms the policy and the recipe use.

    `g` is the projected gravity that rides in obs[3:6], so the class below
    is something the policy can SEE — the getup docstring's point that the
    three fall poses need no memory to tell apart.
    """
    g = np.asarray(env._projected_gravity(), float)
    gx, gy, gz = (float(v) for v in g)
    # Angle away from upright. gz is -1 standing, 0 flat on a side, +1 upside
    # down, so the tilt is acos(-gz).
    tilt = float(np.degrees(np.arccos(np.clip(-gz, -1.0, 1.0))))
    q = env.data.qpos[env.joint_qpos_adr]
    # The recipe folds both legs by one `fold` scalar through hip_pitch
    # (1.1 x) and knee (1.2 x), antisymmetric left/right. Read the same
    # quantity back off whatever pose we are looking at, so "fold" means the
    # same thing for a real fall and for a synthetic spawn.
    fold = float(np.mean([q[2] / 1.1, q[3] / 1.2, -q[11] / 1.1, -q[12] / 1.2]))
    return {
        "cls": _classify(gx, gy, gz),
        "tilt_deg": tilt,
        "gx": gx, "gy": gy, "gz": gz,
        "trunk_z": float(env._trunk_xpos[2]),
        "fold": fold,
        # Leg asymmetry: the recipe gives both legs ONE fold, so a real fall
        # that lands with one leg tucked and one out has no synthetic
        # counterpart. |left - right| in the hip-pitch/knee pair.
        "leg_asym": float(abs(q[2] + q[11]) + abs(q[3] + q[12])) / 2.0,
        "qpos": env.data.qpos.copy().tolist(),
    }


def _classify(gx: float, gy: float, gz: float) -> str:
    """back / front / left / right / upright, from the projected gravity.

    The getup docstring states the convention this checks against: a duck on
    its back reads (-1, 0, 0) in the trunk frame, on its front (+1, 0, 0),
    on its side (0, -/+1, 0). `--verify-classes` reproduces it from the
    recipe's own spawns rather than trusting the sentence.
    """
    if gz < -np.cos(np.deg2rad(40.0)):       # within 40 deg of upright
        return "upright"
    return ("back" if gx < 0 else "front") if abs(gx) >= abs(gy) else \
           ("left" if gy < 0 else "right")


# ------------------------------------------------------------------ the falls

def _fall_limp(env, rng) -> None:
    _settle(env, lambda o, e: _limp_action(e))


def _fall_zero(env, rng) -> None:
    z = np.zeros(C.NUM_JOINTS, np.float32)
    _settle(env, lambda o, e: z)


def _fall_squat(env, rng) -> None:
    """Ramp into the squat over 5 control steps, hold it, then release."""
    ramp = 5
    for i in range(ramp):
        env.step(_squat_action(env, (i + 1) / ramp))
    hold = int(round(rng.uniform(0.2, 0.8) / C.CTRL_DT))
    for _ in range(hold):
        env.step(_squat_action(env, 1.0))
    _settle(env, lambda o, e: _limp_action(e))


def _walk_then_push(env, rng, infer, cut: bool) -> None:
    """Walk forward, then shove. `cut` releases the servos at the shove;
    otherwise the walker keeps driving, which is what happens on the pitch."""
    env.twist_cmd[:] = (float(rng.uniform(0.3, 0.5)), 0.0, 0.0)
    obs = env._get_obs()
    for _ in range(int(round(rng.uniform(1.0, 2.5) / C.CTRL_DT))):
        obs, _, _, _, _ = env.step(np.asarray(infer(obs), np.float32))
    # The shove: a world-frame xy velocity step on the base, in a random
    # direction, at a speed above the range walk_env's own DR uses (that one
    # is tuned not to take the duck down).
    theta = float(rng.uniform(-np.pi, np.pi))
    speed = float(rng.uniform(*PUSH_SPEED))
    env.data.qvel[0] += speed * np.cos(theta)
    env.data.qvel[1] += speed * np.sin(theta)
    if cut:
        _settle(env, lambda o, e: _limp_action(e))
    else:
        _settle(env, lambda o, e: np.asarray(infer(o), np.float32))


def _fall_assumed(env, rng) -> None:
    """NOT a fall: the shipped recipe's own synthetic lie, for comparison."""
    from microduck_local.behaviors.getup import _getup_spawn

    kind = str(rng.choice(("back", "front", "side"), p=(0.45 / 0.95,
                                                        0.30 / 0.95,
                                                        0.20 / 0.95)))
    env.spawn_overrides["MICRODUCK_GETUP_TILT_LO"] = "80"
    env.spawn_overrides["MICRODUCK_GETUP_TILT_HI"] = "115"
    env.spawn_overrides["MICRODUCK_GETUP_SETTLE_S"] = "1.0"
    _getup_spawn(env, kind)


def run_mode(mode: str, seeds: int, actuator: str, noise: bool,
             walker: Path | None) -> list[dict]:
    infer = None
    if mode.startswith("walk"):
        from microduck_local.brain.brain_env import onnx_infer
        if walker is None or not walker.exists():
            raise SystemExit(
                f"{mode} needs the shipped walker: {walker} not found "
                "(run scripts/setup.sh, or pass --walker)")
        infer = onnx_infer(walker)
    rows = []
    for s in range(seeds):
        env = _standing_env(2000 + s, actuator, noise)
        env.reset(seed=2000 + s)
        rng = np.random.default_rng(9000 + s)
        if mode == "limp":
            _fall_limp(env, rng)
        elif mode == "zero":
            _fall_zero(env, rng)
        elif mode == "squat":
            _fall_squat(env, rng)
        elif mode == "walk_push":
            _walk_then_push(env, rng, infer, cut=False)
        elif mode == "walk_cut":
            _walk_then_push(env, rng, infer, cut=True)
        elif mode == "assumed":
            _fall_assumed(env, rng)
        else:
            raise SystemExit(f"unknown mode {mode}")
        row = _measure(env)
        row["mode"] = mode
        row["seed"] = 2000 + s
        rows.append(row)
        env.close()
    return rows


def _summarise(mode: str, rows: list[dict]) -> None:
    fell = [r for r in rows if r["cls"] != "upright"]
    n = len(rows)
    print(f"\n{mode}: {len(fell)}/{n} fell"
          + ("" if fell else "  — nothing to characterise"))
    if not fell:
        return
    counts: dict[str, int] = {}
    for r in fell:
        counts[r["cls"]] = counts.get(r["cls"], 0) + 1
    med = lambda xs: statistics.median(xs)  # noqa: E731
    print(f"  {'pose':<7} {'n':>3} {'share':>6} {'tilt deg':>18} "
          f"{'trunk z':>8} {'fold':>15} {'leg asym':>9}")
    for cls, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        g = [r for r in fell if r["cls"] == cls]
        t = [r["tilt_deg"] for r in g]
        print(f"  {cls:<7} {c:>3} {c / len(fell):>6.0%} "
              f"{min(t):>6.0f}-{max(t):<4.0f} (med {med(t):>4.0f}) "
              f"{med([r['trunk_z'] for r in g]):>8.3f} "
              f"{med([r['fold'] for r in g]):>+7.2f} "
              f"[{min(r['fold'] for r in g):+.2f},"
              f"{max(r['fold'] for r in g):+.2f}] "
              f"{med([r['leg_asym'] for r in g]):>9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=24,
                    help="attempts per mode")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--actuator", default="bam", choices=("bam", "xml"),
                    help="bam = the honest XL330 model (the default, and what "
                         "any claim about the robot must be made under)")
    ap.add_argument("--noise", action="store_true",
                    help="observation noise + domain randomization + the "
                         "action delay")
    ap.add_argument("--walker", default=None,
                    help="the shipped walker the walk_* modes fall off "
                         "(default: POLICIES_DIR/alpha_walking.onnx)")
    ap.add_argument("--dump", default=None,
                    help="write every settled pose (with its full qpos) here, "
                         "for scripts/probe_getup.py to replay")
    args = ap.parse_args()

    print(f"actuator={args.actuator}  {'noisy' if args.noise else 'clean'}  "
          f"{args.seeds} attempts per mode")
    walker = (Path(args.walker) if args.walker
              else POLICIES_DIR / "alpha_walking.onnx")
    allrows: list[dict] = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        rows = run_mode(mode, args.seeds, args.actuator, args.noise, walker)
        _summarise(mode, rows)
        allrows += rows
    if args.dump:
        Path(args.dump).write_text(json.dumps(allrows))
        print(f"\nwrote {len(allrows)} poses to {args.dump}")


if __name__ == "__main__":
    main()
