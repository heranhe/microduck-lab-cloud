"""`repair_warmstart_norm`: measure, and optionally repair, a warm-start
donor's observation statistics on slots the donor never really saw.

    # just LOOK (what you want before deciding anything):
    uv run python scripts/repair_warmstart_norm.py --donor runs/.distill/<hash> \
        --behavior dribble --slots 51:58 --policy runs/x/policy.onnx --dry-run

    # ...and write a repaired copy:
    uv run python scripts/repair_warmstart_norm.py --donor runs/.distill/<hash> \
        --out runs/walker-ballslots --behavior dribble --slots 51:58 \
        --policy runs/x/policy.onnx

**The failure this exists for.** A recipe puts its task in the command slots
(`find_ball`, `lastmetre` and `dribble` all do; AGENTS.md: "a task the robot
must SENSE puts its sensing in the command slots"). A donor trained on a
DIFFERENT task only ever saw keep-alive noise there, so its `VecNormalize`
carries a variance of ~1e-4 on those slots — and a real value then enters the
network at an absurd z. Measured on the duck, 2026-09-24:

  * obs[51:55] (the ball): donor std 0.009-0.042, a real rollout presents
    **max |z| 115**, median 12.
  * obs[55:58] (a target vector): donor std 0.0029 against written values of
    +-1.0 — **200x the range those slots ever carried**. The effect is not
    subtle: the shipped walker went from 400/400 steps to **30**, and blanking
    exactly those three slots before inference restored exactly 400/400.

`lastmetre` hit this and chose to train from scratch rather than face it
("a bearing of 1.0 would enter the network at ~34 and the running statistics
would take another 2M steps to notice").

**What the statistics are worth, and why overwriting them discards nothing.**
The donor's numbers on those slots describe a task with no ball and no target.
They are an artefact of keep-alive noise, not a measurement of anything the
new task will present. The replacements here are sampled from the NEW task's
own rollouts, which is what the running statistics would converge to anyway —
this just skips the convergence.

**IMPORTANT, and the opposite of what is often assumed:** `train_behavior`
does NOT freeze the normalizer on `--init-from`. Nothing sets
`venv.training = False`, and a loaded `VecNormalize` comes back `training=True`
(check it yourself: this script prints it). So an unrepaired warm start is not
permanently wrong — it is wrong for as long as the running mean and variance
take to move, which for a variance of 1e-4 is long enough to wreck the policy
first. Repairing is a way to skip a transient, not to fix a freeze.

Every non-listed slot is left byte-identical, and the script prints an
untouched control slot so you can see that.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _slots(spec: str) -> tuple[int, int]:
    a, b = spec.split(":")
    return int(a), int(b)


def sample(behavior: str, policy: str | None, lo: int, hi: int, episodes: int,
           steps: int, blank: bool, seed0: int) -> np.ndarray:
    """Roll the NEW task and collect what its command slots actually carry."""
    import microduck_local.behaviors  # noqa: F401  (registers the recipes)
    from microduck_local.behaviors.env import BehaviorEnv

    env = BehaviorEnv(behavior, seed=3, max_episode_s=steps * 0.02 + 0.5,
                      domain_rand=False, random_yaw=False, obs_noise=False,
                      action_delay=False)
    infer = None
    if policy:
        from microduck_local.brain.brain_env import onnx_infer
        infer = onnx_infer(policy)
    rng = np.random.default_rng(0)
    out = []
    for ep in range(episodes):
        res = env.reset(seed=seed0 + ep)
        obs = res[0] if isinstance(res, tuple) else res
        for _ in range(steps):
            o = np.asarray(obs, np.float32).copy()
            if blank:
                # Keep a donor-derived policy ALIVE while sampling: the very
                # slots being measured are what break it, so zero them for the
                # inference call only. The recorded sample is the real obs.
                o[lo:hi] = 0.0
            act = (infer(o.reshape(-1)) if infer is not None
                   else rng.uniform(-0.2, 0.2, env.action_space.shape[0]).astype(np.float32))
            step = env.step(act)
            obs = step[0]
            out.append(np.asarray(obs, float)[lo:hi])
            if step[2] or step[3]:
                break
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--donor", required=True, help="run dir with model.zip + vecnormalize.pkl")
    ap.add_argument("--out", help="where to write the repaired copy (omit with --dry-run)")
    ap.add_argument("--behavior", required=True, help="the NEW task's recipe id")
    ap.add_argument("--slots", required=True, help="obs slice to repair, e.g. 51:58")
    ap.add_argument("--policy", help="ONNX rolled to generate rollouts; random actions if omitted")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--no-blank", action="store_true",
                    help="do NOT zero the slots for the inference call (see `sample`)")
    ap.add_argument("--dry-run", action="store_true", help="report the z-scores, write nothing")
    a = ap.parse_args()

    lo, hi = _slots(a.slots)
    donor = Path(a.donor)
    for f in ("model.zip", "vecnormalize.pkl"):
        if not (donor / f).exists():
            sys.exit(f"{donor} has no {f} — --init-from needs both")

    with open(donor / "vecnormalize.pkl", "rb") as fh:
        vn = pickle.load(fh)
    print(f"donor {donor}")
    print(f"  normalizer .training = {vn.__dict__.get('training')}  "
          f"(True: the statistics keep updating during the warm start)\n")

    S = sample(a.behavior, a.policy, lo, hi, a.episodes, a.steps, not a.no_blank, a.seed0)
    if len(S) == 0:
        sys.exit("no samples — every episode ended immediately")
    mean, var = vn.obs_rms.mean, vn.obs_rms.var
    Z = np.abs((S - mean[lo:hi]) / np.sqrt(var[lo:hi] + 1e-8))
    print(f"{len(S)} samples of obs[{lo}:{hi}] from `{a.behavior}`\n")
    print(f"{'slot':>6} {'donor std':>10} {'task std':>10} {'ratio':>8} {'max |z| now':>12}")
    for k, i in enumerate(range(lo, hi)):
        ds = float(np.sqrt(var[i]))
        ts = float(S[:, k].std())
        print(f"{i:>6} {ds:10.5f} {ts:10.4f} {ts / max(ds, 1e-9):8.0f}x {Z[:, k].max():12.1f}")
    print(f"\n  overall max |z| {Z.max():.1f}, median {np.median(Z):.2f}")
    if Z.max() < 5:
        print("  -> already in distribution; repairing would buy little.")
    else:
        print("  -> out of distribution. An unrepaired warm start eats this as a")
        print("     transient while the running statistics catch up.")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return
    if not a.out:
        sys.exit("give --out, or use --dry-run")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("model.zip", "vecnormalize.pkl"):
        shutil.copyfile(donor / f, out / f)
    with open(out / "vecnormalize.pkl", "rb") as fh:
        vn = pickle.load(fh)
    # SB3's VecNormalize.__getstate__ deletes exactly these three, and its
    # __getattr__ RECURSES on a missing one — so touch __dict__ directly and
    # never hasattr(). Both were live bugs on the way to this script.
    for k, v in (("venv", None), ("class_attributes", {}), ("returns", np.zeros(1))):
        vn.__dict__.setdefault(k, v)
    ctrl = 0 if lo != 0 else hi
    before_ctrl = float(np.sqrt(vn.obs_rms.var[ctrl]))
    for k, i in enumerate(range(lo, hi)):
        vn.obs_rms.mean[i] = float(S[:, k].mean())
        vn.obs_rms.var[i] = float(max(S[:, k].var(), 1e-4))
    tmp = out / "vecnormalize.pkl.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(vn, fh)
    os.replace(tmp, out / "vecnormalize.pkl")

    with open(out / "vecnormalize.pkl", "rb") as fh:
        chk = pickle.load(fh)
    Z2 = np.abs((S - chk.obs_rms.mean[lo:hi]) / np.sqrt(chk.obs_rms.var[lo:hi] + 1e-8))
    print(f"\nwrote {out}")
    print(f"  |z| {Z.max():.1f} -> {Z2.max():.1f} (max), {np.median(Z):.2f} -> {np.median(Z2):.2f} (median)")
    print(f"  untouched control slot[{ctrl}] std {before_ctrl:.4f} -> "
          f"{float(np.sqrt(chk.obs_rms.var[ctrl])):.4f}")
    print(f"\n  now: --init-from {out}")


if __name__ == "__main__":
    main()
