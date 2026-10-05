"""Distil the PLANNED retract into the policy's weights.

    uv run python scripts/distill_retract.py <stow-run> <out-run> [episodes]

Training could not find the fold: 3M steps plateaued at 2.66 rad, more pay
REGRESSED everything (delivered 14->9), and the policy stops COMMANDING the
fold 1.5 rad short while nothing blocks it. Measured, the tuck pose and the
post-delivery pose are each collision-free but the bin wall sits between them,
so a straight line in any space jams — ten hand-built paths scored 0-4/24.

A one-waypoint planner solves the geometry (8/8 seeds, forward reach landing
exactly on the tuck value). This turns that planner into a TEACHER: it flies
the delivery with the trained policy and the fold with the planner, records
every (observation, action) pair, and behaviour-clones them into a fresh
policy. The planner is then scaffolding — the retract lives in the weights,
the deployed artifact is one ONNX, and RL can fine-tune past it afterwards.

That is the answer to "can the student keep what the teacher taught": yes, and
nothing is computed on the robot.
"""
import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from microduck_local.distill import fit
from microduck_local.robots import moss
import microduck_local.robots.moss_env as me
from microduck_local.robots.moss_env import (ARM_DELTA_RAD, MossStowEnv,
                                             eval_env_kwargs)

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "plan_retract", HERE / "plan_retract.py")
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

TUCK = np.asarray(moss.tuck_pose(), float)


def demo(policy: str, episodes: int, seed0: int = 0):
    """Roll the teacher out: policy delivers, planner folds. Record it all."""
    sess = ort.InferenceSession(policy, providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name
    kw = eval_env_kwargs(policy)
    kw.pop("base_lock", None)
    me.RETRACT_MAX_STEPS = 3000
    O, A = [], []
    folded = 0
    for ep in range(episodes):
        e = MossStowEnv(seed=seed0 + ep, retract=6.0, bin_scrape=0.25,
                        prop_variety=True, **kw)
        obs, _ = e.reset()
        n = 0
        # --- the DELIVERY, flown by the trained policy
        while n < 200 and not getattr(e, "_retracting", False):
            a = sess.run(None, {nm: obs.reshape(1, -1).astype(np.float32)})[0][0]
            O.append(obs.copy())
            A.append(np.asarray(a, np.float32).copy())
            obs, _r, t, tr, _i = e.step(a)
            n += 1
            if t or tr:
                break
        if not getattr(e, "_retracting", False):
            continue
        # --- the FOLD, flown by the planner
        m, d = e.model, e.data
        adrs = [m.joint(j).qposadr[0] for j in moss.ARM_JOINTS[:5]]
        lo = np.array([m.joint(j).range[0] for j in moss.ARM_JOINTS[:5]])
        hi = np.array([m.joint(j).range[1] for j in moss.ARM_JOINTS[:5]])
        # THE SAME ROUTE EVERY TIME. Planning per-episode gives a different
        # waypoint each run, and a teacher that answers differently to the
        # same observation cannot be cloned — measured, 250 episodes at MSE
        # 0.022 still folded 0/24. `UNIVERSAL_WAYPOINT` is the planner's
        # answer, fixed: it connects 14/14 real post-delivery poses to tuck.
        for tgt in [pr.UNIVERSAL_WAYPOINT, TUCK]:
            for _ in range(200):
                cur = np.asarray([d.qpos[a_] for a_ in adrs], float)
                if float(np.abs(cur - tgt).max()) < 0.05:
                    break
                a = np.zeros(e.action_space.shape[0], np.float32)
                a[moss.ACT_ARM] = np.clip((tgt - cur) / ARM_DELTA_RAD, -1, 1)
                O.append(obs.copy())
                A.append(a.copy())
                obs, _r, t, tr, _i = e.step(a)
                if t or tr:
                    break
        cur = np.asarray([d.qpos[a_] for a_ in adrs], float)
        folded += int(float(np.abs(cur - TUCK).max()) < 0.15)
    print(f"  {len(O)} demonstration steps from {episodes} episodes; "
          f"the teacher folded home on {folded}/{episodes}")
    return np.asarray(O, np.float32), np.asarray(A, np.float32)


def main() -> None:
    src = sys.argv[1]
    out = sys.argv[2]
    eps = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    O, A = demo(f"runs/{src}/policy.onnx", eps)
    mse = fit(O, A, Path("runs") / out, epochs=120, robot="moss", task="stow")
    print(f"  cloned into runs/{out}   action MSE {mse:.5f}")
    print("  (fidelity decides everything here: clone failure tracks action "
          "MSE at r=0.93 in this repo's own measurements)")


if __name__ == "__main__":
    main()
