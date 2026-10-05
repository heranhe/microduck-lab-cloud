"""Score a MOSS pickup policy: how often does it grip and lift the can?

    uv run python scripts/eval_moss_pick.py runs/<run>/policy.onnx [rung] [seeds]

Deterministic (the ONNX is the mean action), one episode per seed, and it
reports what happened rather than a reward: picked, gripped-but-not-lifted,
and the gap it closed. A reward number has lied about this task before.
"""
import sys

import numpy as np
import onnxruntime as ort

from microduck_local.robots.moss_env import (MossPickEnv, arm_camera_mount_of,
                                             obs_env_kwargs)


def main() -> None:
    path = sys.argv[1]
    rung = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    picked = held = 0
    gaps, steps = [], []
    for seed in range(seeds):
        env = MossPickEnv(seed=seed, pick_rung=rung,
                          arm_camera_mount=arm_camera_mount_of(path),
                          **obs_env_kwargs(path))
        obs, _ = env.reset()
        info = {}
        for _ in range(int(env.max_steps)):
            act = sess.run(None, {name: obs.reshape(1, -1).astype(np.float32)})[0][0]
            obs, _r, term, trunc, info = env.step(act)
            if term or trunc:
                break
        picked += bool(info.get("picked"))
        held += bool(info.get("held"))
        gaps.append(float(info.get("gap", 0.0)))
        steps.append(env.step_count)
    print(f"{path}  rung {rung}, {seeds} seeds")
    print(f"  picked   {picked}/{seeds}   ({100 * picked / seeds:.0f}%)")
    print(f"  gripped  {held}/{seeds}")
    print(f"  final gap median {np.median(gaps) * 1000:.0f} mm   "
          f"steps median {int(np.median(steps))} of {env.max_steps}")


if __name__ == "__main__":
    main()
