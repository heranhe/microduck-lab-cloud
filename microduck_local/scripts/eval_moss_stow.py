"""Score a MOSS stow policy: does the can end up IN the bin?

    uv run python scripts/eval_moss_stow.py runs/<run>/policy.onnx [rung] [seeds]

Deterministic (the ONNX is the mean action), one episode per seed. Holding
the can is not the goal and is not reported alone: a policy that carries it
for the whole episode and never lets go scores a flat reward that looks
calm, so this reports where the can ACTUALLY ended — in the bin, still in
the jaws at the end, or dropped on the floor.
"""
import sys

import numpy as np
import onnxruntime as ort

from microduck_local.robots.moss_env import MossStowEnv


def main() -> None:
    path = sys.argv[1]
    rung = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    stowed = dropped = still_held = 0
    gaps, steps = [], []
    for seed in range(seeds):
        env = MossStowEnv(seed=seed, stow_rung=rung)
        obs, _ = env.reset()
        info = {}
        for _ in range(int(env.max_steps)):
            act = sess.run(
                None, {name: obs.reshape(1, -1).astype(np.float32)})[0][0]
            obs, _r, term, trunc, info = env.step(act)
            if term or trunc:
                break
        stowed += bool(info.get("stowed"))
        dropped += bool(info.get("dropped"))
        still_held += bool(info.get("carrying"))
        gaps.append(float(info.get("stow_gap", 0.0)))
        steps.append(env.step_count)
    print(f"{path}  rung {rung}, {seeds} seeds")
    print(f"  IN THE BIN    {stowed}/{seeds}   ({100 * stowed / seeds:.0f}%)")
    print(f"  dropped       {dropped}/{seeds}")
    print(f"  still holding {still_held}/{seeds}   (never let go)")
    print(f"  can-to-bin gap at the end: median "
          f"{np.median(gaps) * 1000:.0f} mm   (starts at ~490 mm)")
    print(f"  steps median {int(np.median(steps))} of {env.max_steps}")


if __name__ == "__main__":
    main()
