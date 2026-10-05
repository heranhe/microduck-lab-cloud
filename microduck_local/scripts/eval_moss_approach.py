"""Score a MOSS approach policy: does it arrive WITHOUT bulldozing the can?

    uv run python scripts/eval_moss_approach.py runs/<run>/policy.onnx [rung] [seeds]

Deterministic (the ONNX is the mean action), one episode per seed. Arriving
is not the only thing that matters and is not reported alone: a policy that
reaches the handover pose having shoved the can across the room has failed
the next leg before it starts, so the can's DISPLACEMENT is reported beside
the arrival rate, and so is how far the arm stuck out of the chassis while
driving — it spawns jammed 74 mm proud and is paid to fold itself in.
"""
import sys

import numpy as np
import onnxruntime as ort

#: How far the can has to be shoved before the approach counts as having
#: disturbed it. The can settles by microns on its own; a real nudge is
#: millimetres.
DISTURBED_M = 0.005

from microduck_local.robots.moss_env import MossApproachEnv


def main() -> None:
    path = sys.argv[1]
    rung = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    arrived = touched = 0
    shifts, shells, steps = [], [], []
    for seed in range(seeds):
        env = MossApproachEnv(seed=seed, approach_rung=rung)
        obs, _ = env.reset()
        start = np.array(env.data.xpos[env.can_body][:2], float)
        info, worst_shell, worst_shift = {}, 0.0, 0.0
        for _ in range(int(env.max_steps)):
            act = sess.run(
                None, {name: obs.reshape(1, -1).astype(np.float32)})[0][0]
            obs, _r, term, trunc, info = env.step(act)
            worst_shell = max(worst_shell, float(info.get("out_of_shell", 0.0)))
            # `disturbed` is a DISTANCE in metres, not a flag. Read as a bool
            # it is true for a nanometre of settling jitter, which reported
            # "touched the can 12/12" for a policy that moved it 0 mm.
            worst_shift = max(worst_shift, float(info.get("disturbed", 0.0)))
            if term or trunc:
                break
        arrived += bool(info.get("arrived"))
        touched += worst_shift > DISTURBED_M
        shifts.append(float(np.linalg.norm(
            np.array(env.data.xpos[env.can_body][:2], float) - start)))
        shells.append(worst_shell)
        steps.append(env.step_count)
    print(f"{path}  rung {rung}, {seeds} seeds")
    print(f"  arrived       {arrived}/{seeds}   ({100 * arrived / seeds:.0f}%)")
    print(f"  disturbed can {touched}/{seeds}   (moved it >5 mm: hands the pickup a rolling can)")
    print(f"  can moved     median {np.median(shifts) * 1000:.0f} mm   "
          f"max {np.max(shifts) * 1000:.0f} mm")
    print(f"  arm out of shell  median {np.median(shells) * 1000:.0f} mm   "
          f"(spawns jammed at 74 mm)")
    print(f"  steps median {int(np.median(steps))} of {env.max_steps}")


if __name__ == "__main__":
    main()
