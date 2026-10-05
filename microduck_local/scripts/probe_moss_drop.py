"""Find a release pose the arm can REACH from the lift, carrying a can.

    uv run python scripts/probe_moss_drop.py

`moss.DROP_POSE` was chosen as a point over the bin that is reachable and
contact-free — checked as a POSE. Approached from `LIFT_POSE` with a can in
the jaws it is none of those things: `shoulder_lift` stalls 0.379 rad short
against `bin_x1`, the bin's own front wall, and the can is let go 8.5 cm high
and 7 cm forward of where the pose says, bounces off the near rim and ends on
the floor. 0 of 10 deliveries. Same mistake as the tuck, which also jams: a
pose validated on its own rather than along the path that has to reach it.

So the test here is the whole delivery — ramp from the lift, open, settle,
and ask whether the can is IN THE BIN.
"""
import numpy as np
import mujoco

from microduck_local.robots import moss
from microduck_local.robots.moss_env import MossStowEnv

RAMP_S = 3.0
SETTLE_S = 1.5


def main() -> None:
    env = MossStowEnv(stow_rung=0)
    lo = np.array([env.model.joint(j).range[0] for j in moss.ARM_JOINTS])
    hi = np.array([env.model.joint(j).range[1] for j in moss.ARM_JOINTS])

    def deliver(pose, seed):
        env.reset(seed=seed)
        if not env._gripped_now():
            return None
        start = np.array([env.arm_cmd[j] for j in moss.ARM_JOINTS], float)
        goal = np.asarray(pose, float)
        n = int(RAMP_S / moss.GRASP_PHYSICS_DT)
        for k in range(n):
            f = (k + 1) / n
            env.arm_cmd.update(
                zip(moss.ARM_JOINTS, start + f * (goal - start)))
            env.driver.set_arm(env.arm_cmd)
            env.driver.step(env.data)
            mujoco.mj_step(env.model, env.data)
        held = env._gripped_now()
        err = max(abs(float(env.data.qpos[env.model.joint(j).qposadr[0]]) - v)
                  for j, v in zip(moss.ARM_JOINTS, pose))
        env.arm_cmd[moss.GRIPPER_JOINT] = 0.041
        env.driver.set_arm(env.arm_cmd)
        for _ in range(int(SETTLE_S / moss.GRASP_PHYSICS_DT)):
            env.driver.step(env.data)
            mujoco.mj_step(env.model, env.data)
        return env._in_bin(), held, err

    rng = np.random.default_rng(0)
    base = np.asarray(moss.DROP_POSE, float)
    print(f"{'candidate':>10}{'reach err':>11}{'held':>7}{'in bin':>8}")
    best = []
    for i in range(60):
        # Search AROUND the documented drop, widening as it goes.
        scale = 0.15 + 0.45 * (i / 60)
        pose = np.clip(base + rng.normal(0.0, scale, 5), lo, hi)
        got = deliver(pose, 0)
        if got is None:
            continue
        in_bin, held, err = got
        if in_bin or (held and err < 0.08):
            print(f"{i:>10}{err:>11.3f}{str(held):>7}{str(in_bin):>8}")
        if in_bin:
            best.append((err, pose))
    if not best:
        print("\nno candidate delivered the can -- widen or rethink")
        return
    best.sort(key=lambda t: t[0])
    err, pose = best[0]
    print(f"\nbest: {tuple(np.round(pose, 4).tolist())}  (reach err {err:.3f})")
    ok = sum(1 for s in range(8) if (deliver(pose, s) or (False,))[0])
    print(f"  repeat over 8 seeds: {ok}/8 landed in the bin")


if __name__ == "__main__":
    main()
