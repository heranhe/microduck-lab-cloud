"""Find a tuck the arm can actually REACH, not just one that looks folded.

    uv run python scripts/probe_moss_tuck.py

The first `moss.TUCK_POSE` was chosen by checking the POSE: inside the
rover's footprint, clear of the bin, no contacts. It was never checked that
the arm could travel there from where it starts. MEASURED: commanded from
HOME it jams on the hull and the track (5 contacts, `hull/palm`,
`track_1/palm`) and stalls 0.744 rad short permanently, leaving a link 74 mm
OUTSIDE the shell while the robot drives -- the opposite of what the tuck is
for. This searches with the servo in the loop, so reachability is the test.
"""
import numpy as np
import mujoco

from microduck_local.robots import moss
from microduck_local.robots.moss_drive import MossDriver

SHELL_X, SHELL_Y = (-0.145, 0.115), 0.141
SETTLE_S = 3.0


def main() -> None:
    m = moss.model()
    d = mujoco.MjData(m)
    drv = MossDriver(m, "")
    arm_geoms = [g for g in range(m.ngeom)
                 if int(m.geom_group[g]) == moss.COLLISION_GROUP
                 and int(m.geom_bodyid[g]) != m.body(moss.BASE_BODY).id
                 and int(m.body_rootid[m.geom_bodyid[g]])
                 == int(m.body_rootid[m.body(moss.BASE_BODY).id])]
    lo = np.array([m.joint(j).range[0] for j in moss.ARM_JOINTS])
    hi = np.array([m.joint(j).range[1] for j in moss.ARM_JOINTS])
    probe = mujoco.MjData(m)

    def footprint(q):
        """Worst overshoot of the shell at pose q, by forward kinematics."""
        probe.qpos[:] = m.key_qpos[0]
        for j, v in zip(moss.ARM_JOINTS, q):
            probe.qpos[m.joint(j).qposadr[0]] = v
        mujoco.mj_forward(m, probe)
        worst, top = 0.0, 0.0
        for g in arm_geoms:
            p = probe.geom_xpos[g]
            worst = max(worst, p[0] - SHELL_X[1], SHELL_X[0] - p[0],
                        abs(p[1]) - SHELL_Y)
            top = max(top, float(p[2]))
        return worst, top, int(probe.ncon)

    def reach(q):
        """Command q from HOME and return what the arm ACHIEVES."""
        drv.spawn(d)
        drv.set_arm(dict(zip(moss.ARM_JOINTS, q)))
        for _ in range(int(SETTLE_S / moss.GRASP_PHYSICS_DT)):
            drv.step(d)
            mujoco.mj_step(m, d)
        got = np.array([float(d.qpos[m.joint(j).qposadr[0]])
                        for j in moss.ARM_JOINTS])
        return got, float(np.abs(got - np.asarray(q)).max())

    rng = np.random.default_rng(0)
    cands = []
    for _ in range(40000):
        q = rng.uniform(lo, hi)
        out, top, ncon = footprint(q)
        if out <= 0.0 and ncon == 0 and top < moss.BIN_RIM_Z:
            cands.append((top, q))
    cands.sort(key=lambda t: t[0])
    print(f"{len(cands)} poses are inside the shell, clear and contact-free")
    print(f"{'top z':>7}{'joint err':>11}{'achieved out':>14}  verdict")
    best = None
    for top, q in cands[:40]:
        got, err = reach(q)
        out, _t, _n = footprint(got)
        ok = err < 0.05 and out <= 0.0
        print(f"{top:>7.3f}{err:>11.3f}{out * 1000:>12.0f}mm  "
              f"{'REACHABLE, inside' if ok else ''}")
        if ok and best is None:
            best = (q, got, top)
        if best is not None:
            break
    if best is None:
        print("\nno reachable tuck in this sample -- widen the search")
        return
    q, got, top = best
    print(f"\nTUCK_POSE = {tuple(np.round(q, 4).tolist())}")
    print(f"  achieved {np.round(got, 4).tolist()}, top z {top:.3f} m")


if __name__ == "__main__":
    main()
