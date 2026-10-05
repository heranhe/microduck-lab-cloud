"""Drill rung, done properly: IK the jaws onto a STANDING can, close, lift.

The jaw geometry, measured: pad CENTRES are 16 mm apart shut and 98 mm at the
41 mm slide limit, so centre = 16 + 2*ctrl (mm). The pads are 8 mm boxes, so
the inner faces are 8 mm closer than that. A 66 mm can wants a few mm of
interference: inner 60-64 mm -> ctrl 26-28 mm. 20 mm (the first try) was a
12 mm over-squeeze, which ejects rather than grips.
"""
import mujoco
import numpy as np

from microduck_local.robots import moss
from microduck_local.robots.moss_drive import MossDriver

CAN_R, CAN_HALF, CAN_M = 0.033, 0.0575, 0.018

def build():
    spec = moss.scene_spec()
    can = spec.worldbody.add_body(name="can", pos=[0.25, 0, CAN_HALF])
    can.add_freejoint(name="can_free")
    can.add_geom(name="can_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[CAN_R, CAN_HALF, 0], mass=CAN_M,
                 rgba=[0.93, 0.45, 0.38, 1], condim=6, friction=[1.2, 0.02, 0.001])
    k = spec.key(moss.HOME_KEY)
    k.qpos = list(np.asarray(k.qpos, float)) + [0.25, 0, CAN_HALF, 1, 0, 0, 0]
    return spec.compile()

m = build(); d = mujoco.MjData(m); drv = MossDriver(m, "")
tcp = m.site("tcp").id
ARM = list(moss.ARM_JOINTS)
qadr = [int(m.joint(j).qposadr[0]) for j in ARM]
dofs = [int(m.joint(j).dofadr[0]) for j in ARM]
lo = np.array([m.joint(j).range[0] for j in ARM])
hi = np.array([m.joint(j).range[1] for j in ARM])

def ik(target, q0, iters=400, lam=0.25):
    """Damped least squares onto the tcp, joint limits clamped."""
    probe = mujoco.MjData(m)
    q = np.array(q0, float)
    jac = np.zeros((3, m.nv))
    for _ in range(iters):
        probe.qpos[:] = d.qpos
        probe.qpos[qadr] = q
        mujoco.mj_kinematics(m, probe); mujoco.mj_comPos(m, probe)
        err = np.asarray(target) - probe.site_xpos[tcp]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(m, probe, jac, None, tcp)
        J = jac[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)
        q = np.clip(q + dq, lo, hi)
    return q, float(np.linalg.norm(err))

print(f"{'spot (x,y)':>14} {'ctrl':>6} {'IK err':>8} {'lift':>7} {'held':>5}")
SPOTS = [(x, y) for x in (0.22, 0.25, 0.28) for y in (-0.06, 0.0, 0.06)]
held_n = 0
for cx, cy in SPOTS:
    for ctrl in (0.027,):
        gz = 0.050
        drv.spawn(d)
        d.qpos[m.joint("can_free").qposadr[0]:][:3] = [cx, cy, CAN_HALF]
        q, err = ik([cx, cy, gz], [m.key_qpos[0][a] for a in qadr])
        drv.set_arm({**dict(zip(ARM, q)), "finger_left": 0.041, "finger_right": 0.041})
        for _ in range(900): drv.step(d); mujoco.mj_step(m, d)
        z0 = float(d.xpos[m.body("can").id][2])
        drv.set_arm({"finger_left": ctrl, "finger_right": ctrl})
        for _ in range(500): drv.step(d); mujoco.mj_step(m, d)
        drv.set_arm(dict(zip(ARM, [moss.ARM_HOME[j] for j in ARM])))
        for _ in range(1500): drv.step(d); mujoco.mj_step(m, d)
        zf = float(d.xpos[m.body("can").id][2])
        ok = zf > z0 + 0.03
        held_n += ok
        print(f"{str((cx,cy)):>14} {ctrl:>6.3f} {err:>8.4f} {zf - z0:>+7.3f} "
              f"{'YES' if ok else 'no':>5}")
print(f"held {held_n}/{len(SPOTS)}")
