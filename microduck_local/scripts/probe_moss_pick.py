"""The loop, with the gripper PRE-POSITIONED and the base driving the can in.

Why this order and not a top-down pick: his jaws hang below a palm that sits
~60 mm up the approach axis, and his can is 115 mm tall, so a descent onto a
standing can lands the palm on its lid and stops 35 mm short (measured). The
static sweep that held 7/9 had the can already between the pads. A tracked
litter-picker does the same thing: put the open jaws at can height, then
drive up to it."""
import mujoco
import numpy as np

from microduck_local.robots import moss
from microduck_local.robots.moss_drive import MossDriver

CAN_R, CAN_HALF, CAN_M = 0.033, 0.0575, 0.018
TUCK = [-0.403, 1.321, 1.047, 1.17, 0.773]
DROP = [-0.13, -0.03, 0.36]
DEPLOY_AT, GRASP_X = 0.55, 0.26

def build(cans):
    spec = moss.scene_spec(); k = spec.key(moss.HOME_KEY)
    qpos = list(np.asarray(k.qpos, float))
    for i, (x, y) in enumerate(cans):
        b = spec.worldbody.add_body(name=f"can{i}", pos=[x, y, CAN_HALF])
        b.add_freejoint(name=f"can{i}_free")
        b.add_geom(name=f"can{i}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[CAN_R, CAN_HALF, 0], mass=CAN_M, condim=6,
                   rgba=[.93,.45,.38,1], friction=[1.2, .02, .001])
        qpos += [x, y, CAN_HALF, 1, 0, 0, 0]
    k.qpos = qpos
    return spec.compile()

def run(can_xy, verbose=True):
    global m, d, drv, tcp, qadr, dofs, lo, hi
    m = build([can_xy]); d = mujoco.MjData(m); drv = MossDriver(m, "")
    tcp = m.site("tcp").id
    ARM = list(moss.ARM_JOINTS)
    qadr = [int(m.joint(j).qposadr[0]) for j in ARM]
    dofs = [int(m.joint(j).dofadr[0]) for j in ARM]
    lo = np.array([m.joint(j).range[0] for j in ARM]); hi = np.array([m.joint(j).range[1] for j in ARM])
    HOME_Q = [moss.ARM_HOME[j] for j in ARM]

    def ik(t, q0, iters=500, lam=0.18):
        p = mujoco.MjData(m); q = np.array(q0, float); jac = np.zeros((3, m.nv)); err = 9e9
        for _ in range(iters):
            p.qpos[:] = d.qpos; p.qpos[qadr] = q
            mujoco.mj_kinematics(m, p); mujoco.mj_comPos(m, p)
            e = np.asarray(t) - p.site_xpos[tcp]; err = float(np.linalg.norm(e))
            if err < 1e-4: break
            mujoco.mj_jacSite(m, p, jac, None, tcp); J = jac[:, dofs]
            q = np.clip(q + J.T @ np.linalg.solve(J@J.T + lam**2*np.eye(3), e), lo, hi)
        return q, err
    def step(vx=0., wz=0., n=1):
        for _ in range(n):
            drv.set_cmd(vx, wz, d.time); drv.step(d); mujoco.mj_step(m, d)
    def pose_of(p3):
        x, y, yaw = drv.pose(d); dx, dy = p3[0]-x, p3[1]-y
        c, s_ = np.cos(-yaw), np.sin(-yaw)
        return np.array([dx*c - dy*s_, dx*s_ + dy*c, p3[2]])
    def can_base(): return pose_of(d.xpos[m.body("can0").id])
    def to_world(pb):
        x, y, yaw = drv.pose(d); c, s_ = np.cos(yaw), np.sin(yaw)
        return np.array([x + pb[0]*c - pb[1]*s_, y + pb[0]*s_ + pb[1]*c, pb[2]])

    drv.spawn(d, 0, 0, 0)
    drv.set_arm(dict(zip(ARM, TUCK))); step(n=700)
    while can_base()[0] > DEPLOY_AT:
        step(0.30, 2.0 * float(np.arctan2(can_base()[1], can_base()[0])))
    step(n=300)
    # DEPLOY straight to the grasp pose: jaws open, at can height, out front.
    qg, _ = ik(to_world([GRASP_X, 0.0, moss.GRASP_HEIGHT_M]), HOME_Q)
    drv.set_arm({**dict(zip(ARM, qg)), "finger_left": 0.041, "finger_right": 0.041})
    step(n=1000)
    if verbose: print(f"  deployed, can {np.round(can_base(),3)}")
    # CREEP: drive the can into the open jaws.
    for _ in range(4000):
        cb = can_base()
        if cb[0] <= GRASP_X: break
        step(0.08, 1.2 * float(np.arctan2(cb[1], cb[0])))
    step(n=200)
    cb = can_base()
    if verbose: print(f"  can in the jaws at {np.round(cb,3)}")
    drv.set_arm({"finger_left": moss.GRASP_JAW_CTRL_M,
                 "finger_right": moss.GRASP_JAW_CTRL_M}); step(n=500)
    ql, _ = ik(to_world([cb[0], cb[1], 0.25]), qg); drv.set_arm(dict(zip(ARM, ql))); step(n=900)
    lifted = can_base()[2] > 0.12
    qd, _ = ik(to_world(DROP), ql); drv.set_arm(dict(zip(ARM, qd))); step(n=1700)
    drv.set_arm({"finger_left": 0.041, "finger_right": 0.041}); step(n=900)
    drv.set_arm(dict(zip(ARM, TUCK))); step(n=1200)
    cb = can_base()
    inbin = (moss.BIN_INTERIOR_X[0] < cb[0] < moss.BIN_INTERIOR_X[1]
             and moss.BIN_INTERIOR_Y[0] < cb[1] < moss.BIN_INTERIOR_Y[1]
             and cb[2] > moss.BIN_FLOOR_Z)
    if verbose: print(f"  lifted {'Y' if lifted else 'n'}  final {np.round(cb,3)}  "
                      f"IN BIN {'YES' if inbin else 'no'}")
    return lifted, inbin

ok = binned = 0
for xy in [(0.9, 0.0), (0.9, 0.15), (0.9, -0.15), (1.1, 0.0), (0.8, 0.08)]:
    print(f"can at {xy}:")
    l, b = run(xy)
    ok += l; binned += b
print(f"\nlifted {ok}/5   stowed in its own bin {binned}/5")
