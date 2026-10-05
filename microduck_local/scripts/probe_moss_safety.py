"""Would a real MOSS survive a tidy run? Measure the SERVOS, not the score.

    uv run python scripts/probe_moss_safety.py 300 0,1,2,3

`eval_tidy_moss.py` answers "how many props reached the bin". This answers the
question that decides whether the run may be repeated on hardware: **what did
the arm have to do to get them there.** Nothing in this repo has ever driven a
MOSS, so the reference is not a datasheet we have measured against — it is the
envelope Laurent's own MJCF declares, which is the only honest one available:

    five arm servos   position, kp 70, kd 3, forcerange +-2.2 N m
    one finger servo  position, kp 700, kd 8, forcerange +-8 N
    joint ranges      +-1.92 / +-1.75 / +-1.69 / +-1.66 / -2.74..+2.84 rad

A position servo pinned at its `forcerange` is a STALLED servo. On the
STS3215-class hardware this arm derives from that is the failure mode that
actually breaks things — not a fast sweep, not a hard knock, but a joint
holding full torque against something for tens of seconds while its winding
heats and its nylon gear train creeps. So the headline numbers here are
DURATIONS, not peaks: the longest unbroken stretch each joint spent at its
clamp while not moving, and its torque duty cycle over the whole run.

Sampled every PHYSICS step (2 ms), not every control tick, by wrapping
`mujoco.mj_step`: a contact that lasts three substeps is invisible at 50 Hz,
and `world/arena.py` already learned that lesson for bump sensing.

Six families, each with the reason it is a hazard:

`stall`      torque at >=90% of the clamp with the joint moving <0.05 rad/s.
             The winding is at full current and none of it is becoming motion.
             Reported as the LONGEST unbroken episode per joint.
`duty`       mean (torque/clamp)^2 over the run — proportional to I^2R heating
             for a current-proportional servo. A continuous-duty hobby serial
             servo wants this well under 0.3.
`limit`      time a joint spent within 2 deg of its hard stop. The sim clamps
             `qpos` for free; the hardware has a printed end stop to break.
`clip`       the VISIBLE arm's clearance from the rover's own hull and bin
             (`brain/moss_motion.ArmClearance`, 10 Hz). Negative means the arm
             you see on /sim is inside the robot — and the collision proxies
             are thinner than the meshes, so the physics does not object.
             `clearance()`'s two SENTINELS are skipped, not counted: -1.0 (a
             proxy contact) and -0.2 (jaws down inside the bin) are states the
             mission is SUPPOSED to reach — the release puts the jaws in the
             bin — and they are flags, not depths, so averaging them in would
             report the robot doing its job as 200 mm of penetration.
`contact`    real contacts between an arm geom and the rover/bin, with their
             normal force and penetration depth. A pry bar moment on a 2.2 N m
             wrist is how a printed link snaps.
`jerk`       the biggest one-tick command step per joint. At 50 Hz a 0.3 rad
             step is 15 rad/s of demanded speed, which a servo answers with
             its whole stall torque.

Prints one block per seed and a pooled verdict. Every threshold is a module
constant with its reason above it; none of them is a pass mark this repo has
earned on hardware, and the report says so.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

from microduck_local.brain import moss_motion
from microduck_local.robots import moss
from microduck_local.viz_server import load_policy_infer
from microduck_local.world import scenario as S
from microduck_local.world_server import WorldState

#: Fraction of `forcerange` at which a position servo is AT its clamp. Not 1.0:
#: MuJoCo's solver lands a saturated actuator a hair under the range, and the
#: hardware's current limiter is not a step function either.
SAT_FRAC = 0.90
#: Below this a joint is not doing work (rad/s for a hinge, m/s for the slide).
#: 0.05 rad/s is 3 deg/s — an arm creeping this slowly under full torque is
#: pushing on something that is not going to move.
STALL_SPEED = 0.05
#: Within this of a hard stop. 2 deg for a HINGE — one servo step is ~0.09 deg,
#: so this is 20 steps of slack and not a rounding band — and 1 mm for the
#: finger SLIDE, whose whole travel is 41 mm. The first cut used the radian
#: number for both and reported the jaw against its stop 100% of every run:
#: 0.035 of a 0.041 m slide is 85% of the travel, so the band was the joint.
LIMIT_MARGIN = 0.035
LIMIT_MARGIN_SLIDE = 0.001
#: Report a stall episode at all once it lasts this long, s. A tenth of a
#: second of clamp is a normal contact transient; a second is a decision.
STALL_REPORT_S = 0.5
#: Base speed above which a pad resting on the floor is a pad being DRAGGED
#: across it — `robots/moss_env.ARM_FLOOR_DRIVE_MPS`, the same number, so this
#: report and that reward term count the same event.
FLOOR_DRIVE_MPS = 0.1
#: Clearance samples per second. The visual-mesh sweep is 5 meshes x 7 hull
#: boxes of `mj_geomDistance` and costs ~1.5 ms; at 10 Hz it adds ~1% to a run.
CLIP_HZ = 10.0


def _override(brain) -> None:
    """`MICRODUCK_MOSS_SAFETY_PARAMS="arm_rate_cap=0"` — the A/B knob.

    Through `dataclasses.replace` and then READ BACK: assigning an attribute on
    a frozen dataclass raises, but assigning one the class does not declare
    does not, and this repo has already run a two-arm battery where both arms
    silently held the same config and reported a null.
    """
    spec = os.environ.get("MICRODUCK_MOSS_SAFETY_PARAMS", "").strip()
    if not spec:
        return
    want = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if not hasattr(brain.p, k):
            raise SystemExit(f"tidy_moss has no param {k!r}")
        cur = getattr(brain.p, k)
        want[k] = type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true")
    brain.p = dataclasses.replace(brain.p, **want)
    for k, v in want.items():
        got = getattr(brain.p, k)
        if got != v:
            raise SystemExit(f"override of {k} did not take: {got!r} != {v!r}")
    print(f"  params: {want}")


class Watch:
    """Streaming per-joint extremes over a run. Nothing is kept per sample."""

    def __init__(self, names, clamp, lo, hi, dt, margin):
        self.names = list(names)
        self.clamp = np.asarray(clamp, float)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.dt = float(dt)
        self.margin = np.asarray(margin, float)
        n = len(self.names)
        self.peak = np.zeros(n)          # worst |torque|
        self.speed = np.zeros(n)         # worst |velocity|
        self.sat_s = np.zeros(n)         # seconds at the clamp
        self.stall_s = np.zeros(n)       # ...of which, not moving
        self.run_s = np.zeros(n)         # current unbroken stall
        self.worst_run = np.zeros(n)     # longest unbroken stall
        self.limit_s = np.zeros(n)       # seconds against a hard stop
        self.duty = np.zeros(n)          # sum (torque/clamp)^2 dt
        self.step = np.zeros(n)          # worst one-tick command jump
        #: The brain state each extreme happened in — a number you can go and
        #: watch. Without it "peak 2.2 N m" names no code to look at.
        self.peak_at = [""] * n
        self.speed_at = [""] * n
        self.stall_at = [""] * n
        self.step_at = [""] * n
        self.state = ""
        self.n = 0

    def sample(self, tau, qvel, qpos):
        f = np.abs(tau) / self.clamp
        for i in np.flatnonzero(np.abs(tau) > self.peak):
            self.peak_at[i] = self.state
        for i in np.flatnonzero(np.abs(qvel) > self.speed):
            self.speed_at[i] = self.state
        self.peak = np.maximum(self.peak, np.abs(tau))
        self.speed = np.maximum(self.speed, np.abs(qvel))
        self.duty += f * f * self.dt
        sat = f >= SAT_FRAC
        self.sat_s += sat * self.dt
        stalled = sat & (np.abs(qvel) < STALL_SPEED)
        self.stall_s += stalled * self.dt
        self.run_s = np.where(stalled, self.run_s + self.dt, 0.0)
        for i in np.flatnonzero(self.run_s > self.worst_run):
            self.stall_at[i] = self.state
        self.worst_run = np.maximum(self.worst_run, self.run_s)
        self.limit_s += ((qpos - self.lo < self.margin)
                         | (self.hi - qpos < self.margin)) * self.dt
        self.n += 1

    def rows(self, total_s):
        out = []
        for i, n in enumerate(self.names):
            out.append(dict(
                joint=n, clamp=float(self.clamp[i]),
                peak=float(self.peak[i]), peak_frac=float(self.peak[i] / self.clamp[i]),
                max_speed=float(self.speed[i]),
                sat_s=float(self.sat_s[i]), stall_s=float(self.stall_s[i]),
                worst_stall_s=float(self.worst_run[i]),
                duty=float(self.duty[i] / max(total_s, 1e-9)),
                limit_s=float(self.limit_s[i]),
                max_cmd_step=float(self.step[i]),
                peak_at=self.peak_at[i], speed_at=self.speed_at[i],
                stall_at=self.stall_at[i], step_at=self.step_at[i]))
        return out


def run_one(sc0, seed: int, seconds: float) -> dict:
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc0, seed=seed), sc0
    st.set_brain("m0", "tidy_moss")
    _override(st.brains["m0"])
    w = st.world
    m, d = w.model, w.data
    pre = "m0/"
    jn = list(moss.ARM_JOINTS) + [moss.GRIPPER_JOINT]
    act = [m.actuator(pre + j).id for j in jn]
    jid = [m.joint(pre + j).id for j in jn]
    qadr = [m.jnt_qposadr[i] for i in jid]
    vadr = [m.jnt_dofadr[i] for i in jid]
    clamp = np.array([abs(m.actuator_forcerange[a][1]) for a in act])
    lo = np.array([m.jnt_range[i][0] for i in jid])
    hi = np.array([m.jnt_range[i][1] for i in jid])
    dt = float(m.opt.timestep)
    margin = [LIMIT_MARGIN_SLIDE if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE
              else LIMIT_MARGIN for i in jid]
    watch = Watch(jn, clamp, lo, hi, dt, margin)

    # Which geoms belong to the arm, and which to the rover it must not hit.
    arm_bodies = ("upper_arm", "lower_arm", "wrist", "gripper", "finger", "pad")
    def bname(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
    arm_g = {g for g in range(m.ngeom)
             if bname(g).startswith(pre) and any(k in bname(g) for k in arm_bodies)}
    hull_g = set()
    for n in moss.HULL_GEOMS:
        try:
            hull_g.add(m.geom(pre + n).id)
        except KeyError:
            pass
    hits = {"n": 0, "force": 0.0, "depth": 0.0, "at": "", "pairs": {}}
    floor_g = {g for g in range(m.ngeom)
               if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}
    base_v = [m.jnt_dofadr[m.joint(pre + j).id] for j in moss.BASE_JOINTS[:2]]
    floor = {"n": 0, "drive": 0, "force": 0.0, "geoms": {}}
    six = np.zeros(6)

    # Every PHYSICS step, not every tick: `world/arena.py` calls this inside
    # its substep loop, and a three-substep contact is invisible at 50 Hz.
    real_step = mujoco.mj_step
    def watched_step(mm, dd, nstep=1):
        real_step(mm, dd, nstep)
        if mm is not m:
            return
        watch.sample(dd.actuator_force[act], dd.qvel[vadr], dd.qpos[qadr])
        for c in range(dd.ncon):
            con = dd.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in arm_g and g2 in floor_g) or (g2 in arm_g and g1 in floor_g):
                mujoco.mj_contactForce(mm, dd, c, six)
                floor["n"] += 1
                floor["force"] = max(floor["force"], float(np.linalg.norm(six[:3])))
                if float(np.linalg.norm(dd.qvel[base_v])) > FLOOR_DRIVE_MPS:
                    floor["drive"] += 1
                gn = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM,
                                       g1 if g1 in arm_g else g2)
                floor["geoms"][gn] = floor["geoms"].get(gn, 0) + 1
            if (g1 in arm_g and g2 in hull_g) or (g2 in arm_g and g1 in hull_g):
                mujoco.mj_contactForce(mm, dd, c, six)
                f = float(np.linalg.norm(six[:3]))
                hits["n"] += 1
                if f > hits["force"]:
                    hits["at"] = watch.state
                hits["force"] = max(hits["force"], f)
                hits["depth"] = min(hits["depth"], float(con.dist))
                key = f"{mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, g1)} / " \
                      f"{mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, g2)}"
                p = hits["pairs"].setdefault(key, [0, 0.0, 0.0])
                p[0] += 1
                p[1] = max(p[1], f)
                p[2] = min(p[2], float(con.dist))
    mujoco.mj_step = watched_step

    ac = moss_motion.ArmClearance()
    clip = {"min": 1.0, "neg_s": 0.0, "n": 0, "at": ""}
    ticks = int(seconds / 0.02)
    every = max(1, int(round(1.0 / (CLIP_HZ * 0.02))))
    last_cmd = None
    try:
        for k in range(ticks):
            br = st.brains.get("m0")
            watch.state = str(getattr(br, "state", ""))
            # The pinch's PHASE, because that is the scope a rate cap is
            # written in — "pinch" alone names 7 different blends.
            phase = ""
            pcd = getattr(br, "_pinch", None)
            if watch.state == "pinch" and isinstance(pcd, dict):
                phase = "/" + str(pcd.get("phase", ""))
            st.drive(np.zeros(3), "auto")
            w.step()
            cmd = d.ctrl[act].copy()
            if last_cmd is not None:
                jump = np.abs(cmd - last_cmd)
                for i in np.flatnonzero(jump > watch.step):
                    watch.step_at[i] = watch.state + phase
                watch.step = np.maximum(watch.step, jump)
            last_cmd = cmd
            if k % every == 0:
                q = d.qpos[qadr[:5]]
                c = ac.clearance(q, jaw=float(d.qpos[qadr[5]]))
                if c > -0.15:                 # -1.0 / -0.2 are sentinels, not depths
                    if c < clip["min"]:
                        clip["at"] = watch.state
                    clip["min"] = min(clip["min"], float(c))
                    clip["neg_s"] += (c < 0.0) * every * 0.02
                clip["n"] += 1
    finally:
        mujoco.mj_step = real_step

    x, y, yaw = w.ducks["m0"].driver.pose(d)
    cs, sn = np.cos(-yaw), np.sin(-yaw)
    binned = 0
    for p in sc0.props:
        q = d.xpos[m.body(p.id).id]
        dx, dy = q[0] - x, q[1] - y
        bx, by = dx * cs - dy * sn, dx * sn + dy * cs
        if (moss.BIN_INTERIOR_X[0] < bx < moss.BIN_INTERIOR_X[1]
                and moss.BIN_INTERIOR_Y[0] < by < moss.BIN_INTERIOR_Y[1]
                and q[2] > moss.BIN_FLOOR_Z):
            binned += 1
    return dict(seed=seed, seconds=seconds, samples=watch.n,
                binned=binned, props=len(sc0.props),
                joints=watch.rows(seconds), clip=clip,
                floor=dict(n=floor["n"], driving=floor["drive"],
                           max_force=floor["force"], geoms=floor["geoms"],
                           ticks=ticks * w.substeps),
                contacts=dict(n=hits["n"], max_force=hits["force"],
                              worst_depth=hits["depth"], at=hits["at"],
                              pairs={k: v for k, v in sorted(
                                  hits["pairs"].items(), key=lambda kv: -kv[1][0])[:6]}))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    seconds = float(argv[0]) if argv else 180.0
    seeds = [int(x) for x in (argv[1].split(",") if len(argv) > 1 else ["0"])]
    out = Path(argv[2]) if len(argv) > 2 else None
    sc0 = S.Scenario.from_dict(json.loads(Path("scenarios/moss-yard.json").read_text()))
    runs = []
    for s in seeds:
        r = run_one(sc0, s, seconds)
        runs.append(r)
        print(f"\n=== seed {s}: {r['binned']}/{r['props']} binned, "
              f"{r['samples']} physics samples ===")
        print(f"{'joint':15s} {'peak':>10s} {'duty':>7s} {'sat s':>7s} "
              f"{'stall s':>8s} {'worst':>7s} {'limit s':>8s} {'max spd':>8s} {'d cmd':>7s}")
        for j in r["joints"]:
            print(f"{j['joint']:15s} {j['peak']:6.2f}/{j['clamp']:<4.1f} "
                  f"{j['duty']:7.3f} {j['sat_s']:7.1f} {j['stall_s']:8.1f} "
                  f"{j['worst_stall_s']:7.2f} {j['limit_s']:8.1f} "
                  f"{j['max_speed']:8.2f} {j['max_cmd_step']:7.3f}   "
                  f"peak in {j['peak_at'] or '-'}, fastest in {j['speed_at'] or '-'}"
                  + f", biggest step in {j['step_at'] or '-'}"
                  + (f", stalled in {j['stall_at']}" if j['worst_stall_s'] >= STALL_REPORT_S else ""))
        c = r["clip"]
        print(f"visible arm: min clearance {c['min'] * 1000:+.1f} mm, "
              f"{c['neg_s']:.1f} s inside the rover, worst in {c['at'] or '-'} "
              f"({c['n']} samples)")
        h = r["contacts"]
        print(f"arm/body contacts: {h['n']} substeps, max force {h['max_force']:.1f} N, "
              f"worst penetration {h['worst_depth'] * 1000:+.2f} mm, "
              f"hardest in {h['at'] or '-'}")
        for k, v in h["pairs"].items():
            print(f"    {k}: {v[0]} substeps, {v[1]:.1f} N, {v[2] * 1000:+.2f} mm")
        fl = r["floor"]
        print(f"arm on the FLOOR: {fl['n']} substeps ({100.0 * fl['n'] / fl['ticks']:.2f}% "
              f"of the run), {fl['driving']} of them while driving, "
              f"max {fl['max_force']:.1f} N   {fl['geoms'] or '{}'}")
    if len(runs) > 1:
        print("\n=== pooled ===")
        for i, name in enumerate(runs[0]["joints"]):
            w = [r["joints"][i] for r in runs]
            print(f"{name['joint']:15s} duty {np.mean([x['duty'] for x in w]):.3f} "
                  f"worst stall {max(x['worst_stall_s'] for x in w):.2f} s "
                  f"peak {max(x['peak'] for x in w):.2f} N m "
                  f"limit {np.mean([x['limit_s'] for x in w]):.1f} s")
        print(f"worst visible clipping {min(r['clip']['min'] for r in runs) * 1000:+.1f} mm; "
              f"worst arm/body force {max(r['contacts']['max_force'] for r in runs):.1f} N")
    if out is not None:
        out.write_text(json.dumps(runs, indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
