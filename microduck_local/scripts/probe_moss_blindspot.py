"""WHY the rover does not see what it drives into — vertical, lateral, or not
a sensing problem at all.

    uv run python scripts/probe_moss_blindspot.py 300 0,1,2,3 out.json

`probe_moss_near_band.py` asked how often litter sits in the near band and
answered "about a second a run", but it gated on the BRAIN's 60 deg bearing —
which excludes exactly the geometry a track runs something over with (a prop
at x 0.20, y 0.15 is at bearing 75 deg). This takes the other route: find the
contacts, then look BACKWARD from each one and ask where the object was and
which sensor could have had it.

For every contact between a driving surface and a loose prop, the three
seconds before it are replayed from a ring buffer: the prop's base-frame
position, and whether it was inside the head camera's frustum at the shipped
pitch and at four candidates, inside the brain's own target gate, and whether
the object MEMORY held anything there. A prop that was in frame and in the
memory is not a sensing failure.
"""
import collections
import json
import os
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from microduck_local.robots import moss
from microduck_local.viz_server import load_policy_infer
from microduck_local.world import scenario as S
from microduck_local.world_server import WorldState

SECONDS = float(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split(",")]
sc0 = S.Scenario.from_dict(json.loads(Path("scenarios/moss-yard.json").read_text()))
#: Head-camera pitches to score, rad down. 0.0 is what ships.
PITCHES = [0.0, 0.175, 0.349, 0.524]          # 0, 10, 20, 30 deg
LOOKBACK_S = 3.0
#: How near a memory entry has to be to count as "the brain knew about this
#: object". Loose enough and a neighbouring entry answers for it, so the
#: headline was re-run at 0.10 as well as 0.20 and moved 60% -> see the
#: roadmap. `MEM_R` overrides it.
MEM_R = float(os.environ.get("MEM_R", "0.20"))
HALF_H = math.radians(moss.CAMERA_HFOV_DEG) / 2.0
HALF_V = math.radians(moss.CAMERA_VFOV_DEG) / 2.0


def in_frustum(px, py, pz, pitch):
    """Is a point (base frame, relative to the camera) inside the frustum of a
    head camera pitched `pitch` rad DOWN? Centre only — an object straddling
    the edge counts as out, which errs toward "it could not have seen it"."""
    c, s = math.cos(pitch), math.sin(pitch)
    lx = px * c - pz * s
    ly = py
    lz = px * s + pz * c
    if lx <= 0.0:
        return False
    return (abs(math.atan2(ly, lx)) < HALF_H
            and abs(math.atan2(lz, math.hypot(lx, ly))) < HALF_V)


out = []
for seed in seeds:
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc0, seed=seed), sc0
    st.set_brain("m0", "tidy_moss")
    br, w = st.brains["m0"], st.world
    m, d = w.model, w.data
    pre = "m0/"

    def bname(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""

    def gname(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

    # the DRIVING surfaces: hull and tracks, never the arm and never the bin
    base_g = {g for g in range(m.ngeom)
              if bname(g).startswith(pre)
              and not any(k in bname(g) for k in
                          ("upper_arm", "lower_arm", "wrist", "gripper", "finger"))
              and not gname(g).startswith(pre + "bin")}
    prop_g, pid = {}, {}
    for p_ in sc0.props:
        bid = m.body(p_.id).id
        pid[p_.id] = bid
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) == bid:
                prop_g[g] = p_.id

    ring = collections.deque(maxlen=int(LOOKBACK_S / 0.02) + 1)
    touching = set()
    episodes = []
    hit_now = set()
    real = mujoco.mj_step

    def watched(mm, dd, nstep=1):
        real(mm, dd, nstep)
        if mm is not m:
            return
        for c in range(dd.ncon):
            con = dd.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            pg = prop_g.get(g1) or prop_g.get(g2)
            if pg is None or not ((g1 in base_g) or (g2 in base_g)):
                continue
            if dd.xpos[m.body(pg).id][2] >= moss.BIN_FLOOR_Z:
                continue                   # riding in the basket
            hit_now.add(pg)

    mujoco.mj_step = watched
    try:
        for k in range(int(SECONDS / 0.02)):
            t = k * 0.02
            hit_now.clear()
            st.drive(np.zeros(3), "auto")
            w.step()
            bx, by, byaw = w.ducks["m0"].driver.pose(d)
            cy, sy = math.cos(-byaw), math.sin(-byaw)
            snap = {"t": t, "state": str(getattr(br, "state", "")), "props": {}}
            for p_ in sc0.props:
                q = d.xpos[pid[p_.id]]
                dx, dy = q[0] - bx, q[1] - by
                x, y, z = dx * cy - dy * sy, dx * sy + dy * cy, float(q[2])
                px = x - moss.CAMERA_POS[0]
                pz = z - moss.CAMERA_POS[2]
                mem = None
                if getattr(br, "_mem", None) is not None:
                    e = br._mem.near(float(q[0]), float(q[1]), MEM_R)
                    mem = None if e is None else bool(br._mem.confirmed(e))
                snap["props"][p_.id] = {
                    "x": x, "y": y, "z": z,
                    "bearing": math.atan2(y, px) if px > 0 else math.pi,
                    "fov": {f"{p:.3f}": in_frustum(px, y, pz, p) for p in PITCHES},
                    "gate": bool(abs(math.atan2(y, px) if px > 0 else math.pi)
                                 < br.p.max_bearing and x >= br.p.min_x),
                    "mem": mem}
            ring.append(snap)
            for name in hit_now - touching:
                hist = [s for s in ring]
                at = hist[-1]["props"][name]
                ep = {"t": t, "prop": name, "state": snap["state"],
                      "at_x": at["x"], "at_y": at["y"],
                      "at_bearing": math.degrees(at["bearing"]),
                      "mem_at_contact": at["mem"],
                      "ever_fov": {f"{p:.3f}": any(s["props"][name]["fov"][f"{p:.3f}"]
                                                   for s in hist) for p in PITCHES},
                      "ever_gate": any(s["props"][name]["gate"] for s in hist),
                      "ever_mem": any(s["props"][name]["mem"] for s in hist),
                      "lookback": len(hist) * 0.02}
                episodes.append(ep)
            touching = set(hit_now)
    finally:
        mujoco.mj_step = real

    n = len(episodes)
    print(f"\nseed {seed}: {n} contact episodes with loose props")
    if n:
        for p in PITCHES:
            k_ = f"{p:.3f}"
            c = sum(1 for e in episodes if e["ever_fov"][k_])
            print(f"    in frame at pitch {math.degrees(p):4.0f} deg during the "
                  f"{LOOKBACK_S:.0f} s before contact: {c}/{n} "
                  f"({100*c/n:3.0f}%)")
        g = sum(1 for e in episodes if e["ever_gate"])
        mm_ = sum(1 for e in episodes if e["ever_mem"])
        mc = sum(1 for e in episodes if e["mem_at_contact"])
        print(f"    inside the brain's own target gate at some point: {g}/{n}")
        print(f"    in the object memory at some point:                {mm_}/{n}")
        print(f"    in the object memory AT contact:                   {mc}/{n}")
        bys = collections.Counter(e["state"] for e in episodes)
        print("    state at contact: "
              + "  ".join(f"{k or '-'}:{v}" for k, v in bys.most_common()))
        bb = [abs(e["at_bearing"]) for e in episodes]
        print(f"    |bearing| at contact: median {sorted(bb)[len(bb)//2]:.0f} deg, "
              f"range {min(bb):.0f}..{max(bb):.0f}; "
              f"{sum(1 for v in bb if v > math.degrees(HALF_H))} of {n} are "
              f"outside the {moss.CAMERA_HFOV_DEG:.0f} deg lens")
    out.append({"seed": seed, "episodes": episodes})
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
