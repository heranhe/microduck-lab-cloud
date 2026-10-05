"""TARGET or BYSTANDER: is the litter the rover touches the thing it drove AT?

    uv run python scripts/probe_moss_whom.py 300 0,1,2,3

The question that decides which fix is worth building. If the robot mostly
shoulders its own TARGET on the way in, the fix is in choosing targets. If it
mostly hits things it is not going for, target choice is irrelevant and the
fix is in the PATH.

MEASURED (2026-09-29, three 300 s moss-yard seeds): **bystander 95%, 91%,
100%** — it hits what it is not going for. Classified by comparing each
touched prop against the fix the brain is acting on (`_fix` -> world; note
`_target_world` is only set inside `creep` and reads None at almost every
contact, which is why it is the fallback and not the source).
"""
import json
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
out = []
for seed in seeds:
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc0, seed=seed), sc0
    st.set_brain("m0", "tidy_moss")
    import dataclasses
    import os
    br = st.brains["m0"]
    spec = os.environ.get("PARAMS", "")
    if spec.strip():
        want = {}
        for part in spec.split(","):
            k, _, v = part.partition("=")
            want[k.strip()] = type(getattr(br.p, k.strip()))(v)
        br.p = dataclasses.replace(br.p, **want)
        for k, v in want.items():
            assert getattr(br.p, k) == v, f"override of {k} did not take"
    w = st.world
    m, d = w.model, w.data
    pre = "m0/"
    def bname(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
    # the ROVER itself: hull, bin and tracks — not the arm (already measured)
    # The DRIVING surfaces only. The first cut kept the bin, so every prop
    # resting in the basket counted as the rover touching litter — 3.7M
    # contact-substeps in a run that has 150k substeps in total, which is the
    # instrument failing, not a finding.
    def gname(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

    base_g = {g for g in range(m.ngeom)
              if bname(g).startswith(pre)
              and not any(k in bname(g) for k in
                          ("upper_arm", "lower_arm", "wrist", "gripper", "finger"))
              and not gname(g).startswith(pre + "bin")}
    prop_g = {}
    for p in sc0.props:
        bid = m.body(p.id).id
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) == bid:
                prop_g[g] = p.id
    hits = {}
    #: Substeps on which the rover touched ANY loose prop, and the longest
    #: unbroken run of them. Counted per SUBSTEP, not per contact pair: an
    #: object wedged against the hull reports three or four points at once,
    #: and summing pairs gave 864 "contact seconds" in a 300 s run — a number
    #: that cannot be a duration, which is how the bug announced itself.
    touch = {"substeps": 0, "run": 0, "worst_run": 0, "max_points": 0}
    #: Was the touched prop the one the brain is working on? `_target_world`
    #: is the brain's OWN record of that, in world coords.
    whom = {"target": 0, "bystander": 0, "no_target": 0}
    tgt = [None]
    six = np.zeros(6)
    real = mujoco.mj_step
    state = [""]
    def watched(mm, dd, nstep=1):
        real(mm, dd, nstep)
        if mm is not m:
            return
        points = 0
        for c in range(dd.ncon):
            con = dd.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            pg = prop_g.get(g1) or prop_g.get(g2)
            bg = (g1 in base_g) or (g2 in base_g)
            # ...and only while the prop is NOT in the bin: a delivered object
            # rides in the basket and touches the hull through it all run.
            pb = m.body(pg).id if pg else -1
            q = dd.xpos[pb] if pg else None
            if pg and bg and q is not None and q[2] < moss.BIN_FLOOR_Z:
                mujoco.mj_contactForce(mm, dd, c, six)
                r = hits.setdefault(pg, {"n": 0, "f": 0.0, "states": {}})
                r["n"] += 1
                r["f"] = max(r["f"], float(np.linalg.norm(six[:3])))
                r["states"][state[0]] = r["states"].get(state[0], 0) + 1
                points += 1
                q = dd.xpos[mm.body(pg).id]
                if tgt[0] is None:
                    whom["no_target"] += 1
                elif math.hypot(q[0] - tgt[0][0], q[1] - tgt[0][1]) < 0.15:
                    whom["target"] += 1
                else:
                    whom["bystander"] += 1
        if points:
            touch["substeps"] += 1
            touch["run"] += 1
            touch["worst_run"] = max(touch["worst_run"], touch["run"])
            touch["max_points"] = max(touch["max_points"], points)
        else:
            touch["run"] = 0
    def count_binned():
        bx0, by0, byaw = w.ducks["m0"].driver.pose(d)
        cb, sb = np.cos(-byaw), np.sin(-byaw)
        n = 0
        for p_ in sc0.props:
            q = d.xpos[m.body(p_.id).id]
            dx, dy = q[0] - bx0, q[1] - by0
            qx, qy = dx * cb - dy * sb, dx * sb + dy * cb
            if (moss.BIN_INTERIOR_X[0] < qx < moss.BIN_INTERIOR_X[1]
                    and moss.BIN_INTERIOR_Y[0] < qy < moss.BIN_INTERIOR_Y[1]
                    and q[2] > moss.BIN_FLOOR_Z):
                n += 1
        return n

    # Score at a horizon where the task is NOT yet finished as well as at the
    # end: 88 of 88 props binned cannot tell a slower robot from a faster one,
    # and the arm rate cap read free at 300 s and -1.50/seed at 180 s.
    marks = {}
    mujoco.mj_step = watched
    try:
        for k in range(int(SECONDS / 0.02)):
            if k and k % 1500 == 0:
                marks[round(k * 0.02)] = count_binned()
            state[0] = str(getattr(br, "state", ""))
            # `_target_world` is only set inside `creep`; the fix the brain
            # is ACTING on (`_fix`, base frame) is live in every driving
            # state, so convert that to world instead.
            tgt[0] = getattr(br, "_target_world", None)
            if tgt[0] is None:
                fx = getattr(br, "_fix", None)
                if fx is not None:
                    try:
                        tgt[0] = br._world(fx)
                    except Exception:
                        tgt[0] = None
            st.drive(np.zeros(3), "auto")
            w.step()
    finally:
        mujoco.mj_step = real
    # The MISSION cost, in the same run: a safety change that stops the rover
    # touching litter by never reaching it is not a safety change. Counted the
    # way `probe_moss_runover.py` counts it — in the bin's frame, above its
    # floor.
    binned = count_binned()
    tot = sum(v["n"] for v in hits.values())
    tw = whom["target"] + whom["bystander"] + whom["no_target"]
    print(f"  WHOM: target {100*whom['target']/max(tw,1):.0f}%  "
          f"bystander {100*whom['bystander']/max(tw,1):.0f}%  "
          f"no target held {100*whom['no_target']/max(tw,1):.0f}%   (of {tw} contacts)")
    print(f"\nseed {seed}: {touch['substeps'] * 0.002:.1f} s of contact "
          f"({touch['substeps']} substeps), longest unbroken "
          f"{touch['worst_run'] * 0.002:.1f} s, up to {touch['max_points']} "
          f"points at once, {len(hits)} distinct props "
          f"({tot} contact-pair substeps)", flush=True)
    for name, v in sorted(hits.items(), key=lambda kv: -kv[1]["n"]):
        top = sorted(v["states"].items(), key=lambda kv: -kv[1])[:3]
        print(f"    {name:8s} {v['n']:6d} substeps  max {v['f']:6.1f} N   "
              + ", ".join(f"{s or '-'}:{n}" for s, n in top))
    print(f"    binned {binned}/{len(sc0.props)}  "
          + " ".join(f"{t_}s:{n_}" for t_, n_ in sorted(marks.items())))
    out.append({"seed": seed, "touch": dict(touch),
                "binned": binned, "props": len(sc0.props), "marks": marks,
                "whom": dict(whom),
                "hits": {k: {"n": v["n"], "f": v["f"], "states": v["states"]}
                         for k, v in hits.items()}})
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
