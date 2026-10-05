"""Does the ROVER hit the litter, and how hard?

    uv run python scripts/probe_moss_shove.py 300 0,1 out.json

Contacts between the driving surfaces (hull and tracks — NOT the bin) and any
prop that is not in the bin, every 2 ms physics step, with the brain state.
A continuous measure where the run-over count is a handful of discrete
episodes, which is why it is the instrument to A/B against.

MEASURED (2026-09-29): 7.0 s and 16.0 s of contact per 300 s seed, peaking at
26.3 N against an 18 g can, in `approach`, `creep` and `deploy` — while driving
at a target. Excluding the BIN is the whole trick: the first cut left it in and
counted 3.7M contact-substeps in a run that has 150k, because every delivered
object rests in the basket for the remainder of the run.
"""
import json
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
        if points:
            touch["substeps"] += 1
            touch["run"] += 1
            touch["worst_run"] = max(touch["worst_run"], touch["run"])
            touch["max_points"] = max(touch["max_points"], points)
        else:
            touch["run"] = 0
    mujoco.mj_step = watched
    try:
        for k in range(int(SECONDS / 0.02)):
            state[0] = str(getattr(br, "state", ""))
            st.drive(np.zeros(3), "auto")
            w.step()
    finally:
        mujoco.mj_step = real
    tot = sum(v["n"] for v in hits.values())
    print(f"\nseed {seed}: {touch['substeps'] * 0.002:.1f} s of contact "
          f"({touch['substeps']} substeps), longest unbroken "
          f"{touch['worst_run'] * 0.002:.1f} s, up to {touch['max_points']} "
          f"points at once, {len(hits)} distinct props "
          f"({tot} contact-pair substeps)", flush=True)
    for name, v in sorted(hits.items(), key=lambda kv: -kv[1]["n"]):
        top = sorted(v["states"].items(), key=lambda kv: -kv[1])[:3]
        print(f"    {name:8s} {v['n']:6d} substeps  max {v['f']:6.1f} N   "
              + ", ".join(f"{s or '-'}:{n}" for s, n in top))
    out.append({"seed": seed, "touch": dict(touch),
                "hits": {k: {"n": v["n"], "f": v["f"], "states": v["states"]}
                         for k, v in hits.items()}})
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
