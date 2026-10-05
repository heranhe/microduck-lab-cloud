"""THE FLOOR SCRAPE: the gripper pads touching the ground, and what it costs.

    uv run python scripts/probe_moss_scrape.py 300 0,1,2,3,4,5,6,7 out.json

`probe_moss_safety.py`'s `floor` family counts substeps and a peak force but
says nothing about WHEN or for how long. Measured there (2026-09-30, 8 seeds
at 180 s): 2292 substeps, 90% of them while driving, peak 186 N — and 71% of
it is ONE seed, with two seeds clean. So the question is not "how much" but
"what is the event".

This groups the contact into episodes and reports, per episode: the brain
state, how long it lasts, the peak normal force, how far the pad SLID along
the floor while loaded (abrasion is slip x force, not force), what it was
reaching for, and how much of it `shoulder_lift` spent at its 2.2 N.m clamp.
A pad pressed briefly is a knock; a pad dragged for a metre is a consumable
being ground away, and the joints above it are holding the moment.

MEASURED (2026-09-30, 32 seeds x 300 s):

    28 of 32 seeds affected, 129 episodes, ALL of them in `creep`
    pad-on-floor per run   median 0.56 s   max 3.04 s   (a run is 300 s)
    slip per run           median   45 mm  max  592 mm
    peak force / episode   median 33.4 N   p90 58.9 N   max 186.3 N
    episode length         median 0.12 s   max 1.18 s
    shoulder_lift at >=90% of clamp: 72% of loaded substeps

**It is the learned pickup creeping on a TOPPLED can.** The targets are
`can0`/`can1`/`can2`/`tall0` — the tall props — at a median height of **30 mm**
where a standing can's body sits at 57.5 mm and `tall0`'s at 85 mm. Knocked
over, they read as low objects, the brain lowers the jaws to their height, and
`creep` — which is the LEARNED policy driving base and arm together — ploughs
the pads in. Nothing in that policy's env costs it a floor contact.

**Severity, in context.** Brief and common rather than severe: the median
episode is 0.12 s at 33 N, and although `shoulder_lift` is at its clamp for
72% of the loaded substeps, the scrape accounts for at most **8%** of that
joint's total saturation (median 0.45 s of a median 5.48 s per run). The
hardware cost is pad abrasion and the odd shock, not a joint being held out.
"""
import collections
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

#: Base speed above which a pad on the floor is being DRAGGED, m/s.
DRIVING_MPS = 0.03
#: Gap in contact longer than this ends an episode, s.
EPISODE_GAP_S = 0.20

SECONDS = float(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split(",")]
sc0 = S.Scenario.from_dict(json.loads(Path("scenarios/moss-yard.json").read_text()))
out = []
for seed in seeds:
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc0, seed=seed), sc0
    st.set_brain("m0", "tidy_moss")
    br, w = st.brains["m0"], st.world
    m, d = w.model, w.data

    def gn(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

    arm_g = {g for g in range(m.ngeom)
             if any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                            int(m.geom_bodyid[g])) or "")
                    for k in ("gripper", "finger"))}
    floor_g = {g for g in range(m.ngeom)
               if int(m.geom_bodyid[g]) == 0 and gn(g)}
    lift_act = [a for a in range(m.nu)
                if int(m.actuator_trnid[a][0]) == m.joint("m0/shoulder_lift").id][0]
    six = np.zeros(6)
    eps: list[dict] = []
    cur: list[dict | None] = [None]        # one cell, so the hook can rebind it
    tick = [0]
    real = mujoco.mj_step

    def watched(mm, dd, n=1):
        real(mm, dd, n)
        if mm is not m:
            return
        t = tick[0] * 0.02
        force, where = 0.0, None
        for c in range(dd.ncon):
            con = dd.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in arm_g and g2 in floor_g) or (g2 in arm_g and g1 in floor_g):
                mujoco.mj_contactForce(mm, dd, c, six)
                f = float(np.linalg.norm(six[:3]))
                if f > force:
                    force, where = f, gn(g1 if g1 in arm_g else g2)
        if where is None:
            return
        xy = dd.geom_xpos[mm.geom(where).id][:2].copy()
        e = cur[0]
        if e is None or t - e["last"] > EPISODE_GAP_S:
            e = {"t0": t, "last": t, "force": 0.0, "n": 0, "slip": 0.0,
                 "geoms": collections.Counter(), "states": collections.Counter(),
                 "xy": xy}
            eps.append(e)
            cur[0] = e
        else:
            # Abrasion is SLIP under load, not force: how far the pad's own
            # origin travelled across the floor while it was touching.
            e["slip"] += float(np.linalg.norm(xy - e["xy"]))
        e["xy"] = xy
        e["last"] = t
        e["force"] = max(e["force"], force)
        e["n"] += 1
        e["states"][str(br.state)] += 1
        e["geoms"][where] += 1
        # WHAT is it reaching for? The nearest prop to the pad, and how tall
        # it stands: the jaws are commanded to the target's grasp height, so
        # a flat object puts them on the floor by construction.
        if e["n"] == 1:
            best, bd = None, 1e9
            for p_ in sc0.props:
                q = dd.xpos[mm.body(p_.id).id]
                dd_ = math.hypot(q[0] - xy[0], q[1] - xy[1])
                if dd_ < bd:
                    best, bd = p_, dd_
            e["target"] = None if best is None else best.id
            e["target_m"] = None if best is None else round(float(bd), 3)
            e["target_h"] = None if best is None else round(
                float(dd.xpos[mm.body(best.id).id][2]), 4)
        # and the joint that is holding it up
        e["lift_sat"] = e.get("lift_sat", 0) + int(
            abs(float(dd.actuator_force[lift_act])) >= 0.9 * 2.2)

    mujoco.mj_step = watched
    try:
        for k in range(int(SECONDS / 0.02)):
            tick[0] = k
            st.drive(np.zeros(3), "auto")
            w.step()
    finally:
        mujoco.mj_step = real
    eps = [e for e in eps if e["n"] >= 5]
    print(f"\n=== seed {seed}: {len(eps)} scrape episodes ===")
    for e in sorted(eps, key=lambda e: -e["n"])[:8]:
        top = ", ".join(f"{k or '-'}:{v}" for k, v in e["states"].most_common(3))
        print(f"   t={e['t0']:6.1f}s  {e['n']*0.002:5.2f}s  peak {e['force']:6.1f} N  "
              f"slid {e['slip']*1000:6.1f} mm  target {e.get('target')} "
              f"({e.get('target_m')} m away, {1000*(e.get('target_h') or 0):.0f} mm tall)  "
              f"shoulder_lift at clamp {100*e.get('lift_sat',0)/max(e['n'],1):.0f}%   {top}")
    for e in eps:
        e.pop("xy", None)
        e["geoms"] = dict(e["geoms"])
        e["states"] = dict(e["states"])
        e["seconds"] = e["n"] * 0.002
    out.append({"seed": seed, "episodes": eps,
                "total_s": sum(e["seconds"] for e in eps),
                "total_slip_mm": sum(e["slip"] for e in eps) * 1000.0})
    print(f"   TOTAL {sum(e['seconds'] for e in eps):.2f}s of pad-on-floor, "
          f"{sum(e['slip'] for e in eps)*1000:.0f} mm slid, "
          f"peak {max([e['force'] for e in eps] or [0]):.1f} N")
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
