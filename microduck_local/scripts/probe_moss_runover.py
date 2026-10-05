"""Where does MOSS put what it drops, and does it then drive over it?

    uv run python scripts/probe_moss_runover.py 300 0,1,2,3 out.json
    PARAMS="drop_back_m=0.18" uv run python scripts/probe_moss_runover.py ...

A DROP: a prop that was carried comes back to the floor outside the bin — its
position is recorded in the BASE frame, which is what `min_x` filters on. A
RUN-OVER: the base's own footprint passes over a prop while the tracks drive.

MEASURED with it (2026-09-29, 8 seeds): drops land a median 0.293 m directly
ahead and 55% fall inside `min_x`, where the brain cannot see them at all —
but 13 of 16 run-overs are on objects that were NEVER picked up. See
docs/roadmap.md. The run-over count is 0-10 per seed with a paired standard
error of +-2.5, so it CANNOT resolve an intervention at this sample size; use
`probe_moss_shove.py`'s contact seconds for that.
"""
import json
import sys
from pathlib import Path

import numpy as np

from microduck_local.robots import moss
from microduck_local.viz_server import load_policy_infer
from microduck_local.world import scenario as S
from microduck_local.world_server import WorldState

SECONDS = float(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split(",")]
sc0 = S.Scenario.from_dict(json.loads(Path("scenarios/moss-yard.json").read_text()))
# The rover's own plan-view half-extents, base frame (hull + tracks).
HALF_X, HALF_Y = 0.16, 0.13
DRIVING = 0.03           # m/s of base speed above which a pass is under power

out = []
for seed in seeds:
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc0, seed=seed), sc0
    st.set_brain("m0", "tidy_moss")
    import dataclasses
    import os
    spec = os.environ.get("PARAMS", "")
    if spec.strip():
        br0 = st.brains["m0"]
        want = {}
        for part in spec.split(","):
            k, _, v = part.partition("=")
            cur = getattr(br0.p, k.strip())
            want[k.strip()] = type(cur)(v)
        br0.p = dataclasses.replace(br0.p, **want)
        for k, v in want.items():
            assert getattr(br0.p, k) == v, f"override of {k} did not take"
    w = st.world
    m, d = w.model, w.data
    br = st.brains["m0"]
    pid = {p.id: m.body(p.id).id for p in sc0.props}
    carried = {k: False for k in pid}
    drops, overs, over_ids = [], [], set()
    prev = None
    for k in range(int(SECONDS / 0.02)):
        st.drive(np.zeros(3), "auto")
        w.step()
        x, y, yaw = w.ducks["m0"].driver.pose(d)
        spd = 0.0 if prev is None else np.hypot(x - prev[0], y - prev[1]) / 0.02
        prev = (x, y)
        c, s_ = np.cos(-yaw), np.sin(-yaw)
        for name, b in pid.items():
            q = d.xpos[b]
            dx, dy = q[0] - x, q[1] - y
            bx, by = dx * c - dy * s_, dx * s_ + dy * c
            inbin = (moss.BIN_INTERIOR_X[0] < bx < moss.BIN_INTERIOR_X[1]
                     and moss.BIN_INTERIOR_Y[0] < by < moss.BIN_INTERIOR_Y[1]
                     and q[2] > moss.BIN_FLOOR_Z)
            up = q[2] > 0.06 and not inbin
            if carried[name] and not up and not inbin:
                drops.append((round(k * 0.02, 1), name, round(float(bx), 3),
                              round(float(by), 3), str(br.state)))
            carried[name] = up
            # under the rover, on the floor, while it is moving
            if (not inbin and q[2] < 0.12 and abs(bx) < HALF_X and abs(by) < HALF_Y
                    and spd > DRIVING):
                overs.append((round(k * 0.02, 1), name, round(float(bx), 3),
                              round(float(by), 3), str(br.state)))
                over_ids.add(name)
    # collapse run-overs into episodes (same prop, within 1 s)
    eps, last = [], {}
    for t_, name, bx, by, stt in overs:
        if name not in last or t_ - last[name] > 1.0:
            eps.append((t_, name, bx, by, stt))
        last[name] = t_
    print(f"\nseed {seed}: {len(drops)} drops, {len(eps)} run-over episodes "
          f"on {len(over_ids)} distinct props")
    for e in drops[:8]:
        print(f"    drop  {e[0]:6.1f}s {e[1]:8s} base ({e[2]:+.3f},{e[3]:+.3f})  in {e[4]}")
    for e in eps[:8]:
        print(f"    OVER  {e[0]:6.1f}s {e[1]:8s} base ({e[2]:+.3f},{e[3]:+.3f})  in {e[4]}")
    x, y, yaw = w.ducks["m0"].driver.pose(d)
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    binned = 0
    for p_ in sc0.props:
        q = d.xpos[pid[p_.id]]
        dx, dy = q[0]-x, q[1]-y
        bx, by = dx*c - dy*s_, dx*s_ + dy*c
        if (moss.BIN_INTERIOR_X[0] < bx < moss.BIN_INTERIOR_X[1]
                and moss.BIN_INTERIOR_Y[0] < by < moss.BIN_INTERIOR_Y[1]
                and q[2] > moss.BIN_FLOOR_Z):
            binned += 1
    print(f"    binned {binned}/{len(sc0.props)}")
    out.append({"seed": seed, "drops": drops, "overs": eps, "binned": binned,
                "props": len(sc0.props)})
Path(sys.argv[3]).write_text(json.dumps(out, indent=1)) if len(sys.argv) > 3 else None
