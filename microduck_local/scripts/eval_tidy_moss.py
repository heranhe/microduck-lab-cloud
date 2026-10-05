"""How many cans does tidy_moss actually get into the bin? Count, don't read."""
import json
import sys
from pathlib import Path

import numpy as np

from microduck_local.robots import moss
from microduck_local.viz_server import load_policy_infer
from microduck_local.world import scenario as S
from microduck_local.world_server import WorldState

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0"])]
sc0 = S.Scenario.from_dict(json.loads(Path("scenarios/moss-yard.json").read_text()))

for seed in seeds:
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc0, seed=seed), sc0
    st.set_brain("m0", "tidy_moss")
    import os
    if os.environ.get("MICRODUCK_MOSS_POLICY"):
        print(f"  (learned pickup: {os.environ['MICRODUCK_MOSS_POLICY']})")
    w = st.world; r = w.ducks["m0"]
    cans = [p.id for p in sc0.props]
    for _ in range(int(SECONDS / 0.02)):
        st.drive(np.zeros(3), "auto")
        w.step()
    x, y, yaw = r.driver.pose(w.data)
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    inbin = []
    for cid in cans:
        p = w.data.xpos[w.model.body(cid).id]
        dx, dy = p[0] - x, p[1] - y
        bx, by = dx*c - dy*s_, dx*s_ + dy*c
        ok = (moss.BIN_INTERIOR_X[0] < bx < moss.BIN_INTERIOR_X[1]
              and moss.BIN_INTERIOR_Y[0] < by < moss.BIN_INTERIOR_Y[1]
              and p[2] > moss.BIN_FLOOR_Z)
        inbin.append(ok)
        print(f"  seed {seed} {cid}: base ({bx:+.3f},{by:+.3f},{p[2]:.3f}) "
              f"{'IN BIN' if ok else 'on the floor'}")
    brain = st.brains["m0"]
    print(f"seed {seed}: {sum(inbin)}/{len(cans)} cans in the bin after "
          f"{SECONDS:.0f} s   (brain counted {brain.picked} releases, "
          f"state {brain.state})")
