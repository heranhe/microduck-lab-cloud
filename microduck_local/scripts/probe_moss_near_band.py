"""THE COMPLEMENT: how often is litter actually inside the near blind band?

    uv run python scripts/probe_moss_near_band.py 300 0,1,2

Counting what a filter throws away says nothing about what it costs until you
count what was there to throw. Every tick the rover spends DRIVING
(`search`/`approach`), this asks the truth — prop world poses, which the brain
never sees — how many props lie between the front bumper and `min_x`, in
front, on the floor, and whether they are inside the camera's vertical field
of view at all.

MEASURED (2026-09-29, three 300 s moss-yard seeds): **41, 62 and 55 ticks —
about one second of a five-minute run**, and every instance is an upright
CAN. Never a cap or a card, and the geometry says why: at `CAMERA_VFOV_DEG`
62 deg from `CAMERA_POS` 0.075 m up, a flat object clears the bottom of the
frame only past **x = 0.281 m**. `min_x` at 0.30 is sitting on the camera's
own floor horizon, so there is no blind band to open — a rover that is to see
what it is about to drive over needs a camera that looks DOWN, not a looser
filter.
"""
import collections
import json
import math
import sys
from pathlib import Path
import numpy as np
from microduck_local.robots import moss
from microduck_local.viz_server import load_policy_infer
from microduck_local.world import scenario as S
from microduck_local.world_server import WorldState

SECONDS = float(sys.argv[1]); seeds = [int(x) for x in sys.argv[2].split(",")]
sc0 = S.Scenario.from_dict(json.loads(Path("scenarios/moss-yard.json").read_text()))
HALF_V = math.radians(moss.CAMERA_VFOV_DEG) / 2.0
for seed in seeds:
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc0, seed=seed), sc0
    st.set_brain("m0", "tidy_moss")
    br, w = st.brains["m0"], st.world
    m, d = w.model, w.data
    pid = {p.id: m.body(p.id).id for p in sc0.props}
    band = collections.Counter(); seen = collections.Counter()
    inframe = collections.Counter(); elev = collections.defaultdict(list)
    for k in range(int(SECONDS / 0.02)):
        st.drive(np.zeros(3), "auto"); w.step()
        if br.state not in ("search", "approach"):
            continue
        bx, by, byaw = w.ducks["m0"].driver.pose(d)
        c, s_ = math.cos(-byaw), math.sin(-byaw)
        for p_ in sc0.props:
            q = d.xpos[pid[p_.id]]
            dx, dy, dz = q[0] - bx, q[1] - by, q[2]
            x, y = dx * c - dy * s_, dx * s_ + dy * c
            if not (moss.FRONT_EXTENT_M < x < 0.30): continue
            if abs(math.atan2(y, x - moss.CAMERA_POS[0])) > br.p.max_bearing: continue
            if dz > moss.BIN_FLOOR_Z and abs(x) < 0.2: continue     # in the basket
            xc = x - moss.CAMERA_POS[0]
            e = math.atan2(dz - moss.CAMERA_POS[2], max(xc, 1e-3))
            band[p_.id] += 1
            elev[p_.id].append(math.degrees(e))
            if e > -HALF_V:
                inframe[p_.id] += 1
    tb, ti = sum(band.values()), sum(inframe.values())
    print(f"\nseed {seed}: {tb} tick-sightings of floor litter in the "
          f"0.19-0.30 m band while driving")
    print(f"    inside the camera's {moss.CAMERA_VFOV_DEG:.0f} deg vertical "
          f"FOV: {ti} ({100*ti/max(tb,1):.0f}%)")
    for name in sorted(band, key=lambda n: -band[n]):
        es = elev[name]
        print(f"    {name:8s} {band[name]:5d} ticks   elevation "
              f"{min(es):+6.1f}..{max(es):+6.1f} deg   in frame "
              f"{inframe[name]:5d} ({100*inframe[name]/band[name]:3.0f}%)")
    print(f"    the camera's floor horizon: a 0 m-high object clears "
          f"-{moss.CAMERA_VFOV_DEG/2:.0f} deg only past x = "
          f"{moss.CAMERA_POS[0] + moss.CAMERA_POS[2]/math.tan(HALF_V):.3f} m")
