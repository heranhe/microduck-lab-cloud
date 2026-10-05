"""Does the COLOUR camera report close litter, and does the brain bin it?

`_toys_in_view` applies three filters and feeds BOTH targeting and the object
memory. This records every raw `cls == "toy"` detection and which filter, if
any, discarded it — so "we never see it" and "we see it and throw it away" can
be told apart.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from microduck_local.brain import tidy_moss as TM
from microduck_local.robots import moss
from microduck_local.viz_server import load_policy_infer
from microduck_local.world import scenario as S
from microduck_local.world_server import WorldState

SECONDS = float(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split(",")]
sc0 = S.Scenario.from_dict(json.loads(Path("scenarios/moss-yard.json").read_text()))

for seed in seeds:
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc0, seed=seed), sc0
    st.set_brain("m0", "tidy_moss")
    br = st.brains["m0"]
    tally = {"kept": 0, "bearing": 0, "min_x": 0, "own_bin": 0}
    close_x = []
    orig = TM.TidyMoss._toys_in_view

    # `_toys_in_view` now takes the range floor as a second argument (the
    # split, `near_min_x`); pass it straight through, or a call from
    # `_remember` would bind the floor to `_orig` and this probe would run
    # the brain's sensing against itself.
    def spy(self, frame, min_x=None, _orig=orig):
        if frame is not None and frame.detections:
            for det in frame.detections:
                if det.cls != "toy":
                    continue
                r = self._range(det)
                x = moss.CAMERA_POS[0] + r * math.cos(det.bearing)
                y = moss.CAMERA_POS[1] + r * math.sin(det.bearing)
                if abs(det.bearing) > self.p.max_bearing:
                    tally["bearing"] += 1
                elif x < self.p.min_x:
                    tally["min_x"] += 1
                    close_x.append(float(x))
                elif (moss.BIN_INTERIOR_X[0] - 0.05 < x < moss.BIN_INTERIOR_X[1] + 0.05
                      and moss.BIN_INTERIOR_Y[0] < y < moss.BIN_INTERIOR_Y[1]):
                    tally["own_bin"] += 1
                else:
                    tally["kept"] += 1
        return _orig(self, frame, min_x)

    TM.TidyMoss._toys_in_view = spy
    try:
        w = st.world
        for _ in range(int(SECONDS / 0.02)):
            st.drive(np.zeros(3), "auto")
            w.step()
    finally:
        TM.TidyMoss._toys_in_view = orig
    tot = sum(tally.values())
    print(f"\nseed {seed}: {tot} 'toy' detections from the COLOUR camera")
    for k, v in tally.items():
        print(f"    {k:10s} {v:7d}  {100 * v / max(tot, 1):5.1f}%")
    if close_x:
        cx = sorted(close_x)
        print(f"    the min_x ones sit at x = {cx[0]:.3f} .. {cx[-1]:.3f} m, "
              f"median {cx[len(cx) // 2]:.3f}")
        print(f"    of those, {sum(1 for v in cx if v > 0.186)} "
              f"({100 * sum(1 for v in cx if v > 0.186) / len(cx):.0f}%) are "
              f"AHEAD of the front bumper (0.186 m) — real litter, not the chassis")
