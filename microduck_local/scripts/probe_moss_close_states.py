"""WHICH OBJECTS is `min_x` throwing away — by brain state, and by caller.

    uv run python scripts/probe_moss_close_states.py 300 0,1,2

`probe_moss_dropped_dets.py` says the colour camera reports 7.6-13.3% of its
toy detections inside `min_x` and that the brain discards every one. That
number is true and it is not a cost until you ask what those detections are
OF. This splits them by the state the brain was in and by which caller asked
(`_see` every 50 Hz tick, `_remember` once per detector frame).

MEASURED (2026-09-29, three 300 s moss-yard seeds): they land in `creep`,
`pinch`, `lift`, `stow`, `tuck`, `deploy` and `release` — the arm is out and
the close thing in frame is the object in the jaws or the one being reached
for. Only **15/26/0 happen in `search`** and 4/16/13 in `approach`, so
`near_min_x` — which lifts the floor for `_remember` alone, and only while
the arm is stowed — admits **2, 7 and 2 detections per run**. That is why its
eight-seed A/B came back byte-identical. See `probe_moss_near_band.py` for
the same conclusion from the complement.
"""
import collections
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

SECONDS = float(sys.argv[1]); seeds = [int(x) for x in sys.argv[2].split(",")]
sc0 = S.Scenario.from_dict(json.loads(Path("scenarios/moss-yard.json").read_text()))
for seed in seeds:
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc0, seed=seed), sc0
    st.set_brain("m0", "tidy_moss")
    by_state = collections.Counter(); by_caller = collections.Counter()
    admitted = collections.Counter()
    orig = TM.TidyMoss._toys_in_view
    def spy(self, frame, min_x=None, _orig=orig):
        caller = "see" if min_x is None else "remember"
        if frame is not None and frame.detections:
            for det in frame.detections:
                if det.cls != "toy":
                    continue
                r = self._range(det)
                x = moss.CAMERA_POS[0] + r * math.cos(det.bearing)
                if abs(det.bearing) > self.p.max_bearing:
                    continue
                if x < self.p.min_x:
                    by_state[self.state] += 1
                    by_caller[caller] += 1
                    if x >= moss.FRONT_EXTENT_M:
                        admitted[(caller, self.state)] += 1
        return _orig(self, frame, min_x)
    TM.TidyMoss._toys_in_view = spy
    try:
        w = st.world
        for _ in range(int(SECONDS / 0.02)):
            st.drive(np.zeros(3), "auto"); w.step()
    finally:
        TM.TidyMoss._toys_in_view = orig
    tot = sum(by_state.values())
    print(f"\nseed {seed}: {tot} detections rejected by min_x")
    print("   by state:  " + "  ".join(f"{k or '-'}:{v}" for k, v in by_state.most_common()))
    print("   by caller: " + "  ".join(f"{k}:{v}" for k, v in by_caller.most_common()))
    print("   past the bumper, per (caller,state): "
          + "  ".join(f"{c}/{s}:{v}" for (c, s), v in admitted.most_common()))
    hit = sum(v for (c, s), v in admitted.items()
              if c == "remember" and s in ("search", "approach"))
    print(f"   >>> what the split actually admits: {hit}")
