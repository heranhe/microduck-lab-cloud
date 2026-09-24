"""Where does a line-up at the boards die? (roadmap 12al) kick_gym's board placement with the
real brain; per episode the seconds in each brain state, lineup timeouts
(lineup -> search with the clock past lineup_s), declines, pushes, the
reachability counters and whether a swing came.
    cd microduck_local
    uv run python scripts/probe_board_states.py --arm "base=" --arm "reach=spot_reach=0.129"
"""
import argparse, os, sys, statistics as st
from collections import Counter
import numpy as np
sys.path.insert(0, "scripts")
from kick_gym import EPISODE_S, _board_rect, _drive, _place_at_boards, gym_scenario  # noqa: E402
from microduck_local.brain import REGISTRY  # noqa: E402
from microduck_local.brain.controllers import tof_clearance_bearings  # noqa: E402
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer  # noqa: E402
from microduck_local.world.arena import World  # noqa: E402

def run(seed, episodes, knobs, margin, cove=0.0, corner=0.0):
    if knobs: os.environ["MICRODUCK_CHASE"] = knobs
    else: os.environ.pop("MICRODUCK_CHASE", None)
    sc = gym_scenario(cove=cove, corner=corner); infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={x.id: infer for x in sc.ducks}, seed=seed)
    bk = __import__("microduck_local.brain.team", fromlist=["brain_kwargs"]).brain_kwargs
    brains = {x.id: REGISTRY.make("chase", **bk(x, w, {})) for x in sc.ducks}
    brain = brains["d0"]; d = w.ducks["d0"]; rng = np.random.default_rng(seed); out = []
    for ep in range(episodes):
        q, v = _place_at_boards(w, rng, margin)
        for b in brains.values(): b.reset()
        t0 = w.t; states = Counter(); timeouts = 0; prev_state = brain.state; prev_t_state = brain.t_state; swung = False; prev_skill = None
        pushes0, declines0, dropped0, corners0 = brain.pushes, brain.declines, brain.unreach_dropped, brain.unreach_corners
        n_lineups = 0; last_spot = None; min_dist = 9.0; tos = []; ends = Counter()
        while w.t - t0 < EPISODE_S:
            od = w.odom(d) or (0.0, 0.0, 0.0)
            if brain.spot is not None:
                last_spot = brain.spot
                dist = ((brain.spot[0] - od[0]) ** 2 + (brain.spot[1] - od[1]) ** 2) ** 0.5
                min_dist = min(min_dist, dist)
            sn = getattr(brain, "_senses", None)
            fr = None if sn is None else sn.fresh_tof(brain.TOF_MAX_AGE)
            ahead = None if fr is None else float(tof_clearance_bearings(fr)[0])
            _drive(w, brains); w.step()
            states[brain.state] += 0.02
            if prev_state in ("lineup", "settle") and brain.state not in ("lineup", "settle"):
                ends[f"{prev_state}->{brain.state}"] += 1        # how each line-up ENDS
            if prev_state == "lineup" and brain.state == "search" and (w.t - prev_t_state) > brain.p.lineup_s - 0.05:
                timeouts += 1
                if last_spot is not None:
                    dist = ((last_spot[0] - od[0]) ** 2 + (last_spot[1] - od[1]) ** 2) ** 0.5
                    import math as _m
                    herr = abs(_m.atan2(_m.sin(last_spot[3] - od[2]), _m.cos(last_spot[3] - od[2])))
                    hx, hy = _board_rect(w)
                    gap_spot = min(hx - abs(last_spot[0]), hy - abs(last_spot[1]))
                    gap_duck = min(hx - abs(od[0]), hy - abs(od[1]))
                    tos.append({"dist": dist, "min_dist": min_dist, "herr": herr, "ahead": ahead, "gap_spot": gap_spot, "gap_duck": gap_duck, "speed": float(d.speed(w.data)) if hasattr(d, "speed") else None})
                min_dist = 9.0
            if prev_state != "lineup" and brain.state == "lineup":
                n_lineups += 1
            prev_state, prev_t_state = brain.state, brain.t_state
            if d.skill is not None and prev_skill is None and str(d.skill).startswith("kick"):
                swung = True; break
            prev_skill = d.skill
        out.append({"ends": dict(ends), "timeout_rows": tos, "swung": swung, "states": dict(states), "timeouts": timeouts, "lineups": n_lineups,
                    "pushes": brain.pushes - pushes0, "declines": brain.declines - declines0,
                    "dropped": brain.unreach_dropped - dropped0, "corners": brain.unreach_corners - corners0,
                    "final": brain.state})
    return out

ap = argparse.ArgumentParser(); ap.add_argument("--arm", action="append", default=None); ap.add_argument("--episodes", type=int, default=20); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--margin", type=float, default=0.15); ap.add_argument("--cove", type=float, default=0.0); ap.add_argument("--corner", type=float, default=0.0)
a = ap.parse_args()
for spec in (a.arm or ["base="]):
    label, knobs = spec.split("=", 1)
    rows = [r for s in range(a.seeds) for r in run(s, a.episodes, knobs, a.margin, a.cove, a.corner)]
    tot = Counter()
    for r in rows: tot.update(r["states"])
    n = len(rows); secs = sum(tot.values())
    print(f"\n== {label} ({knobs or 'shipped'}): {n} episodes, swung {sum(r['swung'] for r in rows)}, pushes {sum(r['pushes'] for r in rows)}, declines {sum(r['declines'] for r in rows)}, lineups started {sum(r['lineups'] for r in rows)}, lineup timeouts {sum(r['timeouts'] for r in rows)}, fan drops {sum(r['dropped'] for r in rows)}, corner fans {sum(r['corners'] for r in rows)}")
    print("   share of time by state:", {k: f"{v/secs:.0%}" for k, v in tot.most_common(8)})
    print("   final state:", dict(Counter(r["final"] for r in rows).most_common(6)))
    ends = Counter()
    for r in rows: ends.update(r.get("ends", {}))
    print("   how line-ups end:", dict(ends.most_common(8)))
    tos = [x for r in rows for x in r["timeout_rows"]]
    if tos:
        import statistics as _st
        med = lambda k: _st.median(x[k] for x in tos if x.get(k) is not None)
        print(f"   at the {len(tos)} timeouts: dist to spot med {med('dist')*100:.0f} cm (closest during that lineup med {med('min_dist')*100:.0f} cm), |heading err| med {_st.median(x['herr'] for x in tos)*57.3:.0f} deg, "
              f"ToF ahead med {med('ahead')*100 if any(x.get('ahead') is not None for x in tos) else float('nan'):.0f} cm, spot's gap to a board med {med('gap_spot')*100:.0f} cm, DUCK's gap med {med('gap_duck')*100:.0f} cm; "
              f"within 5 cm of the spot at timeout: {sum(1 for x in tos if x['dist']<0.05)/len(tos):.0%}, ever within 5 cm: {sum(1 for x in tos if x['min_dist']<0.05)/len(tos):.0%}")
