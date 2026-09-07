"""What the head and the body are doing when the ball is not in sight.

    cd microduck_local
    uv run python scripts/probe_search.py --seeds 24 --seconds 300 --per-side 2 --jobs 12

Three questions the goal column cannot answer, all of them counted per TICK so
they resolve on a handful of seeds instead of a hundred (`docs/roadmap.md`
Track 4.1.5):

1. **How much of a run is spent frozen?** `Chase.step`'s search stops the body
   dead every `search_dip_every` for `search_dip_s` and looks down at
   `dip_range`. Reported as seconds a duck spends in `search`, and the part of
   that with a zero twist.
2. **How long is the duck blind?** A "sighting" is a detector FRAME carrying a
   ball; a loss runs from `DET_MAX_AGE` after the last one to the next. The
   distribution of those gaps is the re-acquisition time, pooled over every
   loss event in the battery rather than averaged per run.
3. **Does a turned head still make the ToF report a wall that is not there?**
   Every tick this recomputes the clearance BOTH ways — `tof_clearance_bearings`
   (what ships) and the pre-`554e4de` rule of "the middle two columns are
   ahead" — and counts the ticks where the old rule would have stopped the duck
   (`ahead < tof_stop`) while the bearing rule says the way is clear, binned by
   how far the head is yawed. That is the coupling that killed every head-gaze
   variant, measured in play instead of argued from the diff.

Everything `eval-pitch` reports for the same run rides along (possession, the
spin fractions, falls, kicks, goals), so one battery screens an arm on the
cheap metric and the expensive ones at once.

Arms are selected with `MICRODUCK_CHASE`, and the values are read back off the
CONSTRUCTED brain — never off a fresh `ChaseParams()`, which will happily agree
with you while the live brain runs the defaults (playbook rule 0).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.controllers import (
    Chase,
    ChaseParams,
    WanderParams,
    _column_clearance,
    tof_clearance_bearings,
)
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.world import World, make_pitch
from microduck_local.world.metrics import PitchMetrics, SpinMetrics

YAW_BINS = (0.10, 0.35, 0.70)          # rad: level-ish, turned, well off the line


def _ahead_columns(frame) -> float:
    """The pre-554e4de rule: 'ahead' is the two middle sensor columns, whatever
    the head is pointing at. Kept here (not imported) so the counterfactual
    survives the shipped function changing under it."""
    cols = _column_clearance(frame.depth_mm, frame.valid, WanderParams(rows=(2, 5)))
    return float(cols[3:5].min())


def run(seed: int, seconds: float, per_side: int, roles: str | None = None) -> dict:
    sc = make_pitch(per_side=per_side)
    if roles:
        # Static jobs, the same on both sides, in spawn order (as probe_threat):
        # `--roles defender,midfielder,striker` for 3v3.
        names = [r.strip() for r in roles.split(",")]
        assert len(names) == per_side, f"--roles needs {per_side} names, got {names}"
        for i, d in enumerate(sc.ducks):
            d.role = names[i % per_side]
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams: dict = {}
    brains: dict[str, Chase] = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams))
                                for d in sc.ducks}
    # The knobs this arm asked for, read off a brain that is actually running.
    live = {k: getattr(next(iter(brains.values())).p, k) for k in sorted(ChaseParams.env_names())}
    want = ChaseParams.from_env()
    for k, v in live.items():
        assert v == getattr(want, k), f"{k}: live brain has {v!r}, MICRODUCK_CHASE asked {getattr(want, k)!r}"
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    metrics = PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})
    spin = SpinMetrics(w)
    goal_seq = 0

    dt = None
    state_s: Counter = Counter()
    frozen_s: Counter = Counter()          # zero twist, by state
    yaw_sum = defaultdict(float)           # |head yaw| summed by state
    ticks = 0
    seen_ticks = 0
    losses: list[float] = []
    last_ball_t: dict[str, float] = {d.id: 0.0 for d in sc.ducks}
    last_frame_t: dict[str, float] = {d.id: -1.0 for d in sc.ducks}
    blind_since: dict[str, float | None] = {d.id: None for d in sc.ducks}
    tof_ticks = 0
    stop_bear = stop_col = false_stop = 0
    # The restarts (roadmap B.3): who kicked off, and whether the next goal
    # inside the window went to the side that had the ball.
    restarts: list[dict] = []
    # (ticks, old-rule stops, new-rule stops) per head-yaw bin
    bins = [[0, 0, 0] for _ in range(len(YAW_BINS) + 1)]

    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            b = brains[d.id]
            intent = b.step(s)
            spin.tick(d, intent.twist)
            vx, _, wz = intent.twist
            hy = abs(intent.head[2]) if intent.head is not None else 0.0
            state_s[b.state] += 1
            yaw_sum[b.state] += hy
            if vx == 0.0 and wz == 0.0:
                frozen_s[b.state] += 1
            ticks += 1
            # A sighting is a NEW detector frame carrying a ball. In view =
            # one arrived inside DET_MAX_AGE.
            if det is not None and det.t > last_frame_t[d.id]:
                last_frame_t[d.id] = det.t
                if any(x.cls == "ball" for x in det.detections):
                    if blind_since[d.id] is not None:
                        losses.append(round(w.t - blind_since[d.id], 2))
                        blind_since[d.id] = None
                    last_ball_t[d.id] = w.t
            if w.t - last_ball_t[d.id] <= Chase.DET_MAX_AGE:
                seen_ticks += 1
            elif blind_since[d.id] is None:
                blind_since[d.id] = last_ball_t[d.id] + Chase.DET_MAX_AGE
            # The ToF counterfactual, on the frame the brain just used.
            fr = s.fresh_tof(Chase.TOF_MAX_AGE)
            if fr is not None:
                a_bear = tof_clearance_bearings(fr)[0]
                a_col = _ahead_columns(fr)
                tof_ticks += 1
                sb = a_bear < b.p.tof_stop
                scol = a_col < b.p.tof_stop
                stop_bear += sb
                stop_col += scol
                false_stop += (scol and not sb)
                k = sum(hy > e for e in YAW_BINS)
                bins[k][0] += 1
                bins[k][1] += scol
                bins[k][2] += sb
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        metrics.tick()
        if dt is None and w.t > 0:
            dt = w.t
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            scorer = w.team_defending("left" if w.last_goal == "right" else "right")
            if restarts:
                restarts[-1]["nextGoalDt"] = round(w.t - restarts[-1]["t"], 1)
                restarts[-1]["nextGoalByKicker"] = (scorer == restarts[-1]["team"])
            restarts.append({"t": round(w.t, 1), "team": w.kickoff_team, "scorer": scorer})
            kickoff_brains(brains, teams, w)
    dt = dt or 0.02
    nd = len(sc.ducks)
    per_duck = dt / nd                     # ticks -> seconds a duck spends, averaged over the roster
    row = {
        "seed": seed, "perSide": per_side, "seconds": seconds, "simSeconds": round(w.t, 1),
        "live": {k: (v if not isinstance(v, float) else round(v, 4)) for k, v in live.items()},
        "stateS": {k: round(v * per_duck, 1) for k, v in state_s.most_common()},
        "frozenS": {k: round(v * per_duck, 1) for k, v in frozen_s.most_common()},
        "headYaw": {k: round(yaw_sum[k] / state_s[k], 3) for k in state_s},
        "viewFrac": round(seen_ticks / max(ticks, 1), 4),
        "losses": losses,
        "tofTicks": tof_ticks,
        "stopBearing": stop_bear, "stopColumn": stop_col, "falseStopColumn": false_stop,
        "restarts": restarts,
        "yawBins": bins,
        "kicks": {k: b.kicks for k, b in brains.items()},
        "falls": {k: d.falls for k, d in w.ducks.items()},
        **spin.row(), **metrics.row(),
    }
    sc_ = w.soccer_score()
    row.update(left=sc_["left"], right=sc_["right"], kickGoals=sc_["kicked"], bumpGoals=sc_["bumped"])
    return row


def _mean_team(rows: list[dict], field: str) -> float | None:
    vals = [v for r in rows if isinstance(r.get(field), dict) for v in r[field].values() if v is not None]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict], label: str) -> dict:
    n = len(rows)
    search = np.array([r["stateS"].get("search", 0.0) for r in rows])
    froz_search = np.array([r["frozenS"].get("search", 0.0) for r in rows])
    froz_all = np.array([sum(v for k, v in r["frozenS"].items() if k != "kick") for r in rows])
    view = np.array([r["viewFrac"] for r in rows])
    loss = np.array([x for r in rows for x in r["losses"]])
    kicks = np.array([sum(r["kicks"].values()) for r in rows])
    falls = np.array([sum(r["falls"].values()) for r in rows])
    goals = np.array([r["left"] + r["right"] for r in rows])
    poss = _mean_team(rows, "possession")
    prog = _mean_team(rows, "ballProgress")
    adv = _mean_team(rows, "ballAdvance")
    spinf = np.array([r["spinFrac"] for r in rows if r.get("spinFrac") is not None])
    tofn = sum(r["tofTicks"] for r in rows)
    rest = [x for r in rows for x in r.get("restarts", [])]
    followed = [x for x in rest if "nextGoalDt" in x and x["nextGoalDt"] <= 30.0]
    out = {
        "label": label, "seeds": n,
        "searchS": search.mean(), "searchSd": search.std(ddof=1) if n > 1 else 0.0,
        "frozenSearchS": froz_search.mean(), "frozenAllS": froz_all.mean(),
        "viewFrac": view.mean(),
        "losses": len(loss),
        "lossMedian": float(np.median(loss)) if len(loss) else float("nan"),
        "lossMean": float(loss.mean()) if len(loss) else float("nan"),
        "lossP90": float(np.percentile(loss, 90)) if len(loss) else float("nan"),
        "lossOver2s": float((loss > 2.0).mean()) if len(loss) else float("nan"),
        "kicks": kicks.mean(), "kicksTotal": int(kicks.sum()),
        "falls": falls.mean(), "fallsTotal": int(falls.sum()),
        "goals": goals.mean(), "goalsTotal": int(goals.sum()),
        "possession": poss, "ballProgress": prog, "ballAdvance": adv,
        "spinFrac": float(spinf.mean()) if len(spinf) else float("nan"),
        "tofTicks": tofn,
        "restarts": len(rest), "restartsFollowed30": len(followed),
        "restartKickerScored30": sum(1 for x in followed if x["nextGoalByKicker"]),
        "waitS": float(np.mean([r["stateS"].get("wait", 0.0) for r in rows])),
        "stopBearing": sum(r["stopBearing"] for r in rows),
        "stopColumn": sum(r["stopColumn"] for r in rows),
        "falseStopColumn": sum(r["falseStopColumn"] for r in rows),
        "yawBins": [[sum(r["yawBins"][i][j] for r in rows) for j in range(3)]
                    for i in range(len(YAW_BINS) + 1)],
    }
    return out


def report(s: dict) -> None:
    print(f"\n=== {s['label']}  ({s['seeds']} seeds) ===")
    print(f"  search        {s['searchS']:6.1f} s a duck a run  (sd {s['searchSd']:.1f})")
    print(f"  frozen IN search {s['frozenSearchS']:5.1f} s  = "
          f"{100 * s['frozenSearchS'] / max(s['searchS'], 1e-9):.0f}% of the search, "
          f"{100 * s['frozenSearchS'] / 300.0:.0f}% of a 300 s run")
    print(f"  frozen anywhere (not the kick window) {s['frozenAllS']:.1f} s")
    print(f"  ball in view  {100 * s['viewFrac']:.1f}% of ticks")
    print(f"  losses        {s['losses']} events; median {s['lossMedian']:.2f} s, "
          f"mean {s['lossMean']:.2f}, p90 {s['lossP90']:.2f}, {100 * s['lossOver2s']:.0f}% over 2 s")
    print(f"  kicks {s['kicks']:.2f}/run ({s['kicksTotal']})   falls {s['falls']:.2f} ({s['fallsTotal']})"
          f"   goals {s['goals']:.2f} ({s['goalsTotal']})")
    print(f"  possession {s['possession']:.2f} s/min   ballProgress {s['ballProgress']:.3f}"
          f"   ballAdvance {s['ballAdvance']:.3f}   spinFrac {s['spinFrac']:.3f}")
    print(f"  restarts {s['restarts']}; a goal within 30 s of one: {s['restartsFollowed30']}, "
          f"of which by the side that kicked off: {s['restartKickerScored30']}; "
          f"wait {s['waitS']:.1f} s a duck a run")
    print(f"  ToF: {s['tofTicks']} frames; the shipped bearing rule stops on {s['stopBearing']} "
          f"({100 * s['stopBearing'] / max(s['tofTicks'], 1):.1f}%), the old column rule on "
          f"{s['stopColumn']} ({100 * s['stopColumn'] / max(s['tofTicks'], 1):.1f}%), of which "
          f"{s['falseStopColumn']} are walls the bearing rule says are not ahead")
    edges = ["|yaw| <= 0.10"] + [f"{a:.2f} < |yaw| <= {b:.2f}" for a, b in zip(YAW_BINS, YAW_BINS[1:])] \
        + [f"|yaw| > {YAW_BINS[-1]:.2f}"]
    print(f"  {'head yaw bin':<20}{'frames':>9}{'old rule stops':>16}{'bearing stops':>15}")
    for name, (n, col, bear) in zip(edges, s["yawBins"]):
        if not n:
            continue
        print(f"  {name:<20}{n:>9}{col:>10} {100 * col / n:>4.1f}%{bear:>9} {100 * bear / n:>4.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first seed — a confirmation runs on seeds the effect was NOT found on")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--roles", default=None,
                    help="static jobs per side in spawn order, both sides alike, e.g. 'defender,midfielder,striker'")
    ap.add_argument("--out", default=None, help="write each run as a JSON line")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    label = args.label or (os.environ.get("MICRODUCK_CHASE", "") or "baseline")
    todo = [(s, args.seconds, args.per_side, args.roles) for s in range(args.seed0, args.seed0 + args.seeds)]
    rows: list[dict] = []
    if args.jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(min(args.jobs, len(todo))) as pool:
            rows = list(pool.starmap(run, todo))
    else:
        rows = [run(*a) for a in todo]
    rows.sort(key=lambda r: r["seed"])
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print(f"MICRODUCK_CHASE={os.environ.get('MICRODUCK_CHASE', '')!r}  "
          f"({args.seeds} seeds from {args.seed0} x {args.seconds:g} s of "
          f"{args.per_side}v{args.per_side})")
    print(f"live brain: {json.dumps(rows[0]['live'])}")
    report(summarize(rows, label))


if __name__ == "__main__":
    main()
