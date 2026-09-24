"""`probe_gaze`: is the duck LOOKING at the ball on the way in, and for how
long is it blind before it swings?

    cd microduck_local
    uv run python scripts/probe_gaze.py --seeds 24 --seconds 300 --per-side 2 --jobs 8
    MICRODUCK_CHASE="head_down=1.2" uv run python scripts/probe_gaze.py ... --tag deep

`probe_kick_line.py` says WHERE the ball was when the swing fired (a median
0.24 m ahead of a trunk whose sweet spot is 0.06-0.10) and that the plan the
swing stands on is 3.3 s old. This says WHY: how long the duck has not seen
the ball when it kicks, how far it walked in that state, and what its head was
doing meanwhile.

Everything here is read off the REAL detector (`d.detector.last`), not off the
brain's tracker — the question is what the camera could physically see, and a
track that is still alive on a 2 s memory is not a sighting.

Three numbers to read:
  * `blind at the swing` — seconds since the detector last reported the ball,
    and metres walked since. This is the stale part of the plan, measured.
  * `ball under the feet, unseen` — the fraction of the run where the ball is
    truly inside 0.40 m and the detector is NOT reporting it. The owner's
    "they walk around looking even when the ball is under their feet".
  * `head down` — the fraction of the run-up with a non-zero head-pitch
    command, and the mean command in the last second before the swing. The
    gaze is dropped whenever the duck stops (`vx > 0` gates it), so this is
    where "they keep scaling back up" shows up as a number.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.world import World, make_pitch

WINDOW = 3.5      # s of run-up summarised per kick
NEAR = 0.40       # "under the feet": true ground distance trunk -> ball


def run(seed: int, seconds: float, per_side: int) -> dict:
    sc = make_pitch(per_side=per_side)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    # Rule 10 / rule 0: read the knob off the brain that is RUNNING.
    any_b = next(iter(brains.values()))
    knobs = {"head_down": any_b.p.head_down, "refresh_min": any_b.p.refresh_min,
             "gaze_still": getattr(any_b.p, "gaze_still", None),
             "gaze_neck": getattr(any_b.p, "gaze_neck", None)}
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    last_sight: dict[str, float] = {}          # duck -> sim t of the last frame with a ball in it
    sight_rng: dict[str, float] = {}           # ...and the TRUE trunk->ball distance then
    last_frame_t: dict[str, float] = {}
    hist: dict[str, list] = {d: [] for d in brains}    # (t, head_pitch_cmd, vx)
    kicks: list[dict] = []
    steps = near_unseen = near_total = 0
    goal_seq = 0
    while w.t < seconds:
        bx, by = w.ball_xy()
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            if det is not None and last_frame_t.get(d.id) != det.t:
                last_frame_t[d.id] = det.t
                if any(x.cls == "ball" for x in det.detections):
                    last_sight[d.id] = det.t
                    tp = d.trunk_pos(w.data)
                    sight_rng[d.id] = math.hypot(bx - tp[0], by - tp[1])
            b = brains[d.id]
            intent = b.step(s)
            ox, oy, _ = w.odom(d)
            pos = d.trunk_pos(w.data)
            true_rng = math.hypot(bx - pos[0], by - pos[1])
            h = hist[d.id]
            h.append((w.t, float(intent.head[1]), float(intent.twist[0]), ox, oy, b.state))
            while h and w.t - h[0][0] > WINDOW:
                h.pop(0)
            steps += 1
            if true_rng < NEAR:
                near_total += 1
                fresh = (last_sight.get(d.id) is not None and w.t - last_sight[d.id] < 0.25)
                near_unseen += 0 if fresh else 1
            if intent.skill in ("kick_left", "kick_right"):
                ls = last_sight.get(d.id)
                walked = 0.0
                if ls is not None:
                    pts = [(x, y) for (tt, _, _, x, y, _) in h if tt >= ls]
                    walked = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
                run_up = [r for r in h]
                kicks.append({
                    "seed": seed, "duck": d.id, "t": round(w.t, 1),
                    "blind_s": None if ls is None else round(w.t - ls, 2),
                    "blind_m": None if ls is None else round(walked, 3),
                    # How far away the ball TRULY was when it was last seen —
                    # the range at which this duck went blind.
                    "sight_rng": None if ls is None else round(sight_rng.get(d.id, float("nan")), 3),
                    "head_now": round(float(intent.head[1]), 3),
                    "head_last_s": round(float(np.mean([hh for (tt, hh, _, _, _, _) in run_up
                                                        if w.t - tt <= 1.0] or [0.0])), 3),
                    "head_up_frac": round(float(np.mean([hh <= 0.02 for (_, hh, _, _, _, _) in run_up]
                                                        or [1.0])), 3),
                    "still_frac": round(float(np.mean([vv <= 0.02 for (_, _, vv, _, _, _) in run_up]
                                                      or [0.0])), 3),
                    "ahead": round((bx - pos[0]) * math.cos(d.yaw(w.data))
                                   + (by - pos[1]) * math.sin(d.yaw(w.data)), 3),
                    "settle_frac": round(float(np.mean([st in ("settle", "lineup")
                                                        for (_, _, _, _, _, st) in run_up] or [0.0])), 3),
                })
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)          # with the world: the kickoff rule the benchmark plays under
            last_sight.clear()
    return {"kicks": kicks, "steps": steps, "near_total": near_total,
            "near_unseen": near_unseen, "knobs": knobs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tag", default="shipped")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    todo = [(s, args.seconds, args.per_side) for s in range(args.seed0, args.seed0 + args.seeds)]
    res: list[dict] = []
    if args.jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(min(args.jobs, len(todo))) as pool:
            res = pool.starmap(run, todo)
    else:
        res = [run(*a) for a in todo]
    kicks = [k for r in res for k in r["kicks"]]
    steps = sum(r["steps"] for r in res)
    near_total = sum(r["near_total"] for r in res)
    near_unseen = sum(r["near_unseen"] for r in res)
    knobs = res[0]["knobs"]
    print(f"tag {args.tag!r}  MICRODUCK_CHASE={os.environ.get('MICRODUCK_CHASE', '')!r}")
    print("  the brain that RAN: " + ", ".join(f"{k}={v}" for k, v in knobs.items()))
    print(f"  {len(kicks)} kicks over {args.seeds} seeds (from {args.seed0}) x "
          f"{args.seconds:g} s of {args.per_side}v{args.per_side}, {steps} duck-steps\n")
    if args.out:
        with open(args.out, "w") as fh:
            for k in kicks:
                fh.write(json.dumps(k) + "\n")

    def col(name, default=None):
        return np.array([k[name] for k in kicks if k.get(name) is not None], dtype=float)

    bs, bm, sr = col("blind_s"), col("blind_m"), col("sight_rng")
    print(f"  blind at the swing:  {np.median(bs):.2f} s median (mean {bs.mean():.2f}, "
          f"IQR {np.percentile(bs, 25):.2f}-{np.percentile(bs, 75):.2f}), "
          f"{np.median(bm):.3f} m walked (mean {bm.mean():.3f})")
    print(f"     never seen at all in the {WINDOW:g} s run-up: "
          f"{sum(1 for k in kicks if k['blind_s'] is None or k['blind_s'] > WINDOW)} of {len(kicks)}")
    print(f"     ball's TRUE range when it was last seen: median {np.median(sr):.3f} m "
          f"(IQR {np.percentile(sr, 25):.3f}-{np.percentile(sr, 75):.3f})")
    hn, hl, hu, sf = col("head_now"), col("head_last_s"), col("head_up_frac"), col("still_frac")
    print(f"  head pitch command:  at the swing {np.median(hn):.3f}, mean over the last second "
          f"{hl.mean():.3f}; the run-up is head-UP {hu.mean():.0%} of the time "
          f"and standing still {sf.mean():.0%}")
    print(f"  ball under the feet (<{NEAR:g} m, true) and NOT being detected: "
          f"{near_unseen / max(near_total, 1):.0%} of {near_total} duck-steps "
          f"({near_total / max(steps, 1):.0%} of the run is spent that close)")
    ah = col("ahead")
    print(f"  ball ahead of the trunk at the swing: median {np.median(ah):.3f} m "
          f"(the sweet spot is 0.06-0.10)")
    print(f"  the run-up is lineup/settle {col('settle_frac').mean():.0%} of the time")


if __name__ == "__main__":
    main()
