"""Can a shot gate SEE the thing it would have to gate on? (roadmap 4b)

    cd microduck_local
    uv run python scripts/probe_shot_gate.py --seeds 24 --seconds 300

The ball's sideways offset at the swing IS the kick's aim error — measured
over 462 kicks, **+1.90 deg per cm**, and no rotation can remove it because
the kick spot is laid out in the body heading, so every rotation moves the
offset itself (`ChaseParams.kick_deflect_*`). That leaves one lever:
DECLINE the swing when the offset is bad (`kick_side_max`).

A gate is only as good as its estimate, and the brain's only fresh one is
`Chase.predicted` — the track's last position propagated by its own
velocity. The camera has already lost a floor ball by 0.35 m and the ball on
the kick spot sits 37 deg off the nose against a 31 deg half-field, so this
estimate is doing real work rather than reading a number off a sensor.

So: at every swing, the side offset the brain PREDICTS against the one the
ball ACTUALLY had. Read the correlation and the confusion table. A gate can
only pay if a wide prediction really means a wide shot.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.world import World, make_pitch


def run(seed: int, seconds: float = 300.0, per_side: int = 2) -> dict:
    """Per swing: (predicted side, true side, sigma). Per sampled tick with
    a live estimate (every 25th, a duck): (age of the hit, sigma, error of
    the prediction against the true ball) - the calibration set for
    `Track.sigma` (roadmap C.1)."""
    sc = make_pitch(per_side=per_side)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    goal_seq, out = 0, []
    ticks: list[tuple[float, float, float]] = []
    n_tick = 0
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill,
                       bumped=w.bumped(d))
            b = brains[d.id]
            it = b.step(s)
            n_tick += 1
            if b.predicted is not None and n_tick % 25 == 0:
                bx, by = w.ball_xy()
                tr = b.tracker.best(b.p.target_cls, w.t, min_hits=1)
                age = (w.t - tr.xy_t) if tr is not None and tr.xy is not None else 0.0
                ticks.append((age, b.predicted_sigma or 0.0, math.hypot(b.predicted[0] - bx, b.predicted[1] - by)))
            if it.skill in ("kick_left", "kick_right"):
                ox, oy, oyaw = w.odom(d)
                bx, by = w.ball_xy()
                true = -(bx - ox) * math.sin(oyaw) + (by - oy) * math.cos(oyaw)
                pred = None
                if b.predicted is not None:
                    px, py = b.predicted
                    pred = -(px - ox) * math.sin(oyaw) + (py - oy) * math.cos(oyaw)
                out.append((pred, true, b.predicted_sigma))
            w.apply_intent(d, it)
            if d.skill is None:
                d.set_cmd(w.data, it.twist, it.head)
        w.step()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
    return {"swings": out, "ticks": ticks}


def calibration(ticks: list, swings: list) -> None:
    """Does `Track.sigma` predict the estimate's error? (roadmap C.1) Per
    age of the last hit: the median sigma and error, and the share of
    errors inside 1 and 2 sigma (a calibrated Gaussian: 68% / 95%)."""
    if not ticks:
        print("no live estimates sampled\n")
        return
    a = np.array(ticks, float)
    print(f"THE BALL'S UNCERTAINTY: {len(a)} sampled estimates (every 25th tick a duck with one)")
    print(f"  {'hit age':<14}{'n':>7}{'median sigma':>14}{'median err':>12}{'in 1s':>8}{'in 2s':>8}{'r(sig,err)':>12}")
    edges = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]
    for lo, hi in edges:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum() < 5:
            continue
        sg, er = a[m, 1], a[m, 2]
        r = float(np.corrcoef(sg, er)[0, 1]) if sg.std() > 1e-9 else float("nan")
        print(f"  {f'{lo:.1f}-{hi:.1f} s':<14}{int(m.sum()):>7}{np.median(sg):>14.3f}{np.median(er):>12.3f}"
              f"{(er <= sg).mean():>8.0%}{(er <= 2 * sg).mean():>8.0%}{r:>12.2f}")
    sg, er = a[:, 1], a[:, 2]
    print(f"  {'all':<14}{len(a):>7}{np.median(sg):>14.3f}{np.median(er):>12.3f}"
          f"{(er <= sg).mean():>8.0%}{(er <= 2 * sg).mean():>8.0%}{float(np.corrcoef(sg, er)[0, 1]):>12.2f}")
    if swings:
        s = np.array([[x[2], abs(x[0] - x[1])] for x in swings], float)
        print(f"  at the swing ({len(s)}): median sigma {np.median(s[:, 0]):.3f}, median |side err| "
              f"{np.median(s[:, 1]):.3f}, in 1s {(s[:, 1] <= s[:, 0]).mean():.0%}, in 2s "
              f"{(s[:, 1] <= 2 * s[:, 0]).mean():.0%}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--thresholds", default="0.09,0.12")
    args = ap.parse_args()

    todo = [(s, args.seconds, args.per_side) for s in range(args.seed0, args.seed0 + args.seeds)]
    got: list = []
    ticks: list = []
    if args.jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(min(args.jobs, len(todo))) as pool:
            for r in pool.starmap(run, todo):
                got += r["swings"]
                ticks += r["ticks"]
    else:
        for a in todo:
            r = run(*a)
            got += r["swings"]
            ticks += r["ticks"]

    calibration(ticks, [(p, t, s) for p, t, s in got if p is not None and s is not None])
    have = [(p, t) for p, t, _ in got if p is not None]
    print(f"{len(got)} kicks over {args.seeds} seeds x {args.seconds:g} s of "
          f"{args.per_side}v{args.per_side}; the brain had a fresh estimate on "
          f"{len(have)} of them ({len(have) / max(1, len(got)):.0%}).\n")
    if not have:
        print("No fresh estimates: a gate on this signal can never fire.")
        return
    p = np.array([x[0] for x in have])
    t = np.array([x[1] for x in have])
    r = float(np.corrcoef(np.abs(p), np.abs(t))[0, 1])
    n = len(p)
    tv = r * math.sqrt((n - 2) / max(1e-9, 1 - r * r))
    print(f"|predicted side| vs |actual side|:  r = {r:+.3f}  "
          f"(n = {n}, p = {math.erfc(abs(tv) / math.sqrt(2)):.2g})")
    print(f"  predicted |side|   median {np.median(np.abs(p)):.3f} m")
    print(f"  actual    |side|   median {np.median(np.abs(t)):.3f} m")
    print(f"  the estimate's own error: median {np.median(np.abs(p - t)):.3f} m, "
          f"90th {np.percentile(np.abs(p - t), 90):.3f} m")
    for thr in (float(x) for x in args.thresholds.split(",")):
        ref, wide = np.abs(p) > thr, np.abs(t) > thr
        tp, fp = int((ref & wide).sum()), int((ref & ~wide).sum())
        fn = int((~ref & wide).sum())
        print(f"\n  as a gate at {thr:.2f} m: refuses {ref.mean():.0%} of swings, and "
              f"{tp / (tp + fp) if tp + fp else 0:.0%} of those really were wide")
        print(f"    it still lets through {fn} of the {tp + fn} genuinely wide shots "
              f"({fn / (tp + fn) if tp + fn else 0:.0%})")
    print("\nREAD IT AS: a gate can only pay if a wide PREDICTION means a wide SHOT. "
          "At r near zero\nit is refusing touches at random, which is what the "
          "`kick_side_max` battery measured.")


if __name__ == "__main__":
    main()
