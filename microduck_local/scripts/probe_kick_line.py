"""Does the chase brain know which way its own kick will send the ball?

    cd microduck_local
    uv run python scripts/probe_kick_line.py --seeds 24 --seconds 300 --per-side 2

The brain lays its kick spot on a line `u` — the direction it INTENDS the ball
to go — and then, after the swing, hunts along that same `u` to find the ball
again. Both assume the ball leaves along the body heading. The kick map says
it does not: measured on a standing duck, a ball on the left foot's sweet spot
leaves at **+21.6°** to the body heading and one on the right at **−11°**
(`ChaseParams.kick_deflect_*`, which ship at 0 — the compensation was measured
off in the first form, on 8 seeds, on goals, before anything could see where a
kick actually went).

This measures it in PLAY rather than on a bench: for every kick, the intended
line, the direction the ball actually travelled over the next `CARRY_S`, and
the angle between them — split by foot, so a systematic bias separates from
the scatter. A bias is worth removing even when it is smaller than the noise,
because it lands on every kick of that foot in the same direction; scatter
does not.

Prints the per-foot mean and median error, the fraction of kicks that go the
wrong side of the intended line, and what the shipped map would predict.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.world import World, make_pitch
from microduck_local.world.metrics import CARRY_S

MAP_DEG = {"kick_left": 21.6, "kick_right": -11.0}   # the bench measurement, for comparison


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def run(seed: int, seconds: float, per_side: int) -> list[dict]:
    sc = make_pitch(per_side=per_side)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    pending: list[dict] = []
    out: list[dict] = []
    goal_seq = 0
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            b = brains[d.id]
            intent = b.step(s)
            # The brain clears its spot in the same tick it fires, so the
            # intended line is read from `_hunt_u` — which IS that line, and
            # is also what the duck will walk along afterwards to find the
            # ball again. Both halves of "which way is it going" in one
            # number.
            if intent.skill in MAP_DEG and b._hunt_u is not None:
                pending.append({"t": w.t, "duck": d.id, "foot": intent.skill,
                                "u": b._hunt_u, "heading": w.odom(d)[2], "ball0": w.ball_xy()})
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams)
            pending = [k for k in pending if False]        # the ball teleported: nothing to settle
        keep = []
        for k in pending:
            if w.t - k["t"] < CARRY_S:
                keep.append(k)
                continue
            bx, by = w.ball_xy()
            dx, dy = bx - k["ball0"][0], by - k["ball0"][1]
            dist = math.hypot(dx, dy)
            if dist < 0.10:                    # the swing missed: no line to speak of
                out.append({**k, "dist": dist, "went": None, "err": None, "off_heading": None, "ball0": None})
                continue
            went = math.atan2(dy, dx)
            out.append({"t": round(k["t"], 1), "duck": k["duck"], "foot": k["foot"],
                        "dist": round(dist, 3),
                        "err": round(math.degrees(wrap(went - k["u"])), 2),
                        "off_heading": round(math.degrees(wrap(went - k["heading"])), 2)})
        pending = keep
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=None, help="write every kick as a JSON line")
    args = ap.parse_args()
    todo = [(s, args.seconds, args.per_side) for s in range(args.seeds)]
    rows: list[dict] = []
    if args.jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(min(args.jobs, len(todo))) as pool:
            for r in pool.starmap(run, todo):
                rows += r
    else:
        for a in todo:
            rows += run(*a)
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    hit = [r for r in rows if r.get("err") is not None]
    print(f"{len(rows)} kicks over {args.seeds} seeds x {args.seconds:g} s of "
          f"{args.per_side}v{args.per_side}; {len(rows) - len(hit)} moved the ball < 10 cm (a whiff)\n")
    print(f"{'foot':<12}{'n':>5}{'mean err':>10}{'median':>9}{'sd':>8}{'wrong side':>12}"
          f"{'the map says':>14}{'mean off heading':>18}")
    for foot in ("kick_left", "kick_right"):
        e = np.array([r["err"] for r in hit if r["foot"] == foot])
        h = np.array([r["off_heading"] for r in hit if r["foot"] == foot])
        if not len(e):
            continue
        print(f"{foot:<12}{len(e):>5}{e.mean():>+10.1f}{np.median(e):>+9.1f}{e.std(ddof=1):>8.1f}"
              f"{(np.sign(e) != np.sign(e.mean())).mean():>11.0%}"
              f"{MAP_DEG[foot]:>+14.1f}{h.mean():>+18.1f}")
    e = np.array([r["err"] for r in hit])
    print(f"\nall kicks: mean {e.mean():+.1f}°, mean |error| {np.abs(e).mean():.1f}°, "
          f"sd {e.std(ddof=1):.1f}°")
    print("READ IT AS: 'mean err' is the SYSTEMATIC part — it lands on every kick of that foot "
          "the same way and can be designed out.\n'sd' is the scatter, which cannot. "
          "'mean off heading' is the same error measured against the BODY, which is what the "
          "bench map measured.")


if __name__ == "__main__":
    main()
