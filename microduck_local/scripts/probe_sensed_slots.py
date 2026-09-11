"""WHAT THE SENSED KICK ACTUALLY READS WHILE IT SWINGS (roadmap 12as follow-up H).

`behaviors/lastmetre.py` trained the sensed kicks with the TRUE ball projected
through the head camera at the recipe's own cadence and jitter. In play
(`world/arena.py::_sensed_head`) the same four slots are written off a
smoothed, associated, odometry-coasted `brain/tracker.py` TRACK, whose
placement carries ~5.5 cm of error at line-up range (`Tracker._place`'s own
note). Nobody had priced what that costs the kick, and it was the last
standing explanation on 12as's list for why a foot that reads the ball
perfectly on the bench reads it badly enough in play to be a nudge.

This measures the gap TICK BY TICK inside real kick windows, in the per-swing
gym (`scripts/kick_gym.py`, read-only — nothing here edits it):

  * the four slot values the kick RECEIVED (the track), and
  * the four the recipe's own projection of the TRUE ball would have written
    (`World.truth_update` / `World.truth_slots`, the same code the
    `MICRODUCK_SENSED_TRUTH=1` ablation runs, so the probe and the truth arm
    cannot describe different things), and
  * the exact geometry underneath both: the true ball in the duck's own
    odometry frame, so each estimate has an absolute error and not just a
    difference from the other.

Nothing on disk is patched: two `World` methods are wrapped in memory. The
truth projection is warmed EVERY tick (wrapping `_ball_tracker`, which the
world calls once per duck per tick immediately before the skill command is
built) because that is what training did — a memory that only ran during the
0.5 s swing would start every window blind and measure that instead.

    uv run python scripts/probe_sensed_slots.py --pair <dir with kick_*.onnx+json> \
        --seeds 4 --episodes 40 --jobs 3 --out runs/sensedplay/slots-right.json

The pair directory is pinned with MICRODUCK_SKILL_KICK_{LEFT,RIGHT}; pin only
the foot you mean to read by passing --left / --right instead of --pair. The
run PREFLIGHTS what it pinned off the constructed World (playbook rule 0, and
the snapshot trap in memory note shared-checkout-ab-on-copies): a pair that
silently fell back to the Hub kicks would report about the wrong policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from microduck_local.behaviors.lastmetre import LM_RANGE_SCALE  # noqa: E402
from microduck_local.world.arena import SENSED_BALL_CLS, World  # noqa: E402

TICKS: list[dict] = []
_INSTALLED = False


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def install() -> None:
    """Wrap the two World methods this probe reads. Idempotent, and safe to
    call inside a worker process (which is where it has to happen: a pool
    started with `spawn` re-imports this module and loses the parent's
    patches)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _ball_tracker, _sensed_head = World._ball_tracker, World._sensed_head

    def ball_tracker(self, d):
        # The world calls this once per duck per tick, right before
        # `_skill_cmd` builds the command block — exactly where training's
        # sensing ran. Skipped when the ablation is ON, which already steps it.
        if not self.sensed_truth:
            self.truth_update(d)
        return _ball_tracker(self, d)

    def sensed_head(self, d):
        _sensed_head(self, d)
        TICKS.append(_row(self, d))

    World._ball_tracker, World._sensed_head = ball_tracker, sensed_head


def _row(w: World, d) -> dict:
    """One tick of a sensed kick window: what the kick got, what the recipe's
    projection of the truth would have given it, and the truth itself."""
    hc = np.asarray(d.head_cmd, np.float64)
    truth = np.asarray(w.truth_slots(d), np.float64)
    x, y, yaw = w.odom(d)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    bx, by = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = bx - x, by - y
    ahead, beside = c * dx + s * dy, -s * dx + c * dy
    r = {
        "foot": str(d.skill), "t": round(w.t, 3),
        # the slots, as the network saw them
        "psi": hc[0], "rng": hc[1], "seen": hc[2], "conf": hc[3],
        "t_psi": truth[0], "t_rng": truth[1], "t_seen": truth[2], "t_conf": truth[3],
        # the truth, in the duck's own odometry frame
        "true_bear": math.atan2(beside, ahead), "true_rng": math.hypot(ahead, beside),
        "ball": [bx, by], "pose": [x, y, yaw],
    }
    trk = w._ball_trackers.get(d.id)
    tr = None if trk is None else trk.best(SENSED_BALL_CLS, w.t, min_hits=1)
    r["track_xy"] = None if tr is None or tr.xy is None else [float(tr.xy[0]), float(tr.xy[1])]
    r["track_age"] = None if tr is None else float(tr.age(w.t))
    st = w._truth_sense.get(d.id)
    r["truth_xy"] = None if st is None or st["world"] is None else [float(st["world"][0]),
                                                                    float(st["world"][1])]
    return r


def _work(args):
    seed, episodes, spread = args
    install()
    TICKS.clear()
    import kick_gym as kg  # noqa: PLC0415  (worker-local; keeps the parent import out of the pool)

    kg.run(seed, episodes, spread, 0, 0.0, "", 0.0, 0.0, 0.0, 0.0)
    for t in TICKS:
        t["seed"] = seed
    return list(TICKS)


# -- reading ----------------------------------------------------------------

def _q(xs, p):
    return float(np.percentile(np.asarray(xs, float), p)) if len(xs) else float("nan")


def _stat(xs, unit="", scale=1.0):
    xs = [x * scale for x in xs]
    if not xs:
        return "        —"
    return f"{np.median(xs):8.2f}{unit} (p90 {_q([abs(v) for v in xs], 90):6.2f}{unit}, n {len(xs)})"


def report(rows: list[dict]) -> None:
    by_foot = defaultdict(list)
    for r in rows:
        by_foot[r["foot"]].append(r)
    print(f"\nSENSED-KICK SLOT ERROR, {len(rows)} window ticks over "
          f"{len({r['seed'] for r in rows})} seeds\n" + "=" * 78)
    print("The TRACK is what the kick read; the TRUTH arm is the recipe's own projection of\n"
          "the true ball (the same code MICRODUCK_SENSED_TRUTH=1 runs). A slot unit of\n"
          f"bearing is pi/2 = 90 deg; a slot unit of range is LM_RANGE_SCALE = {LM_RANGE_SCALE} m.\n")
    for foot in sorted(by_foot):
        rs = by_foot[foot]
        live = [r for r in rs if r["conf"] > 0.0 or r["rng"] > 0.0]
        t_live = [r for r in rs if r["t_conf"] > 0.0 or r["t_rng"] > 0.0]
        both = [r for r in rs if r in live and r in t_live]
        print(f"-- {foot}  ({len(rs)} ticks) " + "-" * (52 - len(foot)))
        print(f"  ticks with a ball in the slots   track {len(live) / len(rs):5.1%}    "
              f"truth {len(t_live) / len(rs):5.1%}")
        print(f"  slot[53] seen                    track {np.mean([r['seen'] for r in rs]):5.1%}    "
              f"truth {np.mean([r['t_seen'] for r in rs]):5.1%}    "
              f"agree {np.mean([r['seen'] == r['t_seen'] for r in rs]):5.1%}")
        print(f"  slot[54] confidence  median      track {np.median([r['conf'] for r in rs]):5.2f}     "
              f"truth {np.median([r['t_conf'] for r in rs]):5.2f}")
        print(f"  track age at the tick            {_stat([r['track_age'] for r in rs if r['track_age'] is not None], ' s')}")
        if both:
            d_psi = [_wrap((r["psi"] - r["t_psi"]) * math.pi / 2) for r in both]
            d_rng = [r["rng"] - r["t_rng"] for r in both]
            clip = np.mean([r["rng"] >= 1.0 or r["t_rng"] >= 1.0 for r in both])
            print("  TRACK - TRUTH, slot by slot (the ablation's whole reach):")
            print(f"    bearing slot[51]  {_stat([abs(v) / (math.pi / 2) for v in d_psi])}")
            print(f"    bearing, degrees  {_stat([abs(math.degrees(v)) for v in d_psi], ' deg')}")
            print(f"    range slot[52]    {_stat([abs(v) for v in d_rng])}   "
                  f"(one or both clipped at 1.0 on {clip:.0%} of ticks)")
            print(f"    range, cm         {_stat([abs(v) for v in d_rng], ' cm', LM_RANGE_SCALE * 100)}")
        print("  ABSOLUTE error against the true ball in the duck's own frame:")
        for tag, bk, rk, xk in (("track", "psi", "rng", "track_xy"), ("truth", "t_psi", "t_rng", "truth_xy")):
            live_r = [r for r in rs if r[rk] > 0.0 or r[bk] != 0.0]
            if not live_r:
                print(f"    {tag:<6} —")
                continue
            be = [abs(math.degrees(_wrap(r[bk] * math.pi / 2 - r["true_bear"]))) for r in live_r]
            # range is compared UNCLIPPED where it can be: a slot at 1.0 only
            # says "0.25 m or more" and its error is a floor, not a number.
            unc = [r for r in live_r if r[rk] < 1.0]
            re_ = [abs(r[rk] * LM_RANGE_SCALE - r["true_rng"]) * 100 for r in unc]
            pl = [math.dist(r[xk], r["ball"]) * 100 for r in live_r if r.get(xk)]
            print(f"    {tag:<6} bearing {_stat(be, ' deg')}")
            print(f"           range   {_stat(re_, ' cm')}")
            print(f"           placement (xy vs the true ball) {_stat(pl, ' cm')}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", help="directory holding kick_left/right .onnx + .json; pins BOTH feet")
    ap.add_argument("--left", help="pin kick_left to this onnx (sidecar beside it)")
    ap.add_argument("--right", help="pin kick_right to this onnx")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--spread", type=float, default=0.8)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None, help="write the per-tick rows here as jsonl")
    a = ap.parse_args()
    if a.pair:
        for foot in ("left", "right"):
            os.environ[f"MICRODUCK_SKILL_KICK_{foot.upper()}"] = str(Path(a.pair) / f"kick_{foot}.onnx")
    for foot, path in (("left", a.left), ("right", a.right)):
        if path:
            os.environ[f"MICRODUCK_SKILL_KICK_{foot.upper()}"] = str(path)
    # PREFLIGHT, off the constructed World and not off the command line.
    from microduck_local.world import make_pitch  # noqa: PLC0415
    w = World(make_pitch(per_side=1), seed=0)
    for n in ("kick_left", "kick_right"):
        print(f"[preflight] {n}: {World.skill_path(n)}  sensed={w._skill_sensed[n]}  "
              f"exit={World.skill_sidecar(n).get('exit_rad')}")
    if not w._sensed_kick:
        raise SystemExit("neither foot is a SENSED kick — this probe would record nothing")

    args = [(s, a.episodes, a.spread) for s in range(a.seed0, a.seed0 + a.seeds)]
    rows: list[dict] = []
    if a.jobs > 1 and len(args) > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(_work, args):
                rows += r
    else:
        for x in args:
            rows += _work(x)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"[out] {len(rows)} ticks -> {a.out}")
    report(rows)


if __name__ == "__main__":
    main()
