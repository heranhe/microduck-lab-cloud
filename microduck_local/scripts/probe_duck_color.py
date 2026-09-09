"""Does the colour vote survive CONTACT RANGE? — the duel's precondition.

The open-field duel is the largest remaining block of dead time (roadmap item
11c: 263 of 508 line-ups die to `avoid` in 0.4 s with an opponent 0.33 m
ahead), and every attempt to fix it by geometry has been a null, including
`lineup_keepout`. The reason it is parked (bead mdl-23b) is that a duck cannot
tell a teammate from an opponent close in: `_is_mate` reads `Track.color`, a
VOTE over frames, and `use_color` ships False because one frame of the
classifier is a coin at the hostile preset.

Nobody has measured the VOTE at the range the duel happens. That is what this
does, against the truth the sim detector hands out (`Track.name` is the real
object), over a real match:

    uv run python scripts/probe_duck_color.py --seeds 4 --per-side 3

Two numbers per range band, and they trade off: COVERAGE (how often the track
has voted a colour at all) and ACCURACY (of those, how often it is right).
`_is_mate` treats unknown as an opponent, so low coverage is safe and merely
wasteful, while low accuracy at contact range is what makes the duel
unbuildable — a duck that walks into a teammate it thinks is a stranger, or
shoulders an opponent it thinks is a friend.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.world import World, make_pitch

BANDS = ((0.0, 0.35), (0.35, 0.5), (0.5, 0.8), (0.8, 1.5), (1.5, 9.9))


def run(seed: int, seconds: float, per_side: int, detector: str | None) -> list[dict]:
    sc = make_pitch(per_side=per_side, formation=True)
    if detector:
        for d in sc.ducks:
            d.detector = detector
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed, ball_out_s=5.0)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    team_of = {d.id: d.team for d in sc.ducks}
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    rows: list[dict] = []
    goal_seq = 0
    while w.t < seconds:
        for did, b in brains.items():
            d = w.ducks[did]
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill,
                       bumped=w.bumped(d))
            it = b.step(s)
            w.apply_intent(d, it)
            if d.skill is None:
                d.set_cmd(w.data, it.twist, it.head)
            # Every LIVE duck track this brain holds, against the truth.
            for tr in b.tracker.tracks:
                if tr.cls != "duck" or tr.age(w.t) > 0.6:
                    continue
                truth = team_of.get(tr.name)
                if truth is None:
                    continue
                rows.append({"r": float(tr.range), "hits": int(tr.hits),
                             "voted": tr.color is not None,
                             "right": tr.color == truth,
                             "mate_truth": truth == team_of[did],
                             # The decision COUNTERFACTUALLY: `_is_mate` is
                             # gated on `use_color`, which ships False, so
                             # asking it would answer "no" every time and make
                             # the two error columns below meaningless. This is
                             # what it WOULD say with the sense switched on.
                             "said_mate": tr.color == team_of[did]})
        w.step()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
    return rows


def report(rows: list[dict]) -> None:
    if not rows:
        print("no duck tracks")
        return
    print(f"\n{len(rows)} live duck-track ticks against the truth\n")
    print(f"{'range band':<16}{'ticks':>8}{'voted a colour':>16}{'…and RIGHT':>13}{'of ALL ticks':>14}")
    for lo, hi in BANDS:
        b = [r for r in rows if lo <= r["r"] < hi]
        if not b:
            continue
        v = [r for r in b if r["voted"]]
        lab = f"{lo:.2f}-{hi:.2f} m" if hi < 9 else f">= {lo:.2f} m"
        acc = 100 * np.mean([r["right"] for r in v]) if v else float("nan")
        print(f"{lab:<16}{len(b):>8}{100 * len(v) / len(b):>15.0f}%{acc:>12.0f}%"
              f"{100 * np.mean([r['right'] for r in b]):>13.0f}%")
    close = [r for r in rows if r["r"] < 0.5]
    v = [r for r in close if r["voted"]]
    print(f"\nCONTACT RANGE (< 0.50 m): {len(close)} ticks, "
          f"{100 * len(v) / max(len(close), 1):.0f}% voted, "
          f"{100 * np.mean([r['right'] for r in v]) if v else float('nan'):.0f}% of those right")
    # The two errors the duel actually pays for.
    wrong_mate = [r for r in close if r["said_mate"] and not r["mate_truth"]]
    missed = [r for r in close if not r["said_mate"] and r["mate_truth"]]
    print(f"  called an OPPONENT a teammate : {len(wrong_mate)}  (the dangerous one)")
    print(f"  called a TEAMMATE a stranger  : {len(missed)}  (merely wasteful — unknown counts as opponent)")
    by_hits = defaultdict(list)
    for r in close:
        by_hits[min(r["hits"] // 5 * 5, 20)].append(r["right"])
    print("\n  accuracy at contact range by how many frames the vote rests on:")
    for k in sorted(by_hits):
        print(f"    {k:>3}+ hits  n={len(by_hits[k]):<6} {100 * np.mean(by_hits[k]):.0f}%")
    print("\nIf accuracy at contact range is near a coin, the duel stays parked and "
          "`use_color` stays off — that is the finding, not a failure.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--per-side", type=int, default=3)
    ap.add_argument("--detector", default=None, help="override the detector preset (e.g. hostile)")
    a = ap.parse_args()
    rows: list[dict] = []
    for s in range(a.seed0, a.seed0 + a.seeds):
        rows += run(s, a.seconds, a.per_side, a.detector)
    report(rows)
