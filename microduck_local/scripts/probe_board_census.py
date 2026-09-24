"""How often could the boards line ACT in a real match? — the deciding census.

The match-level `board_margin` arms came back a tight null on possession
(MDE 5%) and flat on kicks (MDE 18%), and solving the MDE backwards showed a
boards-confined effect of the size the gym measured would have had to fire on
20-39% of kick chances to be visible. Nobody has counted the actual rate. If it
is under ~25%, no affordable `eval-pitch` can settle the scoped claim and the
boards line belongs in the gym with this census printed beside it.

    uv run python scripts/probe_board_census.py --seeds 4 --per-side 2

**Counting the ACT set, not the trigger set.** `board_margin` is its own
feasibility limit: it switches to the boards line when the normal spot is not
clear of the boards, and then `_along_the_boards` must ALSO find a spot that
clears the same margin. Below `gap = margin - kick_side` no such spot exists, so
the branch runs and falls straight through to the shipped spot. A census of
"plans within `margin` of a board" would count those fall-throughs as reach and
report a large firing rate for a knob that is a structural no-op there — the
opposite of the truth. So every plan lands in exactly one of:

    act        the trigger fired AND `_along_the_boards` returned a spot
    wasted     the trigger fired and it returned None (a no-op that still ran)
    clear      the trigger did not fire (the normal spot was already fine)

and the denominator is EVERY kick plan, with `no plan` counted separately
rather than skipped (a rate whose denominator is defined by the outcome is
about the filter, not the physics).

**Corners are reported apart from the flat boards** because they are the only
asymmetry in the geometry: a corner ball is constrained by two boards at once,
so `_along_the_boards` fails there far more often. Pooling them averages a
0%-waste population with a ~47%-waste one and hides the thing worth seeing.

Counterfactual, and honestly so: the run is the SHIPPED brain (`board_margin`
0.0), and each candidate margin is evaluated by swapping the brain's own params
and calling `_clear_of_boards` / `_along_the_boards` — the real methods, never a
reimplementation of them. The ball position is the brain's own estimate, not the
world's truth, because that is what the planner actually steers by.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.world import World, make_pitch

MARGINS = (0.10, 0.15, 0.20, 0.25)
CORNER = 0.30            # within this of BOTH boards counts as a corner


def classify(b, bx: float, by: float, margin: float) -> str:
    """`act` / `wasted` / `clear`, by calling the brain's own methods with its
    params swapped to `margin`. Never reimplements the predicates."""
    spot = b.spot
    if spot is None:
        return "noplan"
    old = b.p
    try:
        b.p = replace(old, board_margin=margin)
        if b._clear_of_boards(float(spot[0]), float(spot[1])):
            return "clear"                       # trigger does not fire
        return "act" if b._along_the_boards(bx, by) is not None else "wasted"
    finally:
        b.p = old


def run(seed: int, seconds: float, per_side: int) -> dict:
    sc = make_pitch(per_side=per_side, formation=True)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed, ball_out_s=5.0)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    hx, hy = w.scenario.floor[0] / 2 - 0.25, w.scenario.floor[1] / 2 - 0.25
    tally: dict = defaultdict(int)
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

            tally["ticks"] += 1
            if b.spot is None or b.spot[4] != "kick":
                tally["noplan"] += 1
                continue
            ball = b.tracker.best("ball", w.t)
            if ball is None or ball.xy is None:
                tally["noplan"] += 1
                continue
            bx, by = ball.xy
            near_x, near_y = hx - abs(bx), hy - abs(by)
            where = "corner" if (near_x < CORNER and near_y < CORNER) else "flat"
            tally["plans"] += 1
            tally[f"plans/{where}"] += 1
            for m in MARGINS:
                tally[f"{m}/{where}/{classify(b, bx, by, m)}"] += 1
        w.step()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
    return tally


def report(t: dict) -> None:
    plans = t.get("plans", 0)
    print(f"\n{t.get('ticks', 0)} duck-ticks, {plans} kick plans, "
          f"{t.get('noplan', 0)} ticks with no kick plan")
    if not plans:
        print("!! NO KICK PLANS AT ALL. That is a broken measurement, not a 0% census:"
              "\n   check the roster has kickers and that the run is long enough.")
        return
    for where in ("flat", "corner"):
        n = t.get(f"plans/{where}", 0)
        print(f"\n  {where.upper()} boards — {n} plans ({100 * n / plans:.0f}% of all plans)")
        if not n:
            continue
        print(f"    {'margin':>7}{'act':>9}{'wasted':>9}{'clear':>9}   act as % of ALL plans")
        for m in MARGINS:
            a = t.get(f"{m}/{where}/act", 0)
            wst = t.get(f"{m}/{where}/wasted", 0)
            c = t.get(f"{m}/{where}/clear", 0)
            print(f"    {m:>7.2f}{100 * a / n:>8.0f}%{100 * wst / n:>8.0f}%{100 * c / n:>8.0f}%"
                  f"{100 * a / plans:>16.1f}%")
    print("\nACT as a share of ALL kick plans is the number the match arms needed:"
          "\na gym-sized effect had to fire on 20-39% of kick chances to clear their"
          "\nkick MDE. `wasted` is the trigger firing into a fall-through — a no-op"
          "\nthat a trigger-set census would have counted as reach.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--per-side", type=int, default=2)
    a = ap.parse_args()
    total: dict = defaultdict(int)
    for s in range(a.seeds):
        for k, v in run(s, a.seconds, a.per_side).items():
            total[k] += v
        print(f"  seed {s} done ({total.get('plans', 0)} plans)")
    report(total)


if __name__ == "__main__":
    main()
