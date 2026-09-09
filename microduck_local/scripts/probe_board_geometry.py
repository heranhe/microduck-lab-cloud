"""Where `board_margin` can act, measured on the brain's own geometry.

No simulation: this calls `Chase._along_the_boards` and `_clear_of_boards`
directly, so it answers "could the knob ever act here?" for the price of a
few thousand trig calls, where a battery answers "did it pay?" for 20
minutes and cannot separate the two.

Two questions, and only the first is assumption-free:

1. FEASIBILITY — does a legal along-the-boards spot exist for a ball `gap`
   off a board?  `_along_the_boards` picks its own line, so no aim model
   enters.  The answer is a clean step: a spot exists iff

       gap >= board_margin - kick_side          (kick_side = 0.06 default)

   This is the "feasibility law" item 12q fitted as `margin <= gap + 0.05`.
   The constant is not 0.05 and is not empirical: it is `kick_side`.

2. WHERE THE KNOB CHANGES ANYTHING — the branch in `_hold_target` runs only
   when the DEFAULT spot is inside the margin, and then only helps if a legal
   alternative exists.  Both conditions use the same `board_margin`, so
   raising it makes the knob fire more AND fail more.  Reporting the aim
   sweep needs an aim model; this uses a uniform sweep over `u` and both
   feet, which is an ASSUMPTION and is labelled as one in the output.

The result that matters (2026-09-09): at `board_margin=0.25` the knob cannot
act below gap 0.20 at all — it triggers and falls through 100% of the time,
a no-op that still runs the branch.  So an arm at 0.25 cannot reach the
sub-0.20 m region where 12p's swing-rate cliff lives, and its null says
nothing about the idea.  At 0.10 the knob does act across gap 0.04-0.16,
which is the cliff, and item 12t's match arm still measured nothing — so
"no legal spot exists" is not what the cliff is made of.

    uv run python scripts/probe_board_geometry.py
    uv run python scripts/probe_board_geometry.py --bounds 1.5 1.25 --margins 0.10 0.25
"""
from __future__ import annotations

import argparse
import math

from microduck_local.brain.controllers import Chase, ChaseParams, _wrap

GAPS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.60)


class _Geom:
    """Only what the two geometry methods touch: `bounds`, `p`, `goal`."""

    _clear_of_boards = Chase._clear_of_boards
    _along_the_boards = Chase._along_the_boards


def _default_spot(p: ChaseParams, bx: float, by: float, u: float, foot: str):
    """The spot `_hold_target` would use without the knob (same arithmetic)."""
    side = -p.kick_side if foot == "kick_left" else p.kick_side
    h = _wrap(u - (p.kick_deflect_left if foot == "kick_left" else p.kick_deflect_right))
    return (bx - p.kick_ahead * math.cos(h) - side * math.sin(h),
            by - p.kick_ahead * math.sin(h) + side * math.cos(h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bounds", nargs=2, type=float, default=(1.5, 1.25),
                    metavar=("HX", "HY"), help="pitch half-extents (default: the gym's)")
    ap.add_argument("--margins", nargs="*", type=float,
                    default=(0.10, 0.15, 0.20, 0.25, 0.30))
    ap.add_argument("--aim-steps", type=int, default=36,
                    help="aim directions in the uniform sweep for question 2")
    a = ap.parse_args()

    bounds = (a.bounds[0], a.bounds[1])
    g = _Geom()
    g.bounds = bounds
    g.goal = (bounds[0], 0.0)
    p0 = ChaseParams()
    xs = [i * 0.1 - 1.0 for i in range(21)]

    print(f"pitch half-extents {bounds}   kick_ahead={p0.kick_ahead}  kick_side={p0.kick_side}")
    print(f"shipped board_margin default = {p0.board_margin}\n")

    print("1. FEASIBILITY — % of ball positions with a legal along-the-boards spot")
    print("   (assumption-free: _along_the_boards picks its own line)\n")
    head = "   gap(m) " + "".join(f"{m:>9.2f}" for m in a.margins)
    print(head + "\n   " + "-" * (len(head) - 3))
    for gap in GAPS:
        by = bounds[1] - gap
        row = f"   {gap:<7.3f}"
        for m in a.margins:
            g.p = ChaseParams(board_margin=m)
            ok = sum(g._along_the_boards(bx, by) is not None for bx in xs)
            row += f"{100 * ok / len(xs):>8.0f}%"
        print(row)
    print(f"\n   the step sits at gap = margin - kick_side ({p0.kick_side})\n")

    print("2. WHERE THE KNOB CHANGES ANYTHING — trigger fires AND a legal spot exists")
    print("   ASSUMES a uniform sweep over aim direction and both feet.\n")
    print("   " + f"{'gap(m)':>7} | " + " | ".join(f"m={m:.2f} act/wasted" for m in a.margins))
    for gap in GAPS:
        by = bounds[1] - gap
        cells = []
        for m in a.margins:
            p = ChaseParams(board_margin=m)
            g.p = p
            act = waste = tot = 0
            for bx in xs:
                for k in range(a.aim_steps):
                    u = k * 2.0 * math.pi / a.aim_steps
                    for foot in ("kick_left", "kick_right"):
                        tot += 1
                        sx, sy = _default_spot(p, bx, by, u, foot)
                        if not g._clear_of_boards(sx, sy):
                            if g._along_the_boards(bx, by) is not None:
                                act += 1
                            else:
                                waste += 1
            cells.append(f"{100 * act / tot:5.1f}%/{100 * waste / tot:5.1f}%")
        print(f"   {gap:>7.3f} | " + " | ".join(cells))
    print("\n   act    = the knob re-aims along the boards")
    print("   wasted = the branch runs, no legal spot, falls through to the default")


if __name__ == "__main__":
    main()
