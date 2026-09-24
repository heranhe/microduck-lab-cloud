"""What could each A/B battery on disk ever have resolved?

    cd microduck_local
    uv run python scripts/audit_power.py
    uv run python scripts/audit_power.py --target-pct 10 --metric kickCount

Written after every knob in a run of soccer experiments came back "null" and
the obvious question was whether the knobs were dull or the instrument was.
It is the instrument.  Reading all thirteen A/B batteries on disk:

* **The minimum detectable effect is huge.** Median MDE is 28% of baseline on
  kicks, 19% on ball advance, 33% on falls.  A real 10% improvement cannot be
  seen at these sizes, so it reads as a null.  `goals` and `ballProgress` are
  worse than useless -- hundreds to tens of thousands of seeds for 10%.
* **The paired design buys nothing.** Median between-arm correlation r = 0.05,
  variance reduction 1.03x.  The sim diverges within seconds of any knob that
  fires, so the shared seed cancels no variance.  The one battery where it did
  pay (`t9 hunt`, r = 0.6) is the one whose knob barely fired -- pairing gain
  is a measure of how little the arm perturbed the run, not of a good design.
* **`ballProgress` is noise**: ~21,000 seeds for a 10% change.

`--old-normal` also reports the verdicts that were significant only under the
normal approximation the script used before Student's t was written out (scipy
is not a dependency here, so the normal fallback is what actually ran).  At 24
seeds the untrustworthy band is a reported p between 0.039 and 0.05.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from compare_pitch import (
    COUNTS,
    FIELDS,
    UNQUOTABLE,
    load,
    paired,
    pairing_gain,
    seeds_for,
    t_ppf975,
    value,
    verdict,
)

RUNS = Path(__file__).resolve().parents[1] / "runs"

# The A/B batteries on disk, as (label, baseline, arm).  A pair whose files are
# gone is skipped, so this list is safe to keep as history.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("t4 clamp", "t4-base24-2v2", "t4-clamp24-2v2"),
    ("t4 goal", "t4-base24-2v2", "t4-goal24-2v2"),
    ("t5 roles D", "t5-plainD", "t5-rolesD"),
    ("t5 roles F", "t5-plainF", "t5-rolesF"),
    ("t6 roles3 D", "t6-plain3D", "t6-roles3D"),
    ("t6 roles3 F", "t6-plain3F", "t6-roles3F"),
    ("t7 colour D", "t7-base3", "t7-color3"),
    ("t7 colour F", "t7-base3F", "t7-color3F"),
    ("t8 comp", "t8-base", "t8-comp"),
    ("t9 hunt D", "t9-nohunt", "t9-hunt"),
    ("t9 hunt F", "t9-nohuntF", "t9-huntF"),
    ("led2 i8", "led2-base24", "led2-i8-24"),
    ("thr i8", "thr-base48b", "thr-i8-48"),
)

METRICS: tuple[tuple[str, str], ...] = (
    (("goals", "sum"), ("falls", "sum"))
    + tuple((f, how) for f, how, _ in FIELDS)
    + tuple((f, "sum") for f, _ in COUNTS if f == "kickCount")
)


def normal_p(t: float) -> float:
    """The two-sided p the old fallback produced: the NORMAL, not Student's."""
    return math.erfc(abs(t) / math.sqrt(2))


def series(rows: dict, seeds: list[int], field: str, how: str) -> np.ndarray | None:
    try:
        v = [value(rows[s], field, how, None) for s in seeds]
    except KeyError:
        return None
    return None if any(x is None for x in v) else np.asarray(v, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-pct", type=float, default=10.0,
                    help="the effect size to size batteries for (default: %(default)s%%)")
    ap.add_argument("--metric", default=None, help="only this metric")
    ap.add_argument("--old-normal", action="store_true",
                    help="also flag verdicts that held only under the old normal p")
    args = ap.parse_args()

    rows: list[tuple] = []
    flips: list[tuple[str, str, float, float]] = []
    for label, fa, fb in PAIRS:
        pa, pb = RUNS / f"{fa}.jsonl", RUNS / f"{fb}.jsonl"
        if not (pa.exists() and pb.exists()):
            continue
        A, B = load(str(pa)), load(str(pb))
        seeds = sorted(set(A) & set(B))
        if len(seeds) < 3:
            continue
        for field, how in METRICS:
            if args.metric and field != args.metric:
                continue
            x, y = series(A, seeds, field, how), series(B, seeds, field, how)
            if x is None or y is None or x.mean() == 0 or (y - x).std(ddof=1) == 0:
                continue
            d, half, p, _ = paired(x, y)
            base = abs(x.mean())
            pct = 100.0 * half / base
            n = len(seeds)
            rows.append((label, field, n, float(np.corrcoef(x, y)[0, 1]),
                         pairing_gain(x, y), pct, verdict(p, pct), p,
                         seeds_for(half, n, base, args.target_pct)))
            # Only significant under the normal?  se comes back through half.
            se = half / t_ppf975(n - 1)
            if se > 0 and p >= 0.05 > normal_p(d / se):
                flips.append((label, field, normal_p(d / se), p))

    print(f"{'battery':<13}{'metric':<13}{'n':>3}{'r':>7}{'pair':>7}{'MDE%':>7}"
          f"{'p':>7}  {'verdict':<10}{'n for ' + str(int(args.target_pct)) + '%':>12}")
    print("-" * 82)
    for r in rows:
        note = "  (unreachable)" if r[8] > 400 else ""
        tag = "unquotable" if r[1] in UNQUOTABLE else r[6]
        print(f"{r[0]:<13}{r[1]:<13}{r[2]:>3}{r[3]:>+7.2f}{r[4]:>6.2f}x{r[5]:>6.0f}%"
              f"{r[7]:>7.3f}  {tag:<10}{r[8]:>12}{note}")

    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r[1]].append(r)
    print(f"\n{'metric':<14}{'med r':>7}{'med pair':>10}{'med MDE%':>10}"
          f"{'med n for ' + str(int(args.target_pct)) + '%':>16}   reading")
    print("-" * 82)
    for k, v in sorted(by.items(), key=lambda kv: -float(np.median([x[5] for x in kv[1]]))):
        mde = float(np.median([x[5] for x in v]))
        need = int(np.median([x[8] for x in v]))
        read = ("noise -- do not quote" if k in UNQUOTABLE else
                "usable" if mde <= 15.0 else "too blunt for a null")
        print(f"{k:<14}{np.median([x[3] for x in v]):>+7.2f}"
              f"{np.median([x[4] for x in v]):>9.2f}x{mde:>9.0f}%{need:>16}   {read}")

    gains = [r[4] for r in rows]
    print(f"\nthe shared seeds bought a {np.median(gains):.2f}x variance reduction "
          f"(median over {len(gains)} readings).\n1.0 means the pairing is decorative: "
          "the two arms diverge and share nothing but the layout.")

    if args.old_normal:
        print(f"\n{len(flips)} reading(s) were significant ONLY under the old normal p:")
        for label, field, pn, pt in flips:
            print(f"  {label:<13}{field:<14}normal p = {pn:.3f}  →  Student's t p = {pt:.3f}")


if __name__ == "__main__":
    main()
