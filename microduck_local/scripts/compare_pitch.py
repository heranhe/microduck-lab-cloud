"""Compare two `eval-pitch` / `eval-striker` batteries on the seeds they share.

    cd microduck_local
    uv run python scripts/compare_pitch.py runs/a.jsonl runs/b.jsonl
    uv run python scripts/compare_pitch.py runs/a.jsonl runs/b.jsonl --side home

Everything here is the playbook's reading rules made mechanical, because all
four have been got wrong in this repo at least once:

* **Paired, on the shared seeds only.** Both arms run the same layouts, so the
  per-seed DIFFERENCE is the measurement and comparing two means throws away
  most of the power. Seeds present in only one file are dropped and counted.
* **Student's t, not 1.96.** At the sizes these batteries run, the normal
  value manufactures significance; at n = 2 it understates the interval more
  than sixfold.
* **Events for the counts.** Goals, falls, own goals and kicks are counted,
  not averaged, and printed with their totals — a difference without its event
  count is not quotable.
* **`kicksBack` as a PROPORTION.** As a per-run mean it needs ~151 seeds to
  resolve a 25% shift; as a fraction of the arm's kick events (183 over 24
  seeds of 2v2) it needs about nine. Same measurement, two orders of cost —
  so it is tested here as two proportions, with a z on the pooled events.

`--side` reads ONE team of an asymmetric matchup (`home` = the side that
spawns at −x, `away` = the other) instead of pooling both, which is what an
arm that changes only one side's roster has to be read on. Team names come
off the row, so a file written before the teams were renamed still reads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Per-team dicts worth a paired reading, and how a row's teams combine into
# the number for that row: "sum" over the pair for a quantity both sides
# contribute to, "mean" for one that is already a per-team average.
FIELDS: tuple[tuple[str, str, str], ...] = (
    ("possession", "sum", "s/min"),
    ("ballAdvance", "sum", "m/min"),
    ("ballProgress", "sum", "m/min"),
    ("spread", "mean", "m"),
    ("crowd", "mean", ""),
    ("depth", "mean", "m"),
    ("ballOwnHalf", "mean", "s/min"),
)
COUNTS: tuple[tuple[str, str], ...] = (
    ("goals", "goals"), ("falls", "falls"), ("ownGoals", "own goals"),
    ("kickCount", "kicks"), ("kicksBack", "back-kicks"),
)


def load(path: str) -> dict[int, dict]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["seed"])] = r
    return rows


def teams_of(r: dict, side: str | None) -> list[str]:
    """The team keys to read from this row: both, or the one `--side` names.
    `home`/`away` are on the row for an eval-striker battery; otherwise the
    home team is the one whose sign is +1, i.e. the one attacking +x."""
    keys = sorted((r.get("possession") or r.get("ballAdvance") or {}))
    if side is None:
        return keys
    if r.get("home"):
        want = r["home"] if side == "home" else r.get("away")
        return [want] if want in keys else []
    return keys[:1] if side == "home" else keys[1:]


def value(r: dict, field: str, how: str, side: str | None) -> float | None:
    if field == "goals":
        # Pooled: the run's goals. Per side: the goals that TEAM scored, off
        # the ledger — never off the `left`/`right` mouth keys, which are the
        # inversion this benchmark carries a warning about.
        if side is None:
            return float(r["left"] + r["right"])
        gf = r.get("goalsFor") or {}
        return float(sum(gf[t] for t in teams_of(r, side) if t in gf)) if gf else None
    if field == "falls":
        if side is None:
            return float(sum(r["falls"].values()))
        of = r.get("team") or {}                       # duck -> team, on an eval-striker row
        want = set(teams_of(r, side))
        if not of or not want:
            return None
        return float(sum(v for k, v in r["falls"].items() if of.get(k) in want))
    v = r.get(field)
    if not v:
        return None
    vals = [v[t] for t in teams_of(r, side) if v.get(t) is not None]
    if not vals:
        return None
    return float(np.sum(vals) if how == "sum" else np.mean(vals))


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int]:
    """(mean difference b−a, its 95% half-width, p, seeds b beat a on)."""
    d = b - a
    n = len(d)
    if n < 2:
        return float(d.mean()) if n else 0.0, float("inf"), 1.0, int((d > 0).sum())
    sd = d.std(ddof=1)
    se = sd / math.sqrt(n) if sd else 0.0
    if se == 0:
        return float(d.mean()), 0.0, 1.0 if d.mean() == 0 else 0.0, int((d > 0).sum())
    t = d.mean() / se
    try:
        from scipy import stats  # noqa: PLC0415
        half, p = stats.t.ppf(0.975, n - 1) * se, 2 * stats.t.sf(abs(t), n - 1)
    except ImportError:
        # Student's t at 95%, by table, for the sizes these batteries run.
        tab = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 8: 2.36, 10: 2.26,
               12: 2.20, 16: 2.13, 20: 2.09, 24: 2.07, 30: 2.05, 60: 2.00}
        k = min(tab, key=lambda v: abs(v - (n - 1)))
        half = tab[k] * se
        p = math.erfc(abs(t) / math.sqrt(2))                       # normal, and say so
    return float(d.mean()), float(half), float(p), int((d > 0).sum())


def two_proportions(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """(difference in proportion, p) for two independent event counts."""
    if not n1 or not n2:
        return 0.0, 1.0
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p2 - p1, 1.0
    return p2 - p1, math.erfc(abs((p2 - p1) / se) / math.sqrt(2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="the baseline battery (.jsonl)")
    ap.add_argument("b", help="the arm to judge")
    ap.add_argument("--side", choices=("home", "away"), default=None,
                    help="read ONE side (an asymmetric roster A/B) instead of pooling both teams")
    ap.add_argument("--label", nargs=2, metavar=("A", "B"), default=None)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    seeds = sorted(set(A) & set(B))
    la, lb = args.label or (Path(args.a).stem, Path(args.b).stem)
    dropped = len(set(A) ^ set(B))
    side = f" · side {args.side}" if args.side else ""
    print(f"{lb} against {la}: {len(seeds)} shared seeds{side}"
          + (f" ({dropped} seed(s) in only one file, dropped)" if dropped else ""))
    print(f"\n{'metric':<14}{la[:11]:>11}{lb[:11]:>11}{'Δ':>9}{'95%':>9}{'p':>8}{'up':>6}")
    for field, how, unit in (("goals", "", ""), ("falls", "", "")) + FIELDS:
        how = how or "sum"
        va = [value(A[s], field, how, args.side) for s in seeds]
        vb = [value(B[s], field, how, args.side) for s in seeds]
        keep = [i for i, (x, y) in enumerate(zip(va, vb)) if x is not None and y is not None]
        if not keep:
            continue
        x, y = np.array([va[i] for i in keep]), np.array([vb[i] for i in keep])
        d, half, p, up = paired(x, y)
        star = "  ←" if p < 0.05 else ""
        print(f"{field:<14}{x.mean():>11.3f}{y.mean():>11.3f}{d:>+9.3f}{half:>9.3f}{p:>8.3f}"
              f"{up:>4}/{len(keep)}{star}   {unit}")
    print("\nevents (totals over the shared seeds):")
    for field, label in COUNTS:
        ta = sum(value(A[s], field, "sum", args.side) or 0 for s in seeds)
        tb = sum(value(B[s], field, "sum", args.side) or 0 for s in seeds)
        print(f"  {label:<12}{ta:>8.0f}{tb:>8.0f}")
    # The one that has to be read as a proportion, not as a per-run mean.
    ka = sum(value(A[s], "kickCount", "sum", args.side) or 0 for s in seeds)
    kb = sum(value(B[s], "kickCount", "sum", args.side) or 0 for s in seeds)
    ba = sum(value(A[s], "kicksBack", "sum", args.side) or 0 for s in seeds)
    bb = sum(value(B[s], "kicksBack", "sum", args.side) or 0 for s in seeds)
    if ka and kb:
        d, p = two_proportions(int(ba), int(ka), int(bb), int(kb))
        print(f"\nkicks sent back toward the kicker's own goal: {ba:.0f}/{ka:.0f} = {ba / ka:.0%}"
              f"  →  {bb:.0f}/{kb:.0f} = {bb / kb:.0%}   ({d:+.1%}, p = {p:.4f} on the events)")


if __name__ == "__main__":
    main()
