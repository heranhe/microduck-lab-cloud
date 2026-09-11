"""Compare two `eval-pitch` / `eval-striker` batteries on the seeds they share.

    cd microduck_local
    uv run python scripts/compare_pitch.py runs/a.jsonl runs/b.jsonl
    uv run python scripts/compare_pitch.py runs/a.jsonl runs/b.jsonl --side home

Everything here is the playbook's reading rules made mechanical, because all
four have been got wrong in this repo at least once:

* **Paired, on the shared seeds only.** Both arms run the same layouts, so the
  per-seed difference is the measurement. It is kept because it cannot hurt and
  because a rarely-firing knob does keep its pairing (`t9 hunt`, r = 0.6) — but
  do NOT expect it to buy power. Measured over all thirteen A/B batteries on
  disk, the median between-arm correlation is r = 0.05 and the variance
  reduction is 1.03x. The sim diverges within seconds of any knob that fires,
  so by 300 s the two arms are effectively independent runs. The observed gain
  is printed per metric; when it is ~1.0 the seeds bought nothing.
  Seeds present in only one file are dropped and counted.

* **A null needs its MDE or it is not a null.** Every "measured off" verdict in
  this repo was reported as a p-value alone, and a p-value alone cannot tell
  "no effect" from "no instrument". At the 24 seeds these batteries run, the
  minimum detectable effect on kicks is 28% of baseline and on falls 33% — a
  real 10% improvement is invisible BY CONSTRUCTION and reads as a null. So
  every row prints its MDE, and a non-significant row is called `null` only
  when the MDE is tight enough to mean it; otherwise it prints `NO RESULT`.
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
* **AND `kicksBack` IS NOT A DIRECTION.** Measured per swing in the gym (12at):
  among touches struck straight at the mouth, `advance < 0` runs 31-60% below
  1 m of travel and 0-5% above it, so it counts the short weak touch the duck
  walks back into inside the 2 s window, not the kick that pointed the wrong
  way. `kicksBackLine` — the line the ball LEFT on over its first 0.5 s, more
  than 90 deg from the attacked mouth — is the direction one, out of
  `kickLineCount` (kicks that moved the ball far enough to have a line). Both
  proportions are printed; quote the second for aim. Rows written before
  2026-09-10 carry neither column and print `—` rather than a zero.

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
    ("kickLineCount", "kicks w/ line"), ("kicksBackLine", "back-line"),
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


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz).  Numerical Recipes
    6.4; converges in tens of iterations over the range we ask of it."""
    tiny, eps = 1e-30, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                    -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + num * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + num / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: int) -> float:
    """P(T > |t|) for Student's t.  Written out because scipy is NOT a
    dependency of this workspace and never has been -- so the old fallback,
    which took the half-width from a t-table but the p-value from `erfc` (the
    NORMAL), is the path that computed every p-value in this repo's soccer
    history.  That mixes two distributions: the interval and the test
    disagreed, and the normal is anti-conservative at these sizes, which is
    the very thing this script's own docstring warns about."""
    if df <= 0:
        return 1.0
    return 0.5 * _betainc(0.5 * df, 0.5, df / (df + t * t))


def t_ppf975(df: int) -> float:
    """The two-sided 95% critical value, by bisection on `t_sf`."""
    lo, hi = 0.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 2.0 * t_sf(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
    # One distribution for both, so the interval and the test agree: a
    # difference is significant exactly when it exceeds the half-width.
    half, p = t_ppf975(n - 1) * se, 2.0 * t_sf(abs(t), n - 1)
    return float(d.mean()), float(half), float(p), int((d > 0).sum())


# A non-significant row is only a null if the battery could have SEEN the
# effect it is denying.  MDE at or under this fraction of baseline is tight
# enough to call a null; above it the honest verdict is "no result".
TIGHT_PCT = 15.0

# Metrics whose MDE has never once been under 100% of baseline on a real
# battery — quoting a difference in them is quoting noise.  Measured over the
# thirteen A/B batteries on disk: ballProgress needs ~21,000 seeds for 10%.
UNQUOTABLE: frozenset[str] = frozenset({"ballProgress"})


def pairing_gain(a: np.ndarray, b: np.ndarray) -> float:
    """How much the shared seeds actually bought: the SD of the difference if
    the arms were independent, over its real SD.  1.0 = the pairing is
    decorative, which is what this harness measures on nearly every battery."""
    sd_d = (b - a).std(ddof=1)
    if sd_d == 0 or len(a) < 2:
        return 1.0
    return float(math.sqrt(a.var(ddof=1) + b.var(ddof=1)) / sd_d)


def verdict(p: float, mde_pct: float, tight: float = TIGHT_PCT) -> str:
    """`effect`, `null`, or `NO RESULT` — never `null` for a battery too small
    to have resolved the effect it is denying."""
    if p < 0.05:
        return "effect"
    return "null" if mde_pct <= tight else "NO RESULT"


def seeds_for(half: float, n: int, base: float, target_pct: float = 10.0) -> int:
    """Seeds needed to resolve `target_pct` of baseline, from this battery's
    own spread.  MDE scales as 1/sqrt(n), so n scales as (MDE/target)^2."""
    want = abs(base) * target_pct / 100.0
    if want <= 0 or half <= 0 or not math.isfinite(half):
        return 0
    return max(2, math.ceil(n * (half / want) ** 2))


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
    ap.add_argument("--tight-pct", type=float, default=TIGHT_PCT,
                    help="MDE (%% of baseline) at or under which a non-significant row is a "
                         "real null rather than NO RESULT (default: %(default)s)")
    ap.add_argument("--target-pct", type=float, default=10.0,
                    help="the effect size the footer sizes a battery for (default: %(default)s%%)")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    seeds = sorted(set(A) & set(B))
    la, lb = args.label or (Path(args.a).stem, Path(args.b).stem)
    dropped = len(set(A) ^ set(B))
    side = f" · side {args.side}" if args.side else ""
    print(f"{lb} against {la}: {len(seeds)} shared seeds{side}"
          + (f" ({dropped} seed(s) in only one file, dropped)" if dropped else ""))
    print(f"\n{'metric':<14}{la[:10]:>10}{lb[:10]:>10}{'Δ':>9}{'±MDE':>8}"
          f"{'MDE%':>7}{'p':>7}{'pair':>6}  verdict")
    thin: list[tuple[str, int]] = []
    for field, how, unit in (("goals", "", ""), ("falls", "", "")) + FIELDS:
        how = how or "sum"
        va = [value(A[s], field, how, args.side) for s in seeds]
        vb = [value(B[s], field, how, args.side) for s in seeds]
        keep = [i for i, (x, y) in enumerate(zip(va, vb)) if x is not None and y is not None]
        if not keep:
            continue
        x, y = np.array([va[i] for i in keep]), np.array([vb[i] for i in keep])
        d, half, p, up = paired(x, y)
        # The 95% half-width IS the minimum detectable effect: a difference is
        # significant exactly when it exceeds it.  It was always printed here
        # and never read as one, which is how underpowered batteries came to be
        # written up as nulls.
        base = abs(x.mean())
        pct = 100.0 * half / base if base else float("inf")
        v = verdict(p, pct, args.tight_pct)
        if field in UNQUOTABLE:
            v = "unquotable"
        elif v == "NO RESULT":
            thin.append((field, seeds_for(half, len(keep), base, args.target_pct)))
        mark = "  ←" if v == "effect" else ""
        pc = "  inf" if not math.isfinite(pct) else f"{pct:>5.0f}%"
        print(f"{field:<14}{x.mean():>10.3f}{y.mean():>10.3f}{d:>+9.3f}{half:>8.3f}"
              f"{pc:>7}{p:>7.3f}{pairing_gain(x, y):>5.2f}x  {v}{mark}   {unit}")

    if thin:
        print(f"\n{len(thin)} metric(s) returned NO RESULT — the battery could not have seen"
              f"\na {args.target_pct:.0f}% change, so it is not evidence of one being absent."
              f"\nSeeds that would resolve {args.target_pct:.0f}% of baseline, from this battery's own spread:")
        for field, n_need in sorted(thin, key=lambda t: t[1]):
            print(f"  {field:<14}{n_need:>7} seeds"
                  + ("   (unreachable — use a per-event metric or the gym)" if n_need > 400 else ""))
    if any(f in UNQUOTABLE for f, _, _ in FIELDS):
        print(f"\nunquotable: {', '.join(sorted(UNQUOTABLE))} — MDE has never been under"
              f" 100% of baseline on a real battery here. Do not quote a difference in it.")
    print("\nevents (totals over the shared seeds):")

    def tot(rows: dict[int, dict], field: str) -> tuple[float, int]:
        """(total over the shared seeds, how many of them carried the field).
        A row written before a column existed contributes NOTHING rather than
        a zero — the pre-12au files have no direction column, and averaging a
        missing column as 0 would report the honest answer "not measured" as
        the finding "it never happened"."""
        vals = [value(rows[s], field, "sum", args.side) for s in seeds]
        got = [v for v in vals if v is not None]
        return float(sum(got)), len(got)

    for field, label in COUNTS:
        (ta, na), (tb, nb) = tot(A, field), tot(B, field)
        sa = f"{ta:>8.0f}" if na else f"{'—':>8}"
        sb = f"{tb:>8.0f}" if nb else f"{'—':>8}"
        note = ""
        if (na or nb) and (na != len(seeds) or nb != len(seeds)):
            note = f"   (carried by {la}: {na}, {lb}: {nb}, of {len(seeds)} seeds)"
        print(f"  {label:<14}{sa}{sb}{note}")
    # The two that have to be read as PROPORTIONS, not as per-run means — and
    # not as each other: the first is a weak-touch measure, the second is the
    # direction one (see the module docstring).
    for num, den, what in (("kicksBack", "kickCount",
                            "kicks whose ball ended up nearer the kicker's own goal 2 s later"),
                           ("kicksBackLine", "kickLineCount",
                            "kicks that LEFT on a backward line (>90° from the attacked mouth at 0.5 s)")):
        (ka, na), (kb, nb) = tot(A, den), tot(B, den)
        (ba, _), (bb, _) = tot(A, num), tot(B, num)
        if not (na and nb and ka and kb):
            print(f"\n{what}: not in these rows — {la} has {na} and {lb} has {nb} of {len(seeds)} "
                  f"seeds carrying `{den}` (a battery run before the column existed).")
            continue
        d, p = two_proportions(int(ba), int(ka), int(bb), int(kb))
        print(f"\n{what}:\n  {ba:.0f}/{ka:.0f} = {ba / ka:.1%}"
              f"  →  {bb:.0f}/{kb:.0f} = {bb / kb:.1%}   ({d:+.1%}, p = {p:.4f} on the events)")


if __name__ == "__main__":
    main()
