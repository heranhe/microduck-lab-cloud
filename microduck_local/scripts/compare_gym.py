"""Paired A/B of two (or more) kick_gym --out files on the same seeds.
    uv run python scripts/compare_gym.py vendored=runs/widekick/gym-vendored.jsonl wide=runs/widekick/gym-warm.jsonl
The gym's own compare() (funnel per arm, two-proportion z + MDE on whiff),
then the per-seed reading the roadmap reports beside it: whiff per seed,
connected kicks, sweet-spot rate, median |side|, and a sign test."""
import json, math, sys
from collections import defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import kick_gym  # noqa: E402

arms = {}
for spec in sys.argv[1:]:
    label, path = spec.split("=", 1)
    arms[label] = [json.loads(l) for l in open(path) if l.strip()]
kick_gym.compare(arms)

labels = list(arms)
base = labels[0]

def per_seed(rows):
    out = defaultdict(lambda: {"n": 0, "whiff": 0, "conn": 0, "spot": 0})
    for r in rows:
        if not r.get("swing"):
            continue
        s = out[r["seed"]]
        s["n"] += 1; s["whiff"] += int(r["whiff"]); s["conn"] += int(not r["whiff"])
        s["spot"] += int(0.06 <= r["ahead"] <= 0.10 and 0.04 <= abs(r["side"]) <= 0.08)
    return out

def binom_two_sided(k, n):
    if n == 0:
        return float("nan")
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2
    return min(1.0, p)

print("\n" + "=" * 78 + "\nper seed (paired)\n" + "=" * 78)
b = per_seed(arms[base])
for lab in labels[1:]:
    a = per_seed(arms[lab])
    seeds = sorted(set(b) | set(a))
    better = worse = ties = 0
    print(f"\n{'seed':>5}{base + ' whiff':>16}{lab + ' whiff':>16}{'conn ' + base:>12}{'conn ' + lab:>12}")
    for s in seeds:
        wb = b[s]["whiff"] / b[s]["n"] if b[s]["n"] else float("nan")
        wa = a[s]["whiff"] / a[s]["n"] if a[s]["n"] else float("nan")
        if wa < wb: better += 1
        elif wa > wb: worse += 1
        else: ties += 1
        print(f"{s:>5}{100 * wb:>14.0f}% {100 * wa:>14.0f}% {b[s]['conn']:>12}{a[s]['conn']:>12}")
    swb = [r for r in arms[base] if r.get("swing")]
    swa = [r for r in arms[lab] if r.get("swing")]
    def med_side(rs): return sorted(abs(r["side"]) for r in rs)[len(rs) // 2] if rs else float("nan")
    def spot(rs): return sum(0.06 <= r["ahead"] <= 0.10 and 0.04 <= abs(r["side"]) <= 0.08 for r in rs) / max(len(rs), 1)
    print(f"\n{lab} vs {base}: whiff better on {better}/{len(seeds)} seeds, worse {worse}, ties {ties}"
          f" (sign test p = {binom_two_sided(better, better + worse):.3f})")
    fb = [r for r in swb if r.get("fell") is not None]; fa = [r for r in swa if r.get("fell") is not None]
    if fb and fa:
        d_, p_, mde_ = kick_gym.two_proportions(sum(r["fell"] for r in fb), len(fb), sum(r["fell"] for r in fa), len(fa))
        print(f"fell inside the carry window: {sum(r['fell'] for r in fb)}/{len(fb)} ({100 * sum(r['fell'] for r in fb) / len(fb):.1f}%) -> "
              f"{sum(r['fell'] for r in fa)}/{len(fa)} ({100 * sum(r['fell'] for r in fa) / len(fa):.1f}%), shift {100 * d_:+.1f} pp, MDE {100 * mde_:.1f}, p = {p_:.3f}")
    elif fb or fa:
        print("fell: only one arm carries the fall column (rows written before 2026-09-10 have none)")
    print(f"connected kicks: {sum(not r['whiff'] for r in swb)} -> {sum(not r['whiff'] for r in swa)}"
          f" | sweet spot: {100 * spot(swb):.1f}% -> {100 * spot(swa):.1f}%"
          f" | median |side|: {med_side(swb):.3f} -> {med_side(swa):.3f} m"
          f" | median travel of connected: "
          f"{sorted(r['travel'] for r in swb if not r['whiff'])[max(0, sum(not r['whiff'] for r in swb) // 2)] if swb else float('nan'):.2f} -> "
          f"{sorted(r['travel'] for r in swa if not r['whiff'])[max(0, sum(not r['whiff'] for r in swa) // 2)] if swa else float('nan'):.2f} m")
