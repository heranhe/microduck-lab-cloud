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

# THE DUEL (roadmap C.4's second half). A duel row file has no swings in it, so
# every table below would print a page of NaN on one. `kick_gym.compare` has
# already printed the pooled arm-by-arm reading (it dispatches on the row);
# this is the paired one the roadmap reports beside it, and then we stop.
if any(r.get("duel") for rs in arms.values() for r in rs):
    def duel_per_seed(rows):
        out = defaultdict(lambda: {"n": 0, "ours": 0, "theirs": 0, "none": 0, "fell": 0,
                                   "their_adv": [], "our_adv": []})
        for r in rows:
            if not r.get("duel"):
                continue
            s = out[r.get("seed")]
            s["n"] += 1
            s["ours"] += int(r.get("first") == "ours")
            s["theirs"] += int(r.get("first") == "theirs")
            s["none"] += int(r.get("first") is None)
            s["fell"] += int((r.get("falls_us") or 0) > 0)
            for k, col in (("their_advance", "their_adv"), ("our_advance", "our_adv")):
                if r.get(k) is not None:
                    s[col].append(float(r[k]))
        return out

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 78 + "\nper seed (paired) — we touch first\n" + "=" * 78)
    b = duel_per_seed(arms[base])
    for lab in labels[1:]:
        a = duel_per_seed(arms[lab])
        seeds = sorted(set(b) | set(a))
        better = worse = ties = 0
        print(f"\n{'seed':>5}{base + ' 1st':>16}{lab + ' 1st':>16}"
              f"{'their adv ' + base:>18}{'their adv ' + lab:>18}")
        for sd in seeds:
            fb = b[sd]["ours"] / b[sd]["n"] if b[sd]["n"] else float("nan")
            fa = a[sd]["ours"] / a[sd]["n"] if a[sd]["n"] else float("nan")
            if fa > fb:
                better += 1
            elif fa < fb:
                worse += 1
            else:
                ties += 1
            print(f"{sd:>5}{100 * fb:>14.0f}% {100 * fa:>14.0f}% "
                  f"{mean(b[sd]['their_adv']):>+17.3f} {mean(a[sd]['their_adv']):>+17.3f}")
        print(f"\n{lab} vs {base}: we touch first MORE OFTEN on {better}/{len(seeds)} seeds, "
              f"less on {worse}, ties {ties} (sign test p = "
              f"{binom_two_sided(min(better, worse), better + worse):.3f})")
        tb = [v for sd in seeds for v in b[sd]["their_adv"]]
        ta = [v for sd in seeds for v in a[sd]["their_adv"]]
        ob = [v for sd in seeds for v in b[sd]["our_adv"]]
        oa = [v for sd in seeds for v in a[sd]["our_adv"]]
        print(f"their advance (m toward OUR goal, per THEIR touch): "
              f"{mean(tb):+.3f} (n={len(tb)}) -> {mean(ta):+.3f} (n={len(ta)})")
        print(f"our advance   (m toward THEIR goal, per OUR touch): "
              f"{mean(ob):+.3f} (n={len(ob)}) -> {mean(oa):+.3f} (n={len(oa)})")
        fb_ = sum(b[sd]["fell"] for sd in seeds), sum(b[sd]["n"] for sd in seeds)
        fa_ = sum(a[sd]["fell"] for sd in seeds), sum(a[sd]["n"] for sd in seeds)
        d_, p_, mde_ = kick_gym.two_proportions(fb_[0], fb_[1], fa_[0], fa_[1])
        print(f"THE VETO — episodes our duck fell in: {fb_[0]}/{fb_[1]} -> {fa_[0]}/{fa_[1]}, "
              f"shift {100 * d_:+.1f} pp, MDE {100 * mde_:.1f}, p = {p_:.3f}")
    sys.exit(0)

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

# THE EXIT AND THE BACK SHARE (roadmap 12at). Silent on row files written
# before the exit column existed, so every older pair compares as it did.
if any(kick_gym.exit_summ(arms[lab]) for lab in labels):
    print("\n" + "=" * 78 + "\nin-play exit per foot, and the back share\n" + "=" * 78)
    print(f"{'arm':<22}{'foot':<12}{'n':>5}{'exit med':>10}{'IQR':>17}{'sidecar':>9}"
          f"{'off by':>8}{'|err| med':>11}{'>±34°':>7}{'back':>7}")
    for lab in labels:
        s = kick_gym.exit_summ(arms[lab])
        for foot, d in s.items():
            deg = math.degrees
            iqr = f"{deg(d['exit_q1']):+.0f}..{deg(d['exit_q3']):+.0f}°"
            sc = "  -  " if d["assumed"] is None else f"{deg(d['assumed']):+.0f}°"
            off = "  -  " if d["assumed"] is None else f"{deg(d['exit_med'] - d['assumed']):+.0f}°"
            ea = f"{deg(d['err_abs_med']):.0f}°" if "err_abs_med" in d else "  -  "
            eo = f"{100 * d['err_outside']:.0f}%" if "err_outside" in d else "  - "
            print(f"{lab:<22}{foot:<12}{d['n']:>5}{deg(d['exit_med']):>+9.0f}°{iqr:>17}{sc:>9}"
                  f"{off:>8}{ea:>11}{eo:>7}{100 * d['back']:>6.0f}%")
    # The back share as a PROPORTION OF THE TOUCH EVENTS, which is how the
    # ledger's `kicksBack` is read (memory: soccer-metric-power-table) -- never
    # as a per-seed rate. Same two-proportion z + MDE as the whiff column.
    def back(rows):
        rs = [r for r in rows if r.get("swing") and r.get("back") is not None]
        return sum(bool(r["back"]) for r in rs), len(rs)
    xb, nb = back(arms[base])
    if nb:
        print(f"\n{'arm':<22}{'touches':>9}{'back':>7}{'share':>8}{'vs base':>9}{'±MDE':>7}{'p':>8}  verdict")
        print(f"{base + ' (base)':<22}{nb:>9}{xb:>7}{100 * xb / nb:>7.0f}%{'—':>9}{'—':>7}{'—':>8}")
        for lab in labels[1:]:
            xa, na = back(arms[lab])
            if not na:
                continue
            d_, p_, mde_ = kick_gym.two_proportions(xb, nb, xa, na)
            print(f"{lab:<22}{na:>9}{xa:>7}{100 * xa / na:>7.0f}%{100 * d_:>+8.0f}%"
                  f"{100 * mde_:>6.0f}%{p_:>8.3f}  {kick_gym.verdict_prop(p_, mde_)}")
        print("  back = the ledger's own rule per touch: signed displacement along the attacked"
              "\n  axis over the carry window < 0 (`Metrics._resolve_kicks`).")
