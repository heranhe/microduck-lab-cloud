"""The reachable set of a swing-time freshness gate (roadmap 12aj): whiff by
the ball track's age and sigma at the decision tick, what each candidate gate
would keep and refuse, and the belief's offset from the truth. Reads the rows
`scripts/kick_gym.py --out` writes (columns track_age / track_sigma / pred_*).

    cd microduck_local
    uv run python scripts/read_swing_freshness.py runs/x/gym.jsonl [more.jsonl ...]
"""
import json, sys, statistics as st

def q(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    sw = [r for r in rows if r.get("swing")]
    print(f"\n== {path}: {len(rows)} episodes, {len(sw)} swings, whiff {sum(r['whiff'] for r in sw)/max(len(sw),1):.0%}")
    ages = [r["track_age"] for r in sw if r.get("track_age") is not None]
    sig = [r["track_sigma"] for r in sw if r.get("track_sigma") is not None]
    print(f"  track at the swing: {len(ages)}/{len(sw)} have one; age median {st.median(ages):.2f} s (q1 {q(ages,.25):.2f}, q3 {q(ages,.75):.2f}, p90 {q(ages,.9):.2f});"
          f" sigma median {st.median(sig)*100:.1f} cm (q3 {q(sig,.75)*100:.1f})")
    for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 99)):
        g = [r for r in sw if r.get("track_age") is not None and lo <= r["track_age"] < hi]
        if g:
            print(f"    age {lo:>3}-{hi:<3} s  swings {len(g):4} ({len(g)/len(sw):4.0%})  whiff {sum(r['whiff'] for r in g)/len(g):4.0%}"
                  f"  sigma med {st.median(r['track_sigma'] for r in g)*100:4.1f} cm"
                  f"  |pred-true| ahead med {st.median(abs(r['pred_ahead']-r['ahead']) for r in g if r.get('pred_ahead') is not None)*100 if any(r.get('pred_ahead') is not None for r in g) else float('nan'):4.1f} cm")
    for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 9)):
        g = [r for r in sw if r.get("track_sigma") is not None and lo <= r["track_sigma"] < hi]
        if g:
            print(f"    sigma {lo*100:>2.0f}-{hi*100:<3.0f} cm  swings {len(g):4} ({len(g)/len(sw):4.0%})  whiff {sum(r['whiff'] for r in g)/len(g):4.0%}")
    # the gate as registered, and two looser ones: what share of swings each keeps, and the whiff of what it keeps vs refuses
    for a_max, s_max in ((0.3, 0.05), (0.6, 0.05), (1.0, 0.05), (1.0, 0.08), (2.0, 0.10)):
        keep = [r for r in sw if r.get("track_age") is not None and r["track_age"] <= a_max and r["track_sigma"] <= s_max]
        ref = [r for r in sw if r not in keep]
        print(f"    gate age<={a_max} sigma<={s_max*100:.0f}cm: keeps {len(keep)/len(sw):4.0%} of swings, whiff kept {sum(r['whiff'] for r in keep)/max(len(keep),1):4.0%} / refused {sum(r['whiff'] for r in ref)/max(len(ref),1):4.0%}")
    box = [r for r in sw if r.get("pred_ahead") is not None and 0.04 <= r["pred_ahead"] <= 0.16 and 0.01 <= abs(r.get("pred_side") or 0) <= 0.13]
    print(f"  belief inside the box at the swing: {len(box)}/{len(sw)} ({len(box)/max(len(sw),1):.0%}); truth inside it: "
          f"{sum(1 for r in sw if 0.04 <= r['ahead'] <= 0.16 and 0.01 <= abs(r['side']) <= 0.13)/max(len(sw),1):.0%}")
