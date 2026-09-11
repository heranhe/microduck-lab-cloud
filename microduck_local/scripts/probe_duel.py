"""The DUEL's reachable set (roadmap C.4, second half): how often is an
opponent nearer the ball than this duck, at contact range, while this duck is
going for it — and what happens next when it is.

Playbook rule 5, and the specific lesson of `scripts/probe_contest.py`: the
duel's three previous answers (`lineup_keepout`, `opp_keepout`,
`contest_margin`) were all published as nulls and all three fired on 0.07% of
ticks, which is arithmetic and not a finding. So before building a fourth,
count the population — on the SHIPPED brain, with no knob set:

    uv run python scripts/probe_duel.py --seeds 4 --seconds 120 --per-side 2
    uv run python scripts/probe_duel.py --seeds 4 --seconds 120 --per-side 3 --formation

Two populations are counted, and they are NOT the same number:

* **truth** — the world's own positions. This is the OPPORTUNITY: how often
  the duel situation exists at all, whatever anybody can see.
* **seen** — what `Chase._opponents` and the duck's own ball track say. This
  is the CEILING on any rule built into the brain, because a rule cannot act
  on a duel it cannot perceive. Per roadmap 12n the gap between the two is a
  camera result, not a geometry one.

For every trigger tick the probe also books the next `--window` seconds:
which team is on the ball (`POSSESSION_R`, the metric `eval-pitch` uses), how
far the ball travels toward the goal this duck attacks, and whether this duck
falls. The same three are booked for the NON-trigger chase ticks, because
"what the duck does today in those ticks" is only readable against what it
does in the ticks that are not duels.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from microduck_local import contract as C
from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains, throw_in_brains
from microduck_local.world import World, make_pitch
from microduck_local.world.metrics import POSSESSION_R

GOING = ("chase", "lineup", "settle", "turn", "duel")   # this duck is going for the ball
# ("duel" never occurs on the shipped brain, so the shipped numbers are what they
#  were; it is here so the chain still reads when the probe is run with the knob on.)


def run(seed: int, seconds: float, per_side: int, formation: bool,
        near_ball: float, ball_out_s: float) -> tuple[list[dict], list[dict]]:
    """(per-tick duck rows, per-tick world frames) for one seed."""
    sc = make_pitch(per_side=per_side, formation=formation)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed, ball_out_s=ball_out_s)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    team_of = {d.id: (d.team or d.id) for d in sc.ducks}
    # +1 if this duck's team attacks +x. `Scenario.attacks` maps team -> mouth.
    sign = {did: (1.0 if sc.attacks.get(tm) == "right" else -1.0) for did, tm in team_of.items()}
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    rows: list[dict] = []
    frames: list[dict] = []
    goal_seq, out_seq, k = 0, w.ball_out_seq, 0
    while w.t < seconds:
        ball = w.ball_xy()
        pos = {did: (lambda p: (float(p[0]), float(p[1])))(d.trunk_pos(w.data))
               for did, d in w.ducks.items()}
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
            # ---- TRUTH: the world's own geometry -------------------------
            mine = math.dist(pos[did], ball)
            opp_d = [math.dist(pos[o], ball) for o in pos if team_of[o] != team_of[did]]
            near = min(opp_d, default=math.inf)
            t_close = near <= near_ball
            t_duel = t_close and near < mine
            # ---- SEEN: what the brain could act on ------------------------
            tr = b.tracker.best(b.p.target_cls, w.t, min_hits=1)
            sees_ball = tr is not None and tr.xy is not None and tr.age(w.t) < b.p.lost_s
            s_close = s_duel = False
            if sees_ball:
                bxy = (float(tr.xy[0]), float(tr.xy[1]))
                od = w.odom(d)
                my_r = math.dist((od[0], od[1]), bxy)
                seen_near = min((math.dist(o, bxy) for o in b._opponents(w.t)), default=math.inf)
                s_close = seen_near <= near_ball
                s_duel = s_close and seen_near < my_r
            rows.append({"k": k, "duck": did, "team": team_of[did], "sign": sign[did],
                         "state": b.state, "going": b.state in GOING,
                         "t_close": t_close, "t_duel": t_duel,
                         "sees_ball": sees_ball, "s_close": s_close, "s_duel": s_duel,
                         "falls": d.falls})
        w.step()
        # The world AFTER the step: who is on the ball, and where it is.
        ball2 = w.ball_xy()
        pos2 = {did: (lambda p: (float(p[0]), float(p[1])))(d.trunk_pos(w.data))
                for did, d in w.ducks.items()}
        who = min(pos2, key=lambda i: math.dist(pos2[i], ball2))
        frames.append({"ball": ball2,
                       "holder": team_of[who] if math.dist(pos2[who], ball2) <= POSSESSION_R else None,
                       "falls": {did: d.falls for did, d in w.ducks.items()},
                       "seq": (w.goal_seq, w.ball_out_seq)})
        k += 1
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
        if w.ball_out_seq != out_seq:
            out_seq = w.ball_out_seq
            throw_in_brains(brains, teams)
    return rows, frames


def outcome(r: dict, frames: list[dict], win: int) -> dict | None:
    """What the next `win` ticks did to the ball, from this duck's point of
    view. None if the window runs off the end of the run, or if a goal or a
    ball-out teleports the ball inside it — the referee's metres are not the
    duel's (the same guard `world/metrics.py` carries)."""
    i, j = r["k"], r["k"] + win
    if j >= len(frames):
        return None
    if frames[j]["seq"] != frames[i]["seq"]:
        return None
    hold = Counter(frames[n]["holder"] for n in range(i, j))
    return {"mine": hold[r["team"]] / win,
            "theirs": sum(v for kk, v in hold.items() if kk not in (None, r["team"])) / win,
            "none": hold[None] / win,
            "gain": r["sign"] * (frames[j]["ball"][0] - frames[i]["ball"][0]),
            "fell": frames[j]["falls"][r["duck"]] > frames[i]["falls"][r["duck"]]}


def report(rows: list[dict], frames_by_seed: dict[int, list[dict]],
           win: int, near_ball: float, label: str) -> None:
    n = len(rows)
    print(f"\n=== {label} — {n} duck-ticks ===\n")
    print(f"{'the chain of preconditions':<52}{'ticks':>9}{'share':>9}")
    chain = (
        ("going", f"this duck is going for the ball {GOING}"),
        ("t_close", f"TRUTH: an opponent within {near_ball:.2f} m of the ball"),
        ("t_duel", "TRUTH: …and nearer to it than this duck"),
        ("going_t_duel", "TRUTH: …and this duck is going for the ball  ← the DUEL"),
        ("sees_ball", "SEEN: this duck has a live ball track"),
        ("s_duel", f"SEEN: …and an opponent it can see is nearer to it, within {near_ball:.2f} m"),
        ("going_s_duel", "SEEN: …and this duck is going for the ball  ← the REACHABLE SET"),
    )
    for key, lab in chain:
        c = sum((r["going"] and r["t_duel"]) if key == "going_t_duel" else
                (r["going"] and r["s_duel"]) if key == "going_s_duel" else r[key] for r in rows)
        print(f"{lab:<52}{c:>9}{100 * c / n:>8.2f}%")

    def table(title: str, pick, split) -> None:
        print(f"\n{title}")
        print(f"{'state':<12}{'ticks':>8}{'% of all':>10}{'poss mine':>11}{'theirs':>9}"
              f"{'nobody':>9}{'ball gain m':>13}{'falls/1k':>10}")
        buckets: dict[str, list[dict]] = {}
        for r in rows:
            if not pick(r):
                continue
            o = outcome(r, frames_by_seed[r["seed"]], win)
            if o is not None:
                buckets.setdefault(split(r), []).append(o)
        for name in sorted(buckets, key=lambda s: -len(buckets[s])):
            o = buckets[name]
            m = len(o)
            print(f"{name:<12}{m:>8}{100 * m / n:>9.2f}%{np.mean([x['mine'] for x in o]):>11.2f}"
                  f"{np.mean([x['theirs'] for x in o]):>9.2f}{np.mean([x['none'] for x in o]):>9.2f}"
                  f"{np.mean([x['gain'] for x in o]):>+13.4f}{1000 * np.mean([x['fell'] for x in o]):>10.1f}")
        if not buckets:
            print("  (no ticks)")

    print(f"\nWhat happens in the next {win * C.CTRL_DT:.1f} s — possession is the SHARE of those"
          f"\nticks a team is within {POSSESSION_R} m of the ball; ball gain is metres toward"
          "\nthe goal THIS duck attacks; falls are this duck's, per 1000 trigger ticks.")
    table("TRUTH duel ticks, by the state the duck is in:",
          lambda r: r["t_duel"], lambda r: r["state"])
    table("…against the ticks that are NOT duels, same states:",
          lambda r: not r["t_duel"] and r["going"], lambda r: r["state"])
    table("SEEN (reachable) duel ticks, by state:",
          lambda r: r["s_duel"], lambda r: r["state"])
    fires = sum(r["going"] and r["s_duel"] for r in rows)
    if not fires:
        print("\n!! THE REACHABLE SET IS EMPTY. Before believing it, check the probe could have"
              "\n   seen a non-zero: are there opponents on the pitch (--per-side >= 2), does"
              "\n   `Chase._opponents` return anything at all, and is the ball ever tracked?"
              "\n   A zero from an instrument that could not have measured one reads exactly"
              "\n   like a finding (probe_contest.py's warning, same trap).")
        return
    print(f"\nThe duel is reachable on {100 * fires / len(rows):.2f}% of duck-ticks. Read any null"
          "\nfrom a whole-match battery against that number FIRST (roadmap: 'reading a null"
          "\nagainst how often the rule fires' — required per-firing effect = MDE / firing rate).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--formation", action="store_true", help="stamp the roster's roles (the 3v3 the lab runs)")
    ap.add_argument("--near-ball", type=float, default=0.35, help="contact range to the BALL (m)")
    ap.add_argument("--window", type=float, default=2.0, help="seconds of consequence booked after a trigger")
    ap.add_argument("--ball-out-s", type=float, default=5.0)
    a = ap.parse_args()
    win = max(1, int(round(a.window / C.CTRL_DT)))
    rows: list[dict] = []
    frames_by_seed: dict[int, list[dict]] = {}
    for sd in [a.seed0 + k for k in range(a.seeds)]:
        r, f = run(sd, a.seconds, a.per_side, a.formation, a.near_ball, a.ball_out_s)
        for x in r:
            x["seed"] = sd
        rows += r
        frames_by_seed[sd] = f
        print(f"  seed {sd} done ({len(r)} duck-ticks)", flush=True)
    label = f"{a.per_side}v{a.per_side}{' with roles' if a.formation else ''}, {a.seeds} seeds x {a.seconds:.0f} s"
    report(rows, frames_by_seed, win, a.near_ball, label)


if __name__ == "__main__":
    main()
