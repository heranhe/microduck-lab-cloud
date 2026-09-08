"""`eval-striker`: one side's ROSTER against another, on the pitch, on
identical seeds (roadmap 4.4, generalised by Track 4.1.4).

    uv run python -m microduck_local.eval_striker --left chase  --solo --seeds 8 --out runs/s.jsonl --tag solo-chase
    uv run python -m microduck_local.eval_striker --left striker:striker-v1 --solo --seeds 8 --out runs/k.jsonl --tag solo-v1
    uv run python -m microduck_local.eval_striker --left striker:striker-v1 --seeds 8   # 1v1 against a scripted chase
    uv run python -m microduck_local.eval_striker --per-side 2 --seeds 24 \
        --left "chase+defender,chase+striker" --right chase --out runs/def.jsonl --tag defender

Why not `eval-pitch`: that runs `chase` on both sides by construction. This
is the same world, the same walker, the same metrics and the same resume
machinery, with ONE SIDE'S ROSTER swappable — so the arms differ in one
thing. A roster is one entry a duck (`parse_roster`): a brain kind, with an
optional `+role`; a single entry covers the whole side. `--solo` drops the
away side (a 1v0 pitch), which is what the striker is trained on and the
cleanest reading of "can it take the ball to the goal".

`--left` and `--right` are the two SIDES of the pitch: `--left` is the side
that spawns at −x. The TEAMS are colorways (cream at −x, lavender at +x), and
every per-team number in a row is keyed by those.

READ THE THREE BALL NUMBERS TOGETHER (`eval_pitch`'s docstring is the long
version): `ballAdvance` keeps only the forward part and is inflated by
churn, so it is quoted with signed `ballProgress` beside it and with
advance PER KICK, which says whether each touch was worth anything. And
read the event counts: at 8 seeds a battery holds tens of kicks but a
handful of goals and falls, so goals here are reported, not judged.

TWO WARNINGS FROM THE FIRST BATTERY RUN THROUGH THIS.
(1) `kickGoals` / `bumpGoals` come from the World, which attributes a goal
to "kicked" if ANY kick started within 4 s of it. Against a brain that
kicks every two seconds that attribution says nothing: striker-v1's 1v1
battery reads 50 kicked / 0 bumped where the scripted arm on the same seeds
reads 3 / 34. Do not read it as "it scored by kicking".
(2) The seed split matters here as much as anywhere. Swapping the left duck
for striker-v1 looked like it RAISED the untouched scripted opponent's
`ballAdvance` by +0.26 m/min over seeds 0-23 (t=+4.39, 20 seeds up of 24) -
and on twelve fresh seeds the same contrast was +0.002 (t=+0.01, 6 up of
12). It did not replicate; the opponent's falls (11 -> 3, then 2 -> 1) did.
`--seed0` is how you find that out.

The `left`/`right` keys of `goals` are goal MOUTHS, not teams: a ball
crossing at +x lands in `goals["right"]` and the cream team attacks +x, so
a row's `right` count is what cream scored. `metrics` keys are teams (the
colorways) and need no such translation — and since Track 4.1.1 a row also
carries `goalsFor` per team, which is that translation done once, in the
one place that cannot get it wrong.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from .brain import REGISTRY, Senses
from .brain.brain_env import POLICIES_DIR, onnx_infer
from .brain.striker import LearnedStriker
from .eval_pitch import load_done
from .world import World, make_pitch
from .world.metrics import PitchMetrics, SpinMetrics
from .world.scenario import ROLES


def home_away(sc) -> tuple[str, str | None]:
    """The two teams of a pitch, home first — `make_pitch` spawns the home
    team at −x. Read off the scenario rather than named here, so a pitch built
    with a different pair of colorways still works."""
    teams = [d.team for d in sc.ducks if d.team]
    home = teams[0] if teams else None
    away = next((t for t in teams if t != home), None)
    return home, away


def pitch_scenario(per_side: int, solo: bool):
    """`make_pitch`'s pitch, with the away side removed for `--solo`. Both
    arms build it the same way, so a solo battery compares brains and not
    worlds."""
    sc = make_pitch(per_side=per_side)
    if solo:
        home, _ = home_away(sc)
        sc = replace(sc, ducks=[d for d in sc.ducks if d.team == home])
    return sc


def parse_roster(spec: str, n: int) -> list[tuple[str, str | None]]:
    """A side's roster: one entry a duck, in spawn order.

        "chase"                    every duck on the side
        "chase,chase+defender"     d0 chases, d1 chases as a defender
        "striker:striker-v1"       a trained brain from brains/

    An entry is a brain kind, optionally `+role` (`world.scenario.ROLES`).
    One entry is applied to the whole side, which is what every battery
    before rosters existed was doing implicitly."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise SystemExit(f"empty roster {spec!r}")
    if len(parts) == 1:
        parts = parts * n
    if len(parts) != n:
        raise SystemExit(f"roster {spec!r} has {len(parts)} entries for {n} ducks a side")
    out = []
    for p in parts:
        kind, _, role = p.partition("+")
        if role and role not in ROLES:
            raise SystemExit(f"roster {spec!r}: {role!r} is not a role ({', '.join(ROLES)})")
        out.append((kind, role or None))
    return out


def apply_roster(sc, home_spec: str, away_spec: str):
    """The scenario with each side's brains and roles written onto its ducks,
    so every downstream reader — `brain_kwargs`, the team board, the row — sees
    one roster and not a pile of command-line arguments."""
    home, away = home_away(sc)
    per_side = {t: sum(1 for d in sc.ducks if d.team == t) for t in (home, away) if t}
    rosters = {t: parse_roster(spec, per_side[t])
               for t, spec in ((home, home_spec), (away, away_spec)) if t in per_side}
    seen: dict[str, int] = {}
    ducks = []
    for d in sc.ducks:
        if d.team in rosters:
            k = seen.get(d.team, 0)
            seen[d.team] = k + 1
            kind, role = rosters[d.team][k]
            d = replace(d, brain=kind, role=role)
        ducks.append(d)
    return replace(sc, ducks=ducks)


def make_brain(kind: str, spec, world, teams: dict):
    """A brain for one duck: the scripted `chase` with the pitch kwargs the
    world gives it, or `striker:<name>` — a trained brain from brains/, which
    gets the same goal, bounds and team."""
    from .brain.team import Team, brain_kwargs
    if kind.startswith("striker:"):
        d = world.ducks[spec.id]
        hx, hy = world.scenario.floor[0] / 2 - 0.25, world.scenario.floor[1] / 2 - 0.25
        team = teams.setdefault(spec.team, Team(spec.team)) if spec.team else None
        return LearnedStriker(kind.split(":", 1)[1], goal=world.goal_for(d), duck_id=spec.id,
                              bounds=(hx, hy), goal_w=world.goal_width, team=team)
    return REGISTRY.make(kind, **brain_kwargs(spec, world, teams))


def run_one(seed: int, seconds: float, left: str = "chase", right: str = "chase",
            per_side: int = 1, solo: bool = False) -> dict:
    """One run. Byte-for-byte `eval_pitch.run_one` when left == right ==
    "chase" and solo is False — the test in tests/test_striker.py pins that,
    which is what makes the scripted arm here the published baseline and not
    a re-implementation of it."""
    from .brain.team import kickoff_brains
    sc = apply_roster(pitch_scenario(per_side, solo), left, right)
    home, away = home_away(sc)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams: dict = {}
    brains = {d.id: make_brain(d.brain or "chase", d, w, teams) for d in sc.ducks}
    # A little seed-dependent asymmetry: nudge the ball off centre.
    rng = np.random.default_rng(seed)
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    metrics = PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})
    spin = SpinMetrics(w)
    goal_seq = 0
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            intent = brains[d.id].step(s)
            spin.tick(d, intent.twist)
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        metrics.tick()
        if w.goal_seq != goal_seq:              # a goal: play restarts from the spawns
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
    score = w.soccer_score()
    return {"seed": seed, "perSide": per_side, "solo": solo, "left": score["left"], "right": score["right"],
            "leftBrain": left, "rightBrain": right if not solo else None,
            "home": home, "away": None if solo else away,
            "roles": {d.id: d.role for d in sc.ducks},
            "kickGoals": score["kicked"], "bumpGoals": score["bumped"],
            "ballOuts": score["ballOuts"], "ballOutS": 0.0, "getupS": 0.0,
            "getups": w.getups, "getupTimeouts": w.getup_timeouts,
            "kicks": {k: b.kicks for k, b in brains.items()}, "pushes": {k: b.pushes for k, b in brains.items()},
            "falls": {k: d.falls for k, d in w.ducks.items()},
            "team": {d.id: (d.team or d.id) for d in sc.ducks},
            "simSeconds": round(w.t, 1), "seconds": seconds, **spin.row(), **metrics.row()}


# --- reading a battery --------------------------------------------------------
def team_of(row: dict, side: str) -> dict:
    return {k: v for k, v in row["team"].items() if v == side}


def _scored(r: dict, side: str) -> int:
    """Goals this team scored. Off the ledger when the row has one — exact,
    per team, and no mouth to misread. Off the mouth rule otherwise, which is
    the trap the ledger exists to close: a ball crossing at +x is recorded
    under "right" and is scored by the side that spawns at −x."""
    gf = r.get("goalsFor")
    if gf and side in gf:
        return int(gf[side])
    return int(r["right" if side == "left" else "left"])


def _sum_field(rows: list[dict], field: str, side: str) -> int | None:
    vals = [r[field][side] for r in rows if r.get(field) and side in r[field]]
    return int(np.sum(vals)) if vals else None


def side_reading(rows: list[dict], side: str = "left") -> dict:
    """One TEAM's numbers over a battery: the three ball readings that have to
    be quoted together, its event counts, and the Track 4 ledger (own goals,
    back-kicks) where the rows carry it. `side` is a team name — a colorway on
    any pitch built since teams became colorways, "left"/"right" on a row
    written before that."""
    adv = [r["ballAdvance"][side] for r in rows if r.get("ballAdvance")]
    prog = [r["ballProgress"][side] for r in rows if r.get("ballProgress")]
    poss = [r["possession"][side] for r in rows if r.get("possession")]
    kicks = [sum(v for k, v in r["kicks"].items() if k in team_of(r, side)) for r in rows]
    falls = [sum(v for k, v in r["falls"].items() if k in team_of(r, side)) for r in rows]
    scored = [_scored(r, side) for r in rows]
    tot_adv = float(np.sum([a * r["simSeconds"] / 60.0 for a, r in zip(adv, rows)])) if adv else 0.0
    crowd = [r["crowd"][side] for r in rows if r.get("crowd") and r["crowd"].get(side) is not None]
    return {"seeds": len(rows), "advance": float(np.mean(adv)) if adv else None,
            "advanceSd": float(np.std(adv, ddof=1)) if len(adv) > 1 else None,
            "progress": float(np.mean(prog)) if prog else None,
            "progressSd": float(np.std(prog, ddof=1)) if len(prog) > 1 else None,
            "possession": float(np.mean(poss)) if poss else None,
            "kicks": int(np.sum(kicks)), "falls": int(np.sum(falls)), "goals": int(np.sum(scored)),
            "advPerKick": (tot_adv / np.sum(kicks)) if np.sum(kicks) else None,
            "own": _sum_field(rows, "ownGoals", side),
            "conceded": _sum_field(rows, "goalsAgainst", side),
            "kicksBack": _sum_field(rows, "kicksBack", side),
            "kickCount": _sum_field(rows, "kickCount", side),
            "crowd": float(np.mean(crowd)) if crowd else None,
            "perSeedAdvance": adv, "perSeedGoals": scored}


def _fmt(v, nd=2, sign=False):
    if v is None:
        return "—"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def summarize(rows: list[dict], side: str = "left") -> str:
    s = side_reading(rows, side)
    line = (f"{side}: ballAdvance {_fmt(s['advance'])} ± {_fmt(s['advanceSd'])} m/min · "
            f"ballProgress {_fmt(s['progress'], sign=True)} ± {_fmt(s['progressSd'])} m/min · "
            f"advance/kick {_fmt(s['advPerKick'], 3)} m · possession {_fmt(s['possession'], 1)} s/min\n"
            f"    events over {s['seeds']} seeds: {s['kicks']} kicks · {s['goals']} goals · {s['falls']} falls")
    if s["own"] is not None:
        line += (f" · own {s['own']} of {s['conceded']} conceded"
                 f" · back-kicks {s['kicksBack']} of {s['kickCount']}")
        if s["crowd"] is not None:
            line += f" · crowd {s['crowd']:.0%}"
    return line


def _run_args(a: tuple) -> dict:
    return run_one(*a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="chase",
                    help="the roster of the side that spawns at −x, one entry a duck: "
                         "'chase' | 'chase,chase+defender' | 'striker:<brains/ name>'")
    ap.add_argument("--right", default="chase", help="the opposite side's roster (ignored with --solo)")
    ap.add_argument("--solo", action="store_true", help="1v0: no opponent duck (what the striker trains on)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0, help="first seed — extends a battery onto FRESH seeds")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--per-side", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", help="append each seed as a JSON line AND resume from it")
    ap.add_argument("--tag", default="", help="recorded in --out rows; a resume refuses to mix tags")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    seeds = [args.seed0 + k for k in range(args.seeds)]
    done = load_done(args.out, args.tag, args.per_side, args.seconds)
    rows = [done[sd] for sd in seeds if sd in done]
    todo = [sd for sd in seeds if sd not in done]
    out = open(args.out, "a") if args.out else None

    def keep(r: dict) -> None:
        rows.append(r)
        if out is not None:
            out.write(json.dumps({**r, "tag": args.tag}) + "\n")
            out.flush()
        if not args.json:
            print(f"seed {r['seed']}: goals L{r['left']}/R{r['right']} · kicks {sum(r['kicks'].values())}"
                  f" · falls {sum(r['falls'].values())}"
                  f" · advance {r['ballAdvance']} · progress {r['ballProgress']}", flush=True)

    todo_args = [(sd, args.seconds, args.left, args.right, args.per_side, args.solo) for sd in todo]
    try:
        if args.jobs > 1 and len(todo) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
            with ctx.Pool(min(args.jobs, len(todo))) as pool:
                for r in pool.imap(_run_args, todo_args):
                    keep(r)
        else:
            for a in todo_args:
                keep(run_one(*a))
    finally:
        if out is not None:
            out.close()
    rows.sort(key=lambda r: r["seed"])
    if args.json:
        print(json.dumps(rows))
        return
    label = f"{args.left} vs {'nobody' if args.solo else args.right}"
    print(f"\n{label}, {len(rows)} seeds x {args.seconds:g} s:")
    # The rows say which team is which side; a row written before teams were
    # colorways says "left"/"right" and reads the same way.
    home = rows[0].get("home", "left")
    away = rows[0].get("away", "right")
    print("  " + summarize(rows, home))
    if not args.solo and away:
        print("  " + summarize(rows, away))


if __name__ == "__main__":
    main()
