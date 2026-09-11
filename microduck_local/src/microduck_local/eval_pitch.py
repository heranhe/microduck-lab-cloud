"""`eval-pitch`: the soccer benchmark (first form) — two `chase` brains,
one ball, goals in a fixed time, over seeds.

    uv run eval-pitch --seeds 4 --seconds 300 --jobs 2
    uv run eval-pitch --seeds 4 --per-side 2      # 2v2 (teams share a blackboard, brain/team.py)

Per seed: goals scored on each side, kicks and pushes attempted, falls, and
the CONTINUOUS metrics below (ball progress, possession); then the means per
run. Headless, the same World `pitch` / `pitch-2v2` / `pitch-3v3` streams
on /sim.

READ THE `left`/`right` GOAL KEYS THE WAY THE WORLD WRITES THEM. They are
goal MOUTHS, not team scores: `World._check_goal` puts a ball crossing at +x
into `goals["right"]`, and the team spawned at −x attacks +x — so a row's
`right` count is the number of goals THE TEAM AT −x SCORED. The run TOTAL
(`left + right`) is unaffected, but every per-side reading of a row is
inverted if this is missed, and reading it the natural way flipped the sign
of a whole correlation table here before it was caught.

Two things now stand between that trap and a reader. The teams are
COLORWAYS (`cream` at −x, `lavender` at +x — Track 4.2), so a team name and a
mouth name can no longer be the same word; and `goalsFor` / `goalsAgainst`
per team are in the row (below), which is the translation done once in the
one place that cannot get it wrong.

Measured, 16 seeds x 300 s of 1v1 per arm, shipped lens vs a 120x93 deg one
(two independent 8-seed batteries, both replicated), coefficient of variation
and the seeds an arm needs to resolve a 25% shift in the metric at p<0.05 /
80% power — i.e. the metric's own noise, with the size of this particular
contrast divided out:

READ `ballAdvance` WITH `ballProgress` BESIDE IT. Advance keeps only the
forward part (max(0, dx)), so it is INFLATED BY CHURN: a change that merely
makes the ball move more scores higher on it without sending the ball
anywhere. Measured, the attacker-handover fix: kicks +64%, advance +0.18
+/- 0.06 (2.9 sigma) — and signed `ballProgress` FLAT at -0.003 +/- 0.136
(0.0 sigma) with advance per kick HALVED, 0.202 -> 0.106. The ball moved
more and no further toward the goal. So: advance is the sensitive
instrument, signed progress is the one that says the motion had a
direction, and advance-per-kick says whether each touch was worth more.
Quote all three or none.

    goals            CV 0.76   146 seeds     r with goals   —
    kicks            CV 0.49    62 seeds     r  0.30 [-0.06, +0.59]
    falls            CV 1.22   376 seeds     r  0.07 [-0.29, +0.40]
    ballAdvance      CV 0.40    43 seeds     r  0.50 [+0.19, +0.72]
    possession       CV 0.16     9 seeds     r  0.33 [-0.02, +0.61]
    possessionWide   CV 0.11     6 seeds     r  0.23 [-0.13, +0.53]
    ballProgress     CV 4.20      —          r  0.11 [-0.25, +0.44]

`ballAdvance` is the one to judge a variant on: it is the only metric here
whose association with goals is resolved away from zero (and the only one at
all, `kicks` included), at ~3x fewer seeds than goals. `possession` is the
cheapest DETECTOR of any difference at all — 6-9 seeds against goals' 146 —
but it is not evidence a variant is better, because what it tracks is how
much of the run a duck spends standing over the ball. Ball progress summed
over both teams is near zero by construction in self-play (the two sides
attack opposite goals) and carries nothing; it is a per-TEAM metric, for an
asymmetric matchup.

THE PER-TEAM LEDGER (roadmap Track 4.1). The `left`/`right` mouth keys above
cannot tell an own goal from a scored one, and the first four seeds measured
through them hid the fact that 8 of 8 goals in a 2v2 battery were own goals.
So a row also carries, per TEAM (`world/metrics.py` documents the credit
rule):

    goalsFor/goalsAgainst  the mouth this team attacks / defends. Exact — no
                           attribution, so nothing to get wrong.
    ownGoals               the subset of goalsAgainst this team put in itself,
                           credited to the kicker inside KICK_GOAL_S else to
                           the last team on the ball inside GOAL_CREDIT_S.
    goalsUnattributed      goals neither test could place (a run-scalar).
    kickCount/kicksBack    kicks, and the ones whose ball ended up nearer the
                           kicker's OWN goal CARRY_S later. Measured off the
                           ball, never off the brain's plan: the plan says
                           what it meant to do (rule 6).
    kickCarry              metres toward the attacked goal summed over that
                           team's kicks; divide by kickCount for the mean.
    ballOwnHalf            s/min the ball spends in this team's own half.
    spread/crowd/depth     mean distance between teammates; the fraction of
                           ticks two of them are within 0.5 m of the ball
                           (a pile-up); how far back the deepest one keeps.
                           spread and crowd are None for a one-duck team.

Baseline, 4 seeds x 300 s of shipped `chase` both sides (2026-09-05):
1v1 3 own goals of 6, 14 of 26 kicks back; 2v2 8 of 8, 14 of 27 back.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .brain import REGISTRY, Senses
from .brain.brain_env import POLICIES_DIR, onnx_infer
from .world import World, make_pitch
from .world.arena import (
    KICK_GOAL_S,  # noqa: F401  (the attribution window; the World keeps the counts)
)
from .world.metrics import (  # noqa: F401  (re-exported: tooling imports these from here)
    CARRY_S,
    GOAL_CREDIT_S,
    GOAL_FIELDS,
    METRIC_FIELDS,
    POSSESSION_R,
    POSSESSION_WIDE_R,
    SHAPE_FIELDS,
    SPIN_FIELDS,
    PitchMetrics,
    SpinMetrics,
)

# Every per-team dict a row may carry. A row written before one of them
# existed gets None for it on resume, never 0.0 (see `load_done`).
ROW_FIELDS = METRIC_FIELDS + GOAL_FIELDS + SHAPE_FIELDS


def run_one(seed: int, seconds: float, per_side: int = 1, walker: str | None = None,
            getup_s: float = 0.0, ball_out_s: float = 0.0, getup_policy: str | None = None,
            cove: float = 0.0, corner: float = 0.0) -> dict:
    from .brain.team import brain_kwargs, kickoff_brains, throw_in_brains
    sc = make_pitch(per_side=per_side, cove=cove, corner=corner)
    infer = onnx_infer(Path(walker) if walker else POLICIES_DIR / "alpha_walking.onnx")
    # A real get-up instead of the teleport stand-in (roadmap B.1): the
    # fallen duck is driven by this policy until it stands, and `--getup-s` is
    # the timeout rather than a fixed lie-down. `alpha_stand` is the one that
    # works (100% from back/front/side in 0.2-1.3 s on the bench).
    # A flag that changes nothing is broken, so this is checked and not assumed:
    # the file has to exist, `--getup-s` has to be > 0 (at 0 a fallen duck is
    # respawned on the tick it falls and is never handed to the policy at all,
    # which would run a whole battery of the BASELINE under a get-up flag), and
    # the constructed World has to be carrying it.
    getup_infer = None
    if getup_policy:
        if getup_s <= 0.0:
            raise SystemExit("--getup-policy needs --getup-s > 0: it is the get-up's timeout, and at 0 a "
                             "fallen duck is respawned on the tick it falls, so the policy never runs.")
        path = Path(getup_policy)
        if not path.exists():
            raise SystemExit(f"--getup-policy {getup_policy}: no such file")
        getup_infer = onnx_infer(path)
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed, getup_s=getup_s,
              ball_out_s=ball_out_s, getup_infer=getup_infer)
    assert (w.getup_infer is not None) == bool(getup_policy), "the get-up policy did not reach the World"
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    # A little seed-dependent asymmetry: nudge the ball off centre.
    rng = np.random.default_rng(seed)
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    metrics = PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})
    goal_seq = 0
    out_seq = w.ball_out_seq
    spin = SpinMetrics(w)
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
        if w.ball_out_seq != out_seq:           # the referee moved the ball: drop stale ball beliefs only
            out_seq = w.ball_out_seq
            throw_in_brains(brains, teams)
    score = w.soccer_score()
    return {"seed": seed, "perSide": per_side, "left": score["left"], "right": score["right"],
            "kickGoals": score["kicked"], "bumpGoals": score["bumped"],   # attributed by the World (KICK_GOAL_S)
            "ballOuts": score["ballOuts"],                                 # the ball-out rule's placements (0 unless --ball-out-s)
            "ballOutS": ball_out_s, "getupS": getup_s,   # the physics this row was measured under (`load_done` refuses to mix)
            "cove": cove, "corner": corner,              # …and the boards' geometry (Scenario.cove, make_pitch corner)
            "getups": w.getups, "getupTimeouts": w.getup_timeouts,   # falls it stood up from / ran the timeout out
            "getupDownS": w.getup_down_s,                # …and how long each spell on the floor lasted: a fall's PRICE
            "getupPolicy": getup_policy or "",           # which policy drove them ("" = the teleport stand-in)
            "kicks": {k: b.kicks for k, b in brains.items()}, "pushes": {k: b.pushes for k, b in brains.items()},
            "falls": {k: d.falls for k, d in w.ducks.items()}, "simSeconds": round(w.t, 1),
            "seconds": seconds,
            **spin.row(), **metrics.row()}


def load_done(path: str | None, tag: str, per_side: int, seconds: float,
              knobs: dict[str, float] | None = None, getup_policy: str | None = None) -> dict[int, dict]:
    """Seeds already measured into `path` (JSON lines, one row a seed), for a
    resume. A battery is the best part of an hour and this machine reclaims
    its container mid-run, so a killed run should cost the seed it was on and
    nothing else. Rows written under different settings are REFUSED rather
    than silently mixed: the brain's own parameters do not appear in a row,
    so `--tag` is how a caller says which variant a file belongs to.

    `knobs` are the WORLD's physics for this battery — `--ball-out-s`,
    `--getup-s` — checked the same way and for a sharper reason than the tag:
    they change what the ball and the ducks DO, and a caller who resumes with
    `uv run eval-pitch --seeds 12 --out runs/x.jsonl` after an interrupted
    `--ball-out-s 5` run would otherwise average seeds measured under a
    referee against seeds measured without one, in one file, silently. Every
    row written before a knob existed was measured at its default, so a
    missing field reads as that default rather than as "unknown".

    Resumable, not concurrency-safe: two batteries appending to the same
    file interleave, and a seed can land twice (identically — the loop is
    deterministic in the seed, so the duplicates agree and `done` keys by
    seed, but the line count then lies about how many seeds a file holds).
    One `--out` file per arm.

    A row written BEFORE the ball-progress/possession metrics existed is not
    refused — the goals, kicks and falls in it are as good as they ever were,
    and re-running an hour of seeds to recover metrics nobody measured then
    would be the wrong trade. Its missing metrics come back as None, which
    every consumer here treats as "not measured": `_seed_line` prints a dash
    and the summary averages the seeds that have the field and SAYS how many
    that was. Filling them with 0.0 instead would quietly drag a mean toward
    zero, which is the one outcome worth engineering against."""
    if not path or not os.path.exists(path):
        return {}
    done: dict[int, dict] = {}
    with open(path) as fh:
        lines = fh.readlines()
    last = len(lines)
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            # A battery is killed mid-write often enough here (this machine
            # reclaims its container) that the LAST line of the file can be
            # half a row. That seed simply was not measured: drop it and
            # resume, which is the whole point of the file. Anywhere else it
            # is corruption, and silently skipping rows would quietly shrink
            # a battery — so that is fatal, and says which line.
            if n == last:
                print(f"{path}:{n}: last line is truncated ({e.msg}); that seed will be re-run", flush=True)
                continue
            raise SystemExit(f"{path}:{n} is not valid JSON ({e.msg}). The file is corrupt, not merely "
                             "interrupted — a truncated row is only ever the last one.") from e
        if (r.get("tag", ""), r.get("perSide"), r.get("seconds")) != (tag, per_side, seconds):
            raise SystemExit(
                f"{path}:{n} was measured with tag={r.get('tag', '')!r} perSide={r.get('perSide')} "
                f"seconds={r.get('seconds')}, not tag={tag!r} perSide={per_side} seconds={seconds}. "
                "Write a different variant to a different file.")
        # …and the one knob that is not a number. A get-up arm and a respawn arm
        # in one file is the same silent mixing `knobs` exists to refuse, and it
        # is the exact pair this item's battery runs.
        if getup_policy is not None and (r.get("getupPolicy") or "") != getup_policy:
            raise SystemExit(
                f"{path}:{n} was measured with getupPolicy={(r.get('getupPolicy') or '')!r}, not "
                f"{getup_policy!r}. That is a different world, not a resume: pass the same flags, "
                "or write it to a different file.")
        for k, want in (knobs or {}).items():
            got = float(r.get(k) or 0.0)              # a row from before the knob: its default
            if got != float(want):
                raise SystemExit(
                    f"{path}:{n} was measured with {k}={got}, not {k}={float(want)}. That is a different "
                    "world, not a resume: pass the same flags, or write it to a different file.")
        for f in ROW_FIELDS:
            r.setdefault(f, None)
        r.setdefault("goalsUnattributed", None)
        done[int(r["seed"])] = r
    return done


def _total(r: dict, field: str) -> float | None:
    """A row's metric summed over both teams, or None if this row predates it."""
    v = r.get(field)
    return None if v is None else float(sum(v.values()))


def _mean_field(rows: list[dict], field: str) -> tuple[float | None, int]:
    """(mean over both teams' total, how many rows carried the field)."""
    vals = [v for v in (_total(r, field) for r in rows) if v is not None]
    return (float(np.mean(vals)) if vals else None), len(vals)


def _fmt(v: dict | None, unit: str) -> str:
    if v is None:
        return "—"
    return " ".join(f"{k} {x:+.2f}" if unit == "m/min" else f"{k} {x:.1f}" for k, x in sorted(v.items())) + f" {unit}"


def _ratio(num: dict | None, den: dict | None) -> str:
    """"back 8/27" — the count and what it is out of, both summed over teams.
    Events, per the playbook's first rule; a dash if the row predates them."""
    if num is None or den is None:
        return "—"
    return f"{int(sum(num.values()))}/{int(sum(den.values()))}"


def _seed_line(r: dict) -> str:
    own = r.get("ownGoals")
    return (f"seed {r['seed']}: goals left {r['left']} · right {r['right']} ({r['kickGoals']} kicked, {r['bumpGoals']} bumped)"
            f" · own {'—' if own is None else int(sum(own.values()))}"
            f" · kicks {sum(r['kicks'].values())} (back {_ratio(r.get('kicksBack'), r.get('kickCount'))})"
            f" · pushes {sum(r['pushes'].values())} · falls {r['falls']}"
            f" · progress {_fmt(r.get('ballProgress'), 'm/min')}"
            f" · possession {_fmt(r.get('possession'), 's/min')}")


def _getup_line(rows: list[dict]) -> str | None:
    """What the falls actually COST, which is the whole point of the flag:
    how many the duck stood up from, how many ran the timeout out, and the
    distribution of time spent on the floor. None when nobody ever went down
    (or the rows predate the counters), so the baseline prints nothing new."""
    ups = sum(int(r.get("getups") or 0) for r in rows)
    outs = sum(int(r.get("getupTimeouts") or 0) for r in rows)
    down = sorted(float(x) for r in rows for x in (r.get("getupDownS") or []))
    if not (ups or outs or down):
        return None
    n = ups + outs
    part = f"get-ups: {ups} stood, {outs} timed out"
    if n:
        part += f" ({ups / n:.0%} of {n})"
    if down:
        part += (f" · time down median {np.median(down):.2f} s"
                 f" (p90 {np.percentile(down, 90):.2f}, max {max(down):.2f}, n={len(down)})")
    return part


def _run_one_args(a: tuple) -> dict:
    return run_one(*a)


def _events(rows: list[dict], field: str) -> tuple[float, int]:
    """(total over both teams and every seed, seeds that carried the field)."""
    vals = [v for v in (_total(r, field) for r in rows) if v is not None]
    return float(np.sum(vals)), len(vals)


def _team_totals(rows: list[dict], field: str) -> dict[str, float]:
    """One team's field summed over the battery — the form an event count is
    read in (`ownGoals` left 3 right 5, not a mean of 4)."""
    out: dict[str, float] = {}
    for r in rows:
        for t, v in (r.get(field) or {}).items():
            if v is not None:
                out[t] = out.get(t, 0.0) + float(v)
    return out


def _team_means(rows: list[dict], field: str) -> dict[str, float]:
    """One team's field averaged over the seeds that carried it — the form a
    RATE or a mean is read in (`crowd`, `spread`, `ballOwnHalf`)."""
    acc: dict[str, list[float]] = {}
    for r in rows:
        for t, v in (r.get(field) or {}).items():
            if v is not None:
                acc.setdefault(t, []).append(float(v))
    return {t: float(np.mean(v)) for t, v in acc.items()}


def _print_ledger(rows: list[dict]) -> None:
    """The goal ledger and the shape numbers (roadmap Track 4.1), per TEAM.

    Per team and not summed, for the same reason `ballProgress` is a per-team
    metric: in self-play the two sides mirror each other and several of these
    are degenerate over the pair (`ballOwnHalf` sums to 60 s/min by
    construction, whatever the brains do). The battery that reads them is an
    asymmetric one — one side's roster changed, `eval_striker`-style — and
    there the per-team split IS the measurement.

    Counts are printed as "n of N": a difference in own goals or back-kicks
    means nothing without the total it came out of (the playbook's rule 1)."""
    if not any(r.get("ownGoals") for r in rows):
        return
    fors, againsts = _team_totals(rows, "goalsFor"), _team_totals(rows, "goalsAgainst")
    owns = _team_totals(rows, "ownGoals")
    unatt = sum(r.get("goalsUnattributed") or 0 for r in rows)
    print("goals: " + " · ".join(
        f"{t} for {fors.get(t, 0):.0f} against {againsts.get(t, 0):.0f} (own {owns.get(t, 0):.0f})"
        for t in sorted(owns)) + (f" · {unatt} unattributed" if unatt else ""))
    kicks, back = _team_totals(rows, "kickCount"), _team_totals(rows, "kicksBack")
    carry = _team_totals(rows, "kickCarry")
    if any(kicks.values()):
        print("kicks: " + " · ".join(
            f"{t} {kicks.get(t, 0):.0f} (back {back.get(t, 0):.0f}"
            + (f", carry {carry.get(t, 0.0) / kicks[t]:+.3f} m each)" if kicks.get(t) else ")")
            for t in sorted(kicks)))
    parts = []
    for f, unit in (("ballOwnHalf", "s/min"), ("spread", "m"), ("crowd", ""), ("depth", "m")):
        m = _team_means(rows, f)
        if not m:
            continue
        vals = " ".join((f"{t} {v:.0%}" if f == "crowd" else f"{t} {v:.2f}") for t, v in sorted(m.items()))
        parts.append(f"{f} {vals}" + (f" {unit}" if unit else ""))
    if parts:
        print("shape: " + " · ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first seed: --seeds 12 --seed0 12 EXTENDS a 12-seed battery instead of redoing it "
                         "(a promising result found on one set of seeds has to be confirmed on fresh ones)")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--per-side", type=int, default=1, help="ducks a side: 1 (1v1), 2, 3")
    ap.add_argument("--walker", default=None, metavar="PATH",
                    help="run this walk policy instead of the shipped alpha_walking.onnx "
                         "(roadmap 3.7: comparing two LOCALLY trained walkers, never one against the shipped one)")
    ap.add_argument("--getup-s", type=float, default=0.0, metavar="S",
                    help="a fallen duck lies where it fell for S seconds before it respawns (roadmap B.1: "
                         "falls cost time, the stand-in for a get-up policy); 0 = respawn at once, the baseline")
    ap.add_argument("--ball-out-s", type=float, default=0.0, metavar="S",
                    help="the referee's throw-in (roadmap Track 4 item 11b): a ball at rest against the boards for "
                         "S seconds is placed 0.45 m in; the lab's pitches play at 5. 0 = off, the benchmark's baseline "
                         "(the ball is at the boards 72%% of a 3v3 run and unkickable there: kicks 2.9 -> 7.7 a run at 5)")
    ap.add_argument("--cove", type=float, default=0.0, metavar="R",
                    help="a quarter-round cove of this radius (m) along the base of the boards, so a ball rolling "
                         "into a wall climbs it and rolls back out (Scenario.cove; the sim's wall is otherwise "
                         "dead, e = 0.06). 0 = flat boards, the benchmark's baseline")
    ap.add_argument("--corner", type=float, default=0.0, metavar="L",
                    help="chamfer each corner at 45 degrees starting L m along each wall from the corner point, "
                         "so a ball cannot wedge in it. 0 = square corners, the baseline")
    ap.add_argument("--getup-policy", default=None, metavar="ONNX",
                    help="drive a fallen duck with this policy until it stands, instead of teleporting it "
                         "(roadmap B.1); ../microduck/policies/alpha_stand.onnx is the one that works. "
                         "--getup-s becomes the timeout. Needs --getup-s > 0 to do anything.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", help="append each seed's result here as a JSON line AND resume from it: "
                                  "seeds already in the file are not re-run")
    ap.add_argument("--tag", default="", help="recorded in --out rows; a resume refuses to mix tags, so a file "
                                              "written under different brain settings is never reused by mistake")
    args = ap.parse_args()
    seeds = [args.seed0 + k for k in range(args.seeds)]
    # Each seed PRINTS as it finishes, rather than the battery printing at the
    # end: a 12-seed 3v3 battery is the best part of an hour, and a machine
    # that reclaims its container mid-run should cost one seed, not all of
    # them (it cost all of them, twice). Resume the rest with --seed0.
    if args.getup_policy and args.getup_s <= 0.0:
        raise SystemExit("--getup-policy needs --getup-s > 0: it is the get-up's timeout, and at 0 a fallen "
                         "duck is respawned on the tick it falls, so the policy never runs.")
    if args.getup_policy:
        print(f"get-up: fallen ducks driven by {Path(args.getup_policy).resolve()} "
              f"(timeout {args.getup_s:g} s)", flush=True)
    done = load_done(args.out, args.tag, args.per_side, args.seconds,
                     {"ballOutS": args.ball_out_s, "getupS": args.getup_s,
                      "cove": args.cove, "corner": args.corner},
                     getup_policy=args.getup_policy or "")
    rows = [done[sd] for sd in seeds if sd in done]
    if not args.json:
        for r in rows:
            print(_seed_line(r) + "  (already measured)", flush=True)
    todo = [sd for sd in seeds if sd not in done]
    out = open(args.out, "a") if args.out else None

    def keep(r: dict) -> None:
        rows.append(r)
        if out is not None:
            out.write(json.dumps({**r, "tag": args.tag}) + "\n")
            out.flush()
        if not args.json:
            print(_seed_line(r), flush=True)

    args_list = [(sd, args.seconds, args.per_side, args.walker, args.getup_s, args.ball_out_s,
                  args.getup_policy, args.cove, args.corner) for sd in todo]
    try:
        if args.jobs > 1 and len(todo) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
            with ctx.Pool(min(args.jobs, len(todo))) as pool:
                for r in pool.imap(_run_one_args, args_list):
                    keep(r)
        else:
            for a in args_list:
                keep(run_one(*a))
    finally:
        if out is not None:
            out.close()
    rows.sort(key=lambda r: r["seed"])
    if args.json:
        print(json.dumps(rows))
        return
    goals = [r["left"] + r["right"] for r in rows]
    print(f"{args.per_side}v{args.per_side}: mean goals {np.mean(goals):.2f}/run ({np.mean([r['kickGoals'] for r in rows]):.2f} kicked, "
          f"{np.mean([r['bumpGoals'] for r in rows]):.2f} bumped) · kicks "
          f"{np.mean([sum(r['kicks'].values()) for r in rows]):.1f}/run · pushes "
          f"{np.mean([sum(r['pushes'].values()) for r in rows]):.1f}/run"
          f" · falls {np.mean([sum(r['falls'].values()) for r in rows]):.2f}/run"
          f" (per duck {np.mean([sum(r['falls'].values()) for r in rows]) / (2 * args.per_side):.2f})")
    # The continuous metrics, both teams together (in self-play the two sides
    # run the same brain, so the pair's total is the run's number; the per-team
    # split in each seed line is what an asymmetric matchup is read from).
    parts, missing = [], 0
    for f, unit in (("ballProgress", "m/min"), ("ballAdvance", "m/min"),
                    ("possession", "s/min"), ("possessionWide", "s/min")):
        m, n = _mean_field(rows, f)
        missing = max(missing, len(rows) - n)
        parts.append(f"{f} {'—' if m is None else f'{m:+.2f}' if unit == 'm/min' else f'{m:.1f}'} {unit}")
    print("both teams: " + " · ".join(parts)
          + (f"   ({len(rows) - missing}/{len(rows)} seeds; the rest predate these metrics)" if missing else ""))
    gl = _getup_line(rows)
    if gl:
        print(gl)
    _print_ledger(rows)


if __name__ == "__main__":
    main()
