"""When the ball is rolling into a team's OWN goal, what does that team do?

    cd microduck_local
    uv run python scripts/probe_threat.py --seeds 24 --seconds 300 --per-side 2

WHY THIS EXISTS. Goals cannot judge a defensive change: the benchmark's own
power table (docs/roadmap.md 4.1.5) says goals need 136 seeds to resolve a
25% shift and `ownGoals` 347, on 19 events over 24 seeds. So a defensive idea
has to be judged on a PER-EVENT metric with far more events a battery, and
this builds one: the THREAT.

A threat is the situation the repo owner described watching a 2v2 — the ball
trickling toward a duck's own goal while the duck sets up a proper kick
instead of simply getting in the way. Made exact:

  * the ball is moving at least `--v-min` (0.05 m/s),
  * its straight-line path crosses the defending team's own goal line
    INSIDE the mouth (|y| < goal_width/2), and
  * it gets there within `--eta` seconds,
  * and all three have held continuously for `--arm` seconds.

That last clause is not tidiness. Without it half the events are one tick
long: a ball under a duck's foot has a velocity that points somewhere new
every frame, so the predicate flickers on and off and every flicker counts as
a threat that was instantly "cleared" — an instrument that scores contact
noise as defensive success. `--arm` 0.4 s costs nothing real (a genuine roll
at the mouth lasts seconds) and removes it.

There is no deceleration term in that predicate, and that is a MEASUREMENT,
not an omission. `ChaseParams.ball_decel` = 0.04 m/s^2 is the brain's model
of this floor; the floor itself has no rolling friction to speak of. Rolled
at 0.15 / 0.3 / 0.6 / 1.0 m/s, the ball drops at once to 0.597 of its launch
speed (the slide-to-roll transition) and then holds that speed for the six
seconds it takes to cross the pitch — 0.089, 0.179, 0.358, 0.597 m/s, flat
to the millimetre per second. **A ball rolling at the mouth in this sim does
not stop by itself; it is a goal unless somebody touches it.** That is what
makes "conceded" a fair verdict on the defence and not on the friction.

Outcomes, per threat: `conceded` (a goal in the mouth this team defends while
the threat is live), `cleared` (the path stops crossing the mouth for
`--hold` seconds — the ball was touched, deflected or turned round),
`reset` (a goal at the OTHER end teleported the ball) and `expired` (the run
ended first).

MEASURED WITH IT, so a reader knows what the numbers look like
(`runs/thr-base48b.jsonl`, 48 seeds x 300 s of 2v2): 71 threats, 1.5 a run,
**54 of 70 decided are conceded**; 40 of the 70 are declared with the ball
already inside 0.3 m of the goal line and 38 of those are conceded; a block
was geometrically available in 35 of 71 and in only 20 of the 54 conceded
ones. A defender within 0.15 m of the ball's path clears 16 of 33 and one
0.15-0.50 m off it concedes 31 of 31. `docs/roadmap.md` 4d has the rest,
including what happened when a block was built on it.

WHAT THE DEFENDER WAS DOING is the other half. Per threat the probe names the
best-placed defender — the defending duck nearest to the ball's own path
between it and the goal, which is the duck a block would fall to — and
records the fraction of the threat's ticks it spent in each brain state, plus
whether it could SEE the ball at all and what its tracker made of the ball's
velocity. That last part is the check that any fix is buildable: a brain
cannot act on a velocity it does not have, and the tracker's is differenced
from sightings that stop at `refresh_min`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.contract import CTRL_DT
from microduck_local.world import World, make_pitch

GOAL_INSET = 0.08          # World._check_goal: the ball crosses at |x| > hx - 0.08
# The danger clock: seconds a minute the ball spends this near a team's own
# goal line. A run-level continuous number, and unlike `ballOwnHalf` it is NOT
# degenerate over a self-play pair — the two goal areas do not partition the
# pitch, so the ball can be in neither and the pair's total is a real
# measurement of how much of the run was spent in front of somebody's goal.
DANGER_NEAR, DANGER_FAR = 0.45, 0.9


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _cross(bx: float, by: float, vx: float, vy: float, line_x: float,
           mouth: float) -> tuple[float, float] | None:
    """Where and when a ball at (bx, by) rolling at (vx, vy) crosses the plane
    x = `line_x`, if it does so inside +-`mouth` of the centre and while
    travelling toward it. (y at the crossing, seconds to get there) or None.

    Straight line, no decel: measured above, this floor has none."""
    if abs(vx) < 1e-9:
        return None
    eta = (line_x - bx) / vx
    if eta <= 0:
        return None                                  # going the other way
    ycross = by + vy * eta
    if abs(ycross) >= mouth:
        return None
    return ycross, eta


class Threat:
    """One live threat against one team."""

    def __init__(self, team: str, t: float, speed: float, dist: float, eta: float,
                 ycross: float, best: str | None, cover: float, ball_rng: float,
                 goal_side: bool, slack: float):
        self.team, self.t0 = team, t
        self.speed0, self.dist0, self.eta0, self.ycross0 = speed, dist, eta, ycross
        self.best, self.cover0, self.ball_rng0, self.goal_side0 = best, cover, ball_rng, goal_side
        self.slack0 = slack
        self.states: Counter = Counter()
        self.roles: Counter = Counter()
        self.ticks = 0
        self.seen = 0                 # ticks the best defender had a FRESH ball track
        self.tracked = 0              # …or any track at all
        self.verr: list[float] = []   # its tracker's ball-heading error vs truth, deg
        self.vratio: list[float] = []
        self.off_t: float | None = None      # when the predicate first went false
        self.outcome = "expired"
        self.end = t

    def row(self, seed: int) -> dict:
        n = max(self.ticks, 1)
        return {"kind": "threat", "seed": seed, "team": self.team, "t0": round(self.t0, 1),
                "dur": round(self.end - self.t0, 2), "outcome": self.outcome,
                "speed0": round(self.speed0, 3), "dist0": round(self.dist0, 3),
                "eta0": round(self.eta0, 2), "ycross0": round(self.ycross0, 3),
                "best": self.best, "cover0": round(self.cover0, 3),
                "ballRng0": round(self.ball_rng0, 3), "goalSide0": self.goal_side0,
                "slack0": None if not np.isfinite(self.slack0) else round(self.slack0, 2),
                "ticks": self.ticks, "seenFrac": round(self.seen / n, 3),
                "trackFrac": round(self.tracked / n, 3),
                "velErrDeg": None if not self.verr else round(float(np.median(np.abs(self.verr))), 1),
                "velRatio": None if not self.vratio else round(float(np.median(self.vratio)), 2),
                "states": {k: round(v / n, 3) for k, v in self.states.items()},
                "roles": {k: round(v / n, 3) for k, v in self.roles.items()}}


def run(seed: int, seconds: float, per_side: int, v_min: float, eta_max: float,
        hold: float, arm: float = 0.4, zone: float = 1.0, roles: str | None = None) -> list[dict]:
    sc = make_pitch(per_side=per_side)
    if roles:
        # Static jobs, the same on both sides, in spawn order: `--roles
        # keeper,striker` makes d0/d2 keepers and d1/d3 strikers.
        names = [r.strip() for r in roles.split(",")]
        assert len(names) == per_side, f"--roles needs {per_side} names, got {names}"
        for i, d in enumerate(sc.ducks):
            d.role = names[i % per_side]
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    vadr = int(w.model.jnt_dofadr[w._ball_joint])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)

    hx = sc.floor[0] / 2 - 0.25
    line_x = hx - GOAL_INSET
    mouth = sc.goal_width / 2
    # Each team's own goal line and the mouth key the World writes a goal
    # against there ("right" for +x). Read off `goal_for`, never guessed: the
    # mouth names and the team names are two different words for a reason
    # (`eval_pitch`'s warning).
    sides: dict[str, tuple[float, str]] = {}
    members: dict[str, list[str]] = {}
    for d in sc.ducks:
        g = w.goal_for(w.ducks[d.id])
        team = d.team or d.id
        own_x = -line_x if g[0] >= 0 else line_x
        sides[team] = (own_x, "right" if own_x > 0 else "left")
        members.setdefault(team, []).append(d.id)

    live: dict[str, Threat] = {}
    arming: dict[str, float] = {}          # team → when the predicate first went true
    out: list[dict] = []
    inc: dict[str, list] = {}              # team → the incursion under way [t0, min dist, ticks]
    incs: list[dict] = []
    danger = {t: [0, 0] for t in sides}    # ticks the ball spent inside DANGER_NEAR / DANGER_FAR
    steps = 0
    goal_seq = 0
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill,
                       bumped=w.bumped(d))
            intent = brains[d.id].step(s)
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()

        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            for team, th in list(live.items()):
                th.end = w.t
                th.outcome = "conceded" if w.last_goal == sides[team][1] else "reset"
                out.append(th)
            for team, v in list(inc.items()):
                incs.append(_inc_row(seed, team, w.t, v,
                                     "goal" if w.last_goal == sides[team][1] else "reset"))
            live.clear()
            arming.clear()
            inc.clear()
            kickoff_brains(brains, teams, w)
            continue
        if w.in_kickoff:
            continue
        steps += 1

        bx, by = w.ball_xy()
        vx, vy = float(w.data.qvel[vadr]), float(w.data.qvel[vadr + 1])
        speed = math.hypot(vx, vy)
        # The INCURSION, and the danger clock. The strict threat above is the
        # honest question ("is this a goal unless somebody moves") but it is
        # rare, because a 0.70 m mouth subtends +-7 deg from midfield and a
        # ball's line is almost never that good until it is nearly in. This
        # is the plentiful companion: every visit the ball pays to a team's
        # goal area, how near it got, and whether it went in.
        for team, (own_x, _k) in sides.items():
            # Distance to the MOUTH, not to the goal line: a ball in the
            # corner has reached the line and is not dangerous, and reading
            # the line put half the incursions at "nearest 0.000 m" while
            # only a third were goals.
            d = math.hypot(own_x - bx, by)
            if d <= DANGER_NEAR:
                danger[team][0] += 1
            if d <= DANGER_FAR:
                danger[team][1] += 1
            v = inc.get(team)
            if v is None:
                if d <= zone:
                    inc[team] = [w.t, d, 0, Counter()]
            else:
                v[1] = min(v[1], d)
                v[2] += 1
                for did in members[team]:
                    v[3][brains[did].state] += 1
                if d > zone + 0.25:
                    incs.append(_inc_row(seed, team, w.t, v, "left"))
                    del inc[team]
        for team, (own_x, _mouth_key) in sides.items():
            hit = _cross(bx, by, vx, vy, own_x, mouth) if speed >= v_min else None
            on = hit is not None and hit[1] <= eta_max
            th = live.get(team)
            if th is None:
                # Arming: the predicate has to hold for `arm` before this is a
                # threat and not a ball twitching under someone's foot.
                if not on:
                    arming.pop(team, None)
                    continue
                t0 = arming.setdefault(team, w.t)
                if w.t - t0 < arm:
                    continue
                arming.pop(team, None)
                ycross, eta = hit
                best, cover, brng, gs = _best_placed(w, members[team], bx, by, vx, vy, speed, own_x)
                slack = _slack(w, members[team], bx, by, vx, vy, speed, own_x)
                live[team] = Threat(team, t0, speed, abs(own_x - bx), eta, ycross,
                                    best, cover, brng, gs, slack)
                th = live[team]
            elif on:
                th.off_t = None
            else:
                if th.off_t is None:
                    th.off_t = w.t
                elif w.t - th.off_t >= hold:
                    th.end, th.outcome = th.off_t, "cleared"
                    out.append(th)
                    del live[team]
                    continue
            _sample(th, w, brains, bx, by, vx, vy, speed)
            th.end = w.t
    for th in live.values():
        out.append(th)
    for team, v in inc.items():
        incs.append(_inc_row(seed, team, w.t, v, "end"))
    rows = [th.row(seed) for th in out] + incs
    mins = max(steps * CTRL_DT / 60.0, 1e-9)
    for team, (near, far) in danger.items():
        rows.append({"kind": "run", "seed": seed, "team": team,
                     "nearSecPerMin": round(near * CTRL_DT / mins, 3),
                     "farSecPerMin": round(far * CTRL_DT / mins, 3)})
    return rows


def _inc_row(seed: int, team: str, t: float, v: list, how: str) -> dict:
    """One visit the ball paid to a team's goal area: how near it got, how
    long it stayed, what the team was doing, and how it ended.

    `nearest` is the instrument that the strict threat cannot be — a
    continuous number with hundreds of events a battery, where "conceded" is
    a binary bounded by the goal count and therefore by goals' own power."""
    t0, near, ticks, states = v
    n = max(ticks, 1)
    return {"kind": "incursion", "seed": seed, "team": team, "t0": round(t0, 1),
            "dur": round(t - t0, 2), "nearest": round(near, 3), "how": how,
            "ticks": ticks,
            "states": {k: round(c / n, 3) for k, c in states.most_common(8)}}


def _best_placed(w, ids: list[str], bx: float, by: float, vx: float, vy: float,
                 speed: float, own_x: float) -> tuple[str | None, float, float, bool]:
    """The defending duck a block would fall to: the one nearest to the ball's
    own path between the ball and the goal line. Also its range to the ball
    and whether it is goal-side of the ball (in the way already)."""
    ux, uy = vx / speed, vy / speed
    reach = abs(own_x - bx) / max(abs(ux), 1e-6)
    best, bestd, bestr, bestg = None, math.inf, math.inf, False
    for did in ids:
        px, py, _ = w.odom(w.ducks[did])
        s = float(np.clip((px - bx) * ux + (py - by) * uy, 0.0, reach))
        d = math.hypot(bx + s * ux - px, by + s * uy - py)
        if d < bestd:
            best, bestd = did, d
            bestr = math.hypot(px - bx, py - by)
            bestg = s > 0.0
    return best, bestd, bestr, bestg


# The walker, as `brain/team.py` measured it: 0.45 m/s, 0.7 rad/s in place
# once the gait is going, 0.4 s of cold start, and 0.5 rad of bearing the
# walk's own steering absorbs for free.
WALK, TURN_RATE, TURN_FREE, COLD_S = 0.45, 0.7, 0.5, 0.4


def _walk_time(px: float, py: float, pyaw: float, tx: float, ty: float) -> float:
    """Seconds for a duck at (px, py, pyaw) to stand on (tx, ty)."""
    bear = abs(_wrap(math.atan2(ty - py, tx - px) - pyaw))
    turn = max(0.0, bear - TURN_FREE)
    return turn / TURN_RATE + (COLD_S if turn > 0 else 0.0) + math.hypot(tx - px, ty - py) / WALK


def _slack(w, ids: list[str], bx: float, by: float, vx: float, vy: float,
           speed: float, own_x: float) -> float:
    """COULD anybody have stood in the way? The best margin, over every
    defending duck and every point on the ball's path between it and the goal
    line, of (the ball's time to that point) minus (the duck's walk time to
    it). Positive means a block was geometrically available and was not made;
    -inf means nobody could have reached the path anywhere.

    This is the number that says whether a defensive behaviour has anything to
    work with, and it is deliberately generous to the duck: it grants perfect
    knowledge of the ball's velocity and ignores the ball's own radius."""
    ux, uy = vx / speed, vy / speed
    reach = abs(own_x - bx) / max(abs(ux), 1e-6)
    best = -math.inf
    for did in ids:
        px, py, pyaw = w.odom(w.ducks[did])
        for s in np.linspace(0.0, reach, 25):
            m = float(s) / speed - _walk_time(px, py, pyaw, bx + s * ux, by + s * uy)
            if m > best:
                best = m
    return best


def _sample(th: Threat, w, brains: dict, bx: float, by: float, vx: float, vy: float,
            speed: float) -> None:
    """One tick of "what was the best-placed defender doing": its state, its
    role, whether it saw the ball, and what its own tracker made of the ball's
    velocity (the check that a fix is buildable from the duck's own senses)."""
    th.ticks += 1
    if th.best is None:
        return
    b = brains[th.best]
    th.states[b.state] += 1
    th.roles[getattr(b, "role", "?")] += 1
    tr = b.tracker.best("ball", w.t, min_hits=1)
    if tr is None:
        return
    age = tr.age(w.t)
    if age < b.p.lost_s:
        th.tracked += 1
    if age <= b.DET_MAX_AGE:
        th.seen += 1
    if tr.vel_hits >= 2:
        tvx, tvy = tr.vel
        tsp = math.hypot(tvx, tvy)
        if tsp > 1e-3:
            th.verr.append(math.degrees(_wrap(math.atan2(tvy, tvx) - math.atan2(vy, vx))))
            th.vratio.append(tsp / max(speed, 1e-6))


def _pct(n: int, d: int) -> str:
    return "—" if not d else f"{n}/{d} = {n / d:.0%}"


def summarise(all_rows: list[dict], seeds: int, seconds: float, per_side: int) -> None:
    rows = [r for r in all_rows if r.get("kind", "threat") == "threat"]
    incs = [r for r in all_rows if r.get("kind") == "incursion"]
    runs = [r for r in all_rows if r.get("kind") == "run"]
    n = len(rows)
    if not n:
        print("no threats")
        _incursions(incs, runs, seeds)
        return
    by = Counter(r["outcome"] for r in rows)
    decided = by["conceded"] + by["cleared"]
    print(f"{n} threats over {seeds} seeds x {seconds:g} s of {per_side}v{per_side} "
          f"({n / seeds:.1f} a run)\n")
    print(f"{'outcome':<12}{'n':>6}{'of decided':>13}")
    for k in ("conceded", "cleared", "reset", "expired"):
        share = f"{by[k] / decided:.0%}" if (decided and k in ("conceded", "cleared")) else "—"
        print(f"{k:<12}{by[k]:>6}{share:>13}")
    print(f"\nCONCEDED FRACTION  {_pct(by['conceded'], decided)}   "
          f"(of decided threats; `reset` and `expired` are neither)")

    sp = np.array([r["speed0"] for r in rows])
    et = np.array([r["eta0"] for r in rows])
    di = np.array([r["dist0"] for r in rows])
    cv = np.array([r["cover0"] for r in rows])
    print("\nat the moment a threat starts (median, IQR):")
    print(f"  ball speed      {np.median(sp):.3f} m/s   {np.percentile(sp, 25):.3f}-{np.percentile(sp, 75):.3f}")
    print(f"  seconds to line {np.median(et):.2f} s     {np.percentile(et, 25):.2f}-{np.percentile(et, 75):.2f}")
    print(f"  ball to line    {np.median(di):.2f} m     {np.percentile(di, 25):.2f}-{np.percentile(di, 75):.2f}")
    print(f"  best defender off the ball's path {np.median(cv):.2f} m   "
          f"{np.percentile(cv, 25):.2f}-{np.percentile(cv, 75):.2f}")
    gs = sum(1 for r in rows if r["goalSide0"])
    print(f"  already goal-side of the ball: {_pct(gs, n)}")
    sl = [r["slack0"] for r in rows if r.get("slack0") is not None]
    if sl:
        ok = sum(1 for v in sl if v > 0)
        conc = [r for r in rows if r["outcome"] == "conceded" and r.get("slack0") is not None]
        cok = sum(1 for r in conc if r["slack0"] > 0)
        print(f"  a block was AVAILABLE (some duck could reach the path before the ball): "
              f"{_pct(ok, len(sl))}; of the conceded ones {_pct(cok, len(conc))}")
        print(f"  best margin  median {np.median(sl):+.2f} s   "
              f"{np.percentile(sl, 25):+.2f}..{np.percentile(sl, 75):+.2f}")

    # What the defender was DOING, weighted by the ticks of the threat it was
    # doing it for — over all threats, and over the conceded ones alone.
    for label, sel in (("all threats", rows),
                       ("conceded only", [r for r in rows if r["outcome"] == "conceded"]),
                       ("cleared only", [r for r in rows if r["outcome"] == "cleared"])):
        if not sel:
            continue
        acc: Counter = Counter()
        for r in sel:
            for k, v in r["states"].items():
                acc[k] += v * r["ticks"]
        tot = sum(acc.values()) or 1
        top = " · ".join(f"{k} {v / tot:.0%}" for k, v in acc.most_common(7))
        print(f"\nbest-placed defender's state, {label} ({len(sel)}):\n  {top}")
        rl: Counter = Counter()
        for r in sel:
            for k, v in r["roles"].items():
                rl[k] += v * r["ticks"]
        rt = sum(rl.values()) or 1
        print("  role: " + " · ".join(f"{k} {v / rt:.0%}" for k, v in rl.most_common()))

    seen = np.array([r["seenFrac"] for r in rows])
    trk = np.array([r["trackFrac"] for r in rows])
    ve = [r["velErrDeg"] for r in rows if r["velErrDeg"] is not None]
    vr = [r["velRatio"] for r in rows if r["velRatio"] is not None]
    print(f"\ncould the best-placed defender SEE it? fresh detection on "
          f"{seen.mean():.0%} of threat ticks, any live track {trk.mean():.0%}")
    if ve:
        print(f"  its tracker's ball VELOCITY, where it had one ({len(ve)} of {n} threats): "
              f"heading off truth by a median {np.median(ve):.0f} deg, "
              f"speed {np.median(vr):.2f}x truth")
    _incursions(incs, runs, seeds)
    print("\nREAD IT AS: the conceded fraction is the number to move, counted in EVENTS. "
          "`reset` threats end when the other team scores and say nothing about this "
          "defence; `expired` ones ran into the end of the run.")


def _incursions(incs: list[dict], runs: list[dict], seeds: int) -> None:
    """The plentiful companion. A strict threat is "this is a goal unless
    somebody moves", which is the right question and a rare event — 70
    decided over 48 seeds, with 54 of them conceded, so the ratio carries
    barely more than the goal count it is bounded by. An INCURSION is every
    visit the ball pays to a goal area, and `nearest` (how close it got) is
    continuous, so hundreds of events a battery each carry information."""
    if not incs:
        return
    d = np.array([r["nearest"] for r in incs])
    dur = np.array([r["dur"] for r in incs])
    goals = sum(1 for r in incs if r["how"] == "goal")
    print(f"\nINCURSIONS — every visit the ball paid to a goal area ({len(incs)}, "
          f"{len(incs) / seeds:.1f} a run):")
    print(f"  nearest the ball got to the mouth       mean {d.mean():.3f} m   "
          f"median {np.median(d):.3f}   sd {d.std(ddof=1):.3f}")
    print(f"  how long it stayed                      mean {dur.mean():.2f} s")
    print(f"  ended in a goal                         {_pct(goals, len(incs))}")
    acc: Counter = Counter()
    for r in incs:
        for k, v in r["states"].items():
            acc[k] += v * r["ticks"]
    tot = sum(acc.values()) or 1
    print("  the defending TEAM's states while it was there: "
          + " · ".join(f"{k} {v / tot:.0%}" for k, v in acc.most_common(7)))
    if runs:
        near = np.array([r["nearSecPerMin"] for r in runs])
        far = np.array([r["farSecPerMin"] for r in runs])
        print(f"  danger clock (both teams, s/min the ball is that near SOME mouth): "
              f"within {DANGER_NEAR} m {near.sum() / seeds:.2f} · "
              f"within {DANGER_FAR} m {far.sum() / seeds:.2f}")


def _z2(a: int, na: int, b: int, nb: int) -> str:
    """Two proportions, pooled z — the form `kicksBack` is read in, and the
    reason a per-event fraction resolves where a per-run mean cannot."""
    if not na or not nb:
        return "—"
    pa, pb = a / na, b / nb
    pool = (a + b) / (na + nb)
    se = math.sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
    if se < 1e-12:
        return "p = 1.00"
    z = (pb - pa) / se
    from math import erfc
    return f"z = {z:+.2f}, p = {erfc(abs(z) / math.sqrt(2)):.3f}"


def _paired(name: str, per_seed: dict[int, list[float]], other: dict[int, list[float]],
            unit: str, how: str = "mean", good: int = -1) -> None:
    """Student's t on the per-seed difference over the SHARED seeds. Paired,
    because both arms ran the same layouts and comparing two means throws
    most of the power away; Student's, because at these sizes 1.96
    manufactures significance (the playbook's reading rules)."""
    seeds = sorted(set(per_seed) & set(other))
    if len(seeds) < 3:
        print(f"  {name:<34} too few shared seeds")
        return
    f = np.mean if how == "mean" else np.sum
    a = np.array([f(per_seed[s]) if per_seed[s] else np.nan for s in seeds])
    b = np.array([f(other[s]) if other[s] else np.nan for s in seeds])
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        print(f"  {name:<34} too few shared seeds")
        return
    d = b - a
    se = d.std(ddof=1) / math.sqrt(len(d))
    from statistics import NormalDist
    t = d.mean() / se if se > 1e-12 else 0.0
    # Student's t, two-sided, via the normal only as a last resort.
    try:
        from scipy import stats  # noqa: PLC0415
        pv = float(2 * stats.t.sf(abs(t), len(d) - 1))
    except Exception:
        pv = float(2 * (1 - NormalDist().cdf(abs(t))))
    wins = int((d < 0).sum() if good < 0 else (d > 0).sum())
    print(f"  {name:<34} {a.mean():8.3f} -> {b.mean():8.3f} {unit:<7} "
          f"delta {d.mean():+.3f} +/- {se:.3f}  t = {t:+.2f}, p = {pv:.3f}  "
          f"(better on {wins} of {len(d)} seeds)")


def compare(pa: str, pb: str) -> None:
    """Two probe files, read the way the playbook says to read them."""
    A = [json.loads(x) for x in open(pa) if x.strip()]
    B = [json.loads(x) for x in open(pb) if x.strip()]

    def kind(rows, k):
        return [r for r in rows if r.get("kind", "threat") == k]

    for lbl, rows in ((pa, A), (pb, B)):
        th = kind(rows, "threat")
        d = [r for r in th if r["outcome"] in ("conceded", "cleared")]
        print(f"{lbl}: {len(th)} threats, {len(kind(rows, 'incursion'))} incursions, "
              f"{len(set(r['seed'] for r in rows))} seeds, "
              f"{sum(1 for r in d if r['outcome'] == 'conceded')}/{len(d)} conceded")
    seeds = sorted(set(r["seed"] for r in A) & set(r["seed"] for r in B))
    A = [r for r in A if r["seed"] in seeds]
    B = [r for r in B if r["seed"] in seeds]
    print(f"\nshared seeds: {len(seeds)}")
    da = [r for r in kind(A, "threat") if r["outcome"] in ("conceded", "cleared")]
    db = [r for r in kind(B, "threat") if r["outcome"] in ("conceded", "cleared")]
    ca = sum(1 for r in da if r["outcome"] == "conceded")
    cb = sum(1 for r in db if r["outcome"] == "conceded")
    print(f"\nCONCEDED THREATS   {ca}/{len(da)} = {ca / max(len(da), 1):.0%}  ->  "
          f"{cb}/{len(db)} = {cb / max(len(db), 1):.0%}   {_z2(ca, len(da), cb, len(db))}")
    far_a = [r for r in da if r["dist0"] >= 0.62]
    far_b = [r for r in db if r["dist0"] >= 0.62]
    fa = sum(1 for r in far_a if r["outcome"] == "conceded")
    fb = sum(1 for r in far_b if r["outcome"] == "conceded")
    print(f"  …of the ones a block could reach (ball 0.62 m+ out, the clearance a "
          f"block needs):\n     {fa}/{len(far_a)}  ->  {fb}/{len(far_b)}   "
          f"{_z2(fa, len(far_a), fb, len(far_b))}")
    print("\npaired per seed. THE GOOD DIRECTION IS NOT THE SAME FOR EVERY ROW: for "
          "`nearest`\nfurther from the mouth is better, for everything else less is "
          "better. The\n'better on N seeds' column is signed per row, so read that "
          "rather than the delta.")

    def bucket(rows, k, field, transform=None):
        fn = transform if transform is not None else (lambda r: r[field])
        out: dict[int, list[float]] = {s: [] for s in seeds}
        for r in kind(rows, k):
            out[r["seed"]].append(fn(r))
        return out

    _paired("incursion: nearest the mouth", bucket(A, "incursion", "nearest"),
            bucket(B, "incursion", "nearest"), "m", good=+1)
    _paired("incursion: ended in a goal",
            bucket(A, "incursion", "how", lambda r: 1.0 * (r["how"] == "goal")),
            bucket(B, "incursion", "how", lambda r: 1.0 * (r["how"] == "goal")), "/inc")
    _paired("incursions a run", bucket(A, "incursion", "dur", lambda r: 1.0),
            bucket(B, "incursion", "dur", lambda r: 1.0), "n", how="sum")
    _paired(f"danger clock (< {DANGER_NEAR} m)", bucket(A, "run", "nearSecPerMin"),
            bucket(B, "run", "nearSecPerMin"), "s/min", how="sum")
    _paired(f"danger clock (< {DANGER_FAR} m)", bucket(A, "run", "farSecPerMin"),
            bucket(B, "run", "farSecPerMin"), "s/min", how="sum")
    _paired("threats a run", bucket(A, "threat", "dur", lambda r: 1.0),
            bucket(B, "threat", "dur", lambda r: 1.0), "n", how="sum")


def _run_args(a: tuple) -> list[dict]:
    return run(*a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first seed: --seeds 24 --seed0 100 confirms on fresh layouts")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--v-min", type=float, default=0.05, help="a ball slower than this is parked")
    ap.add_argument("--eta", type=float, default=8.0, help="seconds to the line that count as a threat")
    ap.add_argument("--hold", type=float, default=0.5,
                    help="the path must miss the mouth this long before a threat counts as cleared")
    ap.add_argument("--zone", type=float, default=1.0,
                    help="a ball this near a team's own goal MOUTH is an incursion")
    ap.add_argument("--arm", type=float, default=0.4,
                    help="the path must hold on the mouth this long before it counts as a threat at all")
    ap.add_argument("--out", default=None, help="write every event as a JSON line")
    ap.add_argument("--roles", default=None,
                    help="static jobs per side in spawn order, both sides alike, e.g. 'keeper,striker' "
                         "or 'defender,striker'; default: none (the role-free chase-vs-chase control)")
    ap.add_argument("--vs", nargs=2, metavar=("BASE", "ARM"),
                    help="do not measure: read two --out files and compare them, paired on the "
                         "seeds they share")
    args = ap.parse_args()
    if args.vs:
        compare(*args.vs)
        return
    todo = [(args.seed0 + k, args.seconds, args.per_side, args.v_min, args.eta, args.hold,
             args.arm, args.zone, args.roles) for k in range(args.seeds)]
    print(f"roles={args.roles!r}  MICRODUCK_CHASE={os.environ.get('MICRODUCK_CHASE', '')!r}")
    rows: list[dict] = []
    if args.jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(min(args.jobs, len(todo))) as pool:
            for r in pool.imap_unordered(_run_args, todo):
                rows += r
    else:
        for a in todo:
            rows += run(*a)
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    summarise(rows, args.seeds, args.seconds, args.per_side)


if __name__ == "__main__":
    main()
