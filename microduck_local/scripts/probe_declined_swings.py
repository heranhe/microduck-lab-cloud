"""THE DECLINED SWING: does the corrected kick sidecar refuse swings, and were
they worth taking? (roadmap 12au's "what settles it next" (3), 2026-09-10)

12au measured the corrected left-foot sidecar (`exit_rad` -0.225 -> +0.209) on
the 2v2 ledger and found the kick COUNT fall 327 -> 249 over 48 paired seeds
(-24%, p 0.001) while carry per kick rose 0.332 -> 0.511 m and total carry held.
It read that as "the selector declines about a quarter of its swings" and asked
whether the declined ones were worth taking. That reading has two testable
halves, and this file measures both on the gym's population:

  1. *Does the SELECTOR decline?* `kickselect.select` returns None only when
     EVERY candidate line puts more than `kick_select_t_own` of its samples in
     our own net — and `Chase._plan` does not treat that as "no kick": it keeps
     the clamp's line and its own foot and swings anyway. So a selector
     "decline" cannot remove a swing at all. The only path in the brain that
     refuses a settled swing is `declines` (`_too_wide` / `_too_far`), which
     reads the BELIEVED ball offset and never reads the exit.
  2. *Does the aim window bind?* The candidate fan is `los +- aim_max`, and the
     exit enters the roll-out as `u + exit`, so a foot whose exit moved 0.43 rad
     needs a line 0.43 rad the other way to send the ball where it used to. If
     that line is outside the window, the selector must take a worse one — the
     mechanism 12au's question names. `detour` (the chosen line's offset from
     the line of sight) against `aim_max` says whether it is ever pinned.

So every row here is an EVENT with the selector's own price on it, plus the
COUNTERFACTUAL: at every plan the brain makes, the same candidate fan is
re-scored under the OTHER arm's exit on the SAME random stream (the generator's
state is snapshotted and restored, so the live run is bit-identical to an
un-probed one — `--selfcheck` asserts exactly that), and the row carries what
the other model would have chosen. That answers "for the swings this arm took
that the other model would have declined, what did they actually do" without
re-running anything.

    uv run python scripts/probe_declined_swings.py --episodes 40 --seeds 12 \
        --jobs 3 --label w12 --out runs/declined/w12-b0.jsonl
    uv run python scripts/probe_declined_swings.py --selfcheck      # determinism

The scenario, the placement and the swing detection are `scripts/kick_gym.py`'s,
imported rather than copied, so this probe's population is that instrument's
population and nothing here can drift away from it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kick_gym import (  # noqa: E402  (the gym IS the population; never a copy)
    EPISODE_S,
    _board_rect,
    _drive,
    _place,
    _place_at_boards,
    gym_scenario,
)

from microduck_local.brain import REGISTRY  # noqa: E402
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer  # noqa: E402
from microduck_local.brain.controllers import Chase  # noqa: E402
from microduck_local.world.arena import World  # noqa: E402

# How long after an event the ball's fate is read. 4 s, not the ledger's 2 s
# carry window: a DECLINE is not a touch, and the question about it is what
# happened to the ball next — the duck walks round and swings again, or the
# ball sits. Two seconds is shorter than one re-approach.
FATE_S = 4.0


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# --- the counterfactual patch -------------------------------------------------
# Module state, not an argument: `_select_kick_line` is a method on the brain and
# the probe has no other seam into it. One process runs one seed, so this is
# per-seed state and never shared.
_CF: dict = {"exits": None, "log": None, "orig": None}


def _install(cf_exits: tuple[float, float] | None):
    """Wrap `Chase._select_kick_line` so every call logs what the selector
    priced, and — with `cf_exits` — what the OTHER exit model would have
    chosen from the identical candidate fan on the identical random stream.

    The live call runs FIRST and its post-state is restored afterwards, so the
    counterfactual consumes no randomness the brain would have had. The rng is
    force-created here exactly as the brain creates it, so the snapshot exists
    before the first call."""
    # Idempotent: a second `run` in one process must wrap the PRISTINE method,
    # not the patch (nesting it triples the census and was caught by
    # `--selfcheck` before any block was run).
    if _CF["orig"] is None:
        _CF["orig"] = Chase._select_kick_line
    orig = _CF["orig"]
    _CF["exits"] = cf_exits

    def patched(self, odom, ball_xy, los, u_clamp):
        if self._kick_rng is None:
            self._kick_rng = np.random.default_rng(zlib.crc32(self.duck_id.encode() or b"duck"))
        st = self._kick_rng.bit_generator.state
        out = orig(self, odom, ball_xy, los, u_clamp)
        v = self.last_select
        post = self._kick_rng.bit_generator.state
        cf_out = cf_v = None
        if cf_exits is not None:
            p0 = self.p
            self.p = replace(p0, kick_exit_left=cf_exits[0], kick_exit_right=cf_exits[1])
            self._kick_rng.bit_generator.state = st
            try:
                cf_out = orig(self, odom, ball_xy, los, u_clamp)
                cf_v = self.last_select
            finally:
                self.p = p0
                self._kick_rng.bit_generator.state = post
                self.last_select = v
        log = _CF.get("log")
        if log is not None:
            gx, gy = (self.goal if self.goal is not None else (0.0, 0.0))
            log.append({
                "los": round(float(los), 4),
                "u_clamp": round(float(u_clamp), 4),
                "aim_max": round(float(self.p.aim_max), 4),
                # Where the goal is FROM THE BALL, in the same odom frame as
                # `los`: the line the selector would take if nothing bent it.
                "goal_dir": round(float(math.atan2(gy - ball_xy[1], gx - ball_xy[0])), 4),
                "goal_cone": round(float(self.goal_cone(*ball_xy)), 4),
                "ball": [round(float(ball_xy[0]), 4), round(float(ball_xy[1]), 4)],
                "none": out is None,
                "foot": None if v is None else str(v.foot),
                "u": None if out is None else round(float(out[0]), 4),
                "detour": None if out is None else round(float(_wrap(out[0] - los)), 4),
                "p_own": None if v is None else round(float(v.p_own), 4),
                "p_goal": None if v is None else round(float(v.p_goal), 4),
                "value": None if v is None else round(float(v.value), 4),
                "cf_none": None if cf_exits is None else (cf_out is None),
                "cf_foot": None if cf_v is None else str(cf_v.foot),
                "cf_u": None if (cf_exits is None or cf_out is None) else round(float(cf_out[0]), 4),
                "cf_detour": None if (cf_exits is None or cf_out is None)
                else round(float(_wrap(cf_out[0] - los)), 4),
                "cf_p_own": None if cf_v is None else round(float(cf_v.p_own), 4),
                "cf_p_goal": None if cf_v is None else round(float(cf_v.p_goal), 4),
                "cf_value": None if cf_v is None else round(float(cf_v.value), 4),
            })
        return out

    Chase._select_kick_line = patched
    return orig


def _ball_xy(w: World, q: int) -> tuple[float, float]:
    return (float(w.data.qpos[q]), float(w.data.qpos[q + 1]))


def _uninstall() -> None:
    """Put the PRISTINE `Chase._select_kick_line` back and forget the patch.

    The wrapper runs the selector twice per call when a counterfactual is
    asked for; left installed after a `run` it doubles the fan for every
    later `Chase` in the process — which is how `tests/test_spot_reach.py`
    went red on CI after this probe's tests ran before it (2026-09-11)."""
    if _CF["orig"] is not None:
        Chase._select_kick_line = _CF["orig"]
    _CF["orig"] = None
    _CF["exits"] = None
    _CF["log"] = None


def run(*args, **kwargs):
    """One seed of the probe; the selector patch never outlives the call."""
    try:
        return _run_patched(*args, **kwargs)
    finally:
        _uninstall()


def _run_patched(seed: int, episodes: int, spread: float, at_boards: float,
        cf_exits: tuple[float, float] | None, knobs: str = "", opponents: int = 0) -> list[dict]:
    """One seed. One row per EVENT (a swing or a decline) plus one row per
    episode, so a rate has the denominator its numerator came from."""
    if knobs:
        os.environ["MICRODUCK_CHASE"] = knobs
    else:
        os.environ.pop("MICRODUCK_CHASE", None)
    _install(cf_exits)
    sc = gym_scenario(opponents=opponents)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={x.id: infer for x in sc.ducks}, seed=seed)
    bk = __import__("microduck_local.brain.team", fromlist=["brain_kwargs"]).brain_kwargs
    teams: dict = {}
    brains = {x.id: REGISTRY.make("chase", **bk(x, w, teams)) for x in sc.ducks}
    brain = brains["d0"]
    d = w.ducks["d0"]
    # Verification rule 0: the exits are read off the CONSTRUCTED brain, never
    # off a fresh ChaseParams() and never off the file.
    exits_live = (round(float(brain.p.kick_exit_left), 4), round(float(brain.p.kick_exit_right), 4))
    gx, gy = brain.goal if brain.goal is not None else (0.0, 0.0)
    bx_h, by_h = _board_rect(w)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for ep in range(episodes):
        q, v = (_place_at_boards(w, rng, at_boards) if at_boards > 0.0 else _place(w, rng, spread))
        place_xy = _ball_xy(w, q)
        place_board = round(min(bx_h - abs(place_xy[0]), by_h - abs(place_xy[1])), 4)
        for b in brains.values():
            b.reset()
        _CF["log"] = log = []
        t0 = w.t
        prev_skill = None
        prev_declines = brain.declines
        prev_pushes = brain.pushes
        n_swing = n_decline = n_push = 0
        t_first_swing = None
        pending: list[dict] = []            # events waiting for their FATE_S sample
        ev_rows: list[dict] = []
        while w.t - t0 < EPISODE_S:
            _drive(w, brains)
            w.step()
            ball = _ball_xy(w, q)
            # …the fate window of anything already fired
            keep = []
            for pe in pending:
                if w.t - pe["_t"] >= FATE_S:
                    b0 = pe["ball"]
                    pe["fate_dx"] = round(ball[0] - b0[0], 4)          # + = toward the attacked (+x) mouth
                    pe["fate_travel"] = round(math.dist(b0, ball), 4)
                    pe["fate_ball"] = [round(ball[0], 4), round(ball[1], 4)]
                    pe["fate_swings"] = n_swing - pe["_sw0"]
                    pe.pop("_t"), pe.pop("_sw0")
                    ev_rows.append(pe)
                else:
                    keep.append(pe)
            pending = keep
            swung = d.skill is not None and prev_skill is None and str(d.skill).startswith("kick")
            declined = brain.declines > prev_declines
            pushed = brain.pushes > prev_pushes
            prev_declines, prev_pushes, prev_skill = brain.declines, brain.pushes, d.skill
            if not (swung or declined or pushed):
                continue
            if swung:
                n_swing += 1
                t_first_swing = (w.t - t0) if t_first_swing is None else t_first_swing
            n_decline += bool(declined)
            n_push += bool(pushed)
            last = log[-1] if log else {}
            yaw = float(d.yaw(w.data))
            pos = d.trunk_pos(w.data)
            pending.append({
                "kind": "swing" if swung else ("push" if pushed else "decline"),
                "ep": ep, "seed": seed, "t": round(w.t - t0, 2),
                "foot": str(d.skill) if swung else None,
                "ball": [round(ball[0], 4), round(ball[1], 4)],
                # …and where that is relative to the MOUTH this duck attacks
                "mouth_range": round(math.hypot(gx - ball[0], gy - ball[1]), 4),
                "mouth_bearing": round(_wrap(math.atan2(gy - ball[1], gx - ball[0]) - yaw), 4),
                "ball_board": round(min(bx_h - abs(ball[0]), by_h - abs(ball[1])), 4),
                "duck": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(yaw, 4)],
                "pred_side": None if brain.predicted is None else round(
                    -(brain.predicted[0] - w.odom(d)[0]) * math.sin(w.odom(d)[2])
                    + (brain.predicted[1] - w.odom(d)[1]) * math.cos(w.odom(d)[2]), 4),
                "pred_ahead": None if brain.predicted is None else round(
                    (brain.predicted[0] - w.odom(d)[0]) * math.cos(w.odom(d)[2])
                    + (brain.predicted[1] - w.odom(d)[1]) * math.sin(w.odom(d)[2]), 4),
                "sel": {k: last.get(k) for k in
                        ("none", "foot", "u", "detour", "aim_max", "p_own", "p_goal", "value",
                         "los", "goal_dir", "goal_cone",
                         "cf_none", "cf_foot", "cf_u", "cf_detour", "cf_p_own", "cf_p_goal", "cf_value")},
                "_t": w.t, "_sw0": n_swing,
            })
        for pe in pending:                  # the episode ran out before the window: no fate, still an event
            pe.pop("_t"), pe.pop("_sw0")
            ev_rows.append(pe)
        rows += ev_rows
        # THE SELECT CENSUS for the episode: the reachable set of the own-goal
        # filter and of the aim window, counted over every plan the brain made.
        pinned = sum(1 for r in log if r["detour"] is not None
                     and abs(abs(r["detour"]) - r["aim_max"]) < 1e-6)
        cf_diff_foot = sum(1 for r in log if r["cf_foot"] is not None and r["cf_foot"] != r["foot"])
        cf_diff_line = sum(1 for r in log if r["cf_u"] is not None and r["u"] is not None
                           and abs(_wrap(r["cf_u"] - r["u"])) > 1e-6)
        rows.append({
            "kind": "episode", "ep": ep, "seed": seed,
            "swings": n_swing, "declines": n_decline, "pushes": n_push,
            "t_first_swing": None if t_first_swing is None else round(t_first_swing, 2),
            "place": [round(c, 4) for c in place_xy], "place_board": place_board,
            "selects": len(log),
            "select_none": sum(1 for r in log if r["none"]),
            "select_pinned": pinned,
            "select_p_own_pos": sum(1 for r in log if r["p_own"] is not None and r["p_own"] > 0),
            "cf_none": sum(1 for r in log if r["cf_none"]),
            "cf_diff_foot": cf_diff_foot,
            "cf_diff_line": cf_diff_line,
            "exits_live": exits_live, "cf_exits": None if cf_exits is None else list(cf_exits),
        })
        _CF["log"] = None
    return rows


def _run(a):
    return run(*a)


def _load(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            out += [json.loads(x) for x in fh if x.strip()]
    return out


def _pct(k, n):
    return "—" if not n else f"{k}/{n} = {100.0 * k / n:.1f}%"


def report(rows: list[dict], label: str = "") -> None:
    eps = [r for r in rows if r["kind"] == "episode"]
    sw = [r for r in rows if r["kind"] == "swing"]
    dc = [r for r in rows if r["kind"] == "decline"]
    n_ep = len(eps)
    print(f"\n=== {label or 'arm'}: {n_ep} episodes, {len(sw)} swings, {len(dc)} declines")
    if not n_ep:
        return
    print(f"  episodes with no swing : {_pct(sum(1 for e in eps if not e['swings']), n_ep)}")
    ts = [e["t_first_swing"] for e in eps if e["t_first_swing"] is not None]
    if ts:
        ts.sort()
        print(f"  time to first swing    : med {ts[len(ts)//2]:.2f}s  "
              f"IQR {ts[len(ts)//4]:.2f}..{ts[3*len(ts)//4]:.2f}")
    sel = sum(e["selects"] for e in eps)
    print("  --- the selector's reachable set (every plan, not every swing) ---")
    print(f"  select calls           : {sel}")
    print(f"  returned None (all lines own-goal): {_pct(sum(e['select_none'] for e in eps), sel)}")
    print(f"  chosen line PINNED at aim_max     : {_pct(sum(e['select_pinned'] for e in eps), sel)}")
    print(f"  any own-goal share at all         : {_pct(sum(e['select_p_own_pos'] for e in eps), sel)}")
    if any(e["cf_exits"] for e in eps):
        print("  --- counterfactual: the OTHER exit model on the same fan/stream ---")
        print(f"  would have returned None : {_pct(sum(e['cf_none'] for e in eps), sel)}")
        print(f"  picks a different FOOT   : {_pct(sum(e['cf_diff_foot'] for e in eps), sel)}")
        print(f"  picks a different LINE   : {_pct(sum(e['cf_diff_line'] for e in eps), sel)}")
    print("  --- what the events did over the next 4 s ---")
    for kind, evs in (("swing", sw), ("decline", dc)):
        f = [e["fate_dx"] for e in evs if e.get("fate_dx") is not None]
        if not f:
            print(f"  {kind:8s}: no fate window closed")
            continue
        f.sort()
        print(f"  {kind:8s}: n {len(f)}  mean dx {sum(f)/len(f):+.3f} m  med {f[len(f)//2]:+.3f}  "
              f"back {_pct(sum(1 for x in f if x < 0), len(f))}")


def compare(arms: dict[str, list[dict]]) -> None:
    for lab, rows in arms.items():
        report(rows, lab)


def selfcheck() -> None:
    """The probe must not move the thing it measures. Two seeds, 6 episodes,
    with the counterfactual OFF and ON: every event row identical."""
    a = run(0, 6, 0.8, 0.0, None)
    b = run(0, 6, 0.8, 0.0, (0.209, -0.036))
    strip = lambda rs: [{k: v for k, v in r.items() if not k.startswith("cf_") and k != "sel"}  # noqa: E731
                        for r in rs]
    ok = strip(a) == strip(b)
    print(f"selfcheck: {len(a)} rows vs {len(b)} rows -> "
          f"{'IDENTICAL (the counterfactual consumes no randomness)' if ok else 'DIVERGED'}")
    if not ok:
        for x, y in zip(strip(a), strip(b)):
            if x != y:
                print("  first divergence:\n   ", x, "\n   ", y)
                break
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=40, help="episodes PER seed")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--spread", type=float, default=0.8)
    ap.add_argument("--at-boards", type=float, default=0.0, metavar="M")
    ap.add_argument("--opponents", type=int, default=0,
                    help="contest the ball: the one thing the clean gym lacks and the pitch has")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--knobs", default="", help="a MICRODUCK_CHASE string for the arm")
    ap.add_argument("--cf-exits", default=None, metavar="LEFT,RIGHT",
                    help="the OTHER arm's (left,right) exit in rad; every plan is re-scored under it")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--report", nargs="+", default=None, metavar="FILE",
                    help="read row files back and report instead of running")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        raise SystemExit(0)
    if a.report:
        compare({Path(p).stem: _load([p]) for p in a.report})
        raise SystemExit(0)
    cf = None
    if a.cf_exits:
        l_, r_ = a.cf_exits.split(",")
        cf = (float(l_), float(r_))
    # THE PREFLIGHT (roadmap 12at/12au discipline): the pinned pair, the exits
    # the World will hand the brain, and the exits read back off a CONSTRUCTED
    # brain — printed before a minute of compute is spent.
    print(f"[{a.label}] MICRODUCK_SKILL_KICK_LEFT  = {os.environ.get('MICRODUCK_SKILL_KICK_LEFT', '(unset)')}")
    print(f"[{a.label}] MICRODUCK_SKILL_KICK_RIGHT = {os.environ.get('MICRODUCK_SKILL_KICK_RIGHT', '(unset)')}")
    print(f"[{a.label}] World.skill_path(kick_left)  = {World.skill_path('kick_left')}")
    print(f"[{a.label}] World.kick_exits()           = {World.kick_exits()}")
    print(f"[{a.label}] counterfactual exits         = {cf}")
    args = [(s, a.episodes, a.spread, a.at_boards, cf, a.knobs, a.opponents)
            for s in range(a.seed0, a.seed0 + a.seeds)]
    rows: list[dict] = []
    if a.jobs > 1 and len(args) > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(_run, args):
                rows += r
    else:
        for x in args:
            rows += run(*x)
    live = {tuple(e["exits_live"]) for e in rows if e["kind"] == "episode"}
    print(f"[{a.label}] exits READ BACK off the constructed brains: {live}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "label": a.label}) + "\n")
    report(rows, a.label)
