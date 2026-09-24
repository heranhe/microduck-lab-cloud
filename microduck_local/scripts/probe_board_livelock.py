"""WHY does a duck stand in a corner? — the re-plan loop and the throw-in it blocks.

`probe_corner_dwell.py` says HOW LONG. This says WHY, by counting the three
things the mechanism predicts, on the lab's own pitch:

1. **UNREACHABLE SPOTS.** `_plan` lays the kick spot `kick_ahead` behind the
   ball and `kick_side` across it, and NOTHING checks the duck's body fits
   there — `board_margin` is the only such test and it ships at 0.0 (roadmap
   12v). A spot closer to a board than the walking body's own extent (0.129 m,
   measured over 499 walking ticks) is a spot the trunk cannot occupy, so the
   line-up can never reach `lineup_tol` and never swings.

2. **THE RE-PLAN LOOP.** `lineup` times out after `lineup_s` = 4 s into
   `search` (controllers.py:2937) — and the very next tick, with the ball still
   seen and still close, re-plans THE SAME SPOT and re-enters `lineup`. Nothing
   remembers that the attempt just failed. This counts those cycles: a
   `lineup` timeout whose replacement plan lands within `SAME_M` of the one
   that just failed.

3. **THE THROW-IN THE DUCK BLOCKS.** `World._check_ball_out` gives the ball
   back to play after `ball_out_s` at rest by the boards, but the rest timer is
   reset by ANY single tick over 0.05 m/s (`arena.py:_check_ball_out`). A duck
   standing on the ball nudges it, so the referee never fires. This measures the
   siege directly: how long the ball is continuously by a board, how much of
   that time the rest timer was actually accumulating, and how many throw-ins
   came out the other end -- with a duck within `CONTACT_M` or not, which is the
   comparison that says whether the duck is the cause.

    uv run python scripts/probe_board_livelock.py --seeds 8 --seconds 180

Reachability is computed from the SPOT, never from the ball (12v's finding:
"the ball's distance predicts the swing only THROUGH the spot").
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains, throw_in_brains
from microduck_local.world import World, make_pitch

BODY = 0.129        # m, the walking duck's max horizontal trunk extent (roadmap 12v)
NEAR_BOARD = 0.30   # the ball counts as "at the boards" inside this
CORNER = 0.30
SAME_M = 0.05       # a re-plan within this of the failed spot is the SAME attempt
CONTACT_M = 0.25    # a duck this close to the ball is standing on it
BALL_OUT_S, GETUP_S = 5.0, 5.0
REST_SPEED = 0.05   # arena.py's own threshold


def run(seed: int, seconds: float, per_side: int) -> dict:
    sc = make_pitch(per_side=per_side, formation=True)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    getup = onnx_infer(POLICIES_DIR / "alpha_stand.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed,
              ball_out_s=BALL_OUT_S, getup_s=GETUP_S, getup_infer=getup)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    hx, hy = w.scenario.floor[0] / 2 - 0.25, w.scenario.floor[1] / 2 - 0.25
    t: dict = defaultdict(float)
    prev_state = {did: None for did in brains}
    prev_spot: dict = {did: None for did in brains}
    last_failed = {did: None for did in brains}
    goal_seq, out_seq = 0, w.ball_out_seq
    siege_t0: float | None = None
    siege_rest = 0.0
    siege_contact = 0.0
    sieges: list[tuple[float, float, float, bool]] = []      # (length, rest_share, contact_share, ended_in_throw_in)
    j = w._ball_joint
    qadr, vadr = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
    while w.t < seconds:
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

            bxy = w.ball_xy()
            ball_gap = min(hx - abs(bxy[0]), hy - abs(bxy[1])) if bxy else 9.0
            ball_corner = bxy is not None and (hx - abs(bxy[0])) < CORNER and (hy - abs(bxy[1])) < CORNER
            spot = b.spot
            if spot is not None and bxy is not None:
                gap = min(hx - abs(float(spot[0])), hy - abs(float(spot[1])))
                where = "corner" if ball_corner else ("board" if ball_gap < NEAR_BOARD else "open")
                t[f"plan/{where}"] += 1
                if gap < BODY:
                    t[f"unreach/{where}"] += 1
                if gap < 0.0:
                    t[f"inwall/{where}"] += 1
            # The re-plan loop: a lineup that timed out, then planned the same
            # spot again. The failed spot is the PREVIOUS tick's -- the timeout
            # branch sets `self.spot = None` before returning, so reading
            # `b.spot` after the step reads the clearing and never the spot that
            # failed (this probe did exactly that and reported 0% for it).
            if prev_state[did] == "lineup" and b.state == "search":
                last_failed[did] = prev_spot[did]
                t["lineup_timeouts"] += 1
                t[f"timeout_at/{'corner' if ball_corner else 'board' if ball_gap < NEAR_BOARD else 'open'}"] += 1
            elif b.state == "lineup" and prev_state[did] != "lineup" and last_failed[did] is not None \
                    and spot is not None:
                if math.hypot(spot[0] - last_failed[did][0], spot[1] - last_failed[did][1]) < SAME_M:
                    t["replan_same"] += 1
                last_failed[did] = None
            prev_state[did] = b.state
            prev_spot[did] = None if spot is None else (float(spot[0]), float(spot[1]))

        # the siege: the ball continuously by a board, and who is standing on it
        bxy = w.ball_xy()
        speed = float(np.hypot(w.data.qvel[vadr], w.data.qvel[vadr + 1]))
        by_board = bxy is not None and min(hx - abs(bxy[0]), hy - abs(bxy[1])) < w.ball_out_m
        contact = bxy is not None and any(
            math.hypot(float(w.ducks[did].trunk_pos(w.data)[0]) - bxy[0],
                       float(w.ducks[did].trunk_pos(w.data)[1]) - bxy[1]) < CONTACT_M for did in brains)
        if by_board:
            if siege_t0 is None:
                siege_t0, siege_rest, siege_contact = w.t, 0.0, 0.0
            if speed < REST_SPEED:
                siege_rest += 1.0
            if contact:
                siege_contact += 1.0
            t["siege_ticks"] += 1
            if contact:
                t["siege_contact_ticks"] += 1
                if speed >= REST_SPEED:
                    t["siege_contact_moving"] += 1
            else:
                t["siege_alone_ticks"] += 1
                if speed >= REST_SPEED:
                    t["siege_alone_moving"] += 1
        w.step()
        if by_board and (bxy is None or min(hx - abs(w.ball_xy()[0]), hy - abs(w.ball_xy()[1])) >= w.ball_out_m
                         or w.ball_out_seq != out_seq):
            n = max(1.0, (w.t - siege_t0) * 50.0)
            sieges.append(((w.t - siege_t0), siege_rest / n, siege_contact / n, w.ball_out_seq != out_seq))
            siege_t0 = None
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
        if w.ball_out_seq != out_seq:
            out_seq = w.ball_out_seq
            t["throw_ins"] += 1
            throw_in_brains(brains, teams)
    t["duck_seconds"] = len(brains) * w.t
    t["_sieges"] = 0
    return {"t": t, "sieges": sieges}


def report(t: dict, sieges: list, seeds: int, seconds: float) -> None:
    ds = t.get("duck_seconds", 0.0)
    print(f"\n{ds:.0f} duck-seconds ({seeds} seeds x {seconds:.0f} s)")
    print("\n1. IS THE PLANNED SPOT SOMEWHERE THE BODY FITS?  (body extent 0.129 m)")
    print(f"   {'ball is':<10}{'plans':>9}{'spot < body of a board':>24}{'spot INSIDE a board':>22}")
    for where in ("open", "board", "corner"):
        n = t.get(f"plan/{where}", 0)
        if not n:
            continue
        print(f"   {where:<10}{n:>9.0f}{100 * t.get(f'unreach/{where}', 0) / n:>23.1f}%"
              f"{100 * t.get(f'inwall/{where}', 0) / n:>21.1f}%")
    print("\n2. THE RE-PLAN LOOP")
    to = t.get("lineup_timeouts", 0)
    print(f"   line-up timeouts: {to:.0f}  ({t.get('timeout_at/corner', 0):.0f} with the ball in a corner, "
          f"{t.get('timeout_at/board', 0):.0f} at a flat board, {t.get('timeout_at/open', 0):.0f} in open play)")
    print(f"   of which re-planned WITHIN {SAME_M} m of the spot that just failed: "
          f"{t.get('replan_same', 0):.0f}  ({100 * t.get('replan_same', 0) / to if to else 0:.0f}%)")
    print("\n3. THE SIEGE — the ball by the boards, and the throw-in that should end it")
    st = t.get("siege_ticks", 0)
    if st:
        ca, al = t.get("siege_contact_ticks", 0), t.get("siege_alone_ticks", 0)
        print(f"   ball within {0.20:.2f} m of a board for {st / 50:.0f} s of {seconds * seeds:.0f} s "
              f"({100 * st / (seconds * seeds * 50):.0f}% of the run)")
        print(f"   with a duck within {CONTACT_M} m: {100 * ca / st:>4.0f}% of that time; "
              f"the ball is MOVING (> {REST_SPEED} m/s, which RESETS the referee's timer) on")
        print(f"       {100 * t.get('siege_contact_moving', 0) / ca if ca else 0:>5.1f}% of the ticks a duck is on it")
        print(f"       {100 * t.get('siege_alone_moving', 0) / al if al else 0:>5.1f}% of the ticks it is alone"
              "     <-- the control: the same ball, no duck")
    d = np.array([s[0] for s in sieges], dtype=float)
    if len(d):
        ends = sum(1 for s in sieges if s[3])
        print(f"\n   {len(d)} sieges: median {np.median(d):.1f} s, p90 {np.percentile(d, 90):.1f} s, "
              f"MAX {d.max():.1f} s;  {ends} ended in a throw-in ({100 * ends / len(d):.0f}%)")
        long = [s for s in sieges if s[0] > 10.0]
        if long:
            print(f"   the {len(long)} sieges over 10 s: mean contact {100 * np.mean([s[2] for s in long]):.0f}% "
                  f"of ticks, ball at rest {100 * np.mean([s[1] for s in long]):.0f}%, "
                  f"{sum(1 for s in long if s[3])} ended in a throw-in")
    print(f"\n   throw-ins fired: {t.get('throw_ins', 0):.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--per-side", type=int, default=2)
    a = ap.parse_args()
    total: dict = defaultdict(float)
    sieges: list = []
    for s in range(a.seed0, a.seed0 + a.seeds):
        got = run(s, a.seconds, a.per_side)
        for k, v in got["t"].items():
            total[k] += v
        sieges += got["sieges"]
        print(f"  seed {s} done", flush=True)
    report(total, sieges, a.seeds, a.seconds)


if __name__ == "__main__":
    main()
