"""How long does a duck actually spend in a corner, and does it get OUT? — the
dwell census.

From /sim, again: *"the duck is getting stuck in the corner"*. Item 12m answered
a NEIGHBOURING question — it counted the *stuck-at-a-board* ticks (within 0.35 m,
pointing at it within 35 deg, no ball sighting fresher than 0.3 s) and found 9 s a
duck a 180 s run, with the ball within half a metre 79.7% of the time. It then
attributed the whole thing to the near-field blind radius **of a camera the robot
does not have** (62.3 x 48.8, roadmap 12z). The corner claim inside it - "corners
are not traps, 3.79 s a visit against 5.17 s at a flat wall" - is the number this
probe re-measures, because it was taken on that same wrong lens and a lens twice
as wide changes exactly the quantity the mechanism rests on.

**A total is not a trap.** 9 s spread over twenty 0.5 s brushes is a duck that
handles corners; 9 s in one visit is a duck that is stuck. Both give the same
total, and only the second is what the user is watching. So this probe segments
the occupancy into VISITS and reports the distribution - count, median, p90 and
the worst single visit - alongside the total. `--csv` writes one row a visit so a
distribution can be re-read without a re-run.

    uv run python scripts/probe_corner_dwell.py --seeds 4 --seconds 180
    MICRODUCK_CAMERA="fov_h_deg=62.3,fov_v_deg=48.8,px_h=320,projection=pinhole" \
        uv run python scripts/probe_corner_dwell.py --seeds 4 --seconds 180

The second line is the A/B that matters: the same census under the camera 12m was
measured on. If corner dwell is unchanged between them, the blind radius is not
what holds a duck in a corner and 12m's mechanism does not carry over to the
corner claim.

**Truth, not the brain's estimate**, for where the duck is: this asks whether the
duck IS in the corner, which is a fact about the world. The ball distance is truth
for the same reason; what the brain believes is reported separately as `det_age`,
because the gap between them is the hypothesis under test.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains, throw_in_brains
from microduck_local.sensors import DetectorSpec
from microduck_local.world import World, make_pitch

CORNER = 0.30       # within this of BOTH boards: a corner (probe_board_census's figure)
WALL = 0.30         # within this of ONE board and not a corner: a flat board
GAP_S = 0.5         # leave for longer than this and the next entry is a NEW visit
STUCK_M, STUCK_DEG, STUCK_AGE = 0.35, 35.0, 0.3      # 12m's stuck-at-a-board definition
BALL_OUT_S, GETUP_S = 5.0, 5.0   # world_server.PITCH_BALL_OUT_S / PITCH_GETUP_S: the LAB's pitch


def facing_board(x: float, y: float, yaw: float, hx: float, hy: float) -> float:
    """Angle between the heading and the direction of the NEAREST board, rad."""
    dx, dy = hx - abs(x), hy - abs(y)
    if dx < dy:
        out = math.atan2(0.0, math.copysign(1.0, x))     # +-x board
    else:
        out = math.atan2(math.copysign(1.0, y), 0.0)     # +-y board
    return abs(math.atan2(math.sin(yaw - out), math.cos(yaw - out)))


class Visits:
    """Occupancy -> segments, with `GAP_S` of hysteresis so a one-tick wobble
    out of the zone does not split one visit into two."""

    def __init__(self) -> None:
        self.open_t0: float | None = None
        self.last_in: float | None = None
        self.done: list[float] = []

    def tick(self, t: float, inside: bool) -> None:
        if inside:
            if self.open_t0 is None:
                self.open_t0 = t
            self.last_in = t
        elif self.open_t0 is not None and self.last_in is not None and t - self.last_in > GAP_S:
            self.done.append(self.last_in - self.open_t0)
            self.open_t0 = self.last_in = None

    def close(self) -> None:
        if self.open_t0 is not None and self.last_in is not None:
            self.done.append(self.last_in - self.open_t0)
        self.open_t0 = self.last_in = None


def run(seed: int, seconds: float, per_side: int, rows: list) -> dict:
    """One run under THE LAB'S physics, not the benchmark's: the /sim page is
    what the observation came from, so `ball_out_s`, the get-up and the
    throw-in belief reset are all the world_server's (PITCH_BALL_OUT_S,
    PITCH_GETUP_S, throw_in_brains). eval-pitch's defaults differ and would
    measure a different pitch."""
    sc = make_pitch(per_side=per_side, formation=True)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    getup = onnx_infer(POLICIES_DIR / "alpha_stand.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed,
              ball_out_s=BALL_OUT_S, getup_s=GETUP_S, getup_infer=getup)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    hx, hy = w.scenario.floor[0] / 2 - 0.25, w.scenario.floor[1] / 2 - 0.25
    tally: dict = defaultdict(float)
    visits = {did: {"corner": Visits(), "flat": Visits()} for did in brains}
    goal_seq, out_seq = 0, w.ball_out_seq
    dt = w.scenario.dt if hasattr(w.scenario, "dt") else None
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

            pos = d.trunk_pos(w.data)
            x, y, yaw = float(pos[0]), float(pos[1]), d.yaw(w.data)
            near_x, near_y = hx - abs(x), hy - abs(y)
            corner = near_x < CORNER and near_y < CORNER
            flat = (not corner) and min(near_x, near_y) < WALL
            visits[did]["corner"].tick(w.t, corner)
            visits[did]["flat"].tick(w.t, flat)

            # THE BALL's sighting age, from the brain's own tracker. Using the
            # freshness of the last detection FRAME instead reads 0% blind
            # forever: a frame with no ball in it is still a fresh frame, and
            # the detector produces one every tick. That bug returned exactly
            # 0.0 s for 12m's stuck definition -- a zero from a question not
            # asked, which is the failure 12o is a monument to.
            trk = b.tracker.best(b.p.target_cls, t=w.t, min_hits=1)
            ball_age = None if trk is None else trk.age(w.t)
            blind = ball_age is None or ball_age > STUCK_AGE
            tally["ticks"] += 1
            # THE MECHANISM ITSELF, which fires thousands of times where a
            # 100 s corner visit fires five: a supporter with NO ball belief
            # (`_support`'s early return -- `post` is None exactly then)
            # standing within 0.30 m of a board. That is the state the trap is
            # made of, and unlike the visit lengths it is not a rare event, so
            # it is the arm with the power. `stranded_corner` is the same thing
            # with two boards instead of one.
            # NOT `b.post is None`: `support_home` sets a post in exactly this
            # branch, so that test reads 0% under the fix and measures the
            # fix's own bookkeeping rather than the duck (12o's failure 3,
            # "measured through the fix"). Recompute the belief the way
            # `_support` does instead -- own track within `lost_s`, else the
            # team board's led ball -- which is true of both arms.
            believes = (trk is not None and ball_age is not None and ball_age < b.p.lost_s) or (
                b.team is not None and b.team.led_ball(w.t) is not None)
            if b.state in ("support", "wait") and not believes:
                tally["noball_support"] += 1
                if corner or flat:
                    tally["stranded"] += 1
                if corner:
                    tally["stranded_corner"] += 1
            if corner:
                tally["corner_ticks"] += 1
                tally[f"state/{b.state}"] += 1
                bxy = w.ball_xy()
                if bxy is not None and math.hypot(bxy[0] - x, bxy[1] - y) < 0.5:
                    tally["corner_ball_near"] += 1
                if blind:
                    tally["corner_blind"] += 1
                # Is a POST what put it here? `post` is where a supporter is
                # holding; if the post is in the corner too, the duck was SENT.
                if b.post is not None:
                    px, py = float(b.post[0]), float(b.post[1])
                    tally["corner_posted"] += 1
                    if (hx - abs(px)) < CORNER and (hy - abs(py)) < CORNER:
                        tally["corner_post_in_corner"] += 1
                    if min(hx - abs(px), hy - abs(py)) < 0.0:
                        tally["corner_post_outside"] += 1
            if flat:
                tally["flat_ticks"] += 1
            # 12m's own definition, re-measured verbatim so the two are comparable
            if min(near_x, near_y) < STUCK_M and facing_board(x, y, yaw, hx, hy) < math.radians(STUCK_DEG) and blind:
                tally["stuck12m_ticks"] += 1
        w.step()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
        if w.ball_out_seq != out_seq:          # the referee moved it: drop stale ball beliefs (12s)
            out_seq = w.ball_out_seq
            throw_in_brains(brains, teams)
    for did, v in visits.items():
        for where, vv in v.items():
            vv.close()
            for dur in vv.done:
                rows.append({"seed": seed, "duck": did, "where": where, "seconds": round(dur, 3)})
    tally["duck_seconds"] = len(brains) * w.t
    tally["_hx"], tally["_hy"] = hx, hy
    tally["_dt"] = w.t / max(1.0, tally["ticks"] / len(brains))
    return tally


def report(t: dict, rows: list, seconds: float, seeds: int) -> None:
    ds = t.get("duck_seconds", 0.0)
    dt = t.get("_dt", 0.0) / max(1, seeds)
    if not ds:
        print("!! no duck-seconds: broken run")
        return
    print(f"\n{ds:.0f} duck-seconds ({seeds} seeds x {seconds:.0f} s), tick {dt * 1000:.0f} ms")
    # THE CONTROL THAT DECIDES HOW TO READ THESE. A 0.30 m band round a
    # 3.4 x 2.85 pitch is a THIRD of its area, so a duck wandering at random
    # is "at a board" a third of the time. Without this line 17% reads as a
    # pathology when it is under chance. (Corner = four 0.30 m squares.)
    hx, hy = t.get("_hx", 0.0) / max(1, seeds), t.get("_hy", 0.0) / max(1, seeds)
    area = (2 * hx) * (2 * hy)
    corner_share = 4 * CORNER * CORNER / area if area else 0.0
    band_share = (area - (2 * hx - 2 * WALL) * (2 * hy - 2 * WALL)) / area if area else 0.0
    print(f"\n  TIME, as seconds per duck per 180 s run (12m's unit)"
          f"   [chance = uniform wandering on a {2 * hx:.2f} x {2 * hy:.2f} m pitch]")
    for k, label, chance in (
            ("corner_ticks", "in a CORNER (< 0.30 m of both boards)", corner_share),
            ("flat_ticks", "at a FLAT board (< 0.30 m, not a corner)", band_share - corner_share),
            ("stuck12m_ticks", "12m 'stuck': < 0.35 m, facing it, ball unseen > 0.3 s", None)):
        secs = t.get(k, 0.0) * dt
        pct = 100 * secs / ds
        tail = "" if chance is None else f"   chance {100 * chance:>4.1f}%  ->  {pct / (100 * chance):.2f}x"
        print(f"    {label:<54} {secs / ds * 180:>6.1f} s ({pct:>4.1f}%){tail}")
    nb, strand = t.get("noball_support", 0.0), t.get("stranded", 0.0)
    print(f"\n  THE MECHANISM — a supporter that does not know where the ball is")
    print(f"    no-ball support ticks                       {nb * dt / ds * 180:>6.1f} s a duck a 180 s run")
    print(f"    ...of them, spent within 0.30 m of a board  {strand * dt / ds * 180:>6.1f} s "
          f"({100 * strand / nb if nb else 0:.0f}% of them)   <-- the trap")
    print(f"    ...in a CORNER (two boards)                 "
          f"{t.get('stranded_corner', 0.0) * dt / ds * 180:>6.1f} s")
    cn = t.get("corner_ticks", 0.0)
    if cn:
        print(f"\n  WHILE IN A CORNER")
        print(f"    ball truly within 0.5 m           {100 * t.get('corner_ball_near', 0) / cn:>5.1f}%")
        print(f"    ball NOT seen for over 0.3 s      {100 * t.get('corner_blind', 0) / cn:>5.1f}%")
        pt = t.get("corner_posted", 0)
        print(f"    holding a post at all             {100 * pt / cn:>5.1f}%"
              f"   of which the post is itself in the corner: "
              f"{(100 * t.get('corner_post_in_corner', 0) / pt) if pt else 0:.0f}%"
              f", outside the boards: {(100 * t.get('corner_post_outside', 0) / pt) if pt else 0:.0f}%")
        st = sorted(((k[6:], v) for k, v in t.items() if k.startswith("state/")),
                    key=lambda kv: -kv[1])
        print("    state: " + ", ".join(f"{k} {100 * v / cn:.0f}%" for k, v in st[:6]))
    for where in ("corner", "flat"):
        d = np.array([r["seconds"] for r in rows if r["where"] == where], dtype=float)
        print(f"\n  {where.upper()} VISITS — is it a TRAP or a brush?")
        if not len(d):
            print("    none")
            continue
        print(f"    {len(d)} visits, {len(d) / ds * 180:.1f} a duck a 180 s run")
        print(f"    median {np.median(d):.2f} s   mean {d.mean():.2f} s   "
              f"p90 {np.percentile(d, 90):.2f} s   MAX {d.max():.2f} s")
        for thr in (3.0, 5.0, 10.0):
            n = int((d > thr).sum())
            print(f"    visits longer than {thr:>4.0f} s: {n:>4}  "
                  f"({100 * n / len(d):.1f}% of visits, {100 * d[d > thr].sum() / d.sum():.0f}% of the time)")
    print("\nA TRAP is the tail, not the total. Read MAX and the >5 s share first:"
          "\nthe same total time is a handled corner if the visits are short and a"
          "\nstuck duck if one visit holds it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0, help="first seed, so runs can be sharded across cores")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--csv", help="write one row a visit here")
    a = ap.parse_args()
    cam = os.environ.get("MICRODUCK_CAMERA", "")
    spec = DetectorSpec.from_env()
    print(f"camera: fov {spec.fov_h_deg:.1f} x {spec.fov_v_deg:.1f}, px_h {spec.px_h}, "
          f"{spec.projection}" + (f"   [MICRODUCK_CAMERA={cam}]" if cam.strip() else "   [shipped]"))
    total: dict = defaultdict(float)
    rows: list = []
    for s in range(a.seed0, a.seed0 + a.seeds):
        for k, v in run(s, a.seconds, a.per_side, rows).items():
            total[k] += v
        print(f"  seed {s} done ({len(rows)} visits)")
    report(total, rows, a.seconds, a.seeds)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=["seed", "duck", "where", "seconds"])
            wcsv.writeheader()
            wcsv.writerows(rows)
        print(f"\nwrote {len(rows)} visit rows to {a.csv}")


if __name__ == "__main__":
    main()
