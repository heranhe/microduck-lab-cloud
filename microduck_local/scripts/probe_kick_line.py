"""Does the chase brain know which way its own kick will send the ball?

    cd microduck_local
    uv run python scripts/probe_kick_line.py --seeds 24 --seconds 300 --per-side 2

The brain lays its kick spot on a line `u` — the direction it INTENDS the ball
to go — and then, after the swing, hunts along that same `u` to find the ball
again. Both assume the ball leaves along the body heading. The kick map says
it does not: measured on a standing duck, a ball on the left foot's sweet spot
leaves at **+21.6°** to the body heading and one on the right at **−11°**
(`ChaseParams.kick_deflect_*`, which ship at 0 — the compensation was measured
off in the first form, on 8 seeds, on goals, before anything could see where a
kick actually went).

This measures it in PLAY rather than on a bench: for every kick, the intended
line, the direction the ball actually travelled over the next `CARRY_S`, and
the angle between them — split by foot, so a systematic bias separates from
the scatter. A bias is worth removing even when it is smaller than the noise,
because it lands on every kick of that foot in the same direction; scatter
does not.

Prints the per-foot mean and median error, the fraction of kicks that go the
wrong side of the intended line, and what the shipped map would predict.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.world import World, make_pitch
from microduck_local.world.metrics import CARRY_S

MAP_DEG = {"kick_left": 21.6, "kick_right": -11.0}   # the bench measurement, for comparison


def _joint(w, d, suffix: str) -> float | None:
    """This duck's joint angle (rad) for the joint whose name ends in `suffix`."""
    import mujoco
    for j in range(w.model.njnt):
        n = mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n and n.startswith(d.adr.prefix) and n.endswith(suffix):
            return float(w.data.qpos[w.model.jnt_qposadr[j]])
    return None


def _head_joint(w, d) -> float | None:
    """This duck's head-pitch joint angle (rad) right now, or None."""
    return _joint(w, d, "head_pitch")


def _trunk_pitch(w, d) -> float:
    """Trunk pitch from projected gravity (rad, nose-down positive)."""
    import math
    g = d.projected_gravity(w.data)
    return float(math.atan2(-g[0], -g[2]))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def run(seed: int, seconds: float, per_side: int, ball_out_s: float = 0.0, dump_state: bool = False) -> list[dict]:
    sc = make_pitch(per_side=per_side)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed, ball_out_s=ball_out_s)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    pending: list[dict] = []
    out: list[dict] = []
    plan: dict[str, tuple[float, tuple[float, float]] | None] = {}
    prev_spot: dict[str, object] = {}
    # The head yaw the duck was HOLDING on the tick before the swing. The
    # kick tick itself is not it: the skill takes the head over. A held yaw
    # is a mass off the centre line, so this is the column to correlate an
    # aim bias against when one appears.
    prev_yaw: dict[str, float] = {}
    goal_seq = 0
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            b = brains[d.id]
            intent = b.step(s)
            # Roadmap 12i: after a swing, when does this duck next SEE the ball?
            for k in pending:
                if k["duck"] == d.id and k.get("reacq") is None and w.t - k["t"] >= 0.5:
                    trk = b.tracker.best(b.p.target_cls, w.t, min_hits=1)
                    if trk is not None and trk.age(w.t) <= 0.2:
                        k["reacq"] = round(w.t - k["t"], 2)
            # The brain clears its spot in the same tick it fires, so the
            # intended line is read from `_hunt_u` — which IS that line, and
            # is also what the duck will walk along afterwards to find the
            # ball again. Both halves of "which way is it going" in one
            # number.
            if intent.skill in MAP_DEG and b._hunt_u is not None:
                # The head-pitch JOINT at the instant of the swing - not the
                # command. The kick skill was trained from a level head
                # (~+0.39 rad) and whiffs 12/12 from +0.97 (benched), so a
                # gaze that leaves the joint down explains a whiff directly.
                hj = _head_joint(w, d)
                nj = _joint(w, d, "neck_pitch")
                tp = _trunk_pitch(w, d)
                # Where the ball REALLY is relative to the kicking body at the
                # instant of the swing. The sweet spot is `kick_ahead` 0.08 m
                # forward and `kick_side` 0.06 m to the foot's side, and the
                # deflection is a steep function of that offset — so this says
                # whether a kick was ever going to go where it was aimed.
                bx, by = w.ball_xy()
                ox, oy, oyaw = w.odom(d)
                dx, dy = bx - ox, by - oy
                ahead = dx * math.cos(oyaw) + dy * math.sin(oyaw)
                side = -dx * math.sin(oyaw) + dy * math.cos(oyaw)
                # Did it REACH its spot, and was the spot in the right place?
                # Two different failures with the same symptom: `spot_dist` is
                # trunk-to-spot (should be inside `lineup_tol` = 3 cm if the
                # walk-in finished) and `spot_ball` is spot-to-BALL (should be
                # `kick_ahead` = 8 cm if the plan was right when it was made).
                sp = prev_spot.get(d.id)
                state = None
                if dump_state:
                    # The duck's and the ball's FULL state at the instant the
                    # swing is asked for (before the world applies it), so a
                    # bench can replay this exact swing (roadmap item 12a).
                    import mujoco
                    a, q, v = d.adr, w.data.qpos, w.data.qvel
                    jb = w._ball_joint
                    bq, bv = int(w.model.jnt_qposadr[jb]), int(w.model.jnt_dofadr[jb])
                    state = {"root_qpos": q[a.root_qpos:a.root_qpos + 7].round(5).tolist(),
                             "root_qvel": v[a.root_qvel:a.root_qvel + 6].round(5).tolist(),
                             "joint_qpos": q[a.joint_qpos].round(5).tolist(),
                             "joint_qvel": v[a.joint_qvel].round(5).tolist(),
                             "last_action": d.last_action.round(5).tolist(),
                             "prev_joint_vel": d.prev_joint_vel.round(5).tolist(),
                             "ctrl": w.data.ctrl[a.actuators].round(5).tolist(),
                             "ball_qpos": q[bq:bq + 7].round(5).tolist(),
                             "ball_qvel": v[bv:bv + 6].round(5).tolist()}
                    # …and the CONTEXT a single-duck bench scene has not got.
                    # A replay deletes the boards and the other ducks, so a
                    # cause that lives in either of them would be invisible in
                    # it and would come back as "the state is innocent".
                    # Recorded here so both can be ruled out on the play rows
                    # instead (roadmap 12a: neither is the cause — the ball is
                    # a median 0.87 m from a board and the only thing touching
                    # it at the swing, in 79 of 79 swings, is the floor, which
                    # is candidate (iii) falsified outright).
                    hx, hy = sc.floor[0] / 2.0, sc.floor[1] / 2.0
                    state["ball_board"] = round(float(min(hx - abs(bx), hy - abs(by))), 3)
                    state["ball_duck"] = round(min(
                        (math.dist((bx, by), (float(q[o.adr.root_qpos]), float(q[o.adr.root_qpos + 1])))
                         for o in w.ducks.values() if o.id != d.id), default=9.9), 3)
                    # The gain the ACTUATORS are at as the swing is asked for.
                    # `start_skill` drops it to STANDING_GAIN_RATIO on this
                    # same tick, so a replay that wants the arena's swing has
                    # to apply 0.8 itself — this column is what says so.
                    state["gain_ratio"] = round(float(d.gain_ratio), 3)
                    bb, ball_body = w.model.geom_bodyid, int(w.model.jnt_bodyid[jb])
                    touch = []
                    for con in range(w.data.ncon):
                        g1, g2 = int(w.data.contact.geom1[con]), int(w.data.contact.geom2[con])
                        oth = g2 if bb[g1] == ball_body else (g1 if bb[g2] == ball_body else None)
                        if oth is not None:
                            touch.append(mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_GEOM, oth)
                                         or f"geom{oth}")
                    state["touching"] = touch
                # What the BRAIN believed at the swing: its ball track's age and
                # its fresh predicted position (the ahead gate reads the latter;
                # None = nothing fresh, the gate cannot fire) - item 12c coverage.
                trk = b.tracker.best(b.p.target_cls, w.t, min_hits=1)
                pred = getattr(b, "predicted", None)
                pending.append({"t": w.t, "duck": d.id, "foot": intent.skill, "state": state,
                                "track_age": None if trk is None else round(trk.age(w.t), 2),
                                "pred_ahead": None if pred is None else round((pred[0] - ox) * math.cos(oyaw) + (pred[1] - oy) * math.sin(oyaw), 3),
                                "u": b._hunt_u, "heading": oyaw, "ball0": (bx, by),
                                # The goal this duck attacks, so a kick can be
                                # scored as toward / away from it and as "would
                                # cross our own mouth" (kickselect's labels).
                                "goal": None if b.goal is None else [round(b.goal[0], 3), round(b.goal[1], 3)],
                                "select": None if getattr(b, "last_select", None) is None else
                                {"p_goal": round(b.last_select.p_goal, 3), "p_own": round(b.last_select.p_own, 3),
                                 "value": round(b.last_select.value, 3),
                                 "p_pass": round(getattr(b.last_select, "p_pass", 0.0), 3)},
                                # Teammates' TRUE positions at the swing, so a kick can be
                                # scored as RECEIVED (the ball stops within reach of one).
                                "mates": [[round(float(w.odom(o)[0]), 3), round(float(w.odom(o)[1]), 3)]
                                          for o in w.ducks.values()
                                          if o.id != d.id and w.team_of.get(o.id) == w.team_of.get(d.id)],
                                "ahead": round(ahead, 3), "side": round(side, 3),
                                "spot_dist": None if sp is None else round(math.dist((ox, oy), sp[:2]), 3),
                                "spot_ball": None if sp is None else round(math.dist((bx, by), sp[:2]), 3),
                                # How far the ball has drifted since the spot
                                # this swing is standing on was planned.
                                "moved": None if plan.get(d.id) is None else
                                round(math.dist((bx, by), plan[d.id][1]), 3),
                                "plan_age": None if plan.get(d.id) is None else
                                round(w.t - plan[d.id][0], 2),
                                "head_yaw": round(prev_yaw.get(d.id, 0.0), 4),
                                "head_jt": None if hj is None else round(hj, 3),
                                "neck_jt": None if nj is None else round(nj, 3),
                                "trunk_pitch": round(tp, 3)})
            # Remember when this duck last laid a spot, and where the ball was
            # then: the plan's age and the ball's drift since are the two ways
            # a line-up goes wrong that aiming cannot fix.
            if b.spot is not None and b.state == "lineup" and prev_spot.get(d.id) != b.spot:
                plan[d.id] = (w.t, w.ball_xy())
            prev_spot[d.id] = b.spot
            if intent.skill is None:
                prev_yaw[d.id] = float(intent.head[2])
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)          # with the world: the kickoff rule the benchmark plays under
            pending = [k for k in pending if False]        # the ball teleported: nothing to settle
        keep = []
        for k in pending:
            if w.t - k["t"] < CARRY_S:
                keep.append(k)
                continue
            bx, by = w.ball_xy()
            dx, dy = bx - k["ball0"][0], by - k["ball0"][1]
            dist = math.hypot(dx, dy)
            rec = {"seed": seed, "t": round(k["t"], 1), "duck": k["duck"], "foot": k["foot"],
                   "ball0": [round(k["ball0"][0], 3), round(k["ball0"][1], 3)], "goal": k["goal"],
                   "select": k["select"], "u": round(k["u"], 4), "mates": k.get("mates"),
                   "ball1": [round(bx, 3), round(by, 3)],
                   "dist": round(dist, 3), "ahead": k["ahead"], "side": k["side"],
                   "moved": k["moved"], "plan_age": k["plan_age"], "head_yaw": k["head_yaw"],
                   "head_jt": k.get("head_jt"), "neck_jt": k.get("neck_jt"), "trunk_pitch": k.get("trunk_pitch"),
                   "spot_dist": k["spot_dist"], "spot_ball": k["spot_ball"], "state": k.get("state"),
                   "track_age": k.get("track_age"), "pred_ahead": k.get("pred_ahead"), "reacq": k.get("reacq")}
            if dist < 0.10:                    # the swing missed: no line to speak of
                out.append({**rec, "err": None, "off_heading": None})
                continue
            went = math.atan2(dy, dx)
            out.append({**rec,
                        "err": round(math.degrees(wrap(went - k["u"])), 2),
                        "off_heading": round(math.degrees(wrap(went - k["heading"])), 2)})
        pending = keep
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first seed — a confirmation runs on seeds the effect was NOT found on")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=None, help="write every kick as a JSON line")
    ap.add_argument("--dump-state", action="store_true",
                    help="record the duck's and the ball's full state at each swing into --out rows (bench_kick_headdown --from-swings)")
    ap.add_argument("--ball-out-s", type=float, default=0.0,
                    help="the ball-out rule (World.ball_out_s); the lab's pitches play at 5 - it more than doubles the kicks a run")
    args = ap.parse_args()
    todo = [(s, args.seconds, args.per_side, args.ball_out_s, args.dump_state)
            for s in range(args.seed0, args.seed0 + args.seeds)]
    rows: list[dict] = []
    if args.jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(min(args.jobs, len(todo))) as pool:
            for r in pool.starmap(run, todo):
                rows += r
    else:
        for a in todo:
            rows += run(*a)
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    hit = [r for r in rows if r.get("err") is not None]
    print(f"{len(rows)} kicks over {args.seeds} seeds (from {args.seed0}) x {args.seconds:g} s of "
          f"{args.per_side}v{args.per_side}, ball-out {args.ball_out_s:g} s; {len(rows) - len(hit)} moved the ball < 10 cm (a whiff)")
    print(f"MICRODUCK_CHASE={os.environ.get('MICRODUCK_CHASE', '')!r}\n")
    print(f"{'foot':<12}{'n':>5}{'mean err':>10}{'median':>9}{'sd':>8}{'wrong side':>12}"
          f"{'the map says':>14}{'mean off heading':>18}")
    for foot in ("kick_left", "kick_right"):
        e = np.array([r["err"] for r in hit if r["foot"] == foot])
        h = np.array([r["off_heading"] for r in hit if r["foot"] == foot])
        if not len(e):
            continue
        print(f"{foot:<12}{len(e):>5}{e.mean():>+10.1f}{np.median(e):>+9.1f}{e.std(ddof=1):>8.1f}"
              f"{(np.sign(e) != np.sign(e.mean())).mean():>11.0%}"
              f"{MAP_DEG[foot]:>+14.1f}{h.mean():>+18.1f}")
    e = np.array([r["err"] for r in hit])
    print(f"\nall kicks: mean {e.mean():+.1f}°, mean |error| {np.abs(e).mean():.1f}°, "
          f"sd {e.std(ddof=1):.1f}°")
    # Why a kick scatters: where the ball actually was, and how stale the plan
    # was. The sweet spot is 6-10 cm ahead and 4-8 cm to the side.
    sweet = [r for r in rows if r.get("ahead") is not None]
    if sweet:
        ah = np.array([r["ahead"] for r in sweet])
        sd = np.array([abs(r["side"]) for r in sweet])
        mv = np.array([r["moved"] for r in sweet if r.get("moved") is not None])
        ag = np.array([r["plan_age"] for r in sweet if r.get("plan_age") is not None])
        onspot = ((ah >= 0.06) & (ah <= 0.10) & (sd >= 0.04) & (sd <= 0.08)).mean()
        whiff = np.array([r.get("err") is None for r in rows])
        print(f"\nwhere the ball WAS when the swing fired ({len(sweet)} kicks; "
              f"the sweet spot is 0.06-0.10 m ahead, 0.04-0.08 m to the side):")
        print(f"  ahead of the trunk   median {np.median(ah):.3f} m   IQR "
              f"{np.percentile(ah, 25):.3f}-{np.percentile(ah, 75):.3f}")
        print(f"  to the side          median {np.median(sd):.3f} m   IQR "
              f"{np.percentile(sd, 25):.3f}-{np.percentile(sd, 75):.3f}")
        print(f"  actually ON the sweet spot: {onspot:.0%} of kicks;  whiffed "
              f"(<10 cm of ball travel): {whiff.mean():.0%}")
        if len(mv):
            print(f"  ball drift since the spot was planned  median {np.median(mv):.3f} m "
                  f"(90th {np.percentile(mv, 90):.3f});  plan age median {np.median(ag):.2f} s")
        hj = np.array([r["head_jt"] for r in sweet if r.get("head_jt") is not None])
        if len(hj):
            wh = np.array([r.get("err") is None for r in sweet if r.get("head_jt") is not None])
            print(f"  head-pitch JOINT at the swing  median {np.median(hj):+.2f} rad "
                  f"(level ~ +0.39; the kick whiffs 12/12 from +0.97);  whiff when joint > 0.6: "
                  f"{wh[hj > 0.6].mean() if (hj > 0.6).any() else float('nan'):.0%} of {(hj > 0.6).sum()}, "
                  f"when <= 0.6: {wh[hj <= 0.6].mean() if (hj <= 0.6).any() else float('nan'):.0%} of {(hj <= 0.6).sum()}")
        sdst = np.array([r["spot_dist"] for r in sweet if r.get("spot_dist") is not None])
        sbal = np.array([r["spot_ball"] for r in sweet if r.get("spot_ball") is not None])
        if len(sdst):
            print(f"  did it REACH the spot?  trunk-to-spot median {np.median(sdst):.3f} m "
                  f"(lineup_tol is 0.030)")
            print(f"  was the SPOT right?     spot-to-ball median {np.median(sbal):.3f} m "
                  f"(kick_ahead is 0.080)")
    print("READ IT AS: 'mean err' is the SYSTEMATIC part — it lands on every kick of that foot "
          "the same way and can be designed out.\n'sd' is the scatter, which cannot. "
          "'mean off heading' is the same error measured against the BODY, which is what the "
          "bench map measured.")


if __name__ == "__main__":
    main()
