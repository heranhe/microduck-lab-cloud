"""Does a drifting odometry lose the goal? (roadmap Track 4.4.3)

    cd microduck_local
    uv run python scripts/probe_odom_goal.py --seconds 300 --seeds 3

The pitch tells every `chase` brain where the goal is in its ODOMETRY frame
(`World.goal_for`), which is the truth at spawn and drifts with the duck
afterwards. The honest question before anyone builds a goal detector is
whether that drift ever matters: a shot misses because of it only if the
odometry error at the moment of the kick, carried down the line to the goal,
is larger than the goal's half-width.

So this measures the error the ducks actually accumulate while playing —
position and heading, per `OdomNoise` preset — and turns the heading half into
the miss it causes at the goal line from where the ducks really kick from.
Read the last column: if the 95th percentile miss is well inside the 0.35 m
half-width, the known-pitch assumption holds and a goal detector buys nothing.

The final column is the other thing drift costs, and it is about the TEAM: two
teammates putting the same ball in different places. The blackboard
(`brain/team.py`) passes a ball position and a pose in "my odometry frame",
and every duck's frame is anchored at its own spawn — which is the SAME frame
only for as long as nobody has drifted. This measures how far apart two
teammates' frames have wandered, which is the error a shared ball fix carries
on top of the detector's own.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.world import World, make_pitch

GOAL_HALF_W = 0.35              # make_pitch's goal_width / 2


def run(seed: int, seconds: float, preset: str, per_side: int) -> dict:
    sc = make_pitch(per_side=per_side)
    for d in sc.ducks:
        d.odom = preset
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    pos_err, yaw_err, miss, disagree = [], [], [], []
    # The same four columns for the brain's LOCALISED pose, when the chase
    # brain runs the goal-post particle filter (`MICRODUCK_CHASE=localize=1`).
    loc_pos, loc_yaw, loc_miss, loc_disagree = [], [], [], []
    goal_seq = 0
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            intent = brains[d.id].step(s)
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        # Sample once a second: consecutive ticks are the same error.
        if w.tick % 50 == 0:
            errs: list[tuple[float, float]] = []
            lerrs: list[tuple[float, float]] = []
            for d in w.ducks.values():
                true = d.trunk_pos(w.data)
                est = w.odom(d)
                ex, ey = float(est[0]) - float(true[0]), float(est[1]) - float(true[1])
                dyaw = float(np.arctan2(np.sin(est[2] - d.yaw(w.data)), np.cos(est[2] - d.yaw(w.data))))
                goal = w.goal_for(d)
                rng = math.hypot(goal[0] - float(true[0]), goal[1] - float(true[1]))
                pos_err.append(math.hypot(ex, ey))
                yaw_err.append(abs(dyaw))
                errs.append((ex, ey))
                # What the heading error costs at the goal line: the whole
                # aiming chain (the ball's placement, the kick line) is laid
                # out in this frame, so a yaw error rotates the shot.
                miss.append(abs(math.sin(dyaw)) * rng)
                loc = getattr(brains[d.id], "loc", None)
                if loc is not None and loc.pose is not None:
                    lx, ly, lyaw = loc.pose
                    lex, ley = lx - float(true[0]), ly - float(true[1])
                    ldyaw = float(np.arctan2(np.sin(lyaw - d.yaw(w.data)), np.cos(lyaw - d.yaw(w.data))))
                    loc_pos.append(math.hypot(lex, ley))
                    loc_yaw.append(abs(ldyaw))
                    loc_miss.append(abs(math.sin(ldyaw)) * rng)
                    lerrs.append((lex, ley))
            for i in range(len(lerrs)):
                for j in range(i + 1, len(lerrs)):
                    loc_disagree.append(math.dist(lerrs[i], lerrs[j]))
            # What the blackboard costs: two teammates put the SAME ball in
            # different places, because each one's fix is its own pose plus a
            # bearing and a range. The message says "the ball is at (x, y)"
            # in a frame the two only agree on while their odometry does.
            for i in range(len(errs)):
                for j in range(i + 1, len(errs)):
                    disagree.append(math.dist(errs[i], errs[j]))
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams)
    return {"pos": np.array(pos_err), "yaw": np.array(yaw_err), "miss": np.array(miss),
            "disagree": np.array(disagree),
            "loc_pos": np.array(loc_pos), "loc_yaw": np.array(loc_yaw), "loc_miss": np.array(loc_miss),
            "loc_disagree": np.array(loc_disagree)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--per-side", type=int, default=1)
    ap.add_argument("--presets", default="ideal,datasheet,hostile")
    args = ap.parse_args()
    print(f"{args.per_side}v{args.per_side}, {args.seeds} seeds x {args.seconds:g} s; "
          f"the goal's half-width is {GOAL_HALF_W} m\n")
    print(f"{'odom':<21}{'pos err med':>12}{'95%':>8}{'yaw err med':>13}{'95%':>8}"
          f"{'miss at goal med':>18}{'95%':>8}{'over half-width':>17}{'mates disagree':>16}")
    import os
    print(f"MICRODUCK_CHASE={os.environ.get('MICRODUCK_CHASE', '')!r}")
    for preset in args.presets.split(","):
        got = [run(s, args.seconds, preset, args.per_side) for s in range(args.seeds)]
        for tag, keys in (("", ("pos", "yaw", "miss", "disagree")),
                          (" localised", ("loc_pos", "loc_yaw", "loc_miss", "loc_disagree"))):
            out = {k[4:] if k.startswith("loc_") else k: np.concatenate([g[k] for g in got]) for k in keys}
            if not len(out["pos"]):
                continue
            over = float((out["miss"] > GOAL_HALF_W).mean())
            print(f"{preset + tag:<21}{np.median(out['pos']):>11.3f}m{np.percentile(out['pos'], 95):>7.3f}"
                  f"{math.degrees(np.median(out['yaw'])):>12.2f}°{math.degrees(np.percentile(out['yaw'], 95)):>7.2f}"
                  f"{np.median(out['miss']):>17.3f}m{np.percentile(out['miss'], 95):>7.3f}"
                  f"{over:>16.0%}"
                  f"{np.median(out['disagree']) if len(out['disagree']) else 0.0:>15.3f}m")


if __name__ == "__main__":
    main()
