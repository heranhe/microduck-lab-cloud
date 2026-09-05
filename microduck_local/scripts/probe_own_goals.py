"""Baseline probe for docs/roadmap.md Track 4 (positional soccer): how many
goals today are OWN goals — the credited team's attacked mouth is not the
mouth scored on, credit going to the last kick within KICK_GOAL_S, else to
the last team on the ball — and how many kicks are aimed AWAY from the goal
the kicker attacks (the chase brain's `aim_max` fallback to the line of
sight). The same loop as `eval_pitch.run_one`, plus the kick line read off
the brain (`Chase._hunt_u`) the tick a kick fires.

    cd microduck_local && uv run python scripts/probe_own_goals.py <seed> 300 <per_side>

Prints one JSON row. Measured 2026-09-05 (seeds 0-3): 1v1 14 of 26 kicks
aimed away from the attacked goal, 3 of 6 goals own; 2v2 14 of 27 and 8 of
8. These fields become `eval-pitch`'s under roadmap item 1.1.
"""
import json
import math
import sys

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.world import World, make_pitch
from microduck_local.world.metrics import PitchMetrics


def run(seed, seconds, per_side):
    sc = make_pitch(per_side=per_side)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    rng = np.random.default_rng(seed)
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    team_of = {d.id: d.team for d in sc.ducks}
    metrics = PitchMetrics(w, team_of)
    attacks = {tm: ("right" if s > 0 else "left") for tm, s in metrics.sign.items()}
    goal_seq = 0
    goals, kicks = [], []
    own_half_ticks = {tm: 0 for tm in attacks}   # ticks the ball sits in a team's OWN half
    ticks = 0
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            b = brains[d.id]
            intent = b.step(s)
            if intent.skill is not None and b.goal is not None and b._hunt_u is not None:
                bx, by = w.ball_xy()
                gdir = math.atan2(b.goal[1] - by, b.goal[0] - bx)
                kicks.append({"t": round(w.t, 2), "duck": d.id, "team": team_of[d.id],
                              "cos": round(math.cos(b._hunt_u - gdir), 3)})
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        ticks += 1
        bx, _ = w.ball_xy()
        for tm, sgn in metrics.sign.items():
            if bx * sgn < 0:
                own_half_ticks[tm] += 1
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            holder, ago = metrics._holder, (w.t - metrics._holder_t)
            mouth = w.last_goal
            goals.append({"t": round(w.t, 1), "mouth": mouth, "by": holder,
                          "ago": round(ago, 1), "own": bool(holder and attacks[holder] != mouth)})
            metrics.tick()
            kickoff_brains(brains, teams)
        else:
            metrics.tick()
    back = [k for k in kicks if k["cos"] < 0]
    return {"seed": seed, "perSide": per_side, "goals": goals, "nGoals": len(goals),
            "ownGoals": sum(g["own"] for g in goals), "kicks": len(kicks), "backKicks": len(back),
            "kickList": kicks,
            "ballOwnHalfFrac": {tm: round(v / max(ticks, 1), 3) for tm, v in own_half_ticks.items()},
            "falls": {k: d.falls for k, d in w.ducks.items()}, **metrics.row()}

if __name__ == "__main__":
    seed, seconds, per_side = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    print(json.dumps(run(seed, seconds, per_side)), flush=True)
