"""THE KICK GYM: one duck, one ball, one swing, repeated.

The kick is measured in two places today and they disagree, which is why
nobody can fix it:

  * `bench_kick_headdown.py` spawns a STANDING duck at HOME with the ball on
    a chosen offset and fires the skill. 0% whiff. Too clean: no walk-in, no
    settle, no gait phase, a perfectly still ball.
  * `probe_kick_line.py` reads swings out of a contested 2v2 match. 94% whiff
    where the plan puts the ball. Too dirty: opponents, teammates, a
    blackboard, avoid/yield/block, and only 83 swings in the band that
    matters out of 48 seeds x 300 s of compute.

This is the rung between them. ONE duck and ONE ball on a pitch with a goal,
driven by the real `chase` brain, so the approach, the settle, the gait phase
and the stale plan are all REAL - and nothing else is. Every episode is one
placement and one swing, so the swings-per-minute is set by the kick and not
by how long it takes to win possession, and the ball's start is CHOSEN rather
than whatever the match happened to produce.

    uv run python kick_gym.py --episodes 60 --out gym.jsonl
    uv run python kick_gym.py --episodes 200 --jobs 4 --out gym.jsonl

It prints the same funnel table as roadmap Track 4 item 12, so the two are
read side by side. The question it exists to answer first: does a clean
single-duck approach reproduce the 94%? If it does, the fix can be iterated
here in minutes instead of an hour a battery. If it does NOT, then what
breaks the kick is something only the match has, and that is the finding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

from microduck_local import contract as C
from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.controllers import ChaseParams
from microduck_local.world.arena import World
from microduck_local.world.metrics import CARRY_S
from microduck_local.world.scenario import Ball, Duck, Scenario, Wall

# EXACTLY the match probe's definitions, so the two funnels are comparable:
# `probe_kick_line.py` calls a swing a whiff when the ball moved under 0.10 m
# in the CARRY_S window after it. Using anything shorter here would count more
# whiffs than the match does and flatter the gym.
WHIFF_M = 0.10
SETTLE_S = CARRY_S
EPISODE_S = 25.0        # a walk-in from ~1 m plus a line-up; beyond this the episode is a no-swing


def gym_scenario(size=(3.0, 2.5), goal_width=0.7, opponents: int = 0) -> Scenario:
    """One duck at the centre facing +x, one ball, boards, and a goal to aim
    at (the brain needs one to lay a kick line). No team: with a single duck
    `brain_kwargs` gives no blackboard, so there is no attacker/support
    churn, no yielding and no avoid - the swing is the only thing happening."""
    hx, hy = size[0] / 2, size[1] / 2
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    walls = [Wall(corners[i], corners[(i + 1) % 4], 0.3, 0.02) for i in range(4)]
    ducks = [Duck("d0", (-0.9, 0.0, 0.0), None, "datasheet", "datasheet", "chase", team="cream")]
    # An OPPONENT is the one thing the match has that the clean gym does not:
    # another body contesting the same ball, which brings `avoid`, `yield`,
    # `blocked`, contact, and a ball that is being pushed by somebody else.
    # Spawned facing the ball from the far side so it genuinely contests.
    for i in range(opponents):
        ducks.append(Duck(f"o{i}", (0.9 + 0.3 * i, 0.0, math.pi), None, "datasheet", "datasheet",
                          "chase", team="graphite"))
    return Scenario(name="kick-gym", floor=(size[0] + 0.5, size[1] + 0.5), walls=walls,
                    balls=[Ball((0.5, 0.0))], ducks=ducks, goal_width=goal_width)


def _drive(w: World, brains: dict) -> None:
    """One control tick for every duck in the gym."""
    for did, b in brains.items():
        x = w.ducks[did]
        tof, det = x.tof.last, x.detector.last
        s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                   det=det, det_age=None if det is None else w.t - det.t,
                   speed=x.heading_speed(w.data), odom=w.odom(x), skill=x.skill,
                   bumped=w.bumped(x))
        it = b.step(s)
        w.apply_intent(x, it)
        if x.skill is None:
            x.set_cmd(w.data, it.twist, it.head)


def _place(w: World, rng: np.random.Generator, spread: float):
    """One episode's start: the duck on its spawn (a little yaw jitter so the
    approach is never the same twice), the ball ahead of it at a drawn range
    and bearing. The RANGE is the thing being swept - a ball put down 0.5 m
    away is a short walk-in, 1.4 m is a long one, and the plan's age at the
    swing scales with it."""
    d = w.ducks["d0"]
    yaw = float(rng.uniform(-0.25, 0.25))
    d.spawn = (-0.9, float(rng.uniform(-0.3, 0.3)), yaw)
    w._respawn(d)
    rng_m = float(rng.uniform(0.45, 1.4))
    bear = float(rng.uniform(-spread, spread))
    bx = d.spawn[0] + rng_m * math.cos(yaw + bear)
    by = d.spawn[1] + rng_m * math.sin(yaw + bear)
    hx, hy = w.scenario.floor[0] / 2 - 0.35, w.scenario.floor[1] / 2 - 0.35
    bx, by = float(np.clip(bx, -hx, hx)), float(np.clip(by, -hy, hy))
    j = w._ball_joint
    q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
    w.data.qpos[q:q + 7] = [bx, by, w.scenario.balls[0].radius + 0.005, 1.0, 0.0, 0.0, 0.0]
    w.data.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(w.model, w.data)
    return q, v


def run(seed: int, episodes: int, spread: float, opponents: int = 0, ball_out_s: float = 0.0,
        knobs: str = "") -> list[dict]:
    # An ARM is a `MICRODUCK_CHASE` string, applied here so it lands in the
    # worker process before any brain is built (`brain_kwargs` reads
    # `ChaseParams.from_env()`). Every arm of a comparison runs the same seeds
    # and the same episodes, so the difference is the knob and nothing else.
    if knobs:
        os.environ["MICRODUCK_CHASE"] = knobs
    else:
        os.environ.pop("MICRODUCK_CHASE", None)
    sc = gym_scenario(opponents=opponents)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={x.id: infer for x in sc.ducks}, seed=seed, ball_out_s=ball_out_s)
    bk = __import__("microduck_local.brain.team", fromlist=["brain_kwargs"]).brain_kwargs
    teams: dict = {}
    brains = {x.id: REGISTRY.make("chase", **bk(x, w, teams)) for x in sc.ducks}
    brain = brains["d0"]
    d = w.ducks["d0"]
    # Playbook rule 0: read the knobs back off the CONSTRUCTED brain, never
    # off a fresh ChaseParams(), so an arm that changes nothing says so.
    live = {k: getattr(brain.p, k) for k in sorted(ChaseParams.env_names())} if knobs else {}
    live["_tracker_rest_coast_s"] = brain.tracker.p.rest_coast_s
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        q, v = _place(w, rng, spread)
        for b in brains.values():
            b.reset()
        t0 = w.t
        outs0 = w.ball_outs
        swing = None
        prev_skill = None
        while w.t - t0 < EPISODE_S:
            _drive(w, brains)
            w.step()
            if d.skill is not None and prev_skill is None and str(d.skill).startswith("kick"):
                # THE SWING. Everything the three candidates of item 12a need,
                # captured at the instant the skill takes the body.
                p = d.trunk_pos(w.data)
                yaw = d.yaw(w.data)
                bx, by = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
                dx, dy = bx - float(p[0]), by - float(p[1])
                joints = np.asarray(w.data.qpos[d.adr.joint_qpos], float)
                swing = {
                    "ep": ep, "foot": str(d.skill), "t": round(w.t - t0, 2),
                    # where the ball was, in the duck's own yaw frame
                    "ahead": dx * math.cos(yaw) + dy * math.sin(yaw),
                    "side": -dx * math.sin(yaw) + dy * math.cos(yaw),
                    # candidate (ii): was it moving?
                    "ball_speed": float(math.hypot(w.data.qvel[v], w.data.qvel[v + 1])),
                    # candidate (i): did the settle arrive at the pose the kick trained from?
                    "pose_max_dev": float(np.max(np.abs(joints - C.DEFAULT_POSE))),
                    "pose_rms_dev": float(np.sqrt(np.mean((joints - C.DEFAULT_POSE) ** 2))),
                    "head_pitch": float(joints[C.JOINT_NAMES.index("head_pitch")])
                    if "head_pitch" in C.JOINT_NAMES else None,
                    "neck_pitch": float(joints[C.JOINT_NAMES.index("neck_pitch")])
                    if "neck_pitch" in C.JOINT_NAMES else None,
                    "ball0": (bx, by),
                    "outs_during_approach": w.ball_outs - outs0,
                    "plan_age": round(w.t - getattr(brain, "t_state", w.t), 2),
                }
                prev_skill = d.skill
                break
            prev_skill = d.skill
        if swing is None:
            rows.append({"ep": ep, "swing": False, "arm": knobs, "live": live})
            continue
        # let the ball run, then measure how far the swing actually sent it
        ts = w.t
        while w.t - ts < SETTLE_S:
            _drive(w, brains)
            w.step()
        bx1, by1 = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
        travel = math.dist(swing["ball0"], (bx1, by1))
        swing.update(swing=True, travel=travel, whiff=travel < WHIFF_M, seed=seed,
                     arm=knobs, live=live)
        swing.pop("ball0")
        rows.append(swing)
    return rows


def _run(a):
    return run(*a)


BANDS = ((0.00, 0.08), (0.08, 0.11), (0.11, 0.15), (0.15, 0.20), (0.20, 9.9))


def report(rows: list[dict]) -> None:
    sw = [r for r in rows if r.get("swing")]
    print(f"\n{len(rows)} episodes, {len(sw)} produced a swing "
          f"({100 * len(sw) / max(len(rows), 1):.0f}%); whiff = travel < {WHIFF_M} m\n")
    if not sw:
        print("no swings: the duck never got to a kick. Raise --episodes or EPISODE_S.")
        return
    print(f"{'ball ahead of the root':<34}{'swings':>8}{'moved < 10 cm':>16}")
    for lo, hi in BANDS:
        b = [r for r in sw if lo <= r["ahead"] < hi]
        if not b:
            continue
        lab = f"{lo:.2f}-{hi:.2f} m" if hi < 9 else f">= {lo:.2f} m"
        if abs(lo - 0.08) < 1e-9:
            lab += "  (kick_ahead: where the plan puts it)"
        print(f"{lab:<34}{len(b):>8}{100 * np.mean([r['whiff'] for r in b]):>15.0f}%")
    spot = [r for r in sw if 0.06 <= r["ahead"] <= 0.10 and 0.04 <= abs(r["side"]) <= 0.08]
    if spot:
        print(f"{'on the sweet spot':<34}{len(spot):>8}{100 * np.mean([r['whiff'] for r in spot]):>15.0f}%")
    print(f"\noverall whiff {100 * np.mean([r['whiff'] for r in sw]):.0f}%  "
          f"| median ahead {np.median([r['ahead'] for r in sw]):.3f} m  "
          f"side {np.median([abs(r['side']) for r in sw]):.3f} m")
    hit = [r for r in sw if not r["whiff"]]
    miss = [r for r in sw if r["whiff"]]

    def col(rs, k):
        return f"{np.median([r[k] for r in rs]):.3f}" if rs else "  -  "
    print("\nthe three candidates of item 12a, connected swings vs whiffs:")
    print(f"{'':<26}{'connected':>12}{'whiffed':>12}")
    for k, lab in (("ball_speed", "ball speed m/s (ii)"),
                   ("pose_max_dev", "max joint dev (i)"),
                   ("pose_rms_dev", "rms joint dev (i)"),
                   ("head_pitch", "head pitch rad"),
                   ("neck_pitch", "neck pitch rad")):
        if sw[0].get(k) is None:
            continue
        print(f"{lab:<26}{col(hit, k):>12}{col(miss, k):>12}")
    print("\nREAD IT AGAINST roadmap Track 4 item 12's 372-swing match table. If the "
          "0.08-0.11 row is ~94% here too, the failure reproduces with ONE duck and "
          "can be iterated in minutes. If it is low, what breaks the kick is something "
          "only the match has.")


def compare(arms: "dict[str, list[dict]]") -> None:
    """Two or more arms on the same seeds, read the way this repo reads a
    soccer result: the funnel per arm, then the whiff as a PROPORTION of the
    swing events with a two-proportion z on the pooled counts. Swings are
    reported beside it, because a gate that improves the rate by refusing
    most of the touches has not improved anything (playbook rule 6)."""
    labels = list(arms)
    print("\n" + "=" * 78)
    print("A/B on the same seeds and episodes")
    print("=" * 78)
    for lab in labels:
        sw = [r for r in arms[lab] if r.get("swing")]
        live = next((r.get("live") for r in arms[lab] if r.get("live")), None) or {}
        changed = {k: v for k, v in live.items() if not k.startswith("_")}
        print(f"\n--- {lab} --- {len(arms[lab])} episodes, {len(sw)} swings"
              + (f"   [tracker rest_coast_s={live.get('_tracker_rest_coast_s')}]" if live else ""))
        report(arms[lab])
        if changed:
            on = {k: v for k, v in changed.items() if v}
            print("  live knobs (off the constructed brain):", on or "none set")
    if len(labels) < 2:
        return
    base = labels[0]
    b_sw = [r for r in arms[base] if r.get("swing")]
    print("\n" + "-" * 78)
    print(f"{'arm':<28}{'swings':>8}{'whiff':>9}{'vs base':>10}{'p':>9}")
    n1, x1 = len(b_sw), sum(r["whiff"] for r in b_sw)
    print(f"{base + ' (base)':<28}{n1:>8}{100 * x1 / max(n1, 1):>8.0f}%{'—':>10}{'—':>9}")
    for lab in labels[1:]:
        sw = [r for r in arms[lab] if r.get("swing")]
        n2, x2 = len(sw), sum(r["whiff"] for r in sw)
        if not (n1 and n2):
            continue
        pooled = (x1 + x2) / (n1 + n2)
        se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)) if 0 < pooled < 1 else 0.0
        z = (x1 / n1 - x2 / n2) / se if se else 0.0
        pv = math.erfc(abs(z) / math.sqrt(2))
        print(f"{lab:<28}{n2:>8}{100 * x2 / n2:>8.0f}%"
              f"{100 * (x2 / n2 - x1 / n1):>+9.0f}%{pv:>9.3f}")
    print("\nA drop in whiff on FEWER swings is not a win: read both columns.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=40, help="episodes PER seed")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--spread", type=float, default=0.8, help="bearing spread of the ball's placement (rad)")
    ap.add_argument("--opponents", type=int, default=0,
                    help="contesting ducks to add (0 = the clean gym; 1 = the match's one difference)")
    ap.add_argument("--ball-out-s", type=float, default=0.0,
                    help="the referee's throw-in, as the match funnel was measured (World.ball_out_s)")
    ap.add_argument("--arm", action="append", default=None, metavar="LABEL=KNOBS",
                    help="an arm to measure, as a MICRODUCK_CHASE string: "
                         "--arm 'shipped=' --arm 'memory=rest_predict_s=6,rest_coast_s=20'. "
                         "Repeat it; every arm runs the SAME seeds and episodes. "
                         "Without it the ambient environment is measured as one arm.")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    specs = [(s.split("=", 1)[0], s.split("=", 1)[1] if "=" in s else "") for s in (a.arm or ["ambient="])]
    arms: dict[str, list[dict]] = {}
    for label, knobs in specs:
        args = [(s, a.episodes, a.spread, a.opponents, a.ball_out_s, knobs)
                for s in range(a.seed0, a.seed0 + a.seeds)]
        rows: list[dict] = []
        if a.jobs > 1 and len(args) > 1:
            with ProcessPoolExecutor(a.jobs) as ex:
                for r in ex.map(_run, args):
                    rows += r
        else:
            for x in args:
                rows += run(*x)
        arms[label] = rows
        if a.out:
            with open(a.out, "a") as fh:
                for r in rows:
                    fh.write(json.dumps({**r, "label": label}) + "\n")
    if len(arms) == 1:
        report(next(iter(arms.values())))
    else:
        compare(arms)
