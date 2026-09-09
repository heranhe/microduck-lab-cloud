"""Is the ToF floor blob the BALL, in the state where it would be used?

`tof_ball_m` has been measured off twice, and both times by pooling every tick
the blob fired: it is redundant on 88-94% of them (the camera already has the
ball) and wrong about two times in three in the blind case, which ends in a
line-up on somebody's foot. Item 12e proposes something those numbers cannot
answer, because they never split on it: use the blob **only during the line-up
and the settle**, and only when no body is beside us.

That is a different population. During the settle the camera IS blind - that
is the whole of item 12 - so the blob is not redundant there, and `_beside`
already knows about the failure case. So this counts the blob's events the way
the decision would use them:

    uv run python scripts/probe_tof_ball.py --seeds 4 --episodes 30

For every tick the blob fires it records the brain's state, whether a body is
beside the duck, whether the camera had the ball anyway, and whether the blob
was actually the ball (its implied position within `--tol` of the truth). The
number that decides 12e is the last column of the `lineup/settle, nobody
beside` row: a sensor that is right less than about four times in five there
cannot be allowed to move a kick spot, because the cost of the other case is a
fall.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.controllers import tof_floor_ball
from microduck_local.brain.team import brain_kwargs
from microduck_local.world import World, make_pitch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kick_gym import _place, gym_scenario  # noqa: E402


def run(seed: int, episodes: int, r_max: float, tol: float, spread: float,
        per_side: int = 0, seconds: float = 300.0) -> list[dict]:
    """`per_side` 0 runs the gym (line-up dense, barely contested); 2 or 3 runs
    a real match, which is where the old measurement's failure mode lives - a
    blob that is the OTHER DUCK'S FOOT. The gym alone would flatter this."""
    if per_side:
        sc = make_pitch(per_side=per_side, formation=True)
    else:
        sc = gym_scenario(opponents=1)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={x.id: infer for x in sc.ducks}, seed=seed, ball_out_s=5.0)
    teams: dict = {}
    brains = {x.id: REGISTRY.make("chase", **brain_kwargs(x, w, teams)) for x in sc.ducks}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    if per_side:
        q = int(w.model.jnt_qposadr[w._ball_joint])
        w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
        episodes, seconds_per = 1, seconds
    else:
        seconds_per = 20.0
    for _ in range(episodes):
        if not per_side:
            _place(w, rng, spread)
            for b in brains.values():
                b.reset()
        t0 = w.t
        while w.t - t0 < seconds_per:
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
                if did != "d0":
                    continue
                fr = s.fresh_tof(b.TOF_MAX_AGE)
                blob = None if fr is None else tof_floor_ball(fr, r_max=r_max)
                if blob is None:
                    continue
                # Where the blob says the ball is, in the odometry frame the
                # brain steers by, against where it truly is.
                ox, oy, oyaw = w.odom(d)
                bx = ox + blob[1] * math.cos(oyaw + blob[0])
                by = oy + blob[1] * math.sin(oyaw + blob[0])
                true = w.ball_xy()
                seen_cam = bool(det is not None and any(x.cls == "ball" for x in det.detections)
                                and w.t - det.t <= b.DET_MAX_AGE)
                rows.append({"state": b.state, "beside": bool(b._beside(w.t)),
                             "cam": seen_cam,
                             "is_ball": bool(true is not None and math.dist((bx, by), true) <= tol)})
            w.step()
    return rows


def report(rows: list[dict], tol: float) -> None:
    if not rows:
        print("the blob never fired")
        return
    print(f"\n{len(rows)} blob events; 'is the ball' = implied position within {tol:.2f} m of the truth\n")
    print(f"{'population':<40}{'events':>8}{'camera had it':>15}{'IS THE BALL':>13}")

    def line(lab, rs):
        if not rs:
            print(f"{lab:<40}{0:>8}{'—':>15}{'—':>13}")
            return
        print(f"{lab:<40}{len(rs):>8}{100 * np.mean([r['cam'] for r in rs]):>14.0f}%"
              f"{100 * np.mean([r['is_ball'] for r in rs]):>12.0f}%")

    line("every tick it fires", rows)
    lu = [r for r in rows if r["state"] in ("lineup", "settle")]
    line("lineup / settle", lu)
    line("lineup / settle, nobody beside", [r for r in lu if not r["beside"]])
    line("lineup / settle, a body beside", [r for r in lu if r["beside"]])
    blind = [r for r in lu if not r["cam"] and not r["beside"]]
    line("…and the camera blind (the case for it)", blind)
    print("\nstates it fires in:", dict(Counter(r["state"] for r in rows).most_common(6)))
    print("\nREAD THE LAST TWO ROWS. The blob is only worth having where the camera is "
          "blind, and it is only safe where no body is beside the duck. If it is not the "
          "ball in most of that population, 12e is dead however it is gated.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--r-max", type=float, default=0.5, help="the ToF floor-blob radius (ChaseParams.tof_ball_m)")
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--spread", type=float, default=0.8)
    ap.add_argument("--per-side", type=int, default=0,
                    help="0 = the gym; 2 or 3 = a real match, where a blob can be a foot")
    ap.add_argument("--seconds", type=float, default=300.0, help="match length when --per-side is set")
    a = ap.parse_args()
    rows: list[dict] = []
    for s in range(a.seed0, a.seed0 + a.seeds):
        rows += run(s, a.episodes, a.r_max, a.tol, a.spread, a.per_side, a.seconds)
    report(rows, a.tol)
