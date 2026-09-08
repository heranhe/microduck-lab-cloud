"""Roadmap Track 4 item 12a: replay recorded PLAY swings on the kick bench.

    uv run python scripts/probe_kick_line.py --seeds 48 --ball-out-s 5 \
        --dump-state --out swings.jsonl
    uv run python scripts/replay_kick_swings.py --swings swings.jsonl

The answer, on 386 swings over 48 seeds (2026-09-08): the swing is fine and
the ball is not in front of it. `bench_kick_headdown --from-swings` shows
that nothing about the recorded STATE matters; this shows the other half,
which a ladder of "change one thing" cells cannot reach without a positive
control — move the ball back onto the recipe's sweet spot, hold the exact
play state, and the whiff goes 56% -> 0% with the foot reaching the ball in
100% of swings. Under the arena's whole swing protocol as well (rung 10).

The bench (`scripts/bench_kick_headdown.py`) whiffs 0% from 0.09 m ahead;
play whiffs ~90% from the same offsets. This replays each recorded swing —
root pose, every joint angle and velocity, the lagged `joint_vel` and
`last_action` blocks the obs carries, the servo targets, and the ball's own
pose and velocity — in the kick recipe's own env, under a LADDER of cells
that walks from the bench's world to the arena's.

The ladder, not a single number, is the point: each rung adds exactly one
difference between the two, so the rung where the whiff rate jumps names
the cause. The roadmap's three candidates are rungs 2-4 (the arrival pose,
the moving ball); the rest are differences the roadmap did not list and
that fall out of reading the two step loops side by side:

  * the arena runs a kick window at STANDING_GAIN_RATIO = 0.8 x the
    walking Kp (`arena._set_gain_ratio`, robotd's standing transition).
    The bench, and the recipe that TRAINED the kick, run at 1.0.
  * the arena's ball is condim 6 with rolling friction 0.002 (`compose`,
    the 2026-09-06 rolling-resistance fix). The bench's ball is upstream's
    `ball.xml`, condim 3 — where MuJoCo ignores the rolling coefficient
    entirely. The bench's 0% was measured on a frictionless ball.
  * the arena hands the kick network the reflex tier for KICK_S = 0.5 s
    and then hands back to the WALKER; the bench runs the kick for 1.2 s.

Every cell reports the whiff two ways, because play and bench did not use
the same definition: `moved` is the ball's total displacement over 2 s
(the play probe's `dist < 0.10`), `along` is displacement along the body
heading (the bench's). A cell is compared against the play outcome of the
SAME swings, printed at the top.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from microduck_local import contract as C
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.behaviors.kick import _kick_ball_ids
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer

CARRY_S = 2.0                    # world/metrics.CARRY_S: the window play scores a kick over
KICK_S = 0.5                     # world/arena.KICK_S
STANDING_GAIN_RATIO = 0.8        # world/arena.STANDING_GAIN_RATIO
PLAY_BALL_FRICTION = (0.5, 0.005, 0.002)   # world/scenario.Ball.rolling, on a condim-6 geom

# label -> the one thing this rung changes from the rung above it.
# state:  "recipe" (the recipe's own reset), "home" (HOME pose, still, ball
#         where play had it), "exact" (the recorded state).
# ball_v: keep the recorded ball velocity, or zero it.
# gain:   actuator Kp scale.  ball: "bench" | "play".  window: None | 0.5 s.
LADDER = (
    # The 2x2 that separates the DUCK's state from the BALL's placement.
    # Rungs 1 and 2 put the ball back on the recipe's sweet spot (0.09 m
    # ahead, 0.042 to the foot's side of the duck's ACTUAL root and yaw);
    # rungs 3 and 4 leave it where play had it.
    ("1 home  + spot ball",   dict(state="home",  ball_xy="recipe", ball_v=False, gain=1.0, ball="bench", window=None)),
    ("2 exact + spot ball",   dict(state="exact", ball_xy="recipe", ball_v=False, gain=1.0, ball="bench", window=None)),
    ("3 home  + play ball",   dict(state="home",  ball_xy="play",   ball_v=False, gain=1.0, ball="bench", window=None)),
    ("4 exact + play ball",   dict(state="exact", ball_xy="play",   ball_v=False, gain=1.0, ball="bench", window=None)),
    # …then the rest of the differences between the two step loops.
    ("5  + the ball's motion", dict(state="exact", ball_xy="play",  ball_v=True,  gain=1.0, ball="bench", window=None)),
    ("6  + play ball physics", dict(state="exact", ball_xy="play",  ball_v=True,  gain=1.0, ball="play",  window=None)),
    ("7  + play gain x0.8",   dict(state="exact", ball_xy="play",   ball_v=True,  gain=STANDING_GAIN_RATIO, ball="play", window=None)),
    ("8  + play 0.5 s window", dict(state="exact", ball_xy="play",  ball_v=True,  gain=STANDING_GAIN_RATIO, ball="play", window=KICK_S)),
    ("9 recipe reset",        dict(state="recipe", ball_xy="play",  ball_v=True,  gain=1.0, ball="bench", window=None)),
    # The decision cell: the play state AND the arena's whole swing protocol
    # (its ball, its 0.8 gain, its 0.5 s window then the walker) — with the
    # ball moved back onto the recipe's spot. If this is ~0%, then putting
    # the ball where the foot goes is sufficient in play as it stands, and
    # nothing about the swing itself needs retraining.
    ("10 play protocol, spot ball",
     dict(state="exact", ball_xy="recipe", ball_v=False, gain=STANDING_GAIN_RATIO, ball="play", window=KICK_S)),
)


def _ball_geom(env) -> int:
    return mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")


def _yaw_of(q) -> float:
    qw, qx, qy, qz = (float(v) for v in q[3:7])
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


_ENVS: dict[tuple[str, int], tuple] = {}


def env_for(foot: str, steps: int):
    """One env per (foot, episode length), reused across every swing and cell.

    Building a BehaviorEnv outside a `shared_model_scope` COMPILES the MJCF
    privately (walk_env: `scope is None` -> `from_xml_path`), which is ~1 s;
    553 of those is the whole experiment's runtime. Reuse is safe only
    because every model field a cell touches is restored from the pristine
    copy taken here — never multiplied in place, which is how a per-episode
    `*=` silently walks Kp to zero (AGENTS.md rule 0).
    """
    key = (foot, steps)
    got = _ENVS.get(key)
    if got is None:
        # obs_noise=False: the ARENA hands a skill a noise-free 61-obs
        # (`WorldDuck.obs`), while BehaviorEnv defaults it True even under
        # domain_rand=False — so the shipped bench has been grading the kick
        # on NOISIER observations than play gives it. That difference cannot
        # explain play whiffing MORE, but a replay of play must not carry it.
        env = BehaviorEnv(f"kick_{foot}", seed=7, max_episode_s=steps * C.CTRL_DT + 0.5,
                          domain_rand=False, random_yaw=False, obs_noise=False)
        env.reset(seed=7)
        gid = _ball_geom(env)
        pristine = {"kp": env.model.actuator_gainprm[:, 0].copy(),
                    "bias": env.model.actuator_biasprm[:, 1].copy(),
                    "condim": int(env.model.geom_condim[gid]),
                    "friction": env.model.geom_friction[gid].copy()}
        got = _ENVS[key] = (env, gid, pristine)
    return got


def setup(row: dict, cell: dict, foot: str, steps: int):
    """Put one recorded play swing into the bench env under one cell.

    The ONLY place a cell is turned into physics — `look12a.py` renders
    through this same function, because a renderer that grew its own copy of
    this setup silently drew a different experiment from the one the table
    reported (it did, once, before this was factored out)."""
    st = row["state"]
    env, gid, pristine = env_for(foot, steps)
    env.reset(seed=7)
    d, m = env.data, env.model

    # --- the actuator gain the swing runs at ------------------------------
    # Set ABSOLUTELY, off the pristine copy, never by *= .
    m.actuator_gainprm[:, 0] = pristine["kp"] * cell["gain"]
    m.actuator_biasprm[:, 1] = pristine["bias"] * cell["gain"]

    # --- the ball's contact model -----------------------------------------
    if cell["ball"] == "play":
        m.geom_condim[gid] = 6
        m.geom_friction[gid] = PLAY_BALL_FRICTION
    else:
        m.geom_condim[gid] = pristine["condim"]
        m.geom_friction[gid] = pristine["friction"]

    # --- the state ---------------------------------------------------------
    _, qadr, dadr = _kick_ball_ids(env)
    if cell["state"] != "recipe":
        d.qpos[0:7] = st["root_qpos"]
        d.qpos[qadr:qadr + 7] = st["ball_qpos"]
        d.qvel[dadr:dadr + 6] = st["ball_qvel"]
        if cell["ball_xy"] == "recipe":
            # kick.py `_kick_reset_for`, from the duck's ACTUAL root and yaw:
            # the sweet spot the recipe trains on, under this play state.
            yaw0 = _yaw_of(d.qpos)
            sgn = -1.0 if foot == "right" else 1.0
            ox, oy = 0.09, sgn * 0.042                       # kick.BALL_OFFSET
            d.qpos[qadr + 0] = float(d.qpos[0]) + math.cos(yaw0) * ox - math.sin(yaw0) * oy
            d.qpos[qadr + 1] = float(d.qpos[1]) + math.sin(yaw0) * ox + math.cos(yaw0) * oy
            d.qpos[qadr + 2] = 0.035                         # kick.BALL_Z
            d.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
            d.qvel[dadr:dadr + 6] = 0.0
        if cell["state"] == "exact":
            d.qvel[0:6] = st["root_qvel"]
            d.qpos[env.joint_qpos_adr] = st["joint_qpos"]
            d.qvel[env.joint_qvel_adr] = st["joint_qvel"]
            d.ctrl[:] = st["ctrl"]
            env.last_action[:] = np.asarray(st["last_action"], np.float32)
            env.prev_action = env.last_action.copy()
            env.prev_joint_vel[:] = np.asarray(st["prev_joint_vel"], np.float32)
        else:                                   # "home": the pose the recipe trains from
            d.qvel[0:6] = 0.0
            d.qpos[env.joint_qpos_adr] = C.DEFAULT_POSE
            d.qvel[env.joint_qvel_adr] = 0.0
            d.ctrl[:] = C.DEFAULT_POSE
            env.last_action[:] = 0.0
            env.prev_action = env.last_action.copy()
            env.prev_joint_vel[:] = 0.0
    if not cell["ball_v"]:
        d.qvel[dadr:dadr + 6] = 0.0
    env.twist_cmd[:] = 0.0
    env.head_cmd[:] = 0.0
    env.body_cmd[:] = 0.0
    if getattr(env, "bam", None) is not None:
        env.bam.reset(d.qpos[env.joint_qpos_adr])
    mujoco.mj_forward(m, d)
    yaw = _yaw_of(d.qpos)
    env._kick_dir = (math.cos(yaw), math.sin(yaw))

    return env, gid, qadr, dadr


def one(row: dict, cell: dict, foot: str, walker, kick, steps: int, verify: dict | None) -> dict:
    st = row["state"]
    env, gid, qadr, dadr = setup(row, cell, foot, steps)
    d, m = env.data, env.model

    # Read the knobs back off the env that is about to RUN (AGENTS.md rule 0:
    # never assert on a freshly constructed one). Recorded once per cell.
    if verify is not None and "kp" not in verify:
        verify["kp"] = float(m.actuator_gainprm[0, 0])
        verify["bias"] = float(m.actuator_biasprm[0, 1])
        verify["condim"] = int(m.geom_condim[gid])
        verify["rolling"] = float(m.geom_friction[gid][2])
        verify["obs_noise"] = bool(getattr(env, "obs_noise", None))
        verify["actuator"] = getattr(env, "actuator_model", "?")
        if cell["state"] == "exact":
            verify["joint_max_err"] = float(np.abs(
                np.asarray(d.qpos[env.joint_qpos_adr]) - np.asarray(st["joint_qpos"])).max())

    obs = env._get_obs()
    x0, y0 = float(d.qpos[qadr]), float(d.qpos[qadr + 1])
    n_kick = steps if cell["window"] is None else int(round(cell["window"] / C.CTRL_DT))
    fell, touched = False, 0
    for k in range(steps):
        infer = kick if k < n_kick else walker
        obs, _, term, _, _ = env.step(infer(obs))
        # Did the kicking foot ever reach the ball at all? A swing that never
        # made contact and one that made a bad contact are different failures.
        for c in range(int(d.ncon)):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if gid in (g1, g2) and m.geom_bodyid[g1 if g2 == gid else g2] != 0:
                touched += 1
                break
        if term:
            fell = True
            break
    dx, dy = float(d.qpos[qadr]) - x0, float(d.qpos[qadr + 1]) - y0
    return {"moved": math.hypot(dx, dy),
            "along": dx * env._kick_dir[0] + dy * env._kick_dir[1],
            "fell": fell, "touched": touched > 0,
            "ahead": row.get("ahead"), "side": row.get("side"),
            "play_moved": row.get("dist")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swings", required=True)
    ap.add_argument("--limit", type=int, default=0, help="replay at most this many swings (0 = all)")
    ap.add_argument("--seconds", type=float, default=CARRY_S, help="how long to run each replay")
    ap.add_argument("--cells", default="", help="comma-separated rung prefixes; default all")
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.swings) if x.strip()]
    rows = [r for r in rows if r.get("state") and r["foot"] in ("kick_left", "kick_right")]
    if args.limit:
        rows = rows[:args.limit]
    steps = int(round(args.seconds / C.CTRL_DT))
    ladder = [(lab, c) for lab, c in LADDER
              if not args.cells or lab.split()[0] in args.cells.split(",")]

    walker = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    # The kick each foot ACTUALLY runs in play — env override, local export,
    # then the shipped file — not a hardcoded path that can silently differ
    # from the policy the funnel was measured with (AGENTS.md rule 0).
    from microduck_local.world import World
    kicks = {f: onnx_infer(World.skill_path(f"kick_{f}")) for f in ("left", "right")}

    play_whiff = np.mean([r["dist"] < 0.10 for r in rows])
    print(f"{len(rows)} play swings from {args.swings}; replayed for {args.seconds:g} s each")
    print(f"PLAY, these same swings: whiff (ball moved < 10 cm in {CARRY_S:g} s) {play_whiff:.0%}\n")
    print(f"  {'cell':<24}{'n':>4}{'whiff':>7}{'moved med':>11}{'along med':>11}{'touched':>9}{'falls':>7}")
    out: dict[str, list[dict]] = {}
    for label, cell in ladder:
        verify: dict = {}
        res = [one(r, cell, r["foot"].split("_")[1], walker, kicks[r["foot"].split("_")[1]],
                   steps, verify) for r in rows]
        out[label] = res
        w = np.mean([x["moved"] < 0.10 for x in res])
        print(f"  {label:<24}{len(res):>4}{w:>7.0%}"
              f"{np.median([x['moved'] for x in res]):>10.2f}m"
              f"{np.median([x['along'] for x in res]):>10.2f}m"
              f"{np.mean([x['touched'] for x in res]):>9.0%}"
              f"{sum(x['fell'] for x in res):>7}")
        print(f"      [ran with Kp {verify['kp']:.4f} bias {verify['bias']:.4f} "
              f"ball condim {verify['condim']} rolling {verify['rolling']:g} "
              f"actuator {verify['actuator']} obs_noise {verify['obs_noise']}"
              + (f" pose set to {verify['joint_max_err']:.1e} rad]" if "joint_max_err" in verify else "]"))

    for last in [lb for lb, _ in ladder if lb.split()[0] in ("2", "4")]:
        print(f"\n  '{last}', by where the ball was ahead of the root — the roadmap's funnel:")
        print(f"    {'band':<14}{'n':>4}{'bench whiff':>13}{'play whiff':>12}{'foot touched':>14}")
        for lo, hi in ((0.0, 0.08), (0.08, 0.11), (0.11, 0.15), (0.15, 0.20), (0.20, 9.0)):
            g = [x for x in out[last] if x["ahead"] is not None and lo <= x["ahead"] < hi]
            if g:
                print(f"    {lo:.2f}-{hi:.2f} m{len(g):>7}"
                      f"{np.mean([x['moved'] < 0.10 for x in g]):>13.0%}"
                      f"{np.mean([x['play_moved'] < 0.10 for x in g]):>12.0%}"
                      f"{np.mean([x['touched'] for x in g]):>14.0%}")
    # The recipe's sweet spot is a POINT (0.09 ahead, 0.042 to the foot's
    # side +- 0.015), not an "ahead" band: the funnel above collapses the
    # side away, and a ball at the right distance on the wrong side is the
    # same row as one the foot passes straight through.
    picked = [lb for lb, _ in ladder if lb.startswith("4")]
    if not picked:
        return
    ex = out[picked[0]]
    print("\n  the same swings by BOTH offsets ('exact + play ball'; the recipe trains on "
          "0.075-0.105 ahead x 0.027-0.057 to the kicking foot's side):")
    print(f"    {'side |0.027-0.057|':<22}{'n':>4}{'bench whiff':>13}{'play whiff':>12}{'foot touched':>14}")
    for lo, hi, name in ((0.0, 0.027, "side < 0.027"), (0.027, 0.057, "side 0.027-0.057 (spot)"),
                         (0.057, 0.09, "side 0.057-0.09"), (0.09, 9.0, "side >= 0.09")):
        g = [x for x in ex if x["side"] is not None and lo <= abs(x["side"]) < hi]
        if g:
            print(f"    {name:<22}{len(g):>4}{np.mean([x['moved'] < 0.10 for x in g]):>13.0%}"
                  f"{np.mean([x['play_moved'] < 0.10 for x in g]):>12.0%}"
                  f"{np.mean([x['touched'] for x in g]):>14.0%}")
    box = [x for x in ex if x["ahead"] is not None
           and 0.075 <= x["ahead"] <= 0.105 and 0.027 <= abs(x["side"]) <= 0.057]
    print(f"    ON the recipe's own box (both offsets): {len(box)} of {len(ex)} swings"
          + (f", bench whiff {np.mean([x['moved'] < 0.10 for x in box]):.0%}, "
             f"play whiff {np.mean([x['play_moved'] < 0.10 for x in box]):.0%}" if box else ""))
    json.dump({k: v for k, v in out.items()}, open(Path(args.swings).with_suffix(".replay.json"), "w"))


if __name__ == "__main__":
    main()
