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
from microduck_local.brain.knob_gates import warning_for
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

# THE EXIT WINDOW (roadmap 12at, 2026-09-10). The sidecar `exit_rad` the
# selector aims with is a BENCH median (`scripts/bench_kick_headdown.py`), and
# the sidecar of the pair it replaced records that the same foot read -0.16 on
# the bench and +0.26 in play. Nothing here measured the in-play exit per
# swing, so the selector's aim error was never a number.
#
# The exit is the ball's travel direction over the FIRST 0.5 s after the touch,
# not over the whole carry window: at 1.4 m/s off the foot and 0.3 m/s^2 of
# rolling resistance the ball has run ~0.66 m by then and has not yet reached a
# board on this 3.0 x 2.5 m pitch from most placements, so the direction is the
# kick's and not the wall's. `advance` below keeps the full CARRY_S window,
# because that is what the ledger's `kicksBack` scores.
EXIT_S = 0.5
# Under this the direction is numerical noise, not a line: a ball nudged 2 cm
# has an angle, but not one worth a row. Half the whiff threshold, so every
# connected swing (>= 0.10 m over CARRY_S) that actually left in the first
# half-second is measured and a stationary ball reads None rather than a
# uniformly-distributed angle that would bias a median toward nothing.
EXIT_MIN_M = 0.05


def _wrap(a: float) -> float:
    """An angle into (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def exit_angle(ball0, ball1, yaw: float, min_m: float = EXIT_MIN_M) -> float | None:
    """The direction a ball travelled from `ball0` to `ball1`, in the BODY
    frame of a duck whose world yaw was `yaw` at the swing — the same sign
    convention as the sidecar's `exit_rad` and `ChaseParams.kick_exit_*`:
    POSITIVE is to the duck's LEFT.

    None when the ball moved under `min_m`: a stationary ball has no
    direction, and giving it one would put uniform noise in the median.
    """
    dx, dy = float(ball1[0]) - float(ball0[0]), float(ball1[1]) - float(ball0[1])
    if math.hypot(dx, dy) < min_m:
        return None
    return _wrap(math.atan2(dy, dx) - yaw)


def aim_error(exit_play: float | None, aim_body: float | None) -> float | None:
    """Realised minus intended, wrapped: how far off the line the selector
    laid the ball actually left. Both arguments are in the body frame at the
    swing, so no odometry drift enters. None when either is missing."""
    if exit_play is None or aim_body is None:
        return None
    return _wrap(exit_play - aim_body)


def gym_scenario(size=(3.0, 2.5), goal_width=0.7, opponents: int = 0,
                 cove: float = 0.0, corner: float = 0.0) -> Scenario:
    """One duck at the centre facing +x, one ball, boards, and a goal to aim
    at (the brain needs one to lay a kick line). No team: with a single duck
    `brain_kwargs` gives no blackboard, so there is no attacker/support
    churn, no yielding and no avoid - the swing is the only thing happening."""
    hx, hy = size[0] / 2, size[1] / 2
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    if corner > 0:
        # THE LAB'S BOARDS (roadmap item 14): chamfered corners and, below, the
        # cove - the quarter-round that parks a ball off the wall and returns
        # one rolled into it. Flat boards stay the default: every gym number
        # before 2026-09-10 is on them, and a flat wall has no restitution,
        # so a ball pushed into it simply dies - which is the difference a
        # board PUSH is most sensitive to (12an).
        c = min(float(corner), hx - 0.05, hy - 0.05)
        pts = [(-hx + c, -hy), (hx - c, -hy), (hx, -hy + c), (hx, hy - c),      # make_pitch's own list, counter-
               (hx - c, hy), (-hx + c, hy), (-hx, hy - c), (-hx, -hy + c)]      # clockwise, endpoints shared
        walls = [Wall(pts[i], pts[(i + 1) % len(pts)], 0.3, 0.02) for i in range(len(pts))]
    else:
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
                    balls=[Ball((0.5, 0.0))], ducks=ducks, goal_width=goal_width, cove=float(cove))


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


def _board_rect(w: World) -> tuple[float, float]:
    """The BOARD half-extents, read off the scenario's walls rather than
    recomputed — the floor is 0.25 m larger than the boards on each side, and
    confusing the two is what makes a placement look 25 cm further out than it
    is."""
    xs = [abs(c[0]) for wall in w.scenario.walls for c in (wall.start, wall.end)]
    ys = [abs(c[1]) for wall in w.scenario.walls for c in (wall.start, wall.end)]
    return max(xs), max(ys)


def _place_at_boards(w: World, rng: np.random.Generator, margin: float):
    """One episode's start with the ball AT A BOARD — the population the
    boards line (`board_margin`) exists for, and the one the open-play draw
    almost never reaches.

    Measured on the shipped draw (200k samples): the ball lands within 0.40 m
    of a board on 6.2% of episodes and within 0.15 m on 1.0%, because the duck
    spawns at x = -0.9 and draws 0.45-1.4 m ahead, which does not reach the
    end boards at all and reaches the side boards only in the tail. The clip
    is against the FLOOR (0.25 m outside the boards), so it caps placement at
    10 cm from a board rather than excluding the band — but only 0.62% of
    draws are clipped at all. Either way there is no population to measure, so
    an arm about the boards run in the open-play gym would measure the
    baseline with extra steps.

    The ball is drawn on a side chosen in proportion to its LENGTH, uniformly
    along it (corners left in: they are part of the population, and 12m
    measured them as slower to leave but not traps), and `margin` out from it.
    The duck is then spawned a normal walk-in away — the SAME 0.45-1.4 m range
    the open-play draw uses, so the two modes differ in where the ball is and
    not in how far the duck walks — at a bearing that keeps it on the pitch."""
    bx_h, by_h = _board_rect(w)
    d = w.ducks["d0"]
    r_ball = w.scenario.balls[0].radius
    # Sides in proportion to length, so a long board is not under-sampled.
    lens = np.array([2 * by_h, 2 * by_h, 2 * bx_h, 2 * bx_h], dtype=float)
    side = int(rng.choice(4, p=lens / lens.sum()))
    out = float(rng.uniform(r_ball + 0.01, max(margin, r_ball + 0.02)))
    if side < 2:                                   # the +x / -x end boards
        sx = 1.0 if side == 0 else -1.0
        bx, by = sx * (bx_h - out), float(rng.uniform(-by_h + 0.05, by_h - 0.05))
        inward = (-sx, 0.0)
    else:                                          # the +y / -y side boards
        sy = 1.0 if side == 2 else -1.0
        bx, by = float(rng.uniform(-bx_h + 0.05, bx_h - 0.05)), sy * (by_h - out)
        inward = (0.0, -sy)

    # The duck: the same walk-in range as open play, on a bearing that leaves
    # it inside the boards. Biased toward `inward` because a spawn behind the
    # board is not a spawn.
    base = math.atan2(inward[1], inward[0])
    for _ in range(40):
        rng_m = float(rng.uniform(0.45, 1.4))
        th = base + float(rng.uniform(-1.2, 1.2))
        dx, dy = bx + rng_m * math.cos(th), by + rng_m * math.sin(th)
        if abs(dx) < bx_h - 0.25 and abs(dy) < by_h - 0.25:
            break
    else:                                          # straight in from the board
        rng_m = 0.9
        dx, dy = bx + rng_m * inward[0], by + rng_m * inward[1]
        dx = float(np.clip(dx, -bx_h + 0.25, bx_h - 0.25))
        dy = float(np.clip(dy, -by_h + 0.25, by_h - 0.25))
    yaw = math.atan2(by - dy, bx - dx) + float(rng.uniform(-0.25, 0.25))
    d.spawn = (dx, dy, yaw)
    w._respawn(d)
    j = w._ball_joint
    q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
    w.data.qpos[q:q + 7] = [bx, by, r_ball + 0.005, 1.0, 0.0, 0.0, 0.0]
    w.data.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(w.model, w.data)
    return q, v


def _place_at_corner(w: World, rng: np.random.Generator, margin: float):
    """The ball in a CORNER — within `margin` of TWO boards — and the duck a
    normal walk-in away, spawned on the diagonal into the pitch.

    This exists to falsify the feasibility model, not to explore. Measured by
    a 1 cm sweep of real `_along_the_boards` calls on `gym_scenario()` bounds:
    a corner spot for `board_margin` 0.25 is **infeasible for every gap below
    0.33 m**, so a draw capped at 0.30 or under can never let that knob act.

    The cap therefore matters more than it looks. At `--at-corners 0.35` or
    above, part of the draw is genuinely feasible, a real effect there would be
    EXPECTED, and reading it as a falsification would be a false rejection —
    the opposite direction to every other error in this file. Keep the cap at
    0.30 or below and the whole draw sits inside the infeasible region.
    """
    bx_h, by_h = _board_rect(w)
    d = w.ducks["d0"]
    r_ball = w.scenario.balls[0].radius
    lo, hi = r_ball + 0.01, max(margin, r_ball + 0.02)
    sx = 1.0 if rng.random() < 0.5 else -1.0
    sy = 1.0 if rng.random() < 0.5 else -1.0
    bx = sx * (bx_h - float(rng.uniform(lo, hi)))
    by = sy * (by_h - float(rng.uniform(lo, hi)))
    base = math.atan2(-sy, -sx)                       # the diagonal, into the pitch
    for _ in range(40):
        rng_m = float(rng.uniform(0.45, 1.4))
        th = base + float(rng.uniform(-0.9, 0.9))
        dx, dy = bx + rng_m * math.cos(th), by + rng_m * math.sin(th)
        if abs(dx) < bx_h - 0.25 and abs(dy) < by_h - 0.25:
            break
    else:
        dx = float(np.clip(bx - 0.9 * sx, -bx_h + 0.25, bx_h - 0.25))
        dy = float(np.clip(by - 0.9 * sy, -by_h + 0.25, by_h - 0.25))
    yaw = math.atan2(by - dy, bx - dx) + float(rng.uniform(-0.25, 0.25))
    d.spawn = (dx, dy, yaw)
    w._respawn(d)
    j = w._ball_joint
    q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
    w.data.qpos[q:q + 7] = [bx, by, r_ball + 0.005, 1.0, 0.0, 0.0, 0.0]
    w.data.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(w.model, w.data)
    return q, v


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
        knobs: str = "", at_boards: float = 0.0, at_corners: float = 0.0,
        cove: float = 0.0, corner: float = 0.0) -> list[dict]:
    # An ARM is a `MICRODUCK_CHASE` string, applied here so it lands in the
    # worker process before any brain is built (`brain_kwargs` reads
    # `ChaseParams.from_env()`). Every arm of a comparison runs the same seeds
    # and the same episodes, so the difference is the knob and nothing else.
    if knobs:
        os.environ["MICRODUCK_CHASE"] = knobs
    else:
        os.environ.pop("MICRODUCK_CHASE", None)
    sc = gym_scenario(opponents=opponents, cove=cove, corner=corner)
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
    # The board rectangle, for the re-bin: every distance below is to the
    # nearest BOARD, not the floor edge (they differ by 0.25 m).
    bx_h, by_h = _board_rect(w)

    def to_board(x: float, y: float) -> float:
        return min(bx_h - abs(x), by_h - abs(y))

    last_out_t = None                  # when the referee last teleported the ball
    rows = []
    for ep in range(episodes):
        q, v = (_place_at_corner(w, rng, at_corners) if at_corners > 0.0
                else _place_at_boards(w, rng, at_boards) if at_boards > 0.0
                else _place(w, rng, spread))
        # THE EPISODE'S INPUT, recorded at placement and put on EVERY row.
        # A rate needs its denominator binned on the same axis as its
        # numerator: the first version wrote the ball's distance only inside
        # the swing branch, so the no-swing episodes had no axis at all and the
        # curve this exists for could not be computed. `place_board` is where
        # the ball STARTED; `ball_board` on a swing row is where it was when
        # the skill fired, which is not the same after an approach.
        place_xy = (float(w.data.qpos[q]), float(w.data.qpos[q + 1]))
        place_board = round(to_board(*place_xy), 4)
        for b in brains.values():
            b.reset()
        t0 = w.t
        outs0 = w.ball_outs
        swing = None
        prev_skill = None
        last_spot, last_spot_t = None, None
        last_sel = None             # kick_select's Verdict for that latched plan (12at)
        pre_track = None            # (age, sigma, hits) of the ball track at the decision tick
        pushes0 = brain.pushes      # a PUSH is a touch too (roadmap 12g): the brain counts them, the skill never starts
        while w.t - t0 < EPISODE_S:
            # WHAT A SWING GATE WOULD SEE (roadmap 12c, second half, 2026-09-10):
            # the best ball track's age and 1-sigma error at the tick the
            # brain decides on, read BEFORE the brain steps - at the swing it
            # has already called `tracker.disturb`, and a row that reads the
            # track afterwards measures the disturbance, not the decision.
            trk = brain.tracker.best(brain.p.target_cls, w.t, min_hits=1)
            pre_track = None if trk is None else (
                trk.age(w.t), trk.sigma(w.t, brain.tracker.p.vel_prior, brain.tracker.p.vel_sig_after_s), trk.hits)
            _drive(w, brains)
            # LATCH the plan. `brain.spot` is consumed by the time the kick
            # skill takes the body -- reading it AT the swing returns None -- so
            # the last non-None value is the plan the swing was decided on.
            # Still read off the brain, never recomputed from the ball.
            if brain.spot is not None:
                last_spot, last_spot_t = brain.spot, w.t
                # `_plan` sets `last_select` on its way to setting `spot`, so
                # the verdict latched here is the one that chose this plan's
                # line and foot (None when `kick_select` declined or is off).
                last_sel = brain.last_select
            outs_before = w.ball_outs
            w.step()
            if w.ball_outs != outs_before:
                # A throw-in teleports the ball and tells no brain (roadmap
                # 12n): beliefs are corrupted for about a second afterwards, so
                # the row records how long ago it was and the analysis can split
                # those episodes out instead of averaging the corruption in.
                last_out_t = w.t
            pushed = brain.pushes > pushes0
            if pushed or (d.skill is not None and prev_skill is None and str(d.skill).startswith("kick")):
                # THE SWING - or the PUSH, the moment the brain commits to
                # walking through the ball (12g): the same row, `touch` says
                # which, and the carry window below measures both alike.
                # Everything the three candidates of item 12a need,
                # captured at the instant the skill takes the body.
                p = d.trunk_pos(w.data)
                yaw = d.yaw(w.data)
                odom_now = w.odom(d) or (0.0, 0.0, 0.0)
                bx, by = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
                dx, dy = bx - float(p[0]), by - float(p[1])
                joints = np.asarray(w.data.qpos[d.adr.joint_qpos], float)
                swing = {
                    "ep": ep, "foot": "push" if pushed else str(d.skill), "touch": "push" if pushed else "kick",
                    "t": round(w.t - t0, 2),
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
                    # THE FALL COLUMN (2026-09-10, after 12ai's ledger read falls
                    # 4 -> 9 on 24 seeds and could not resolve them): the kicking
                    # duck's arena fall counter at the swing, so `fell` below is
                    # "went down between the swing and the end of the carry
                    # window" - the swing's own fall, not the approach's.
                    "falls_before": int(d.falls),
                    "outs_during_approach": w.ball_outs - outs0,
                    "plan_age": round(w.t - getattr(brain, "t_state", w.t), 2),
                    # The quantity `_too_far` gates on, recorded at the swing:
                    # how far AHEAD the brain's own predicted ball is, in its
                    # own odom frame. Not the same as `ahead` above, which is
                    # the TRUE ball in the trunk frame -- the gate sees the
                    # belief, not the truth. With the gate off this says which
                    # swings it WOULD have declined, so the value of declining
                    # them can be measured instead of assumed.
                    "pred_ahead": None if brain.predicted is None else round(
                        (brain.predicted[0] - odom_now[0]) * math.cos(odom_now[2])
                        + (brain.predicted[1] - odom_now[1]) * math.sin(odom_now[2]), 4),
                    # ...and its SIDE offset in the same frame (+: left), so the
                    # belief's whole offset from the sweet spot can be read.
                    "pred_side": None if brain.predicted is None else round(
                        -(brain.predicted[0] - odom_now[0]) * math.sin(odom_now[2])
                        + (brain.predicted[1] - odom_now[1]) * math.cos(odom_now[2]), 4),
                    # The belief's own 1-sigma error (m) at the decision tick,
                    # and the track it rests on: seconds since its last hit and
                    # its hit count. The "now" gate 12c asks for (no swing at a
                    # track older than 0.3 s or wider than 5 cm) can only act
                    # on swings these columns show it; measure that first.
                    "pred_sigma": None if brain.predicted_sigma is None else round(brain.predicted_sigma, 4),
                    "track_age": None if pre_track is None else round(pre_track[0], 3),
                    "track_sigma": None if pre_track is None else round(pre_track[1], 4),
                    "track_hits": None if pre_track is None else int(pre_track[2]),
                    # --- the re-bin ---
                    # The ball's own distance to the nearest board, so the curve
                    # can be binned by where the ball ACTUALLY was rather than by
                    # the arm's `--at-boards` cap (which is a cumulative bound,
                    # not a distance).
                    "ball_board": round(to_board(bx, by), 4),
                    "place_board": place_board,
                    "place": [round(c, 4) for c in place_xy],
                    # The planned spot READ OFF THE BRAIN at the instant the
                    # swing was decided -- never recomputed from `ball0` at
                    # analysis time. Those differ by the plan's staleness, so a
                    # reconstruction would answer "where would the spot be for a
                    # ball there" instead of "where was the spot when the swing
                    # was decided" (roadmap 12a: median plan age 3.6 s).
                    "spot": None if last_spot is None
                    else [round(float(last_spot[0]), 4), round(float(last_spot[1]), 4),
                          last_spot[2] or last_spot[4]],
                    "spot_board": None if last_spot is None
                    else round(to_board(float(last_spot[0]), float(last_spot[1])), 4),
                    "spot_age": None if last_spot_t is None else round(w.t - last_spot_t, 2),
                    # --- THE EXIT (12at) ---
                    # The three raw angles the exit and the aim error are
                    # built from, all recorded rather than reconstructed, so
                    # an analysis that disagrees with `exit_play` below can
                    # say where. `spot_head` is the BODY HEADING `_plan` laid
                    # the spot in, in the ODOM frame (`self.spot[3]`, which
                    # the old three-field `spot` column above throws away);
                    # `odom_yaw` is the body's odom yaw at the swing, so the
                    # two differ by the line-up's residual; `swing_yaw` is the
                    # true world yaw, which is what the exit is measured
                    # against. Odometry drift cancels in `spot_head -
                    # odom_yaw` and never enters `exit_play`.
                    "spot_head": None if last_spot is None else round(float(last_spot[3]), 4),
                    "odom_yaw": round(float(odom_now[2]), 4),
                    "swing_yaw": round(float(yaw), 4),
                    # What the brain BELIEVES this foot's exit is: the sidecar
                    # value `World.kick_exits()` put into `ChaseParams` (or the
                    # shipped default when there is no sidecar). This is the
                    # number the selector aimed with, and the one `exit_play`
                    # is the audit of. A push leaves along the walk: 0.0.
                    "exit_assumed": round(float(
                        brain.p.kick_exit_left if str(d.skill) == "kick_left"
                        else brain.p.kick_exit_right if str(d.skill) == "kick_right" else 0.0), 4),
                    # The selector's own verdict for the latched plan, so "the
                    # aim was back toward our own mouth" and "the kick went
                    # back" can be told apart: a line the selector KNEW was
                    # 20% own-goal is a different failure from one it thought
                    # was safe and the foot bent round.
                    "sel_foot": None if last_sel is None else str(last_sel.foot),
                    "sel_p_own": None if last_sel is None else round(float(last_sel.p_own), 4),
                    "sel_p_goal": None if last_sel is None else round(float(last_sel.p_goal), 4),
                    # Seconds since the last throw-in, or None if there has not
                    # been one this run.
                    "since_out": None if last_out_t is None else round(w.t - last_out_t, 2),
                }
                prev_skill = d.skill
                break
            prev_skill = d.skill
        if swing is None:
            # The LATCHED PLAN goes on the no-swing rows too. `place_board`
            # gave the rate a denominator; the mechanism question -- is the
            # collapse "no legal spot exists"? -- needs the spot on the
            # episodes that did NOT swing, which are exactly the ones it is
            # about. Recording it only where a swing happened is the same
            # selection error one level down.
            rows.append({"ep": ep, "swing": False, "arm": knobs, "live": live,
                         "place_board": place_board, "place": [round(c, 4) for c in place_xy],
                         "spot": None if last_spot is None
                         else [round(float(last_spot[0]), 4), round(float(last_spot[1]), 4),
                               last_spot[2] or last_spot[4]],
                         "spot_board": None if last_spot is None
                         else round(to_board(float(last_spot[0]), float(last_spot[1])), 4)})
            continue
        # let the ball run, then measure how far the swing actually sent it
        ts = w.t
        b_exit = None               # where the ball was EXIT_S after the touch (12at)
        while w.t - ts < SETTLE_S:
            _drive(w, brains)
            w.step()
            if b_exit is None and w.t - ts >= EXIT_S:
                b_exit = (float(w.data.qpos[q]), float(w.data.qpos[q + 1]))
        bx1, by1 = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
        travel = math.dist(swing["ball0"], (bx1, by1))
        # ...and how much of it went TOWARD the goal (+x is the attacked
        # mouth in this gym): the number a board touch is for (12g).
        advance = bx1 - swing["ball0"][0]
        # THE EXIT, realised (12at): the ball's line over the first EXIT_S in
        # the body frame at the swing, and the error against the line the plan
        # laid. `aim_body` is the intended BALL direction relative to the body
        # at the swing -- the spot's heading turned into the body frame plus
        # the exit the brain assumed for the foot it swung, which is exactly
        # the quantity `kickselect.evaluate` rolls out (`u + model.exit`). With
        # `kick_deflect_*` at their shipped 0 the spot heading IS the aim line
        # `u`, so `aim_body` reduces to `exit_assumed` for a duck that finished
        # its line-up, and the residual says how much of the error is the
        # line-up rather than the foot.
        e_play = None if b_exit is None else exit_angle(swing["ball0"], b_exit, swing["swing_yaw"])
        aim_body = (None if swing["spot_head"] is None
                    else _wrap(swing["spot_head"] - swing["odom_yaw"]) + swing["exit_assumed"])
        swing.update(swing=True, travel=travel, advance=round(advance, 4), whiff=travel < WHIFF_M, seed=seed,
                     arm=knobs, live=live, fell=int(d.falls) > swing["falls_before"],
                     # The ledger's own rule, per swing: `Metrics._resolve_kicks`
                     # counts a kick BACK when the signed displacement along the
                     # attacked axis over CARRY_S is negative. Same window, same
                     # sign, so the gym's back-share and `kicksBack` are the same
                     # quantity measured on different populations.
                     back=bool(advance < 0.0),
                     exit_travel=None if b_exit is None else round(math.dist(swing["ball0"], b_exit), 4),
                     exit_play=None if e_play is None else round(e_play, 4),
                     aim_body=None if aim_body is None else round(_wrap(aim_body), 4),
                     aim_err=None if aim_error(e_play, aim_body) is None else round(aim_error(e_play, aim_body), 4))
        swing.pop("ball0")
        rows.append(swing)
    return rows


def _run(a):
    return run(*a)


BANDS = ((0.00, 0.08), (0.08, 0.11), (0.11, 0.15), (0.15, 0.20), (0.20, 9.9))

# The scatter `kickselect` already assumes about the exit line
# (`ChaseParams.kick_select_dir_sd`, shipped 0.6 rad = 34.4 deg). A systematic
# aim error INSIDE this is already in the model's variance; one outside it is a
# bias the roll-out cannot see, and every own-goal filter rests on the roll-out.
SELECTOR_DIR_SD = 0.6


def _q(xs, f: float) -> float:
    return float(np.quantile(np.asarray(xs, float), f))


def exit_summ(rows: list[dict]) -> dict:
    """Per foot: the realised in-play exit and the aim error, as median and
    IQR, plus the share of swings whose aim error is outside the scatter the
    selector assumes. Keys are the skill names (`kick_left`, `kick_right`,
    `push`); a foot with no measured exit is absent rather than zero.

    Reads only `.get`, so a row file written before 12at summarises to `{}`
    and every caller of this module keeps working on it."""
    out: dict[str, dict] = {}
    by_foot: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("swing") and r.get("exit_play") is not None:
            by_foot.setdefault(str(r.get("foot")), []).append(r)
    for foot, rs in sorted(by_foot.items()):
        ex = [float(r["exit_play"]) for r in rs]
        er = [float(r["aim_err"]) for r in rs if r.get("aim_err") is not None]
        assumed = [float(r["exit_assumed"]) for r in rs if r.get("exit_assumed") is not None]
        d = {"n": len(rs), "exit_med": _q(ex, 0.5), "exit_q1": _q(ex, 0.25), "exit_q3": _q(ex, 0.75),
             "assumed": (float(np.median(assumed)) if assumed else None),
             "back": sum(bool(r.get("back")) for r in rs) / len(rs)}
        if er:
            d.update(n_err=len(er), err_med=_q(er, 0.5), err_q1=_q(er, 0.25), err_q3=_q(er, 0.75),
                     err_abs_med=_q([abs(v) for v in er], 0.5),
                     err_outside=sum(abs(v) > SELECTOR_DIR_SD for v in er) / len(er))
        out[foot] = d
    return out


def report_exit(rows: list[dict]) -> None:
    """`exit_summ` as the roadmap reads it: degrees, per foot, against the
    sidecar the selector aimed with. Silent on a row file with no exit
    column, so every pre-12at `--out` file prints exactly what it did."""
    s = exit_summ(rows)
    if not s:
        return
    deg = math.degrees
    print(f"\nin-play exit over the first {EXIT_S:g} s, body frame at the swing "
          f"(+ = to the duck's LEFT; sidecar = what the selector aimed with):")
    print(f"{'foot':<12}{'n':>5}{'exit med':>10}{'IQR':>17}{'sidecar':>9}{'off by':>8}"
          f"{'aim err med':>13}{'|err| med':>10}{'>±34°':>7}{'back':>7}")
    for foot, d in s.items():
        iqr = f"{deg(d['exit_q1']):+.0f}..{deg(d['exit_q3']):+.0f}°"
        sc = "  -  " if d["assumed"] is None else f"{deg(d['assumed']):+.0f}°"
        off = "  -  " if d["assumed"] is None else f"{deg(d['exit_med'] - d['assumed']):+.0f}°"
        em = f"{deg(d['err_med']):+.0f}°" if "err_med" in d else "  -  "
        ea = f"{deg(d['err_abs_med']):.0f}°" if "err_abs_med" in d else "  -  "
        eo = f"{100 * d['err_outside']:.0f}%" if "err_outside" in d else "  - "
        print(f"{foot:<12}{d['n']:>5}{deg(d['exit_med']):>+9.0f}°{iqr:>17}{sc:>9}{off:>8}"
              f"{em:>13}{ea:>10}{eo:>7}{100 * d['back']:>6.0f}%")
    print(f"  `off by` is the in-play median minus the sidecar `exit_rad` the brain aimed with;"
          f"\n  `>±34°` is the share of swings whose aim error exceeds the {SELECTOR_DIR_SD:g} rad scatter"
          "\n  kickselect already samples (`kick_select_dir_sd`) — a bias inside it is priced in,"
          "\n  one outside it is invisible to the own-goal filter. `back` is the ledger's rule"
          "\n  (`advance` < 0 over the carry window), per foot.")


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
    fk = [r for r in sw if r.get("fell") is not None]      # rows written before the column exists carry no verdict
    if fk:
        print(f"fell inside the {SETTLE_S:g} s after the swing: {sum(r['fell'] for r in fk)} of {len(fk)} "
              f"({100 * np.mean([r['fell'] for r in fk]):.1f}%)")
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
    report_exit(sw)
    print("\nREAD IT AGAINST roadmap Track 4 item 12's 372-swing match table. If the "
          "0.08-0.11 row is ~94% here too, the failure reproduces with ONE duck and "
          "can be iterated in minutes. If it is low, what breaks the kick is something "
          "only the match has.")


# A whiff-rate null is only a null if this many swings could have SEEN the
# shift it denies.  Same rule the pitch battery now carries (playbook item 3):
# the MDE is the interval half-width, and a difference is significant exactly
# when it exceeds it.
TIGHT_PP = 0.08         # 8 percentage points -- tight enough to call a null
TARGET_PP = 0.10        # the shift the footer sizes an arm for


def two_proportions(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float, float]:
    """(shift in the whiff rate, p, MDE) for two independent event counts.
    The MDE is 1.96 SE, so `p < 0.05` holds exactly when |shift| > MDE."""
    if not (n1 and n2):
        return 0.0, 1.0, float("inf")
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2)) if 0 < pool < 1 else 0.0
    if se == 0:
        return p2 - p1, 1.0, float("inf")
    return p2 - p1, math.erfc(abs((p2 - p1) / se) / math.sqrt(2)), 1.96 * se


def outcome_key(rows: "list[dict]") -> tuple:
    """An arm's outcome, episode by episode, to the precision that matters.
    Two arms that produce this identically did not differ in the physics."""
    return tuple((bool(r.get("swing")), None if not r.get("swing")
                  else round(float(r.get("travel", 0.0)), 9)) for r in rows)


def is_identical(base_rows: "list[dict]", arm_rows: "list[dict]") -> bool:
    """Playbook rule 0, made mechanical: a knob that changes NOTHING is broken,
    not null.  The gym is deterministic per seed and episode, so an arm whose
    every episode matches the baseline's did not reach the code it was meant to
    change -- usually a precondition it does not satisfy (`contest_margin` is
    gated on `use_color`, which is OFF by default, so an arm that sets only the
    margin runs the shipped path and reports a beautiful, meaningless null)."""
    return len(base_rows) == len(arm_rows) and outcome_key(base_rows) == outcome_key(arm_rows)


def verdict_prop(p: float, mde: float, tight: float = TIGHT_PP) -> str:
    """`effect`, `null`, or `NO RESULT` -- never `null` for an arm too small
    to have resolved the shift it is denying."""
    if p < 0.05:
        return "effect"
    return "null" if mde <= tight else "NO RESULT"


def swings_for(mde: float, n1: int, n2: int, target: float = TARGET_PP) -> int:
    """Swings PER ARM that would resolve `target`.  MDE falls as 1/sqrt(n)."""
    if not math.isfinite(mde) or mde <= 0 or target <= 0:
        return 0
    per = 0.5 * (n1 + n2)
    return max(2, math.ceil(per * (mde / target) ** 2))


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
    print(f"{'arm':<24}{'swings':>8}{'whiff':>8}{'vs base':>9}{'±MDE':>7}{'p':>8}  verdict"
          f"{'':4}{'fell':>6}{'p':>7}")
    n1, x1 = len(b_sw), sum(r["whiff"] for r in b_sw)

    def falls(rs):
        fk = [r for r in rs if r.get("fell") is not None]
        return (sum(r["fell"] for r in fk), len(fk)) if fk else (None, 0)
    f1, m1 = falls(b_sw)
    print(f"{base + ' (base)':<24}{n1:>8}{100 * x1 / max(n1, 1):>7.0f}%{'—':>9}{'—':>7}{'—':>8}"
          f"{'':13}{(f'{100 * f1 / m1:.1f}%' if m1 else '—'):>6}{'—':>7}")
    thin, broken = [], []
    for lab in labels[1:]:
        sw = [r for r in arms[lab] if r.get("swing")]
        n2, x2 = len(sw), sum(r["whiff"] for r in sw)
        if not (n1 and n2):
            continue
        d, pv, mde = two_proportions(x1, n1, x2, n2)
        if is_identical(arms[base], arms[lab]):
            broken.append(lab)
            v = "BROKEN"
        else:
            v = verdict_prop(pv, mde)
            if v == "NO RESULT":
                thin.append((lab, swings_for(mde, n1, n2)))
        f2, m2 = falls(sw)
        fp = two_proportions(f1, m1, f2, m2)[1] if (m1 and m2) else None
        print(f"{lab:<24}{n2:>8}{100 * x2 / n2:>7.0f}%{100 * d:>+8.0f}%"
              f"{100 * mde:>6.0f}%{pv:>8.3f}  {v:<11}"
              f"{(f'{100 * f2 / m2:.1f}%' if m2 else '—'):>6}{(f'{fp:.3f}' if fp is not None else '—'):>7}")
    for lab in broken:
        print(f"\n!! {lab} reproduced the baseline EPISODE FOR EPISODE. A knob that changes"
              "\n   nothing is BROKEN, not null (playbook rule 0): it never reached the code"
              "\n   it meant to change. Check its preconditions -- `contest_margin`,"
              "\n   `opp_keepout` and `_is_mate` are all gated on `use_color`, which is OFF"
              "\n   by default, so an arm must set `use_color=1` alongside them.")
    print("\nA drop in whiff on FEWER swings is not a win: read both columns.")
    print("±MDE is the smallest whiff shift this many swings could have resolved:"
          "\na verdict is `null` only under " f"{100 * TIGHT_PP:.0f}" " points, else NO RESULT.")
    for lab, need in thin:
        print(f"  {lab}: {need} swings an arm would resolve "
              f"{100 * TARGET_PP:.0f} points (had {len(arms[lab])} episodes).")


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
    ap.add_argument("--at-boards", type=float, default=0.0, metavar="M",
                    help="place the ball within M metres of a BOARD instead of in open "
                         "play, and spawn the duck a normal walk-in away. 0 (default) is "
                         "the open-play draw, which reaches within 0.40 m of a board on "
                         "only 6.2%% of episodes — there is no boards population in it to "
                         "measure. Try --at-boards 0.25.")
    ap.add_argument("--at-corners", type=float, default=0.0, metavar="M",
                    help="place the ball within M of TWO boards (a corner). Keep M <= 0.30: "
                         "a corner spot for board_margin 0.25 is infeasible below a 0.33 m "
                         "gap, so 0.30 and under is a clean falsifier and 0.35+ silently "
                         "stops being one. EXPECTED VERDICT FOR 0.25 HERE IS *BROKEN*.")
    ap.add_argument("--cove", type=float, default=0.0, metavar="R",
                    help="the lab's quarter-round at the boards (0 = flat boards, the gym's default; the lab runs 0.15)")
    ap.add_argument("--corner", type=float, default=0.0, metavar="L",
                    help="chamfer each corner at 45 deg starting this far along each wall (the lab runs 0.3)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    specs = [(s.split("=", 1)[0], s.split("=", 1)[1] if "=" in s else "") for s in (a.arm or ["ambient="])]
    if a.at_corners > 0.0:
        print(f"\n--at-corners {a.at_corners}: EXPECTED VERDICT for board_margin >= 0.20 is"
              "\n  BROKEN, and here that is the PASS. The feasibility model says no corner"
              "\n  spot exists below a 0.33 m gap, so the knob cannot act and must reproduce"
              "\n  the baseline episode for episode. Any effect FALSIFIES the model."
              "\n  A BROKEN line here is the prediction holding, not a defect -- do not"
              "\n  'fix' the knob until it stops. It also does NOT show that acting in a"
              "\n  corner would help: that is a separate question this arm cannot answer.\n")
    # Preflight, before a minute of compute is spent: a knob gated behind a
    # knob that ships off measures the shipped path and reports a null about
    # nothing.  `is_identical` catches that afterwards; this catches the
    # commonest form of it now, in about a second, from the source.
    for label, knobs in specs:
        warn = warning_for(knobs)
        if warn:
            print(f"\n[{label}] {warn}\n")
    arms: dict[str, list[dict]] = {}
    for label, knobs in specs:
        args = [(s, a.episodes, a.spread, a.opponents, a.ball_out_s, knobs,
                 a.at_boards, a.at_corners, a.cove, a.corner)
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
