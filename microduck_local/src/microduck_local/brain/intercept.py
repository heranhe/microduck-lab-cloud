"""Getting in the way of a ball that is rolling into our own goal.

The repo owner, watching a 2v2: the ball trickles toward a duck's OWN goal
and the duck keeps trying to set up a proper kick instead of simply getting
in the way — "come in from the side and deflect it", "push it out of the
way".

Why this is a different problem from the kick, and why it is worth trying
where three aim-side fixes died (`ChaseParams.spot_lead`, `kick_deflect_*`,
`two_stage`): **a block does not need centimetre precision.** Over 191 kicks
in 24 seeds of 2v2 not one had the ball on the kick's sweet spot, because the
line-up aims at where the ball WAS and swings 3.3 s later. A body standing on
the ball's path is a target the width of a duck with a second of arrival
window, and a glancing contact is a perfectly good defensive actuator — every
"kick" in this benchmark already IS one (17.7 cm of ball travel, README).

MEASURED, and read this before tuning anything (`scripts/probe_threat.py`,
48 seeds x 300 s of 2v2, 71 threats, `runs/threat-base48.jsonl`):

  * 54 of 70 decided threats are CONCEDED, so there is room to move.
  * But **40 of the 70 are declared with the ball already inside 0.3 m of
    the goal line, and those are 95% conceded** — a goal in progress, not a
    situation. The threats a block can address are the 29 that start from
    0.8 m or further out, and those are ~50% conceded.
  * The tell that this is worth building at all: **a defender within 0.15 m
    of the ball's path clears 48% of the time, and one 0.15-0.50 m off it
    concedes 31 of 31.** Being one step out of the way scores exactly as
    badly as being on the other side of the pitch. Nothing in the brain
    takes that step.

MEASURED, AND IT SHIPS OFF. `ChaseParams.intercept_eta` carries the full
table; the short version is that the block engages exactly as designed (21%
of a threat's ticks, with `lineup` 21% -> 6%), keeps the ball measurably
further from the mouths (-1.53 s/min inside 0.9 m, p = 0.010 over 96 paired
seeds, replicated on both blocks), costs neither possession nor falls — and
does not resolve the conceded fraction it was built to move (79% -> 74%,
p = 0.34). The ceiling is in the baseline: a block was geometrically
available in only 20 of the 54 conceded threats, so the lever is earlier
than the block.

Two decisions in here are worth reading before changing anything.

**The closing rate is the TRIGGER; the geometry comes from positions.**
Measured over the same threats, a duck's own tracker gets the ball's velocity
DIRECTION wrong by a median 25 deg. A block point laid on that ray would miss
by most of a metre. But a ball that is a threat is, by definition, running
down the line from where it is to our goal — and both ENDS of that line are
known to centimetres (the sighting, and the goal, which does not move). So
the velocity is used only to answer "is it coming at us, and how soon", as a
SCALAR closing rate differenced from the ball's distance to our own goal, and
the standing point is laid on the ball-to-goal line.

**The history is fed from sightings only.** A coasting track's odometry-frame
position moves with the duck that holds it (the bearing turns with the body,
the range does not update), so differencing a coasted track measures the
duck's own walk and calls it a rolling ball. `Team` documents the same trap
for its own fixes and answers it with a speed floor it cannot use here — a
threat ball rolls at 0.26 m/s and `Team.vel_use` is 0.7. So this keeps its
own short history and only ever appends a FRESH sighting.
"""

from __future__ import annotations

import math

# The walker as `brain/team.py` measured it (`walker-facts`): 0.45 m/s, a
# turn in place at ~0.7 rad/s once the gait is going, 0.4 s of cold start,
# and 0.5 rad of bearing the walk's own steering absorbs for free. Constants
# rather than a reach into `Team`, because a lone duck has no board and this
# arithmetic must not change when it gains one.
WALK, TURN_RATE, TURN_FREE, COLD_S = 0.45, 0.7, 0.5, 0.4


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def walk_time(pose: tuple[float, float, float], target: tuple[float, float]) -> float:
    """Seconds for a duck at (x, y, yaw) to stand on `target`."""
    dx, dy = target[0] - pose[0], target[1] - pose[1]
    turn = max(0.0, abs(_wrap(math.atan2(dy, dx) - pose[2])) - TURN_FREE)
    return turn / TURN_RATE + (COLD_S if turn > 0 else 0.0) + math.hypot(dx, dy) / WALK


def block_point(ball: tuple[float, float], own_goal: tuple[float, float],
                me: tuple[float, float], ahead: float, keep: float
                ) -> tuple[float, float] | None:
    """Where to stand: the point of the ball-to-goal line nearest to me, kept
    `ahead` clear of the ball (standing ON it is a line-up, not a block) and
    `keep` clear of the goal (standing in the mouth is an own goal waiting to
    happen, and the walker cannot turn against the boards).

    Nearest-to-me is the earliest point a straight-line walker can reach, so
    it is the interception that leaves the most margin without needing the
    ball's speed to any precision.

    None when those two clearances leave NO room — the ball is already inside
    `ahead + keep` of the goal. That is not a nicety, it is the first thing
    this got wrong: clamped into the gap instead, the standing point lands a
    few centimetres goal-side of a ball that is already at the mouth, and
    walking to it walks the ball in. It is also the case with nothing to win.
    Measured over the baseline threats, a threat declared with the ball
    inside 0.3 m of the line is conceded 38 times in 40 whatever anyone does;
    the ones a block can address start from 0.8 m out."""
    dx, dy = own_goal[0] - ball[0], own_goal[1] - ball[1]
    d = math.hypot(dx, dy)
    if d < ahead + keep:
        return None
    ux, uy = dx / d, dy / d
    s = (me[0] - ball[0]) * ux + (me[1] - ball[1]) * uy
    s = min(max(s, ahead), d - keep)
    return ball[0] + s * ux, ball[1] + s * uy


class Interceptor:
    """The decision and its hysteresis: does this duck leave the play and put
    itself in front of the ball, and where does it stand to do it.

    Stateless between kickoffs except for the sighting history and the
    commitment clock; `reset()` clears both."""

    def __init__(self) -> None:
        self.hist: list[tuple[float, float]] = []      # (t, distance to our own goal)
        self.reset()

    def reset(self) -> None:
        self.hist = []
        self.until = -1e9                        # committed to a block until this time
        self.target: tuple[float, float] | None = None
        self.ball: tuple[float, float] | None = None   # what the block is aimed at, for the caller to face
        self.rate = 0.0                          # the closing rate that fired it, m/s
        self.eta: float | None = None

    def see(self, t: float, sight: tuple[float, float], own_goal: tuple[float, float],
            window: float) -> None:
        """One FRESH sighting of the ball, as its distance to our own goal.
        A scalar, because that is the only part of the ball's motion this has
        to be right about."""
        self.hist.append((t, math.hypot(sight[0] - own_goal[0], sight[1] - own_goal[1])))
        cut = t - 3.0 * window
        while len(self.hist) > 2 and self.hist[0][0] < cut:
            self.hist.pop(0)

    def closing(self, t: float, window: float, max_age: float) -> tuple[float, float] | None:
        """(metres a second the ball is eating into our distance, seconds until
        it arrives), from the sighting history alone. None when the history is
        too short, too old, or the ball is not coming."""
        if len(self.hist) < 2 or t - self.hist[-1][0] > max_age:
            return None
        t1, d1 = self.hist[-1]
        old = [h for h in self.hist if t1 - h[0] >= window]
        if not old:
            return None
        t0, d0 = old[-1]
        rate = (d0 - d1) / (t1 - t0)
        if rate <= 1e-6:
            return None
        return rate, d1 / rate

    def update(self, t: float, p, ball: tuple[float, float] | None,
               sight: tuple[float, float] | None, own_goal: tuple[float, float],
               odom: tuple[float, float, float], duck_id: str,
               mates: list[tuple[str, tuple[float, float, float] | None]]
               ) -> tuple[float, float] | None:
        """Where to stand to block, or None to carry on playing.

        `sight` is a fresh ball position (mine, or the freshest on the team
        board) and is the only thing the trigger is differenced from; `ball`
        is the best current estimate, which is what the geometry uses. Both
        are in this duck's odometry frame, like everything else here."""
        if p.intercept_eta <= 0:
            self.reset()
            return None
        if sight is not None:
            self.see(t, sight, own_goal, p.intercept_dt)
        self.ball = ball
        if ball is None:
            self.target = None
            return None
        c = self.closing(t, p.intercept_dt, p.intercept_age)
        if c is not None and c[0] >= p.intercept_vmin and c[1] <= p.intercept_eta:
            self.rate, self.eta = c
            self.until = t + p.intercept_hold
        elif t >= self.until:
            self.target = None
            return None
        target = block_point(ball, own_goal, (odom[0], odom[1]),
                             p.intercept_ahead, p.intercept_keep)
        if target is None:
            self.target = None
            return None
        # ONE duck goes: whoever has the shortest walk to its own block point.
        # Every teammate runs the same arithmetic on the same blackboard, so
        # nobody has to be told, and ties break on the id — which every duck
        # also agrees on. Without this a threat pulls the whole team off the
        # ball, which is the way this idea most obviously fails.
        mine = walk_time(odom, target)
        for mid, pose in mates:
            if pose is None:
                continue
            spot = block_point(ball, own_goal, (pose[0], pose[1]),
                               p.intercept_ahead, p.intercept_keep)
            if spot is not None and (walk_time(pose, spot), mid) < (mine, duck_id):
                self.target = None
                return None
        # The sweep, the owner's "push it out of the way": with the ball
        # nearly here, aim ACROSS the line instead of standing in it, toward
        # the touchline the ball is already nearer — the shortest way out of
        # the mouth. Off by default (`intercept_clear` = 0), because standing
        # in the way is the claim being tested and a sweep is a second one.
        if p.intercept_clear > 0 and math.dist(ball, target) <= p.intercept_clear:
            dx, dy = own_goal[0] - ball[0], own_goal[1] - ball[1]
            d = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / d, dx / d                       # across the line
            wide = ball[1] if abs(ball[1]) > 1e-3 else (odom[1] - ball[1])
            side = 1.0 if ny * wide >= 0 else -1.0         # the side the ball is already on
            target = (target[0] + side * p.intercept_clear * nx,
                      target[1] + side * p.intercept_clear * ny)
        self.target = target
        return target


__all__ = ["Interceptor", "block_point", "walk_time"]
