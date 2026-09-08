"""Supporter positioning by potential field (roadmap Track 4 §6 D.2).

RoboCup supporters stand where a potential field over the pitch has its
minimum — the ball, the teammates, the opponents and the goals each add a
term (Berlin United's Mellmann et al.; B-Human's supporter roles; the MSL
teams tile the pitch by Voronoi instead). Ours stood at fixed POSTS laid
out from the ball (`ChaseParams.strike_ahead`, `mid_side`, `support_back`),
and the 3v3 push-first measurement said what that costs: a post is a
distance from the ball, so when the ball is WALKED up the pitch with a
duck attached the whole team walks with it and compresses around the
carrier (crowd 0.19 -> 0.30, spread 1.51 -> 1.23 m, p<0.001), and the
carrier's own rule (`defender_clears`) did not touch it.

This field asks a different question of a supporter — not "how far from
the ball" but "where is a place the carrier is NOT going, from which the
ball can still be reached":

- `ahead`: along the carrier's LANE (the line from the ball to the goal it
  is taking it to), be this far up-pitch of the ball — the role's number
  (a striker's `strike_ahead`, a midfielder's 0, a plain supporter's
  `support_back` behind);
- the lane itself is a REPULSOR strip ahead of the ball: a supporter in it
  is a duck the carrier walks into, and the ball cannot be passed to a
  place the carrier is already walking to;
- `wide`: a lateral offset from the lane to prefer, either side, which
  with the lane term puts the spot beside the lane, not on it;
- teammates repel (spread), opponents repel where they stand AND where
  they stand between the ball and the spot (an open pass lane);
- the pitch potential of `kickselect` pulls a little toward the goal, so
  a striker's wide spot drifts to the far post, not the corner flag;
- the boards' margin, the role's zone and the attacker's room are hard.

A grid of candidate spots over the pitch, costed in numpy on every call
(a 3v3 pitch is ~900 spots at 0.1 m); a small hysteresis keeps the last
spot unless a new minimum is clearly better, so the servo target does not
hop between cells as the world moves. Pure numbers in; the brain
(`Chase._field_spot`) supplies them. Nothing here changes a duck with the
ball or a defender / keeper, whose posts measured well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .kickselect import Pitch


@dataclass(frozen=True)
class FieldParams:
    grid: float = 0.1                  # m between candidate spots
    ahead_w: float = 1.0               # weight of (along - ahead)^2, per m^2
    lane_w: float = 0.2                # the carrier's lane: half-width (Gaussian sigma) of the repulsor strip
    lane_cost: float = 1.5             # its cost on the centre line
    wide: float = 0.5                  # preferred |lateral| offset from the lane
    wide_w: float = 0.5
    mate_r: float = 0.5                # teammates repel with this sigma
    mate_cost: float = 2.0
    opp_r: float = 0.4                 # opponents: the same, on their distance to the ball->spot segment
    opp_cost: float = 1.5
    goal_pull: float = 0.3             # x the pitch potential (spans about -1..+2)
    me_w: float = 0.15                 # the cost of getting there, per m^2 from where the supporter stands
    hysteresis: float = 0.1            # keep the previous spot unless a new minimum beats it by this


def potential_grid(x: np.ndarray, y: np.ndarray, pitch: Pitch) -> np.ndarray:
    """`kickselect.potential` over arrays: the same field, vectorised."""
    ax = pitch.attack_sign * x / max(pitch.half_x, 1e-6)
    sx_own, sx_opp, sy = 0.75 * pitch.half_x, 0.5 * pitch.half_x, 0.27 * pitch.half_x
    own = np.exp(-0.5 * (((x - pitch.own_x) / sx_own) ** 2 + (y / sy) ** 2))
    opp = np.exp(-0.5 * (((x - pitch.opp_x) / sx_opp) ** 2 + (y / sy) ** 2))
    return ax - own + opp


class Field:
    """The candidate spots of one pitch (built once: `bounds` inside the
    boards, less `margin`) and the cost over them."""

    def __init__(self, bounds: tuple[float, float], margin: float, params: FieldParams | None = None):
        self.p = params or FieldParams()
        hx, hy = max(bounds[0] - margin, 0.0), max(bounds[1] - margin, 0.0)
        # linspace, not arange: arange dropped the +y row whenever 2*hy was not
        # a multiple of the step, so a 2v2 supporter stood 5 cm nearer one board.
        xs = np.linspace(-hx, hx, int(round(2 * hx / self.p.grid)) + 1)
        ys = np.linspace(-hy, hy, int(round(2 * hy / self.p.grid)) + 1)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.x, self.y = gx.ravel(), gy.ravel()
        self._pot: dict[float, np.ndarray] = {}

    def cost(self, ball, lane_u, ahead: float, mates, opps, pitch: Pitch,
             zone: tuple[float, float] | None = None, keep_out: float = 0.0,
             me: tuple[float, float] | None = None) -> np.ndarray:
        """The cost of every spot (inf where a spot is not allowed). `me` is
        where the supporter stands now: the spot on ITS side of the lane
        costs less than the mirror image across the carrier's path."""
        p = self.p
        bx, by = float(ball[0]), float(ball[1])
        ux, uy = float(lane_u[0]), float(lane_u[1])
        dx, dy = self.x - bx, self.y - by
        along = dx * ux + dy * uy
        lat = -dx * uy + dy * ux
        c = p.ahead_w * (along - ahead) ** 2 + p.wide_w * (np.abs(lat) - p.wide) ** 2
        # The lane: ahead of the ball only (behind it the carrier is not coming).
        c = c + p.lane_cost * np.exp(-0.5 * (lat / p.lane_w) ** 2) * (along > -0.05)
        for mx, my in mates or ():
            c = c + p.mate_cost * np.exp(-0.5 * ((self.x - mx) ** 2 + (self.y - my) ** 2) / p.mate_r ** 2)
        for ox, oy in opps or ():
            # Distance from the opponent to the ball->spot segment: on the
            # spot, or between it and the ball, is the same closed lane.
            l2 = np.maximum(dx * dx + dy * dy, 1e-9)
            f = np.clip(((ox - bx) * dx + (oy - by) * dy) / l2, 0.0, 1.0)
            dseg = np.hypot(ox - (bx + f * dx), oy - (by + f * dy))
            c = c + p.opp_cost * np.exp(-0.5 * (dseg / p.opp_r) ** 2)
        if me is not None and p.me_w:
            c = c + p.me_w * ((self.x - me[0]) ** 2 + (self.y - me[1]) ** 2)
        if p.goal_pull:
            key = pitch.attack_sign
            if key not in self._pot:
                self._pot[key] = potential_grid(self.x, self.y, pitch)
            c = c - p.goal_pull * self._pot[key]
        ok = np.ones(len(self.x), bool)
        if keep_out > 0:
            ok &= np.hypot(dx, dy) >= keep_out
        if zone is not None and pitch.half_x > 0:
            a = pitch.attack_sign * self.x / pitch.half_x
            ok &= (a >= zone[0] - 1e-9) & (a <= zone[1] + 1e-9)
        return np.where(ok, c, np.inf)

    def spot(self, ball, lane_u, ahead: float, mates, opps, pitch: Pitch,
             zone: tuple[float, float] | None = None, keep_out: float = 0.0,
             prev: tuple[float, float] | None = None,
             me: tuple[float, float] | None = None) -> tuple[float, float] | None:
        """The best spot, or `prev` when it is still within `hysteresis` of
        the best; None when no spot is allowed (the caller keeps its post)."""
        c = self.cost(ball, lane_u, ahead, mates, opps, pitch, zone, keep_out, me)
        i = int(np.argmin(c))
        if not np.isfinite(c[i]):
            return None
        if prev is not None:
            j = int(np.argmin((self.x - prev[0]) ** 2 + (self.y - prev[1]) ** 2))
            if np.isfinite(c[j]) and c[j] <= c[i] + self.p.hysteresis:
                return prev
        return float(self.x[i]), float(self.y[i])


def lane_unit(ball, goal, fallback_yaw: float) -> tuple[float, float]:
    """The carrier's lane: from the ball toward the goal it attacks; along
    the duck's own attack heading when the ball is on the goal line."""
    gx, gy = goal[0] - ball[0], goal[1] - ball[1]
    n = math.hypot(gx, gy)
    if n < 1e-6:
        return math.cos(fallback_yaw), math.sin(fallback_yaw)
    return gx / n, gy / n
