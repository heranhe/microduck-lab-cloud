"""Ball progress and possession: the two continuous soccer metrics.

They live here, not in `eval_pitch`, because they are properties of the
world the /sim page streams as much as of the benchmark - the page shows
them live, and a number on the page has to be the same number the battery
reports. See the class docstring for why these two and not the obvious
alternatives, and `eval_pitch`'s module docstring for what each costs in
seeds (goals 146, ballAdvance 43, possession 9, for a 25% shift).
"""

from __future__ import annotations

import math

from .. import contract as C
from .arena import World

# --- the continuous metrics ---------------------------------------------------
# Goals are a terrible ruler for this benchmark: ~2.5 a run against ~13 kicks,
# so the same brain measured twice gave 3 falls and 6, and four brain variants
# landed inside each other's noise on goals in one day. What the ducks actually
# do thousands of times a run is *reach the ball* and *move it*; these two
# accumulate that at the control tick.
#
# POSSESSION_R — a duck is "on the ball" inside this radius (trunk centre to
# ball centre, in the plane). The kick sweet spot is 6-10 cm ahead and 4-8 cm
# to the side of the trunk (`ChaseParams.kick_ahead/kick_side`), i.e. ~0.12 m
# out; 0.25 m is about twice that, so a duck lined up on the ball or one step
# short of it counts, while it stays well inside `duck_keepout` (0.40 m) — the
# range at which the chase brain starts avoiding the other duck — so the two
# contesting ducks are rarely both on the ball. Only the NEAREST duck can hold
# possession anyway, so the radius sets how much loitering counts, not who wins
# a contest.
POSSESSION_R = 0.25
# The same clock at the keepout radius, recorded only as a robustness check:
# it says whether a conclusion turns on the exact choice of POSSESSION_R.
#
# It did, and not in the direction that was expected: over 16 seeds an arm the
# WIDER radius came out both quieter (CV 0.11 vs 0.16) and far more sensitive
# to the lens contrast (p=0.0001 vs p=0.06). The primary stays 0.25 m anyway.
# 0.25 m was fixed before the battery and 0.40 m was not, and a radius chosen
# because it won on the one battery that could confirm it is exactly the kind
# of result this benchmark has already had to withdraw twice; 0.40 m also sits
# at the chase brain's own `duck_keepout` and is occupied 45 s in every 60 in
# 1v1, which is close enough to saturation to be measuring the pitch rather
# than the brain. Promote it when a battery run FOR that question says so.
POSSESSION_WIDE_R = 0.40
# After the last tick a team was on the ball it keeps CREDIT for the ball's
# motion for this long. A kick is exactly the case that matters: the ball
# leaves at speed and travels most of its distance with no duck within
# POSSESSION_R, so progress credited only while inside the radius would score
# dribbling and ignore the shot — the one thing the benchmark is about. 2.0 s
# is deliberately shorter than the World's own kick→goal attribution window
# (KICK_GOAL_S = 4.0 s): long enough for a kicked ball to run out, short
# enough that a ball rattling round the boards minutes later is nobody's.
CARRY_S = 2.0

# The per-team metric keys a row carries. A row written before these existed
# has none of them; `load_done` fills them with None rather than 0.0, and the
# summary reports how many seeds it could use (see `_mean_field`).
METRIC_FIELDS = ("ballProgress", "ballAdvance", "possession", "possessionWide")

# --- the goal ledger (roadmap Track 4.1.1) ------------------------------------
# `World.goals` is keyed by MOUTH, so a ball the lavender team puts into its own net
# and one the cream team scores are the same row - which is why the first
# measured fact about this pitch (8 of 8 goals in a 4-seed 2v2 battery were own
# goals) needed a scratchpad probe to see at all. These fields are per TEAM.
#
# `goalsFor` / `goalsAgainst` need no attribution and are exact: the mouth a
# team attacks and the mouth it defends are known from its spawn heading
# (`World.goal_for`), so every goal is one of each. `ownGoals` is the subset of
# `goalsAgainst` this team put in ITSELF, and that one needs credit:
#
#   * a kick inside `KICK_GOAL_S` of the ball crossing - the World records
#     which duck took it (`goal_credit_duck`), decided on exactly the test its
#     own kicked/walked-in split uses, so the two can never disagree;
#   * failing that, the last team strictly on the ball (`POSSESSION_R`) within
#     `GOAL_CREDIT_S` - which is what a walked-in goal is, and 13 of the first
#     14 goals measured here were walked in;
#   * failing that the goal is nobody's, and `goalsUnattributed` counts it
#     rather than a mean quietly absorbing a guess.
GOAL_FIELDS = ("goalsFor", "goalsAgainst", "ownGoals", "kickCount", "kicksBack",
               "kickLineCount", "kicksBackLine", "kickCarry")
# How long after a touch a goal is still that team's. The same 4 s the World
# allows a kick: "whoever last kicked or held this ball put it in".
GOAL_CREDIT_S = 4.0

# --- the kick's LINE (roadmap 12au) -------------------------------------------
# `kicksBack` above is `advance < 0` over CARRY_S, and 12at measured what that
# actually counts in the gym: among touches struck FORWARD (the 0.5 s line
# within 45 deg of the attacked mouth) it runs 31-60% below 1 m of travel and
# 0-5% above it. It is a WEAK-TOUCH measure — the short touch the duck walks
# back into inside the 2 s window — and is therefore anti-correlated with how
# well the kick connected. So the ledger could not read the back-kick lean
# 12aq reported: the arm that HALVES the backward lines raises `kicksBack`
# (20.6 -> 23.5%) while whiff falls.
#
# `kicksBackLine` is the direction measure beside it: the line the ball
# actually LEFT on, more than 90 deg from the mouth this team attacks. Same
# rule, same window and the same minimum travel as `scripts/kick_gym.py`'s
# `exit_play` (its EXIT_S / EXIT_MIN_M, named in `tests/test_metrics_kickline.py`
# so a drift in either is one honest failure), so the gym's backward-line
# share and this column are the same quantity on two populations.
EXIT_S = 0.5
# Why 0.5 s and not the carry window: at ~1.4 m/s off the foot and 0.3 m/s^2 of
# rolling resistance the ball has run ~0.66 m by then and has usually not
# reached a board, so the direction is the kick's and not the wall's.
EXIT_MIN_M = 0.05
# Under this the direction is numerical noise: a ball nudged 2 cm has an angle
# but not a line. Such a kick is counted in `kickCount` and NOT in
# `kickLineCount`, which is why the two denominators are both in the row — a
# rate whose denominator is not the population its numerator came from is
# about the filter (AGENTS.md, "a rate's denominator").

# --- the per-kick LIST (roadmap E.2, follow-up 2) -----------------------------
# Everything above is a per-team SUM, and a sum cannot answer a per-swing
# question. E.2 registered a gym win of +0.28 m a swing and had to read it on
# the ledger through `kickCarry`, a run total with a 35% MDE over 48 seeds —
# while the quantity the claim was about, carry PER KICK, was sitting in ~300
# events the ledger threw away as it accumulated them. So every resolved kick
# is also kept, in the order it settled:
KICK_EVENT = ("t", "carry", "adv", "line", "back", "foot")
#   t      when the swing STARTED (`skill_t0`), seconds of sim time.
#   carry  the signed metres toward the attacked mouth over CARRY_S — exactly
#          the quantity summed into `kickCarry`, so the list adds up to the
#          column (to rounding) and a consumer can never be measuring a
#          different thing from the total beside it. `carry < 0` is the
#          `kicksBack` event.
#   adv    the same displacement over the first EXIT_S — how far the ball ran
#          BEFORE the duck could walk it back — or None when no exit sample
#          was taken (the kick was settled inside the window by a goal or a
#          ball-out). None and not 0.0: "not measured" is not "it went
#          nowhere" (AGENTS.md).
#   line   1 if this kick had a readable line (it is in `kickLineCount`).
#   back   1 if that line was backward (it is in `kicksBackLine`). A kick with
#          `line` 0 always has `back` 0 — a whiff has no direction.
#   foot   "L"/"R" for the two shipped kicks, "" when the swing did not come
#          from one of them (a kick a test appends by hand). 12as's "the right
#          foot alone" reading needs this column and the ledger had no way to
#          say it.
#
# It is NOT part of `row()`: the /sim page calls that every frame, and a list
# that grows for as long as the lab is up does not belong in a 50 Hz payload.
# `events_row()` is the battery's door to it (`eval_pitch.run_one`).
EVENT_FIELDS = ("kickEvents",)


def _foot(skill: str | None) -> str:
    """Which foot a kick skill is, in one character. Anything that is not one
    of the two shipped kicks has no foot rather than a guessed one."""
    return {"kick_left": "L", "kick_right": "R"}.get(skill or "", "")


# --- shape (roadmap Track 4.1.3) ----------------------------------------------
# What "they all pile onto the ball" and "somebody stayed back" are as numbers.
# Goals cannot say either, and the README's crowding figures ("two teammates
# inside 0.5 m of the ball 24.5% of the time") came from a one-off trace; these
# accumulate at the control tick like everything else here, so a roster change
# is judged on them the same way a brain change is judged on possession.
SHAPE_FIELDS = ("ballOwnHalf", "spread", "crowd", "depth")
CROWD_R = 0.5              # two teammates this near the ball at once: a pile-up


class PitchMetrics:
    """Ball progress and possession, accumulated at the 50 Hz control tick.

    `tick()` is called once per control step, AFTER `World.step()`, and costs
    one distance per duck plus a few floats — nothing measurable against an
    `mj_step` of a 2-duck pitch.

    **Ball progress** is the ball's displacement toward the goal a team
    attacks (the pitch's long axis is x; `World.goal_for` gives the sign),
    summed over the ticks that team is in control. Why not the other two
    candidates:

    - *Signed per-step displacement over the whole run* telescopes. Sum every
      Δx and all that survives is the ball's net start-to-end position, and
      since every goal recentres the ball, that quantity is ~1.25 m × (goals
      for − goals against) plus a remainder — goals in disguise, with goals'
      variance. It cannot carry more signal than the thing it collapses to.
    - *Positive-only displacement, uncredited* measures how much the ball
      moved at all, which both teams get paid for equally; it is a ball-motion
      counter, not a per-side metric.

    Attributing each tick's displacement to the team in control keeps the
    telescoping LOCAL — inside one possession, where net start-to-end is
    exactly what "this team took the ball 40 cm toward the goal" means — and
    a run contains hundreds of possessions instead of two or three goals.
    `progress` is the signed sum (a team that shoves the ball back toward its
    own goal is charged for it); `advance` sums only the forward part, and is
    recorded alongside so the two can be compared on the same battery. They
    were, over 32 runs, and `advance` won on both counts: the signed sum has a
    CV of 4.2 as a run total, because in self-play the two sides' signs oppose
    and the run total is a difference of two similar numbers, while `advance`
    has a CV of 0.40 and is the only metric here whose correlation with goals
    is resolved away from zero (r 0.50, 95% CI [+0.19, +0.72]). Keep both:
    the signed one is the meaningful per-TEAM reading in an asymmetric
    matchup, where the cancellation that kills its run total is the point.

    **Possession** is the strict clock: seconds where the nearest duck to the
    ball is within POSSESSION_R and belongs to that team. No carry — the task
    it measures is "was one of mine on the ball", which is either true or not.
    It is by far the quietest number the benchmark produces (CV 0.16, against
    goals' 0.76) and therefore the cheapest way to tell that two variants
    differ AT ALL; it is not a measure of playing well, and the battery could
    not resolve its correlation with goals away from zero. Screen with it,
    judge with `advance`.
    """

    def __init__(self, world: World, team_of: dict[str, str]):
        self.w = world
        self.team_of = dict(team_of)
        self.teams = sorted(set(team_of.values()))
        # Which way is "forward" for each team, from the goal it attacks.
        self.sign: dict[str, float] = {}
        for did, tm in team_of.items():
            g = world.goal_for(world.ducks[did])
            s = 1.0 if g is None or g[0] >= 0 else -1.0
            if self.sign.setdefault(tm, s) != s:
                raise ValueError(f"team {tm!r} has ducks attacking opposite goals")
        # The mouth each team attacks, in the World's own keys: a ball crossing
        # at +x is recorded under "right" (`World._check_goal`) and the team
        # whose sign is +1 attacks it. The other mouth is the one it defends,
        # which is where its own goals land.
        self.attacks = {t: ("right" if sg > 0 else "left") for t, sg in self.sign.items()}
        self.defends = {t: ("left" if sg > 0 else "right") for t, sg in self.sign.items()}
        self.ducks_of: dict[str, list[str]] = {t: [] for t in self.teams}
        for did, tm in sorted(self.team_of.items()):
            self.ducks_of[tm].append(did)
        self.half_x = world.scenario.floor[0] / 2 - 0.25      # the goal lines (arena._check_goal)
        self.progress = {t: 0.0 for t in self.teams}
        self.advance = {t: 0.0 for t in self.teams}
        self.possession = {t: 0.0 for t in self.teams}
        self.possession_wide = {t: 0.0 for t in self.teams}
        # The goal ledger (GOAL_FIELDS above).
        self.goals_for = {t: 0 for t in self.teams}
        self.goals_against = {t: 0 for t in self.teams}
        self.own_goals = {t: 0 for t in self.teams}
        self.goals_unattributed = 0
        # Kicks, judged on where the BALL went and not on where the brain
        # meant it to go (the playbook's rule 6: a plan is not an outcome).
        # A kick is pending until CARRY_S has run or a goal ends it.
        self.kick_count = {t: 0 for t in self.teams}
        self.kicks_back = {t: 0 for t in self.teams}
        self.kick_carry = {t: 0.0 for t in self.teams}
        # …and the DIRECTION the ball left on (EXIT_S above): how many kicks
        # had a readable line at all, and how many of those left backward.
        self.kick_lines = {t: 0 for t in self.teams}
        self.kicks_back_line = {t: 0 for t in self.teams}
        # …and the same kicks ONE BY ONE (KICK_EVENT above), which is what
        # makes a per-swing claim readable on the pitch at all.
        self.kick_events: dict[str, list[list]] = {t: [] for t in self.teams}
        # A pending kick is (team, t0, ball at t0, ball at t0 + EXIT_S or None,
        # foot). The fourth field is filled in by `_sample_exits` when the
        # window runs out and stays None when the kick is settled before then
        # (a goal or a ball-out teleports the ball, and a teleport has no
        # direction). SHORTER entries are still accepted everywhere here: a
        # test that hands this list a kick by hand writes the three-field shape
        # (`tests/test_ball_out.py`), and they simply have no line and no foot.
        self._pending: list[tuple] = []
        # A kick is read off the DUCK that is taking it (`skill_t0`), not off
        # the World's `last_kick_t`: the World keeps one last-kick stamp, so
        # two ducks kicking on the same control step would silently collapse
        # into one — the "knob that changes nothing" failure with a different
        # hat on. A kick already in the air when this object is built belongs
        # to whatever came before it and is not counted.
        self._kick_t0 = {did: self._kick_start(d) for did, d in world.ducks.items()}
        # Shape (SHAPE_FIELDS above).
        self.own_half = {t: 0.0 for t in self.teams}
        self._spread = {t: 0.0 for t in self.teams}
        self._crowd = {t: 0 for t in self.teams}
        self._depth = {t: 0.0 for t in self.teams}
        self.ticks = 0
        self._prev = world.ball_xy()
        self._holder: str | None = None       # team credited with the ball right now
        self._holder_t = -1e9                 # when it was last strictly on the ball
        self._goal_seq = world.goal_seq
        # …and the ball-out counter, for the same reason (see `tick`).
        self._ball_outs = world.ball_outs

    def positions(self) -> dict[str, tuple[float, float]]:
        """Every duck's trunk in the plane, read once a tick (the shape
        metrics, the possession test and `nearest` all want it)."""
        out = {}
        for did, d in self.w.ducks.items():
            p = d.trunk_pos(self.w.data)
            out[did] = (float(p[0]), float(p[1]))
        return out

    def nearest(self, pos: dict[str, tuple[float, float]] | None = None) -> tuple[str | None, float]:
        """(duck id, planar distance) of the duck closest to the ball."""
        ball = self.w.ball_xy()
        if ball is None:
            return None, math.inf
        pos = self.positions() if pos is None else pos
        best, best_d = None, math.inf
        for did, (x, y) in pos.items():
            if self.w.ducks[did].down_until > self.w.t:
                # Lying where it fell (`World.getup_s`), on a zero command,
                # for as long as a get-up would cost. It is not "on the
                # ball" in any sense the possession clock means, and while
                # it lay there it was also the `_holder` credited with the
                # ball's motion — a duck flat on the floor beside the ball
                # was being paid for whatever the other side did to it.
                continue
            r = math.hypot(x - ball[0], y - ball[1])
            if r < best_d:
                best, best_d = did, r
        return best, best_d

    # -- the goal ledger and the kick ledger ---------------------------------
    def _credit(self) -> str | None:
        """Whose goal the one that just crossed is: the team of the duck the
        World says kicked it in, else the last team on the ball if that touch
        is inside GOAL_CREDIT_S, else nobody's."""
        w = self.w
        who = w.goal_credit_duck
        if who is not None and who in self.team_of:
            return self.team_of[who]
        if self._holder is not None and w.t - self._holder_t <= GOAL_CREDIT_S:
            return self._holder
        return None

    def _score_goal(self) -> None:
        """A goal has just been counted by the World (`goal_seq` moved). Its
        MOUTH is `World.last_goal`; every team gets a for or an against off
        that alone, and the credited team gets an own goal if the mouth is the
        one it defends."""
        mouth = self.w.last_goal
        if mouth is None:
            return
        for t in self.teams:
            if self.attacks[t] == mouth:
                self.goals_for[t] += 1
            else:
                self.goals_against[t] += 1
        by = self._credit()
        if by is None:
            self.goals_unattributed += 1
        elif self.defends[by] == mouth:
            self.own_goals[by] += 1

    @staticmethod
    def _kick_start(d) -> float | None:
        """When this duck's current kick window began, or None if it is not
        kicking (`World.start_skill` / `_skill_cmd` own both fields)."""
        return d.skill_t0 if (d.skill or "").startswith("kick") else None

    def _note_kick(self, ball: tuple[float, float]) -> None:
        """Any duck that has just STARTED a kick: remember where the ball was,
        so the swing can be judged on where it ends up."""
        for did, d in self.w.ducks.items():
            t0 = self._kick_start(d)
            if t0 is not None and t0 != self._kick_t0.get(did) and did in self.team_of:
                self._pending.append((self.team_of[did], self.w.t, ball, None, _foot(d.skill)))
            self._kick_t0[did] = t0

    def _sample_exits(self, ball: tuple[float, float]) -> None:
        """Where the ball is EXIT_S after each pending kick — the one sample
        the direction column needs, taken once per kick on the first tick past
        the window and never again. Called only on a tick the ball moved on
        its own: on the tick a goal or a ball-out teleports it every pending
        kick has already been settled, so nothing here can read a jump."""
        if not self._pending:
            return
        out = []
        for p in self._pending:
            xy_exit = p[3] if len(p) > 3 else None
            if xy_exit is None and self.w.t - p[1] >= EXIT_S:
                xy_exit = ball
            out.append((p[0], p[1], p[2], xy_exit, p[4] if len(p) > 4 else ""))
        self._pending = out

    def _back_line(self, tm: str, xy0: tuple[float, float],
                   xy_exit: tuple[float, float] | None) -> bool | None:
        """Did this kick LEAVE on a backward line: is the world direction the
        ball travelled over its first EXIT_S more than 90 deg from the mouth
        this team attacks? None when there is no line to read — the sample was
        never taken (the kick was settled inside the window) or the ball moved
        under EXIT_MIN_M, which is a whiff and has no direction.

        The attacked mouth is at +x for a team whose `sign` is +1 and at -x for
        the other, so "more than 90 deg from it" is exactly `kick_gym`'s rule
        (`|world dir| > pi/2` from the mouth, its `exit_play` measured in the
        body frame and the mouth at +x in that gym)."""
        if xy_exit is None:
            return None
        dx, dy = xy_exit[0] - xy0[0], xy_exit[1] - xy0[1]
        if math.hypot(dx, dy) < EXIT_MIN_M:
            return None
        mouth = 0.0 if self.sign[tm] > 0 else math.pi
        ang = math.atan2(dy, dx) - mouth                     # …wrapped into (-pi, pi]
        return abs(math.atan2(math.sin(ang), math.cos(ang))) > math.pi / 2

    def _resolve_kicks(self, ball: tuple[float, float], force: bool = False) -> None:
        """Every kick whose CARRY_S has run (all of them, on a goal — the ball
        is about to be teleported back to the centre spot). What is scored is
        the SAME quantity `ballProgress` scores, the signed displacement along
        the pitch's long axis toward the goal that team attacks, so "the kick
        went backwards" and "the team lost ground" cannot disagree.

        Beside it, and NOT the same question (12at): the LINE the ball left on
        over its first EXIT_S. A kick can leave straight at the mouth and still
        end the window behind where it started — that is the weak touch the
        duck walks back into — so `kicksBack` and `kicksBackLine` are expected
        to disagree, and only the second one is about aim."""
        keep = []
        for p in self._pending:
            tm, t0, xy0 = p[0], p[1], p[2]
            if not force and self.w.t - t0 < CARRY_S:
                keep.append(p)
                continue
            xy_exit = p[3] if len(p) > 3 else None
            d = self.sign[tm] * (ball[0] - xy0[0])
            self.kick_count[tm] += 1
            self.kick_carry[tm] += d
            if d < 0.0:
                self.kicks_back[tm] += 1
            back = self._back_line(tm, xy0, xy_exit)
            if back is not None:
                self.kick_lines[tm] += 1
                if back:
                    self.kicks_back_line[tm] += 1
            # The same event, kept whole (KICK_EVENT). Every column here is
            # read off the numbers the four columns above were just made of,
            # in the same place, so the list and the totals cannot drift.
            self.kick_events[tm].append(
                [round(t0, 3), round(d, 4),
                 None if xy_exit is None else round(self.sign[tm] * (xy_exit[0] - xy0[0]), 4),
                 int(back is not None), int(bool(back)), p[4] if len(p) > 4 else ""])
        self._pending = keep

    def _shape(self, ball: tuple[float, float], pos: dict[str, tuple[float, float]]) -> None:
        """One tick of the shape metrics: where the ball is, how spread out a
        team is, whether two of it are on the ball at once, and how deep its
        deepest duck is."""
        self.ticks += 1
        for tm, ids in self.ducks_of.items():
            if ball[0] * self.sign[tm] < 0:
                self.own_half[tm] += C.CTRL_DT              # the ball is in our half
            near = sum(1 for i in ids if math.dist(pos[i], ball) <= CROWD_R)
            if near >= 2:
                self._crowd[tm] += 1
            if len(ids) > 1:
                pairs = [math.dist(pos[a], pos[b])
                         for k, a in enumerate(ids) for b in ids[k + 1:]]
                self._spread[tm] += sum(pairs) / len(pairs)
            # The deepest duck's distance from the goal line it defends: how
            # far back this team keeps anybody at all.
            line = -self.sign[tm] * self.half_x
            self._depth[tm] += min(abs(pos[i][0] - line) for i in ids)

    def tick(self) -> None:
        w = self.w
        ball = w.ball_xy()
        if ball is None:
            return
        # The displacement first, and it is credited to whoever held the ball
        # THROUGH the step that just ended, not to whoever is standing over it
        # now: the tick a duck arrives to win the ball back is a tick of motion
        # its opponent caused, and charging a couple of centimetres to the
        # wrong side at every changeover — hundreds a run — is a real bias, not
        # a rounding one.
        if w.goal_seq != self._goal_seq:
            # A goal: the World teleported the ball back to the centre spot.
            # That jump is not anybody's progress (counted, it would cancel
            # almost exactly the goal it followed), and the next possession
            # starts clean. The ledger is written FIRST, while `_holder` still
            # says who had the ball and `_prev` still holds the last position
            # before the teleport — which is also the only honest place to
            # settle a kick that is still in the air.
            self._score_goal()
            if self._prev is not None:
                self._resolve_kicks(self._prev, force=True)
            self._goal_seq = w.goal_seq
            self._ball_outs = w.ball_outs
            self._prev, self._holder = ball, None
        elif w.ball_outs != self._ball_outs:
            # A BALL-OUT (`World.ball_out_s`): the referee picked the ball off
            # the boards and placed it back in play. That is the same kind of
            # jump as the goal recentre above and gets the same treatment —
            # up to `ball_out_in` (0.45 m) of it, credited to whoever last
            # touched the ball, would be the referee's progress reported as a
            # team's. MEASURED before this guard existed: a ball parked on a
            # team's own end board with one of its ducks inside POSSESSION_R
            # booked +0.400 m of `progress` AND `advance` on the tick the rule
            # fired. A kick still in the air is settled at `_prev`, the last
            # position the ball reached on its own, for the same reason.
            if self._prev is not None:
                self._resolve_kicks(self._prev, force=True)
            self._ball_outs = w.ball_outs
            self._prev, self._holder = ball, None
        elif self._prev is not None and self._holder is not None and w.t - self._holder_t <= CARRY_S:
            dx = self.sign[self._holder] * (ball[0] - self._prev[0])
            self.progress[self._holder] += dx
            self.advance[self._holder] += max(0.0, dx)
        self._prev = ball
        self._note_kick(ball)
        self._sample_exits(ball)
        self._resolve_kicks(ball)
        # Then who is on the ball at the end of the step, which is who the NEXT
        # step's motion belongs to.
        pos = self.positions()
        who, r = self.nearest(pos)
        if who is not None and r <= POSSESSION_R:
            tm = self.team_of[who]
            self.possession[tm] += C.CTRL_DT
            self._holder, self._holder_t = tm, w.t          # control (and credit) passes
        if who is not None and r <= POSSESSION_WIDE_R:
            self.possession_wide[self.team_of[who]] += C.CTRL_DT
        self._shape(ball, pos)

    def row(self) -> dict:
        """Everything this class accumulated, per team.

        The four continuous metrics and `ballOwnHalf` are RATES per minute of
        play, so a row is comparable across `--seconds` (the totals divide
        out; `simSeconds` is in the row if anyone wants them back). Goals and
        kicks are EVENT COUNTS, deliberately: the playbook's first rule is to
        quote the events, and a count that has been divided by anything can no
        longer be added up across a battery. `kickCarry` is a SUM of metres
        for the same reason — divide it by `kickCount` for the per-kick mean.
        `spread`, `crowd` and `depth` are means over the run's ticks.

        The two back-kick columns have DIFFERENT denominators, and reading one
        out of the other is the error this pair exists to prevent: `kicksBack`
        is out of `kickCount` (every kick has a carry), `kicksBackLine` is out
        of `kickLineCount` (only a kick that moved the ball EXIT_MIN_M in its
        first EXIT_S left on a line at all).

        No side effects: the /sim page calls this every frame. A kick still in
        the air when the run ends therefore never lands in `kickCount`, which
        costs at most the last 2 s of a run and costs it to both teams."""
        per_min = 60.0 / max(self.w.t, 1e-9)
        n = max(self.ticks, 1)
        return {"ballProgress": {t: round(v * per_min, 3) for t, v in self.progress.items()},
                "ballAdvance": {t: round(v * per_min, 3) for t, v in self.advance.items()},
                "possession": {t: round(v * per_min, 3) for t, v in self.possession.items()},
                "possessionWide": {t: round(v * per_min, 3) for t, v in self.possession_wide.items()},
                "goalsFor": dict(self.goals_for),
                "goalsAgainst": dict(self.goals_against),
                "ownGoals": dict(self.own_goals),
                "goalsUnattributed": self.goals_unattributed,
                "kickCount": dict(self.kick_count),
                "kicksBack": dict(self.kicks_back),
                "kickLineCount": dict(self.kick_lines),
                "kicksBackLine": dict(self.kicks_back_line),
                "kickCarry": {t: round(v, 3) for t, v in self.kick_carry.items()},
                "ballOwnHalf": {t: round(v * per_min, 2) for t, v in self.own_half.items()},
                "spread": {t: (round(self._spread[t] / n, 3) if len(self.ducks_of[t]) > 1 else None)
                           for t in self.teams},
                "crowd": {t: (round(self._crowd[t] / n, 4) if len(self.ducks_of[t]) > 1 else None)
                          for t in self.teams},
                "depth": {t: round(self._depth[t] / n, 3) for t in self.teams}}

    def events_row(self) -> dict:
        """The per-kick list (KICK_EVENT), per team, in the order the kicks
        settled — `{"kickEvents": {team: [[t, carry, adv, line, back, foot], ...]}}`.

        Deliberately NOT in `row()`. The /sim page puts `row()` in every
        streamed frame, and this list grows for as long as the lab is up; a
        battery calls it once, at the end of a run (`eval_pitch.run_one`). A
        row written before the list existed has no key at all, which
        `load_done` turns into None and `compare_pitch.py` prints as a dash —
        never as "this arm took no kicks".

        A copy, so a caller that mutates what it gets cannot corrupt the
        column the run's own totals were made from."""
        return {"kickEvents": {t: [list(e) for e in ev] for t, ev in self.kick_events.items()}}


SPIN_FIELDS = ("spinFrac", "steerFrac", "spinYaw", "spinRate")


class SpinMetrics:
    """How much of a run is the robot rotating on the spot?

    A PER-TICK tally, which is the point: it resolves where goals and falls
    cannot. Measured with it, a 1v1 run is 47% in-place turning at the
    walker's 0.655 rad/s ceiling, and `ANG_VEL_Z_RANGE`'s +-1.0 is already
    asking for all of that - the largest measured lever in this repo
    (roadmap 3.7, docs/turn-rate-experiment.md).

    `spinYaw` integrates the yaw the body ACTUALLY swept while a turn was
    commanded, so a faster walker shows up as the same yaw demand in fewer
    ticks rather than as a different number. `spinRate` is the rate it
    managed, which is how you check a new walker is really delivering.

    Lives here rather than in either benchmark because `eval_pitch` and
    `eval_striker` run the same loop and their rows must stay identical -
    a test pins that, and it caught this class being added to only one.
    """

    def __init__(self, world):
        self.w = world
        self.spin = self.steer = self.ticks = 0
        self.yaw = 0.0
        self._prev = {d.id: world.odom(d)[2] for d in world.ducks.values()}

    def tick(self, duck, twist) -> None:
        """Once per DUCK per control step, with the twist the brain commanded."""
        from ..brain.gait import TURN_KICK
        vx, _, wz = twist
        self.ticks += 1
        y = self.w.odom(duck)[2]
        if wz != 0.0 and vx <= TURN_KICK:
            self.spin += 1
            self.yaw += abs(math.atan2(math.sin(y - self._prev[duck.id]),
                                       math.cos(y - self._prev[duck.id])))
        elif wz != 0.0:
            self.steer += 1
        self._prev[duck.id] = y

    def row(self) -> dict:
        n = max(self.ticks, 1)
        return {"spinFrac": round(self.spin / n, 4),
                "steerFrac": round(self.steer / n, 4),
                "spinYaw": round(self.yaw, 1),
                "spinRate": round(self.yaw / max(self.spin * C.CTRL_DT, 1e-9), 3)}
