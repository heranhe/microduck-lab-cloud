"""The soccer benchmark's continuous metrics (`eval_pitch.PitchMetrics`).

Goals are too rare to separate anything in a 300 s run — the same brain
measured twice gave 3 falls and 6 — so the benchmark's real ruler is the ball
progress and possession clock accumulated at the control tick. Those are sums
over 15 000 ticks, which is exactly the shape of thing that can be silently
sign-flipped or double-counted and still look plausible in a battery, so they
are driven here on hand-built states: the ball is put where the test wants it
and the accumulator is ticked directly.
"""

import json
import math

import mujoco
import pytest

from microduck_local import contract as C
from microduck_local.eval_pitch import (
    CARRY_S,
    GOAL_CREDIT_S,
    METRIC_FIELDS,
    POSSESSION_R,
    ROW_FIELDS,
    PitchMetrics,
    _mean_field,
    _print_ledger,
    _seed_line,
    _total,
    load_done,
    run_one,
)
from microduck_local.world import World, make_pitch
from microduck_local.world.metrics import CROWD_R
from microduck_local.world.scenario import PITCH_TEAMS

# The two teams `make_pitch` puts on a pitch, read from the source: the
# PAIR has already changed once (cream v sky read alike at 60 px), and a
# test that hard-codes it fails for a reason that is not about the test.
HOME, AWAY = PITCH_TEAMS
FAR = (1.4, 1.1)          # a corner: no duck is anywhere near the ball


@pytest.fixture(scope="module")
def world():
    """A 1v1 pitch, never stepped: these tests own the state."""
    return World(make_pitch(), seed=0)


@pytest.fixture(scope="module")
def world2():
    """A 2v2 pitch, for the metrics a one-duck team cannot have."""
    return World(make_pitch(per_side=2), seed=0)


def _put(w, ball=None, **duck_xy) -> None:
    if ball is not None:
        q = int(w.model.jnt_qposadr[w._ball_joint])
        w.data.qpos[q:q + 2] = ball
    for did, xy in duck_xy.items():
        a = w.ducks[did].adr
        w.data.qpos[a.root_qpos:a.root_qpos + 2] = xy
    mujoco.mj_forward(w.model, w.data)


def _fresh(w, ball, **duck_xy) -> PitchMetrics:
    """A new run on the shared world: the clock back to zero and nothing left
    over from the last test — a duck still holding a kick window would make
    the next kick at t = 0 look like the same one (the World ends a window
    after KICK_S; a test that sets one has to end it too)."""
    w.t = 0.0
    w.goal_seq = 0
    w.last_goal, w.goal_credit_duck, w.last_kick_duck = None, None, None
    w.last_kick_t = -1e9
    for d in w.ducks.values():
        d.skill, d.skill_t0 = None, 0.0
    _put(w, ball=ball, **duck_xy)
    return PitchMetrics(w, {d.id: d.team for d in w.scenario.ducks})


def _tick(w, m, ball=None, **duck_xy) -> None:
    """One control step of the accumulator, with the world moved first."""
    _put(w, ball=ball, **duck_xy)
    w.t += C.CTRL_DT
    m.tick()


def _goal(w, m, mouth: str, credit: str | None, **duck_xy) -> None:
    """What `World._check_goal` leaves behind for the next tick: the mouth the
    ball crossed, the duck whose kick is inside KICK_GOAL_S (None = walked in),
    the counter moved, and the ball already back on the centre spot."""
    w.last_goal, w.goal_credit_duck = mouth, credit
    w.goal_seq += 1
    _tick(w, m, ball=(0.0, 0.0), **duck_xy)


def _kick(w, m, by: str, ball, **duck_xy) -> None:
    """What `World.start_skill` leaves behind: this duck holds a kick window
    that began now, and the World's own last-kick stamp points at it."""
    d = w.ducks[by]
    d.skill, d.skill_t0 = "kick_left", w.t
    w.last_kick_t, w.last_kick_duck = w.t, by
    _tick(w, m, ball=ball, **duck_xy)


# -- which way is forward -------------------------------------------------------

def test_the_two_sides_attack_opposite_goals(world):
    m = _fresh(world, (0.0, 0.0), d0=FAR, d1=FAR)
    assert m.sign == {HOME: 1.0, AWAY: -1.0}      # make_pitch: left attacks +x


def test_the_same_ball_motion_is_progress_for_one_side_and_a_loss_for_the_other(world):
    """The one bug this metric could carry into a published result: a sign
    read off the pitch instead of off the team. The ball going +x is the cream
    team's progress and the lavender team's giveaway, and vice versa."""
    got = {}
    for holder, xy in (("d0", (0.0, 0.0)), ("d1", (0.0, 0.0))):
        m = _fresh(world, (0.0, 0.0), **{holder: xy, ("d1" if holder == "d0" else "d0"): FAR})
        _tick(world, m, ball=(0.0, 0.0), **{holder: xy})       # take possession
        _tick(world, m, ball=(0.30, 0.0), **{holder: xy})      # ball moves +30 cm in x
        got[holder] = (m.progress, m.advance)
    (cp, ca), (sp, sa) = got["d0"], got["d1"]
    assert cp[HOME] == pytest.approx(0.30) and cp[AWAY] == 0.0
    assert ca[HOME] == pytest.approx(0.30)                  # forward for cream: it advanced
    assert sp[AWAY] == pytest.approx(-0.30) and sp[HOME] == 0.0
    assert sa[AWAY] == 0.0                                    # backwards for the away side: no advance


def test_advance_keeps_only_the_forward_part_where_progress_nets_out(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    for x in (0.0, 0.40, 0.10, 0.25):                          # +0.40, −0.30, +0.15
        _tick(world, m, ball=(x, 0.0), d0=(x, 0.0))
    assert m.progress[HOME] == pytest.approx(0.25)           # net: telescopes inside the possession
    assert m.advance[HOME] == pytest.approx(0.55)            # 0.40 + 0.15


def test_motion_across_y_and_the_ball_sitting_still_move_nothing(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    for y in (0.0, 0.3, -0.3, 0.0):
        _tick(world, m, ball=(0.0, y), d0=(0.0, y))
    assert m.progress[HOME] == pytest.approx(0.0) and m.advance[HOME] == pytest.approx(0.0)
    assert m.possession[HOME] > 0                            # it was on the ball the whole time


# -- possession ------------------------------------------------------------------

def test_possession_is_the_nearest_ducks_team_inside_the_radius_and_nobody_outside_it(world):
    m = _fresh(world, (0.0, 0.0), d0=FAR, d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(POSSESSION_R * 0.5, 0.0), d1=FAR)      # d0 on the ball
    _tick(world, m, ball=(0.0, 0.0), d0=FAR, d1=(0.0, POSSESSION_R * 0.5))      # d1 on the ball
    _tick(world, m, ball=(0.0, 0.0), d0=FAR, d1=FAR)                            # free ball
    assert m.possession[HOME] == pytest.approx(C.CTRL_DT)
    assert m.possession[AWAY] == pytest.approx(C.CTRL_DT)
    assert m.possession_wide[HOME] == pytest.approx(C.CTRL_DT)                # the wider clock agrees here


def test_only_the_nearer_duck_holds_the_ball_when_both_are_inside_the_radius(world):
    m = _fresh(world, (0.0, 0.0), d0=FAR, d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.20, 0.0), d1=(0.05, 0.0))
    assert m.possession[AWAY] == pytest.approx(C.CTRL_DT) and m.possession[HOME] == 0.0
    assert m.nearest()[0] == "d1"


def test_the_wide_clock_sees_a_duck_the_primary_radius_does_not(world):
    m = _fresh(world, (0.0, 0.0), d0=FAR, d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.33, 0.0), d1=FAR)   # between 0.25 and 0.40
    assert m.possession[HOME] == 0.0 and m.possession_wide[HOME] == pytest.approx(C.CTRL_DT)


# -- credit for the ball nobody is touching --------------------------------------

def test_a_kicked_ball_running_free_is_credited_to_whoever_last_had_it(world):
    """Most of a kick's distance happens with no duck inside POSSESSION_R. A
    progress metric that only counted contact ticks would score dribbling and
    ignore the shot, which is the thing the benchmark is about."""
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.0, 0.0))            # d0 takes possession
    _tick(world, m, ball=(0.5, 0.0), d0=FAR)                   # ball away, nobody near
    _tick(world, m, ball=(0.9, 0.0), d0=FAR)
    assert m.progress[HOME] == pytest.approx(0.9)
    assert m.possession[HOME] == pytest.approx(C.CTRL_DT)     # the clock, though, stops at contact


def test_credit_lapses_after_carry_s_so_a_ball_rattling_round_the_boards_is_nobodys(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.0, 0.0))
    world.t += CARRY_S + C.CTRL_DT                              # long after the last touch
    _tick(world, m, ball=(0.9, 0.0), d0=FAR)
    assert m.progress[HOME] == 0.0 and m.advance[HOME] == 0.0


def test_the_other_team_taking_the_ball_takes_the_credit_with_it(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.0, 0.0))
    _tick(world, m, ball=(0.1, 0.0), d0=FAR, d1=(0.1, 0.0))     # d1 wins it: +0.1 still d0's
    _tick(world, m, ball=(0.4, 0.0), d0=FAR, d1=FAR)            # now d1's, and it is going the wrong way
    assert m.progress[HOME] == pytest.approx(0.1)
    assert m.progress[AWAY] == pytest.approx(-0.3)


# -- the goal restart --------------------------------------------------------------

def test_the_ball_teleporting_back_to_the_centre_spot_after_a_goal_is_nobodys_progress(world):
    """`World.kickoff` puts the ball back on the spot. Counted as motion it
    would cancel the goal it followed almost exactly — 1.25 m the wrong way."""
    m = _fresh(world, (1.0, 0.0), d0=(1.0, 0.0), d1=FAR)
    _tick(world, m, ball=(1.0, 0.0), d0=(1.0, 0.0))
    before = dict(m.progress)
    world.goal_seq += 1                                          # a goal, and the World recentres
    _tick(world, m, ball=(0.0, 0.0), d0=FAR)
    assert m.progress == before
    _tick(world, m, ball=(0.6, 0.0), d0=FAR)                     # and the next possession starts clean
    assert m.progress == before


# -- the goal ledger: for, against, and who put it in (roadmap Track 4.1.1) ---------

def test_a_goal_is_for_one_team_and_against_the_other_by_the_mouth_alone(world):
    """No attribution needed for these two, which is the point of splitting them
    out: the mouth a team attacks is known from its spawn heading, so every goal
    is exactly one `for` and one `against` however murky the credit is."""
    m = _fresh(world, (1.6, 0.0), d0=FAR, d1=FAR)
    _goal(world, m, "right", None, d0=FAR, d1=FAR)               # cream attacks the +x mouth
    assert m.goals_for == {HOME: 1, AWAY: 0}
    assert m.goals_against == {HOME: 0, AWAY: 1}
    _goal(world, m, "left", None, d0=FAR, d1=FAR)
    assert m.goals_for == {HOME: 1, AWAY: 1}
    assert m.goals_against == {HOME: 1, AWAY: 1}


def test_an_own_goal_is_the_kicker_scoring_on_the_mouth_it_defends(world):
    """The World says which duck kicked it in (`goal_credit_duck`, decided on
    exactly the test its own kicked/walked-in split uses). d1 is the right
    team, and the +x mouth is the one the right team defends."""
    m = _fresh(world, (1.6, 0.0), d0=FAR, d1=(1.5, 0.0))
    _goal(world, m, "right", "d1", d0=FAR, d1=(1.5, 0.0))
    assert m.own_goals == {HOME: 0, AWAY: 1} and m.goals_unattributed == 0
    assert m.goals_for[HOME] == 1                              # …and the left team is still credited a goal for
    # The same kicker at the other end is a goal, not an own goal.
    _goal(world, m, "left", "d1", d0=FAR, d1=FAR)
    assert m.own_goals == {HOME: 0, AWAY: 1}


def test_a_walked_in_goal_is_charged_to_the_last_team_on_the_ball(world):
    """13 of the first 14 goals measured on this pitch were walked in, so the
    possession fallback is not a corner case — it is the common one."""
    m = _fresh(world, (1.4, 0.0), d0=FAR, d1=(1.45, 0.0))
    _tick(world, m, ball=(1.5, 0.0), d0=FAR, d1=(1.55, 0.0))     # d1 is on it, inside POSSESSION_R
    assert m._holder == AWAY
    _goal(world, m, "right", None, d0=FAR, d1=FAR)               # no kick: the World says nobody kicked
    assert m.own_goals == {HOME: 0, AWAY: 1} and m.goals_unattributed == 0


def test_a_goal_nobody_has_touched_for_seconds_is_nobodys(world):
    """The alternative is a guess, and a guess in a ledger is worse than a
    gap: `goalsUnattributed` is reported so a battery can say how much of the
    column it could not place."""
    m = _fresh(world, (1.4, 0.0), d0=FAR, d1=(1.45, 0.0))
    _tick(world, m, ball=(1.5, 0.0), d0=FAR, d1=(1.55, 0.0))     # d1 had it…
    world.t += GOAL_CREDIT_S + 0.1                               # …a long time ago
    _goal(world, m, "right", None, d0=FAR, d1=FAR)
    assert m.own_goals == {HOME: 0, AWAY: 0} and m.goals_unattributed == 1
    assert m.goals_against[AWAY] == 1                         # the against column is unaffected


# -- kicks, judged on where the ball went (roadmap Track 4.1.2) ---------------------

def test_a_kick_is_scored_on_where_the_ball_ENDS_UP_not_where_it_was_aimed(world):
    """The chase brain's plan says where it MEANT the ball to go; only the ball
    says where it went (the playbook's rule 6). The quantity is the same signed
    displacement `ballProgress` uses, so "the kick went backwards" and "the team
    lost ground" can never disagree."""
    m = _fresh(world, (0.0, 0.0), d0=(-0.1, 0.0), d1=FAR)
    _kick(world, m, "d0", (0.0, 0.0), d0=(-0.1, 0.0), d1=FAR)
    assert m.kick_count == {HOME: 0, AWAY: 0}                # still in the air
    _tick(world, m, ball=(0.6, 0.0), d0=FAR, d1=FAR)
    assert m.kick_count[HOME] == 0
    world.t += CARRY_S
    _tick(world, m, ball=(0.6, 0.0), d0=FAR, d1=FAR)
    assert m.kick_count == {HOME: 1, AWAY: 0}
    assert m.kicks_back == {HOME: 0, AWAY: 0}
    assert m.kick_carry[HOME] == pytest.approx(0.6)             # left attacks +x


def test_a_kick_that_sends_the_ball_toward_its_own_goal_is_a_back_kick(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.1, 0.0), d1=FAR)
    _kick(world, m, "d0", (0.0, 0.0), d0=(0.1, 0.0), d1=FAR)
    world.t += CARRY_S
    _tick(world, m, ball=(-0.5, 0.0), d0=FAR, d1=FAR)
    assert m.kicks_back == {HOME: 1, AWAY: 0}
    assert m.kick_carry[HOME] == pytest.approx(-0.5)
    # The identical ball motion is a FORWARD kick for the other side.
    m2 = _fresh(world, (0.0, 0.0), d0=FAR, d1=(0.1, 0.0))
    _kick(world, m2, "d1", (0.0, 0.0), d0=FAR, d1=(0.1, 0.0))
    world.t += CARRY_S
    _tick(world, m2, ball=(-0.5, 0.0), d0=FAR, d1=FAR)
    assert m2.kicks_back == {HOME: 0, AWAY: 0}
    assert m2.kick_carry[AWAY] == pytest.approx(0.5)


def test_a_kick_still_in_the_air_when_a_goal_lands_is_settled_at_the_goal_line(world):
    """The ball is about to teleport to the centre spot; settling the kick
    against THAT would score a goal as a 1.5 m back-kick."""
    m = _fresh(world, (0.9, 0.0), d0=(0.8, 0.0), d1=FAR)
    _kick(world, m, "d0", (0.9, 0.0), d0=(0.8, 0.0), d1=FAR)
    _tick(world, m, ball=(1.65, 0.0), d0=FAR, d1=FAR)             # rolling toward the mouth
    _goal(world, m, "right", "d0", d0=FAR, d1=FAR)
    assert m.kick_count == {HOME: 1, AWAY: 0}
    assert m.kicks_back == {HOME: 0, AWAY: 0}
    assert m.kick_carry[HOME] == pytest.approx(0.75)            # 0.9 → 1.65, not 0.9 → 0.0


# -- shape: the pile-up and the deepest duck (roadmap Track 4.1.3) ------------------

def test_shape_measures_the_pile_up_the_spread_and_how_far_back_anyone_stays(world2):
    """`crowd` is the fraction of ticks two teammates are within CROWD_R of the
    ball — what "they all group up" means as a number. `depth` is the deepest
    teammate's distance from the goal line it defends."""
    m = PitchMetrics(world2, {d.id: d.team for d in world2.scenario.ducks})
    world2.t = 0.0
    world2.goal_seq = 0
    hx = m.half_x
    far = {"d2": FAR, "d3": FAR}
    # Both left ducks on the ball at the centre: a pile-up, and nobody back.
    _tick(world2, m, ball=(0.0, 0.0), d0=(0.1, 0.0), d1=(0.0, 0.2), **far)
    assert max(math.dist((0.1, 0.0), (0.0, 0.0)), math.dist((0.0, 0.2), (0.0, 0.0))) < CROWD_R
    assert m._crowd[HOME] == 1 and m._crowd[AWAY] == 0
    assert m._spread[HOME] == pytest.approx(math.dist((0.1, 0.0), (0.0, 0.2)))
    assert m._depth[HOME] == pytest.approx(hx)                  # both at x=0, the line is at -hx
    # Spread out, one of them deep: no pile-up, and depth is the DEEPEST one.
    _tick(world2, m, ball=(0.0, 0.0), d0=(0.1, 0.0), d1=(-1.2, 0.0), **far)
    assert m._crowd[HOME] == 1
    assert m._depth[HOME] == pytest.approx(hx + (hx - 1.2))     # tick 1 at x=0, tick 2 at x=-1.2
    assert m.row()["crowd"][HOME] == pytest.approx(0.5)         # one tick of two
    assert m.row()["spread"][AWAY] == pytest.approx(0.0)       # both parked in the same corner


def test_a_one_duck_team_has_no_spread_or_crowd_and_says_so(world):
    """None, not 0.0: a 1v1 team cannot pile up, and a zero would average into
    a 2v2 battery as if it had been measured."""
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.0, 0.0))
    row = m.row()
    assert row["spread"] == {HOME: None, AWAY: None}
    assert row["crowd"] == {HOME: None, AWAY: None}
    assert row["depth"][HOME] is not None                       # …but depth is one duck's own


def test_the_ball_sits_in_exactly_one_teams_own_half(world):
    """Which is why the summary prints `ballOwnHalf` per team and never
    averaged: over the pair it is 60 s/min whatever the brains do."""
    m = _fresh(world, (0.0, 0.0), d0=FAR, d1=FAR)
    _tick(world, m, ball=(-0.5, 0.0), d0=FAR, d1=FAR)             # the left team's half
    _tick(world, m, ball=(0.5, 0.0), d0=FAR, d1=FAR)
    assert m.own_half[HOME] == pytest.approx(C.CTRL_DT)
    assert m.own_half[AWAY] == pytest.approx(C.CTRL_DT)


# -- what goes in the row ------------------------------------------------------------

def test_the_row_is_per_minute_of_play_and_carries_every_metric(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.0, 0.0))
    _tick(world, m, ball=(0.5, 0.0), d0=(0.5, 0.0))
    world.t = 30.0                                               # half a minute of play
    row = m.row()
    assert set(row) == set(ROW_FIELDS) | {"goalsUnattributed"}
    assert row["ballProgress"][HOME] == pytest.approx(1.0, abs=1e-3)      # 0.5 m in 30 s
    assert row["possession"][HOME] == pytest.approx(2 * C.CTRL_DT * 2, abs=1e-3)
    assert _total(row, "ballProgress") == pytest.approx(1.0, abs=1e-3)


def test_run_one_still_returns_every_field_other_tooling_reads():
    r = run_one(0, 1.0)
    for k in ("seed", "perSide", "left", "right", "kickGoals", "bumpGoals",
              "kicks", "pushes", "falls", "simSeconds", "seconds", *METRIC_FIELDS):
        assert k in r, k
    for f in METRIC_FIELDS:
        assert set(r[f]) == {HOME, AWAY}


# -- resuming a file written before these metrics existed ------------------------------

def _row(seed, tag="", per_side=1, seconds=300.0, **kw):
    return {"seed": seed, "tag": tag, "perSide": per_side, "seconds": seconds,
            "left": 1, "right": 0, "kickGoals": 1, "bumpGoals": 0,
            "kicks": {"d0": 5, "d1": 4}, "pushes": {"d0": 1, "d1": 0},
            "falls": {"d0": 0, "d1": 1}, "simSeconds": 300.0, **kw}


def test_an_old_row_resumes_with_the_new_metrics_missing_not_zero(tmp_path):
    """The alternative — refusing the file — would re-run an hour of seeds to
    recover metrics nobody measured then; the other alternative — filling 0.0 —
    would drag every mean toward zero and never say so."""
    f = tmp_path / "old.jsonl"
    f.write_text(json.dumps(_row(0, "shipped")) + "\n")
    done = load_done(str(f), "shipped", 1, 300.0)
    assert set(done) == {0}
    assert done[0]["kicks"] == {"d0": 5, "d1": 4}                 # what it did measure survives
    for field in METRIC_FIELDS:
        assert done[0][field] is None and _total(done[0], field) is None
    assert "progress —" in _seed_line(done[0])                    # and it prints as unmeasured
    assert "back —, back-line —" in _seed_line(done[0])           # both shares, both unmeasured


def test_the_seed_line_reads_each_back_share_out_of_its_own_denominator(tmp_path):
    """`kicksBack` is out of every kick; `kicksBackLine` is out of the kicks
    that moved the ball far enough to HAVE a line, which is a SMALLER number.
    Printing the second over the first's denominator is exactly the confusion
    the second column was added to prevent (12at), so the line carries both
    ratios whole — and a `—` for a row written before the column existed,
    rather than a 0 that would read as "no backward lines"."""
    r = _row(0, kickCount={"left": 5, "right": 2}, kicksBack={"left": 1, "right": 0},
             kickLineCount={"left": 4, "right": 1}, kicksBackLine={"left": 2, "right": 0})
    assert "kicks 9 (back 1/7, back-line 2/5)" in _seed_line(r)

    f = tmp_path / "before-the-column.jsonl"
    f.write_text(json.dumps(_row(1, "shipped", kickCount={"left": 5, "right": 2},
                                 kicksBack={"left": 1, "right": 0})) + "\n")
    old = load_done(str(f), "shipped", 1, 300.0)[1]
    assert "kicks 9 (back 1/7, back-line —)" in _seed_line(old)


def test_the_ledger_summary_says_nothing_rather_than_zero_over_rows_that_predate_it(capsys, tmp_path):
    """Same rule as the metric means: a row written before the ledger existed
    has no own goals to report, and reporting 0 would read as "none happened"."""
    f = tmp_path / "old.jsonl"
    f.write_text(json.dumps(_row(0, "shipped")) + "\n")
    rows = list(load_done(str(f), "shipped", 1, 300.0).values())
    _print_ledger(rows)
    assert capsys.readouterr().out == ""
    rows.append(_row(1, "shipped", goalsFor={"left": 1, "right": 0},
                     goalsAgainst={"left": 0, "right": 1}, ownGoals={"left": 0, "right": 1},
                     goalsUnattributed=2, kickCount={"left": 4, "right": 2},
                     kicksBack={"left": 3, "right": 0}, kickCarry={"left": -0.4, "right": 0.6},
                     ballOwnHalf={"left": 20.0, "right": 40.0}, spread={"left": None, "right": None},
                     crowd={"left": None, "right": None}, depth={"left": 1.0, "right": 1.2}))
    _print_ledger(rows)
    out = capsys.readouterr().out
    assert "right for 0 against 1 (own 1)" in out and "2 unattributed" in out
    assert "left 4 (back 3, carry -0.100 m each)" in out          # -0.4 over 4 kicks
    assert "ballOwnHalf left 20.00 right 40.00 s/min" in out
    assert "spread" not in out                                    # a 1v1 row carries None


def test_a_summary_over_a_half_upgraded_file_averages_only_the_seeds_that_have_it(tmp_path):
    f = tmp_path / "mixed.jsonl"
    new = _row(1, "shipped", ballProgress={"left": 2.0, "right": 1.0},
               ballAdvance={"left": 3.0, "right": 3.0},
               possession={"left": 10.0, "right": 8.0}, possessionWide={"left": 20.0, "right": 18.0})
    f.write_text(json.dumps(_row(0, "shipped")) + "\n" + json.dumps(new) + "\n")
    rows = list(load_done(str(f), "shipped", 1, 300.0).values())
    mean, n = _mean_field(rows, "ballProgress")
    assert (mean, n) == (pytest.approx(3.0), 1)                   # 2.0 + 1.0 from the one row that has it
    assert _mean_field(rows, "possession") == (pytest.approx(18.0), 1)
    assert len(rows) == 2                                         # the old seed is still not re-run
