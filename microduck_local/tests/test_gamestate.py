"""Locks for the game state (roadmap Track 4 §6 B.3): the World is the
GameController — after a goal the side that conceded kicks off and the
state runs set → kickoff → playing; the team board carries the message;
a chase brain with `kickoff_wait` stands off the other side's restart in
its own half until the ball leaves the spot. Ships off until measured."""

import math

import mujoco

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.runtime import Senses
from microduck_local.brain.team import Team, kickoff_brains
from microduck_local.sensors.detector import Detection, DetectionFrame
from microduck_local.world import World, make_pitch


def test_a_goal_hands_the_kickoff_to_the_side_that_conceded_and_the_state_runs_set_kickoff_playing():
    sc = make_pitch()
    w = World(sc, seed=3)
    home, away = sc.ducks[0].team, sc.ducks[1].team               # home attacks +x ("right")
    assert w.team_defending("left") == home and w.team_defending("right") == away
    assert w.kickoff_team is None and w.game_state == "playing"     # the first kickoff is contested
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    hx = sc.floor[0] / 2 - 0.25
    w.data.qpos[q:q + 2] = [-(hx - 0.03), 0.0]                      # into the LEFT mouth: against home
    mujoco.mj_forward(w.model, w.data)
    w.step()
    assert w.goals["left"] == 1 and w.kickoff_team == home
    assert w.game_state == "set" and w.soccer_score()["state"] == "set"
    assert w.soccer_score()["kickoffTeam"] == home
    while w.in_kickoff:
        w.step()
    assert w.game_state == "kickoff"                                # the ball is still on the spot
    bx, by = w.kickoff_ball
    w.data.qpos[q:q + 2] = [bx + 0.2, by]                           # somebody kicked it
    mujoco.mj_forward(w.model, w.data)
    assert w.game_state == "playing"
    # The window: with none, play resumes the moment the hold ends.
    w.data.qpos[q:q + 2] = [-(hx - 0.03), 0.0]
    mujoco.mj_forward(w.model, w.data)
    w.step()
    assert w.goals["left"] == 2 and w.game_state == "set"
    w.kickoff_free_s = 0.0
    while w.in_kickoff:
        w.step()
    assert w.game_state == "playing"
    w.reset()
    assert w.kickoff_team is None and w.kickoff_ball is None and w.game_state == "playing"


def test_the_board_waits_only_for_the_other_sides_kickoff_until_the_ball_leaves_the_spot():
    tm = Team("graphite")
    assert not tm.waits(5.0)                                        # never stamped: never waits
    tm.kickoff(ours=False, until=20.0, ball=(0.02, -0.01))
    assert tm.waits(5.0) and tm.waits(5.0, (0.05, 0.0))             # unseen, or still on the spot
    assert not tm.waits(5.0, (0.3, 0.0))                            # a duck saw it move
    tm.claim("d3", 6.0, 0.5, (0.25, 0.0))                           # the board's own sighting counts too
    assert not tm.waits(6.0)
    assert not tm.waits(25.0, (0.0, 0.0))                           # the window ran out
    tm.kickoff(ours=True, until=20.0, ball=(0.0, 0.0))
    assert not tm.waits(5.0)                                        # our ball: play
    tm.kickoff(ours=False, until=20.0, ball=(0.0, 0.0))
    tm.reset()
    assert not tm.waits(5.0)                                        # a wipe forgets the message


def test_kickoff_brains_stamps_the_boards_from_the_world():
    class Ctl:
        kickoff_team = "cream"
        kickoff_until = 1.0
        kickoff_free_s = 10.0
        kickoff_ball = (0.0, 0.0)
        kickoff_moved_m = 0.1
    teams = {"cream": Team("cream"), "graphite": Team("graphite")}
    kickoff_brains({}, teams, Ctl())
    assert not teams["cream"].waits(5.0) and teams["graphite"].waits(5.0)
    assert not teams["graphite"].waits(11.5)
    kickoff_brains({}, teams)                                       # no controller: nobody waits
    assert not teams["graphite"].waits(5.0)
    Ctl.kickoff_team = None                                         # a contested kickoff
    kickoff_brains({}, teams, Ctl())
    assert not teams["cream"].waits(5.0) and not teams["graphite"].waits(5.0)


def _see(t, odom, ball_xy):
    rng = math.hypot(ball_xy[0] - odom[0], ball_xy[1] - odom[1])
    bearing = math.atan2(ball_xy[1] - odom[1], ball_xy[0] - odom[0]) - odom[2]
    det = DetectionFrame(t=t, detections=[Detection("ball", "", bearing, -0.3, 0.12, rng, 0.9)])
    return Senses(t=t, det=det, det_age=0.0, odom=odom, speed=0.0)


def test_a_waiting_duck_supports_in_its_own_half_and_plays_once_the_ball_leaves_the_spot():
    assert ChaseParams().kickoff_wait is False                      # ships off until measured
    tm = Team("cream", half_x=1.75)
    tm.jobs = {"d0": "defender", "d2": "striker"}
    b = Chase(ChaseParams(kickoff_wait=True), goal=(1.75, 0.0), team=tm, duck_id="d2",
              bounds=(1.75, 1.25), goal_w=0.7, role="striker")
    tm.kickoff(ours=False, until=20.0, ball=(0.0, 0.0))
    odom = (-0.6, 0.3, 0.0)
    for k in range(3):
        b.step(_see(5.0 + 0.02 * k, odom, (0.0, 0.0)))
    # Alone on the board it would attack; standing off, it supports from its own half.
    assert b._kickoff_wait and b.role == "support" and b.state == "wait"
    assert b.post is not None and b._attack_x(b.post[0]) <= -0.3 / 1.75 + 1e-9, b.post
    # The ball leaves the spot: the wait ends and the striker goes for it.
    for k in range(3):
        b.step(_see(5.1 + 0.02 * k, odom, (0.4, 0.0)))
    assert not b._kickoff_wait and b.role == "attack" and b.state != "wait"
    # The knob off: the same board message changes nothing.
    off = Chase(ChaseParams(), goal=(1.75, 0.0), team=Team("cream", half_x=1.75), duck_id="d2",
                bounds=(1.75, 1.25), goal_w=0.7, role="striker")
    off.team.kickoff(ours=False, until=20.0, ball=(0.0, 0.0))
    for k in range(3):
        off.step(_see(5.0 + 0.02 * k, odom, (0.0, 0.0)))
    assert not off._kickoff_wait and off.role == "attack"
    # A kickoff on the brain forgets the flag; the knob reads off the environment.
    b._kickoff_wait = True
    b.kickoff()
    assert b._kickoff_wait is False
    assert ChaseParams.from_env("kickoff_wait=1,kickoff_circle=0.25").kickoff_circle == 0.25
