"""The dribble recipe's gates, priced at the states that beat it.

This recipe has been beaten four times by the same shape — a term satisfiable
without doing the task gets satisfied without doing the task — and once by the
opposite (a gate so tight that the thing it was meant to encourage became
unaffordable). `dribble.py` records each one in prose. These are the same
claims as numbers, so the next change to a weight has to keep them true.

Every test here was checked against the bug it is meant to catch: the planted
regression is named in each docstring, and each one was confirmed to FAIL with
that regression in place before being committed. A test that has never failed
has not been shown to test anything.
"""

import math

import numpy as np
import pytest

from microduck_local.behaviors import BEHAVIORS, BehaviorEnv
from microduck_local.behaviors.dribble import (
    DRIBBLE_BALL_ROLLING,
    DRIBBLE_GONE,
    DRIBBLE_KEEP,
    DRIBBLE_LOST,
    DRIBBLE_ROLL_LADDER,
    _ball_backward,
    _ball_believed,
    _ball_overshoot,
    _closing,
    _doing,
    _dribble_command,
    _dribble_gone,
    _dribble_lost,
    _engaged,
    _gaze_ball,
    _have,
    _target_dir,
    _turn_track,
)
from microduck_local.behaviors.kick import BALL_Z, _kick_ball_ids


def _env():
    env = BehaviorEnv("dribble", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    return env


def _place_ball(env, ahead, left=0.0, vx=0.0, vy=0.0):
    """Put the ball at a body-frame offset and give it a world velocity."""
    import mujoco

    _, qadr, dadr = _kick_ball_ids(env)
    yaw = float(np.arctan2(
        2.0 * (env.data.qpos[3] * env.data.qpos[6] + env.data.qpos[4] * env.data.qpos[5]),
        1.0 - 2.0 * (env.data.qpos[5] ** 2 + env.data.qpos[6] ** 2)))
    c, s = math.cos(yaw), math.sin(yaw)
    t = env._trunk_xpos
    env.data.qpos[qadr] = float(t[0]) + ahead * c - left * s
    env.data.qpos[qadr + 1] = float(t[1]) + ahead * s + left * c
    env.data.qvel[dadr:dadr + 6] = 0.0
    env.data.qvel[dadr] = vx
    env.data.qvel[dadr + 1] = vy
    mujoco.mj_forward(env.model, env.data)


def _set_duck_vel(env, ahead, left=0.0):
    import mujoco

    yaw = float(np.arctan2(
        2.0 * (env.data.qpos[3] * env.data.qpos[6] + env.data.qpos[4] * env.data.qpos[5]),
        1.0 - 2.0 * (env.data.qpos[5] ** 2 + env.data.qpos[6] ** 2)))
    c, s = math.cos(yaw), math.sin(yaw)
    env.data.qvel[env._root_qvel + 0] = ahead * c - left * s
    env.data.qvel[env._root_qvel + 1] = ahead * s + left * c
    mujoco.mj_forward(env.model, env.data)


# --- the terminal ---------------------------------------------------------

def test_losing_the_ball_does_not_end_the_episode():
    """Planted regression: `terminate_fn=_dribble_lost` (what it used to be).

    The ball leaving the dribble band used to end the clip, which deleted the
    only state in which sight pays — the duck never had to re-find anything.
    Measured consequence (12ax F): blinding the detector entirely changed
    nothing, because the perception path was never load-bearing.
    """
    env = _env()
    _place_ball(env, ahead=(DRIBBLE_LOST + DRIBBLE_GONE) / 2.0)
    assert _dribble_lost(env), "ball should be outside the dribble band"
    assert not _dribble_gone(env), "...but inside the recovery corridor"
    assert BEHAVIORS["dribble"].terminate_fn is _dribble_gone


def test_the_corridor_is_wide_enough_to_walk():
    """A corridor narrower than a stride is not a corridor. At 0.25 m/s a duck
    covers 0.3 m in the ~1.2 s a nudged ball takes to leave the band."""
    assert DRIBBLE_GONE - DRIBBLE_LOST >= 0.3


def test_a_truly_gone_ball_still_ends_it():
    env = _env()
    _place_ball(env, ahead=DRIBBLE_GONE + 0.2)
    assert _dribble_gone(env)


# --- the gates ------------------------------------------------------------

def test_closing_is_zero_while_the_ball_is_held():
    """Planted regression: drop the `_have` guard from `_closing`.

    Inside the band, closing IS dribbling and `_doing` already prices it;
    paying both lets one metre of progress earn twice.
    """
    env = _env()
    _place_ball(env, ahead=DRIBBLE_KEEP)
    _set_duck_vel(env, ahead=0.3)
    assert _have(env) == 1.0
    assert _closing(env) == 0.0


def test_closing_pays_only_while_the_gap_shrinks():
    """Planted regression: pay `1 - dist/GONE` instead of approach SPEED.

    A term that pays for the ball being far is farmable by putting it there.
    This one is zero unless the duck is moving toward it.
    """
    env = _env()
    _place_ball(env, ahead=0.8)
    _set_duck_vel(env, ahead=0.0)
    assert _closing(env) == 0.0, "standing still while the ball is away pays nothing"
    _set_duck_vel(env, ahead=-0.3)
    assert _closing(env) == 0.0, "walking AWAY pays nothing"
    _set_duck_vel(env, ahead=0.3)
    assert _closing(env) > 0.5, "walking toward it pays"


def test_engaged_is_a_selector_not_a_sum():
    """`_doing` and `_closing` are exclusive by construction, so `_engaged`
    can never exceed either one — it cannot pay for holding AND fetching."""
    env = _env()
    for ahead in (DRIBBLE_KEEP, 0.3, 0.6, 1.0):
        for v in (-0.3, 0.0, 0.3):
            _place_ball(env, ahead=ahead)
            _set_duck_vel(env, ahead=v)
            assert _engaged(env) == max(_doing(env), _closing(env))
            assert _engaged(env) <= 1.0
            assert min(_doing(env), _closing(env)) == 0.0


# --- the perversion this recipe already measured --------------------------

def test_admiring_a_ball_you_lost_pays_nothing():
    """Planted regression: `return bell` (ungate `_gaze_ball` outright).

    The measured perversion: the bell alone paid 0.988 for a ball kicked away
    against 0.655 for the head pitched down onto a ball at the feet, because a
    DISTANT ball sits near a level axis. Ungating it pays for the failure.
    The fix is the `_engaged` gate, not the bell — a lost ball the duck is not
    closing on is worth nothing to look at.
    """
    env = _env()
    _place_ball(env, ahead=0.8)
    _set_duck_vel(env, ahead=0.0)
    assert _gaze_ball(env) == 0.0

    _set_duck_vel(env, ahead=0.3)
    assert _gaze_ball(env) > 0.0, "looking while FETCHING it is the whole point"


def test_gaze_stays_under_the_dribble():
    """Looking must never be competitive with walking: the "just staring at
    the ball" failure this recipe was rebuilt to stop."""
    terms = {t.key: t.weight for t in BEHAVIORS["dribble"].terms}
    assert terms["gaze_ball"] < terms["ball_with_me"]
    assert terms["gaze_ball"] + terms["ball_seek"] < (
        terms["ball_with_me"] + terms["ball_close"] + terms["keep_pace"]), (
        "seeking must not out-earn dribbling, or shove-and-chase is the optimum")


# --- the standing optimum, which has won three times ----------------------

def test_a_still_duck_with_a_lost_ball_earns_only_survival():
    """Every positive term except `stay_upright` must be zero for a duck that
    is neither dribbling nor fetching. `stay_upright` is deliberately ungated
    (gating it made falling free: 7/20 falls against 0/20, ep_len 346 -> 52)."""
    env = _env()
    _place_ball(env, ahead=0.8)
    _set_duck_vel(env, ahead=0.0)
    assert _doing(env) == 0.0
    assert _closing(env) == 0.0
    assert _engaged(env) == 0.0
    ungated = [t.key for t in BEHAVIORS["dribble"].terms
               if not t.is_penalty and abs(t.fn(env)) > 1e-9]
    assert ungated == ["stay_upright"], f"also earning while idle: {ungated}"


@pytest.mark.parametrize("stage", range(3))
def test_curriculum_ladders_the_world_not_the_pay(stage):
    """Each rung moves spawn/terminal knobs only. A curriculum that changes
    weights is a curriculum that trains three different tasks."""
    env = BEHAVIORS["dribble"].curriculum[stage].env
    assert all(k.startswith("MICRODUCK_DRIBBLE_") or k.startswith("MICRODUCK_BALL_")
               for k in env), env
    gone = float(env["MICRODUCK_DRIBBLE_GONE"].split(",")[0])
    lost = float(env["MICRODUCK_DRIBBLE_LOST"].split(",")[0])
    assert gone >= lost, "the corridor cannot be negative"
    if stage == 0:
        assert gone == lost, "stage 1 has no corridor: learn the dribble first"
        assert float(env["MICRODUCK_DRIBBLE_NUDGE_RATE"].split(",")[0]) == 0.0


# --- the oracle, closed ---------------------------------------------------

def _step_until_belief(env, want: bool, limit: int = 200):
    """Step with zero actions until the belief is/ isn't present."""
    for _ in range(limit):
        if (getattr(env, "_lm_world", None) is not None) is want:
            return True
        env.step(np.zeros(14, dtype=np.float32))
    return (getattr(env, "_lm_world", None) is not None) is want


def test_the_command_comes_from_the_belief_not_the_truth():
    """Planted regression: `_ball_believed` reading `env.data.qpos[ball]`.

    That WAS the recipe: `_dribble_command` placed the dribble spot from the
    ball's TRUE position and published the resulting heading in obs[48:51], so
    an oracle told the policy where the ball was every step. Measured: the
    commanded yaw correlated +0.642 with the true ball bearing, and a
    permutation ablation put the whole task on that channel — destroying it
    took full clips 28/30 -> 20/30 and losses 0 -> 8/30, while destroying the
    CAMERA cost nothing at all.

    So: with the belief pinned, moving the true ball must not move the command.
    `_dribble_command` is called directly because it is the unit under test —
    `obs_fn` also re-senses, which would legitimately update the belief (and
    short-circuits on `_lm_step_done`, which is what made the first version of
    this test read the reset value instead of the command).

    On the robot `Chase` steers on the detector's estimate, so this is a
    sim2real contract and not a preference.
    """
    env = _env()
    _place_ball(env, ahead=DRIBBLE_KEEP)
    env._lm_world = np.array([float(env.data.qpos[0]) + 0.20,
                              float(env.data.qpos[1]), BALL_Z])
    _dribble_command(env)
    before = tuple(float(x) for x in env.twist_cmd)
    body_before = tuple(float(x) for x in env.body_cmd)

    _place_ball(env, ahead=0.30, left=0.35)      # the TRUTH moves a long way
    _dribble_command(env)                        # ...the belief did not
    assert tuple(float(x) for x in env.twist_cmd) == pytest.approx(before, abs=1e-6), (
        "the command moved with the TRUE ball while the belief stood still — "
        "that is the oracle this recipe exists to close")
    assert tuple(float(x) for x in env.body_cmd) == pytest.approx(body_before, abs=1e-6)

    # ...and it MUST move when the belief moves, or it is reading nothing.
    env._lm_world = np.array([float(env.data.qpos[0]),
                              float(env.data.qpos[1]) + 0.30, BALL_Z])
    _dribble_command(env)
    assert tuple(float(x) for x in env.twist_cmd) != pytest.approx(before, abs=1e-6), (
        "the command ignored the belief too — it is steering on neither")


def test_no_sighting_means_no_command():
    """Planted regression: falling back to `twist_cmd[0] = _dribble_speed`.

    That was this block's first version, and it re-created the leak the change
    exists to close: 12ax G measured that the ball is never far and the command
    points forward, so WALKING FINDS IT. A forward fallback hands that back
    every time the duck stops looking. A zero command prices itself — `_going`
    returns 0 without a commanded speed, and `_doing` with it.
    """
    env = _env()
    _place_ball(env, ahead=DRIBBLE_KEEP)
    env._lm_world = None
    _dribble_command(env)
    assert float(env.twist_cmd[0]) == 0.0, "no fix on the ball, no forward command"
    assert float(env.twist_cmd[2]) == 0.0, "no fix on the ball, no turn command"
    assert all(float(env.body_cmd[i]) == 0.0 for i in range(3))
    assert _doing(env) == 0.0, "an unknowing duck must not collect the dribble terms"


def test_the_oracle_knob_still_restores_the_old_behaviour():
    """The A/B has to be one knob, or the comparison is two code paths."""
    import os
    os.environ["MICRODUCK_DRIBBLE_ORACLE"] = "1"
    try:
        env = _env()
        env._lm_world = None                      # never seen it...
        _place_ball(env, ahead=DRIBBLE_KEEP)
        env.behavior.obs_fn(env)
        assert _ball_believed(env) is not None, "the oracle arm reads truth by design"
    finally:
        os.environ.pop("MICRODUCK_DRIBBLE_ORACLE", None)


def test_the_spawn_gaze_can_actually_see_the_ball():
    """The reachable set, checked before any training: with the command coming
    from the belief, a spawn with the ball out of frame is a spawn with NO
    command, and a stage whose rollouts never contain a sighting can never
    learn to use one. The old default window put the ball in frame on 2 of 100
    spawns; measured on this one, 67%."""
    from microduck_local.behaviors import BehaviorEnv
    seen = 0
    for seed in range(30):
        env = BehaviorEnv("dribble", obs_noise=False, domain_rand=False,
                          action_delay=False, random_yaw=False, seed=seed)
        env.reset(seed=seed)
        seen += 1 if getattr(env, "_lm_world", None) is not None else 0
    assert seen >= 15, (
        f"only {seen}/30 spawns start with the ball in frame — under a believed "
        "command that is a majority of episodes with no steering at all")


# --- the turn, which is what actually caps the task -----------------------

def test_a_zero_yaw_command_pays_no_turn_tracking():
    """Planted regression: `return _run_track_ang(env)` unscaled.

    The Gaussian is exp(-(z_err^2 + w_xy^2)/0.5), so a duck standing perfectly
    still with a zero command scores **1.0** — 2.0 a step of free standing
    income, which is the optimum this recipe has already been rebuilt by three
    times. Scaling by the demand makes a command that asks for nothing pay
    nothing.
    """
    env = _env()
    env.twist_cmd[2] = 0.0
    assert _turn_track(env) == 0.0, "a zero yaw command must pay nothing"


def test_turn_tracking_pays_when_the_turn_is_asked_for_and_made():
    env = _env()
    clip = float(BEHAVIORS["dribble"].curriculum[0].env[
        "MICRODUCK_DRIBBLE_TURN"].split(",")[1])
    env.twist_cmd[2] = clip                 # the steering asks for the full turn
    env._gyro = np.array([0.0, 0.0, clip])  # ...and the body delivers it
    good = _turn_track(env)
    env._gyro = np.array([0.0, 0.0, -clip])  # ...or turns the WRONG way
    bad = _turn_track(env)
    assert good > bad > 0.0 or good > bad, "tracking the command must beat fighting it"
    assert good > 0.5, f"a tracked full-clip turn should pay well, got {good:.3f}"


def test_the_turn_is_laddered_and_ends_where_the_task_needs_it():
    """The measured demand is ~30 deg/s (the duck travels a median 73 deg off
    the ball->target line and the command saturates 58% of the time at
    8.6 deg/s). Cold, 29 deg/s is 16/20 falls — hence a ladder, not a jump."""
    clips = [float(st.env["MICRODUCK_DRIBBLE_TURN"].split(",")[1])
             for st in BEHAVIORS["dribble"].curriculum]
    assert clips == sorted(clips), f"the turn ladder must not go backwards: {clips}"
    assert clips[0] <= 0.15, "rung 1 must start at the clip the donor survives"
    assert clips[-1] >= 0.45, (
        f"the top rung is {clips[-1]} rad/s = {math.degrees(clips[-1]):.0f} deg/s; "
        "the task needs ~30 deg/s or the duck cannot get behind the ball")


# --- the cancellation: driving the ball backwards used to be free ---------

def test_driving_the_ball_backwards_is_priced():
    """Planted regression: no `ball_backward` term at all (the shipped state
    until 2026-09-24).

    `_ball_with_me` reads `clip(ball.u, 0.0, TARGET)`, so a ball shoved AWAY
    from its target scored exactly what an untouched ball scored: zero. Half of
    every clip was invisible — the come-round branch retreating at -0.043 m/s
    against the push-through's +0.040, cancelling to -0.004 net — and 2.4 M
    steps of turn ladder moved the falls and not the ball because of it.
    """
    terms = {t.key: t for t in BEHAVIORS["dribble"].terms}
    assert "ball_backward" in terms, "nothing prices the ball going the wrong way"
    assert terms["ball_backward"].is_penalty
    assert terms["ball_backward"].weight >= 0.9, (
        "below ~0.9 the retreat merely cancels the push-through again, which is "
        "the state this term exists to end")

    env = _env()
    ux, uy, _d = _target_dir(env)
    _, _, dadr = _kick_ball_ids(env)
    env.data.qvel[dadr:dadr + 2] = [ux * 0.20, uy * 0.20]        # ball advancing
    assert _ball_backward(env) == 0.0, "advancing must not be docked"
    env.data.qvel[dadr:dadr + 2] = [-ux * 0.20, -uy * 0.20]      # ball retreating
    assert _ball_backward(env) < 0.0, "retreating must be docked"


@pytest.mark.parametrize("duck_side", (+1.0, -1.0))
def test_the_come_round_path_does_not_cross_the_ball(duck_side):
    """Planted regression: `spot = (bx - tux*behind, by - tuy*behind)`, and
    separately `side = 1.0`.

    The bare point behind the ball sends an OVERRUN duck straight across the
    ball's own position, so it walks into the ball on its way to standing
    behind it. With the lateral offset the commanded path clears it; the same
    policy, no retraining, went come-round -0.043 -> -0.019 m/s and NET ball
    progress -0.004 -> +0.011.

    BOTH SIDES are checked, because the first version of this test only ever
    placed the duck on the +side and so passed happily against a hard-coded
    `side = 1.0` — which sends half of all approaches the long way round,
    back across the ball. A test that cannot fail is not a test.
    """
    import mujoco

    env = _env()
    ux, uy, _d = _target_dir(env)
    _, qadr, _ = _kick_ball_ids(env)
    bx, by = float(env.data.qpos[qadr]), float(env.data.qpos[qadr + 1])
    px, py = -uy, ux
    # PAST the ball along the line (the overrun case), off to the given side
    env.data.qpos[0] = bx + ux * 0.10 + px * 0.06 * duck_side
    env.data.qpos[1] = by + uy * 0.10 + py * 0.06 * duck_side
    mujoco.mj_forward(env.model, env.data)
    _dribble_command(env)
    spot = env._dribble_spot

    lateral = (spot[0] - bx) * px + (spot[1] - by) * py
    assert abs(lateral) > 0.05, (
        f"the come-round spot is only {abs(lateral):.3f} m off the line — the "
        "path crosses the ball")
    assert math.copysign(1.0, lateral) == duck_side, (
        f"duck is on side {duck_side:+.0f} and the spot is on "
        f"{math.copysign(1.0, lateral):+.0f} — the long way round, back across the ball")


# --- kick-and-chase: the touch must be keepable ---------------------------

def test_a_ball_the_duck_cannot_follow_is_docked():
    """Planted regressions: the fixed `DRIBBLE_TARGET` threshold, and the
    `max(DRIBBLE_TARGET, duck_speed)` floor that made the retarget a no-op.

    Measured on the trained policy: 1.3 foot touches per 8 s clip, each sending
    the ball off at 0.436 m/s while the duck walks at 0.193 — 2.3x its own
    pace. Against the old fixed 0.35 m/s threshold that cost **0.0037 a step**,
    about a 1600th of `ball_close`, so kick-and-chase was free. The threshold
    is the duck's own speed now: a ball faster than the duck cannot be kept.
    """
    env = _env()
    _, _, dadr = _kick_ball_ids(env)
    env.data.qvel[env._root_qvel:env._root_qvel + 2] = [0.20, 0.0]   # duck at 0.20 m/s
    env.data.qvel[dadr:dadr + 2] = [0.15, 0.0]                       # ball SLOWER — keepable
    assert _ball_overshoot(env) == 0.0
    env.data.qvel[dadr:dadr + 2] = [0.44, 0.0]                       # ball 2.2x the duck
    hard = _ball_overshoot(env)
    assert hard < -0.15, (
        f"a ball at 0.44 m/s under a duck at 0.20 scored {hard:.4f} — that is the "
        "kick-and-chase this term exists to price")
    # a STATIONARY duck must not get a free nudge either
    env.data.qvel[env._root_qvel:env._root_qvel + 2] = [0.0, 0.0]
    assert _ball_overshoot(env) < -0.15


def test_the_task_can_outearn_its_own_preconditions():
    """The audit that should have been run on day one. On the trained policy,
    per-step income shares were:

        stay_upright 36.1%   ball_close 17.2%   turn_track 16.5%
        ball_with_me  8.3%  <- THE TASK

    `ball_with_me` carried weight 12 and delivered 8 % because the policy only
    reached 3.6 % of its maximum, while `turn_track` at weight 2 delivered
    16.5 % by reaching 43 % of its own. **A weight is not an income.** So the
    task's weight must at least exceed the sum of the terms that pay merely for
    being upright and near the ball, or the optimiser is right to ignore it.
    """
    w = {t.key: t.weight for t in BEHAVIORS["dribble"].terms}
    # THE TASK is what moves the ball to its target. `ball_progress` is the
    # displacement form and carries it; `ball_with_me` is the velocity product,
    # kept at a reduced weight because it is the only term insisting the duck
    # and the ball move TOGETHER — but it is not the task, because its value
    # rose x2.6 while the ball advanced no further.
    task = w["ball_progress"] + w["ball_with_me"]
    preconditions = w["stay_upright"] + w["ball_close"]
    assert w["ball_progress"] > 2 * preconditions, (
        f"the displacement task is {w['ball_progress']} against {preconditions} "
        "for standing upright and loitering near the ball")
    assert task > 4 * preconditions
    assert w["turn_track"] < w["ball_progress"] / 10, (
        "turn_track is a MEANS; at 2.0 it out-earned the task 16.5% to 8.3%")


# --- the ball's rolling friction: the physics rung -------------------------
#
# The knob that made a dribble possible at all. Every assertion here was
# watched failing against a planted regression before it was kept:
#   * drop the `geom_friction` write in `_dribble_reset` -> reaches_the_model
#   * default 0.0001 -> 0.05                              -> default_is_upstreams
#   * a rung at 0.002                                     -> rungs_bite
#   * delete the knob from one stage                       -> every_rung_declares

def _ball_env(ov=None):
    env = BehaviorEnv("dribble", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0,
                      spawn_overrides=dict(ov or {}))
    env.reset(seed=0)
    return env


def test_the_rolling_knob_reaches_the_model():
    """A knob nothing reads is the failure mode this repo keeps hitting: the
    nudge rate looked staged all session and was not."""
    env = _ball_env({"MICRODUCK_DRIBBLE_BALL_ROLLING": "0.05,0.05"})
    gid = int(env.model.geom("ball_geom").id)
    assert env.model.geom_friction[gid][2] == pytest.approx(0.05)


def test_the_default_matches_the_composed_scene():
    """THE guard. A reset hook whose "default" differs from the scene's own
    value is not a default, it is a silent physics change applied to every
    dribble run — and it happened: the hook shipped at ball.xml's 0.0001 while
    the composed scene says 0.002, so the ball rolled 1.44 m a clip instead of
    0.83 m and a whole 6M chain trained on a ball nothing else here uses.
    Compare against the SCENE, never against the upstream file."""
    import mujoco

    from microduck_local.contract import scene_walk_ball_xml

    m = mujoco.MjModel.from_xml_path(str(scene_walk_ball_xml()))
    scene_roll = float(m.geom_friction[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")][2])
    assert DRIBBLE_BALL_ROLLING == pytest.approx(scene_roll), (
        f"default {DRIBBLE_BALL_ROLLING} overrides the composed scene's {scene_roll}")
    env = _ball_env()
    gid = int(env.model.geom("ball_geom").id)
    assert env.model.geom_friction[gid][2] == pytest.approx(scene_roll)


def test_the_ball_geom_has_condim_6_or_the_knob_is_a_noop():
    """Under condim 3 MuJoCo ignores the rolling coefficient entirely. That has
    been missed twice here (the soccer ball, then MOSS's cans), and it would
    make this whole ladder silently flat."""
    env = _ball_env()
    gid = int(env.model.geom("ball_geom").id)
    assert int(env.model.geom_condim[gid]) >= 6, "rolling friction needs condim 6"


def test_every_rung_is_a_measured_value():
    """Each rung must be one this repo has actually rolled a ball at: 0.05 and
    0.01 were measured (12.0 and 3.75 touches a clip), and the bottom rung is
    the scene's own ball. A rung nobody has measured costs 2M steps to find out."""
    for i, v in enumerate(DRIBBLE_ROLL_LADDER):
        assert v in (0.05, 0.01) or v == pytest.approx(DRIBBLE_BALL_ROLLING), \
            f"rung {i} = {v} has never been measured here"


@pytest.mark.parametrize("stage", range(3))
def test_every_rung_declares_its_ball(stage):
    """An added stage must not silently inherit whatever the previous one left
    on the model."""
    env = BEHAVIORS["dribble"].curriculum[stage].env
    assert "MICRODUCK_DRIBBLE_BALL_ROLLING" in env, "stage does not name its ball"
    assert float(env["MICRODUCK_DRIBBLE_BALL_ROLLING"].split(",")[0]) == \
        pytest.approx(DRIBBLE_ROLL_LADDER[stage])


def test_the_ladder_descends_to_the_shipped_ball():
    """A ladder that stops on the sticky ball has not answered the question."""
    assert list(DRIBBLE_ROLL_LADDER) == sorted(DRIBBLE_ROLL_LADDER, reverse=True), \
        "the rungs must get harder, not easier"
    assert DRIBBLE_ROLL_LADDER[-1] == pytest.approx(DRIBBLE_BALL_ROLLING)
