"""The kick that SEES the ball (behaviors/lastmetre.py, roadmap 12h / E.1):
the wide kick's world and the wide kick's pay, with the ball's position
riding the four HEAD command slots of the 61-obs contract.

Locks the three things a recipe can get silently wrong: the pay (it must be
the wide kick's, term for term, or "the difference is the observation" is
not what was trained), the ladder (spawn knobs only, never the reward), and
the SLOTS — which is the part no reward curve would ever complain about.
"""

import math

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.behaviors import BEHAVIORS, match_behavior
from microduck_local.behaviors.core import _face_home_pen
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.behaviors.kick import (
    BALL_Z,
    HEAD_DOWN,
    KICK_BOX_AHEAD,
    KICK_BOX_SIDE,
    NECK_DOWN,
    _kick_ball_ids,
)
from microduck_local.behaviors.lastmetre import (
    LM_FAR_AHEAD,
    LM_FAR_SIDE,
    LM_GAZE_STAGE1,
    LM_RANGE_SCALE,
    LM_RANGE_SCALE_FAR,
    _lm_sense,
)

HEAD_YAW_ID = C.JOINT_NAMES.index("head_yaw")
# Every one of these is a SPAWN window (or the clip the spawn gets to play
# out in). Nothing here may ever name a reward key: a stage laddering the pay
# is the failure this allowlist exists to catch, and no reward curve would
# ever complain about it.
STAGE_KNOBS = {"MICRODUCK_KICK_BOX_AHEAD", "MICRODUCK_KICK_BOX_SIDE",
               "MICRODUCK_LM_GAZE_NECK", "MICRODUCK_LM_GAZE_HEAD", "MICRODUCK_LM_GAZE_YAW",
               "MICRODUCK_LM_FAR_PROB", "MICRODUCK_LM_FAR_AHEAD", "MICRODUCK_LM_FAR_SIDE",
               "MICRODUCK_EPISODE_S"}


def _env_id(behavior_id: str, **over):
    return BehaviorEnv(behavior_id, seed=3, max_episode_s=2.0, domain_rand=False,
                       random_yaw=False, obs_noise=False, action_delay=False,
                       spawn_overrides=over or None)


def _env(side: str, **over):
    return _env_id(f"kick_{side}_sensed", **over)


def _place(env, ahead: float, side: float, neck: float, head: float, yaw: float = 0.0):
    """Put the true ball at (ahead, side) in the duck's body frame — + side is
    to its LEFT — set the gaze, and run one detector report."""
    env.reset(seed=11)
    _, qadr, dadr = _kick_ball_ids(env)
    env.data.qpos[qadr:qadr + 7] = [float(env.data.qpos[0]) + ahead,
                                    float(env.data.qpos[1]) + side, BALL_Z, 1.0, 0.0, 0.0, 0.0]
    env.data.qvel[dadr:dadr + 6] = 0.0
    env.data.qpos[env.joint_qpos_adr[5]] = C.DEFAULT_POSE[5] + neck
    env.data.qpos[env.joint_qpos_adr[6]] = C.DEFAULT_POSE[6] + head
    env.data.qpos[env.joint_qpos_adr[HEAD_YAW_ID]] = C.DEFAULT_POSE[HEAD_YAW_ID] + yaw
    env.data.ctrl[:] = env.data.qpos[env.joint_qpos_adr]
    mujoco.mj_forward(env.model, env.data)
    env._lm_world, env._lm_conf, env._lm_det_seen = None, 0.0, False
    env._lm_det_step = -10 ** 9
    _lm_sense(env, force=True)
    return env.head_cmd.copy()


def test_the_sensed_kick_is_the_wide_kicks_pay_and_the_wide_kicks_box():
    """Same terms, same weights, same signs as `kick_<side>_wide` — the two
    recipes differ in the OBSERVATION, so anything else that differs makes
    the comparison meaningless."""
    for side in ("right", "left"):
        b, wide = BEHAVIORS[f"kick_{side}_sensed"], BEHAVIORS[f"kick_{side}_wide"]
        assert b.scene == "ball" and b.episode_s == pytest.approx(2.0) and b.terminate_on_fall
        assert not b.symmetric                                   # it names a foot
        assert [(t.key, t.weight, t.is_penalty) for t in b.terms] == \
               [(t.key, t.weight, t.is_penalty) for t in wide.terms]
        # `_kick_terms` closes a fresh function over the foot each call, so
        # compare what they ARE, not their identity.
        assert [t.fn.__name__ for t in b.terms] == [t.fn.__name__ for t in wide.terms]
        assert [t.friendly for t in b.terms] == [t.friendly for t in wide.terms]
        assert b.terms[-1].key == "face_line" and b.terms[-1].is_penalty and b.terms[-1].fn is _face_home_pen
        assert b.obs_fn is not None and b.reset_fn.__name__ == f"_lm_reset_{side}"
        # By ID, never by identity: `reload_library` (the lab's hot reload,
        # exercised in tests/test_lab.py) rebuilds every recipe object, and an
        # `is` check here passes alone and fails in the full suite.
        assert match_behavior(f"kick_{side}_sensed").id == b.id
        assert match_behavior(f"last metre {side}").id == b.id
        # ...and the plain strike keeps the generic phrase, deliberately:
        # a sensed kick is asked for by name, never by "kick right".
        assert match_behavior(f"kick {side}").id == f"kick_{side}"


def test_the_ladder_moves_the_world_and_never_the_pay():
    """AGENTS.md: a stage may ladder physics, spawns and strictness. Every
    stage knob here is a SPAWN window, and the stages are strictly opening."""
    for side in ("right", "left"):
        b = BEHAVIORS[f"kick_{side}_sensed"]
        assert len(b.curriculum) == 4
        # ...and it only ever OPENS: each rung's box is wider than the last in
        # both axes (the 6 x 6 rung sits 3 mm inside the strike spot's near
        # edge, so this is a width test, not a containment one). Rung 4 keeps
        # rung 3's near box and opens a SECOND window past it, so the widening
        # test is over the three near rungs and rung 4 is checked below.
        prev = None
        for st in b.curriculum[:3]:
            lo_a, hi_a = (float(v) for v in st.env["MICRODUCK_KICK_BOX_AHEAD"].split(","))
            lo_s, hi_s = (float(v) for v in st.env["MICRODUCK_KICK_BOX_SIDE"].split(","))
            box = (hi_a - lo_a, hi_s - lo_s)
            if prev is not None:
                assert box[0] > prev[0] and box[1] > prev[1], st.label
            prev = box
        for st in b.curriculum:
            assert set(st.env) <= STAGE_KNOBS, st.env
            assert st.detail
        assert sum(st.steps for st in b.curriculum) == b.default_steps
        # Stage 2 IS the finished world: the full box, the full gaze, yaw home.
        for st in (b.curriculum[2], b.curriculum[3]):
            assert st.env["MICRODUCK_KICK_BOX_AHEAD"] == f"{KICK_BOX_AHEAD[0]},{KICK_BOX_AHEAD[1]}"
            assert st.env["MICRODUCK_KICK_BOX_SIDE"] == f"{KICK_BOX_SIDE[0]},{KICK_BOX_SIDE[1]}"
            assert st.env["MICRODUCK_LM_GAZE_NECK"] == f"{NECK_DOWN[0]},{NECK_DOWN[1]}"
            assert st.env["MICRODUCK_LM_GAZE_HEAD"] == f"{HEAD_DOWN[0]},{HEAD_DOWN[1]}"
            assert st.env["MICRODUCK_LM_GAZE_YAW"] == "0.0,0.0"
        # ...and stage 2 has no far share at all: it must reproduce the world
        # 12as measured, bit for bit.
        assert not (set(b.curriculum[2].env) & {"MICRODUCK_LM_FAR_PROB", "MICRODUCK_EPISODE_S"})
        # Stage 3 is the APPROACH rung: a share of the episodes out of reach,
        # and a clip long enough to step AND swing.
        appr = b.curriculum[3].env
        assert 0.0 < float(appr["MICRODUCK_LM_FAR_PROB"]) < 1.0     # the strike is still rehearsed
        far_lo, far_hi = (float(v) for v in appr["MICRODUCK_LM_FAR_AHEAD"].split(","))
        assert far_lo >= KICK_BOX_AHEAD[1] + 0.03                   # genuinely past the swing
        assert far_hi > far_lo
        side_lo, side_hi = (float(v) for v in appr["MICRODUCK_LM_FAR_SIDE"].split(","))
        assert side_lo < 0.0 < side_hi                              # either side, not just the foot's
        assert float(appr["MICRODUCK_EPISODE_S"]) >= 2 * b.episode_s


def test_the_spawn_puts_the_ball_in_the_box_and_the_drill_puts_the_gaze_on_it():
    """The default world is the wide kick's (ball over the box, head across
    the gaze range, head yaw HOME as the bench and the arena hand it over);
    stage 1 narrows the box and turns the gaze onto it."""
    for side, sgn in (("right", -1.0), ("left", 1.0)):
        env = _env(side)
        aheads, sides, necks, heads, yaws = [], [], [], [], []
        for k in range(40):
            env.reset(seed=30 + k)
            _, qadr, _ = _kick_ball_ids(env)
            aheads.append(float(env.data.qpos[qadr] - env.data.qpos[0]))
            sides.append(sgn * float(env.data.qpos[qadr + 1] - env.data.qpos[1]))
            necks.append(float(env.data.qpos[env.joint_qpos_adr[5]] - C.DEFAULT_POSE[5]))
            heads.append(float(env.data.qpos[env.joint_qpos_adr[6]] - C.DEFAULT_POSE[6]))
            yaws.append(float(env.data.qpos[env.joint_qpos_adr[HEAD_YAW_ID]]))
            assert _face_home_pen(env) == 0.0                  # the line is the one it began on
        assert KICK_BOX_AHEAD[0] - 1e-6 <= min(aheads) and max(aheads) <= KICK_BOX_AHEAD[1] + 1e-6
        assert KICK_BOX_SIDE[0] - 1e-6 <= min(sides) and max(sides) <= KICK_BOX_SIDE[1] + 1e-6
        assert max(aheads) - min(aheads) > 0.08 and max(sides) - min(sides) > 0.08
        assert NECK_DOWN[0] - 1e-6 <= min(necks) and max(heads) <= HEAD_DOWN[1] + 1e-6
        # Head yaw is HOME in the finished world (the drill's offset is 0,0
        # there), so only the walk env's own spawn noise is left on it — which
        # is exactly what the wide kick's spawn leaves.
        assert max(abs(y) for y in yaws) < 0.05

        st1 = dict(BEHAVIORS[f"kick_{side}_sensed"].curriculum[1].env)   # the 6 x 6 rung
        env = _env(side, **st1)
        seen, yaws = [], []
        for k in range(40):
            env.reset(seed=30 + k)
            _, qadr, _ = _kick_ball_ids(env)
            ahead = float(env.data.qpos[qadr] - env.data.qpos[0])
            assert 0.06 - 1e-6 <= ahead <= 0.12 + 1e-6
            yaws.append(float(env.data.qpos[env.joint_qpos_adr[HEAD_YAW_ID]]))
            seen.append(float(env.head_cmd[2]))
        lo, hi = (float(v) for v in LM_GAZE_STAGE1[2].split(","))
        # Yawed AT the ball (the offset is signed by the foot); the window plus
        # the walk env's own spawn noise on that joint.
        assert all(lo - 0.05 <= sgn * y <= hi + 0.05 for y in yaws), yaws
        assert min(sgn * y for y in yaws) > 0.2
        assert np.mean(seen) > 0.9        # ...and that is what makes it in frame: measured 99%


def test_the_ball_rides_the_head_slots_left_positive_right_negative():
    """The slot contract, and the only thing in this recipe a reward curve
    could never complain about: obs[51] is the ball's bearing in the DUCK's
    own frame, + to the left; obs[52] its ground range; obs[53] whether the
    detector has it; obs[54] how fresh that is. A ball nobody has seen reads
    all zeros."""
    env = _env("right")
    down = (-0.30, 0.60)                                  # a gaze the ball is in frame from
    left = _place(env, 0.16, +0.06, *down)
    right = _place(env, 0.16, -0.06, *down)
    centre = _place(env, 0.16, 0.0, *down)
    assert left[2] == right[2] == centre[2] == 1.0        # all three are seen
    assert left[0] > 0.05 and right[0] < -0.05            # LEFT positive, RIGHT negative
    assert left[0] == pytest.approx(-right[0], abs=0.02)  # and symmetric about the nose
    assert abs(centre[0]) < 0.02
    # The range slot is the ground range, normalized, and monotone in it.
    near = _place(env, 0.08, -0.02, *down)
    far = _place(env, 0.20, -0.02, *down)
    assert 0.0 < near[1] < far[1] < 1.0
    assert far[1] == pytest.approx(math.hypot(0.20, 0.02) / LM_RANGE_SCALE, abs=0.02)
    # Fresh sighting: seen and full confidence.
    assert far[3] == pytest.approx(1.0)


def test_a_ball_out_of_frame_reads_seen_zero_and_the_estimate_fades():
    """`seen` is the DETECTOR's, not the truth's. A level head cannot see a
    ball at its own feet (the camera sits 0.21 m above it; 12k), so the slot
    must say so rather than leaking the truth."""
    env = _env("right")
    blind = _place(env, 0.10, -0.06, 0.0, 0.0)            # level gaze: out of frame below
    assert blind[2] == 0.0 and blind[3] == 0.0            # not seen, nothing known...
    assert blind[0] == 0.0 and blind[1] == 0.0            # ...and the position slots say nothing
    # Seen, then hidden: the estimate is HELD (odometry) and its confidence fades.
    _place(env, 0.10, -0.06, -0.30, 0.60)
    assert env.head_cmd[2] == 1.0 and env.head_cmd[3] == pytest.approx(1.0)
    held = float(env.head_cmd[0])
    env.data.qpos[env.joint_qpos_adr[6]] = C.DEFAULT_POSE[6]      # look up: the ball leaves the frame
    env.data.qpos[env.joint_qpos_adr[5]] = C.DEFAULT_POSE[5]
    mujoco.mj_forward(env.model, env.data)
    for _ in range(50):                                   # 1 s of detector reports with nothing in them
        env.step_count += 1
        _lm_sense(env)
    assert env.head_cmd[2] == 0.0                                  # the detector says nothing
    assert 0.0 < env.head_cmd[3] < 0.5                             # confidence has faded (tau 1 s)
    assert env.head_cmd[0] == pytest.approx(held, abs=0.02)        # the estimate is still there


def test_the_slots_reach_the_observation_and_nothing_else_moves():
    """The 61-obs layout is untouched: the four head slots carry the ball and
    every other block is the contract's."""
    env = _env("right", **dict(BEHAVIORS["kick_right_sensed"].curriculum[1].env))
    obs, _ = env.reset(seed=5)
    assert obs.shape == (C.OBS_DIM,)
    assert np.allclose(obs[51:55], env.head_cmd)
    assert np.allclose(obs[48:51], 0.0)                   # a trick: the twist command stays zero
    assert obs[53] == 1.0                                 # stage 1 spawns with the ball in frame
    for _ in range(10):
        obs, _, _, _, _ = env.step(np.zeros(14, np.float32))
        assert np.allclose(obs[51:55], env.head_cmd)


def test_the_approach_rung_spawns_a_ball_no_swing_can_reach_and_only_then():
    """The 12as follow-up, and the whole of it: the first cut kicked a ball it
    could see and never walked to one, because no rollout ever contained a
    ball worth walking to. This rung puts one there — and every rung before it
    must be untouched, RNG stream included, or the seed-2 replicate and the
    benches stop measuring the same world."""
    for side, sgn in (("right", -1.0), ("left", 1.0)):
        b = BEHAVIORS[f"kick_{side}_sensed"]
        env = _env(side, **dict(b.curriculum[3].env))
        assert env.max_steps == round(4.0 / C.CTRL_DT)     # the clip knob reaches the env
        far, near, sides = 0, 0, []
        for k in range(120):
            env.reset(seed=200 + k)
            _, qadr, _ = _kick_ball_ids(env)
            ahead = float(env.data.qpos[qadr] - env.data.qpos[0])
            beside = sgn * float(env.data.qpos[qadr + 1] - env.data.qpos[1])
            if ahead > KICK_BOX_AHEAD[1] + 1e-6:
                far += 1
                assert 0.20 - 1e-6 <= ahead <= 0.45 + 1e-6
                assert abs(beside) <= 0.13 + 1e-6
                sides.append(beside)
            else:
                near += 1
                assert KICK_BOX_AHEAD[0] - 1e-6 <= ahead <= KICK_BOX_AHEAD[1] + 1e-6
                assert KICK_BOX_SIDE[0] - 1e-6 <= beside <= KICK_BOX_SIDE[1] + 1e-6
        assert far > 30 and near > 30, (far, near)          # both worlds in the rollouts
        assert min(sides) < -0.02 and max(sides) > 0.02     # ...and the far ball is on both sides

        # A far ball is VISIBLE from a level head — the geometry the near box
        # fails (12k: the camera is 0.21 m up, so a ball at its feet is below
        # the frame). That is what makes the approach learnable at all.
        assert _place(env, 0.40, 0.0, 0.0, 0.0)[2] == 1.0


def test_every_rung_before_the_approach_is_the_world_12as_measured():
    """The far spawn is OFF by default and draws no random number when it is,
    so rung 1-3, every bench and every replay see the identical stream. This
    is the test that would catch 'the seed-2 run is not comparable'."""
    for side in ("right", "left"):
        a = _env(side)
        b = _env(side, MICRODUCK_LM_FAR_PROB="0.0")
        for k in range(20):
            a.reset(seed=700 + k)
            b.reset(seed=700 + k)
            _, qa, _ = _kick_ball_ids(a)
            _, qb, _ = _kick_ball_ids(b)
            assert np.allclose(a.data.qpos[qa:qa + 3], b.data.qpos[qb:qb + 3])
            assert np.allclose(a.data.qpos, b.data.qpos)
        assert a.max_steps == round(2.0 / C.CTRL_DT)


def test_the_range_slot_saturates_and_that_is_where_the_approach_stops():
    """The measured ceiling of the approach rung, pinned as a number.

    obs[52] is `min(1, ground range / LM_RANGE_SCALE)`, so for a ball straight
    ahead of the duck EVERY slot is identical past LM_RANGE_SCALE: bearing 0,
    range 1.0, seen 1, conf 1.0. Walking toward such a ball changes nothing
    the policy can observe until it crosses that radius, so there is no
    gradient out there — which is exactly what the trained rung does. The
    approach appears between 0.20 and 0.24 m and dies between 0.24 and 0.26,
    and a second 2M-step arm with the spawns concentrated on 0.25-0.35 m
    moved that cliff by nothing (roadmap 12as follow-up).

    If someone raises LM_RANGE_SCALE, this test is what tells them the
    approach ceiling moved with it — and that every policy trained under the
    old value now reads its range slot wrong.
    """
    assert LM_RANGE_SCALE == pytest.approx(0.25)
    assert LM_FAR_AHEAD[1] > LM_RANGE_SCALE          # the rung spawns past it, deliberately
    env = _env("right")
    down = (-0.30, 0.60)
    inside = _place(env, LM_RANGE_SCALE - 0.03, 0.0, *down)
    just_out = _place(env, LM_RANGE_SCALE + 0.01, 0.0, *down)
    far_out = _place(env, LM_FAR_AHEAD[1], 0.0, *down)
    assert inside[1] < 0.95                                    # informative...
    assert just_out[1] == pytest.approx(1.0)                   # ...then pinned
    assert far_out[1] == pytest.approx(1.0)
    # ...and the two out-of-range balls are INDISTINGUISHABLE, 0.19 m apart.
    assert np.allclose(just_out, far_out, atol=0.02), (just_out, far_out)


# --- the far-range pair: the same recipe, one number different (12as's next cut) ---


def test_the_far_recipe_is_the_sensed_recipe_with_a_wider_range_slot():
    """`kick_<side>_sensed_far` exists to change ONE thing — what obs[52]
    means — so everything else has to be equal by construction, or the pair
    stops being a control for each other.

    The pay, the stages, the spawn knobs, the clip, the scene and the
    asymmetry are all compared term for term and key for key against the
    0.25 m recipe.
    """
    for side in ("right", "left"):
        near, far = BEHAVIORS[f"kick_{side}_sensed"], BEHAVIORS[f"kick_{side}_sensed_far"]
        assert [(t.key, t.weight, t.is_penalty, t.fn.__name__, t.friendly) for t in far.terms] == \
               [(t.key, t.weight, t.is_penalty, t.fn.__name__, t.friendly) for t in near.terms]
        assert far.scene == near.scene and far.episode_s == pytest.approx(near.episode_s)
        assert far.terminate_on_fall == near.terminate_on_fall
        assert far.symmetric is False and far.default_steps == near.default_steps
        assert far.obs_fn is near.obs_fn                       # one sensing code path
        assert far.reset_fn.__name__ == f"_lm_reset_{side}_far"
        # The ladder is the same ladder: same labels, same budgets, same knobs.
        assert [st.label for st in far.curriculum] == [st.label for st in near.curriculum]
        assert [st.steps for st in far.curriculum] == [st.steps for st in near.curriculum]
        for a, b in zip(far.curriculum, near.curriculum):
            assert a.env == b.env, (a.label, a.env, b.env)
            assert set(a.env) <= STAGE_KNOBS, a.env       # ...and the allowlist still holds
            assert a.detail
        assert sum(st.steps for st in far.curriculum) == far.default_steps
        # Asked for by name — and on the shared handle the LONGER phrase wins
        # the phrase it contains, so "last metre right far" does not land on
        # the 0.25 m recipe. (Bare "kick <side>" prose still belongs to the
        # plain strike, deliberately, as the first sensed test asserts.)
        assert match_behavior(f"kick_{side}_sensed_far").id == far.id
        assert match_behavior(f"last metre {side} far").id == far.id
        assert match_behavior(f"last meter {side} far").id == far.id
        assert match_behavior(f"last metre {side}").id == near.id


def test_the_far_recipe_ranges_a_ball_the_first_one_cannot_tell_apart():
    """The whole cut, as a number. obs[52] is `min(1, range / scale)`; under
    0.25 m a ball 0.30 m away reads a pinned 1.00 and so does one at 0.45,
    which is why the approach died there (12as follow-up B). Under 0.60 m the
    same ball reads 0.5 and the one at 0.45 reads 0.75 — a slot that still
    moves when the duck walks.
    """
    assert LM_RANGE_SCALE_FAR == pytest.approx(0.60)
    # The point of the number: the WHOLE of stage 3's far window is inside it,
    # so there is no spawn the policy cannot range.
    assert LM_RANGE_SCALE_FAR > LM_FAR_AHEAD[1] > LM_RANGE_SCALE
    assert LM_RANGE_SCALE_FAR > math.hypot(LM_FAR_AHEAD[1], LM_FAR_SIDE[1])

    down = (-0.30, 0.60)
    for side in ("right", "left"):
        near_env, far_env = _env(side), _env_id(f"kick_{side}_sensed_far")
        # The scale is stamped on the env by the recipe's own reset — one
        # sensing code path, two meanings, and the env knows which it is.
        near_env.reset(seed=1)
        far_env.reset(seed=1)
        assert near_env._lm_range_scale == pytest.approx(LM_RANGE_SCALE)
        assert far_env._lm_range_scale == pytest.approx(LM_RANGE_SCALE_FAR)
        for ahead in (0.30, 0.45):
            n = _place(near_env, ahead, 0.0, *down)
            f = _place(far_env, ahead, 0.0, *down)
            assert n[1] == pytest.approx(1.0)                      # saturated, as it always was
            assert f[1] == pytest.approx(ahead / LM_RANGE_SCALE_FAR, abs=0.03)
            assert f[1] < 0.95                                     # ...and still informative
            # Everything else in the slot vector is the SAME observation.
            assert f[0] == pytest.approx(n[0], abs=1e-9)
            assert f[2] == n[2] == 1.0
            assert f[3] == pytest.approx(n[3])
        # 0.30 m reads 0.5, not 1.0 — the headline of the cut.
        assert _place(far_env, 0.30, 0.0, *down)[1] == pytest.approx(0.5, abs=0.03)
        # Inside the old radius the two agree on the ORDERING and differ only
        # by the scale factor, so the near game is the same problem.
        for ahead in (0.08, 0.16, 0.22):
            n = _place(near_env, ahead, 0.0, *down)
            f = _place(far_env, ahead, 0.0, *down)
            assert f[1] == pytest.approx(n[1] * LM_RANGE_SCALE / LM_RANGE_SCALE_FAR, abs=0.02)


def test_the_first_recipe_still_reads_its_range_slot_under_0_25():
    """The reason this is a new id and not an edit: every policy in 12as's
    tables was trained against `min(1, range / 0.25)`. If the far recipe's
    scale ever leaked onto the original id, all of them would silently read
    their own range wrong and every table above would still pass.
    """
    down = (-0.30, 0.60)
    for side in ("right", "left"):
        env = _env(side)
        for ahead in (0.08, 0.16, 0.22):
            slot = _place(env, ahead, 0.0, *down)[1]
            assert slot == pytest.approx(math.hypot(ahead, 0.0) / LM_RANGE_SCALE, abs=0.02)
        # ...and it still saturates where it always did.
        assert _place(env, 0.26, 0.0, *down)[1] == pytest.approx(1.0)
        assert np.allclose(_place(env, 0.26, 0.0, *down), _place(env, 0.45, 0.0, *down), atol=0.02)


def test_the_far_recipe_spawns_the_same_worlds_as_the_first_one():
    """Same spawn code, same knobs, same RNG stream: at a given seed the ball
    and the duck land in the same place under both ids, so a grid or an
    approach probe run on one arm is measuring the same world as the other.
    Only the slots the policy reads differ.
    """
    for side in ("right", "left"):
        a, b = _env(side), _env_id(f"kick_{side}_sensed_far")
        for k in range(15):
            a.reset(seed=900 + k)
            b.reset(seed=900 + k)
            assert np.allclose(a.data.qpos, b.data.qpos)
        # ...including the approach rung's mixed spawn.
        st3 = dict(BEHAVIORS[f"kick_{side}_sensed"].curriculum[3].env)
        a, b = _env(side, **st3), _env_id(f"kick_{side}_sensed_far", **st3)
        assert a.max_steps == b.max_steps == round(4.0 / C.CTRL_DT)
        for k in range(15):
            a.reset(seed=950 + k)
            b.reset(seed=950 + k)
            assert np.allclose(a.data.qpos, b.data.qpos)


def test_the_distillation_sees_what_the_recipe_sees_and_shows_the_teacher_zeros():
    """`scripts/distil_kick.py` (12as follow-up (2), route b) rests on two
    facts about this recipe, and both are silent when they break.

    The first is WHERE the ball rides. The script blanks `SENSED_SLOTS` to
    build the BLIND teacher's view of the same step; if that slice ever
    stopped being the slice `_lm_sense` writes, the teacher would be fed the
    ball and the student's targets would come from a policy reading an
    observation no vendored kick was ever trained on — and the clone would
    still fit, and the fit's MSE would still look fine.

    The second is that the collection is SENSED at all. The whole point of
    cloning inside this env rather than the blind one is the `VecNormalize`
    statistics: fitted on a blind rollout the four slots are constant, their
    variance is ~0, and a warm start from it reproduces the exact defect
    ("the strike's VecNormalize was fitted with those four slots carrying
    keep-alive noise") that made a warm start impossible in the first place.
    So the collected slots must actually MOVE.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from distil_kick import SENSED_SLOTS, collect  # noqa: PLC0415

    env = _env("left")
    obs, _ = env.reset(seed=5)
    assert np.allclose(obs[SENSED_SLOTS], env.head_cmd)      # the slice IS the slots
    assert obs.shape[0] == C.OBS_DIM and SENSED_SLOTS == slice(51, 55)

    teacher = Path(__file__).resolve().parents[1] / "policies" / "kick" / "kick_left.onnx"
    if not teacher.exists():                                  # a checkout without the local export
        pytest.skip(f"no teacher at {teacher}")
    ob, act, ret, seen = collect("kick_left_sensed", str(teacher), 12, seed=3)
    assert len(ob) == len(act) == len(ret) > 200
    assert act.shape[1] == 14 and ob.shape[1] == C.OBS_DIM
    # The slots carry a ball that is somewhere, seen sometimes, and fading:
    # every one of the four has to have real spread or the normalizer is the
    # degenerate one this route exists to avoid.
    assert 0.02 < seen < 0.98                                 # the detector reports, and not always
    for k in range(51, 55):
        assert ob[:, k].std() > 0.05, (k, ob[:, k].std())
