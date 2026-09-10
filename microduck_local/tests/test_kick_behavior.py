"""The kick trained here (behaviors/kick.py, roadmap item 7 / 4c revisit):
the walk scene grows the kick ball without losing its keyframes, the
recipe spawns the ball on the kicking foot's sweet spot and the head across
the gaze range, a real kick moves the ball under its pay, and the arena
can run a local export in place of the shipped skill."""


import mujoco
import pytest

from microduck_local import contract as C
from microduck_local.behaviors import BEHAVIORS
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.behaviors.kick import (
    BALL_NOISE,
    BALL_OFFSET,
    HEAD_DOWN,
    NECK_DOWN,
    _kick_ball_ids,
    ball_speed_along,
)
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer


def test_the_walk_scene_takes_the_ball_and_keeps_its_keyframes():
    p = C.scene_walk_ball_xml()
    assert p.exists() and (p.parent / "ball.xml").exists() and (p.parent / "scene_walk.xml").is_symlink()
    m = mujoco.MjModel.from_xml_path(str(p))
    assert m.nq == 21 + 7 and m.key("STAND").id >= 0 and m.body("ball").id > 0


def test_the_kick_recipe_spawns_the_ball_on_the_foot_and_the_head_across_the_gaze_range():
    for side, sgn in (("right", -1.0), ("left", 1.0)):
        assert f"kick_{side}" in BEHAVIORS and BEHAVIORS[f"kick_{side}"].scene == "ball"
        env = BehaviorEnv(f"kick_{side}", seed=7, max_episode_s=2.0, domain_rand=False, random_yaw=False)
        necks, heads = [], []
        for k in range(12):
            env.reset(seed=20 + k)
            _, qadr, _ = _kick_ball_ids(env)
            ahead = env.data.qpos[qadr] - env.data.qpos[0]
            beside = env.data.qpos[qadr + 1] - env.data.qpos[1]
            assert abs(ahead - BALL_OFFSET[0]) <= BALL_NOISE + 1e-6 and abs(beside - sgn * BALL_OFFSET[1]) <= BALL_NOISE + 1e-6
            necks.append(env.data.qpos[env.joint_qpos_adr[5]] - C.DEFAULT_POSE[5])
            heads.append(env.data.qpos[env.joint_qpos_adr[6]] - C.DEFAULT_POSE[6])
        assert min(necks) >= NECK_DOWN[0] - 1e-6 and max(necks) <= NECK_DOWN[1] + 1e-6
        assert min(heads) >= HEAD_DOWN[0] - 1e-6 and max(heads) <= HEAD_DOWN[1] + 1e-6
        assert max(heads) > 0.3 and min(necks) < -0.1                     # the gaze poses are really sampled


def test_a_real_kick_moves_the_ball_under_the_recipes_pay():
    env = BehaviorEnv("kick_right", seed=3, max_episode_s=2.0, domain_rand=False, random_yaw=False)
    obs, _ = env.reset(seed=3)
    # The shipped right kick from a LEVEL head (it whiffs head-down - measured, item 7).
    for j in (5, 6):
        env.data.qpos[env.joint_qpos_adr[j]] = C.DEFAULT_POSE[j]
    env.data.ctrl[:] = env.data.qpos[env.joint_qpos_adr]
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    kick = onnx_infer(POLICIES_DIR / "ball_kick_right.onnx")
    _, qadr, _ = _kick_ball_ids(env)
    x0 = float(env.data.qpos[qadr])
    vmax, total = 0.0, 0.0
    for k in range(60):
        obs, r, term, trunc, info = env.step(kick(obs))
        total += r
        vmax = max(vmax, ball_speed_along(env))
        if term or trunc:
            break
    assert vmax > 0.4 and float(env.data.qpos[qadr]) - x0 > 0.3      # measured: 0.65 m/s, +0.68 m
    assert total > 100.0 and not term                                  # paid for it, still standing


def test_the_arena_runs_a_local_export_in_place_of_a_shipped_skill(monkeypatch, tmp_path):
    from microduck_local.world import World, make_pitch
    sc = make_pitch()
    w = World(sc, seed=1)
    d = w.ducks[sc.ducks[0].id]
    monkeypatch.setenv("MICRODUCK_SKILL_KICK_RIGHT", str(POLICIES_DIR / "ball_kick_left.onnx"))   # any 61-obs policy stands in
    assert w.start_skill(d, "kick_right")
    assert d.skill == "kick_right"
    monkeypatch.setenv("MICRODUCK_SKILL_KICK_LEFT", str(tmp_path / "missing.onnx"))
    with pytest.raises(FileNotFoundError, match="MICRODUCK_SKILL_KICK_LEFT"):
        World(sc, seed=1)                                   # a missing override is refused at BUILD, loudly (it used to disable the kick silently)


def test_the_local_kicks_are_the_default_and_their_exits_reach_the_brain(monkeypatch):
    """A kick trained here, vendored under policies/kick, is what the arena
    runs unless MICRODUCK_SKILL_<NAME> says otherwise; the brain's exit
    angles follow the sidecar, unless the command line names them."""
    import json

    from microduck_local.brain.controllers import Chase, ChaseParams
    from microduck_local.brain.team import brain_kwargs
    from microduck_local.world import World, make_pitch
    monkeypatch.delenv("MICRODUCK_SKILL_KICK_RIGHT", raising=False)
    monkeypatch.delenv("MICRODUCK_SKILL_KICK_LEFT", raising=False)
    monkeypatch.delenv("MICRODUCK_CHASE", raising=False)
    p = World.skill_path("kick_right")
    assert p is not None and p.name == "kick_right.onnx" and "policies/kick" in str(p) and p.exists()
    assert World.skill_path("ground_pick").name == "alpha_ground_pick.onnx"        # no local one: the shipped file
    assert World.skill_path("not_a_skill") is None
    side = json.loads(p.with_suffix(".json").read_text())
    assert World.kick_exits() == (json.loads(World.skill_path("kick_left").with_suffix(".json").read_text())["exit_rad"], side["exit_rad"])
    sc = make_pitch()
    w = World(sc, seed=1)
    kw = brain_kwargs(sc.ducks[0], w, {})
    left = json.loads(World.skill_path("kick_left").with_suffix(".json").read_text())
    assert kw["p"].kick_exit_right == side["exit_rad"] == -0.036 and kw["p"].kick_exit_left == left["exit_rad"] == -0.225   # the weight-12 pair, bench 2026-09-10
    assert w.start_skill(w.ducks[sc.ducks[0].id], "kick_right")                     # and it loads
    monkeypatch.setenv("MICRODUCK_CHASE", "kick_exit_left=0.5")                     # the command line is the caller's
    assert Chase(**brain_kwargs(sc.ducks[0], w, {})).p.kick_exit_left == 0.5    # named on the command line: the brain reads it itself
    monkeypatch.delenv("MICRODUCK_CHASE")
    monkeypatch.setenv("MICRODUCK_SKILL_KICK_RIGHT", str(POLICIES_DIR / "ball_kick_right.onnx"))   # an override wins
    assert World.skill_path("kick_right").name == "ball_kick_right.onnx"
    assert World.kick_exits() is None                                               # no sidecar: the brain's defaults
    assert brain_kwargs(sc.ducks[0], w, {}).get("p", ChaseParams()).kick_exit_right == ChaseParams().kick_exit_right


def test_naming_one_exit_on_the_command_line_keeps_the_other_from_its_sidecar(monkeypatch):
    """Until 2026-09-08 a battery that named `kick_exit_left` silently ran the
    shipped default for the RIGHT foot too, 28.7 deg off the local kick's
    sidecar (code review)."""
    from microduck_local.brain.controllers import ChaseParams
    from microduck_local.brain.team import brain_kwargs
    from microduck_local.world import World, make_pitch
    exits = World.kick_exits()
    assert exits is not None, "the local kicks and their sidecars are vendored"
    monkeypatch.setenv("MICRODUCK_CHASE", "kick_exit_left=-0.3")
    sc = make_pitch()
    kw = brain_kwargs(sc.ducks[0], World(sc, seed=1), {})
    p = kw.get("p") or ChaseParams.from_env()
    assert p.kick_exit_left == -0.3 and p.kick_exit_right == exits[1]


def test_a_sidecar_without_an_exit_yet_reads_as_no_exits_and_a_bad_override_raises(tmp_path, monkeypatch):
    import json
    import shutil

    from microduck_local.world import World
    src = World.skill_path("kick_left")
    onnx = tmp_path / "kick_left.onnx"
    shutil.copyfile(src, onnx)
    onnx.with_suffix(".json").write_text(json.dumps({"exit_rad": None, "note": "not measured yet"}))
    monkeypatch.setenv("MICRODUCK_SKILL_KICK_LEFT", str(onnx))
    assert World.kick_exits() is None                            # null: the brain keeps its measured defaults, no crash
    monkeypatch.setenv("MICRODUCK_SKILL_KICK_LEFT", str(tmp_path / "typo.onnx"))
    from microduck_local.world import make_pitch
    with pytest.raises(FileNotFoundError, match="MICRODUCK_SKILL_KICK_LEFT"):
        World(make_pitch(), seed=1)                                # a typo here disabled the kick silently before


def test_the_wide_kick_spawns_the_ball_across_the_box_and_a_stage_narrows_it():
    """Roadmap 12b + 12ab: the same pay as the point-strike recipe plus the
    heading anchor, the ball anywhere in the box play produces, and a stage's
    knobs narrow the box per instance (the lab previews the active stage)."""
    from microduck_local.behaviors.core import _face_home_pen
    from microduck_local.behaviors.kick import KICK_BOX_AHEAD, KICK_BOX_SIDE, KICK_BOX_STAGE1

    def spawns(env, sgn, n=40):
        aheads, sides = [], []
        for k in range(n):
            env.reset(seed=30 + k)
            _, qadr, _ = _kick_ball_ids(env)
            aheads.append(float(env.data.qpos[qadr] - env.data.qpos[0]))
            sides.append(sgn * float(env.data.qpos[qadr + 1] - env.data.qpos[1]))
            assert _face_home_pen(env) == 0.0                       # the line is the one it began on
        return aheads, sides

    for side, sgn in (("right", -1.0), ("left", 1.0)):
        b, narrow = BEHAVIORS[f"kick_{side}_wide"], BEHAVIORS[f"kick_{side}"]
        assert b.scene == "ball" and b.reset_fn.__name__ == f"_kick_reset_{side}_wide"
        assert [(t.key, t.weight) for t in b.terms[:-1]] == [(t.key, t.weight) for t in narrow.terms]   # the same pay...
        assert b.terms[-1].key == "face_line" and b.terms[-1].is_penalty and b.terms[-1].fn is _face_home_pen   # ...plus the line
        assert len(b.curriculum) == 2
        for st in b.curriculum:                                                     # spawn knobs only, never the pay
            assert set(st.env) == {"MICRODUCK_KICK_BOX_AHEAD", "MICRODUCK_KICK_BOX_SIDE"}
        env = BehaviorEnv(f"kick_{side}_wide", seed=7, max_episode_s=2.0, domain_rand=False, random_yaw=False)
        aheads, sides = spawns(env, sgn)
        assert KICK_BOX_AHEAD[0] - 1e-6 <= min(aheads) and max(aheads) <= KICK_BOX_AHEAD[1] + 1e-6
        assert KICK_BOX_SIDE[0] - 1e-6 <= min(sides) and max(sides) <= KICK_BOX_SIDE[1] + 1e-6
        assert max(aheads) - min(aheads) > 0.08 and max(sides) - min(sides) > 0.08      # really spread, not +-1.5 cm
        env = BehaviorEnv(f"kick_{side}_wide", seed=7, max_episode_s=2.0, domain_rand=False, random_yaw=False,
                          spawn_overrides={"MICRODUCK_KICK_BOX_AHEAD": KICK_BOX_STAGE1[0],
                                           "MICRODUCK_KICK_BOX_SIDE": KICK_BOX_STAGE1[1]})
        aheads, sides = spawns(env, sgn)
        assert 0.06 - 1e-6 <= min(aheads) and max(aheads) <= 0.12 + 1e-6
        assert 0.03 - 1e-6 <= min(sides) and max(sides) <= 0.09 + 1e-6
        assert max(aheads) - min(aheads) > 0.03 and max(sides) - min(sides) > 0.03

