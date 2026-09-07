"""Locks for the scenario contract and world composition (roadmap 0.1/0.3):
a one-duck world carries the same 14-joint contract as the walk scene,
N ducks step in one model, and bad scenarios fail loudly."""

import json
import math

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain.brain_env import POLICIES_DIR
from microduck_local.world import (
    Ball,
    Box,
    Duck,
    Person,
    Scenario,
    Wall,
    compose,
    load_scenario,
    make_room,
    validate_scenario,
)
from microduck_local.world.compose import DuckAddress, spawn_duck
from microduck_local.world.scenario import ScenarioError

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")


def test_single_duck_world_matches_the_walk_contract():
    m = compose(Scenario(name="one", ducks=[Duck("d0", (0, 0, 0))]))
    adr = DuckAddress.resolve(m, "d0")
    ref = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    # Same joint count, same contract order, same actuator order, same limits.
    assert m.nq == ref.nq and m.nv == ref.nv and m.nu == ref.nu
    for k, name in enumerate(C.JOINT_NAMES):
        j = m.joint("d0/" + name)
        r = ref.joint(name)
        assert int(j.qposadr[0]) == int(r.qposadr[0]) == adr.joint_qpos[k]
        np.testing.assert_allclose(j.range, r.range)
        assert m.actuator("d0/" + name).trnid[0] == j.id
    assert m.opt.timestep == C.PHYSICS_DT
    assert adr.tof_site >= 0 and adr.gyro_adr >= 0 and min(adr.foot_geoms) >= 0
    assert m.geom_priority[adr.foot_geoms[0]] == 1


def test_spawn_puts_duck_in_stand_pose_at_rest():
    m = compose(Scenario(name="one", ducks=[Duck("d0", (0.3, -0.2, 1.0))]))
    d = mujoco.MjData(m)
    adr = DuckAddress.resolve(m, "d0")
    spawn_duck(m, d, adr, 0.3, -0.2, 1.0)
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.qpos[adr.joint_qpos], C.DEFAULT_POSE, atol=1e-6)
    np.testing.assert_allclose(d.xpos[adr.trunk_body][:2], [0.3, -0.2], atol=1e-6)
    fwd = d.xmat[adr.trunk_body].reshape(3, 3)[:, 0]
    assert fwd @ [math.cos(1.0), math.sin(1.0), 0] > 0.999
    # Held by the servos alone (no policy), the duck stands for a while and
    # then sags — the SAME behaviour as the reference walk env under a zero
    # action (it terminates at ~1.0 s there too). Lock the shared truth: up
    # at 0.4 s, and the sag matches the reference env to the millimetre.
    z_world = []
    for k in range(int(0.8 / C.PHYSICS_DT)):
        mujoco.mj_step(m, d)
        if (k + 1) % int(0.2 / C.PHYSICS_DT) == 0:
            z_world.append(float(d.xpos[adr.trunk_body][2]))
    assert z_world[1] > 0.10
    from microduck_local.walk_env import MicroduckWalkEnv
    env = MicroduckWalkEnv(obs_noise=False, domain_rand=False, action_delay=False,
                           random_yaw=False, seed=0)
    env.reset(seed=0)
    # The reference reset adds ±0.03 rad pose noise and up to 1 cm of height;
    # re-pose it exactly as spawn_duck did so the two trajectories are comparable.
    env.data.qpos[env.joint_qpos_adr] = C.DEFAULT_POSE
    env.data.qpos[0:3] = [0.3, -0.2, 0.12]
    env.data.qpos[3:7] = [math.cos(0.5), 0, 0, math.sin(0.5)]
    env.data.qvel[:] = 0
    env.data.ctrl[:] = C.DEFAULT_POSE
    mujoco.mj_forward(env.model, env.data)
    z_ref = []
    for k in range(int(0.8 / C.PHYSICS_DT)):
        mujoco.mj_step(env.model, env.data)
        if (k + 1) % int(0.2 / C.PHYSICS_DT) == 0:
            z_ref.append(float(env.data.xpos[env.trunk_body_id][2]))
    np.testing.assert_allclose(z_world, z_ref, atol=1e-3)


def test_three_ducks_and_objects_step_in_one_model():
    sc = Scenario(name="three", floor=(5, 5),
                  walls=[Wall((-2, -2), (2, -2))],
                  boxes=[Box((1, 1, 0.1), (0.2, 0.2, 0.2)), Box((-1, 1, 0.3), (0.1, 0.1, 0.1), mass=0.2)],
                  balls=[Ball((0.6, 0.0))],
                  ducks=[Duck(f"d{i}", (0.0, 0.5 * i, 0.0)) for i in range(3)])
    m = compose(sc)
    d = mujoco.MjData(m)
    adrs = [DuckAddress.resolve(m, f"d{i}") for i in range(3)]
    for i, a in enumerate(adrs):
        spawn_duck(m, d, a, 0.0, 0.5 * i, 0.0)
    mujoco.mj_forward(m, d)
    # 3 free ducks + 1 free box + 1 ball: 5 freejoints + 42 hinges.
    assert m.nq == 5 * 7 + 3 * 14 and m.nu == 3 * 14
    # Distinct, non-overlapping addresses.
    qs = np.concatenate([a.joint_qpos for a in adrs])
    assert len(set(qs.tolist())) == 42
    for _ in range(100):
        mujoco.mj_step(m, d)
    for a in adrs:
        assert d.xpos[a.trunk_body][2] > 0.10
    ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball0")
    assert abs(d.xpos[ball][2] - 0.035) < 0.01     # resting on the floor
    assert m.body_mass[ball] == pytest.approx(0.015)


def test_scenario_roundtrip_and_validation(tmp_path):
    sc = make_room(seed=7, n_boxes=3, n_ducks=2)
    assert len(sc.walls) == 4 and len(sc.ducks) == 2 and sc.name == "room-7"
    assert make_room(seed=7, n_boxes=3, n_ducks=2) == sc      # deterministic
    p = tmp_path / "room.json"
    sc.save(p)
    back = load_scenario(p)
    assert back == sc
    raw = json.loads(p.read_text())
    assert raw["version"] == 1 and raw["walls"][0]["from"] == list(sc.walls[0].start)

    def bad(mutate):
        r = json.loads(p.read_text())
        mutate(r)
        with pytest.raises(ScenarioError):
            validate_scenario(r)

    bad(lambda r: r.update(name="../etc"))
    bad(lambda r: r["ducks"].append({"id": "d0", "spawn": [0, 0, 0]}))   # duplicate
    bad(lambda r: r["ducks"].append({"id": "Bad Id", "spawn": [0, 0, 0]}))
    bad(lambda r: r["ducks"][0].update(tof="lidar"))
    bad(lambda r: r["walls"][0].update(height=float("nan")))
    bad(lambda r: r["boxes"][0].update(size=[0, 0, 0]))
    bad(lambda r: r.update(collision="rollers"))
    bad(lambda r: r.update(floor={"size": [100, 1]}))
    bad(lambda r: r["ducks"].extend({"id": f"x{i}", "spawn": [0, 0, 0]} for i in range(20)))


def test_all_collision_robot_variant_composes():
    m = compose(Scenario(name="all", collision="all", ducks=[Duck("d0", (0, 0, 0))]))
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "d0/mouth_tip") >= 0


def test_a_team_is_a_colorway_and_a_legacy_scene_still_loads():
    """Track 4.2: the two teams used to be called after the two SIDES of the
    pitch, which are the same two words the World writes its goal MOUTHS
    under — the ambiguity `eval_pitch` carries a standing warning about. A
    colorway cannot be confused with a mouth, and it is also the only thing a
    person watching six ducks can read at a glance."""
    from microduck_local.world import make_pitch
    from microduck_local.world.scenario import (
        PITCH_TEAMS,
        TEAM_COLORWAYS,
        ScenarioError,
        validate_scenario,
    )
    home, away = PITCH_TEAMS
    sc = make_pitch(per_side=2)
    assert [d.team for d in sc.ducks] == [home, home, away, away]
    assert all(t in TEAM_COLORWAYS for t in PITCH_TEAMS) and home != away
    raw = sc.to_dict()
    raw["ducks"][0]["team"] = "left"                       # a scene saved before the rename
    assert validate_scenario(raw).ducks[0].team == home
    for field, bad in (("team", "red"), ("role", "goalie")):
        raw = sc.to_dict()
        raw["ducks"][0][field] = bad
        with pytest.raises(ScenarioError, match=field):
            validate_scenario(raw)
    raw = sc.to_dict()
    raw["ducks"][0]["role"] = "striker"                    # a role needs a team to have it on
    raw["ducks"][0]["team"] = None
    with pytest.raises(ScenarioError, match="role but no team"):
        validate_scenario(raw)
    assert all(d.role is None for d in sc.ducks)           # make_pitch default is the role-free control


def test_formation_roles_stamp_jobs_by_roster_size_and_eval_pitch_stays_role_free():
    """Lab builtins stamp defender/striker (2v2) and defender/mid/striker (3v3).
    `make_pitch` without `formation` — what `eval-pitch` calls — stays role-free
    so turning roles on inside the battery cannot silently move the baseline."""
    from microduck_local.world import formation_roles, make_pitch
    from microduck_local.world_server import builtin_scenarios
    assert formation_roles(1) == [None]
    assert formation_roles(2) == ["defender", "striker"]
    assert formation_roles(3) == ["defender", "midfielder", "striker"]
    assert all(d.role is None for d in make_pitch(per_side=1).ducks)
    assert all(d.role is None for d in make_pitch(per_side=2).ducks)
    assert all(d.role is None for d in make_pitch(per_side=3).ducks)
    formed2 = make_pitch(name="pitch-2v2", per_side=2, formation=True)
    formed3 = make_pitch(name="pitch-3v3", per_side=3, formation=True)
    home2 = [d.role for d in formed2.ducks if d.team == formed2.ducks[0].team]
    away2 = [d.role for d in formed2.ducks if d.team != formed2.ducks[0].team]
    assert home2 == away2 == ["defender", "striker"]
    home3 = [d.role for d in formed3.ducks if d.team == formed3.ducks[0].team]
    assert home3 == ["defender", "midfielder", "striker"]
    builtins = builtin_scenarios()
    assert [d.role for d in builtins["pitch"].ducks] == [None, None]
    assert [d.role for d in builtins["pitch-2v2"].ducks][:2] == ["defender", "striker"]
    assert [d.role for d in builtins["pitch-3v3"].ducks][:3] == ["defender", "midfielder", "striker"]


def test_a_team_wears_its_colorway_in_the_compiled_model():
    """`MjSpec.attach` prefixes materials per duck, so a colorway is a write to
    THAT duck's shells and beak and nobody else's. Colour is not mass: the
    step-for-step lock against the walk env (tests/test_arena.py) is what says
    this changed nothing about the physics."""
    import mujoco

    from microduck_local.world import make_pitch
    from microduck_local.world.compose import (
        SHELL_MATERIALS,
        TRIM_MATERIALS,
        compose,
        paint_team,
    )
    from microduck_local.world.scenario import PITCH_TEAMS, TEAM_COLORWAYS
    if not C.SCENE_WALK_XML.exists():
        pytest.skip("microduck_rl checkout not found")
    m = compose(make_pitch())

    def rgb(duck, mat):
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"{duck}/{mat}")
        assert mid >= 0, f"{duck}/{mat} is not in the model"
        return tuple(round(float(v), 3) for v in m.mat_rgba[mid][:3])

    for duck, team in (("d0", PITCH_TEAMS[0]), ("d1", PITCH_TEAMS[1])):
        for part, mats in (("shell", SHELL_MATERIALS), ("trim", TRIM_MATERIALS)):
            for mat in mats:
                assert rgb(duck, mat) == tuple(round(v, 3) for v in TEAM_COLORWAYS[team][part])
    assert rgb("d0", "left_shell_material") != rgb("d1", "left_shell_material")
    # The complaint that produced this assert: a duck rendered as a patchwork
    # of five colours, because the printed thigh plate, shoe rim, hip and soft
    # mouth carry CAD-export colours no colorway claimed. Every printed part
    # lands on ONE of the colorway's two colours — a leg is the body colour,
    # not a second shade of it, and a sole is the shoe colour.
    for duck in ("d0", "d1"):
        assert len({rgb(duck, mat) for mat in SHELL_MATERIALS}) == 1
        assert len({rgb(duck, mat) for mat in TRIM_MATERIALS}) == 1
        assert rgb(duck, "leg_material") == rgb(duck, "left_shell_material")
        assert rgb(duck, "sole_left_material") == rgb(duck, "foot_left_material")
    # Every named material is really in the model: 0 would mean an upstream CAD
    # re-export moved the names and the paint silently did nothing.
    assert paint_team(m, "d0", "lavender") == len(SHELL_MATERIALS) + len(TRIM_MATERIALS)
    assert paint_team(m, "d0", "puce") == 0
    assert rgb("d1", "left_shell_material") == tuple(
        round(v, 3) for v in TEAM_COLORWAYS[PITCH_TEAMS[1]]["shell"])


def test_a_team_facing_both_goals_is_refused_at_validation_not_at_load():
    """It used to raise out of `PitchMetrics` — after the world had been swapped
    in, so /sim answered 500 and went on streaming the previous world's score.
    The editor's save is the right place to say no."""
    from microduck_local.world import World, make_pitch
    from microduck_local.world.scenario import PITCH_TEAMS, ScenarioError, validate_scenario
    sc = make_pitch(per_side=2)
    raw = sc.to_dict()
    raw["ducks"][1]["spawn"] = [-0.9, 0.5, math.pi]        # a home duck facing its own goal
    with pytest.raises(ScenarioError, match="faces both goals"):
        validate_scenario(raw)
    # Declaring the mouth is how a roster like that is legal — and then the
    # World hands BOTH home ducks the same goal, whatever way they face.
    raw["attacks"] = {PITCH_TEAMS[0]: "right"}
    sc2 = validate_scenario(raw)
    w = World(sc2)
    hx = sc2.floor[0] / 2 - 0.25
    assert w.goal_for(w.ducks["d0"]) == (hx, 0.0)
    assert w.goal_for(w.ducks["d1"]) == (hx, 0.0)          # …the one it is turned away from
    assert w.goal_for(w.ducks["d2"]) == (-hx, 0.0)


def test_pitch_counts_goals_and_recentres_the_ball():
    """Soccer, first form: a ball across a short wall's line inside the goal
    width is a goal for that side; the ball comes back to the centre."""
    import mujoco

    from microduck_local.world import World, make_pitch
    from microduck_local.world.scenario import validate_scenario
    sc = make_pitch()
    assert validate_scenario(sc.to_dict()) == sc and sc.goal_width == 0.7
    w = World(sc)
    assert w.soccer_score() == {"left": 0, "right": 0, "ball": [0.0, 0.0], "lastGoal": None, "kickoff": 0.0, "kicked": 0, "bumped": 0,
                                "state": "playing", "kickoffTeam": None}
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    hx = sc.floor[0] / 2 - 0.25
    w.data.qpos[q:q + 2] = [hx - 0.03, 0.1]                  # on the right goal line, inside the posts
    mujoco.mj_forward(w.model, w.data)
    w.step()
    s = w.soccer_score()
    assert s["left"] == 0 and s["right"] == 1 and abs(s["ball"][0]) < 0.06 and abs(s["ball"][1]) < 0.06
    assert s["lastGoal"] == "right" and w.goal_seq == 1
    assert s["kicked"] == 0 and s["bumped"] == 1                # no kick ran: walked in
    w.kickoff_until = -1.0                                   # skip the hold for the second probe
    w.data.qpos[q:q + 2] = [-(hx - 0.03), 0.6]               # left line but outside the posts: no goal
    mujoco.mj_forward(w.model, w.data)
    w.step()
    assert w.soccer_score()["left"] == 0 and w.goal_seq == 1


def test_a_goal_restarts_play_from_a_kickoff():
    """After a goal every duck is back on its spawn, standing on a zero
    command for the hold however hard its brain pushes, and play resumes
    when the hold ends; the ball sits on the centre spot within its nudge."""
    import mujoco

    from microduck_local.world import World, make_pitch
    sc = make_pitch()
    w = World(sc, seed=3)
    d0, d1 = (w.ducks[d.id] for d in sc.ducks)
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    hx = sc.floor[0] / 2 - 0.25
    # Walk both ducks somewhere else first.
    for d in (d0, d1):
        x, y, yaw = d.spawn
        from microduck_local.world.compose import spawn_duck
        spawn_duck(w.model, w.data, d.adr, x + 0.4, y - 0.3, yaw + 1.0)
    w.data.qpos[q:q + 2] = [-(hx - 0.03), 0.0]
    mujoco.mj_forward(w.model, w.data)
    w.step()
    assert w.soccer_score()["left"] == 1 and w.in_kickoff
    for d in (d0, d1):
        pos = d.trunk_pos(w.data)
        assert abs(pos[0] - d.spawn[0]) < 0.03 and abs(pos[1] - d.spawn[1]) < 0.03
        assert abs(w.odom(d)[0] - d.spawn[0]) < 1e-6          # the odometry frame is the pitch again
    bx, by = w.soccer_score()["ball"]
    assert abs(bx) <= 0.05 and abs(by) <= 0.05
    # The hold: the walker sees a zero twist even though the caller asks for full speed.
    n = 0
    while w.in_kickoff:
        d0.set_cmd(w.data, (0.6, 0.0, 0.0))
        w.step()
        n += 1
        assert float(d0.twist_cmd[0]) == 0.0
    assert abs(n * C.CTRL_DT - w.kickoff_hold_s) <= 2 * C.CTRL_DT and w.goal_seq == 1
    d0.set_cmd(w.data, (0.6, 0.0, 0.0))
    w.step()
    assert not w.in_kickoff and abs(float(d0.twist_cmd[0]) - 0.6) < 1e-6
    # reset() clears the board.
    w.reset()
    assert w.soccer_score()["left"] == 0 and w.soccer_score()["lastGoal"] is None


@pytest.mark.skipif(not (POLICIES_DIR / "ball_kick_left.onnx").exists(), reason="upstream policies not checked out")
def test_shipped_kick_skill_sends_the_ball_flying():
    """The kicks run as a 0.5 s window with an all-zero command, like robotd.
    Measured: a ball 8 cm ahead and 6 cm to the foot's side flies over a
    metre; on the wrong side it is not touched."""
    import mujoco

    from microduck_local.brain.brain_env import onnx_infer
    from microduck_local.world import Ball, Duck, Scenario, World
    for side, by, expect in (("kick_left", 0.06, 1.0), ("kick_right", -0.06, 1.0), ("kick_right", 0.06, 0.02)):
        sc = Scenario(name="k", floor=(4, 4), ducks=[Duck("d0", (0, 0, 0), None, None, None)], balls=[Ball((0.08, by))])
        w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")}, seed=0)
        d = w.ducks["d0"]
        for _ in range(50):
            d.set_cmd(w.data, [0, 0, 0])
            w.step()
        b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "ball0")
        p0 = w.data.xpos[b].copy()
        assert w.start_skill(d, side) and d.skill == side
        for _ in range(int(1.5 / C.CTRL_DT)):
            w.step()
        moved = float(np.hypot(*(w.data.xpos[b] - p0)[:2]))
        assert d.skill is None and d.falls == 0
        if expect >= 1.0:
            assert moved > expect, (side, by, moved)
        else:
            assert moved < expect, (side, by, moved)


@pytest.mark.skipif(not (POLICIES_DIR / "ball_kick_left.onnx").exists(), reason="upstream policies not checked out")
def test_kick_window_runs_at_the_standing_gain_and_hands_it_back():
    """robotd runs a kick at the standing tuning: the walking Kp softened by
    `standing_gain_ratio` (control.rs). Here the duck's actuators drop to
    that ratio for the 0.5 s window — this duck's only — and come back."""
    from microduck_local.brain.brain_env import onnx_infer
    from microduck_local.world import Ball, Duck, Scenario, World
    from microduck_local.world.arena import KICK_S, STANDING_GAIN_RATIO
    sc = Scenario(name="k2", floor=(4, 4), balls=[Ball((0.08, 0.06))],
                  ducks=[Duck("d0", (0, 0, 0), None, None, None), Duck("d1", (1.5, 0, 0), None, None, None)])
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={"d0": infer, "d1": infer}, seed=0)
    d0, d1 = w.ducks["d0"], w.ducks["d1"]
    kp0 = w.model.actuator_gainprm[d0.adr.actuators, 0].copy()
    kp1 = w.model.actuator_gainprm[d1.adr.actuators, 0].copy()
    assert (kp0 > 0).all()
    for _ in range(50):
        for d in (d0, d1):
            d.set_cmd(w.data, [0, 0, 0])
        w.step()
    assert w.start_skill(d0, "kick_left")
    np.testing.assert_allclose(w.model.actuator_gainprm[d0.adr.actuators, 0], STANDING_GAIN_RATIO * kp0)
    np.testing.assert_allclose(w.model.actuator_biasprm[d0.adr.actuators, 1], -STANDING_GAIN_RATIO * kp0)
    np.testing.assert_array_equal(w.model.actuator_gainprm[d1.adr.actuators, 0], kp1)   # the other duck: untouched
    steps = 0
    while d0.skill is not None:
        w.step()
        steps += 1
    assert abs(steps * C.CTRL_DT - KICK_S) <= C.CTRL_DT + 1e-9
    np.testing.assert_array_equal(w.model.actuator_gainprm[d0.adr.actuators, 0], kp0)
    np.testing.assert_array_equal(w.model.actuator_biasprm[d0.adr.actuators, 1], -kp0)
    # A respawn mid-window (a fall) restores it too.
    assert w.start_skill(d0, "kick_right")
    w.reset_duck("d0")
    assert d0.skill is None and d0.gain_ratio == 1.0
    np.testing.assert_array_equal(w.model.actuator_gainprm[d0.adr.actuators, 0], kp0)


def test_a_polite_person_stops_short_of_a_duck_and_steps_around():
    """`Person.yield_m`: a walker with a duck inside that range on its way
    stands, then after 2.5 s gives the waypoint up; without it the mocap
    capsule walks straight through the duck."""
    from microduck_local.world import World

    def run(yield_m):
        sc = Scenario(name="p", floor=(6, 6), ducks=[Duck("d0", (0.0, 0.0, 0.0))],
                      persons=[Person("p0", (1.0, 0.0), math.pi, path=[(-1.0, 0.0), (1.0, 1.0)], speed=0.5, yield_m=yield_m)])
        assert validate_scenario(sc.to_dict()) == sc
        w = World(sc)
        p = w.persons["p0"]
        nearest, waited = 9.0, 0.0
        for _ in range(int(6.0 / C.CTRL_DT)):
            w.step()
            nearest = min(nearest, math.hypot(p.x, p.y))
            waited = max(waited, p.waiting)
        return nearest, waited, p.yields
    through = run(0.0)
    assert through[0] < 0.1 and through[2] == 0                    # walked through the duck's spot
    polite = run(0.4)
    assert 0.3 < polite[0] < 0.45 and polite[2] == 1               # stopped short, then gave the waypoint up
    assert polite[1] >= 2.4


def test_the_world_senses_a_bump_between_two_ducks_and_the_tof_places_hits_by_the_head_pose():
    """Two ducks stood touching: both are `bumped`; a lone duck is not, and
    the ball does not count. With the head dipped 0.6 rad the ToF's
    body-height clearance still reads the boards, not the floor (the
    unrotated placement read the floor as a wall 0.35 m ahead)."""
    import mujoco

    from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
    from microduck_local.brain.controllers import tof_clearance_3d
    from microduck_local.world import World, make_pitch
    sc = make_pitch(per_side=1)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=0)
    d0, d1 = w.ducks["d0"], w.ducks["d1"]
    for _ in range(30):
        d0.set_cmd(w.data, (0.0, 0.0, 0.0), (0.0, 0.6, 0.0, 0.0))
        w.step()
    assert not w.bumped(d0) and not w.bumped(d1)
    assert tof_clearance_3d(d0.tof.last).min() > 0.8                  # the boards, not the floor under the dipped head
    # Put d1 right against d0: only the FEET collide in the walk scene, so the
    # trunks sit 5 cm apart (feet touching is what a duck-duck fall is).
    q0 = d1.adr.root_qpos
    p0 = d0.trunk_pos(w.data)
    w.data.qpos[q0:q0 + 2] = [p0[0] + 0.05, p0[1]]
    mujoco.mj_forward(w.model, w.data)
    w.step()
    assert w.bumped(d0) and w.bumped(d1)


def test_the_tof_sees_a_ball_at_the_feet_when_the_head_dips():
    """tof_floor_ball: a floor ball 0.3 m ahead of a duck with its head
    dipped 0.6 rad is a cluster of low hits above the floor plane; with no
    ball there, nothing; a level head sees nothing either (it looks over it)."""
    import mujoco

    from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
    from microduck_local.brain.controllers import tof_floor_ball
    from microduck_local.world import World, make_pitch
    sc = make_pitch(per_side=1)
    w = World(sc, infer_for={d.id: onnx_infer(POLICIES_DIR / "alpha_walking.onnx") for d in sc.ducks}, seed=0)
    d = w.ducks["d0"]
    q = int(w.model.jnt_qposadr[w._ball_joint])

    def settle(head, ball_xy):
        w.data.qpos[q:q + 2] = ball_xy
        mujoco.mj_forward(w.model, w.data)
        for _ in range(40):
            d.set_cmd(w.data, (0.0, 0.0, 0.0), head)
            w.step()
        return tof_floor_ball(d.tof.last)

    p0, yaw = d.trunk_pos(w.data), d.yaw(w.data)
    ahead = (p0[0] + 0.30 * math.cos(yaw), p0[1] + 0.30 * math.sin(yaw))
    seen = settle((0.0, 0.6, 0.0, 0.0), ahead)
    assert seen is not None and abs(seen[0]) < 0.25 and 0.2 < seen[1] < 0.4
    assert settle((0.0, 0.6, 0.0, 0.0), (0.0, 1.0)) is None                  # the ball far away: nothing on the floor ahead
    assert settle((0.0, 0.0, 0.0, 0.0), ahead) is None                       # level head: the ToF looks over it
    # Stood 0.35 m from the boards with the head dipped: a wall has hits above the band in the same columns - not a ball.
    hx = sc.floor[0] / 2 - 0.25
    q0 = d.adr.root_qpos
    w.data.qpos[q0:q0 + 2] = [hx - 0.35, 0.0]
    w.data.qpos[q0 + 3:q0 + 7] = [1.0, 0.0, 0.0, 0.0]                        # facing +x, at the right boards
    assert settle((0.0, 0.6, 0.0, 0.0), (0.0, 1.0)) is None


def test_the_chase_brain_tracks_a_ball_the_tof_sees_at_its_feet():
    """`tof_ball_m`: with no ball in the camera frame, the ToF's floor blob
    becomes a ball sighting for the tracker; off, the brain has no ball."""
    import mujoco

    from microduck_local.brain import Senses
    from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
    from microduck_local.brain.controllers import Chase, ChaseParams
    from microduck_local.world import World, make_pitch
    sc = make_pitch(per_side=1)
    w = World(sc, infer_for={d.id: onnx_infer(POLICIES_DIR / "alpha_walking.onnx") for d in sc.ducks}, seed=1)
    d = w.ducks["d0"]
    q = int(w.model.jnt_qposadr[w._ball_joint])
    p0, yaw = d.trunk_pos(w.data), d.yaw(w.data)
    w.data.qpos[q:q + 2] = [p0[0] + 0.28 * math.cos(yaw), p0[1] + 0.28 * math.sin(yaw)]
    mujoco.mj_forward(w.model, w.data)
    for _ in range(40):
        d.set_cmd(w.data, (0.0, 0.0, 0.0), (0.0, 0.6, 0.0, 0.0))
        w.step()
    tof = d.tof.last
    senses = Senses(t=w.t, tof=tof, tof_age=w.t - tof.t, det=None, det_age=None, speed=0.0, odom=w.odom(d))
    on = Chase(ChaseParams(tof_ball_m=0.5), goal=(1.5, 0.0))
    on.step(senses)
    assert on.tof_ball is not None and on.tracker.best("ball", w.t, min_hits=1) is not None
    off = Chase(ChaseParams(), goal=(1.5, 0.0))
    off.step(senses)
    assert off.tof_ball is None and off.tracker.best("ball", w.t, min_hits=1) is None


def test_clearance_is_selected_by_bearing_so_a_turned_head_cannot_report_a_wall_beside_it():
    """The ToF is IN THE HEAD: yaw it and the middle columns report whatever
    is off to the side. `tof_clearance_bearings` picks hits by their bearing
    in the body's heading frame instead, so a head turned off the walking
    line reads +inf ahead - honestly blind - where the column version read a
    wall 0.52 m ahead that was really 69 deg off the nose."""
    import mujoco
    import numpy as np

    from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
    from microduck_local.brain.controllers import tof_clearance_3d, tof_clearance_bearings
    from microduck_local.world import World, make_pitch
    sc = make_pitch(per_side=1)
    w = World(sc, infer_for={d.id: onnx_infer(POLICIES_DIR / "alpha_walking.onnx") for d in sc.ducks}, seed=0)
    d = w.ducks["d0"]
    q0 = d.adr.root_qpos
    hx = sc.floor[0] / 2 - 0.25

    def settle(head):                                   # pinned 0.40 m off the boards, facing them
        w.data.qpos[q0:q0 + 2] = [hx - 0.40, 0.0]
        w.data.qpos[q0 + 3:q0 + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(w.model, w.data)
        for _ in range(60):
            d.set_cmd(w.data, (0.0, 0.0, 0.0), head)
            w.step()
            w.data.qpos[q0:q0 + 2] = [hx - 0.40, 0.0]
            w.data.qpos[q0 + 3:q0 + 7] = [1.0, 0.0, 0.0, 0.0]
        return d.tof.last

    fr = settle((0.0, 0.0, 0.0, 0.0))
    ahead, left, right = tof_clearance_bearings(fr)
    assert 0.25 < ahead < 0.36 and abs(ahead - float(tof_clearance_3d(fr)[3:5].min())) < 0.03   # the old range, kept
    fr = settle((0.0, 0.0, 1.2, 0.0))                   # head 69 deg off the line
    assert float(tof_clearance_3d(fr)[3:5].min()) < 1.1                      # the column version: a "wall" ahead
    assert tof_clearance_bearings(fr) == (np.inf, np.inf, np.inf)            # the truth: nothing is ahead
    # A synthetic frame carries no mount pose: the level-head columns, as before.
    from microduck_local.sensors.tof import TofFrame
    depth = np.full((8, 8), 2000, np.uint16)
    depth[2:5, 0:3] = 150
    syn = TofFrame(t=0.0, depth_mm=depth, valid=np.ones((8, 8), bool))
    a2, l2, r2 = tof_clearance_bearings(syn)
    assert abs(l2 - 0.15) < 1e-9 and abs(a2 - 2.0) < 1e-9 and abs(r2 - 2.0) < 1e-9


def test_a_scene_can_name_a_learned_brain():
    """The inspector could always switch a duck to `learned:follow-v4` live,
    but saving that scene failed validation: the brain field was gated by
    the duck-id pattern, which has no room for the colon or the dashes a
    run name carries. Built-ins never hit it (they are built in Python),
    so the only scenes that could hold a learned brain were the ones you
    could not save."""
    from microduck_local.world.scenario import ScenarioError, validate_scenario
    from microduck_local.world_server import builtin_scenarios
    base = builtin_scenarios()["follow-me"].to_dict()
    assert base["ducks"][0]["brain"] == "learned:follow-v4"      # the built-in itself
    sc = validate_scenario(base)
    assert sc.ducks[0].brain == "learned:follow-v4"
    assert validate_scenario(sc.to_dict()) == sc
    for ok in ("follow", "learned:p-n256-s31", "learned:ab-batch-lr", "learned:x.y_z"):
        base["ducks"][0]["brain"] = ok
        assert validate_scenario(base).ducks[0].brain == ok
    for bad in ("Learned:x", "learned:", "follow me", "learned:../etc", 7):
        base["ducks"][0]["brain"] = bad
        with pytest.raises(ScenarioError):
            validate_scenario(base)
