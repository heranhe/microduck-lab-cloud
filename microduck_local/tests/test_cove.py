"""Locks for the cove (`Scenario.cove`, `make_pitch(cove=, corner=)`,
`eval-pitch --cove/--corner`): a quarter-round along the base of the boards
so a ball rolling into a wall climbs it and rolls back out, and a 45-degree
chamfer across each corner so nothing wedges.

Why it exists: the physics audit (2026-09-06) found the sim's wall is dead -
MuJoCo's soft contact has no restitution, a 1.4 m/s kick rebounds at e = 0.06
where a real hollow ball is 0.5-0.7 - so a kicked ball sits at the wall it
hits and the ball-out rule (`World.ball_out_s`) is the referee that patches
it. The cove returns the ball by gravity, which MuJoCo does model.
"""

import json

import mujoco
import numpy as np
import pytest

from microduck_local.brain.brain_env import POLICIES_DIR
from microduck_local.world import World, make_pitch
from microduck_local.world.compose import compose
from microduck_local.world.scenario import Scenario, ScenarioError


def _roll(cove: float, v0: float, start_off: float, corner: float = 0.0, target: str = "side",
          seconds: float = 8.0) -> tuple[float, float, float]:
    """Roll the ball at `v0` into a board from `start_off` m off it, on an empty
    2v2 pitch. Returns (distance from the wall line at the end, the closest it
    got, how high it climbed)."""
    sc = make_pitch(per_side=2, cove=cove, corner=corner)
    sc.ducks = []
    m = compose(sc)
    d = mujoco.MjData(m)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball0_free")
    q, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
    r = sc.balls[0].radius
    hx, hy = sc.floor[0] / 2 - 0.25, sc.floor[1] / 2 - 0.25
    if target == "side":                                       # the +y board, at x = 0.3
        pos, vel = [0.3, hy - start_off, r], [0, v0, 0, -v0 / r, 0, 0]
        dist = lambda: hy - float(d.qpos[q + 1])               # noqa: E731
    else:                                                      # the +x,+y corner, along its diagonal
        s = start_off / np.sqrt(2)
        c = v0 / np.sqrt(2)
        pos, vel = [hx - s, hy - s, r], [c, c, 0, -c / r, c / r, 0]
        dist = lambda: min(hx - float(d.qpos[q]), hy - float(d.qpos[q + 1]))   # noqa: E731
    d.qpos[q:q + 7] = pos + [1, 0, 0, 0]
    for _ in range(100):
        mujoco.mj_step(m, d)
    d.qvel[v:v + 6] = vel
    zmax, dmin = 0.0, 9.0
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        zmax = max(zmax, float(d.qpos[q + 2]) - r)
        dmin = min(dmin, dist())
    return dist(), dmin, zmax


def test_a_kicked_ball_dies_at_a_flat_board_and_comes_back_off_a_coved_one():
    end, _, climb = _roll(0.0, 1.0, 0.6)
    assert end < 0.2 and climb < 0.02, (end, climb)             # the audit's dead wall: it sits where it hit
    end, closest, climb = _roll(0.15, 1.0, 0.6)
    assert end > 0.8 and 0.03 < climb < 0.15 and closest < 0.1, (end, closest, climb)


def test_a_dribbled_ball_does_not_rest_flush_against_a_coved_board():
    end, closest, _ = _roll(0.0, 0.2, 0.25)
    assert end < 0.06, end                                      # flat: flush against the wall
    end, closest, _ = _roll(0.15, 0.2, 0.25)
    assert closest < 0.16 and 0.12 < end < 0.45, (end, closest)   # reached the cove's foot, settled off it


def test_the_cove_is_mitred_at_a_square_corner_and_a_chamfer_returns_the_ball_too():
    end, _, _ = _roll(0.0, 1.0, 0.6, target="corner")
    assert end < 0.15, end
    end, _, climb = _roll(0.15, 1.0, 0.6, target="corner")
    assert end > 0.5 and climb > 0.03, (end, climb)             # the two coves meet on the bisector
    end, closest, _ = _roll(0.15, 1.0, 0.6, corner=0.3, target="corner")
    assert end > 0.5 and closest > 0.15, (end, closest)         # the chamfer keeps it out of the corner


def _shoot(w: World, x: float, y: float, vx: float) -> None:
    j = w._ball_joint
    q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
    r = w.scenario.balls[0].radius
    w.data.qpos[q:q + 7] = [x, y, r, 1, 0, 0, 0]
    w.data.qvel[v:v + 6] = [vx, 0, 0, 0, vx / r, 0]
    mujoco.mj_forward(w.model, w.data)


def test_the_goal_mouth_stays_open_and_the_cove_ends_are_the_posts():
    for cove in (0.0, 0.15):
        sc = make_pitch(per_side=2, cove=cove)
        w = World(sc, seed=1)
        hx = sc.floor[0] / 2 - 0.25
        _shoot(w, hx - 0.25, 0.0, 0.3)                          # a slow shot, dead centre of the +x mouth
        for _ in range(int(3.0 / 0.02)):
            w.step()
        assert w.goals["right"] == 1, cove                      # scores on both
    sc = make_pitch(per_side=2, cove=0.15)
    w = World(sc, seed=1)
    _shoot(w, hx - 0.25, sc.goal_width / 2 + 0.1, 0.3)          # the same shot 10 cm outside the post
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    for _ in range(int(3.0 / 0.02)):
        w.step()
    assert w.goals == {"left": 0, "right": 0}
    assert hx - float(w.data.qpos[q]) > 0.1                     # turned back by the cove beside the mouth


def test_a_pitch_with_chamfered_corners_has_eight_walls_that_share_their_endpoints():
    sc = make_pitch(per_side=2)
    assert len(sc.walls) == 4 and sc.cove == 0.0                # the benchmark's pitch, untouched
    sc = make_pitch(per_side=2, cove=0.15, corner=0.3)
    assert len(sc.walls) == 8 and sc.cove == 0.15
    for a, b in zip(sc.walls, sc.walls[1:] + sc.walls[:1]):
        assert a.end == b.start
    hx, hy = sc.floor[0] / 2 - 0.25, sc.floor[1] / 2 - 0.25
    assert sc.walls[0].start == (-hx + 0.3, -hy) and sc.walls[1].end == (hx, -hy + 0.3)


def test_every_lab_pitch_is_coved_and_chamfered_and_the_benchmark_pitch_is_not():
    from microduck_local.world_server import PITCH_CORNER, PITCH_COVE, builtin_scenarios
    builtins = builtin_scenarios()
    for name in ("pitch", "pitch-2v2", "pitch-3v3"):
        sc = builtins[name]
        assert sc.cove == PITCH_COVE == 0.15 and len(sc.walls) == 8, name
        hx = sc.floor[0] / 2 - 0.25
        assert sc.walls[0].start[0] == pytest.approx(-hx + PITCH_CORNER)
        m = compose(sc)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        assert sum(n.startswith("cove") for n in names) == 80, name
    assert builtins["living-room"].cove == 0.0
    assert make_pitch(per_side=2).cove == 0.0 and len(make_pitch(per_side=2).walls) == 4   # eval-pitch's baseline


def test_a_scene_round_trips_its_cove_and_refuses_a_silly_one(tmp_path):
    sc = make_pitch(per_side=2, cove=0.15, corner=0.3)
    p = tmp_path / "coved.json"
    sc.save(p)
    back = Scenario.from_dict(json.loads(p.read_text()))
    assert back.cove == 0.15 and len(back.walls) == 8
    assert Scenario.from_dict({"name": "flat", "floor": {"size": [3, 3]}}).cove == 0.0
    with pytest.raises(ScenarioError, match="cove"):
        Scenario.from_dict({"name": "x", "floor": {"size": [3, 3]}, "cove": 0.6})


def test_a_resume_refuses_rows_measured_on_different_boards(tmp_path):
    from microduck_local.eval_pitch import load_done
    f = tmp_path / "x.jsonl"
    row = {"seed": 0, "tag": "a", "perSide": 2, "seconds": 300.0, "cove": 0.15, "corner": 0.3}
    f.write_text(json.dumps(row) + "\n")
    knobs = {"ballOutS": 0.0, "getupS": 0.0, "cove": 0.15, "corner": 0.3}
    assert set(load_done(str(f), "a", 2, 300.0, knobs)) == {0}
    with pytest.raises(SystemExit, match="cove"):
        load_done(str(f), "a", 2, 300.0, {**knobs, "cove": 0.0})
    with pytest.raises(SystemExit, match="corner"):
        load_done(str(f), "a", 2, 300.0, {**knobs, "corner": 0.0})
    old = {"seed": 1, "tag": "a", "perSide": 2, "seconds": 300.0}      # from before the boards had a shape
    f.write_text(json.dumps(old) + "\n")
    assert set(load_done(str(f), "a", 2, 300.0, {**knobs, "cove": 0.0, "corner": 0.0})) == {1}


@pytest.mark.skipif(not (POLICIES_DIR / "alpha_walking.onnx").exists(), reason="upstream policies not checked out")
def test_the_walkers_flat_floor_trajectory_is_bit_identical_with_a_cove_at_the_boards():
    """The arms differ only at the boards: a duck walking in the open on a
    coved pitch takes exactly the steps it takes on the flat one."""
    from microduck_local.brain.brain_env import onnx_infer
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    trunks = []
    for cove, corner in ((0.0, 0.0), (0.15, 0.3)):
        sc = make_pitch(per_side=2, cove=cove, corner=corner)
        w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=3)
        for d in w.ducks.values():
            d.set_cmd(w.data, (0.15, 0.0, 0.0))
        for _ in range(200):
            w.step()
        trunks.append(np.array([d.trunk_pos(w.data) for d in w.ducks.values()]))
    assert np.array_equal(trunks[0], trunks[1])


@pytest.mark.skipif(not (POLICIES_DIR / "alpha_walking.onnx").exists(), reason="upstream policies not checked out")
def test_an_eval_pitch_row_records_the_boards_it_was_measured_on():
    from microduck_local.eval_pitch import run_one
    row = run_one(0, 1.0, per_side=1, cove=0.15, corner=0.3)
    assert row["cove"] == 0.15 and row["corner"] == 0.3 and row["ballOutS"] == 0.0
