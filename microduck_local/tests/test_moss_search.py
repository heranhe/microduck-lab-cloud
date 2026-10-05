"""MOSS's model of the room (`brain/moss_search.py`, `TidyMoss.object_memory`
and `.patrol`): remember what was seen, never chase a one-frame phantom, and
go looking when nothing is in view."""
import dataclasses
import math
import random
from types import SimpleNamespace

import pytest

from microduck_local.brain.moss_search import ObjectMemory, Patrol, area_from_world
from microduck_local.robots import moss
from microduck_local.sensors.detector import Detection, DetectionFrame
from microduck_local.world.scenario import Wall

pytestmark = pytest.mark.skipif(
    not moss.moss_ready(),
    reason="MOSS assets missing — uv run fetch-robot moss")


def test_a_far_object_is_one_memory_not_a_streak():
    """A can at 1.5 m ranged from its apparent width (~10% noise) is ONE
    remembered object near where it is — the first cut's flat 0.15 m gate
    made a streak of five along the line of sight."""
    random.seed(0)
    m = ObjectMemory()
    for k in range(40):
        r = 1.5 * (1 + random.gauss(0, 0.1))
        m.observe([(r, random.gauss(0, 0.02), 0.115, r)], 0.1 * k)
    confirmed = [e for e in m.items if m.confirmed(e)]
    assert len(confirmed) == 1
    assert abs(confirmed[0].x - 1.5) < 0.08


def test_a_phantom_is_never_confirmed_and_is_forgotten():
    m = ObjectMemory()
    m.observe([(0.3, 0.9, 0.05, 0.9)], 0.0)
    assert not m.confirmed(m.items[0])
    m.looked([], set(), m.tentative_s + 0.1)
    assert m.items == []


def test_looking_right_at_a_remembered_spot_and_not_seeing_it_forgets_it():
    m = ObjectMemory()
    for k in range(3):
        m.observe([(0.6, 0.0, 0.1, 0.6)], 0.1 * k)
    e = m.items[0]
    for _ in range(m.miss_frames - 1):
        m.looked([e], set(), 1.0)
    assert m.items == [e]                      # a few missed frames: still there
    m.looked([e], set(), 1.0)
    assert m.items == []                       # looked long enough: gone


def test_the_rim_patrol_is_inset_from_the_yard_walls_and_goes_round():
    yard = SimpleNamespace(scenario=SimpleNamespace(walls=[
        Wall((-1.5, -1.2), (1.5, -1.2)), Wall((1.5, -1.2), (1.5, 1.2)),
        Wall((1.5, 1.2), (-1.5, 1.2)), Wall((-1.5, 1.2), (-1.5, -1.2))]))
    area = area_from_world(yard)
    assert area == (-1.5, 1.5, -1.2, 1.2)
    p = Patrol(area, inset_m=0.45)
    assert all(-1.05 - 1e-9 <= x <= 1.05 + 1e-9 and -0.75 - 1e-9 <= y <= 0.75 + 1e-9
               for x, y in p.waypoints)
    first = p.next_waypoint(-0.9, 0.0)
    assert first == (-1.05, 0.0)               # the nearest, from the spawn
    seen = {first} | {p.next_waypoint(0, 0) for _ in range(7)}
    assert len(seen) == 8                      # every waypoint once round the loop
    assert area_from_world(SimpleNamespace(scenario=SimpleNamespace(walls=[]))) is None


def _frame(t, bearing, rng, name="can0", radius=0.0575):
    w = 2.0 * math.atan(radius / rng)
    return DetectionFrame(t=t, detections=[Detection("toy", name, bearing, -0.05, w, rng, 1.0)])


def _brain(**kw):
    from microduck_local.brain.tidy_moss import TidyMoss
    b = TidyMoss()
    b.p = dataclasses.replace(b.p, **kw)
    return b


def test_with_memory_a_one_frame_phantom_is_not_a_target():
    from microduck_local.brain.runtime import Senses
    b = _brain(object_memory=True)
    b._to("search", 0.0)
    s = Senses(t=0.02, odom=(0.0, 0.0, 0.0), speed=0.0, det=_frame(0.0, 0.2, 0.9, name=""))
    b.step(s)
    assert b.state == "search"                 # seen once: could be a phantom
    b.step(Senses(t=0.04, odom=(0.0, 0.0, 0.0), speed=0.0, det=_frame(0.03, 0.2, 0.9, name="")))
    assert b.state == "approach"               # seen again in the same place: real
    # ...and without memory the first frame is enough (what chased phantoms).
    b = _brain(object_memory=False)
    b._to("search", 0.0)
    b.step(Senses(t=0.02, odom=(0.0, 0.0, 0.0), speed=0.0, det=_frame(0.0, 0.2, 0.9, name="")))
    assert b.state == "approach"


def test_with_nothing_in_view_the_patrol_drives_to_the_rim_after_a_full_turn():
    from microduck_local.brain.runtime import Senses
    yard = SimpleNamespace(scenario=SimpleNamespace(walls=[
        Wall((-1.5, -1.2), (1.5, -1.2)), Wall((1.5, -1.2), (1.5, 1.2)),
        Wall((1.5, 1.2), (-1.5, 1.2)), Wall((-1.5, 1.2), (-1.5, -1.2))]))
    b = _brain(object_memory=True, patrol=True)
    b.world = yard
    b._to("search", 0.0)
    x, y, yaw, t, dt, vx = 0.0, 0.0, 0.0, 0.0, 0.02, 0.0
    drove = False
    for _ in range(int(20 / dt)):
        t += dt
        it = b.step(Senses(t=t, odom=(x, y, yaw), speed=vx))
        vx, _vy, wz = it.twist
        yaw += wz * dt
        x += vx * math.cos(yaw) * dt
        y += vx * math.sin(yaw) * dt
        if vx > 0.05:
            drove = True
            assert t > 2 * math.pi / b.p.search_wz - 0.5    # only after a full turn
            break
    assert drove
    assert b._srch["kind"] == "rim"


# --- the walls from its own depth, and touch (2026-09-28: "is it drawing
# those walls from the actual environment?")

BOX = (-1.5, 1.5, -1.2, 1.2)


def _scan(x, y, yaw, t=0.0, box=BOX, n=88, fov_deg=87.0, mount=(0.156, 0.0), extra=None):
    """A depth row from (x, y, yaw) inside a walled box: each ray's range to
    the first wall (or to `extra`, a disc (cx, cy, r)), as a LidarFrame."""
    import numpy as np

    from microduck_local.sensors.lidar import LidarFrame
    a = np.deg2rad(np.linspace(fov_deg / 2, -fov_deg / 2, n))
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy = x + c * mount[0] - s * mount[1], y + s * mount[0] + c * mount[1]
    rs = []
    for ai in a:
        dx, dy = math.cos(yaw + ai), math.sin(yaw + ai)
        ts = []
        if dx > 1e-9:
            ts.append((box[1] - ox) / dx)
        if dx < -1e-9:
            ts.append((box[0] - ox) / dx)
        if dy > 1e-9:
            ts.append((box[3] - oy) / dy)
        if dy < -1e-9:
            ts.append((box[2] - oy) / dy)
        r = min(ts)
        if extra is not None:
            cx, cy, cr = extra
            px, py = cx - ox, cy - oy
            tc = px * dx + py * dy
            d2 = px * px + py * py - tc * tc
            if tc > 0 and d2 < cr * cr:
                r = min(r, tc - math.sqrt(cr * cr - d2))
        rs.append(r)
    rs = np.array(rs, np.float32)
    return LidarFrame(t=t, ranges=rs, angles=a.astype(np.float32), valid=rs >= 0.52,
                      mount_pos=np.array([mount[0], mount[1], 0.075]))


def test_the_depth_finds_the_walls_once_it_has_looked_round():
    from microduck_local.brain.moss_search import RoomMap
    m = RoomMap((-0.9, 0.0))
    for _ in range(3):
        m.update(_scan(-0.9, 0.0, 0.0), (-0.9, 0.0, 0.0), 6.0)
    assert m.solid().size > 8                   # the far wall is solid...
    assert m.area() is None                     # ...but one way is not a room
    for k in range(1, 12):
        yaw = k * 2 * math.pi / 12
        m.update(_scan(-0.9, 0.0, yaw), (-0.9, 0.0, yaw), 6.0)
    area = m.area()
    assert area is not None
    # To a cell: a return exactly on a grid line lands in the outer cell.
    assert all(abs(a - b) <= m.cell + 1e-9 for a, b in zip(area, BOX))


def test_a_thing_that_moves_is_cleared_by_the_rays_that_pass_where_it_stood():
    from microduck_local.brain.moss_search import RoomMap
    m = RoomMap((0.0, 0.0))
    can = (0.8, 0.0, 0.06)
    for _ in range(3):
        m.update(_scan(0.0, 0.0, 0.0, extra=can), (0.0, 0.0, 0.0), 6.0)
    assert m.blocked(0.8, 0.0, 0.1)
    for _ in range(8):                          # picked: the rays now reach the wall
        m.update(_scan(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6.0)
    assert not m.blocked(0.8, 0.0, 0.1)
    assert m.blocked(1.5, 0.0, 0.1)             # ...which is still there


def _yard():
    return SimpleNamespace(scenario=SimpleNamespace(walls=[
        Wall((-1.5, -1.2), (1.5, -1.2)), Wall((1.5, -1.2), (1.5, 1.2)),
        Wall((1.5, 1.2), (-1.5, 1.2)), Wall((-1.5, 1.2), (-1.5, -1.2))]))


def test_with_depth_the_area_is_its_own_and_without_it_is_given():
    from microduck_local.brain.runtime import Senses
    # Depth, and a world whose walls are somewhere ELSE: the brain must use
    # what it sensed, not what it was handed.
    b = _brain(object_memory=True, patrol=True)
    b.world = SimpleNamespace(scenario=SimpleNamespace(walls=[Wall((-9, -9), (9, 9))]))
    b._to("search", 0.0)
    t = 0.0
    for k in range(40):
        t += 0.1
        yaw = k * 2 * math.pi / 36
        b.step(Senses(t=t, odom=(-0.9, 0.0, yaw), speed=0.0,
                      lidar=_scan(-0.9, 0.0, yaw, t=t), lidar_age=0.0))
    assert b._area_src == "sensed"
    assert all(abs(a - x) <= 0.05 + 1e-9 for a, x in zip(b._patrol.area, BOX))
    assert b.map_payload(t)["room"]["solid"]
    # No depth: after a second of nothing from it, the scenario's walls.
    b = _brain(object_memory=True, patrol=True)
    b.world = _yard()
    b._to("search", 0.0)
    for k in range(60):
        b.step(Senses(t=0.02 * (k + 1), odom=(0.0, 0.0, 0.0), speed=0.0))
    assert b._area_src == "given" and b._patrol.area == BOX


def test_a_patrol_leg_that_stalls_marks_what_it_hit_and_moves_on():
    """Something the depth never saw (a low block) stops a rim leg: the
    robot is commanded forward and makes no progress."""
    from microduck_local.brain.runtime import Senses
    b = _brain(object_memory=True, patrol=True)
    b.world = _yard()
    b._to("search", 0.0)
    x, y, yaw, t, dt, v = 0.0, 0.0, 0.0, 0.0, 0.02, 0.0
    block_x = 0.3
    for _ in range(int(40 / dt)):
        t += dt
        it = b.step(Senses(t=t, odom=(x, y, yaw), speed=v,
                           lidar=_scan(x, y, yaw, t=t), lidar_age=0.0))
        vx, _vy, wz = it.twist
        yaw += wz * dt
        nx, ny = x + vx * math.cos(yaw) * dt, y + vx * math.sin(yaw) * dt
        v = 0.0 if nx > block_x else vx          # the block holds it
        if v:
            x, y = nx, ny
        if b._room is not None and b._room.felt:
            break
    assert b._room is not None and len(b._room.felt) == 1
    fx, fy = b._room.felt[0]
    assert abs(fx - (x + b.p.felt_ahead_m * math.cos(yaw))) < 1e-6
    assert fx > block_x                          # marked in front of it, on the block
    assert b._srch["phase"] == "choose"          # that leg is over
    assert "bumped" in it.note
    assert b._room.blocked(fx, fy, 0.05)


def test_a_rim_waypoint_inside_something_is_skipped():
    p = Patrol(BOX, inset_m=0.45)
    first = p.next_waypoint(-0.9, 0.0, blocked=lambda x, y: (x, y) == (-1.05, 0.0))
    assert first != (-1.05, 0.0)
    assert Patrol(BOX).next_waypoint(0, 0, blocked=lambda x, y: True) is None


def test_moss_yard_mounts_the_realsense_depth_and_it_sees_the_far_wall():
    """The depth row leaves the RealSense frame: from the spawn, facing +x,
    the middle ray meets the far wall's inner face (x = 1.49) 2.234 m from
    the lens (x = -0.9 + 0.156), and no ray returns MOSS itself. A MOSS
    written with `tof: null` has no depth at all."""
    import json
    from pathlib import Path

    import mujoco
    import numpy as np

    from microduck_local.viz_server import load_policy_infer
    from microduck_local.world import scenario as S
    from microduck_local.world_server import WorldState
    raw = json.loads(Path(__file__).resolve().parents[1].joinpath(
        "scenarios/moss-yard.json").read_text())
    sc = S.Scenario.from_dict(raw)
    st = WorldState(load_infer=load_policy_infer)
    w = st.build(sc, seed=0)
    r = w.ducks["m0"]
    assert r.lidar is not None and r.lidar.mount.endswith("moss_camera")
    for _ in range(5):
        w.step()
    f = r.lidar.scan(w.data, w.t)
    assert abs(float(f.angles[0] + f.angles[-1])) < 1e-6      # a fan centred on the lens axis
    assert abs(float(f.angles[0]) - np.deg2rad(43.5)) < 0.01
    mid = int(np.argmin(np.abs(f.angles)))
    assert abs(float(f.ranges[mid]) - (1.49 - (-0.9 + 0.156))) < 0.005
    hit = r.lidar._last_hits
    for g in hit.geomid[hit.geomid >= 0]:
        name = mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_BODY, int(w.model.geom_bodyid[g])) or ""
        assert not name.startswith(r.prefix), name
    raw["ducks"][0]["tof"] = None
    w2 = st.build(S.Scenario.from_dict(raw), seed=0)
    assert w2.ducks["m0"].lidar is None
