"""MOSS: the download, the three lab rewrites, the drive envelope, the room.

Every case here is a statement about somebody ELSE's model — Laurent
Genoud's `moss_robot.xml`, pinned at a sha — so the suite's job is to catch
both a revision that moved under us and a lab rewrite that stopped meaning
what it says. Where a case could pass for the wrong reason it plants the
regression that must break it (AGENTS.md: a test proves nothing until it has
been shown to fail).
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from microduck_local.robots import moss
from microduck_local.robots.moss_drive import MAX_SPIN_RAD_S, MossDriver, clamp_cmd, twist_to_tracks

pytestmark = pytest.mark.skipif(
    not moss.moss_ready(),
    reason="MOSS's assets are not downloaded — `uv run fetch-robot moss`")


# ------------------------------------------------------------ the download

def test_every_downloaded_file_still_hashes_to_the_manifest():
    """The pinned revision, verified on disk rather than at fetch time.

    `fetch` checks each hash on the way in; this checks that what is in the
    cache NOW is still that, which is the difference between "we downloaded
    the right thing once" and "the model these tests measured is the model
    the manifest names".
    """
    d = moss.asset_dir()
    bad = [rel for rel, want in moss.ASSETS if moss._sha256(d / rel) != want]
    assert not bad, f"{bad} no longer hash to robots/moss.ASSETS"


# ------------------------------------------------------- the three rewrites

def test_fresh_asset_fallback_includes_the_arm_camera():
    """An explicit raw XML bypasses optional collision-candidate caches."""
    model = moss.robot_spec(xml=moss.robot_xml_path()).compile()
    assert model.body(moss.ARM_CAMERA_BODY).id > 0


def test_the_lab_variant_has_a_heading_where_his_model_has_a_rail():
    """His base is one X slide; a room needs (x, y, yaw).

    The joint NAMES and their order are `mars.BASE_JOINTS`, because
    `MossDriver` indexes them by that tuple — a body that spelled them
    differently would need a driver of its own for no reason.
    """
    his = mujoco.MjSpec.from_file(str(moss.robot_xml_path())).compile()
    ours = moss.robot_spec().compile()
    assert [his.joint(j).name for j in range(his.njnt)][:1] == ["base_x"]
    assert his.nq == 8, "his model is not the 8-DoF one this rewrite assumes"
    assert [ours.joint(j).name for j in range(3)] == list(moss.BASE_JOINTS)
    assert ours.nq == 10
    x = ours.joint("base_x")
    assert not bool(x.limited), "his 0-0.8 m rail survived into the lab model"


def test_his_base_position_servo_is_gone():
    """A kp-3000 position servo on one of three planar DoFs would hold x
    while the drive steered in y and yaw — a commanded arc coming out as a
    crab back onto his rail. The seven arm/finger servos stay."""
    ours = moss.robot_spec().compile()
    names = [ours.actuator(a).name for a in range(ours.nu)]
    # Five arm servos and ONE gripper: his follower finger lost its actuator
    # with `couple_fingers`, because the real jaw has a single servo.
    assert names == [*moss.ARM_JOINTS, moss.GRIPPER_JOINT], names
    assert "base_x" not in names
    assert moss.FINGER_JOINTS[1] not in names


def test_the_home_keyframe_is_widened_from_his_numbers_not_retyped():
    """The rewrite slices HIS rows; `ARM_HOME` is for readers.

    Planted regression: a keyframe that was retyped from the constant would
    survive his file changing, so the case reads both and requires them equal.
    """
    his = mujoco.MjSpec.from_file(str(moss.robot_xml_path())).compile()
    ours = moss.robot_spec().compile()
    assert ours.nkey == 1 and ours.key(moss.HOME_KEY).id == 0
    # base_x keeps HIS value (9.6e-7 — his own missions start a hair off
    # zero); the two new DoFs spawn at zero. The driver's `spawn` writes all
    # three anyway, so this row only matters to a bare `mj_resetDataKeyframe`.
    # His own value, not retyped — but compared with a tolerance, because
    # `prepare_candidate.py` round-trips the keyframe through a float and
    # writes back more digits than his XML carried: 9.624455271921862e-07
    # against a literal 9.62446e-07. A 5e-13 difference proves the number
    # came from him just as well as bit-equality does, and bit-equality
    # here was really asserting that nobody had re-serialised the file.
    assert ours.key_qpos[0][0] == pytest.approx(his.key_qpos[0][0], abs=1e-11)
    assert list(ours.key_qpos[0][1:3]) == [0.0, 0.0], "the new DoFs spawn at zero"
    # atol 1e-5, not 0: his own `prepare_candidate.py` re-serialises the
    # keyframe at about six significant figures, so four of these seven come
    # back up to 2.02e-6 away from the file it read. That still proves what
    # this test is for — the numbers are HIS, widened, not retyped — because
    # a retyped pose would differ in the second decimal, not the sixth.
    np.testing.assert_allclose(ours.key_qpos[0][3:], his.key_qpos[0][1:],
                               atol=1e-5)
    # six commands now: his trailing base_x goes with the base servo and one
    # finger goes with the coupling
    np.testing.assert_allclose(ours.key_ctrl[0], his.key_ctrl[0][:6],
                               atol=1e-5)
    np.testing.assert_allclose(
        [moss.ARM_HOME[j] for j in (*moss.ARM_JOINTS, moss.GRIPPER_JOINT)],
        his.key_ctrl[0][:6], atol=1e-9)


# ------------------------------------------------------------- the mass

def test_the_rover_carries_the_mass_laurent_weighed_not_the_shipped_one():
    """3.5 kg at his balance point, replacing the shipped 6.138672 kg.

    That figure is an EXPLICIT `<inertial>` in his file (line 86, after the
    arm subtree), frozen by his exporter from a density composite of unmassed
    boxes. Which is why the replacement has to be another explicit inertial:
    editing geom masses would change nothing.

    The planted regression is the model WITHOUT the rewrite: if
    `set_base_inertial` stopped doing anything, the shipped value would still
    be there, so the case measures both and requires them different.
    """
    ours = moss.robot_spec().compile()
    i = ours.body(moss.BASE_BODY).id
    assert float(ours.body_mass[i]) == pytest.approx(moss.BASE_MASS_KG)
    assert sum(kg for _n, kg, _c in moss.BASE_COMPONENTS) == pytest.approx(
        moss.BASE_MASS_KG), "the three weighed components must sum to the whole"
    # The COM is COMPOSED from those three, not asserted at the balance
    # point: that point belongs to the 3.18 kg configuration, and the cover
    # and bin are elsewhere. So it sits behind and above his lumped estimate.
    com = ours.body_ipos[i]
    assert com[0] < moss.BASE_COM_M[0], "the bin should pull the COM rearward"
    assert com[2] > moss.BASE_COM_M[2], "the cover and bin sit above the axle"
    assert abs(com[0]) < 0.02 and abs(com[2] - moss.BASE_COM_M[2]) < 0.02, (
        f"composed COM {com} is implausibly far from the weighed balance point")
    assert float(com[1]) == pytest.approx(0.0), "lateral symmetry is assumed"

    raw = mujoco.MjSpec.from_file(str(moss.robot_xml_path()))
    moss.add_planar_base(raw)
    moss.drop_base_servo(raw)
    moss.rewrite_home_key(raw)
    plain = raw.compile()                      # everything BUT set_base_inertial
    assert float(plain.body_mass[plain.body(moss.BASE_BODY).id]) == pytest.approx(
        moss.SHIPPED_BASE_MASS_KG, abs=1e-6), (
        "the rover no longer compiles to the inertial his file ships — "
        "re-measure SHIPPED_BASE_MASS_KG against moss_robot.xml line 86")


def test_the_com_height_is_a_knob_because_he_asked_for_one():
    """50-100 mm is his sensitivity sweep; the arm stays separate so the
    COMBINED centre of mass still moves with the arm."""
    lo, hi = moss.BASE_COM_Z_SWEEP
    for z in (lo, moss.BASE_COM_M[2], hi):
        spec = moss.load_robot_spec()
        moss.add_planar_base(spec)
        moss.drop_base_servo(spec)
        moss.set_base_inertial(spec, com_z=z)
        moss.rewrite_home_key(spec)
        m = spec.compile()
        # `com_z` moves the BASE component — the part that is an estimate —
        # so the composite follows it without landing exactly on it.
        got = float(m.body_ipos[m.body(moss.BASE_BODY).id][2])
        share = moss.EQUIPPED_BASE_MASS_KG / moss.BASE_MASS_KG
        assert got == pytest.approx(
            z * share + sum(kg * c[2] for n, kg, c in moss.BASE_COMPONENTS
                            if n != "equipped_base") / moss.BASE_MASS_KG,
            abs=1e-6)
    ours = moss.robot_spec().compile()
    arm = sum(float(ours.body_mass[b]) for b in range(ours.nbody)
              if ours.body(b).name not in ("world", moss.BASE_BODY))
    assert arm == pytest.approx(moss.ARM_MASS_KG, abs=1e-3), (
        "the arm links are lumped into the base — his 'keep the arm masses "
        "separate' request is what makes the combined COM move with the arm")


# ------------------------------------------------------------- the drive

def test_the_track_width_is_measured_off_his_geoms_not_his_default():
    """`diffdrive.py` says 0.30 m and the collision tracks say 0.244 m.

    He flagged the first as an uncalibrated default. Using it would make
    every commanded yaw 23% slow, which is the kind of error a brain would
    spend a week attributing to anything but the axle spacing.
    """
    m = moss.robot_spec().compile()
    centres = [float(m.geom(g).pos[1]) for g in moss.TRACK_GEOMS]
    want = (moss.TRACK_CENTRES_M if moss.collision_v04()
            else moss.TRACK_CENTRES_COLLISION_M)
    # RECONCILED on his candidate, which is the default: the collision geoms
    # now sit where the CAD belts do, which is what he asked for on
    # 2026-09-25. Opted out (MICRODUCK_MOSS_COLLISION_V04=0) they are the
    # legacy proxies at 244, and the disagreement is live again — so this
    # asserts the geometry ACTUALLY IN USE rather than one of the two.
    assert abs(centres[1] - centres[0]) == pytest.approx(want, abs=1e-6)
    assert moss.TRACK_CENTRES_M == pytest.approx(0.266)
    assert moss.TRACK_CENTRES_COLLISION_M == pytest.approx(0.244)
    # The one number that is nobody's measurement stays refused either way.
    assert moss.TRACK_CENTRES_M != moss.TRACK_CENTRES_DIFFDRIVE_DEFAULT_M
    if moss.collision_v04():
        assert abs(centres[1] - centres[0]) != pytest.approx(
            moss.TRACK_CENTRES_COLLISION_M, abs=1e-6), (
            "with his candidate adopted the collision geoms must NOT still "
            "be at the legacy 244 mm — that was the discrepancy")


def test_an_unreachable_twist_is_scaled_not_clipped():
    """His rule: both tracks scale together, so the TURN RADIUS survives.

    Clipping each component independently — `mars_drive.clamp_cmd`'s rule,
    right for a holonomic base — would straighten the turn instead of slowing
    it, and the robot would arrive somewhere else entirely.
    """
    vx, wz = 0.5, 3.0
    cvx, cwz = clamp_cmd(vx, wz)
    assert (cvx, cwz) != (vx, wz), "pick a command his tracks cannot deliver"
    assert cvx / cwz == pytest.approx(vx / wz, rel=1e-9), "the turn radius moved"
    left, right = twist_to_tracks(cvx, cwz)
    assert max(abs(left), abs(right)) == pytest.approx(moss.MAX_TRACK_SPEED_MPS)
    # in the envelope, nothing is touched
    assert clamp_cmd(0.3, 1.0) == pytest.approx((0.3, 1.0))
    assert MAX_SPIN_RAD_S == pytest.approx(
        2 * moss.MAX_TRACK_SPEED_MPS / moss.TRACK_CENTRES_M)


def _drive(vx, wz, seconds=3.0):
    m = moss.model()
    d = mujoco.MjData(m)
    drv = MossDriver(m, "")
    drv.spawn(d)
    for _ in range(int(seconds / m.opt.timestep)):
        drv.set_cmd(vx, wz, d.time)
        drv.step(d)
        mujoco.mj_step(m, d)
    return drv, d


def test_it_drives_where_it_is_told():
    """Forward, turn, and an arc — against the twist his tracks can deliver.

    The delivered fraction depends on WHICH COLLISION MODEL is in use, and
    that is physics rather than slack. On his V0.4 hulls the belts rest on
    the floor and the drive pays their residual drag, so it lands 0.869 m of
    a commanded 0.9 and spins 2.881 rad of 3.0 — about 96%. Opted out, the
    legacy proxies sit 4 mm PROUD and the robot touches nothing at all, so it
    tracks almost exactly: 0.894 m and 2.970 rad.

    The tolerance below is tight around whichever of those applies, because
    the failure this test exists to catch is enormous: adopting his geometry
    without `tracks_as_support` travels 3 mm instead of 900.
    """
    on_hulls = moss.collision_v04()
    fwd, spin = (0.869, 2.881) if on_hulls else (0.894, 2.970)
    speed = 0.2915 if on_hulls else 0.300
    # A tracked base cannot strafe, but on the hulls it is on real contacts,
    # so the sideways and yaw slop is larger — still sub-millimetre-ish.
    lat, yaw_slop = (1.5e-3, 5e-3) if on_hulls else (1e-3, 1e-3)

    drv, d = _drive(0.3, 0.0)
    x, y, yaw = drv.pose(d)
    assert x == pytest.approx(fwd, abs=0.02)
    assert abs(y) < lat, f"{y * 1e3:.2f} mm sideways — a tracked base cannot strafe"
    assert abs(yaw) < yaw_slop
    assert drv.velocity(d)[0] == pytest.approx(speed, abs=5e-3)

    drv, d = _drive(0.0, 1.0)
    assert drv.pose(d)[2] == pytest.approx(spin, abs=0.05)
    assert math.hypot(*drv.pose(d)[:2]) < 0.01, "a spin in place moved the robot"

    drv, d = _drive(0.3, 1.0)                  # an arc of radius 0.3 m
    assert math.hypot(*drv.pose(d)[:2]) == pytest.approx(0.6, abs=0.06)


def test_a_stale_command_stops_the_robot():
    """His watchdog, in SIM seconds: a slow host makes a robot late, never
    runaway. The planted half is the first assertion — it has to be MOVING
    before the silence, or the stop means nothing.

    The moving speed is the drive's delivered fraction, not the command: on
    his V0.4 hulls the belts are on the floor and 0.4 m/s arrives as 0.389.
    See `test_it_drives_where_it_is_told`. What must be exact either way is
    the STOP."""
    m = moss.model()
    d = mujoco.MjData(m)
    drv = MossDriver(m, "")
    drv.spawn(d)
    for _ in range(int(1.0 / m.opt.timestep)):
        drv.set_cmd(0.4, 0.0, d.time)
        drv.step(d)
        mujoco.mj_step(m, d)
    moving = 0.389 if moss.collision_v04() else 0.400
    assert drv.velocity(d)[0] == pytest.approx(moving, abs=1e-2)
    for _ in range(int(2.0 / m.opt.timestep)):    # nobody commands
        drv.step(d)
        mujoco.mj_step(m, d)
    assert abs(drv.velocity(d)[0]) < 1e-3


def test_the_hold_reproduces_his_own_settled_keyframe():
    """The driver's equilibrium IS the pose his file ships.

    His `home` keyframe records a commanded `ctrl` row and the `qpos` the arm
    settles to beside it. Running his servos through this driver lands on the
    second to five decimals, which is the strongest available statement that
    the port did not change his robot.
    """
    m = moss.model()
    d = mujoco.MjData(m)
    drv = MossDriver(m, "")
    drv.spawn(d)
    for _ in range(int(2.0 / m.opt.timestep)):
        drv.step(d)
        mujoco.mj_step(m, d)
    settled = np.array([float(d.qpos[m.joint(n).qposadr[0]])
                        for n in moss.JOINT_NAMES])
    # The ARM still lands on his shipped settled row EXACTLY — five joints,
    # 0.00000 rad of difference, through a V0.4 mesh swap and a gripper
    # re-plumb. That is the claim this case exists to make.
    np.testing.assert_allclose(settled[:5], m.key_qpos[0][3:8], atol=2e-5)
    # The two JAWS sit 0.13-0.16 mm below it, and that is the coupling rather
    # than a drift: one servo now pulls both, so the leader carries the
    # follower's stiction as well as its own.
    for i in (5, 6):
        assert abs(settled[i] - m.key_qpos[0][3 + i]) < 3e-4


# -------------------------------------------------------------- the room

def test_a_moss_room_runs_his_solver_and_a_duck_room_does_not():
    """"Its own environment" (2026-09-24), as a compiled model.

    `MjSpec.attach` keeps the PARENT's `<option>`, so a robot attached into a
    default scene silently loses the integrator and the cone its contacts
    were tuned under. A MOSS room adopts them through the body's own hook;
    a duck room must be untouched, because the duck's composed world is a
    golden bit (`tests/test_arena.py`).
    """
    from microduck_local.world import scenario as S
    from microduck_local.world.compose import compose

    sc = S.Scenario.from_dict(json.loads(
        Path(__file__).resolve().parents[1].joinpath(
            "scenarios/moss-yard.json").read_text()))
    m = compose(sc)
    m = m[0] if isinstance(m, tuple) else m
    assert m.opt.timestep == pytest.approx(moss.GRASP_PHYSICS_DT)
    assert int(m.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    assert int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    assert m.opt.iterations == moss.SOLVER_OPTIONS["iterations"]

    duck = S.make_room(seed=0)
    dm = compose(duck)
    dm = dm[0] if isinstance(dm, tuple) else dm
    assert int(dm.opt.cone) != int(mujoco.mjtCone.mjCONE_ELLIPTIC), (
        "a duck room picked up MOSS's cone — the hook is not per-room")


def test_one_moss_drives_in_a_composed_room():
    """The whole stack: scenario -> compose -> arena -> driver.

    MEASURED: from (-0.9, 0) at 0.4 m/s for 3 s it reaches x = 0.29, turns
    1.2 rad/s for 1.5 s to yaw 1.77, and drives into the north wall rather
    than through it.
    """
    from microduck_local.world import scenario as S
    from microduck_local.world.arena import World

    sc = S.Scenario.from_dict(json.loads(
        Path(__file__).resolve().parents[1].joinpath(
            "scenarios/moss-yard.json").read_text()))
    w = World(sc)
    r = w.ducks["m0"]

    def tick(vx, wz, seconds):
        for _ in range(int(seconds / 0.02)):
            r.driver.set_cmd(vx, wz, w.data.time)
            w.step()

    assert tuple(np.round(r.trunk_pos(w.data)[:2], 3)) == (-0.9, 0.0)
    tick(0.4, 0.0, 3.0)
    assert r.trunk_pos(w.data)[0] == pytest.approx(0.29, abs=0.05)
    tick(0.0, 1.2, 1.5)
    assert r.yaw(w.data) == pytest.approx(1.77, abs=0.05)
    tick(0.5, 0.0, 4.0)
    assert r.trunk_pos(w.data)[1] < 1.2, "it drove through the wall"
    assert r.falls == 0, "a planar base cannot fall"


# --------------------------------------------------------------- the eye

def test_the_camera_is_a_massless_frame_up_front():
    """His MJCF has no camera and the real rover does — one is mounted here.

    Massless and geomless, so the dynamics are untouched: the same total mass
    with and without it. The POSE is provisional (one sentence of his build
    log), which is why the case pins the frame's existence and its neutrality
    rather than a position nobody has measured on the robot.
    """
    ours = moss.robot_spec().compile()
    cam = ours.body(moss.CAMERA_BODY)
    assert float(ours.body_mass[cam.id]) == 0.0
    assert ours.body_parentid[cam.id] == ours.body(moss.BASE_BODY).id
    np.testing.assert_allclose(ours.body_pos[cam.id], moss.CAMERA_POS, atol=1e-9)
    assert float(ours.body_mass.sum()) == pytest.approx(
        moss.BASE_MASS_KG + moss.ARM_MASS_KG, abs=1e-3), (
        "the camera frame changed the robot's mass — it must be a frame")


def _geom_points(m, d, gi):
    """A geom's vertices (mesh) or corners (box), in world coordinates."""
    import itertools
    R, p = d.geom_xmat[gi].reshape(3, 3), d.geom_xpos[gi]
    if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[gi]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        return m.mesh_vert[a:a + n] @ R.T + p
    return np.array([p + R @ (np.array(c) * m.geom_size[gi])
                     for c in itertools.product((-1, 1), repeat=3)])


def test_the_wrist_camera_sits_where_a_bracket_can_hold_it():
    """The gripper frame's +z is the APPROACH (the tool point is at z -0.014,
    the housing at z -0.098..-0.042). The first mount read it as "up" and put
    the lens 6.5 cm past the jaw tips — under the floor at GRASP_POSE, looking
    up at the tool point.

    Lens, tool point and housing are rigid in the gripper frame, so those are
    checked ONCE, in its terms: behind the jaws, 1 cm clear of every geom the
    frame and the fingers carry (jaws shut AND open), tool point in the field.
    What does change with the pose is checked where it changes: off the floor
    at every pose the arm is sent to, and clear of the forearm and wrist over
    the whole wrist_flex x wrist_roll range — a mount behind the housing
    corner saw as much and came within 5 mm of the forearm at the stop."""
    from microduck_local.robots import moss_bin

    m = moss.model()
    d = mujoco.MjData(m)
    g, cam = m.body(moss.GRIPPER_FRAME_BODY).id, m.body(moss.ARM_CAMERA_BODY).id
    tcp = m.site("tcp").id

    def pose(q, fingers=0.0):
        mujoco.mj_resetData(m, d)
        for j, v in zip(moss.ARM_JOINTS, q):
            d.qpos[m.joint(j).qposadr[0]] = v
        for j in moss.FINGER_JOINTS:
            d.qpos[m.joint(j).qposadr[0]] = fingers
        mujoco.mj_forward(m, d)

    def in_grip(x):
        return (x - d.xpos[g]) @ d.xmat[g].reshape(3, 3)

    carried = [m.body(n).id for n in (moss.GRIPPER_FRAME_BODY, *moss.FINGER_JOINTS)]
    for fingers in (0.0, float(m.joint(moss.FINGER_JOINTS[0]).range[1])):
        pose(moss.GRASP_POSE, fingers)
        lens, tool = in_grip(d.xpos[cam]), in_grip(d.site_xpos[tcp])
        assert lens[2] < tool[2] - 0.05, ("lens is not behind the jaws", lens, tool)
        for gi in range(m.ngeom):
            if m.geom_bodyid[gi] not in carried:
                continue
            pts = in_grip(_geom_points(m, d, gi))
            gap = np.maximum(pts.min(0) - lens, 0) + np.maximum(lens - pts.max(0), 0)
            assert np.linalg.norm(gap) >= 0.01, (
                f"lens within 1 cm of {m.geom(gi).name or gi}", lens)
        assert moss_bin.in_view(d.xpos[cam], d.xmat[cam], d.site_xpos[tcp])

    for q in (moss.GRASP_POSE, moss.LIFT_POSE, moss.DROP_POSE, moss.TUCK_POSE_V04):
        pose(q)
        assert d.xpos[cam][2] > 0.05, ("lens at the floor", q, d.xpos[cam])

    arm = [gi for gi in range(m.ngeom) if m.geom_group[gi] == 2
           and m.geom_bodyid[gi] in {m.body(n).id for n in
                                     ("upper_arm_link", "lower_arm_link", "wrist_link")}]
    flex, roll = m.joint("wrist_flex").range, m.joint("wrist_roll").range
    worst = 9.0
    for qf in np.linspace(*flex, 13):
        for qr in np.linspace(*roll, 13):
            q = list(moss.GRASP_POSE)
            q[3], q[4] = qf, qr
            pose(q)
            pts = np.concatenate([_geom_points(m, d, gi) for gi in arm])
            worst = min(worst, float(np.linalg.norm(pts - d.xpos[cam], axis=1).min()))
    assert worst >= 0.015, f"lens comes within {worst * 100:.1f} cm of the forearm"


def test_the_wrist_depth_fix_says_where_the_object_sits_in_the_jaws():
    """A range cannot see grip depth (122 yard lifts: 120-124 mm median deep
    or shallow, corr -0.1); a depth camera's 3-D fix can. `grip` is the
    object in the TOOL frame, with the range's noise — checked here against a
    can posed a known distance along the approach, at the grasp pose, where
    the tool frame is nothing like the world's."""
    from microduck_local.robots import moss_env as ME
    from microduck_local.robots.moss_wrist import MossTargetSensors, _Geom

    env = ME.MossPickEnv(seed=0)
    env.reset(seed=0)
    m, d = env.model, env.data
    tcp = d.site_xpos[env.tcp_site].copy()
    R = d.site_xmat[env.tcp_site].reshape(3, 3).copy()
    # along the approach AND sideways along the tool's x, which the shoulder
    # pan has turned well away from the world's x
    d.qpos[env.can_qadr:env.can_qadr + 3] = tcp + R @ np.array([0.03, 0.0, 0.03])
    mujoco.mj_forward(m, d)
    want = R.T @ (d.xpos[env.can_body] - d.site_xpos[env.tcp_site])
    world = d.xpos[env.can_body] - d.site_xpos[env.tcp_site]
    assert np.linalg.norm(world - want) > 3 * ME.ARM_DET_RANGE_NOISE  # frames differ

    ts = MossTargetSensors(m, "", seed=1)
    got = []
    for k in range(60):
        d.time = 0.02 * k
        ts.tick(d, env.driver, [(env.can_body, _Geom("cylinder", 0.033, 0.0575))])
        fix = ts.read()["grip"]
        if fix is not None and (not got or fix != got[-1]):
            got.append(fix)
    assert len(got) >= 8                              # 15 Hz over 1.2 s
    mean = np.mean(got, axis=0)
    assert np.linalg.norm(mean - want) < 3 * ME.ARM_DET_RANGE_NOISE, (mean, want)


def test_the_grip_reaches_the_pick_the_same_way_in_env_and_brain():
    """`publish_grip` puts the depth camera's grip fix in `moss.OBS_GRIP`
    (the duplicate second-jaw slots): (distance from the tool point, 1) when
    fresh, (GRIP_UNSEEN_M, 0) when not — in the env AND in tidy_moss, or a
    leg trained on one is fed something else by the other."""
    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss
    from microduck_local.robots import moss_env as ME

    env = ME.MossPickEnv(seed=0, publish_grip=True)
    o, _ = env.reset(seed=0)
    assert tuple(o[list(moss.OBS_GRIP)]) == (ME.GRIP_UNSEEN_M, 0.0)
    env._grip, env._grip_t = np.array([0.0, 0.03, 0.04]), float(env.data.time)
    o = env._obs()
    assert o[moss.OBS_GRIP[0]] == pytest.approx(0.05) and o[moss.OBS_GRIP[1]] == 1.0
    plain = ME.MossPickEnv(seed=0)
    o2, _ = plain.reset(seed=0)
    assert o2[6] == pytest.approx(o2[5], abs=2e-3)      # near-duplicates, as before

    b = TidyMoss()
    b._flags["pick"] = {"publish_grip": True}
    arm = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
    arm[moss.GRIPPER_JOINT] = 0.03
    sn = Senses(t=0.0, odom=(0.0, 0.0, 0.0), speed=0.0, arm=arm,
                target_obs={"grip": (0.0, 0.03, 0.04)})
    ob = b._policy_obs(sn, (0.3, 0.0), "pick")
    assert ob[moss.OBS_GRIP[0]] == pytest.approx(0.05) and ob[moss.OBS_GRIP[1]] == 1.0
    sn = Senses(t=0.0, odom=(0.0, 0.0, 0.0), speed=0.0, arm=arm, target_obs={})
    ob = b._policy_obs(sn, (0.3, 0.0), "pick")
    assert tuple(ob[list(moss.OBS_GRIP)]) == (ME.GRIP_UNSEEN_M, 0.0)


def test_a_handover_state_episode_starts_where_the_yard_handed_over():
    """The yard's fingertip grips come from the STATES it hands the pick
    (9 of 24 settled grips from them, 3 of 43 from the env's spawns), so
    `handover_states` starts an episode in one: the arm's joints, the target
    object's pose and shape, and a fresh fix of it — a replay that started
    with an empty tracker did not reproduce the yard."""
    import json as _json

    from microduck_local.robots import moss_env as ME

    bank = _json.loads(ME.HANDOVER_STATES_FILE.read_text())
    assert len(bank) >= 500 and {b["seed"] for b in bank}.isdisjoint(range(1, 13))
    env = ME.MossPickEnv(seed=3, handover_states=True, publish_grip=True)
    o, _ = env.reset(seed=3)
    hs = env._hs
    assert hs is not None and env.prop.shape == hs["shape"]
    m, d = env.model, env.data
    for j, v in zip(moss.ARM_JOINTS, hs["arm"]):
        assert d.qpos[m.joint(j).qposadr[0]] == pytest.approx(v, abs=0.02), j
    bx, by, byaw = env.driver.pose(d)
    c, s_ = np.cos(-byaw), np.sin(-byaw)
    dx, dy = d.xpos[env.can_body][0] - bx, d.xpos[env.can_body][1] - by
    got = (dx * c - dy * s_, dx * s_ + dy * c)
    assert np.hypot(got[0] - hs["pos_b"][0], got[1] - hs["pos_b"][1]) < 0.01
    assert o[moss.OBS_TARGET_SEEN.start] == 1.0          # it starts with a fix


def test_a_warm_start_passes_the_repurposed_grip_slots_through(tmp_path):
    """478dad's normaliser describes the duplicate jaw slots there (mean
    -0.0115, sd 0.013); a grip distance of 0.15 would arrive as z = +12 and
    clip. A run that turns `publish_grip` on over a donor that had it off
    must start those slots as pass-through — and one whose donor had it on
    must not. The flags reach run.json."""
    import json as _json
    from types import SimpleNamespace as NS

    from microduck_local.robots.registry import registry

    body = registry()["moss"]
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "run.json").write_text(_json.dumps({"env_kwargs": {}}))
    assert body.repurposed_obs_dims({"publish_grip": True}, donor) == list(moss.OBS_GRIP)
    assert body.repurposed_obs_dims({"publish_grip": False}, donor) == []
    (donor / "run.json").write_text(_json.dumps({"env_kwargs": {"publish_grip": True}}))
    assert body.repurposed_obs_dims({"publish_grip": True}, donor) == []
    import os as _os
    _os.environ["MICRODUCK_MOSS_GRIP_OBS"] = "1"
    _os.environ["MICRODUCK_MOSS_HANDOVER_STATES"] = "1"
    try:
        kw = body.train_env_kwargs(NS(task="pick"))
    finally:
        del _os.environ["MICRODUCK_MOSS_GRIP_OBS"]
        del _os.environ["MICRODUCK_MOSS_HANDOVER_STATES"]
    assert kw["publish_grip"] is True and kw["handover_states"] is True


def test_ball_frac_makes_balls_common_and_changes_nothing_at_zero(monkeypatch):
    """0281e5 asked for balls held through the centre with balls in a sixth
    of its episodes and did not learn it; `ball_frac` raises that share. At
    0 it draws nothing extra, so earlier runs replay the same episodes. It is
    recorded in run.json."""
    from types import SimpleNamespace as NS

    from microduck_local.robots.moss_env import MossPickEnv, sample_prop
    from microduck_local.robots.registry import registry

    r0, r1 = np.random.default_rng(9), np.random.default_rng(9)
    assert ([sample_prop(r0).id for _ in range(50)]
            == [sample_prop(r1, ball_frac=0.0).id for _ in range(50)])
    r = np.random.default_rng(1)
    share = np.mean([sample_prop(r, ball_frac=0.5).id == "ball" for _ in range(600)])
    assert 0.5 < share < 0.66, share                   # half + the variety's sixth
    env = MossPickEnv(seed=0, prop_variety=True, ball_frac=0.5)
    assert sum(env.reset(seed=s)[0] is not None and env.prop.id == "ball"
               for s in range(40)) >= 14
    monkeypatch.setenv("MICRODUCK_MOSS_BALL_FRAC", "0.5")
    assert registry()["moss"].train_env_kwargs(NS(task="pick"))["ball_frac"] == 0.5


def test_the_full_grip_reaches_the_pick_the_same_way_in_env_and_brain():
    """`grip_xyz`: the depth fix in the TOOL frame (is the object centred
    between the jaws, how deep) and a fresh flag in `moss.OBS_GRIP_XYZ`, or
    (0, 0, GRIP_UNSEEN_M, 0) — in the env and in tidy_moss alike."""
    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss
    from microduck_local.robots import moss_env as ME

    env = ME.MossPickEnv(seed=0, grip_xyz=True)
    o, _ = env.reset(seed=0)
    assert tuple(o[list(moss.OBS_GRIP_XYZ)]) == (0.0, 0.0, ME.GRIP_UNSEEN_M, 0.0)
    env._grip, env._grip_t = np.array([0.01, -0.02, 0.03]), float(env.data.time)
    assert tuple(np.round(env._obs()[list(moss.OBS_GRIP_XYZ)], 6)) == (0.01, -0.02, 0.03, 1.0)

    b = TidyMoss()
    b._flags["pick"] = {"grip_xyz": True}
    arm = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
    arm[moss.GRIPPER_JOINT] = 0.03
    sn = Senses(t=0.0, odom=(0.0, 0.0, 0.0), speed=0.0, arm=arm,
                target_obs={"grip": (0.01, -0.02, 0.03)})
    ob = b._policy_obs(sn, (0.3, 0.0), "pick")
    assert tuple(np.round(ob[list(moss.OBS_GRIP_XYZ)], 6)) == (0.01, -0.02, 0.03, 1.0)
    sn = Senses(t=0.0, odom=(0.0, 0.0, 0.0), speed=0.0, arm=arm, target_obs={})
    ob = b._policy_obs(sn, (0.3, 0.0), "pick")
    assert tuple(ob[list(moss.OBS_GRIP_XYZ)]) == (0.0, 0.0, ME.GRIP_UNSEEN_M, 0.0)


def test_handover_frac_mixes_yard_starts_with_the_rung_box():
    """Stage 3 of the grip pick starts half its episodes from yard handover
    states and half from the rung's box; at 1.0 every episode is a yard
    state and no extra draw is made (8956a1's episodes replay)."""
    from microduck_local.robots import moss_env as ME

    half = ME.MossPickEnv(seed=0, pick_rung=2, handover_states=True, handover_frac=0.5)
    n = sum(half.reset(seed=s)[0] is not None and half._hs is not None for s in range(60))
    assert 18 <= n <= 42, n
    full = ME.MossPickEnv(seed=0, pick_rung=2, handover_states=True)
    assert all(full.reset(seed=s)[0] is not None and full._hs is not None for s in range(10))


def test_the_brain_gives_a_pick_the_episode_it_trained_in(tmp_path):
    """A pick trained in 15 s episodes, where a missed grasp does not end
    the episode, must not be cut off at 8 s and sent to tuck — that
    redeploy cycle is 12 s a time and half of every yard run. Runs without
    the record keep 8 s."""
    import json as _json

    from microduck_local.brain.tidy_moss import TidyMoss
    from microduck_local.robots import moss_env as ME

    run = tmp_path / "r"
    run.mkdir()
    (run / "run.json").write_text(_json.dumps({"env_kwargs": {"max_episode_s": 15.0}}))
    assert ME.episode_s_of(run / "policy.onnx") == 15.0
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "run.json").write_text(_json.dumps({"env_kwargs": {}}))
    assert ME.episode_s_of(tmp_path / "old") is None
    b = TidyMoss()
    b._flags["pick"] = {"episode_s": 15.0}
    assert b._pick_window_s() == 15.0
    b._flags["pick"] = {"episode_s": None}
    assert b._pick_window_s() == b.p.learned_window_s == 8.0


def test_the_grip_pick_recipe_is_found_and_ladders_into_the_yard(tmp_path):
    """"pick with the depth camera" reaches MOSS_PICK_GRIP, whose last two
    stages start from yard handover states; a warm start that turns
    `grip_xyz` on over a donor without it passes all four slots through."""
    import json as _json

    from microduck_local.behaviors.core import match_behavior
    from microduck_local.robots.registry import registry

    b = match_behavior("pick with the depth camera", "moss")
    assert b is not None and b.id == "moss_pick_grip"
    assert [st.env.get("MICRODUCK_MOSS_HANDOVER_STATES") for st in b.curriculum] == [
        None, None, None, None, "1", "1"]
    # nothing charged until the pick can grab: every penalty on from step
    # one taught it to hover and never close (0/20 at 900k)
    assert all("MICRODUCK_MOSS_TOPPLE" not in st.env for st in b.curriculum[:4])
    assert b.curriculum[0].env["MICRODUCK_MOSS_PROP_VARIETY"] == "0"
    assert match_behavior("pick up the can", "moss").id == "moss_pick"
    donor = tmp_path / "d"
    donor.mkdir()
    (donor / "run.json").write_text(_json.dumps({"env_kwargs": {}}))
    assert registry()["moss"].repurposed_obs_dims({"grip_xyz": True}, donor) == list(
        moss.OBS_GRIP_XYZ)
    # stage 5 turns the proximity input on: slot 26 changes meaning
    assert registry()["moss"].repurposed_obs_dims(
        {"grip_xyz": True, "publish_proximity": True},
        donor) == list(moss.OBS_GRIP_XYZ) + [moss.OBS_TARGET_BASE.start + 2]


def test_every_moss_run_records_the_wrist_mount_it_trained_behind(tmp_path):
    """478dad trained on the retired mount and nothing recorded it, so it was
    scored on the new one with no warning (-4.4 points, 2026-09-27). Every
    task's run.json now says which mount, and a run.json without the key is
    read as the retired one — the truth for every run before the move."""
    import json as _json
    from types import SimpleNamespace as NS

    from microduck_local.robots.moss_env import arm_camera_mount_of, eval_env_kwargs
    from microduck_local.robots.registry import registry

    body = registry()["moss"]
    for task in ("pick", "approach", "stow"):
        got = body.train_env_kwargs(NS(task=task)).get("arm_camera_mount")
        assert got == moss.ARM_CAMERA_MOUNT, (task, got)

    root = tmp_path
    for name, kw, want in (("legacy", {"publish_attitude": True},
                            moss.ARM_CAMERA_MOUNT_LEGACY),
                           ("current", {"arm_camera_mount": moss.ARM_CAMERA_MOUNT},
                            moss.ARM_CAMERA_MOUNT)):
        run = root / name
        run.mkdir()
        (run / "run.json").write_text(_json.dumps({"env_kwargs": kw}))
        assert arm_camera_mount_of(run / "policy.onnx") == want
        assert eval_env_kwargs(run / "policy.onnx")["arm_camera_mount"] == want
    assert arm_camera_mount_of(root / "nowhere" / "policy.onnx") is None
    assert "arm_camera_mount" not in eval_env_kwargs(root / "nowhere" / "policy.onnx")


def test_a_leg_behind_another_mount_is_measured_but_never_silently():
    """Scoring an old leg on the honest sensor is legitimate — it is how the
    cost of the move was measured — but the env says so, every time."""
    import warnings

    from microduck_local.robots.moss_env import MossPickEnv

    old = moss.ARM_CAMERA_MOUNT_LEGACY
    with pytest.warns(UserWarning, match="wrist-camera mount"):
        MossPickEnv(seed=0, arm_camera_mount=old, publish_proximity=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MossPickEnv(seed=0, arm_camera_mount=moss.ARM_CAMERA_MOUNT,
                    publish_proximity=True)
        MossPickEnv(seed=0, publish_attitude=True)
        # a leg that never saw the wrist camera is not changed by moving it
        MossPickEnv(seed=0, arm_camera_mount=old, publish_attitude=False,
                    publish_proximity=False)


def test_a_wrist_camera_on_the_approach_axis_is_refused(monkeypatch):
    """On the axis there is no "away from it" to call the image's up; the
    unguarded version compiled a NaN frame that silently detected nothing."""
    monkeypatch.setattr(moss, "ARM_CAMERA_POS", (-0.008, 0.0, -0.12))
    with pytest.raises(ValueError, match="approach axis"):
        moss.arm_camera_quat()


def test_it_sees_the_cans_in_its_own_yard():
    """The detector on that frame finds every object in the yard, driving a lap.

    MEASURED: 3/3 distinct cans, ~278 of 1000 ticks with one in frame, median
    range 0.40 m, nearest 0.18 m — and the camera PITCH is a null from 0 to
    20 deg (identical counts at 0 and 10, +4 frames at 20, -17 at 30), which
    is why `CAMERA_PITCH_RAD` is level.
    """
    from microduck_local.world import scenario as S
    from microduck_local.world.arena import World

    sc = S.Scenario.from_dict(json.loads(
        Path(__file__).resolve().parents[1].joinpath(
            "scenarios/moss-yard.json").read_text()))
    assert sc.ducks[0].detector, "the yard stopped giving MOSS its camera"
    w = World(sc)
    r = w.ducks["m0"]
    assert r.detector is not None, "no detector was mounted on the body"
    seen = set()
    for leg in ((0.35, 0.0, 3.0), (0.0, 1.0, 1.6)) * 4:
        for _ in range(int(leg[2] / 0.02)):
            r.driver.set_cmd(leg[0], leg[1], w.data.time)
            w.step()
            last = r.detector.last
            for det in (last.detections if last else []):
                if det.cls == "toy":
                    seen.add(det.name)
    # EVERY object the yard holds — three cans until 2026-09-26, when a block,
    # a squat can and a ball joined them — not a hard-coded three.
    want = {p.id for p in sc.props if p.cls == "toy"}
    assert len(want) >= 6, sorted(want)
    assert seen == want, f"saw {sorted(seen)}, expected {sorted(want)}"


def test_the_jaw_window_is_where_the_measurement_put_it():
    """`GRASP_JAW_CTRL_M` sits inside the measured hold band and outside the
    two failures either side of it — the sweep in its own comment."""
    inner_mm = 16.0 + 2.0 * (moss.GRASP_JAW_CTRL_M * 1e3) - 8.0
    squeeze_mm = 66.0 - inner_mm
    assert 2.0 <= squeeze_mm <= 6.0, (
        f"{squeeze_mm:.1f} mm of interference on his can: 10 mm ejected it "
        "and a gap never gripped")
    assert 0.026 <= moss.GRASP_JAW_CTRL_M <= 0.028, (
        "0.024 ejected the can and 0.030 never lifted it")
    assert moss.GRASP_HEIGHT_M < 0.115 / 2, (
        "the grasp height is above the can's midpoint; 0.065 and 0.080 both "
        "failed at every jaw command")


# ------------------------------------------------------------ the tidy brain

def test_the_tidy_brain_ranges_each_object_by_its_own_size():
    """The 230 mm bug, and the 200 mm one after it, as a test.

    First the detector's `range_est` came from a per-CLASS radius sized on a
    playroom block, so a can read 0.259 m at 0.491 m; the brain answered by
    inverting the apparent width against a CAN's radius. Then the arena began
    passing each prop's own size to the detector, and the can-radius route
    became the bug: MEASURED in moss-yard (2026-09-26), a block read 0.52 m at
    0.26, a ball 0.53 at 0.30 — the robot drove 20 cm too close and the pick
    reached 20 cm past everything that was not a can. So the brain takes
    `range_est`, and this checks it is right for a BLOCK in the yard.
    """
    import dataclasses

    from microduck_local.brain.tidy_moss import TidyMoss
    from microduck_local.world import scenario as S
    from microduck_local.world.arena import World

    brain = TidyMoss()
    assert brain.p.range_from_detector
    det = SimpleNamespace(width=0.1, range_est=0.42)
    assert brain._range(det) == pytest.approx(0.42)
    # the can-width route still exists for a lab that wants it
    brain.p = dataclasses.replace(brain.p, range_from_detector=False)
    R = brain.p.can_radius_m
    det = SimpleNamespace(width=2 * math.atan(R / 0.5), range_est=0.1)
    assert brain._range(det) == pytest.approx(0.5, rel=1e-6)

    sc = S.Scenario.from_dict(json.loads(
        Path(__file__).resolve().parents[1].joinpath(
            "scenarios/moss-yard.json").read_text()))
    w = World(sc)
    r = w.ducks["m0"]
    errs = {}
    for leg in ((0.35, 0.0, 3.0), (0.0, 1.0, 1.6)) * 4:
        for _ in range(int(leg[2] / 0.02)):
            r.driver.set_cmd(leg[0], leg[1], w.data.time)
            w.step()
            last = r.detector.last
            for d in (last.detections if last else []):
                if d.cls != "toy" or not d.name:
                    continue
                cam = w.data.xpos[w.model.body(d.name).id]
                x, y, _yaw = r.driver.pose(w.data)
                true = math.hypot(cam[0] - x, cam[1] - y)
                errs.setdefault(d.name.rstrip("0123456789"), []).append(
                    d.range_est / true)
    assert "block" in errs and "can" in errs, sorted(errs)
    for shape, ratios in errs.items():
        # within 25% of the truth for EVERY shape (lens offset and noise);
        # the can-radius route put the block at ~2x
        assert 0.75 < float(np.median(ratios)) < 1.25, (shape, np.median(ratios))


def test_the_tidy_brain_ignores_its_own_bin_and_anything_behind_it():
    """A can already IN the bin is still a can to a detector.

    Measured on a 90 s contact sheet before this filter existed: one can
    picked, then 80 s of the machine cycling search -> close -> stow on a
    detection of its own load, which sits behind the camera and folds back to
    a small x.
    """
    import dataclasses

    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss
    brain = TidyMoss()
    # The FILTERS, alone: with `object_memory` a sighting also needs a second
    # frame in the same place (tests/test_moss_search.py), which one call
    # with no odometry cannot give it.
    brain.p = dataclasses.replace(brain.p, object_memory=False)
    R = brain.p.can_radius_m

    def seen(bearing, rng):
        det = SimpleNamespace(cls="toy", width=2 * math.atan(R / rng),
                              range_est=rng, bearing=bearing, elevation=0.0,
                              conf=1.0, name="can0")
        frame = SimpleNamespace(t=0.0, detections=[det])
        return brain._see(Senses(t=0.0, det=frame))

    assert seen(0.0, 0.8) is not None, "a can straight ahead is a target"
    assert seen(3.0, 0.25) is None, "a can behind the robot is not"
    assert seen(0.0, 0.10) is None, "a can under the chassis is not"


def test_the_tuck_pose_keeps_the_arm_inside_the_shell():
    """Laurent's request, 2026-09-24: "keep the gripper within the rover's
    footprint, clear of the bin". HOME does not — it reaches 0.312 m against
    a shell that ends at 0.115 — which is the wall-bumping he described."""
    m = moss.model()
    d = mujoco.MjData(m)
    arm_geoms = [g for g in range(m.ngeom)
                 if m.geom_group[g] == moss.COLLISION_GROUP
                 and m.geom_bodyid[g] != m.body(moss.BASE_BODY).id
                 and m.body_rootid[m.geom_bodyid[g]] == m.body(moss.BASE_BODY).id]
    assert arm_geoms, "no arm collision geoms to check"

    def extent(pose):
        d.qpos[:] = m.key_qpos[0]
        for j, v in zip(moss.ARM_JOINTS, pose):
            d.qpos[m.joint(j).qposadr[0]] = v
        mujoco.mj_forward(m, d)
        pts = np.array([d.geom_xpos[g] for g in arm_geoms] + [d.site_xpos[m.site("tcp").id]])
        return pts, min([float(c.dist) for c in d.contact[:d.ncon]], default=1.0)

    pts, worst = extent(moss.tuck_pose())
    assert pts[:, 0].max() < 0.115, f"the tuck reaches x {pts[:, 0].max():.3f}"
    assert pts[:, 0].min() > -0.145
    assert abs(pts[:, 1]).max() < 0.141
    assert pts[:, 2].max() < moss.BIN_RIM_Z, "the tuck stands above the bin"
    assert worst > -1e-3, (
        f"the tuck drives {-worst * 1e3:.2f} mm into the robot. A 0.04 mm "
        "palm-on-hull graze is expected and documented: the pose that "
        "cleared it stowed 0/9 cans against this one's 2/9")

    home, _ = extent([moss.ARM_HOME[j] for j in moss.ARM_JOINTS])
    assert home[:, 0].max() > 0.30, (
        "HOME no longer sticks out — the tuck has nothing to fix")


# ----------------------------------------------------------- the pick env

def test_the_pick_env_shows_the_policy_a_BELIEF_not_the_truth():
    """The can arrives through a modelled camera, not through omniscience.

    A policy fed the truth every step has never had a stale fix and has no
    idea what to do with one — and on the rover the can leaves the frame at
    close range, arrives 50 ms late at 10 Hz, and MOVES when a track clips
    it. So the observation carries the last reported fix, carried forward by
    the base's own motion, with `OBS_TARGET_SEEN` going to 0 when it is
    stale. The REWARD still reads the truth: paying a policy for its own
    belief would pay it for believing whatever is convenient.
    """
    from microduck_local.robots.moss_env import MossPickEnv

    env = MossPickEnv(seed=3, pick_rung=2)
    obs, _ = env.reset()
    # the very first observation cannot have a fix: the first frame is still
    # in flight (DET_LATENCY_S), which is a real property of the camera
    assert obs[moss.OBS_TARGET_SEEN][0] == 0.0
    seen = 0
    for _ in range(60):
        obs, _r, term, trunc, _info = env.step(np.zeros(moss.NUM_ACTIONS, np.float32))
        seen += int(obs[moss.OBS_TARGET_SEEN][0])
        if term or trunc:
            break
    g = env.ghost()
    assert g["belief"] is not None, "the camera never delivered a fix"
    assert seen > 0, "the target was never marked seen"
    err = math.hypot(g["truth"][0] - g["belief"][0],
                     g["truth"][1] - g["belief"][1])
    assert err < 0.10, f"the belief is {err:.3f} m from the truth"
    assert err > 1e-6, "the belief IS the truth — the camera model is a no-op"


def test_the_pick_rungs_are_a_spawn_ladder():
    """The curriculum is in the physics, not the weights: rung 0 puts the can
    already between the pads so the first thing a policy can learn is what
    closing does."""
    from microduck_local.robots.moss_env import RUNG_BOX, MossPickEnv

    gaps = {}
    for rung in RUNG_BOX:
        env = MossPickEnv(seed=5, pick_rung=rung)
        env.reset()
        gaps[rung] = env._gap()
    assert gaps[0] < gaps[1] < gaps[2], gaps
    assert gaps[0] < 0.03, (
        f"rung 0's can is {gaps[0]:.3f} m from the pocket — it is supposed "
        "to spawn BETWEEN the pads, and a drill that is not a drill teaches "
        "nothing about closing")


def test_a_policy_that_parks_on_the_can_earns_less_than_one_that_lifts():
    """The proxy this reward was rebuilt to avoid.

    With a per-step held bonus of 2.0 a script that shut its jaws and sat
    there earned +18.75 while one that actually lifted earned +5.73. Lift is
    progress-paid now, so the order is the other way round.
    """
    from microduck_local.robots.moss_env import ARM_DELTA_RAD, MossPickEnv

    lift_dir = np.clip((np.array(moss.LIFT_POSE) - np.array(moss.GRASP_POSE))
                       / ARM_DELTA_RAD, -1.0, 1.0)

    def run(lift: bool) -> float:
        total = 0.0
        for seed in range(3):
            env = MossPickEnv(seed=seed, pick_rung=0)
            env.reset()
            a = np.zeros(moss.NUM_ACTIONS, np.float32)
            for i in range(70):
                a[:] = 0.0
                a[5] = -1.0 if i < 25 else 0.0
                if lift and i >= 25:
                    a[:5] = lift_dir
                _o, r, term, trunc, _info = env.step(a)
                total += r
                if term or trunc:
                    break
        return total / 3.0

    assert run(lift=True) > run(lift=False) + 2.0


def test_one_servo_drives_both_jaws_and_the_coupling_does_not_ratchet():
    """Laurent's correction, 2026-09-24: the real gripper has ONE servo.

    Two things have to hold, and the second cost a whole training run. The
    jaws must mirror — a policy given two finger commands would learn a
    scissor the hardware cannot perform — and the coupling must be STIFF
    enough that it does not ratchet: at MuJoCo's default softness, contact
    impulses on the unactuated follower pushed the pair open a little per
    grasp until the leader read 0.054 m against its own 0.041 m limit. A grip
    test that reads commanded-minus-achieved then calls every empty jaw a
    catch, which is exactly what happened (12/12 "grips", 0/12 lifts).
    """
    m = moss.model()
    d = mujoco.MjData(m)
    names = [m.actuator(i).name for i in range(m.nu)]
    assert moss.FINGER_JOINTS[1] not in names, "the follower has no servo"
    assert m.neq >= 1 and any(m.equality(i).name == "gripper_coupling"
                              for i in range(m.neq))
    mujoco.mj_resetDataKeyframe(m, d, 0)
    lead = m.joint(moss.GRIPPER_JOINT).qposadr[0]
    follow = m.joint(moss.FINGER_JOINTS[1]).qposadr[0]
    worst_mirror = worst_drift = 0.0
    for _ in range(4):
        for cmd in (0.041, 0.0):
            d.ctrl[m.actuator(moss.GRIPPER_JOINT).id] = cmd
            for _ in range(400):
                mujoco.mj_step(m, d)
            worst_mirror = max(worst_mirror,
                               abs(float(d.qpos[lead]) - float(d.qpos[follow])))
            worst_drift = max(worst_drift, float(d.qpos[lead]) - cmd)
    assert worst_mirror < 1e-3, f"the jaws scissored by {worst_mirror * 1e3:.2f} mm"
    assert worst_drift < 2e-3, (
        f"the coupling ratcheted the leader {worst_drift * 1e3:.2f} mm past "
        "its command with nothing in the jaws — a grip test reading "
        "commanded-minus-achieved would call that a catch")


def test_laurents_corrections_are_all_in_the_model():
    """Every correction he sent on 2026-09-24, asserted in one place.

    Written because these kept being questioned after they had landed, and a
    claim that something "is done" is worth exactly as much as the check that
    fails when it stops being true. Each line here is one of his points:

      "couple the simulated fingers and expose one gripper command... eight
       independent actions rather than nine"
      "the rover weighs 3.5 kg without the SO-101... Blue top cover is 140 g
       and bin is 180 g"
      "12 cm is not a validated depth cutoff for our setup. The published
       minimum is approximately 52 cm"
      "the current product page lists RGB 87 x 62"
      "the CAD wheel/belt centre spacing is 266 mm, whereas the old collision
       boxes are 244 mm apart"

    The last one is asserted as a DISAGREEMENT on purpose: he asked that we
    reconcile it together, so both numbers stay declared until he decides.
    See `moss.reconcile_track_spacing`, which exists and is deliberately not
    called.
    """
    m = moss.model()

    # ONE gripper command, enforced in the model rather than by convention.
    assert moss.NUM_ACTIONS == 8, "five arm + one gripper + (vx, wz)"
    eq = {m.equality(i).name for i in range(m.neq)}
    assert "gripper_coupling" in eq, "the jaws must be tied in the MODEL"
    assert m.nu == 6, "five arm actuators + ONE gripper, not two fingers"

    # The mass he weighed, as separate components at their own centroids.
    parts = {n: kg for n, kg, _ in moss.BASE_COMPONENTS}
    assert parts == pytest.approx({"equipped_base": 3.18, "cover": 0.14,
                                   "bin": 0.18})
    assert sum(parts.values()) == pytest.approx(3.50)

    # His camera numbers, and his depth correction.
    assert moss.CAMERA_HFOV_DEG == pytest.approx(87.0)
    assert moss.DEPTH_MIN_RANGE_M == pytest.approx(0.52), (
        "0.12 m was ours and wrong; the published minimum is ~0.52 m")

    # The spacing is RECONCILED as of 2026-09-24 — his 2026-09-25 call, that
    # the collision geometry match the V0.4 CAD, is what the default build
    # now does. Both numbers stay declared so the history is legible, but the
    # geoms are at the CAD's 266 mm and the 244 mm proxies are the opt-out.
    assert moss.TRACK_CENTRES_M == pytest.approx(0.266)      # V0.4 CAD
    assert moss.TRACK_CENTRES_COLLISION_M == pytest.approx(0.244)
    if moss.collision_v04():
        got = abs(float(m.geom("track_1").pos[1])
                  - float(m.geom("track_-1").pos[1]))
        assert got == pytest.approx(moss.TRACK_CENTRES_M, abs=1e-6), (
            "the DEFAULT build must put the collision geoms on his CAD "
            f"spacing, not the legacy proxies — got {got * 1000:.1f} mm")

    # Nothing of ours is bound to the old `tread_*` names: the V0.4 belts
    # arrive as static visual meshes, which is what he flagged.
    names = {m.geom(i).name or "" for i in range(m.ngeom)}
    assert not any(n.startswith("tread_") for n in names)
    assert {"v04_belt_left", "v04_belt_right"} <= names


def test_the_shipped_policies_speak_the_hardware_contract():
    """The three files a robot would load, checked as files.

    `export-walk` bakes the observation normalizer into the graph, so what
    ships is the whole function and not a checkpoint plus a scaler nobody
    kept in sync. This asserts the shape and the contract stamp; the schema
    (docs/moss-policy-schema.md) is what maps them to hardware.
    """
    ort = pytest.importorskip("onnxruntime")
    import numpy as np
    runs = Path(__file__).resolve().parents[1] / "runs"
    for leg in ("approach", "pick", "stow"):
        p = runs / f"moss-{leg}-v1" / "policy.onnx"
        if not p.is_file():
            pytest.skip(f"{p} not built in this checkout")
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        assert list(s.get_inputs()[0].shape) == [1, moss.OBS_DIM]
        assert list(s.get_outputs()[0].shape) == [1, moss.NUM_ACTIONS]
        stamped = s.get_modelmeta().custom_metadata_map.get("contract_id")
        assert stamped in (None, moss.CONTRACT_ID)
        out = s.run(None, {s.get_inputs()[0].name:
                           np.zeros((1, moss.OBS_DIM), np.float32)})[0]
        assert np.isfinite(out).all()


def test_the_bin_box_is_closed_underneath():
    """A can on the GROUND under the chassis is not a delivery.

    `_in_bin` tested x, y and a rim CEILING with no floor, so the box was
    open downwards and a can that rolled under the rover inside the bin's
    footprint scored as stowed — and `STOW_BONUS` paid for it. Planted the
    regression to check this test bites: with the floor bound removed, the
    can-on-the-ground case below asserts True and the test fails.
    """
    from microduck_local.robots.moss_env import MossStowEnv

    env = MossStowEnv(stow_rung=0)
    env.reset(seed=0)
    x = 0.5 * (moss.BIN_INTERIOR_X[0] + moss.BIN_INTERIOR_X[1])

    def at(z):
        env._can_base = lambda: np.array([x, 0.0, z])
        return env._in_bin()

    assert at(0.5 * (moss.BIN_FLOOR_Z + moss.BIN_RIM_Z)), "inside the bin"
    assert not at(0.033), "on the GROUND under the chassis is not in the bin"
    assert not at(moss.BIN_FLOOR_Z - 0.01), "below the bin floor is not in it"
    assert not at(moss.BIN_RIM_Z + 0.01), "above the rim is not in it"


def test_a_can_still_above_the_rim_is_not_yet_a_drop():
    """A can let go over the bin is STATIONARY before it is anywhere.

    The settle test was "at rest for more than 5 ticks", and a can released
    over the mouth has been held still, so its speed is near zero 0.24 s
    later while it is still above the 0.261 m rim. That scored a delivery in
    mid-air as a drop — the lab's "it went in the basket and counted a fall".

    Gravity is zeroed so the can genuinely holds still above the rim, which
    is the state the old condition misread. Planted the regression to check
    this bites: drop the `below_rim` term and the env terminates here with
    `dropped`, and the final assertion fails.
    """
    from microduck_local.robots.moss_env import MossStowEnv

    env = MossStowEnv(stow_rung=0)
    env.reset(seed=0)
    env.model.opt.gravity[:] = 0.0          # hold the can where we put it

    # Over the mouth, clearly ABOVE the rim, and out of the jaws.
    x = 0.5 * (moss.BIN_INTERIOR_X[0] + moss.BIN_INTERIOR_X[1])
    env.data.qpos[env.can_qadr:env.can_qadr + 3] = [x, 0.0,
                                                    moss.BIN_RIM_Z + 0.04]
    env.data.qvel[env.can_dadr:env.can_dadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert not env._carrying(), "the can must not be in the jaws for this test"
    assert not env._in_bin(), "above the rim is not in the bin"

    # Released a moment ago: past the 5-tick minimum, far short of the
    # 50-tick settle clock.
    env._released_at = env.step_count - 10

    _, _, terminated, _, info = env.step(np.zeros(moss.NUM_ACTIONS, np.float32))
    assert not info["stowed"], "it is not in the bin yet"
    assert not terminated, (
        "a can at rest ABOVE the rim is still falling or perched — judging it "
        "now scores a delivery in flight as a drop")


def test_the_pick_criterion_is_the_instant_one_and_the_stricter_ones_are_kept():
    """`picked` is the INSTANT test, and the two stricter ones stay uncalled.

    Not because the instant test is good — it is uncorrelated with cans
    delivered. Two replacements were built and each was checked against the
    mission before being adopted, which is the whole point:

      * the sustained hold (`PICK_HOLD_STEPS`) ranks existing legs in mission
        order, then INVERTED when trained against — 62% -> 81% sustained
        picks and 27/72 -> 19/72 cans;
      * the handover ramp (`_lift_handover`) mis-ranks without any training:
        79/58/21% against cans of 26/38/22%.

    So they are kept as the record and the scored criterion is unchanged,
    which is also what keeps every number recorded on the runs comparable.
    """
    from microduck_local.robots import moss_env as ME

    assert ME.PICK_HOLD_STEPS > 1, "kept as the record, with its own value"
    assert hasattr(ME.MossPickEnv, "_lift_handover"), "kept as the record"

    src = inspect.getsource(ME.MossPickEnv.step)
    assert "picked = held and lift > SUCCESS_LIFT_M" in src, (
        "the scored criterion is the instant one")
    assert "_lift_handover()" not in src, (
        "the handover ramp mis-ranks against the mission — it must not be "
        "wired into the score without a fresh check that it ranks by cans")


def test_every_one_of_his_corrections_is_live_in_the_default_build():
    """All seven of Laurent's 2026-09-24 corrections, checked in the model
    that actually ships — not in a constant, and not behind a flag.

    This exists because "is it applied?" kept being answered from memory and
    the answer kept being wrong in both directions. Each assertion reads the
    COMPILED default: no environment variables, no opt-ins.
    """
    m = moss.model()

    # 1. One gripper command, enforced by the model rather than convention.
    assert moss.NUM_ACTIONS == 8
    assert m.nu == 6, "five arm actuators and ONE gripper"
    assert "gripper_coupling" in {m.equality(i).name for i in range(m.neq)}

    # 2. His three weighed components at their own centroids, COMPOSED into
    #    the rover's inertial rather than lumped at the balance point.
    parts = {n: kg for n, kg, _ in moss.BASE_COMPONENTS}
    assert parts == pytest.approx({"equipped_base": 3.18, "cover": 0.14,
                                   "bin": 0.18})
    total = sum(parts.values())
    cx = sum(kg * c[0] for _n, kg, c in moss.BASE_COMPONENTS) / total
    assert float(m.body(moss.BASE_BODY).ipos[0]) == pytest.approx(cx, abs=1e-6), (
        "the compiled rover must carry the COMPOSED centre, not his balance "
        "point with everything lumped on it")
    assert float(m.body(moss.BASE_BODY).ipos[0]) != pytest.approx(0.0, abs=1e-6)

    # 3. His camera figures, and his depth correction.
    assert moss.CAMERA_HFOV_DEG == pytest.approx(87.0)
    assert moss.CAMERA_VFOV_DEG == pytest.approx(62.0)
    assert moss.DEPTH_MIN_RANGE_M == pytest.approx(0.52)

    # 4. Nothing bound to the old tread_* names.
    names = {m.geom(i).name or "" for i in range(m.ngeom)}
    assert not any(n.startswith("tread_") for n in names)

    # 5. His V0.4 visual meshes, via his own patcher.
    assert sum(1 for n in names if n.startswith("v04_")) > 40
    assert {"v04_belt_left", "v04_belt_right"} <= names

    # 6. The track spacing, RECONCILED to his CAD in the default build.
    assert moss.collision_v04(), "his collision model is the default"
    spacing = abs(float(m.geom("track_1").pos[1])
                  - float(m.geom("track_-1").pos[1]))
    assert spacing == pytest.approx(0.266, abs=1e-6)

    # 7. The payload is modelled but NOT claimed as measured — the one item
    #    that needs his scale rather than our simulator.
    assert moss.PAYLOAD_UNMEASURED is True, (
        "the loaded-bin tensor is his measurement to make; simulating it "
        "does not make it measured")


def test_anything_that_can_roll_has_a_contact_model_that_can_stop_it():
    """A rolling coefficient on a condim-3 geom is a comment, not physics.

    This cost the repo two separate months. The soccer ball had no rolling
    resistance until 2026-09-06 — the coefficient was set and `condim` 3
    ignores it. On 2026-09-24 the identical bug was found on MOSS's cans,
    because `world/compose.py` granted condim 6 to props whose shape is
    "sphere" and cylinders fell through to a condim-3 branch. A knocked can
    rolled until something stopped it, and what stopped it was the rover's
    own tracks: fixing it took the litter mission from 46% to 83%.

    So the assertion is about the CLASS, not either object: a geom that can
    roll needs condim 6, wherever it is declared.
    """
    from microduck_local.robots.moss_env import MossPickEnv

    rollable = {mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER,
                mujoco.mjtGeom.mjGEOM_CAPSULE}
    m = MossPickEnv(seed=0, pick_rung=2).model
    checked = 0
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue                      # visual only, cannot roll anywhere
        if m.geom_type[i] not in rollable or m.geom_bodyid[i] == 0:
            continue                      # fixed to the world, or not round
        if m.body_dofnum[m.geom_bodyid[i]] == 0:
            continue                      # welded: it cannot roll either
        checked += 1
        assert m.geom_condim[i] >= 6, (
            f"{name!r} is a free rolling shape at condim {m.geom_condim[i]} — "
            "MuJoCo ignores its rolling friction, so nothing slows it down")
        assert m.geom_friction[i][2] > 0.0, (
            f"{name!r} has condim 6 but a rolling coefficient of 0")
    assert checked, "no free rolling geom found — this test stopped testing"


def test_the_scripted_carry_abandons_a_can_it_has_dropped():
    """Every scripted state must check the precondition it assumes.

    The carry ran all three ramps and opened the jaws over the bin without
    ever asking whether it still held anything — and only ~31% of lifts do,
    so four stows in ten were nine seconds of theatre inside a 180 s run.
    A human watching the lab found it; no metric here had.

    The learned stages get watched and scored. The scripted ones get
    trusted, which is where preconditions go unchecked.
    """
    from microduck_local.brain import tidy_moss as TM

    p = TM.TidyMossParams()
    assert p.stow_abort_on_drop is True, (
        "the carry must notice an empty gripper, not finish the delivery")
    assert p.stow_drop_grace_s > 0.0, (
        "the pads flicker during the swing — a single empty tick is not a drop")

    src = inspect.getsource(TM.TidyMoss.step)
    stow = src[src.index('elif self.state == "stow":'):
               src.index('elif self.state == "release":')]
    assert "_gripped" in stow, (
        "the scripted stow must test the grip every tick; it assumed a can")


def test_no_scripted_transition_is_a_step_input_to_the_servos():
    """A scripted pose change must RAMP, or it slams the real arm.

    `deploy` and `tuck` commanded their pose outright: p95 joint speeds of
    9.47 and 7.94 rad/s with peaks of 12.9 and 13.5, against an XL330's
    practical no-load speed of about 5.6. Those are not merely harsh, they
    are unreachable on hardware — the sim only gets there because a position
    servo with no rate limit will happily slam. `lift`, `stow` and `release`
    already ramped and peaked at 2.00, 2.37 and 0.67.

    Asserts the mechanism rather than a speed, because the speed depends on
    the pose and the pose keeps changing.
    """
    from microduck_local.brain import tidy_moss as TM

    p = TM.TidyMossParams()
    assert p.deploy_ramp_s > 0.5 and p.tuck_ramp_s > 0.5

    src = inspect.getsource(TM.TidyMoss.step)
    for state, nxt in (('elif self.state == "deploy":', 'elif self.state == "creep"'),
                       ('elif self.state == "tuck":', None)):
        blk = src[src.index(state):
                  (src.index(nxt) if nxt else len(src))]
        assert "_ramp_from_here" in blk or "_ramp(" in blk, (
            f"{state} commands its pose outright — ramp it, or the real "
            "servos take a step input they cannot follow")


def test_the_can_spawns_in_every_pose_the_room_produces():
    """Upright, on its side, AND mid-topple — and the same in the preview.

    An env that only ever spawned an upright can trained a policy that drove
    the can to 12 mm and oscillated without committing, because a cylinder
    lying across the jaws is a different grasp. The mid-topple case is what
    the approach actually creates: the chassis clips the can and the policy
    meets it at 40 degrees, already going over.

    The second half is the one that kept biting: the pose must NOT depend on
    `domain_rand`, which the lab's preview pins False to protect the shared
    model. A resting pose writes no model field, and gating it there is why
    the watched trainee saw an upright can every single episode.
    """
    from microduck_local.robots.moss_env import MossPickEnv

    def tilts(domain_rand):
        out = []
        for s in range(40):
            e = MossPickEnv(seed=s, pick_rung=2, domain_rand=domain_rand)
            e.reset(seed=s)
            q = e.data.qpos[e.can_qadr + 3:e.can_qadr + 7]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            out.append(float(np.degrees(np.arccos(
                np.clip(R.reshape(3, 3)[2, 2], -1.0, 1.0)))))
        return np.array(out)

    for domain_rand in (True, False):
        t = tilts(domain_rand)
        upright = int((t < 10).sum())
        lying = int((t > 80).sum())
        tilted = int(((t >= 10) & (t <= 80)).sum())
        who = "trainer" if domain_rand else "the lab's PREVIEW"
        assert upright >= 2, f"{who} never spawns an upright can"
        assert lying >= 2, f"{who} never spawns one on its side"
        assert tilted >= 2, (
            f"{who} never spawns one mid-topple — which is the pose the "
            "approach actually creates")


def test_a_roster_slot_reads_the_bodys_env_knobs_too():
    """The rung must reach EVERY lab env, not just the trainer's.

    This knob had to be plumbed four separate times — trainer, trainee
    preview, `/teach` request, and finally the roster slot — and each patch
    left the next path silently on the default. A roster MOSS ran rung 0,
    where the can spawns between the pads within 15 mm every episode, so the
    watched robot appeared never to see a different can. It didn't.

    `slot_env` is the one seam every body env in the lab passes through.
    """
    import os

    from microduck_local.lab import robots as lab_robots

    prev = os.environ.get("MICRODUCK_MOSS_PICK_RUNG")
    os.environ["MICRODUCK_MOSS_PICK_RUNG"] = "2"
    try:
        e = lab_robots.slot_env("moss", 0, {"task": "pick", "seed": 0})
        assert e.rung == 2, (
            f"a roster slot took rung {e.rung} while the lab's environment "
            "said 2 — the watched robot is not doing the task being asked")
        # An explicit kwarg still wins: the trainee preview resolves its own
        # stage and must not be overridden by the lab process's export.
        e2 = lab_robots.slot_env("moss", 0, {"task": "pick", "seed": 0,
                                             "pick_rung": 1})
        assert e2.rung == 1, "an explicit kwarg must beat the environment"
    finally:
        if prev is None:
            os.environ.pop("MICRODUCK_MOSS_PICK_RUNG", None)
        else:
            os.environ["MICRODUCK_MOSS_PICK_RUNG"] = prev


def test_the_axis_slots_are_live_at_every_rung():
    """**The observation the policy trains on must not be dead where it is
    deployed.**

    The wrist camera looks down at the jaws, so it sees nothing until ~20 cm.
    Measured on 2026-09-25, with attitude coming from the wrist ALONE, the
    axis slots carried a value on 67% of ticks at rung 0, 7% at rung 1 and
    **0.00% at rung 2** — which is the rung `brain/tidy_moss.py` hands over
    at. A policy trained that way sees three constant zeros and then gets
    live numbers on the robot: the train/deploy mismatch that cost the stow
    leg 9/12 in its env against 0/9 in a room.

    So the FRONT camera answers for the axis whenever the wrist cannot, and
    this test is the thing that notices if either half stops reporting. It
    fails if the slots go dead at ANY rung, not just the easy one.
    """
    from microduck_local.robots.moss_env import RUNG_BOX, MossPickEnv

    live = {}
    for rung in RUNG_BOX:
        env = MossPickEnv(seed=11, pick_rung=rung, domain_rand=True,
                          obs_noise=True, publish_attitude=True)
        rng = np.random.default_rng(4)
        obs, _ = env.reset()
        seen = 0
        total = 0
        for _ in range(600):
            a = rng.uniform(-1, 1, env.action_space.shape[0]).astype(np.float32)
            obs, _r, term, trunc, _i = env.step(a)
            axis = np.asarray(obs[moss.OBS_TARGET_AXIS], float)
            seen += int(np.abs(axis).max() > 1e-9)
            total += 1
            if term or trunc:
                obs, _ = env.reset()
        live[rung] = seen / total
    # A third of the ticks is well below what the two cameras measure (0.96 /
    # 0.78 / 0.68) and well above the wrist-only rung 2 this guards against.
    for rung, frac in live.items():
        assert frac > 0.33, (
            f"rung {rung}: the can's axis reaches the observation on only "
            f"{frac:.0%} of ticks. The policy cannot align a jaw to an axis "
            f"it is not told — check that the FRONT camera still reports "
            f"attitude (`_front_attitude`) and not just position. All: {live}")


def test_the_axis_slots_stay_DEAD_unless_a_policy_asks_for_them():
    """**The other half, and the one that cost 10/12.**

    Filling slots 28-30 for a policy that trained with them zero does not set
    it a harder task, it breaks its input: the normalizer baked into every
    pre-2026-09-25 export carries sd 1.75e-05 there, so a real axis value of
    0.9 arrives as z = 51,522 and clips to the bound. The shipped pick leg
    went 10/12 -> 0/12 at rung 2 on the same twelve seeds the day the front
    camera started reporting, and the deployed mission runs three such legs.

    So the default is OFF and this test is what keeps it off. A policy trained
    with the axis passes `publish_attitude=True`; everything else — the
    scripted brain, `eval_moss_pick.py`, the lab's palette — keeps the
    observation it was trained on.
    """
    from microduck_local.robots.moss_env import MossPickEnv

    env = MossPickEnv(seed=11, pick_rung=0, domain_rand=True, obs_noise=True)
    assert env.publish_attitude is False, (
        "attitude publication must default OFF — see ATTITUDE_DEFAULT")
    rng = np.random.default_rng(4)
    obs, _ = env.reset()
    worst = 0.0
    for _ in range(400):
        obs, _r, term, trunc, _i = env.step(
            rng.uniform(-1, 1, env.action_space.shape[0]).astype(np.float32))
        worst = max(worst, float(np.abs(obs[moss.OBS_TARGET_AXIS]).max()),
                    float(np.abs(obs[moss.OBS_TARGET_UPRIGHT]).max()))
        if term or trunc:
            obs, _ = env.reset()
    assert worst == 0.0, (
        f"the axis slots carried {worst:.3f} with publish_attitude off — an "
        "old policy's normalizer turns that into a z-score of tens of "
        "thousands and the leg stops picking up cans")


def test_the_wrist_can_actually_aim_the_jaw():
    """**Measured off the GEOMETRY, not off what a policy happens to do.**

    Before blaming a policy for not turning the calipers, this is the question
    that has to be answered from the model: can `wrist_roll` point the jaw at
    an arbitrary can lying on the floor? A parallel jaw is symmetric modulo
    180 degrees, so covering 180 degrees of jaw YAW is the whole requirement,
    and the grasp must not tilt while doing it.

    MEASURED on 2026-09-25 (this SO-101-derived arm): a +-90 degree roll sweep
    takes the jaw's opening axis across 14 to 163 degrees of yaw, about 1:1,
    with elevation inside +-12 degrees. `wrist_roll` itself has 320 degrees of
    range for a job needing 90, and driven open-loop it reaches both limits.

    So when a policy leaves the wrist at a constant angle (sd 2.6 deg at the
    grasp against a can axis spanning 54 deg), that is a LEARNING result and
    not a mechanical one — which is the distinction this test exists to keep
    honest if the arm's model is ever re-exported.
    """
    import math as _math

    from microduck_local.robots.moss_env import MossPickEnv

    env = MossPickEnv(seed=1, pick_rung=2)
    env.reset()
    m, d = env.model, env.data
    roll = m.joint("wrist_roll")
    lo, hi = float(roll.range[0]), float(roll.range[1])
    assert hi - lo > _math.radians(180), (
        f"wrist_roll spans {_math.degrees(hi - lo):.0f} deg; aligning a "
        "symmetric jaw to any can needs 180")

    fl = m.joint("finger_left")
    adr = roll.qposadr[0]
    yaws = []
    elevs = []
    for deg in range(-90, 91, 15):
        d.qpos[adr] = _math.radians(deg)
        mujoco.mj_forward(m, d)
        R = d.xmat[m.jnt_bodyid[fl.id]].reshape(3, 3)
        ax = R @ np.array(fl.axis, float)
        _x, _y, byaw = env.driver.pose(d)
        c, s = _math.cos(-byaw), _math.sin(-byaw)
        ax_b = np.array([c * ax[0] - s * ax[1], s * ax[0] + c * ax[1], ax[2]])
        yaws.append(_math.degrees(_math.atan2(ax_b[1], ax_b[0])) % 180.0)
        elevs.append(abs(_math.degrees(_math.asin(
            float(np.clip(ax_b[2] / np.linalg.norm(ax_b), -1, 1))))))
    span = max(yaws) - min(yaws)
    assert span > 150.0, (
        f"rolling the wrist moves the jaw's opening axis through only "
        f"{span:.0f} deg of yaw, so a can lying across the approach cannot be "
        "aligned to by the wrist at all — check the finger slide axis and the "
        "wrist_roll hinge in the arm model")
    assert max(elevs) < 25.0, (
        f"the jaw tilts {max(elevs):.0f} deg out of horizontal while rolling; "
        "aligning would come at the cost of the grasp")


def test_a_policy_is_evaluated_in_the_observation_it_TRAINED_on():
    """**The mistake this repo made twice in one day, once for real.**

    Nothing in a `.onnx` says whether it was trained with the can's axis in
    slots 28-30. An axis-blind export's normalizer has var 3e-10 there, so
    handing it a live axis pins three of thirty-two inputs at the clip bound:
    the shipped leg measured 10/12 -> 0/12, and a probe written to COMPARE
    the policies set the flag process-wide and measured 59/60 -> 0/60 — the
    same error, once in the code and once in the measurement of the code.

    `obs_env_kwargs` answers it from the RUN, so a consumer cannot get it
    wrong by forgetting. Runs from before the variant was recorded are
    axis-blind, which is both the safe default and the truth.
    """
    import json as _json

    from microduck_local.robots.moss_env import OBS_FLAGS, obs_env_kwargs

    d = Path(__file__).resolve().parents[1] / "runs"
    if not d.is_dir():
        pytest.skip("no runs/ in this checkout")

    # a run that says nothing must come back with EVERY observation flag off,
    # never on. And it must answer for every flag there is: `publish_size` was
    # added after this helper and left out of it, so a probe built the env
    # without it, handed a policy trained on slot 31 a zero, and measured 3
    # grips where it really makes 63 — a partial guard that read as complete.
    blind = obs_env_kwargs(d / "does-not-exist" / "policy.onnx")
    assert blind == {k: False for k in OBS_FLAGS}, blind
    assert set(blind) == set(OBS_FLAGS)

    for run in sorted(d.glob("*/run.json")):
        try:
            kw = (_json.loads(run.read_text()).get("env_kwargs") or {})
        except (OSError, ValueError):
            continue
        got = obs_env_kwargs(run.parent / "policy.onnx")
        assert set(got) == set(OBS_FLAGS), got
        for flag in OBS_FLAGS:
            assert got[flag] == bool(kw.get(flag, False)), (
                f"{run.parent.name}: run.json says {flag}="
                f"{kw.get(flag)} but obs_env_kwargs said {got[flag]}")


def test_the_arm_lives_against_its_torque_clamp():
    """**MOSS has no BAM, and the clamp it does have is free to sit against.**

    The duck trains against a measured actuator model so a policy cannot learn
    to rely on torque the servo does not have. The arm has only the MJCF's
    `forcerange` of +-2.20 N·m, which MuJoCo enforces silently and charges
    nothing for.

    MEASURED 2026-09-25 over 1,700 control ticks of the then-current policy:
    `shoulder_lift` — the joint holding the arm up against gravity — was at
    the clamp on 38.2% of ticks and peaked at 6.15 rad/s against a COMMANDED
    rate of 0.75 (the action is +-0.03 rad at 25 Hz). A human watching the lab
    called it violent movement, and nothing in the reward mentioned it.

    This test is the no-op guard: it fails if the envelope charge stops being
    reachable, which would mean the penalty had quietly become decoration —
    the `board_margin` mistake this repo already made once.
    """
    from microduck_local.robots.moss_env import MossPickEnv

    env = MossPickEnv(seed=1, pick_rung=2, torque_sat=0.05, overspeed=0.3)
    env.reset()
    assert set(env._arm_act) == set(moss.JOINT_NAMES[:5]), env._arm_act
    for name, ai in env._arm_act.items():
        lim = float(env.model.actuator_forcerange[ai][1])
        assert lim > 0.0, f"{name} has no force limit to live against"

    # NOT by holding the deployed pose — that turned out NOT to saturate, and
    # the first version of this test asserted it did. The clamp is reached by
    # COMMANDING the lifting joint, which is what a policy spends its episode
    # doing: 113 of 120 ticks driving it one way, 69 the other.
    ai = env._arm_act["shoulder_lift"]
    lim = float(env.model.actuator_forcerange[ai][1])
    hit = 0
    a = np.zeros(env.action_space.shape[0], np.float32)
    a[moss.ACT_ARM.start + 1] = 1.0
    for _ in range(120):
        env.step(a)
        if abs(float(env.data.actuator_force[ai])) >= 0.995 * lim:
            hit += 1
    assert hit > 20, (
        f"shoulder_lift reached its torque clamp on only {hit}/120 ticks while "
        "being driven — the envelope charge has become a no-op, which is the "
        "`board_margin` mistake: a penalty on something that cannot happen")


def test_the_handover_contract_is_inside_the_arms_reach():
    """**A task the arm cannot do is not a reward problem.**

    `moss.PICK_HANDOVER_BOX` is where `brain/tidy_moss.py` stops driving and
    hands the can to the pick policy. Until 2026-09-25 its far edge was 0.55 m
    while the arm reaches 0.508 m at graspable height and covers the full
    +-0.12 lateral box only to 0.48 — so on roughly a third of hand-overs the
    can was outside the arm's kinematic reach and the ONLY way to succeed was
    to drive the chassis at it.

    That is why every pick policy trained here drove at the can with its base
    command saturated, and why four separate penalties on base motion never
    stopped it: the base was doing NECESSARY work. A penalty cannot price away
    a motion the task requires.

    This test is the invariant: whatever the contract says, the arm must be
    able to reach it from a standing start.
    """
    import math as _math

    from microduck_local.robots.moss_env import MossPickEnv

    env = MossPickEnv(seed=1, pick_rung=2)
    env.reset()
    m, d = env.model, env.data
    names = moss.JOINT_NAMES[:5]
    adr = [m.joint(j).qposadr[0] for j in names]
    lo = np.array([m.joint(j).range[0] for j in names])
    hi = np.array([m.joint(j).range[1] for j in names])
    rng = np.random.default_rng(0)
    lo_x, hi_x, ymax = moss.PICK_HANDOVER_BOX
    reach = []
    for _ in range(20000):
        q = rng.uniform(lo, hi)
        for a, v in zip(adr, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        tcp = np.array(d.site_xpos[env.tcp_site], float)
        bx, by, byaw = env.driver.pose(d)
        c, sn = _math.cos(-byaw), _math.sin(-byaw)
        dx, dy = tcp[0] - bx, tcp[1] - by
        x, y, z = c * dx - sn * dy, sn * dx + c * dy, tcp[2]
        if 0.02 < z < 0.12:
            reach.append((x, abs(y)))
    assert reach, "no graspable-height samples"
    R = np.array(reach)
    # every corner of the box the brain may deliver into must be reachable
    at_far = R[np.abs(R[:, 0] - hi_x) < 0.012]
    assert len(at_far), f"arm never reaches x={hi_x:.2f} m at grasp height"
    cover = float(at_far[:, 1].max())
    assert cover >= ymax, (
        f"the hand-over contract's far edge is {hi_x:.2f} m, where the arm "
        f"covers only |y| <= {cover:.3f} m against the box's {ymax:.3f}. The "
        "brain would hand the pick policy cans it cannot reach, and the policy "
        "would learn to DRIVE at them — which no base penalty can undo.")


def test_the_gripper_is_charged_for_being_dragged_along_the_floor():
    """**A rule about the HARDWARE, not about the score.**

    MEASURED 2026-09-25 on the best policy: an arm or gripper geom is on the
    floor for 3.9% of control ticks and the base is driving for 3.2% of them —
    almost every floor contact is a scrape under power, and it is the pads
    doing it. On the real robot that is the fingers dragged across a floor,
    collecting whatever is on it, with the servos loaded sideways in their
    weakest axis. Nothing in the reward mentioned it.

    Reachability is the point of this test: a safety charge that cannot fire
    is decoration, and this repo has shipped one of those before.
    """
    from microduck_local.robots.moss_env import MossPickEnv

    env = MossPickEnv(seed=1, pick_rung=2, arm_floor=0.2, low_approach=0.1)
    env.reset()
    assert {"pad_left", "pad_right", "palm"} <= env._arm_geoms, env._arm_geoms

    # Drive the arm down into the floor and the base forward: the charge must
    # fire, and fire harder than the same contact standing still.
    def cost(drive: bool) -> float:
        e = MossPickEnv(seed=1, pick_rung=2, arm_floor=0.2, low_approach=0.0)
        e.reset()
        a = np.zeros(e.action_space.shape[0], np.float32)
        # The command that actually lands the gripper on the floor, FOUND by
        # sweeping all 81 sign combinations rather than assumed — "shoulder
        # and wrist down" does not do it, which is what the first version of
        # this test asserted.
        a[moss.ACT_ARM.start + 0] = 1.0
        a[moss.ACT_ARM.start + 1] = 1.0
        a[moss.ACT_ARM.start + 2] = -1.0
        if drive:
            a[moss.ACT_BASE.start] = 1.0
        base = MossPickEnv(seed=1, pick_rung=2, arm_floor=0.0, low_approach=0.0)
        base.reset()
        got = 0.0
        for _ in range(120):
            _o, r, _t, _tr, _i = e.step(a)
            _o2, r2, _t2, _tr2, _i2 = base.step(a)
            got += r - r2
        return got

    still, driving = cost(False), cost(True)
    assert still < 0.0, (
        f"the gripper on the floor cost {still:+.2f} — the charge never fired, "
        "so it is decoration")
    assert driving < still, (
        f"dragging cost {driving:+.2f} but resting cost {still:+.2f}; dragging "
        "under power is the damaging case and must cost more")


def test_every_pick_knob_reaches_the_LAB_PREVIEW_not_just_the_trainer():
    """**The stage must run what the trainer runs, and knobs are how.**

    The trainer is a subprocess: it inherits `MICRODUCK_MOSS_*` and reads them
    when it imports. The lab's preview is IN-PROCESS, so `_body_env_kwargs`
    stages the same variables around a call to `train_env_kwargs` — which only
    works if that function reads the ENVIRONMENT, not a module constant fixed
    at import.

    Every knob added on 2026-09-25 was a module constant. The result: a run
    training on six object shapes with its base locked previewed as a plain
    upright can with a free base, and the person watching was told "here is
    your training" while being shown something else. The rung was read from the
    environment and was the only knob that ever worked.

    So: stage a knob, and the preview kwargs must change.
    """
    import types

    from microduck_local.robots.registry import get

    body = get("moss")
    args = types.SimpleNamespace(task="pick")
    knobs = {
        "MICRODUCK_MOSS_PROP_VARIETY": ("1", "prop_variety", True),
        "MICRODUCK_MOSS_BASE_LOCK": ("1", "base_lock", True),
        "MICRODUCK_MOSS_ATTITUDE": ("1", "publish_attitude", True),
        "MICRODUCK_MOSS_SIZE_OBS": ("1", "publish_size", True),
        "MICRODUCK_MOSS_PROXIMITY": ("1", "publish_proximity", True),
        "MICRODUCK_MOSS_WRIST_FREE": ("1", "wrist_free", True),
        "MICRODUCK_MOSS_TOPPLE": ("15.0", "can_topple", 15.0),
        "MICRODUCK_MOSS_JAW_ALIGN": ("6.0", "jaw_align", 6.0),
        "MICRODUCK_MOSS_ALIGN_HOLD": ("0.3", "align_hold", 0.3),
        "MICRODUCK_MOSS_GAP_SCALE": ("7.0", "gap_scale", 7.0),
        "MICRODUCK_MOSS_TORQUE_SAT": ("0.05", "torque_sat", 0.05),
        "MICRODUCK_MOSS_OVERSPEED": ("0.3", "overspeed", 0.3),
        "MICRODUCK_MOSS_ARM_FLOOR": ("0.2", "arm_floor", 0.2),
        "MICRODUCK_MOSS_LOW_APPROACH": ("0.1", "low_approach", 0.1),
        "MICRODUCK_MOSS_WRIST_START": ("1.5708", "wrist_start_rand", 1.5708),
        "MICRODUCK_MOSS_WRIST_DRILL": ("1", "wrist_drill", True),
    }
    import os as _os
    for var, (val, kwarg, want) in knobs.items():
        prev = _os.environ.get(var)
        _os.environ[var] = val
        try:
            got = body.train_env_kwargs(args).get(kwarg)
        finally:
            if prev is None:
                _os.environ.pop(var, None)
            else:
                _os.environ[var] = prev
        assert got == want, (
            f"{var}={val} did not reach the preview as {kwarg}={want} (got "
            f"{got!r}). It is almost certainly a module constant read at "
            "IMPORT time, which the lab cannot stage — so the stage will show "
            "different physics from the trainer.")


def test_shape_variety_reaches_EVERY_env_that_records_it():
    """**A flag recorded and then ignored is worse than a missing one.**

    `prop_variety` was implemented in `MossPickEnv.reset`. `MossStowEnv`
    OVERRIDES `reset`, so it never ran: `run.json` said `prop_variety: True`
    while every stow episode spawned the same can, and a 2M-step run went by
    testing something its own provenance claimed it was not. The person
    watching asked "are we training the bin with different shapes?" and the
    honest answer was no, despite the config.

    Both resets now call `_maybe_new_prop`, and this is what keeps it that way
    for the next env that subclasses one of them.
    """
    from microduck_local.robots.moss_env import MossPickEnv, MossStowEnv

    for cls, kw in ((MossPickEnv, {}), (MossStowEnv, {"retract": 6.0})):
        env = cls(seed=1, prop_variety=True, **kw)
        seen = set()
        for _ in range(12):
            env.reset()
            seen.add(env.prop.id)
        assert len(seen) > 1, (
            f"{cls.__name__} spawned only {seen} over 12 resets with "
            "prop_variety=True — the flag is recorded and ignored, which "
            "makes the run's own record.json a lie about what it trained on")

    # ...and OFF must still mean off
    env = MossPickEnv(seed=1, prop_variety=False)
    only = set()
    for _ in range(6):
        env.reset()
        only.add(env.prop.id)
    assert only == {"can"}, only


def test_the_retract_waypoint_is_what_makes_the_fold_reachable():
    """**The tuck pose is not reachable in a straight line, and this is why.**

    `brain/tidy_moss.py`'s tuck state has recorded for months that the arm
    "jams on the hull 0.74 rad short of TUCK_POSE and rides ~75 mm proud".
    Re-measured 2026-09-25: commanding the arm straight to the tuck pose and
    HOLDING it for 600 steps leaves it jammed on `bin_x1`, 0.65 rad short,
    while the tuck pose itself is collision-free and stable (0.013 rad drift).
    Both endpoints are fine; the bin sits between them.

    Ten hand-built routes scored 0-4/24 and 3M steps of training plateaued at
    2.66 rad, because no monotone path exists for either a human or an
    optimiser to find. `moss.RETRACT_WAYPOINT` is a planner's answer.

    This test asserts the property the waypoint was selected for: each leg of
    the route is collision-free where the direct line is not.
    """
    import importlib.util
    from pathlib import Path

    from microduck_local.robots.moss_env import MossStowEnv

    here = Path(__file__).resolve().parents[1] / "scripts" / "plan_retract.py"
    if not here.is_file():
        pytest.skip("planner helper not present")
    spec = importlib.util.spec_from_file_location("plan_retract", here)
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)

    env = MossStowEnv(seed=1, retract=6.0)
    env.reset()
    m, d = env.model, env.data
    adrs = [m.joint(j).qposadr[0] for j in moss.ARM_JOINTS[:5]]
    tuck = np.asarray(moss.tuck_pose(), float)
    wp = np.asarray(moss.RETRACT_WAYPOINT, float)

    # the waypoint itself, and the leg from it to the tuck, must be clear
    assert not pr.collides(m, d, adrs, wp, env._arm_geoms), (
        "RETRACT_WAYPOINT is itself in collision")
    assert pr.seg_free(m, d, adrs, wp, tuck, env._arm_geoms), (
        "the waypoint does not connect to the tuck pose — the route it was "
        "chosen for no longer exists, so the fold will jam as it did before")


def test_the_retract_drill_reset_sets_everything_the_stow_reset_does():
    """The drill starts where a delivery ENDS, by its own reset path, and that
    path forgot `_last_contact`: every worker of teach-moss_stow-71087f died
    on its first step while /teach/status said "training" for six hours. A
    second reset path must leave the env in the same SHAPE as the first, so
    compare the attribute sets rather than listing the ones we know about."""
    from microduck_local.robots.moss_env import DROP_DEBOUNCE, MossStowEnv

    def after_reset(drill):
        env = MossStowEnv(prop_variety=True, retract=6.0, retract_staged=True,
                          retract_drill=drill, stow_rung=2)
        env.reset(seed=1)
        return env

    normal, drill = after_reset(False), after_reset(True)
    missing = sorted(set(vars(normal)) - set(vars(drill)))
    assert not missing, f"the drill reset never sets {missing}"
    # "already delivered": let go before tick 0, already paid for the bin
    assert drill._over_once and drill._retracting
    assert drill.step_count - drill._last_contact >= DROP_DEBOUNCE
    # the branch that crashed: not gripped, object inside CARRY_RADIUS_M
    for _ in range(10):
        drill.step(np.zeros(drill.action_space.shape, np.float32))


def test_the_drop_point_avoids_what_the_arm_camera_sees_in_the_bin():
    """Option A (2026-09-26): the arm camera looks into the bin from
    `moss_bin.BIN_LOOK_POSE`, `choose_drop_point` picks the clearest spot,
    and the stow sees it in `moss.OBS_DROP`. The look pose was chosen
    because the post-delivery pose sees NOTHING (0 of 71 items) — so assert
    the camera actually detects things, not just that a point comes back."""
    from microduck_local.robots import moss_bin
    from microduck_local.robots.moss_env import STOW_TARGET, MossStowEnv

    seen = placed = better = 0
    for seed in range(8):
        env = MossStowEnv(seed=seed, prop_variety=True, stow_rung=2,
                          bin_clutter=3, drop_target=True)
        obs, _ = env.reset()
        placed += env._clutter_in_bin()
        seen += env._drop_seen
        assert np.allclose(obs[moss.OBS_DROP], moss_bin.drop_obs(env._drop))
        assert np.allclose(env._target_base()[:2], env._drop)
        if not env._clutter:
            continue

        def clearance(xy):
            out = 9.0
            for k, p in enumerate(env._clutter):
                b = env.model.body(f"clutter{k}").id
                c = env._to_base(env.data.xpos[b][:2])
                out = min(out, float(np.hypot(c[0] - xy[0], c[1] - xy[1]))
                          - moss_bin.footprint_radius(p, env.data.xmat[b])
                          - env.prop.radius)
            return out
        better += clearance(env._drop) >= clearance(moss_bin.BIN_CENTRE)
    assert placed and seen >= 0.6 * placed, (seen, placed)
    assert better >= 6, better
    # an empty bin gets its centre; with the flag off, the old fixed target
    # and dead slots, exactly as every earlier stow trained
    env = MossStowEnv(seed=0, stow_rung=2, drop_target=True)
    env.reset()
    assert np.hypot(env._drop[0] - moss_bin.BIN_CENTRE[0],
                    env._drop[1] - moss_bin.BIN_CENTRE[1]) < 0.005
    env = MossStowEnv(seed=0, stow_rung=2, bin_clutter=3)
    obs, _ = env.reset()
    assert np.allclose(env._target_base(), STOW_TARGET)
    assert not np.any(obs[moss.OBS_DROP])


def test_the_command_leash_stops_a_blocked_joint_storing_a_whip():
    """A joint that cannot move while its command keeps advancing stores the
    difference and releases it when it comes free — measured at 10-14 rad/s
    on the fold (seed 466, jaw pinned by the dropped object and clutter), for
    the learned fold AND the scripted one. With `cmd_leash` the command never
    leads the measured position by more than the leash."""
    from microduck_local.robots.moss_env import MossStowEnv

    def lead_after_blocked_push(leash):
        env = MossStowEnv(seed=0, stow_rung=2, cmd_leash=leash)
        env.reset()
        j = moss.ARM_JOINTS[0]
        adr = env.model.joint(j).qposadr[0]
        a = np.zeros(env.action_space.shape, np.float32)
        a[0] = 1.0
        for _ in range(40):                    # push, while pinning the joint
            q = float(env.data.qpos[adr])
            env.step(a)
            env.data.qpos[adr] = q
            env.data.qvel[env.model.joint(j).dofadr[0]] = 0.0
        return abs(env.arm_cmd[j] - float(env.data.qpos[adr]))

    assert lead_after_blocked_push(0.0) > 0.5          # 40 x 0.03 stored
    assert lead_after_blocked_push(0.08) <= 0.08 + 0.03 + 1e-6


def test_the_brain_folds_with_the_learned_leg_at_25hz_on_a_leash():
    """`tidy_moss`'s tuck state runs the learned fold (when one is on disk)
    the way it trained: seeded from the MEASURED arm, stepped at
    `moss.CONTROL_HZ` not every 50 Hz world tick, and every goal kept within
    `fold_leash_rad` of the measured joint. Run per world tick, its command
    moved at 1.5 rad/s — the fast motion the leg exists to avoid."""
    import dataclasses

    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss

    b = TidyMoss()
    if "fold" not in b._sessions:
        pytest.skip("no fold policy on disk (runs/teach-moss_fold-*)")
    # The learned fold is the FALLBACK since the planned routes landed: with
    # `plan_routes` on, `tuck` goes home by a checked route and this leg never
    # runs (asserted below). Its own behaviour is still pinned here.
    b.p = dataclasses.replace(b.p, plan_routes=False, rest_pose=None)
    arm = dict(zip(moss.ARM_JOINTS, (-1.80, -0.97, 0.10, 1.55, -1.30)))
    arm[moss.GRIPPER_JOINT] = 0.03
    b._to("tuck", 0.0)
    goals, runs = [], 0
    for k in range(10):                               # 10 world ticks, 0.2 s
        before = b._last_policy_t
        intent = b.step(Senses(t=0.02 * k, odom=(0.0, 0.0, 0.0), speed=0.0,
                               arm=dict(arm)))
        runs += b._last_policy_t != before
        goals.append(intent.arm)
    assert b.state == "tuck"
    assert runs == 5, runs                            # 25 Hz, not 50
    for g in goals:                                   # the arm never moved,
        for j in moss.ARM_JOINTS:                     # so no goal may lead it
            assert abs(g[j] - arm[j]) <= b.p.fold_leash_rad + 1e-9, (j, g[j])
    assert goals[0][moss.GRIPPER_JOINT] == pytest.approx(moss.MISSION_OPEN_M,
                                                         abs=0.005)

    # ...and with the routes ON, the fold leg is not what runs: `tuck` goes
    # home by a route checked on the visible meshes instead. Measured: that
    # is what the 48-seed battery ran, and it took arm-on-rover contact from
    # 146 s a run to 6 s.
    b2 = TidyMoss()
    b2._to("tuck", 0.0)
    ran = 0
    for k in range(10):
        before = b2._last_policy_t
        b2.step(Senses(t=0.02 * k, odom=(0.0, 0.0, 0.0), speed=0.0,
                       arm=dict(arm)))
        ran += b2._last_policy_t != before
    assert b2.p.plan_routes and ran == 0, ran


def test_the_pick_runs_at_its_trained_25_hz_in_the_brain():
    """Stepped every 50 Hz world tick, the pick moved arm and jaw at twice the
    speed it trained at; held to 25 Hz, 478dad put 99 objects in the bin over
    24 yard seeds against 71 (2026-09-27)."""
    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss

    b = TidyMoss()
    if b._sess is None:
        pytest.skip("no pick policy on disk")
    arm = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
    arm[moss.GRIPPER_JOINT] = 0.041
    b._to("creep", 0.0)
    b._fix = (0.40, 0.0)
    runs = 0
    for k in range(10):
        before = b._last_policy_t
        b.step(Senses(t=0.02 * k, odom=(0.0, 0.0, 0.0), speed=0.0,
                      arm=dict(arm)))
        runs += b._last_policy_t != before
    assert b.state == "creep"
    assert runs == 5, runs                            # 25 Hz, not 50


def test_a_small_object_can_be_lifted_with_the_jaws_nearly_shut():
    """`min_grasp_m` (20 mm) is a CAN's number: jaws round a cap, block or
    card sit below it, so every grasp of one read as a pinch and the brain
    never lifted — caps, paper and butts reached a lift 0% of the time in the
    yard. For an object the detector sizes as small the floor is 0 (48 yard
    seeds: 281 v 225 in the bin); a big one keeps the pinch guard."""
    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss

    b = TidyMoss()
    arm = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
    arm[moss.GRIPPER_JOINT] = 0.006
    sn = Senses(t=0.0, odom=(0.0, 0.0, 0.0), speed=0.0, arm=arm)
    b._fix_size = 0.03                              # a cap
    assert b._grip_is_a_grasp(sn)
    b._fix_size = 0.12                              # a can: 6 mm is a pinch
    assert not b._grip_is_a_grasp(sn)


def test_the_carry_eases_in_and_out_of_every_leg():
    """A card held by its 4 mm edges sat still in the jaws and left them in
    one tick at a jerk of the carry. Easing every lift and stow leg
    (smoothstep: zero speed at both ends) put 382 objects in the bin over 48
    yard seeds against 358. Checked on the stow leg the brain actually runs."""
    from microduck_local.brain.tidy_moss import TidyMoss

    b = TidyMoss()
    assert b.p.carry_ease
    assert b._ease(0.0) == 0.0 and b._ease(1.0) == 1.0 and b._ease(0.5) == 0.5
    assert b._ease(0.1) < 0.05                           # starts gently
    b.state = "stow"
    b._carry_from = {j: 0.0 for j in moss.ARM_JOINTS}
    target = tuple(1.0 for _ in moss.ARM_JOINTS)
    early = b._ramp(target, 0.1, 1.0)
    assert all(v < 0.05 for v in early.values())         # a linear ramp: 0.1


def test_the_base_lines_up_for_the_pick_at_twice_the_old_speed():
    """Deploy took 26% of every yard run, and the arm was not what took it:
    it reaches the grasp pose in 1.2 s, then the base crept into the handover
    band at 0.06 m/s and 7 deploys in 20 ran to the 9 s cap. At 0.12 m/s,
    48 moss-yard seeds put 225 objects in the bin against 190."""
    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss

    b = TidyMoss()
    assert b.p.band_mps == pytest.approx(0.12)
    arm = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
    arm[moss.GRIPPER_JOINT] = 0.041
    b._to("deploy", 0.0)
    far = b.p.band_far_locked_m + 0.10              # beyond the band
    b._fix, b._fix_t = (far, 0.0), 0.0
    it = b.step(Senses(t=0.02, odom=(0.0, 0.0, 0.0), speed=0.0, arm=arm))
    assert b.state == "deploy"
    assert it.twist[0] == pytest.approx(0.12)


def _drop_on_the_lift(**kw):
    import dataclasses

    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss

    b = TidyMoss()
    b.p = dataclasses.replace(b.p, **kw)
    arm = dict(zip(moss.ARM_JOINTS, moss.LIFT_POSE))
    arm[moss.GRIPPER_JOINT] = 0.02
    b._to("lift", 0.0)
    t = 0.0
    while b.state == "lift" and t < 3.0:
        t += 0.02
        b.step(Senses(t=t, odom=(0.0, 0.0, 0.0), speed=0.0, arm=dict(arm),
                      holding=False))
    assert b.state == "tuck" and b._dropped, (b.state, t)
    assert t <= b.p.lift_drop_grace_s + b.p.stow_drop_grace_s + 0.2, t
    home = dict(zip(moss.ARM_JOINTS, b._rest()))
    rate = b.p.motion_vmax if b.p.plan_routes else b.p.drop_tuck_rate
    q = dict(arm)
    for _ in range(25):                        # 0.5 s, the arm follows exactly
        t += 0.02
        it = b.step(Senses(t=t, odom=(0.0, 0.0, 0.0), speed=0.0, arm=dict(q),
                           holding=False))
        for j in moss.ARM_JOINTS:              # never faster than the rate
            assert abs(it.arm[j] - q[j]) <= rate * 0.02 + 1e-6
            q[j] = it.arm[j]
    moved = sum(abs(q[j] - arm[j]) for j in moss.ARM_JOINTS)
    assert moved > 0.0 and all(abs(q[j] - home[j]) <= abs(arm[j] - home[j]) + 1e-9
                               for j in moss.ARM_JOINTS)


def test_a_can_dropped_on_the_lift_is_noticed_and_the_arm_goes_straight_home():
    """8 of 10 drops in the yard happen at the start of the carry and the
    empty-jaw check only ran in `stow`, so the brain noticed 2.8 s late and
    then took the fold's 7-12 s route round a bin the arm was nowhere near.
    Now: noticed in LIFT, and home — by a clear route to the rest pose
    (`plan_routes`, the lift pose joins it by a straight clear segment), or
    with routes off by the straight, rate-limited, leashed path to the old
    tuck."""
    _drop_on_the_lift()
    _drop_on_the_lift(plan_routes=False, rest_pose=None)


def test_a_valid_start_stow_episode_begins_with_the_object_in_the_jaws():
    """With six shapes, a stow reset handed back the object LYING ON THE
    FLOOR under an open jaw for ~7 of 8 draws — at every rung — and fell
    through silently (the 2026-09-26 bisect: rung 0 held 8/60). `_carrying`
    even said True for it. `valid_start` redraws, then falls back to the
    reference can, so every episode starts held and lifted."""
    from microduck_local.robots.moss_env import MossStowEnv

    for s in range(6):
        env = MossStowEnv(seed=s, stow_rung=0, prop_variety=True,
                          valid_start=True)
        env.reset()
        can = env.data.xpos[env.can_body]
        assert env.start_held and env._gripped_now(), (s, env.prop.id)
        assert can[2] > env.prop.half_height + 0.03, (s, env.prop.id, can[2])


def test_a_sphere_counts_as_picked_only_through_its_centre(monkeypatch):
    """A ball pinched ahead of its centre is squeezed out of the jaws, level
    or not: 478dad's env ball picks under 16 mm from the pad midpoint held
    through the lift and swing 33/33, over 20 mm 2/9. The criterion is
    recorded (run.json) and measured from the PADS, not the tool point."""
    from types import SimpleNamespace as NS

    import mujoco

    from microduck_local.robots.moss_env import GraspProp, MossPickEnv
    from microduck_local.robots.registry import registry

    body = registry()["moss"]
    monkeypatch.setenv("MICRODUCK_MOSS_SPHERE_CENTRE", "0.016")
    assert body.train_env_kwargs(NS(task="pick")).get("sphere_centre_m") == 0.016
    monkeypatch.setenv("MICRODUCK_MOSS_SPHERE_CENTRE", "0")
    assert body.train_env_kwargs(NS(task="pick")).get("sphere_centre_m") == 0.0
    ball = GraspProp(id="ball", shape="sphere", size=(0.025,), mass=0.02,
                     jaw_ctrl_m=moss.GRASP_JAW_CTRL_M,
                     grasp_height_m=moss.GRASP_HEIGHT_M)
    env = MossPickEnv(seed=0, prop=ball, sphere_centre_m=0.016)
    env.reset(seed=0)
    assert env.sphere_centre_m == 0.016
    m, d = env.model, env.data
    # ONE finger moved (not both, as the tied jaw would): the pad midpoint
    # then sits off the tool point, so the two references are told apart
    d.qpos[m.joint(moss.GRIPPER_JOINT).qposadr[0]] = 0.010
    mujoco.mj_forward(m, d)
    mid = (d.geom_xpos[m.geom("pad_left").id]
           + d.geom_xpos[m.geom("pad_right").id]) / 2.0
    d.qpos[env.can_qadr:env.can_qadr + 3] = mid
    mujoco.mj_forward(m, d)
    assert env._sphere_off_centre() < 1e-6
    d.qpos[env.can_qadr + 2] = mid[2] + 0.02
    mujoco.mj_forward(m, d)
    assert env._sphere_off_centre() == pytest.approx(0.02, abs=1e-6)


def test_the_progress_measure_a_pick_trained_on_is_recorded_and_honoured(monkeypatch):
    """`MICRODUCK_MOSS_GAP_TCP` was an import-time constant the trainer never
    passed and run.json never recorded. ad9876 trained with it ON (replay:
    +86/ep on, -3 off; its training said +88), so every fine-tune launched
    without it paid for pulling the object toward the CHASSIS and learned to
    drag objects in (+14..+17 cm in 55-60 of 60 episodes) instead of grasping.
    The trainer's kwargs must carry it, and the env must obey the kwarg."""
    from types import SimpleNamespace as NS

    from microduck_local.robots.moss_env import MossPickEnv
    from microduck_local.robots.registry import registry

    body = registry()["moss"]
    monkeypatch.setenv("MICRODUCK_MOSS_GAP_TCP", "1")
    assert body.train_env_kwargs(NS(task="pick")).get("gap_from_tcp") is True
    monkeypatch.setenv("MICRODUCK_MOSS_GAP_TCP", "0")
    assert body.train_env_kwargs(NS(task="pick")).get("gap_from_tcp") is False
    # the env obeys the KWARG, whatever the process environment says
    on = MossPickEnv(seed=0, gap_from_tcp=True)
    off = MossPickEnv(seed=0, gap_from_tcp=False)
    on.reset(seed=0)
    off.reset(seed=0)
    assert on.gap_from_tcp and not off.gap_from_tcp
    assert on._reward_gap() != off._reward_gap()


def test_the_litter_set_is_opt_in_and_draws_real_litter():
    """`litter` adds a crumpled-paper wad, a cigarette butt and a bottle cap
    to the six shapes; OFF, the draws are exactly the old six so every
    earlier run and evaluation replays. The wad must actually SIT (rolling
    friction is ignored under condim 3 — twice before in this repo)."""
    from microduck_local.robots.moss_env import LITTER_KINDS, MossPickEnv, sample_prop

    rng = np.random.default_rng(0)
    off = {sample_prop(rng).id for _ in range(300)}
    assert not off & set(LITTER_KINDS), off
    rng = np.random.default_rng(0)
    on = {sample_prop(rng, litter=True).id for _ in range(300)}
    assert set(LITTER_KINDS) <= on, on
    paper = next(p for p in (sample_prop(np.random.default_rng(s), True)
                             for s in range(200)) if p.id == "paper")
    assert paper.condim == 6 and paper.friction[2] >= 0.01
    env = MossPickEnv(seed=1, prop_variety=True, litter=True)
    assert env.litter


def test_a_light_object_dropped_into_the_empty_bin_stays_on_its_floor():
    """The cigarette butt (0.6 g, 8 mm) released at the stow's height lands
    on the 4 mm bin floor at ~2 m/s. MEASURED 2026-09-28: it went THROUGH
    the floor in one physics step, into the 8 mm pocket between the floor
    and the hull, rode there ~5 min counted as "in the bin", and slid out of
    the back while the robot drove — the one object lost from the bin in
    every moss-yard run. `bin_floor_governs` lets the floor's own (stiffer)
    contact catch it; his geometry is unchanged."""
    import json as _json

    from microduck_local.viz_server import load_policy_infer
    from microduck_local.world import scenario as S
    from microduck_local.world_server import WorldState
    raw = _json.loads(Path(__file__).resolve().parents[1].joinpath(
        "scenarios/moss-yard.json").read_text())
    w = WorldState(load_infer=load_policy_infer).build(S.Scenario.from_dict(raw), seed=0)
    r = w.ducks["m0"]
    m, d = w.model, w.data
    rover = m.body(r.prefix + moss.BASE_BODY).id
    butt = m.body("butt0")
    qa = int(m.jnt_qposadr[butt.jntadr[0]])
    va = int(m.jnt_dofadr[butt.jntadr[0]])
    for _ in range(5):
        w.step()
    R = d.xmat[rover].reshape(3, 3)
    drop = d.xpos[rover] + R @ np.array([-0.11, 0.0, 0.36])   # where a release lets go
    d.qpos[qa:qa + 3] = drop
    d.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[va:va + 6] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(75):                                        # 1.5 s
        w.step()
    q = R.T @ (d.xpos[butt.id] - d.xpos[rover])
    assert q[2] > moss.BIN_FLOOR_Z, f"under the bin floor: z {q[2]:.4f}"
    assert moss.BIN_INTERIOR_X[0] < q[0] < moss.BIN_INTERIOR_X[1]


def _stow_path_bin_contact(lift: float) -> float:
    """Deepest arm-into-bin penetration (m) along the scripted stow's three
    legs from the lift pose, with `shoulder_lift` held at `lift` over the
    turn — kinematically, the rover parked in moss-yard."""
    import json as _json

    from microduck_local.robots import moss_env as ME
    from microduck_local.viz_server import load_policy_infer
    from microduck_local.world import scenario as S
    from microduck_local.world_server import WorldState
    raw = _json.loads(Path(__file__).resolve().parents[1].joinpath(
        "scenarios/moss-yard.json").read_text())
    w = WorldState(load_infer=load_policy_infer).build(S.Scenario.from_dict(raw), seed=0)
    r = w.ducks["m0"]
    m, d = w.model, w.data
    for _ in range(5):
        w.step()
    qa = [m.joint(r.prefix + j).qposadr[0] for j in moss.ARM_JOINTS]
    bins = {m.geom(r.prefix + n).id for n in moss.HULL_GEOMS if n.startswith("bin_")}
    arm_bodies = {i for i in range(m.nbody) if any(
        k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")
        for k in ("upper_arm", "lower_arm", "wrist", "gripper", "jaw"))}
    high = (ME.STOW_HIGH[0], lift) + tuple(ME.STOW_HIGH[2:])
    turned = (ME.STOW_TURNED[0], lift) + tuple(ME.STOW_TURNED[2:])
    q0, deepest = d.qpos.copy(), 0.0
    for a, b in ((moss.LIFT_POSE, high), (high, turned), (turned, ME.STOW_RELEASE_HIGH)):
        for s in np.linspace(0.0, 1.0, 41):
            d.qpos[:] = q0
            for adr, x, y in zip(qa, a, b):
                d.qpos[adr] = (1 - s) * x + s * y
            mujoco.mj_forward(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                pair = {(c.geom1 in bins), (c.geom2 in bins)}
                arm = {int(m.geom_bodyid[c.geom1]) in arm_bodies,
                       int(m.geom_bodyid[c.geom2]) in arm_bodies}
                if pair == {True, False} and True in arm and c.dist < 0:
                    deepest = max(deepest, -float(c.dist))
    return deepest


def test_the_stow_swings_round_clear_of_the_bin():
    """THE POP: the stow's waypoints held `shoulder_lift` at -1.35, behind the
    bin's front wall — the upper arm stalled on `bin_x1` at 17 N 0.22 rad
    short, scraped it through the turn and snapped free at ~2.8 rad/s
    (70 rad/s^2) on every delivery. The brain's default holds it where the
    whole path is clear; the old value is the control."""
    from microduck_local.brain.tidy_moss import TidyMossParams
    lift = TidyMossParams().stow_clear_lift
    assert lift is not None
    assert _stow_path_bin_contact(lift) == 0.0
    assert _stow_path_bin_contact(-1.35) > 0.004
