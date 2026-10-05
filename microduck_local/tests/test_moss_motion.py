"""MOSS's arm between poses (`brain/moss_motion.py`, `TidyMoss.rest_pose` /
`plan_routes` / `wrist_scan`): a rest pose clear of its own body and out of
the front camera's way, clear routes to and from it, followed together, and a
wrist camera that scans from it."""
import math

import numpy as np
import pytest

from microduck_local.brain.tidy_moss import TidyMossParams
from microduck_local.robots import moss

pytestmark = pytest.mark.skipif(
    not moss.moss_ready(),
    reason="MOSS assets missing — uv run fetch-robot moss")


@pytest.fixture(scope="module")
def clear():
    from microduck_local.brain.moss_motion import ArmClearance
    return ArmClearance()


def _rest():
    return np.array(TidyMossParams().rest_pose)


def test_the_old_tuck_sits_on_the_bin_and_the_rest_pose_clears_it(clear):
    """The screenshot on /sim: the old tuck has the gripper's mesh ON the
    bin's front wall, so every move out of it began in contact."""
    assert clear.clearance(moss.tuck_pose()) <= 0.001
    assert clear.clearance(_rest()) >= 0.015


def test_the_rest_pose_stays_inside_the_turning_circle_and_out_of_the_front_camera(clear):
    """Out of the circle the chassis sweeps (0.222 m, the tracks) the resting
    arm caught a wall while the robot turned to search, was dragged to the
    pan limit and pinned the robot for two minutes. And nothing of it may
    stand in the front camera's view — that is the colour detector AND the
    depth that maps the room."""
    m, d = clear.m, clear.d
    clear.clearance(_rest())                    # poses the model
    rover = m.body(moss.BASE_BODY).id
    R, o = d.xmat[rover].reshape(3, 3), d.xpos[rover]
    cam = m.body(moss.CAMERA_BODY).id
    Rc, pc = d.xmat[cam].reshape(3, 3), d.xpos[cam]
    hf = math.radians(moss.CAMERA_HFOV_DEG / 2)
    vf = math.radians(moss.CAMERA_VFOV_DEG / 2)
    for g in clear.vis:
        mid = m.geom_dataid[g]
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        W = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        B = (W - o) @ R
        assert np.hypot(B[:, 0], B[:, 1]).max() < 0.222
        L = (W - pc) @ Rc
        seen = ((L[:, 0] > 0.005) & (np.abs(np.arctan2(L[:, 1], L[:, 0])) < hf)
                & (np.abs(np.arctan2(L[:, 2], L[:, 0])) < vf))
        assert not seen.any()
    # ...and the wrist camera looks OUT, not at the track
    wc = m.body(moss.ARM_CAMERA_BODY).id
    ax = R.T @ d.xmat[wc].reshape(3, 3)[:, 0]
    assert math.hypot(ax[0], ax[1]) > 0.8 and ax[2] < 0


def test_the_map_is_told_the_floor_the_wrist_camera_can_really_report():
    """The band, against a brute-force grid of the floor in the camera's view
    AND inside its 0.60 m range. The /sim map drew a full wedge from the lens
    out to 0.60 m, claiming the floor beside the tracks — which is below the
    camera's vertical view. Overstating a sensor's reach is exactly how this
    camera went 48 seeds looking like it worked."""
    from microduck_local.brain.tidy_moss import _wrist_floor_band
    from microduck_local.robots.moss_pinch import MossKinematics
    kin = MossKinematics()
    rest = dict(zip(moss.ARM_JOINTS, TidyMossParams().rest_pose))
    pc, Rc = kin.body_in_base(rest, 0.041, moss.ARM_CAMERA_BODY)
    near, far = _wrist_floor_band(pc, Rc)
    hf = math.radians(moss.ARM_CAMERA_HFOV_DEG / 2)
    vf = math.radians(moss.ARM_CAMERA_VFOV_DEG / 2)
    g = 0.004
    X, Y = np.meshgrid(np.arange(-1.0, 1.0, g), np.arange(-1.0, 1.0, g), indexing="ij")
    P = np.stack([X, Y, np.full_like(X, 0.03)], -1) - pc
    L = P @ Rc
    ok = ((L[..., 0] > 0.01)
          & (np.abs(np.arctan2(L[..., 1], L[..., 0])) < hf)
          & (np.abs(np.arctan2(L[..., 2], L[..., 0])) < vf)
          & (np.linalg.norm(P, axis=-1) <= moss.ARM_CAMERA_MAX_RANGE_M))
    assert ok.any()
    h = np.hypot(X - pc[0], Y - pc[1])
    assert near == pytest.approx(float(h[ok].min()), abs=0.005)
    assert far == pytest.approx(float(h[ok].max()), abs=0.005)
    assert 0.05 < near < far <= moss.ARM_CAMERA_MAX_RANGE_M


def test_rest_joins_the_grasp_release_and_lift_poses_by_straight_clear_segments(clear):
    rest = _rest()
    release = (-1.875, -0.695, -0.264, 1.438, -0.05)
    for a, b in ((rest, moss.GRASP_POSE), (release, rest), (moss.LIFT_POSE, rest)):
        assert clear.route(a, b, margin=0.010) is not None
        assert len(clear.route(a, b, margin=0.010)) == 2      # direct
    # the old sweep out (tuck -> grasp) is not clear at all
    assert not clear.segment(moss.tuck_pose(), moss.GRASP_POSE, 0.010)


def test_the_distance_cap_keeps_mj_geomDistance_out_of_its_false_contact_range(clear):
    """`mj_geomDistance` answers 0.0 — touching — for mesh/box pairs that are
    plainly apart once the cutoff grows. At the rest pose the upper arm reads
    0.0 against the bin's front wall at any cutoff >= 0.10 while its nearest
    vertex is 85 mm from that box. A false 0.0 fails `segment`, so a clear
    route is refused: at the first cut's 0.03 cap these pairs, which the brain
    routes between every cycle, had no route at all."""
    from microduck_local.brain.moss_motion import CAP_MAX
    rest = _rest()
    for a, b in ((moss.LIFT_POSE, TidyMossParams().fold_route_wp),
                 (TidyMossParams().fold_route_wp, moss.LIFT_POSE),
                 (moss.RETRACT_WAYPOINT, moss.GRASP_POSE)):
        assert clear.route(a, b, margin=0.010) is not None
    # the cap is CLAMPED, not trusted: asking for the broken range is safe
    assert clear.clearance(rest, cap=0.5) == clear.clearance(rest)
    assert clear.clearance(rest) >= 0.015
    assert CAP_MAX <= 0.02


def test_a_min_jerk_route_peaks_at_its_speed_and_ends_where_it_should():
    from microduck_local.brain.moss_motion import MinJerkRoute
    a, b = np.zeros(5), np.array([1.2, -0.6, 0.3, 0.0, 0.9])
    r = MinJerkRoute([a, b], vmax=1.2)
    ts = np.arange(0.0, r.total + 0.02, 0.005)
    q = np.array([r.at(t) for t in ts])
    v = np.abs(np.diff(q, axis=0)).max(axis=1) / 0.005
    assert v.max() <= 1.2 + 1e-3 and v.max() > 1.1
    assert np.allclose(r.at(r.total + 1.0), b) and np.allclose(r.at(0.0), a)


def test_the_follower_waits_for_a_lagging_joint_instead_of_leaving_the_path():
    """The first cut leashed each joint on its own: a lagging base joint let
    the others run ahead, off the checked line, into the bin."""
    from microduck_local.brain.moss_motion import MinJerkRoute, RouteFollower
    a, b = np.zeros(5), np.array([1.0, 0.5, 0.0, 0.0, 0.0])
    f = RouteFollower(MinJerkRoute([a, b], vmax=1.0), leash=0.1)
    arm = dict(zip(moss.ARM_JOINTS, a))
    held = None
    for k in range(1, 200):                     # the arm never moves: blocked
        q = f.step(0.02 * k, arm)
        held = q if held is None else held
        assert np.abs(q - a).max() <= 0.1 + 1e-9  # never winds up past the leash
    assert not f.done() and f.tau < 0.5
    for k in range(200, 600):                   # freed: it follows to the end
        q = f.step(0.02 * k, dict(zip(moss.ARM_JOINTS, q)))
    assert f.done() and np.allclose(q, b)


def test_the_wrist_camera_scans_into_the_memory_only_at_rest():
    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss
    from microduck_local.robots.moss_pinch import MossKinematics
    from microduck_local.sensors.detector import Detection, DetectionFrame
    b = TidyMoss()
    b._to("search", 0.0)
    rest = dict(zip(moss.ARM_JOINTS, b._rest()))
    kin = MossKinematics()
    pc, Rc = kin.body_in_base(rest, 0.041, moss.ARM_CAMERA_BODY)
    # a can on the floor straight down the wrist camera's axis
    ax = Rc[:, 0]
    s = -pc[2] / ax[2] * 0.97
    spot = pc + ax * s
    rng = float(np.linalg.norm(spot - pc))
    rad = b.p.can_radius_m
    det = Detection("toy", "can9", 0.0, 0.0, 2.0 * math.atan(rad / rng), rng, 1.0)
    for k, arm in enumerate((dict(rest), {**rest, "shoulder_lift": rest["shoulder_lift"] + 0.3})):
        b2 = TidyMoss() if k else b
        b2._to("search", 0.0)
        for i in range(3):
            b2.step(Senses(t=0.02 * (i + 1), odom=(0.0, 0.0, 0.0), speed=0.0, arm=arm,
                           arm_det=DetectionFrame(t=0.02 * i, detections=[det])))
        got = [e for e in b2._mem.items if b2._mem.confirmed(e)]
        if k == 0:
            assert len(got) == 1
            assert math.hypot(got[0].x - spot[0], got[0].y - spot[1]) < 0.05
        else:
            assert got == []                   # not resting: not scanning


def test_the_wrist_camera_does_not_scan_while_the_arm_is_carrying_something():
    """MEASURED: dropped for a joint-speed gate, scanning from any pose put
    5-8 phantoms per run into the memory — the camera stares at the object IN
    THE JAWS during `lift`, and the kinematics place it on the floor 0.3 m
    away. Picks fell (10/10/8 against 9/12/10 on three seeds). The state gate
    is what keeps the camera pointed at floor."""
    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss
    from microduck_local.robots.moss_pinch import MossKinematics
    from microduck_local.sensors.detector import Detection, DetectionFrame
    kin = MossKinematics()
    for state in ("lift", "stow", "release", "creep", "deploy", "tuck"):
        b = TidyMoss()
        b._to(state, 0.0)
        rest = dict(zip(moss.ARM_JOINTS, b._rest()))
        pc, Rc = kin.body_in_base(rest, 0.041, moss.ARM_CAMERA_BODY)
        ax = Rc[:, 0]
        spot = pc + ax * (-pc[2] / ax[2] * 0.97)
        rng = float(np.linalg.norm(spot - pc))
        det = Detection("toy", "can9", 0.0, 0.0,
                        2.0 * math.atan(b.p.can_radius_m / rng), rng, 1.0)
        for i in range(4):
            b.step(Senses(t=0.02 * (i + 1), odom=(0.0, 0.0, 0.0), speed=0.0,
                          arm=dict(rest),
                          arm_det=DetectionFrame(t=0.02 * i, detections=[det])))
        assert b._mem.items == [], f"the wrist scanned in {state}"


def test_in_the_yard_the_brain_is_handed_wrist_camera_frames():
    """The wrist detector existed and the /sim overlay drew its cone, but the
    world never SAMPLED it: `senses.arm_det` was None on every tick and a
    48-seed A/B of wrist scanning came out identical, seed for seed, to not
    scanning. The unit test above hands the brain frames; this asks the
    world for them."""
    import json
    from pathlib import Path

    from microduck_local.viz_server import load_policy_infer
    from microduck_local.world import scenario as S
    from microduck_local.world_server import WorldState
    raw = json.loads(Path(__file__).resolve().parents[1].joinpath(
        "scenarios/moss-yard.json").read_text())
    st = WorldState(load_infer=load_policy_infer)
    w = st.build(S.Scenario.from_dict(raw), seed=0)
    st.world = w
    r = w.ducks["m0"]
    for _ in range(10):
        w.step()
    s = st.senses_for(r)
    assert s.arm_det is not None and s.arm_det.t > 0.0


# --- the FLAT-object pinch (`pinch_flat_m`, `pinch_wide_m`) -----------------
# `moss-yard`'s card0 (60 x 40 x 4 mm) was left on the floor in 47 of 48 runs.
# It is not a missing skill: the card is inside the pick env's own training
# range, and IK'd onto it the gripper lifts it +215 mm. It failed because the
# pinch was gated on WIDTH — the card reads 0.030-0.060 wide, so it fell to the
# learned pick, which grips at a can's 50 mm with a 62 mm gap, 38 mm above it.

def _det(cls, bearing, elevation, width, rng):
    from microduck_local.sensors.detector import Detection
    return Detection(cls, "x", bearing, elevation, width, rng, 1.0)


def _feed(b, z, n, rng=0.30, width=0.12):
    """n head-camera frames of one toy whose implied height is `z`."""
    from microduck_local.brain.runtime import Senses
    from microduck_local.sensors.detector import DetectionFrame
    elev = math.asin(max(-1.0, min(1.0, (z - moss.CAMERA_POS[2]) / rng)))
    for i in range(n):
        b.step(Senses(t=0.02 * (i + 1), odom=(0.0, 0.0, 0.0), speed=0.0,
                      det=DetectionFrame(t=0.02 * i,
                                         detections=[_det("toy", 0.0, elev, width, rng)])))


def test_the_head_camera_gives_the_brain_each_toys_HEIGHT():
    """The camera is level with the base (measured: pitch -0.0 deg), so
    `CAMERA_POS[2] + range * sin(elevation)` is the object's height. Over 11
    yard props the median error is <= 5 mm, and it separates card/butt/cap at
    0.004-0.007 from block/paper/ball/squat at 0.020+ and the cans at 0.058."""
    from microduck_local.brain.tidy_moss import TidyMoss
    b = TidyMoss()
    b._to("search", 0.0)
    _feed(b, 0.004, 6)
    assert b._fix_height is not None
    assert abs(b._fix_height - 0.004) < 0.001


def test_the_flat_gate_reads_a_MEDIAN_and_keeps_each_targets_own_readings():
    """One tick's height is noisy — card0 reads -0.016..0.020 across a run
    while its median is 0.004. Gating on a single reading pinched `paper0`, a
    22 mm cube, 8 times in one run and missed every one; and ONE list shared
    across objects mixed the cube's readings into the card's."""
    from microduck_local.brain.tidy_moss import TidyMoss
    b = TidyMoss()
    b._to("search", 0.0)
    _feed(b, 0.024, 12)                       # a cube, with...
    _feed(b, 0.001, 1)                        # ...one stray low reading
    assert not b._flat_target(), "one stray tick flipped the gate"
    # a different target 0.5 m away starts its own list
    b2 = TidyMoss()
    b2._to("search", 0.0)
    _feed(b2, 0.024, 12)
    n_before = len(b2._fix_heights)
    b2._fix_heights_at = (9.0, 9.0)           # as if the target jumped
    _feed(b2, 0.004, 6)
    assert len(b2._fix_heights) < n_before, "the new target inherited the old readings"
    assert b2._flat_target()


def test_only_a_WIDE_flat_thing_gets_the_deep_descend_and_the_long_axis_grip():
    """The card needs the pads pressed below the floor clearance and the jaws
    on its LONG side. The 8 mm butt and the 12 mm cap are flat too and need
    NEITHER — giving them the deep descend left `cap0` on the floor in 7 of 11
    runs, and letting the butt read as wide left IT on the floor 11 of 20
    against 0. So both ride on flat AND wider than `pinch_wide_m`, and that
    threshold sits in a MEASURED gap: within pick range each prop's apparent
    size is all but exact (p25 = median = p75) — cap 0.015, butt 0.030, block
    0.040, paper 0.044, ball/squat 0.050, card 0.060, cans 0.115."""
    from microduck_local.brain.tidy_moss import TidyMoss, TidyMossParams
    p = TidyMossParams()
    assert p.pinch_flat_floor_m < p.pinch_floor_m      # deeper, and negative
    assert p.pinch_flat_m > 0.0 and p.pinch_wide_m > 0.0
    b = TidyMoss()

    def wide(size):
        b._fix_sizes = [size] * 8
        return b._wide_target()

    assert wide(0.060)                                 # card0, the one target
    for other in (0.015, 0.030, 0.040, 0.044, 0.050):  # cap, butt, block,
        assert not wide(other), other                  # paper, ball/squat
    b._fix_sizes = [0.060] * 3                         # too few to judge
    assert not b._wide_target()


def test_the_jaws_close_ALONG_a_wide_flat_things_long_axis():
    """The most expensive naming trap in this brain: `pinch_grip="across"`
    reads like the right choice for a card and is the wrong one. Measured in
    the room it put the jaw axis 85-89 deg from the card's long axis on EVERY
    close — squeezing its 40 mm short side, which lifted it 0/15 on the bench;
    "along" drops that error to 1.4 deg. The butt must keep the global, which
    is "along" for its 8 mm diameter."""
    from microduck_local.brain.tidy_moss import TidyMoss
    b = TidyMoss()
    b._target_flat, b._fix_sizes = True, [0.060] * 8      # the card
    assert b._grip_axis() == "along"
    b._target_flat, b._fix_sizes = True, [0.030] * 8      # the butt: flat, small
    assert b._grip_axis() == b.p.pinch_grip
    b._target_flat, b._fix_sizes = False, [0.115] * 8     # a can
    assert b._grip_axis() == b.p.pinch_grip


def test_no_tidy_prop_is_thinner_than_the_jaws_can_close_on():
    """MOSS's pads stop 8 mm apart fully shut and are 36 mm tall, so they rest
    ON the floor — a prop much thinner than that cannot be clamped at all, only
    brushed along the floor.

    `moss-yard`'s `card0` used to be a 60 x 40 x 4 mm rigid plate, and that is
    why it was left behind in 47 of 48 runs: not a skill the robot lacked, a
    shape no gripper of this geometry can pick up. MEASURED by remodelling it
    with the SAME mass and volume as a card that is actually crumpled — folded
    in quarters, 30 x 20 x 16 mm — over 16 paired seeds: card0 binned 16/16
    against 10/16, and the whole room got cleaner, +0.50 +- 0.18 objects. The
    repo's own note already said as much: "most real litter is crumpled rather
    than flat".
    """
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    shut_gap = 0.008              # pad inner faces, fully closed
    seen = 0
    for path in sorted((root / "scenarios").glob("*.json")):
        name = path.stem
        raw = json.loads(path.read_text())
        for prop in raw.get("props", []) or []:
            if prop.get("cls") != "toy":
                continue
            size = prop.get("size")
            if not isinstance(size, list) or len(size) < 3:
                continue          # spheres and cylinders carry their own form
            height = float(size[2])
            seen += 1
            assert height >= shut_gap, (
                f"{name}: {prop['id']} is {height * 1000:.0f} mm tall — thinner "
                f"than the {shut_gap * 1000:.0f} mm the jaws shut to, so the pads "
                "can only brush it along the floor")
    assert seen >= 5, f"only {seen} box toys checked — is the glob still right?"
