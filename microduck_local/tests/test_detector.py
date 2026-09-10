"""Locks for the geometric detector (roadmap 1.3): frustum, occlusion, size
gating, class pass-through, noise presets, and the capture-to-availability
latency — on real ducks composed into one world."""

import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.sensors import Detector, DetectorNoise, DetectorSpec, Target
from microduck_local.world import Ball, Duck, Scenario, Wall, compose
from microduck_local.world.compose import DuckAddress, spawn_duck

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")

# THE GEOMETRY THE CHARACTERISATIONS BELOW WERE WRITTEN AGAINST: 62° × 48° at
# 320 px, 295.7 px/rad, read as a pinhole.
#
# It was `DetectorSpec()`'s default until 2026-09-09, when the fitted module
# turned out to be 116° × 60° (docs/camera-hardware.md §1) and the default moved
# to it. Every test here that asserts *what the gates do* — which sizes are
# marginal, which elevations fall outside the frame, what a wide lens costs
# against a narrow one — is a statement about THIS geometry, so it names it
# rather than inheriting whatever the default happens to be.
#
# That is the bug these tests had. A characterisation that silently follows a
# default stops characterising anything the day the default moves, and reports
# it as nine unrelated-looking failures instead of "the camera changed".
NARROW_REF = DetectorSpec(fov_h_deg=62.0, fov_v_deg=48.0, px_h=320, projection="pinhole")


def world(ducks, walls=(), balls=()):
    sc = Scenario(name="det", floor=(8, 8), walls=list(walls), balls=list(balls),
                  ducks=[Duck(i, s, None, None) for i, s in ducks])
    m = compose(sc)
    d = mujoco.MjData(m)
    for i, s in ducks:
        spawn_duck(m, d, DuckAddress.resolve(m, i), *s)
    mujoco.mj_forward(m, d)
    return m, d


def targets(m, ducks=(), balls=()):
    out = [Target(i, "duck", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{i}/trunk_base"), 0.10) for i in ducks]
    out += [Target(f"ball{k}", "ball", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ball{k}"), 0.035) for k in balls]
    return out


def test_camera_site_is_x_forward_and_sees_a_duck_ahead_not_behind():
    m, d = world([("a", (0, 0, 0)), ("b", (0.6, 0.0, math.pi)), ("c", (-0.6, 0.0, 0.0))])
    det = Detector(m, site="a/head_camera", targets=targets(m, ducks=("a", "b", "c")))
    f = det.capture(d, 0.0)
    names = {x.name: x for x in f.detections}
    assert "b" in names and "c" not in names and "a" not in names   # ahead yes, behind no, self never
    b = names["b"]
    assert abs(b.bearing) < 0.05 and -0.4 < b.elevation < 0     # its trunk sits below the camera
    assert 0.35 < b.range_est < 0.8 and b.conf == 1.0


def test_bearing_sign_is_left_positive_and_fov_limits():
    m, d = world([("a", (0, 0, 0)), ("l", (0.5, 0.25, 0)), ("r", (0.5, -0.25, 0)), ("far", (0.3, 0.8, 0))])
    det = Detector(m, site="a/head_camera", targets=targets(m, ducks=("l", "r", "far")))
    names = {x.name: x for x in det.capture(d, 0.0).detections}
    assert names["l"].bearing > 0.3 and names["r"].bearing < -0.3
    assert "far" not in names          # ~70° off-axis: outside a 62° FOV


def test_wall_occludes_and_size_gates_at_range():
    m, d = world([("a", (0, 0, 0)), ("b", (1.2, 0.0, math.pi))], walls=[Wall((0.6, -1), (0.6, 1), 0.6)])
    det = Detector(m, site="a/head_camera", targets=targets(m, ducks=("b",)))
    assert det.capture(d, 0.0).detections == []
    m, d = world([("a", (0, 0, 0)), ("b", (3.9, 0.0, math.pi))])
    det = Detector(m, site="a/head_camera", targets=targets(m, ducks=("b",)), seed=0,
                   spec=NARROW_REF)
    # A 10 cm radius at 3.9 m is ~2.9° wide: found only sometimes, with low
    # confidence — through NARROW_REF. The fitted 116° lens at 640 px resolves it
    # more finely and finds it more often; that is the camera changing, not the gate.
    found = [det.capture(d, 0.0).detections for _ in range(200)]
    hits = [f[0] for f in found if f]
    assert 0 < len(hits) < 200
    # conf = p_find × U(floor, 1) with p_find ≈ 0.66 at 2.9°: found about two
    # times in three, never with more confidence than that.
    assert 0.4 < len(hits) / 200 < 0.9 and all(h.conf < 0.7 for h in hits)


def test_ball_class_and_range_from_width():
    # A floor ball half a metre out is BELOW a 48° vertical field of view from
    # a camera 25 cm up (−26°): the head has to look down for it. At 0.8 m
    # it just makes it in.
    m, d = world([("a", (0, 0, 0))], balls=[Ball((0.5, 0.0))])
    det = Detector(m, site="a/head_camera", targets=targets(m, balls=(0,)), spec=NARROW_REF)
    assert det.capture(d, 0.0).detections == []
    m, d = world([("a", (0, 0, 0))], balls=[Ball((0.8, 0.0))])
    det = Detector(m, site="a/head_camera", targets=targets(m, balls=(0,)), spec=NARROW_REF)
    f = det.capture(d, 0.0).detections
    assert len(f) == 1 and f[0].cls == "ball"
    assert -0.4 < f[0].elevation < -0.2
    assert abs(f[0].range_est - math.hypot(0.8 - 0.064, 0.21)) < 0.08


def test_latency_and_rate():
    m, d = world([("a", (0, 0, 0)), ("b", (0.6, 0.0, math.pi))])
    det = Detector(m, site="a/head_camera", targets=targets(m, ducks=("b",)),
                   noise=DetectorNoise(latency_s=0.05), spec=DetectorSpec(rate_hz=10.0))
    assert det.sample(d, 0.00) is None                  # captured at 0, not available yet
    assert det.sample(d, 0.02) is None and det.last is None
    got = det.sample(d, 0.06)
    assert got is not None and got.t == 0.0 and det.age(0.06) == pytest.approx(0.06)
    assert det.sample(d, 0.08) is None                  # next capture is due at 0.1 …
    assert det.sample(d, 0.12) is None                  # … taken on this first tick after, available at 0.17
    assert det.sample(d, 0.16) is None
    assert det.sample(d, 0.18).t == pytest.approx(0.12)


def test_noise_presets_bearing_spread_and_ghosts():
    m, d = world([("a", (0, 0, 0)), ("b", (0.6, 0.0, math.pi))])
    tg = targets(m, ducks=("b",))
    ideal = Detector(m, site="a/head_camera", targets=tg).capture(d, 0.0).detections[0]
    ds = Detector(m, site="a/head_camera", targets=tg, noise=DetectorNoise.datasheet(), seed=2)
    ho = Detector(m, site="a/head_camera", targets=tg, noise=DetectorNoise.hostile(), seed=2)
    b_ds, ghosts_ds, miss_ds, miss_ho = [], 0, 0, 0
    for _ in range(300):
        f = ds.capture(d, 0.0).detections
        real = [x for x in f if x.name == "b"]
        ghosts_ds += len(f) - len(real)
        miss_ds += not real
        b_ds += [x.bearing for x in real]
        miss_ho += not [x for x in ho.capture(d, 0.0).detections if x.name == "b"]
    assert abs(np.mean(b_ds) - ideal.bearing) < 0.01
    assert 0.01 < np.std(b_ds) < 0.03                   # ~1° bearing noise
    assert 0 < miss_ds < 30 and ghosts_ds >= 1
    assert miss_ho > miss_ds
    with pytest.raises(ValueError):
        DetectorNoise.preset("lidar")


def test_frames_carry_the_camera_pose_and_toys_do_not_occlude():
    """A detection frame says where the camera was (height, depression) —
    the brain ranges floor objects from it — and a toy in the beak does
    not hide the basket behind it (toys are geom group 4, see compose)."""
    from microduck_local.world import Basket, Duck, Pickable, Scenario, World
    sc = Scenario(name="occl", floor=(4, 4), ducks=[Duck("d0", (0, 0, 0), None, "ideal", "ideal")],
                  pickables=[Pickable("t0", "block", (0.6, 0.0))], basket=Basket((1.5, 0.0), (0.3, 0.3), 0.06))
    w = World(sc)
    d = w.ducks["d0"]
    fr = d.detector.capture(w.data, 0.0)
    assert 0.15 < fr.cam_z < 0.3 and -0.1 < fr.cam_pitch < 0.5     # head height; level at the spawn pose
    # The full capture pose: the site's position, and a quaternion whose x is the optical axis.
    assert len(fr.cam_pose) == 7 and abs(fr.cam_pose[2] - fr.cam_z) < 1e-9
    qw, qx, qy, qz = fr.cam_pose[3:]
    fwd = np.array([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)])
    assert abs(-np.arcsin(fwd[2]) - fr.cam_pitch) < 1e-6 and abs(np.linalg.norm(fwd) - 1) < 1e-6
    classes = set()
    for _ in range(20):                                             # a 2 cm toy at 0.6 m is found ~9 frames in 10
        classes |= {x.cls for x in d.detector.capture(w.data, 0.0).detections}
    assert "toy" in classes and "basket" in classes
    # Park the toy right on the line of sight to the basket marker: still seen.
    import mujoco
    q = w.model.jnt_qposadr[mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_JOINT, "t0_free")]
    cam = w.data.site_xpos[d.detector.site_id]
    mk = w.data.xpos[mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "basket_marker")]
    w.data.qpos[q:q + 3] = cam + 0.3 * (mk - cam)
    mujoco.mj_forward(w.model, w.data)
    assert "basket" in {x.cls for x in d.detector.capture(w.data, 0.0).detections}


def test_a_person_is_seen_by_its_legs_up_close_and_a_point_target_is_not():
    """A person is a vertical extent: the part inside the 48 deg frustum is
    reported (its middle leaves the frustum at about 1.2 m from a 24 cm
    camera). A point-like target of the same height is lost there."""

    from microduck_local.world import Duck, Person, Scenario, World
    sc = Scenario(name="near", floor=(6, 6), ducks=[Duck("d0", (0.0, 0.0, 0.0), None, None, "ideal")],
                  persons=[Person("p0", (0.5, 0.0), 0.0, [], 0.0, 0.18, 1.6)])
    w = World(sc)
    d = w.ducks["d0"]
    for _ in range(6):
        w.step()
    fr = d.detector.last
    people = [x for x in fr.detections if x.cls == "person"]
    assert len(people) == 1 and abs(people[0].bearing) < 0.05
    assert abs(people[0].elevation) <= np.deg2rad(24.0) + 1e-6      # reported inside the frustum
    assert 0.4 < people[0].range_est < 0.9                           # width-ranged, at the person
    # The same body as a point target at 0.8 m up: gone.
    tgt = next(t for t in d.detector.targets if t.cls == "person")
    origin = np.ascontiguousarray(w.data.site_xpos[d.detector.site_id], dtype=np.float64)
    R = w.data.site_xmat[d.detector.site_id].reshape(3, 3)
    from dataclasses import replace
    assert d.detector._visible(w.data, replace(tgt, height=0.0), origin, R) is None
    assert d.detector._visible(w.data, tgt, origin, R) is not None


def test_size_gate_thresholds_are_pixel_widths_read_through_the_lens():
    """The two apparent-width thresholds are ~5 px and ~21 px of the NPU's
    320 px frame, so they belong to the SENSOR and not to the angle. At the
    shipped 62° / 320 px they come back bit-for-bit as the shipped 1° and 4°
    (no measurement on record moves); the same 320 px behind a 120° lens
    resolves 1.94× coarser, so both thresholds widen 1.94× — to 1.94° and
    7.74°. Pixels buy the resolution back; an explicit angle still wins."""
    from dataclasses import replace

    s = NARROW_REF
    assert s.px_h == 320 and s.px_per_rad == pytest.approx(295.72, abs=0.01)
    assert s.w_none == float(np.deg2rad(1.0)) and s.w_full == float(np.deg2rad(4.0))   # bit-for-bit
    assert (s.w_none, s.w_full) == (s.w_none_rad, s.w_full_rad) and s._px_coarseness == 1.0
    # px_h named, not inherited: the claim is "same 320 px, wider lens", and it
    # is only that claim if both sides hold the pixels fixed.
    wide = DetectorSpec(fov_h_deg=120.0, fov_v_deg=93.0, px_h=320, projection="pinhole")
    assert wide.w_none / s.w_none == pytest.approx(120.0 / 62.0)      # ~2×, from the pixels alone
    assert wide.w_full / s.w_full == pytest.approx(120.0 / 62.0)
    assert np.rad2deg(wide.w_none) == pytest.approx(1.935, abs=1e-3)
    assert np.rad2deg(wide.w_full) == pytest.approx(7.742, abs=1e-3)
    assert DetectorSpec(fov_h_deg=120.0, px_h=640).w_none < s.w_none  # a 640 px sensor sees finer than shipped
    # An explicit `w_none_rad` does NOT bypass the pixel reading — `w_none` scales
    # it by coarseness like any other. That was invisible while the default WAS
    # the reference (coarseness 1.0), and the docstring's "an explicit angle still
    # wins" read it the wrong way round. On the reference it comes back unchanged;
    # on the fitted 116° / 640 px camera the same input reads ~6% finer.
    assert replace(s, w_none_rad=np.deg2rad(2.0)).w_none == pytest.approx(np.deg2rad(2.0))
    assert DetectorSpec(w_none_rad=np.deg2rad(2.0)).w_none < np.deg2rad(2.0)
    # `replace` carries the pixel meaning — it is how a field-of-view sweep patches the spec.
    assert replace(s, fov_h_deg=120.0, fov_v_deg=93.0).w_none == wide.w_none


def test_a_wide_lens_on_the_same_320_px_sensor_finds_a_distant_duck_less_often():
    """Measured, 400 captures a spec at seed 0: a duck at 3.9 m is 2.99°
    wide and is found 260/400 through the shipped 62° lens but only 73/400
    through a 120° one on the same 320 px. At 2.0 m (5.90° wide) the
    shipped lens is certain, 400/400, and the wide one is not, 271/400. A
    wider lens spends the same pixels over twice the angle: it buys field,
    not sight. (An earlier 120° sweep held the angular gate fixed and so
    charged the wide lens for neither.)"""
    shipped, wide = NARROW_REF, DetectorSpec(fov_h_deg=120.0, fov_v_deg=93.0, px_h=320,
                                             projection="pinhole")

    def found(m, d, tg, spec, n=400):
        det = Detector(m, site="a/head_camera", spec=spec, targets=tg, seed=0)
        return sum(bool(det.capture(d, 0.0).detections) for _ in range(n))

    m, d = world([("a", (0, 0, 0)), ("b", (3.9, 0.0, math.pi))])
    tg = targets(m, ducks=("b",))
    far_shipped, far_wide = found(m, d, tg, shipped), found(m, d, tg, wide)
    assert 0.55 < far_shipped / 400 < 0.75            # p_find ≈ 0.65 at 2.99° through the 1°/4° gate
    assert 0.10 < far_wide / 400 < 0.28               # ≈ 0.18 through the widened 1.94°/7.74° one
    assert far_wide < far_shipped / 2
    # Closer in, the shipped lens is past `w_full` and certain; the wide one is not.
    m, d = world([("a", (0, 0, 0)), ("b", (2.0, 0.0, math.pi))])
    tg = targets(m, ducks=("b",))
    assert found(m, d, tg, shipped) == 400
    assert 0.55 < found(m, d, tg, wide) / 400 < 0.80


def test_a_wide_lens_bearing_is_pushed_outward_by_a_pinhole_reader():
    """`projection`: the detector reports what a consumer INFERS from a box,
    not what the lens physically saw. Under a pinhole model those are the
    same. Under an equidistant (fisheye) one — which the replacement module
    is, since a pinhole focal length solved from its quoted H/V/D disagrees
    while an equidistant one agrees near the quoted EFL — a pinhole-
    calibrated reader is right on axis, right at the edge it calibrated on,
    and wrong in between.

    The magnitude is why this matters: 1.2° worst case at 62° but 9.7° at
    116°, which is larger than the chase brain's 3.4–6.9° aim tolerance. A
    wide lens costs bearing accuracy, and the lens sweep never modelled it
    because at 62° it barely exists."""
    import numpy as np

    from microduck_local.sensors.detector import DetectorSpec

    assert NARROW_REF.projection == "pinhole"                      # the reference, unchanged
    pin = DetectorSpec(projection="pinhole")                       # named: the default is now fisheye
    for th in (0.0, 10.0, 30.0):                                   # identity, exactly
        assert pin.seen_angle(np.deg2rad(th), 62.0) == np.deg2rad(th)

    wide = DetectorSpec(projection="equidistant", fov_h_deg=116.0, fov_v_deg=60.0)
    assert wide.seen_angle(0.0, 116.0) == 0.0                      # right on axis
    edge = np.deg2rad(58.0)
    assert abs(wide.seen_angle(edge, 116.0) - edge) < 1e-9         # right at the calibrated edge
    mid = np.rad2deg(wide.seen_angle(np.deg2rad(28.0), 116.0))
    assert 37.0 < mid < 38.5                                       # pushed OUTWARD in between
    assert wide.seen_angle(np.deg2rad(-28.0), 116.0) == -wide.seen_angle(np.deg2rad(28.0), 116.0)

    worst_wide = max(abs(np.rad2deg(wide.seen_angle(np.deg2rad(t), 116.0)) - t)
                     for t in range(0, 59))
    narrow = DetectorSpec(projection="equidistant")                # same lens law at 62°
    worst_narrow = max(abs(np.rad2deg(narrow.seen_angle(np.deg2rad(t), 62.0)) - t)
                       for t in range(0, 32))
    assert 9.0 < worst_wide < 10.5 and worst_narrow < 1.5          # 8x worse on the wide lens
    assert worst_wide > 6.9                                        # ...and past the aim tolerance


def test_the_camera_sits_eleven_degrees_down_on_a_standing_duck():
    """The head_camera site's tilt and height, MEASURED on a duck standing on
    the walker — the geometry every "the camera loses a floor ball at X"
    claim in this repo rests on.

    The trap this pins: at the model's DEFAULT qpos the site reads level and
    24.8 cm up, and a blind-radius sum built on that is wrong by the whole
    11° (it says the level camera loses a floor ball at 48 cm; the standing
    duck loses it at 28.5). The walker's standing pose is what tilts the
    head, so the duck has to be settled before the site means anything.
    """
    import math

    import mujoco

    from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
    from microduck_local.brain.striker import _gaze_pitch
    from microduck_local.world import World, make_pitch

    sc = make_pitch(per_side=1)
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")}, seed=0)
    d = w.ducks["d0"]
    sid = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_SITE, "d0/head_camera")

    def settle(cmd):
        for _ in range(120):
            d.set_cmd(w.data, (0.0, 0.0, 0.0), (0.0, cmd, 0.0, 0.0))
            w.step()
        ax = w.data.site_xmat[sid].reshape(3, 3)[:, 0]
        return math.atan2(-ax[2], math.hypot(ax[0], ax[1])), w.data.site_xpos[sid][2]

    level, z = settle(0.0)
    assert 0.17 < level < 0.21, f"standing camera tilt {math.degrees(level):.1f}deg, expected ~11"
    assert 0.22 < z < 0.25

    down, z_down = settle(0.6)
    gain = (down - level) / 0.6
    assert 0.70 < gain < 0.85, f"head_pitch gain {gain:.3f} rad/unit, expected ~0.77"
    # `striker._gaze_pitch` inverts exactly this law (cam_level 0.197, gain
    # 0.75). Its constants must keep matching the model, or every head dip
    # aims at the wrong range.
    assert abs(_gaze_pitch(10.0)) < 1e-9                       # far ball: head level
    for rng in (0.30, 0.50, 0.90):
        want = math.atan2(z - 0.035, rng)                      # the true down-angle to a floor ball
        cmd = _gaze_pitch(rng)
        if 0.0 < cmd < 0.6:                                    # unclipped: the axis should land on it
            assert abs((level + gain * cmd) - want) < 0.06, f"gaze off at {rng} m"

    # How much floor 60deg vertical buys over 48deg: the blind radius is
    # where the frame's BOTTOM edge meets the floor.
    def blind(fov_v_deg, tilt, z_cam):
        return (z_cam - 0.035) / math.tan(tilt + math.radians(fov_v_deg) / 2)

    b48, b60 = blind(48.0, level, z), blind(60.0, level, z)
    assert 0.26 < b48 < 0.31 and 0.21 < b60 < 0.25
    assert 0.04 < b48 - b60 < 0.07, "60deg should buy ~5.5 cm of floor, not the 11 cm a level camera gives"
    assert blind(48.0, down, z_down) < 0.11                    # head fully down: ~9 cm


# -- the colour classifier (roadmap Track 4.4.1) ---------------------------------

def test_the_detector_reads_a_ducks_colorway_and_gives_up_at_range():
    """A team IS a colorway, so the one thing a camera could really tell about
    another duck is which team it is on. The classifier fails the two ways a
    real one does — it gives up on a small box, and inside its range it is
    sometimes WRONG (another colorway, not "unknown": a softmax always
    answers)."""
    from microduck_local.sensors.detector import Detector, DetectorNoise, Target

    class Fake(Detector):
        def __init__(self, noise, seed=0):
            self.noise = noise
            self.rng = np.random.default_rng(seed)

    tgt = Target("d1", "duck", 0, 0.10, color="cream")
    sure = Fake(DetectorNoise(color_range=1.0, color_p=1.0))
    assert sure._color(tgt, 0.9) == "cream"
    assert sure._color(tgt, 1.1) is None                    # too far to say
    assert sure._color(Target("ball0", "ball", 0, 0.035), 0.5) is None   # no colour to read
    # At a hostile-ish rate it is wrong sometimes, and wrong means ANOTHER
    # colorway — a brain that treats "not cream" as "opponent" must cope.
    rough = Fake(DetectorNoise(color_range=1.0, color_p=0.75))
    got = [rough._color(tgt, 0.5) for _ in range(400)]
    right = sum(g == "cream" for g in got)
    assert 0.65 < right / len(got) < 0.85                   # about the stated rate
    assert all(g is not None for g in got)                  # never "unknown" inside the range
    assert {g for g in got} - {"cream"}                     # …and the errors are other colorways


def test_the_presets_agree_that_colour_is_harder_than_finding_the_box():
    from microduck_local.sensors.detector import DetectorNoise
    assert DetectorNoise.ideal().color_p == 1.0 and DetectorNoise.ideal().color_range == math.inf
    assert DetectorNoise.datasheet().color_range == 1.0 and DetectorNoise.datasheet().color_p == 0.95
    assert DetectorNoise.hostile().color_range < DetectorNoise.datasheet().color_range
    assert DetectorNoise.hostile().color_p < DetectorNoise.datasheet().color_p


def test_a_pitch_gives_every_duck_target_its_teams_colour():
    from microduck_local.world import World, make_pitch
    if not C.SCENE_WALK_XML.exists():
        pytest.skip("microduck_rl checkout not found")
    sc = make_pitch(per_side=2)
    w = World(sc)
    det = next(d.detector for d in w.ducks.values() if d.detector is not None)
    by_name = {t.name: t for t in det.targets}
    for d in sc.ducks:
        assert by_name[d.id].color == d.team
    assert by_name["ball0"].color is None


def test_a_second_camera_pitched_down_sees_the_ball_at_the_feet_and_reports_it_in_the_head_frame():
    """`DetectorSpec.bottom_pitch_deg` is an ABLATION, not a robot feature:
    the Microduck has one camera. It models the NAO's second, downward lens
    so the sim can ask whether the blind radius is really what the kick is
    waiting on (roadmap Track 4 §6 A.1). Measured geometry, not the NAO's
    number: the duck's camera is 25 cm up, half the NAO's, so 39.7° reaches
    a floor ball at 20 cm and NOT one on the 10 cm kick spot, which needs
    about 60°. A ball the second lens sees is reported with the HEAD's
    bearing (the azimuth the brain steers on), left-positive, ranged from
    its width; a far ball is seen by the head lens first and is identical
    either way; and nothing occludes a ball at the feet — the duck's own
    body never blocks the line (checked with mj_ray)."""
    spot, near, far = (0.10, 0.06), (0.20, 0.05), (1.2, 0.0)
    m, d = world([("a", (0, 0, 0))], balls=[Ball(spot), Ball(near), Ball(far)])
    tg = targets(m, balls=(0, 1, 2))
    from dataclasses import replace
    top = Detector(m, site="a/head_camera", targets=tg, spec=NARROW_REF)
    nao = Detector(m, site="a/head_camera", targets=tg,
                   spec=replace(NARROW_REF, bottom_pitch_deg=39.7))
    steep = Detector(m, site="a/head_camera", targets=tg,
                     spec=replace(NARROW_REF, bottom_pitch_deg=60.0))
    seen_top = {x.name: x for x in top.capture(d, 0.0).detections}
    seen_nao = {x.name: x for x in nao.capture(d, 0.0).detections}
    seen_steep = {x.name: x for x in steep.capture(d, 0.0).detections}
    assert set(seen_top) == {"ball2"}                               # one level camera: only the far ball
    assert set(seen_nao) == {"ball1", "ball2"}                      # the NAO's angle: 20 cm yes, the spot no
    assert set(seen_steep) == {"ball0", "ball1", "ball2"}           # 60 deg: the kick spot too
    b = seen_steep["ball0"]
    assert b.bearing > 0.2                                          # 6 cm left at ~10 cm: left-positive azimuth
    assert 0.05 < b.range_est < 0.30                                # ranged from its width, not the lens
    assert b.elevation < -0.8                                       # far below a level head: the HEAD's frame
    f1, f2 = seen_top["ball2"], seen_steep["ball2"]
    assert (f1.bearing, f1.elevation, f1.width, f1.range_est) == (f2.bearing, f2.elevation, f2.width, f2.range_est)
    # The env knob mirrors MICRODUCK_CHASE: numeric fields only, typos raise.
    assert DetectorSpec.from_env("bottom_pitch_deg=39.7").bottom_pitch_deg == 39.7
    assert DetectorSpec.from_env("").bottom_pitch_deg == 0.0
    with pytest.raises(ValueError):
        DetectorSpec.from_env("bottom_pitch=39.7")
    with pytest.raises(ValueError):
        DetectorSpec.from_env("site=foo")


def test_the_camera_env_sets_the_projection_and_refuses_a_typo():
    """`projection` is the one knob that says whether a wide lens is READ
    correctly (pinhole = calibrated; equidistant = the error a
    pinhole-calibrated reader makes on a fisheye). It was numeric-only in
    `from_env` until 2026-09-09, so the battery that needs it - the new
    module as it ships, uncalibrated - could not be run from a command line
    (docs/camera-hardware.md 2)."""
    import pytest

    from microduck_local.sensors.detector import DetectorSpec

    # The default is "pinhole" — a CALIBRATED reader — since 2026-09-09. It was
    # "equidistant" for part of that day, on the argument that a 116° lens is not
    # rectilinear so reading it as a pinhole is the wrong model. True of the LENS
    # and beside the point for the READER: calibration is a one-off checkerboard,
    # and modelling the robot as though nobody ran one charges it a cost it would
    # not pay. Measured before flipping (kick_gym, 12 seeds × 40 episodes an arm):
    # whiff 31.1% → 17.9%, MDE(80%) 8.6 pp, pinhole better on 11 of 12 seeds.
    # `DetectorSpec.projection` carries the workings.
    assert DetectorSpec().projection == "pinhole"
    wide = DetectorSpec.from_env("fov_h_deg=116,fov_v_deg=60,px_h=640,projection=pinhole")
    assert wide.projection == "pinhole" and wide.px_h == 640 and wide.fov_v_deg == 60.0
    assert DetectorSpec.from_env("projection=equidistant").projection == "equidistant"
    for bad in ("projection=fisheye", "projection=", "fov_h_deg=wide"):
        with pytest.raises(ValueError):
            DetectorSpec.from_env(bad)


# ---- partial visibility and the occlusion fan (2026-09-09) -----------------
#
# The three gates these cover were the answer to a plain observation: the duck
# looked down at the ball, the top of it was plainly in frame, and it still did
# not detect it. docs/camera-hardware.md section 5b has the workings.

LEGACY_GATES = dict(partial_min=0.5, occl_rays=1, seen_full=0.5)
"""The gates every number before 2026-09-09 was measured on: a target had to
have its CENTRE in frame and one ray to that centre had to reach it."""


def _ball_scene(x, walls=()):
    """One duck at the origin, one floor ball `x` ahead, and the pieces needed
    to ask `_visible` about it directly."""
    m, d = world([("a", (0, 0, 0))], balls=[Ball((x, 0.0))], walls=walls)
    det = Detector(m, site="a/head_camera", targets=targets(m, balls=(0,)))
    origin = np.ascontiguousarray(d.site_xpos[det.site_id], dtype=np.float64)
    R = d.site_xmat[det.site_id].reshape(3, 3)
    return m, d, det, det.targets[0], origin, R


def _true_angles(d, det, tgt, origin, R):
    """The target's TRUE centre in the camera frame - what the detector should
    report, clipped or not."""
    local = R.T @ (d.xpos[tgt.body] - origin)
    rng = float(np.linalg.norm(d.xpos[tgt.body] - origin))
    return (math.atan2(local[1], local[0]),
            math.atan2(local[2], math.hypot(local[0], local[1])), rng)


def test_a_ball_with_only_its_top_in_frame_is_still_found():
    """THE ORIGINAL COMPLAINT. At 0.40 m the ball's centre sits about 2.4 deg
    BELOW a 60 deg frustum's bottom edge, so the centre rule threw it away with
    a quarter of it plainly in view - and it is 11 cm across the frame at that
    range, so nothing about this was the size gate."""
    m, d, det, tgt, origin, R = _ball_scene(0.40)
    _, elev, _ = _true_angles(d, det, tgt, origin, R)
    assert elev < -np.deg2rad(det.spec.fov_v_deg) / 2, "the centre must be OUT of frame"

    seen = det._visible(d, tgt, origin, R)
    assert seen is not None, "a quarter of the ball is in frame; it is a detection"
    assert 0.2 < seen[4] < 0.5, seen[4]                 # ...and known to be a partial one

    det.spec = replace(det.spec, **LEGACY_GATES)
    assert det._visible(d, tgt, origin, R) is None       # the rule this replaced


def test_the_old_centre_rule_is_exactly_a_half_visible_rule():
    """Why this went unnoticed for so long: for a ROUND target, centre-in-frame
    IS the 50% contour, so `partial_min` = 0.5 reproduces the old gate exactly.
    The sim was quietly demanding a floor ball be more than half in view."""
    half_v = np.deg2rad(DetectorSpec().fov_v_deg) / 2
    checked = 0
    for x in (0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.50, 0.60):
        m, d, det, tgt, origin, R = _ball_scene(x)
        _, elev, _ = _true_angles(d, det, tgt, origin, R)
        det.spec = replace(det.spec, **LEGACY_GATES)
        assert (det._visible(d, tgt, origin, R) is not None) == (abs(elev) <= half_v), x
        checked += 1
    assert checked == 8


def test_the_occlusion_fan_finds_a_ball_whose_centre_ray_is_blocked():
    """The shipped test was ONE ray to the centre, so anything crossing a
    target's middle deleted it however much of it was in view. A wall edge
    0.5 cm past the line of sight blocks that ray and leaves 38% of the
    silhouette reachable."""
    m, d, det, tgt, origin, R = _ball_scene(
        0.9, walls=[Wall((0.45, -0.6), (0.45, 0.005), 0.30, 0.02)])
    p = d.xpos[tgt.body] - origin
    rng = float(np.linalg.norm(p))

    frac = det._unoccluded(d, tgt, origin, p, rng)
    assert 0.25 < frac < 0.5, frac                       # graded, not binary
    assert det._visible(d, tgt, origin, R) is not None

    det.spec = replace(det.spec, **LEGACY_GATES)
    assert det._unoccluded(d, tgt, origin, p, rng) == 0.0      # the centre ray alone
    assert det._visible(d, tgt, origin, R) is None


def test_partial_visibility_never_moves_the_reported_geometry():
    """Partial visibility decides WHETHER a point target is seen and never what
    you are told about it. Reporting a clipped box's migrated centre instead
    cost `brain/tidy.py`'s elevation-ranging ~2 cm and the pick outright - see
    the note above `width` in the detector."""
    m, d, det, tgt, origin, R = _ball_scene(0.40)
    bearing, elev, rng = _true_angles(d, det, tgt, origin, R)
    seen = det._visible(d, tgt, origin, R)
    assert seen is not None and seen[4] < 1.0, "this target must be clipped"
    assert seen[0] == pytest.approx(bearing, abs=1e-9)
    assert seen[1] == pytest.approx(elev, abs=1e-9)      # NOT the visible midpoint
    assert seen[2] == pytest.approx(2.0 * math.atan(tgt.radius / rng), abs=1e-9)
    assert seen[3] == pytest.approx(rng, abs=1e-9)


def test_seen_full_stops_a_tall_target_being_taxed_for_its_own_height():
    """A person's extent overflows a 60 deg frustum by design, so scaling the
    find probability linearly by the visible fraction charges it for being
    tall: at 1.0 m it fell to 14 detections in 20. `seen_full` gives full marks
    from half-visible up, which is what a detector does with a torso whose head
    is out of frame."""
    from microduck_local.world import Person, World

    for rng_m in (0.8, 1.0, 1.5):
        sc = Scenario(name="tall", floor=(8, 8),
                      ducks=[Duck("d0", (0.0, 0.0, 0.0), None, None, "ideal")],
                      persons=[Person("p0", (rng_m, 0.0), 0.0, [], 0.0, 0.18, 1.6)])
        w = World(sc)
        det = w.ducks["d0"].detector
        for _ in range(6):
            w.step()
        # `ideal` noise, so a p_find of 1.0 is 20 of 20 with no dice at all.
        found = sum(any(x.cls == "person" for x in det.capture(w.data, w.t).detections)
                    for _ in range(20))
        assert found == 20, (rng_m, found)

        # ...and the multiplier itself, rather than a second binomial draw:
        # counting detections under the linear penalty would be a ~0.79^20
        # coin that only happens to be deterministic because the RNG is
        # seeded, and would flip on any change to how draws are consumed.
        tgt = next(t for t in det.targets if t.cls == "person")
        origin = np.ascontiguousarray(w.data.site_xpos[det.site_id], dtype=np.float64)
        R = w.data.site_xmat[det.site_id].reshape(3, 3)
        frac = det._visible(w.data, tgt, origin, R)[4]
        assert frac < 1.0, (rng_m, frac)                  # a linear penalty WOULD bite
        assert min(frac / det.spec.seen_full, 1.0) == 1.0  # `seen_full` spares it
