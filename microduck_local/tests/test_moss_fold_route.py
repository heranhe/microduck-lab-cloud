"""MOSS's short fold route (`TidyMoss.fold_route`): the arm goes home from the
release pose through one planned waypoint, 2.0 rad of the slowest joint
instead of the learned fold's 6.1 (7.8 s a fold in the yard)."""
import mujoco
import numpy as np
import pytest

from microduck_local.robots import moss

pytestmark = pytest.mark.skipif(
    not moss.moss_ready(),
    reason="MOSS assets missing — uv run fetch-robot moss")


def _hits_body(m, d, q) -> bool:
    """Does any arm geom touch the rover's hull, bin or tracks at pose q?"""
    for j, v in zip(moss.ARM_JOINTS, q):
        d.qpos[m.joint(j).qposadr[0]] = v
    for j in moss.FINGER_JOINTS:
        d.qpos[m.joint(j).qposadr[0]] = moss.MISSION_OPEN_M
    mujoco.mj_forward(m, d)
    for c in d.contact[:d.ncon]:
        a, b = int(m.geom_contype[c.geom1]), int(m.geom_contype[c.geom2])
        if {a, b} == {1, 2} and c.dist < 0.0:
            return True
    return False


def _segment_hits(m, d, a, b, step=0.01) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.abs(b - a).max() / step) + 1)
    return any(_hits_body(m, d, a + (b - a) * k / (n - 1)) for k in range(n))


def test_the_short_fold_route_clears_the_body():
    """Both straight segments are collision-free against the robot's own
    collision model — and the straight line home, which the route exists to
    avoid, is not (so this check can fail)."""
    from microduck_local.brain.tidy_moss import TidyMossParams
    p = TidyMossParams()
    m = moss.model()
    d = mujoco.MjData(m)
    start, wp, tuck = p.fold_route_from, p.fold_route_wp, moss.tuck_pose()
    assert not _segment_hits(m, d, start, wp)
    assert not _segment_hits(m, d, wp, tuck)
    assert _segment_hits(m, d, start, tuck)


def _fly_route(**params):
    """Step the brain's tuck from the release pose with a servo that keeps
    up; (seconds to leave `tuck`, largest per-tick joint step, final state)."""
    import dataclasses

    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss

    # The OLD fold (to the old tuck): `plan_routes` and the rest pose off
    # unless a test asks — tests/test_moss_motion.py has the planned one.
    params.setdefault("plan_routes", False)
    params.setdefault("rest_pose", None)
    b = TidyMoss()
    b.p = dataclasses.replace(b.p, fold_route=True, **params)
    b._to("tuck", 0.0)
    arm = dict(zip(moss.ARM_JOINTS, b.p.fold_route_from))
    worst, t = 0.0, 0.0
    for k in range(1, 400):
        t = 0.02 * k
        it = b.step(Senses(t=t, odom=(0.0, 0.0, 0.0), speed=0.0, arm=arm))
        if b.state != "tuck":
            break
        new = {j: it.arm[j] for j in moss.ARM_JOINTS}
        worst = max(worst, max(abs(new[j] - arm[j]) for j in moss.ARM_JOINTS))
        arm = new
    return t, worst, b.state


def test_the_scripted_fold_runs_at_the_cap_and_only_from_the_release_pose():
    import dataclasses

    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss

    t, worst, state = _fly_route()
    assert state == "search"
    assert 2.0 < t < 3.0                           # the learned fold: 7.8 s
    # no faster than the 0.75 rad/s every fold command trained at
    assert worst <= 0.75 * 0.02 + 1e-6

    # Any other start is not what the route was planned from: learned fold.
    b = TidyMoss()
    b.p = dataclasses.replace(b.p, fold_route=True)
    b._to("tuck", 0.0)
    elsewhere = dict(zip(moss.ARM_JOINTS, moss.LIFT_POSE))
    assert b._fold_route(Senses(t=0.02, odom=(0.0, 0.0, 0.0), speed=0.0,
                                arm=elsewhere)) is None
