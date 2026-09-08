"""The get-up in the WORLD (roadmap B.1 / bead mdl-0ad): a fallen duck that
gets up by itself instead of being teleported upright.

`getup_s` alone was always a stand-in — the duck lies on a zero command for
that long and is then respawned. The gap was never a missing policy: the
shipped `alpha_stand.onnx` recovers from lying on its back, front and side
100% of the time in 0.2-1.3 s, while the walker that drives a fallen duck
today manages 0 of 24. So it is a controller switch, and this is the lock on
it: `World(getup_infer=...)` drives a downed duck with that policy, the duck
leaves the down state by STANDING rather than by the clock, and `getup_s`
becomes the timeout that stops a stuck duck stalling a battery.
"""

import math

import mujoco

from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.world import World, make_pitch
from microduck_local.world.arena import zero_infer


def _lay_on_back(w: World, did: str = "d0") -> None:
    """Put a duck on its back where it stands and let the physics take it."""
    d = w.ducks[did]
    q, v = d.adr.root_qpos, d.adr.root_qvel
    w.data.qpos[q + 2] = 0.30
    w.data.qpos[q + 3:q + 7] = [0.0, 0.0, 1.0, 0.0]        # 180 deg about the lateral axis
    w.data.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(w.model, w.data)


def _world(getup_infer, getup_s: float = 5.0) -> World:
    sc = make_pitch(per_side=1)
    walker = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    return World(sc, infer_for={d.id: walker for d in sc.ducks}, seed=0,
                 getup_s=getup_s, getup_infer=getup_infer)


def _drop_and_settle(w: World, steps: int = 80) -> None:
    """Lay it down and let it come to rest. `steps` has to stay well inside
    `getup_s`, or the stand-in respawns the duck before the test starts."""
    _lay_on_back(w)
    for _ in range(steps):
        w.step()
    assert w.ducks["d0"].down_until > w.t, "the fall was never registered"


def test_without_a_policy_the_fallen_duck_is_still_teleported():
    """The default is bit-for-bit the stand-in every earlier number was
    measured on: it lies there for `getup_s` and is respawned."""
    w = _world(None)
    assert w.getup_infer is None and w.getups == 0
    _drop_and_settle(w)
    d = w.ducks["d0"]
    t0 = w.t
    while d.down_until >= 0.0 and w.t - t0 < 8.0:
        w.step()
    assert d.down_until < 0.0                                  # it left the down state…
    assert w.getups == 0 and w.getup_timeouts == 0             # …by the clock, uncounted
    x, y = float(d.trunk_pos(w.data)[0]), float(d.trunk_pos(w.data)[1])
    assert math.dist((x, y), d.spawn[:2]) < 0.15               # teleported to its spawn


def test_with_alpha_stand_the_duck_gets_itself_up_where_it_fell():
    """The substantive difference: it stands under its own power, before the
    timeout, and it is left WHERE IT FELL rather than moved to its spawn."""
    w = _world(onnx_infer(POLICIES_DIR / "alpha_stand.onnx"))
    _drop_and_settle(w)
    d = w.ducks["d0"]
    where = tuple(float(v) for v in d.trunk_pos(w.data)[:2])
    t0 = w.t
    while d.down_until >= 0.0 and w.t - t0 < 8.0:
        w.step()
    assert d.down_until < 0.0, "it never got up"
    assert w.getups >= 1 and w.getup_timeouts == 0             # up by itself, not by the clock
    assert w.t - t0 < 5.0                                      # …and inside the timeout
    assert not d.fallen(w.data)
    x, y = float(d.trunk_pos(w.data)[0]), float(d.trunk_pos(w.data)[1])
    assert math.dist((x, y), where) < 0.5                      # still where it fell: no teleport
    assert math.dist((x, y), d.spawn[:2]) > 0.05               # and NOT put back on its spawn


def test_a_policy_that_cannot_get_up_still_times_out():
    """The timeout is what stops a stuck duck stalling a battery, so a useless
    get-up policy must respawn on `getup_s` exactly as the stand-in does."""
    w = _world(zero_infer, getup_s=2.0)
    _drop_and_settle(w, steps=40)
    d = w.ducks["d0"]
    t0 = w.t
    while d.down_until >= 0.0 and w.t - t0 < 6.0:
        w.step()
    assert d.down_until < 0.0
    assert w.getups == 0 and w.getup_timeouts >= 1             # counted as a timeout, not a get-up
    x, y = float(d.trunk_pos(w.data)[0]), float(d.trunk_pos(w.data)[1])
    assert math.dist((x, y), d.spawn[:2]) < 0.15               # respawned


def test_reset_clears_the_getup_counters():
    w = _world(onnx_infer(POLICIES_DIR / "alpha_stand.onnx"))
    _drop_and_settle(w)
    t0 = w.t
    while w.ducks["d0"].down_until >= 0.0 and w.t - t0 < 8.0:
        w.step()
    assert w.getups >= 1
    w.reset()
    assert w.getups == 0 and w.getup_timeouts == 0
