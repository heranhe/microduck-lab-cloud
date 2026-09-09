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


def _lay_and_register(w: World, limit: int = 60) -> tuple[float, float]:
    """Lay the duck down and step until the World has registered the fall.
    Returns where it went down, so a test can tell a get-up (it stays there)
    from a respawn (it is moved to its spawn)."""
    _lay_on_back(w)
    for _ in range(limit):
        w.step()
        if w.ducks["d0"].down_until > w.t:
            p = w.ducks["d0"].trunk_pos(w.data)
            return float(p[0]), float(p[1])
    raise AssertionError("the fall was never registered")


def _run_until_up(w: World, limit_s: float = 8.0) -> None:
    t0 = w.t
    while w.ducks["d0"].down_until >= 0.0 and w.t - t0 < limit_s:
        w.step()


def test_without_a_policy_the_fallen_duck_is_still_teleported():
    """The default is bit-for-bit the stand-in every earlier number was
    measured on: it lies there for `getup_s` and is respawned."""
    w = _world(None)
    assert w.getup_infer is None and w.getups == 0
    _lay_and_register(w)
    d = w.ducks["d0"]
    _run_until_up(w)
    assert d.down_until < 0.0                                  # it left the down state…
    assert w.getups == 0 and w.getup_timeouts == 0             # …by the clock, uncounted
    x, y = float(d.trunk_pos(w.data)[0]), float(d.trunk_pos(w.data)[1])
    assert math.dist((x, y), d.spawn[:2]) < 0.15               # teleported to its spawn


def test_with_alpha_stand_the_duck_gets_itself_up_where_it_fell():
    """The substantive difference: it stands under its own power, before the
    timeout, and it is left WHERE IT FELL rather than moved to its spawn."""
    w = _world(onnx_infer(POLICIES_DIR / "alpha_stand.onnx"))
    where = _lay_and_register(w)
    d = w.ducks["d0"]
    t0 = w.t
    _run_until_up(w)
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
    _lay_and_register(w)
    d = w.ducks["d0"]
    _run_until_up(w, limit_s=6.0)
    assert d.down_until < 0.0
    assert w.getups == 0 and w.getup_timeouts >= 1             # counted as a timeout, not a get-up
    x, y = float(d.trunk_pos(w.data)[0]), float(d.trunk_pos(w.data)[1])
    assert math.dist((x, y), d.spawn[:2]) < 0.15               # respawned


def test_reset_clears_the_getup_counters():
    w = _world(onnx_infer(POLICIES_DIR / "alpha_stand.onnx"))
    _lay_and_register(w)
    _run_until_up(w)
    assert w.getups >= 1
    w.reset()
    assert w.getups == 0 and w.getup_timeouts == 0


def test_the_walker_gets_it_back_only_once_it_STAYS_up():
    """`fallen()` is a threshold, and a duck handed back the first tick it
    crosses it is still on its way up: it drops straight back over the line.
    Measured before this dwell existed — ONE real fall in seed 0 of a 24-seed
    3v3 battery became TWENTY-FIVE counted falls, 0.1-0.3 s apart, which is
    far too close together to be separate topples (3 falls with the teleport,
    30 with the get-up, on the same seeds). With `getup_hold_s` it is 1 again."""
    w = _world(onnx_infer(POLICIES_DIR / "alpha_stand.onnx"))
    assert w.getup_hold_s > 0.0
    _lay_and_register(w)
    d = w.ducks["d0"]
    while d.up_since < 0.0 and d.down_until >= 0.0:
        w.step()                                              # …to the first upright tick
    assert d.up_since >= 0.0
    first_up = d.up_since
    assert d.down_until >= 0.0, "handed back on the first upright tick: no dwell"
    assert w.getups == 0
    _run_until_up(w)
    assert d.down_until < 0.0 and w.getups == 1
    assert w.t - first_up >= w.getup_hold_s                    # it had to hold it


def test_the_lab_pitches_get_up_and_rooms_and_batteries_do_not():
    """On /sim a fallen duck used to vanish and reappear, which is the one
    thing on that page a robot plainly does not do: the lab built its World
    with `getup_s` 0, i.e. respawn on the tick it fell. Pitches now get the
    real thing. Rooms are left alone (a duck carrying a toy is another
    question) and eval-pitch still defaults to the teleport, so no published
    number moves."""
    from microduck_local.world_server import (
        PITCH_GETUP_POLICY,
        PITCH_GETUP_S,
        WorldState,
    )
    st = WorldState(None)                      # load_infer None: no policies available
    st.preload("pitch-2v2")
    assert st.world.getup_infer is None        # …degrades to exactly the old teleport
    assert PITCH_GETUP_S > 0 and PITCH_GETUP_POLICY == "pollen:alpha_stand"

    stand = onnx_infer(POLICIES_DIR / "alpha_stand.onnx")
    lab = WorldState(lambda pid: stand if pid == PITCH_GETUP_POLICY else None)
    lab.preload("pitch-2v2")
    assert lab.world.getup_infer is not None and lab.world.getup_s == PITCH_GETUP_S
    room = WorldState(lambda pid: stand if pid == PITCH_GETUP_POLICY else None)
    room.preload("living-room")
    assert room.world.getup_infer is None and room.world.getup_s == 0.0

    import inspect

    from microduck_local.eval_pitch import run_one  # the benchmark is untouched
    assert inspect.signature(run_one).parameters["getup_policy"].default is None
