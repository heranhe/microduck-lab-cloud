"""What the arm is allowed to ASK FOR — `TidyMoss.arm_rate_cap` / `arm_slew_cap`.

MOSS's servos take a position command at kp 70 with `forcerange` +-2.2 N m, and
the policy's own increment is 0.03 rad at 25 Hz = 0.75 rad/s. The scripted
paths had no such bound: MEASURED over eight 300 s moss-yard seeds
(`scripts/probe_moss_safety.py`, sampling every 2 ms physics step) the brain
commanded one-tick steps of up to **2.750 rad** — 137 rad/s — and the arm
reached 13.6 rad/s, while `shoulder_pan` sat at its full torque clamp for an
unbroken **9.2 s**. Two separate causes, so two separate caps, and these tests
hold each to the thing it fixes:

* a BLEND ran a fixed duration whatever distance it had to cover -> `_paced`
* a STATE TRANSITION is a step input of the servo's tracking lag -> `_slew`

`tests/test_moss.py::test_no_scripted_transition_is_a_step_input_to_the_servos`
is the older half of this and not a duplicate: it asserts a ramp EXISTS in
`deploy` and `tuck`, deliberately "the mechanism rather than a speed, because
the speed depends on the pose". These assert the speed anyway, because that is
what the pose-dependence turned out to cost — `pinch/align`'s ramp existed the
whole time and still demanded 8.4-10.0 rad/s. (That test also cites an XL330's
5.6 rad/s no-load speed; MOSS's arm is SO-101-derived and its servos are not
the duck's, so the number is the right idea about the wrong motor.)
"""
import dataclasses
import math

import numpy as np
import pytest

from microduck_local.brain.runtime import Senses
from microduck_local.brain.tidy_moss import TidyMoss
from microduck_local.robots import moss

pytestmark = pytest.mark.skipif(
    not moss.moss_ready(),
    reason="MOSS assets missing — uv run fetch-robot moss")

DT = 0.02          # the world's control tick, `contract.CTRL_DT`


def _brain(**params) -> TidyMoss:
    b = TidyMoss()
    if params:
        b.p = dataclasses.replace(b.p, **params)
        for k, v in params.items():          # a frozen dataclass eats attribute sets
            assert getattr(b.p, k) == v, (k, getattr(b.p, k), v)
    return b


def _peak_rate(a, b, seconds: float, eased: bool = True) -> float:
    """rad/s, the biggest one-tick step of the blend the brain actually flies,
    divided by the tick — the same smoothstep `_pinch_step.blend` uses."""
    def k(x):
        x = min(1.0, max(0.0, x / seconds))
        return x * x * (3.0 - 2.0 * x) if eased else x
    worst = 0.0
    for i in range(1, int(seconds / DT) + 2):
        for j in moss.ARM_JOINTS:
            d0, d1 = k((i - 1) * DT), k(i * DT)
            worst = max(worst, abs((b[j] - a[j]) * (d1 - d0)))
    return worst / DT


def _poses(roll_swing: float):
    a = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
    return a, {**a, "wrist_roll": a["wrist_roll"] + roll_swing}


def test_a_paced_blend_never_demands_more_than_the_rated_joint_speed():
    """A quarter turn of the wrist — what `pinch/align` does to put the jaws on
    a card's long axis — flown at the brain's own pacing."""
    b = _brain()
    a, z = _poses(math.pi / 2)
    T = b._paced(0.6 * b.p.pinch_move_s, a, z)
    assert _peak_rate(a, z, T) <= b.p.arm_rate_cap + 1e-6
    # ...and the cap is what BINDS: the unpaced duration is the one measured
    # in the room, and it demands 5.6 rad/s.
    assert T > 0.6 * b.p.pinch_move_s
    assert _peak_rate(a, z, 0.6 * b.p.pinch_move_s) > 5.0


def test_pacing_pays_for_the_smoothsteps_PEAK_rate_not_its_mean():
    """A smoothstep's peak is 1.5x its mean and a straight ramp's is 1.0x.
    Charging the mean would leave every eased blend 50% over the cap."""
    b = _brain()
    a, z = _poses(1.0)
    eased, straight = b._paced(0.01, a, z), b._paced(0.01, a, z, eased=False)
    assert eased == 1.5 * straight
    assert _peak_rate(a, z, eased) <= b.p.arm_rate_cap + 1e-6
    assert _peak_rate(a, z, straight, eased=False) <= b.p.arm_rate_cap + 1e-6
    # the mistake this guards: pacing an EASED blend as if it were straight
    assert _peak_rate(a, z, straight) > b.p.arm_rate_cap


def test_a_zero_cap_is_off_so_the_A_B_has_a_control():
    b = _brain(arm_rate_cap=0.0)
    a, z = _poses(math.pi / 2)
    assert b._paced(0.42, a, z) == 0.42


def test_the_pinch_turns_the_jaws_onto_the_long_axis_at_the_cap():
    """The align phase, flown tick by tick through the brain itself — not
    through a copy of its arithmetic — and it must still ARRIVE."""
    b = _brain()
    hover, over = _poses(math.pi / 2)
    b._to("pinch", 0.0)
    b._pinch = {"t0": b._t0, "phase": "align", "since": 0.0,
                "from": dict(hover), "hover_pose": hover, "over_pose": over,
                "roll": hover["wrist_roll"], "fixes": [], "yaws": []}
    arm, worst = dict(hover), 0.0
    for i in range(1, 400):
        since = i * DT
        out, _ = b._pinch_step(Senses(t=since, odom=(0.0, 0.0, 0.0), speed=0.0,
                                      arm={**arm, moss.GRIPPER_JOINT: 0.041}),
                              since, since)
        if out is None:
            break
        # Record BEFORE testing the phase: the tick that finishes the blend is
        # the tick that flips it, and its command is the endpoint.
        new = {j: float(out[j]) for j in moss.ARM_JOINTS}
        worst = max(worst, max(abs(new[j] - arm[j]) for j in moss.ARM_JOINTS))
        arm = new
        if b._pinch["phase"] != "align":
            break
    assert b._pinch["phase"] == "descend", "the align never finished"
    assert worst <= b.p.arm_rate_cap * DT + 1e-9, f"{worst / DT:.2f} rad/s"
    # the whole quarter turn was covered, not merely approached slowly
    assert abs(arm["wrist_roll"] - over["wrist_roll"]) < 1e-6


def _emit(b, arm, t):
    """One tick as `TidyMoss.step` ends it: limit, then remember what went
    out. Driving `_slew` alone would test a limiter with no memory."""
    out = b._slew(arm, t)
    b._remember_arm(out, t)
    return out


def test_the_slew_limiter_spreads_a_transition_step_without_losing_it():
    """A state transition hands the arm the servo's tracking lag as a step —
    0.449 rad, measured. The limiter pays it off over ticks; it must not
    swallow it, or the leg would never reach its pose."""
    b = _brain(arm_slew_cap=1.5)
    start = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
    _emit(b, dict(start), 0.0)
    goal = {**start, "shoulder_pan": start["shoulder_pan"] + 0.449}
    cmd, worst = dict(start), 0.0
    for i in range(1, 100):
        out = _emit(b, dict(goal), i * DT)
        worst = max(worst, max(abs(out[j] - cmd[j]) for j in moss.ARM_JOINTS))
        cmd = {j: float(out[j]) for j in moss.ARM_JOINTS}
        if abs(cmd["shoulder_pan"] - goal["shoulder_pan"]) < 1e-9:
            break
    assert worst <= 1.5 * DT + 1e-9, f"{worst / DT:.2f} rad/s"
    assert abs(cmd["shoulder_pan"] - goal["shoulder_pan"]) < 1e-9, "never arrived"
    assert i <= math.ceil(0.449 / (1.5 * DT)) + 1, "took longer than the backlog"


def test_an_ARM_LESS_tick_does_not_become_slew_BUDGET():
    """The gap this limiter kept failing on. A tick that commands no arm
    (`_to` handoffs, `pinch: missed`, `_abort_drop` — the state TRANSITIONS
    the limiter exists for) leaves `set_arm` holding the last mapping, so the
    command did not move. Counting that gap as elapsed time hands the next
    tick `cap * gap` of room: over 0.5 s at the shipped cap that is 1.5 rad,
    which is no limit at all."""
    b = _brain(arm_slew_cap=1.5)
    start = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
    _emit(b, dict(start), 0.0)
    for i in range(1, 26):                     # half a second commanding NO arm
        _emit(b, None, i * DT)
    out = _emit(b, {**start, "shoulder_pan": start["shoulder_pan"] + 2.0}, 26 * DT)
    step = abs(out["shoulder_pan"] - start["shoulder_pan"])
    assert step <= 1.5 * DT + 1e-9, (
        f"{step:.3f} rad in one tick ({step / DT:.1f} rad/s) — the arm-less "
        "ticks were spent as budget")


def test_the_slew_limiter_leaves_the_GRIPPER_alone():
    """Its travel is 41 mm of slide on its own servo and nothing here has
    measured that servo's rate — a rad/s cap on it would be invented."""
    b = _brain(arm_slew_cap=1.5)
    shut = {**dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE)), moss.GRIPPER_JOINT: 0.0}
    b._slew(dict(shut), 0.0)
    out = b._slew({**shut, moss.GRIPPER_JOINT: 0.041}, DT)
    assert out[moss.GRIPPER_JOINT] == 0.041


def test_off_by_default_so_the_pacing_ships_on_its_own_measurement():
    assert TidyMoss().p.arm_slew_cap == 0.0
    # 3.0, not the env's 1.5: the tighter cap costs -1.50 +- 0.60 binned per
    # seed at a 180 s horizon and the 300 s score cannot see it (the param).
    assert np.isclose(TidyMoss().p.arm_rate_cap, 3.0)


def test_a_tick_with_NO_ARM_READINGS_does_not_command_the_pose_outright():
    """`_ramp_from_here` snapshots the achieved pose on a state's first tick.
    With no readings there is nothing to snapshot, and the old fallback
    commanded the endpoint OUTRIGHT — an unbounded step input, past the
    `_paced` call that is the whole point of `arm_rate_cap`. The last command
    EMITTED is where the driver is still holding the arm, so it is the honest
    start. `deploy` and `tuck` both ramp through this helper.
    """
    b = _brain()
    here = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
    b._remember_arm(dict(here), 0.0)           # a tick that DID command the arm
    target = list(moss.LIFT_POSE)
    far = max(abs(t - here[j]) for j, t in zip(moss.ARM_JOINTS, target))
    assert far > 0.5, "pick a target far enough for a step to be visible"

    b._to("tuck", 0.0)
    out = b._ramp_from_here(target, 0.0, b.p.tuck_ramp_s,
                            Senses(t=0.0, odom=(0.0, 0.0, 0.0), speed=0.0, arm=None))
    step = max(abs(out[j] - here[j]) for j in moss.ARM_JOINTS)
    assert step <= b.p.arm_rate_cap * DT + 1e-9, (
        f"{step:.3f} rad in one tick ({step / DT:.1f} rad/s) from a tick with "
        "no arm readings")

    # …and it still ARRIVES: the ramp is paced, not abandoned.
    end = b._ramp_from_here(target, 1e3, b.p.tuck_ramp_s,
                            Senses(t=1e3, odom=(0.0, 0.0, 0.0), speed=0.0, arm=None))
    for j, v in zip(moss.ARM_JOINTS, target):
        assert abs(end[j] - v) < 1e-9, f"{j} never reached the target"


def test_a_COLD_start_still_has_only_the_endpoint():
    """No readings AND nothing emitted yet — the first tick of a run. There is
    no pose to ramp from, and pretending otherwise would invent one."""
    b = _brain()
    b._to("tuck", 0.0)
    out = b._ramp_from_here(list(moss.LIFT_POSE), 0.0, b.p.tuck_ramp_s,
                            Senses(t=0.0, odom=(0.0, 0.0, 0.0), speed=0.0, arm=None))
    for j, v in zip(moss.ARM_JOINTS, moss.LIFT_POSE):
        assert abs(out[j] - v) < 1e-9
