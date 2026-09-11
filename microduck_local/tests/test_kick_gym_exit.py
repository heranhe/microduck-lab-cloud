"""The gym could not read the kick's exit angle, and now it can (12at).

The selector aims with a number it never checks: `policies/kick/*.json`'s
`exit_rad` is a BENCH median, the sidecar of the pair it replaced records the
same foot reading -0.16 on the bench and +0.26 in play, and nothing between
them measures the exit per swing. `kick_gym`'s new `exit_play` / `aim_err`
columns are that instrument, so the arithmetic under them has to be locked.

What is locked here:

  * the SIGN. This is the whole trap. `exit_rad`, `ChaseParams.kick_exit_*`
    and `kickselect.KickModel.exit` all mean "positive is to the duck's LEFT",
    and a sign flip here would read a left-foot kick as a right-foot one and
    propose a corrected sidecar with the error doubled instead of removed.
    Checked against a REAL ball in a REAL world, rolled at a known angle past
    a duck at a known yaw — not against the formula that produced it.
  * the WRAP, at the ±180° seam, where an un-wrapped difference reads 350°
    instead of -10° (memory: the headstand reversal metric was fooled by
    exactly this).
  * the SILENCE on a stationary ball: a whiff has no direction, and giving it
    one puts uniform noise in a median that is then read as a bias.
  * BACKWARD COMPATIBILITY: `exit_summ` on a row file written before this
    column must be `{}` and must not raise, or every pre-12at `--out` file
    stops comparing (`scripts/compare_gym.py` reads them).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kick_gym import (  # noqa: E402
    EXIT_MIN_M,
    EXIT_S,
    SELECTOR_DIR_SD,
    aim_error,
    exit_angle,
    exit_summ,
    gym_scenario,
)

from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer  # noqa: E402
from microduck_local.world import World  # noqa: E402


@pytest.fixture(scope="module")
def world():
    sc = gym_scenario()
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    return World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=0)


# --- the pure arithmetic ----------------------------------------------------

@pytest.mark.parametrize("yaw", [0.0, 0.7, -1.3, 3.0])
@pytest.mark.parametrize("exit_true", [0.0, 0.25, -0.25, 1.2, -1.2])
def test_a_ball_rolled_at_a_known_angle_reads_back_that_angle(yaw, exit_true):
    """The definition, in both directions: a ball sent `exit_true` off a body
    at `yaw` reads back `exit_true`, for every combination of the two."""
    r = 0.8
    world_dir = yaw + exit_true
    b0 = (0.3, -0.4)
    b1 = (b0[0] + r * math.cos(world_dir), b0[1] + r * math.sin(world_dir))
    assert exit_angle(b0, b1, yaw) == pytest.approx(exit_true, abs=1e-9)


def test_the_sign_is_positive_to_the_ducks_left():
    """Straight ahead is 0, the duck's left is +, its right is -. The same
    convention as `exit_rad` and `ChaseParams.kick_exit_left` (+23.6 deg for
    the left foot). A flip here is invisible in every symmetric test."""
    assert exit_angle((0, 0), (1, 0), 0.0) == pytest.approx(0.0)
    assert exit_angle((0, 0), (0, 1), 0.0) == pytest.approx(math.pi / 2)      # left
    assert exit_angle((0, 0), (0, -1), 0.0) == pytest.approx(-math.pi / 2)    # right
    # ...and with the body turned a quarter turn left, a ball going due north
    # is now dead ahead.
    assert exit_angle((0, 0), (0, 1), math.pi / 2) == pytest.approx(0.0)


def test_the_answer_is_wrapped_at_the_seam():
    """A ball sent almost straight backwards reads near ±180, never near 360:
    an un-wrapped difference here reads a 10 deg error as a 350 deg one."""
    for yaw in (0.0, 2.9, -2.9, math.pi):
        for off in (math.pi - 0.05, -(math.pi - 0.05)):
            got = exit_angle((0, 0), (math.cos(yaw + off), math.sin(yaw + off)), yaw)
            assert -math.pi < got <= math.pi
            assert abs(abs(got) - (math.pi - 0.05)) < 1e-6


def test_a_ball_that_did_not_move_has_no_direction():
    """A whiff must read None. A stationary ball's angle is whatever the
    solver's last micrometre happened to be, and a median over those is a
    measurement of nothing that looks like a measurement of something."""
    assert exit_angle((0.0, 0.0), (0.0, 0.0), 0.0) is None
    assert exit_angle((0.0, 0.0), (EXIT_MIN_M * 0.5, 0.0), 0.0) is None
    assert exit_angle((0.0, 0.0), (EXIT_MIN_M * 2, 0.0), 0.0) == pytest.approx(0.0)


def test_aim_error_is_realised_minus_intended_and_wraps():
    assert aim_error(0.3, 0.1) == pytest.approx(0.2)
    assert aim_error(-0.3, 0.1) == pytest.approx(-0.4)
    assert aim_error(math.pi - 0.1, -(math.pi - 0.1)) == pytest.approx(-0.2, abs=1e-9)
    assert aim_error(None, 0.1) is None
    assert aim_error(0.1, None) is None


def test_the_selector_scatter_is_the_one_the_brain_actually_samples():
    """`SELECTOR_DIR_SD` is the threshold the report calls "inside the scatter
    the selector already assumes". If `kick_select_dir_sd` moves and this does
    not, the report quietly starts grading against the wrong bar."""
    from microduck_local.brain.controllers import ChaseParams
    assert SELECTOR_DIR_SD == pytest.approx(ChaseParams().kick_select_dir_sd)


# --- the same thing against a real ball in a real world ---------------------

@pytest.mark.parametrize("exit_true", [0.0, 0.4, -0.4])
@pytest.mark.parametrize("yaw", [0.0, 1.1])
def test_a_real_ball_rolled_in_the_world_reads_back_its_angle(world, yaw, exit_true):
    """The frame check the formula cannot do: a real ball given a real
    velocity in MuJoCo, a real duck respawned at a known yaw, `EXIT_S` of real
    stepping, read back through `d.yaw(w.data)` — the exact call the gym's
    swing row makes. This is what catches a world/body frame mix-up
    (memory: the MuJoCo velocity-frame trap cost 30M steps of shuffling)."""
    w = world
    d = w.ducks["d0"]
    d.spawn = (-1.2, -0.9, yaw)
    w._respawn(d)
    got_yaw = d.yaw(w.data)
    assert got_yaw == pytest.approx(yaw, abs=1e-3)

    j = w._ball_joint
    q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
    r = w.scenario.balls[0].radius
    # The ball on the CENTRE SPOT and the duck parked 1.5 m away in a corner:
    # the body cannot foul the roll, and from the centre of a 3.0 x 2.5 m pitch
    # every board is at least 1.25 m off, so a 0.66 m roll cannot reach one in
    # any direction. A ball that bounces measures the wall, not the kick — the
    # first version of this test put the ball at (0.4, 0.9) and read a 4.6 deg
    # "turn" that was the +y board.
    b0 = (0.0, 0.0)
    w.data.qpos[q:q + 7] = [b0[0], b0[1], r + 0.005, 1.0, 0.0, 0.0, 0.0]
    w.data.qvel[v:v + 6] = 0.0
    speed = 1.4
    w.data.qvel[v] = speed * math.cos(yaw + exit_true)
    w.data.qvel[v + 1] = speed * math.sin(yaw + exit_true)
    mujoco.mj_forward(w.model, w.data)

    t0 = w.t
    while w.t - t0 < EXIT_S:
        w.step()                      # the gym's own settle loop, minus the brain
    b1 = (float(w.data.qpos[q]), float(w.data.qpos[q + 1]))

    got = exit_angle(b0, b1, got_yaw)
    assert got is not None, "a ball at 1.4 m/s must clear the minimum travel in 0.5 s"
    # Rolling resistance and the floor's friction shorten the roll; they do not
    # turn it. 3 deg is the whole budget.
    assert got == pytest.approx(exit_true, abs=math.radians(3))


# --- the summary, and the rows written before it existed --------------------

def test_exit_summ_is_per_foot_median_and_iqr():
    rows = [{"swing": True, "foot": "kick_left", "exit_play": e, "aim_err": e - 0.1,
             "exit_assumed": -0.2, "back": e < 0.0}
            for e in (-0.4, -0.2, 0.0, 0.2)]
    rows += [{"swing": True, "foot": "kick_right", "exit_play": 0.05, "aim_err": 0.05,
              "exit_assumed": 0.0, "back": False}]
    rows += [{"swing": True, "foot": "kick_left", "exit_play": None, "whiff": True}]   # a whiff: excluded
    rows += [{"swing": False}]                                                         # a no-swing: excluded
    s = exit_summ(rows)
    assert set(s) == {"kick_left", "kick_right"}
    left = s["kick_left"]
    assert left["n"] == 4
    assert left["exit_med"] == pytest.approx(-0.1)
    assert (left["exit_q1"], left["exit_q3"]) == pytest.approx((-0.25, 0.05))
    assert left["assumed"] == pytest.approx(-0.2)
    assert left["back"] == pytest.approx(0.5)                 # -0.4 and -0.2
    assert left["err_med"] == pytest.approx(-0.2)
    assert s["kick_right"]["n"] == 1


def test_the_share_outside_the_selector_scatter_is_over_the_errors():
    big = SELECTOR_DIR_SD + 0.2
    rows = [{"swing": True, "foot": "kick_left", "exit_play": 0.0, "aim_err": e, "exit_assumed": 0.0}
            for e in (0.0, 0.1, big, -big)]
    s = exit_summ(rows)["kick_left"]
    assert s["n_err"] == 4
    assert s["err_outside"] == pytest.approx(0.5)
    assert s["err_abs_med"] == pytest.approx((0.1 + big) / 2)


def test_a_row_file_from_before_the_exit_column_summarises_to_nothing():
    """Backward compatibility, mechanically: `compare_gym.py` is routinely
    pointed at `runs/*.jsonl` written weeks ago. Those rows have no
    `exit_play`, and the summary must be empty rather than an exception or a
    table of zeros that would be read as "the exit is dead ahead"."""
    old = [{"swing": True, "foot": "kick_left", "ahead": 0.09, "side": 0.05,
            "travel": 0.42, "whiff": False, "seed": 0},
           {"swing": False, "ep": 1}]
    assert exit_summ(old) == {}
    assert exit_summ([]) == {}


def test_the_summary_never_counts_a_swing_twice():
    """One row, one foot, one count — the foot key comes off the row and not
    off the sign of the exit, so a left-foot kick that went right is still a
    left-foot kick."""
    rows = [{"swing": True, "foot": "kick_left", "exit_play": -1.0, "aim_err": 0.0, "exit_assumed": 0.0}]
    s = exit_summ(rows)
    assert list(s) == ["kick_left"]
    assert s["kick_left"]["n"] == 1
    assert s["kick_left"]["exit_med"] == pytest.approx(-1.0)


def test_numpy_quantiles_agree_with_the_plain_median():
    """A cheap guard on the one line every number in the report goes through."""
    from kick_gym import _q
    xs = [3.0, 1.0, 2.0, 4.0]
    assert _q(xs, 0.5) == pytest.approx(float(np.median(xs)))
