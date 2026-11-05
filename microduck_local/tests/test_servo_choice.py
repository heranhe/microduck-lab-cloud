"""The servo choice: Feetech STS3215 (default for training requests) or the
official Dynamixel XL330. The official path must stay byte-for-byte what it was;
the Feetech path must be the upstream model, not a retuned XL330."""

import json
import urllib.request

import mujoco
import numpy as np
import pytest

from microduck_local import bam_actuator as BA
from microduck_local import contract as C
from microduck_local.bam_actuator import BamXL330Actuator
from microduck_local.walk_env import MicroduckWalkEnv

UPSTREAM_M6 = ("https://raw.githubusercontent.com/Rhoban/bam/"
               "31e5b97b3cabfe2b909850d483c4e84afa9781c4/bam/params/feetech_sts3215_7_4V/m6.json")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MICRODUCK_ACTUATOR", raising=False)
    monkeypatch.delenv("MICRODUCK_SERVO", raising=False)


def _actuator(servo, **kw):
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    d = mujoco.MjData(m)
    kw.setdefault("vin_range", None)
    kw.setdefault("vin_drop_gain_range", None)
    kw.setdefault("delay_max_lag", 0)
    return BamXL330Actuator(m, d, C.JOINT_NAMES, dt=C.PHYSICS_DT,
                            rng=np.random.default_rng(0), servo=servo, **kw)


# ------------------------------------------------------------------ provenance


def test_the_feetech_table_is_the_pinned_upstream_file():
    """Network-gated on purpose: it is the only check that the numbers were not mistyped."""
    try:
        with urllib.request.urlopen(UPSTREAM_M6, timeout=10) as r:
            upstream = json.load(r)
    except OSError:
        pytest.skip("cannot reach raw.githubusercontent.com")
    for key, value in BA.STS3215_M6_PARAMS.items():
        assert upstream[key] == value, key
    assert upstream["error_gain_ratio"] == BA.STS3215_ERROR_GAIN_RATIO
    assert upstream["max_velocity"] == BA.STS3215_MAX_VELOCITY
    assert upstream["command_delay"] == BA.STS3215_COMMAND_DELAY
    assert upstream["actuator"] == "sts3215" and upstream["model"] == "m6"


def test_both_servos_expose_the_same_friction_keys():
    assert set(BA.STS3215_M6_PARAMS) == set(BA.XL330_M6_PARAMS)


def test_feetech_is_a_different_motor_not_a_rescaled_xl330():
    f, o = BA.STS3215_M6_PARAMS, BA.XL330_M6_PARAMS
    assert f["kt"] > 3 * o["kt"] and f["armature"] > 10 * o["armature"]
    assert BA.STS3215_KP_FW == 32.0 and BA.DEFAULT_KP_FW == 200.0
    assert BA.DEFAULT_SERVO == "feetech" and set(BA.SERVOS) == {"feetech", "official"}


def test_unknown_servo_is_rejected_everywhere():
    with pytest.raises(ValueError, match="servo"):
        _actuator("dynamixel")
    with pytest.raises(ValueError, match="servo"):
        BA.load_servo_params("dynamixel")
    with pytest.raises(ValueError, match="servo"):
        MicroduckWalkEnv(actuator="bam", servo="dynamixel")


# ------------------------------------------------------------------ the physics


def test_the_official_servo_is_unchanged():
    a = _actuator("official")
    assert a.servo == "official" and a._rate_step is None
    assert a.kp_fw == BA.DEFAULT_KP_FW and a.error_gain == BA.XL330_ERROR_GAIN
    assert a.max_current == BA.XL330_MAX_CURRENT and a.max_pwm == BA.XL330_MAX_PWM
    assert a.max_torque == BA.XL330_MAX_TORQUE
    assert a.p == BA.XL330_M6_PARAMS or a.params_source == "bam package"


def test_feetech_uses_its_own_controller_constants():
    a = _actuator("feetech")
    assert a.servo == "feetech" and a.params_source.startswith("hard-coded")
    assert a.kp_fw == 32.0 and a.max_pwm == 0.97 and a.max_current is None
    assert a.error_gain == pytest.approx(0.166 * BA.STS3215_ERROR_GAIN_RATIO)
    assert a._rate_step == pytest.approx(BA.STS3215_MAX_VELOCITY * C.PHYSICS_DT)
    # No current limiter: the ceiling is the stall torque at 7.4 V, ~5x the XL330's.
    assert a.max_torque == pytest.approx(1.2753202369175398 * 7.4 * 0.97 / 2.753421214375944)
    assert a.max_torque > 4 * BA.XL330_MAX_TORQUE


def test_feetech_torque_saturates_at_the_pwm_ceiling_not_a_current_limit():
    a = _actuator("feetech")
    q = np.zeros(14)
    far = np.full(14, 1.0)  # a 1 rad error saturates the 0.97 duty ceiling
    tau = a.compute_motor_torque(q, q, far, 7.4)
    expected = BA.STS3215_M6_PARAMS["kt"] * 7.4 * 0.97 / BA.STS3215_M6_PARAMS["R"]
    assert np.allclose(tau, expected)
    # The same command on the official servo is cut by the 1.75 A firmware limit.
    assert _actuator("official").compute_motor_torque(q, q, far, 7.4).max() < 0.7


def test_feetech_target_chases_the_command_at_a_bounded_speed():
    a = _actuator("feetech")
    a.data.qpos[a.qpos_adr] = 0.0
    a.reset()
    a.set_target(np.full(14, 1.0))
    for step in range(1, 6):
        a.before_step()
        assert np.allclose(a._rate_target, step * BA.STS3215_MAX_VELOCITY * C.PHYSICS_DT)
    # A command already inside the window is followed exactly.
    a.set_target(a._rate_target + 1e-4)
    a.before_step()
    assert np.allclose(a._rate_target, a._target)


def test_feetech_reset_re_seeds_the_internal_target_from_where_the_joint_is():
    a = _actuator("feetech")
    a.data.qpos[a.qpos_adr] = 0.3
    a.reset()
    assert np.allclose(a._rate_target, 0.3)
    a.set_target(np.full(14, 2.0))
    a.before_step()
    a.data.qpos[a.qpos_adr] = -0.2
    a.reset()
    assert np.allclose(a._rate_target, -0.2)


def test_the_official_servo_never_rate_limits():
    a = _actuator("official")
    a.reset()
    a.set_target(np.full(14, 3.0))
    a.before_step()
    assert a._rate_target is None


# ------------------------------------------------------------------ the env


def test_env_selects_the_servo_and_the_lag_follows_the_identified_delay():
    off = MicroduckWalkEnv(actuator="bam", seed=0)
    fee = MicroduckWalkEnv(actuator="bam", servo="feetech", seed=0)
    assert off.servo_model == "official" and off.bam.servo == "official"
    assert fee.servo_model == "feetech" and fee.bam.servo == "feetech"
    assert (off.bam._delay.min_lag, off.bam._delay.max_lag) == (3, 6)
    assert (fee.bam._delay.min_lag, fee.bam._delay.max_lag) == BA.STS3215_DELAY_LAGS


def test_process_env_overrides_the_argument_and_the_force_kwarg_beats_both(monkeypatch):
    monkeypatch.setenv("MICRODUCK_SERVO", "feetech")
    assert MicroduckWalkEnv(actuator="bam", seed=0).servo_model == "feetech"
    assert MicroduckWalkEnv(actuator="bam", servo_force="official", seed=0).servo_model == "official"


def test_the_xml_servo_ignores_the_choice():
    env = MicroduckWalkEnv(actuator="xml", servo="feetech", seed=0)
    assert env.bam is None and env.servo_model == "feetech"


def test_feetech_rollout_is_finite_deterministic_and_keeps_the_contract():
    def run():
        env = MicroduckWalkEnv(actuator="bam", servo="feetech", seed=3)
        obs, _ = env.reset(seed=3)
        assert obs.shape == (C.OBS_DIM,)
        rng = np.random.default_rng(1)
        trace = []
        for _ in range(60):
            obs, _, term, trunc, _ = env.step(rng.uniform(-1, 1, C.NUM_JOINTS).astype(np.float32))
            assert np.all(np.isfinite(obs))
            trace.append(float(env.bam.applied_torque.sum()))
            if term or trunc:
                break
        assert max(abs(t) for t in trace) <= 14 * env.bam.force_limit + 1e-9
        return trace

    assert run() == run()


def test_the_two_servos_really_play_different_physics():
    def torque():
        out = []
        for servo in ("official", "feetech"):
            env = MicroduckWalkEnv(actuator="bam", servo=servo, seed=0, domain_rand=False, obs_noise=False)
            env.reset(seed=0)
            for _ in range(10):
                env.step(np.ones(C.NUM_JOINTS, dtype=np.float32))
            out.append(float(np.abs(env.bam.applied_torque).max()))
        return out
    official, feetech = torque()
    assert feetech > 1.5 * official


def test_the_torque_penalty_is_normalized_by_the_servos_own_ceiling():
    from microduck_local.behaviors.core import _tau2_max
    off = MicroduckWalkEnv(actuator="bam", servo="official", seed=0)
    fee = MicroduckWalkEnv(actuator="bam", servo="feetech", seed=0)
    assert _tau2_max(off) == pytest.approx(C.NUM_JOINTS * BA.XL330_MAX_TORQUE ** 2)
    assert _tau2_max(fee) == pytest.approx(C.NUM_JOINTS * fee.bam.max_torque ** 2)
