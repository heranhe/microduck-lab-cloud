"""Locks for the two get-up probes (roadmap B.1):
`scripts/probe_fall_poses.py` (what pose does a duck that FELL end up in?)
and `scripts/probe_getup.py` (does a policy recover from those poses?).

Why it exists: the get-up recipe trains and `scripts/bench_getup.py` scores
against a SYNTHETIC lie — a tilt drawn from a window, a declared
back/front/side mix, both legs folded by one shared scalar. The probes
exist to measure the real population instead, and both of them are
instruments whose failure mode is SILENCE:

  * a classifier with the sign convention backwards would report a tidy,
    plausible, wrong distribution;
  * a `_replay` that quietly did not install the recorded pose would score
    every policy from a standing start and report 100% recovery.

AGENTS.md's answer to both is the same: assert on the object that is
RUNNING, and show the measurement producing a non-zero on something already
believed before quoting a zero. So the classifier is checked against the
recipe's own spawns (which declare what they pose), and the replay is
checked by reading the projected gravity back off the env it was replayed
into.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.behaviors.env import BehaviorEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_fall_poses import _classify, _measure, _standing_env  # noqa: E402
from probe_getup import _replay  # noqa: E402


def _spawned(kind: str, seed: int, tilt=("80", "115")):
    """A duck posed by the get-up recipe's own spawn, measured by the probe."""
    from microduck_local.behaviors.getup import _getup_spawn

    env = BehaviorEnv("getup", seed=seed, max_episode_s=4.0, random_yaw=False,
                      actuator="xml",
                      spawn_overrides={"MICRODUCK_SPAWN_FAMILY_PROBS": "0,0,0",
                                       "MICRODUCK_GETUP_TILT_LO": tilt[0],
                                       "MICRODUCK_GETUP_TILT_HI": tilt[1],
                                       # No settle: the spawn's declared
                                       # attitude is the thing under test, and
                                       # a second of physics would roll it.
                                       "MICRODUCK_GETUP_SETTLE_S": "0.0"})
    env.reset(seed=seed)
    _getup_spawn(env, kind)
    row = _measure(env)
    env.close()
    return row


@pytest.mark.parametrize("kind,expect", [("back", {"back"}),
                                         ("front", {"front"}),
                                         ("side", {"left", "right"})])
def test_the_classifier_agrees_with_the_spawn_that_declares_the_pose(kind, expect):
    """The positive control. `getup.py` says a duck on its back reads
    gravity (-1, 0, 0) in the trunk frame, its front (+1, 0, 0), its side
    (0, -/+1, 0); the probe's classifier encodes that. Check it against the
    spawns rather than against the sentence — if the recipe's convention
    ever moves, this fails instead of the probe quietly relabelling every
    fall in the roadmap's table."""
    got = {_spawned(kind, 100 + i)["cls"] for i in range(4)}
    assert got <= expect, f"{kind} spawns classified as {got}"


def test_a_standing_duck_is_not_a_fall_pose():
    """The gate the probe's own share counts rest on. A classifier that
    called the keyframe stand a "front" fall would inflate every count and
    read as a finding (AGENTS.md: a probe that forgets its own gate)."""
    env = _standing_env(7, "xml", noise=False)
    env.reset(seed=7)
    assert _measure(env)["cls"] == "upright", (
        "the STAND keyframe classified as a fall pose — every share the "
        "probe reports would be inflated")
    env.close()


def test_the_tilt_readout_is_the_angle_off_upright():
    """Named, not inherited: 0 deg standing, 90 deg on a side. A test that
    pins a number it does not name is a hostage (AGENTS.md)."""
    assert _classify(0.0, 0.0, -1.0) == "upright"
    env = _standing_env(8, "xml", noise=False)
    env.reset(seed=8)
    assert _measure(env)["tilt_deg"] == pytest.approx(0.0, abs=5.0)
    env.close()
    # A duck on its left side reads gy < 0 with gz ~ 0, i.e. 90 deg over.
    assert _classify(0.0, -1.0, 0.0) == "left"
    assert _classify(0.0, +1.0, 0.0) == "right"


def test_the_fold_readout_inverts_the_spawns_own_fold():
    """`_measure`'s `fold` has to mean the same thing for a real fall and a
    synthetic spawn, or the two distributions cannot be compared at all.
    Pose the legs by the recipe's own parameterisation and read it back."""
    env = _standing_env(9, "xml", noise=False)
    env.reset(seed=9)
    adr = env.joint_qpos_adr
    for want in (0.0, 0.5, 1.0):
        q = env.data.qpos
        q[adr] = 0.0
        q[adr[2]], q[adr[3]] = 1.1 * want, 1.2 * want
        q[adr[11]], q[adr[12]] = -1.1 * want, -1.2 * want
        assert _measure(env)["fold"] == pytest.approx(want, abs=1e-6), (
            f"posed fold {want} and read back {_measure(env)['fold']}")
    env.close()


def test_replaying_a_recorded_pose_actually_installs_it():
    """The check that stops `probe_getup` reporting 100% for everything.

    A replay that silently no-opped would leave the duck at the STAND
    keyframe, every policy would "recover" instantly, and nothing in the
    output would say so — AGENTS.md verification discipline #0, a knob that
    changes nothing is broken, not null. So: record a pose that is
    definitely NOT standing, replay it, and read the attitude back off the
    env that is running.
    """
    lying = _spawned("back", 11)
    assert lying["cls"] == "back" and lying["tilt_deg"] > 60.0

    env = BehaviorEnv("getup", seed=12, max_episode_s=4.0, random_yaw=False,
                      actuator="xml",
                      spawn_overrides={"MICRODUCK_SPAWN_FAMILY_PROBS": "0,0,0"})
    env.reset(seed=12)
    assert _measure(env)["cls"] == "upright", "the control started lying down"
    _replay(env, lying["qpos"])
    got = _measure(env)
    env.close()

    assert got["cls"] == "back", (
        f"replayed a back-lying pose and the env reads {got['cls']} "
        f"(tilt {got['tilt_deg']:.0f} deg) — the replay did not install it")
    assert got["tilt_deg"] == pytest.approx(lying["tilt_deg"], abs=1.0)
    for k in ("gx", "gy", "gz"):
        assert got[k] == pytest.approx(lying[k], abs=1e-3)
    # And the servos hold what the duck fell in, as the recipe's own spawn
    # primes them: a standing target in the buffer would give the policy one
    # control step of a command it never asked for.
    assert env.data.ctrl == pytest.approx(
        np.asarray(lying["qpos"])[env.joint_qpos_adr], abs=1e-6)


def test_the_replay_keeps_the_duck_on_the_floor_under_a_null_control():
    """The null control the recovery numbers are quoted against: a duck
    replayed into a fall pose and left limp stays down. If this ever passes
    trivially — because the replay is broken, or because the pose is not
    actually a fall — every `stood` fraction in the probe is meaningless."""
    lying = _spawned("back", 13)
    env = BehaviorEnv("getup", seed=14, max_episode_s=4.0, random_yaw=False,
                      actuator="bam",
                      spawn_overrides={"MICRODUCK_SPAWN_FAMILY_PROBS": "0,0,0"})
    env.reset(seed=14)
    _replay(env, lying["qpos"])
    from microduck_local.behaviors.getup import _getup_hold_raw

    for _ in range(int(round(3.0 / C.CTRL_DT))):
        q = env.data.qpos[env.joint_qpos_adr]
        env.step((q - C.DEFAULT_POSE).astype(np.float32))
        assert _getup_hold_raw(env) == 0.0, "a limp duck stood up"
    env.close()
