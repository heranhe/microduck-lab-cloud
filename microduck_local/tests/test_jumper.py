"""Jumper integration contract, cache integrity, six-foot physics and PPO smoke."""

import hashlib
import json

import numpy as np
import pytest

from microduck_local.robots import jumper_model as J
from microduck_local.robots import registry as R


def test_listing_and_fetch_hint_do_not_need_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(J, "CACHE_DIR", tmp_path)
    assert not J.ready()
    assert "jumper" in R.ids()
    assert R.setup_hint("jumper") == "uv run fetch-robot jumper"
    assert not R.get("jumper").ready()
    concrete = R.get("jumper")._resolve()
    assert concrete.num_actions == 20
    assert concrete.contract().obs_dim == 71
    assert concrete.lab_spacing_m == J.FACTS["lab_spacing_m"]


def test_a_bad_download_never_becomes_ready(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.xml").write_bytes(b"bad")
    monkeypatch.setattr(J, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setitem(J.FACTS, "files", {"model.xml": hashlib.sha256(b"good").hexdigest()})
    with pytest.raises(RuntimeError, match="SHA256"):
        J.fetch(source)
    assert not J.ready()
    (source / "model.xml").write_bytes(b"good")
    J.fetch(source)
    assert J.ready()
    # Cache corruption is repaired, not silently accepted on re-fetch.
    (J.CACHE_DIR / "model.xml").write_bytes(b"broken")
    J.fetch(source)
    assert (J.CACHE_DIR / "model.xml").read_bytes() == b"good"


def test_the_gait_clock_wraps_balanced_and_is_off_at_idle():
    from microduck_local.robots.jumper_env import GAIT_STEPS, phase_clock

    samples = np.array([phase_clock(i, [0.1, 0, 0]) for i in range(GAIT_STEPS)])
    assert np.allclose(samples.sum(axis=0), 0, atol=1e-6)
    assert np.allclose(phase_clock(GAIT_STEPS, [0.1, 0, 0]), [0, 1])
    assert np.array_equal(phase_clock(7, [0, 0, 0]), [0, 0])


needs_assets = pytest.mark.skipif(not J.ready(), reason="fetch-robot jumper first")


@needs_assets
def test_the_22_model_joints_map_to_20_actions_and_two_held_grippers():
    from microduck_local.robots.jumper_env import JumperStandEnv

    e = JumperStandEnv(obs_noise=False, domain_rand=False, random_yaw=False)
    assert e.model.nq == 29 and e.model.nu == 22
    assert e.action_space.shape == (20,)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (71,)
    for i, name in enumerate(J.JOINT_NAMES):
        assert e.model.joint(name).qposadr[0] == e.joint_qpos_adr[i]
        assert e.model.actuator(name).id == e.ctrl_ids[i]
    for i in range(150):
        obs, reward, done, truncated, info = e.step(np.zeros(20))
        assert not done and np.isfinite(obs).all() and np.isfinite(reward)
        assert all(v <= 0 for k, v in info["reward_terms"].items() if k.endswith("penalty"))
    assert e._contacts().all()
    assert e._projected_gravity()[2] < -0.99
    assert e.data.xpos[e.trunk_body_id, 2] > 0.09
    for name in J.GRIPPERS:
        assert e.data.ctrl[e.model.actuator(name).id] == J.HOME[name]
    with pytest.raises(ValueError):
        e.step(np.zeros(22))


@needs_assets
def test_observation_order_and_non_accumulating_friction():
    from microduck_local.robots.jumper_env import JumperWalkEnv

    e = JumperWalkEnv(obs_noise=False, domain_rand=True, random_yaw=False)
    for i in range(20):
        e.data.qpos[e.joint_qpos_adr[i]] += 0.001 * (i + 1)
        e.data.qvel[e.joint_qvel_adr[i]] = 0.01 * (i + 1)
    e.last_action[:] = np.linspace(-1, 1, 20)
    e.twist_cmd[:] = [0.1, 0.2, 0.3]
    e.step_count = 5
    obs = e._get_obs()
    assert np.allclose(obs[6:26], 0.001 * np.arange(1, 21), atol=1e-6)
    assert np.allclose(obs[26:46], 0.01 * np.arange(1, 21))
    assert np.array_equal(obs[46:66], e.last_action)
    assert np.array_equal(obs[66:69], e.twist_cmd)
    e.reset(seed=17)
    friction = e.model.geom_friction.copy()
    for _ in range(10):
        e.reset(seed=17)
    assert np.array_equal(e.model.geom_friction, friction)
    assert len(e.foot_ids) == 6


@needs_assets
def test_teach_recipes_use_the_correct_body_and_actuators():
    from microduck_local.behaviors import for_robot

    recipes = for_robot("jumper")
    assert {r.id for r in recipes} == {"jumper_stand", "jumper_tripod"}
    for recipe in recipes:
        assert recipe.trainer[-2:] == ("--actuator", "xml")
    from microduck_local.robots.jumper_env import JumperWalkEnv

    with pytest.raises(ValueError, match="BAM"):
        JumperWalkEnv(actuator_force="bam")


@needs_assets
def test_ppo_can_update_then_export_with_its_normalizer(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from microduck_local.export_onnx import export
    from microduck_local.robots.jumper_env import JumperStandEnv
    from microduck_local.robots.policy_contract import PolicyContract

    env = VecNormalize(
        DummyVecEnv([lambda: JumperStandEnv(obs_noise=False, domain_rand=False)]), norm_reward=False
    )
    policy = PPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        policy_kwargs={"net_arch": [16]},
        device="cpu",
        seed=0,
    )
    before = {k: v.clone() for k, v in policy.policy.state_dict().items()}
    policy.learn(64)
    assert any(not (v == before[k]).all() for k, v in policy.policy.state_dict().items())
    policy.save(tmp_path / "model")
    env.save(tmp_path / "vecnormalize.pkl")
    (tmp_path / "run.json").write_text(
        json.dumps({"robot": "jumper", "contract": R.get("jumper").contract().as_dict()})
    )
    path = export(tmp_path, tmp_path / "policy.onnx")
    assert PolicyContract.from_onnx(path) == R.get("jumper").contract()
    env.close()


def test_collection_without_jumper_assets(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/test_body_conformance.py",
            "tests/test_mjcf_body.py",
        ],
        cwd=project,
        env={**os.environ, "MICRODUCK_JUMPER_DIR": str(tmp_path / "missing")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[jumper]" in result.stdout
