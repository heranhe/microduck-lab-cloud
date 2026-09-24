import numpy as np
import mujoco
from microduck_local.behaviors import (
    BEHAVIORS,
    BehaviorEnv,
    LANDING_STABLE_STEPS,
    _lj_hop_combo_award,
    _lj_pop_thrust,
    _lj_straight_pen,
    _lj_update,
)


def test_long_jump_registration():
    """验证 long_jump 成功注册在 BEHAVIORS 中且关键词与参数正确。"""
    assert "long_jump" in BEHAVIORS
    b = BEHAVIORS["long_jump"]
    assert b.id == "long_jump"
    assert b.symmetric is True
    assert b.terminate_on_fall is True
    assert len(b.curriculum) == 4
    assert b.spawn_families[0][1].__name__ == "_lj_spawn_landing"
    term_keys = [t.key for t in b.terms]
    assert term_keys[:3] == ["hop_combo", "hop_window", "landing_settle"]


def test_hop_is_certified_only_after_stable_landing():
    env = BehaviorEnv("long_jump", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False)
    env.reset(seed=42)
    env.step_count = 100
    env._lj.update(stage=3, step=99, stage_steps=0, stable_steps=0,
                   pending_hop_dist=0.04, hop_count=0, total_dist=0.0,
                   bad_fall=False)
    env._foot_contacts = lambda: {"left": True, "right": True}
    env._projected_gravity = lambda: np.array([0.0, 0.0, -1.0])
    env.heading_lin_vel = lambda: (0.0, 0.0, 0.0)
    env._gyro[:] = 0.0
    env.data.qvel[2] = 0.0

    for _ in range(LANDING_STABLE_STEPS - 1):
        _lj_update(env)
        assert env._lj["hop_count"] == 0
        env.step_count += 1

    _lj_update(env)
    assert env._lj["hop_count"] == 1
    assert env._lj["landed_fresh"] is True
    assert _lj_hop_combo_award(env) == 0.2
    env.close()


def test_spawn_drop_is_not_counted_as_takeoff_but_stage_one_is():
    env = BehaviorEnv("long_jump", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False)
    env.reset(seed=42)
    env.data.qpos[2] += 0.05
    mujoco.mj_forward(env.model, env.data)
    env._foot_contacts = lambda: {"left": False, "right": False}
    env._projected_gravity = lambda: np.array([0.0, 0.0, -1.0])
    env.heading_lin_vel = lambda: (0.35, 0.0, 0.0)
    env._gyro[:] = 0.0
    env.data.qvel[2] = 0.35
    joint_vel = np.zeros(14)
    joint_vel[3], joint_vel[12] = -1.0, 1.0
    env._joint_vel = lambda: joint_vel

    env.step_count = 2
    env._lj.update(stage=0, step=1)
    _lj_update(env)
    assert env._lj["stage"] == 0
    assert env._lj["took_off_fresh"] is False

    env.step_count = 3
    env._lj.update(stage=1, step=2, ground_x=float(env._trunk_xpos[0]))
    _lj_update(env)
    assert env._lj["stage"] == 2
    assert env._lj["took_off_fresh"] is True
    assert _lj_pop_thrust(env) > 0.9
    env.close()


def test_landing_spawn_uses_curriculum_knobs():
    env = BehaviorEnv("long_jump", spawn_overrides={
        "MICRODUCK_SPAWN_FAMILY_PROBS": "1.0",
        "MICRODUCK_LJ_MIN_DIST_M": "0.02",
        "MICRODUCK_LJ_STABLE_STEPS": "5",
        "MICRODUCK_LJ_GOAL_HOPS": "1",
    }, obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    env.reset(seed=0)
    assert env.last_spawn == "landing"
    assert env._lj["stage"] == 3
    assert env._lj["min_dist"] == 0.02
    assert env._lj["landing_stable_steps"] == 5
    assert env._lj["goal_hops"] == 1
    assert env.data.qvel[2] < 0.0
    env.close()


def test_bad_fall_terminates_with_remaining_horizon_cost():
    env = BehaviorEnv("long_jump", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False)
    env.reset(seed=42)
    env.step_count = 3
    env._lj["bad_fall"] = True

    _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape))

    assert terminated is True
    assert info["episode_rewards"]["fall_pen_penalty"] < -100.0
    env.close()


def test_straight_penalty_does_not_use_unobservable_world_position():
    env = BehaviorEnv("long_jump", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False)
    env.reset(seed=42)
    env.data.qpos[1] = 10.0
    mujoco.mj_forward(env.model, env.data)
    env.heading_lin_vel = lambda: (0.0, 0.0, 0.0)
    assert _lj_straight_pen(env) == 0.0
    env.close()

if __name__ == "__main__":
    test_long_jump_registration()
    test_hop_is_certified_only_after_stable_landing()
    test_spawn_drop_is_not_counted_as_takeoff_but_stage_one_is()
    test_landing_spawn_uses_curriculum_knobs()
    test_bad_fall_terminates_with_remaining_horizon_cost()
    test_straight_penalty_does_not_use_unobservable_world_position()
    print("LONG JUMP TESTS PASSED")
