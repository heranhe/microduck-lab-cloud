from types import SimpleNamespace

import numpy as np
import pytest

from microduck_local.behaviors import (
    BEHAVIORS, BehaviorEnv, ONE_LEG_SUCCESS_STEPS, _one_leg_5s_update,
    match_behavior,
)


def standing():
    return SimpleNamespace(
        foot_contact_state={"left": True, "right": False},
        foot_geoms={"left": 0, "right": 1},
        data=SimpleNamespace(geom_xpos=np.array([[0., 0., 0.], [0., 0., .08]])),
        _projected_gravity=lambda: np.array([0., 0., -1.]),
        heading_lin_vel=lambda: (0., 0., 0.),
        _trunk_xpos=np.array([0., 0., .12]), stand_z=.12,
        _gyro=np.zeros(3), _one_leg_hold_steps=0, _one_leg_best_steps=0,
    )


def test_five_seconds_is_continuous_not_cumulative():
    env = standing()
    assert ONE_LEG_SUCCESS_STEPS == 250
    for _ in range(249):
        _one_leg_5s_update(env)
    assert env._one_leg_best_steps < ONE_LEG_SUCCESS_STEPS
    env.foot_contact_state["right"] = True
    _one_leg_5s_update(env)
    assert env._one_leg_hold_steps == 0
    env.foot_contact_state["right"] = False
    for _ in range(249):
        _one_leg_5s_update(env)
    assert env._one_leg_best_steps == 249  # not 498!
    _one_leg_5s_update(env)
    assert env._one_leg_best_steps == ONE_LEG_SUCCESS_STEPS


@pytest.mark.parametrize("failure", ["no_support", "low_foot", "tilted", "low_body", "moving", "spinning"])
def test_unstable_stance_resets_timer(failure):
    env = standing()
    env._one_leg_hold_steps = 249
    if failure == "no_support":
        env.foot_contact_state["left"] = False
    elif failure == "low_foot":
        env.data.geom_xpos[1, 2] = .01
    elif failure == "tilted":
        env._projected_gravity = lambda: np.array([.6, 0., -.8])
    elif failure == "low_body":
        env._trunk_xpos[2] = .05
    elif failure == "moving":
        env.heading_lin_vel = lambda: (.2, 0., 0.)
    else:
        env._gyro[2] = 2.
    _one_leg_5s_update(env)
    assert env._one_leg_hold_steps == env._one_leg_best_steps == 0


def test_task_contract_and_reset():
    assert match_behavior("单脚站立5秒").id == "one_leg_5s"
    assert BEHAVIORS["imitate"].episode_s == 4.0  # other clips unchanged
    env = BehaviorEnv("one_leg_5s", obs_noise=False, domain_rand=False,
                      action_delay=False, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (61,) and env.bam is not None and env.max_steps >= 1000
        env._one_leg_hold_steps = env._one_leg_best_steps = 250
        env.reset(seed=1)
        assert env._one_leg_hold_steps == env._one_leg_best_steps == 0
        _, _, _, _, info = env.step(np.zeros(14, dtype=np.float32))
        assert info["is_success"] is False and info["hold_s"] < 5
    finally:
        env.close()
