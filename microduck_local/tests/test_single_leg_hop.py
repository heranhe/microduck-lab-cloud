"""Honest hopping: contract, FK direction and adversarial contact histories."""
from types import SimpleNamespace

import mujoco
import numpy as np

from microduck_local import contract as C, goal_training as G
from microduck_local.behaviors import (
    BEHAVIORS, BehaviorEnv, HOP_CROUCH, HOP_EXTEND, HOP_GAP_STEPS,
    HOP_START_S, _slh_advance, _slh_forward, _slh_head, _slh_reset,
    _slh_command_advance, _slh_rear_gate, _slh_update, _wc_measure, _wc_place,
    _wc_sole_z, match_behavior,
)


def test_contract_clock_gaze_and_physical_crouch_direction():
    assert match_behavior('单脚跳跃前进').id == 'single_leg_hop'
    assert BEHAVIORS['single_leg_hop'].spawn_families[0][1].__name__ == '_slh_spawn_ground'
    env = BehaviorEnv('single_leg_hop', actuator_force='xml', bam_current_scale=2.,
                      action_delay=False, obs_noise=False, domain_rand=False,
                      random_yaw=False, standing_spawns=True, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        assert env.actuator_model == 'bam' and env.action_delay
        assert env.behavior.scene == 'all'
        assert obs.shape == (61,) and env.action_space.shape == (14,)
        assert G.new_goal('single_leg_hop')['hold_seconds'] == 3.
        assert _slh_head(env) > .99
        for step in (0, 70, 180):
            env.step_count = step
            obs = env._get_obs()
            np.testing.assert_allclose(obs[59:61], env.clip.phase(step), atol=1e-6)
        env.step_count = round(HOP_START_S/C.CTRL_DT)
        env.head_cmd[:] = 1.
        obs = env._get_obs()
        np.testing.assert_allclose(obs[48:55], 0., atol=1e-6)
        np.testing.assert_allclose(obs[59:61], [0., .5], atol=1e-6)
        _wc_place(env)
        heights = []
        for pose in (HOP_EXTEND, HOP_CROUCH):
            env.data.qpos[env.joint_qpos_adr[[2, 3, 4]]] = pose
            mujoco.mj_forward(env.model, env.data)
            heights.append(env._trunk_xpos[2]-_wc_sole_z(env, 'left'))
        assert heights[0]-heights[1] > .008
        env._wc_state = _wc_measure(env)
        env.foot_contact_state = dict(left=True, right=False)
        env.data.qvel[0] = .2
        assert _slh_forward(env) == 0.  # sliding cannot earn forward-hop pay
    finally:
        env.close()


def test_real_compress_extend_flight_chain_and_cheats():
    holder = SimpleNamespace()
    _slh_reset(holder)
    s = holder._slh
    step, x = 0, 0.

    def tick(**changes):
        nonlocal step
        step += 1
        sample = dict(left=True, right=False, lifted=True, good=True,
                      knee=-1., knee_speed=0., leg_height=.134, vz=0., clearance=0.,
                      xy=np.array([x, 0.]), forward=np.array([1., 0.]))
        sample.update(changes)
        _slh_advance(s, step, **sample)

    # Born airborne, falling down or extending without loading never count.
    for _ in range(6):
        tick(left=False, clearance=.02, vz=-.1)
    tick()
    assert s['hops'] == 0 and s['best'] == 0
    for _ in range(5):
        tick()
    assert s['locked']
    tick(left=False, clearance=.02, vz=.2)
    tick()
    assert not s['airborne'] and s['hops'] == 0

    def hop():
        nonlocal x
        for _ in range(12):
            tick(knee=0., leg_height=.123)
        tick(knee=-.8, knee_speed=-4., leg_height=.132, vz=.15)
        for _ in range(4):
            tick(left=False, clearance=.02, vz=.15)
            x += .004
        tick()
        assert s['landed']
        for _ in range(12):
            tick()

    hop()
    assert s['hops'] == 1 and s['best']*C.CTRL_DT < .2
    for _ in range(HOP_GAP_STEPS+1):
        tick()
    assert s['hold'] == 0 and s['best']*C.CTRL_DT < 3.
    for _ in range(7):
        hop()
    assert s['hops'] == 7 and s['best']*C.CTRL_DT >= 3.
    # Right-foot touchdown fails even when gaze/body scoring is invalid.
    tick(right=True, good=False)
    assert s['failed'] and s['hold'] == 0
    for _ in range(5):
        tick()
    assert s['failed'] and s['hold'] == 0


def test_hop_certification_follows_whole_robot_com(monkeypatch):
    from microduck_local.behaviors import single_leg_hop as module
    env = BehaviorEnv('single_leg_hop', obs_noise=False, domain_rand=False,
                      spawn_overrides={'MICRODUCK_SPAWN_FAMILY_PROBS': '1'}, seed=0)
    try:
        env.reset(seed=0)
        env.data.qvel[env.joint_qvel_adr[5]] = 2.
        mujoco.mj_forward(env.model, env.data)
        seen = {}
        monkeypatch.setattr(module, '_slh_advance', lambda *args, **kw: seen.update(kw))
        env._slh['step'] = -1
        module._slh_update(env)
        assert seen['vz'] == env.data.subtree_linvel[env._wc_robot_root, 2]
        np.testing.assert_array_equal(seen['xy'], env.data.subtree_com[env._wc_robot_root, :2])
        assert abs(seen['vz']-env.data.qvel[2]) > 1e-4
    finally:
        env.close()


def test_load_and_extension_targets_do_not_contradict_rear_leg_or_balance():
    from microduck_local.behaviors import HOP_LOAD_POSE, HOP_THRUST_POSE
    env = BehaviorEnv('single_leg_hop', obs_noise=False, domain_rand=False,
                      spawn_overrides={'MICRODUCK_SPAWN_FAMILY_PROBS': '1'}, seed=0)
    try:
        env.reset(seed=0)
        heights = []
        for pose in (HOP_LOAD_POSE, HOP_THRUST_POSE):
            env.data.qpos[env.joint_qpos_adr] = pose
            mujoco.mj_forward(env.model, env.data)
            state = _wc_measure(env)
            assert state['margin'] > 0.
            assert state['foot'][0] <= -.06
            assert state['foot'][2] > .05
            assert _slh_head(env) > .9
            heights.append(float(env._trunk_xpos[2])-_wc_sole_z(env, 'left'))
        assert heights[1]-heights[0] > .015
    finally:
        env.close()


def test_thrust_is_relative_extension_not_base_motion():
    from microduck_local.behaviors import _slh_thrust
    env = BehaviorEnv('single_leg_hop', obs_noise=False, domain_rand=False,
                      spawn_overrides={'MICRODUCK_SPAWN_FAMILY_PROBS': '1'}, seed=0)
    try:
        env.reset(seed=0)
        env._slh['stage'] = 2
        env.data.qvel[:] = 0.
        mujoco.mj_forward(env.model, env.data)
        assert _slh_thrust(env) == 0.
        env.data.qvel[:3] = [1., -1., 2.]
        mujoco.mj_forward(env.model, env.data)
        assert abs(_slh_thrust(env)) < 1e-10  # a translated spawn cannot earn thrust
        env.data.qvel[:] = 0.
        env.data.qvel[env.joint_qvel_adr[3]] = -3.
        mujoco.mj_forward(env.model, env.data)
        _slh_thrust(env)
        trunk, foot = env._slh_jac
        rate = float((trunk[2]-foot[2]) @ env.data.qvel)
        before = float(env._trunk_xpos[2]-env.data.geom_xpos[env.foot_geoms['left'], 2])
        mujoco.mj_integratePos(env.model, env.data.qpos, env.data.qvel, 1e-6)
        mujoco.mj_forward(env.model, env.data)
        after = float(env._trunk_xpos[2]-env.data.geom_xpos[env.foot_geoms['left'], 2])
        assert abs((after-before)/1e-6-rate) < 1e-5
    finally:
        env.close()


def test_lift_lock_does_not_depend_on_head_pose():
    holder = SimpleNamespace()
    _slh_reset(holder)
    s = holder._slh
    for step in range(6):
        _slh_advance(s, step, left=True, right=step == 5, lifted=True,
                     good=False, knee=-1., knee_speed=0., leg_height=.134, vz=0.,
                     clearance=0., xy=np.zeros(2), forward=np.array([1., 0.]))
    assert s['locked'] and s['failed']


def test_hop_credit_requires_the_free_leg_to_stay_behind():
    holder = SimpleNamespace(_wc_state={'foot': np.array([-.10, 0., .08])})
    assert _slh_rear_gate(holder) == 1.
    holder._wc_state['foot'][0] = -.04
    assert _slh_rear_gate(holder) == 0.


def test_commander_waits_for_real_load_and_times_out_without_a_jump():
    holder = SimpleNamespace()
    _slh_reset(holder)
    s = holder._slh
    start = round(HOP_START_S/C.CTRL_DT)

    def command(step, **overrides):
        sample = dict(left=True, right=False, knee=-1., head=1.,
                      clearance=.06, valid_landing=False)
        sample.update(overrides)
        _slh_command_advance(s, step, **sample)

    command(start)
    assert s['stage'] == 1
    for step in range(start+1, start+45):
        command(step)
    assert s['stage'] == 1  # not blindly following a 0.36 s clock
    for step in range(start+45, start+49):
        command(step, knee=0.)
    assert s['stage'] == 2
    command(start+70, knee=-1.)
    assert not s['failed']
    command(start+89, knee=-1.)
    assert s['failed']  # fast-looking targets without flight cannot wait forever

    _slh_reset(holder)
    s = holder._slh
    command(start)
    command(start+101)
    assert s['failed'] and s['stage'] == 1


def test_commander_gives_landing_time_before_loading_again():
    holder = SimpleNamespace()
    _slh_reset(holder)
    s = holder._slh
    s.update(stage=4, stage_start=100)
    for step in range(101, 118):
        _slh_command_advance(s, step, left=True, right=False, knee=-1.,
                             head=1., clearance=.06, valid_landing=True)
        assert s['stage'] == 4
    _slh_command_advance(s, 118, left=True, right=False, knee=-1.,
                         head=1., clearance=.06, valid_landing=True)
    assert s['stage'] == 4  # elapsed time is not evidence of stable support
    s['stable'] = 25
    _slh_command_advance(s, 125, left=True, right=False, knee=-1.,
                         head=1., clearance=.06, valid_landing=True)
    assert s['stage'] == 1


def test_recovery_spawn_rewards_and_no_hop_credit(monkeypatch):
    from microduck_local.behaviors import single_leg_hop as hop
    env = BehaviorEnv('single_leg_hop', obs_noise=False, domain_rand=False,
                      spawn_overrides={'MICRODUCK_SPAWN_FAMILY_PROBS': '1',
                                       'MICRODUCK_HOP_RECOVERY': '1'})
    try:
        obs, _ = env.reset(seed=7)
        assert obs.shape == (61,) and env._slh['stage'] == 4 and env._slh['recovery']
        assert env._foot_contacts() == {'left': True, 'right': False}
        assert np.max(np.abs(env.data.qvel)) == 0
        value = hop._slh_landing(env)
        assert 0 < value <= 1 and hop._slh_air(env) == 0
        env.data.qpos[:3] += [.4, -.2, .3]
        mujoco.mj_forward(env.model, env.data)
        env._wc_state = _wc_measure(env)
        assert abs(hop._slh_landing(env)-value) < 1e-6  # no hidden world-height pay
        monkeypatch.setattr(hop, '_slh_head', lambda e: 0.)
        assert hop._slh_landing(env) == 0 and hop._slh_task(env) == 0
        env._slh.update(stable=25, recovery_best=25)
        assert env._one_leg_best_steps == 0 and env._slh['hops'] == 0
        _slh_command_advance(env._slh, env._slh['stage_start']+100,
                             left=True, right=False, knee=-1., head=1.,
                             clearance=.06, valid_landing=False)
        assert not env._slh['failed'] and env._slh['stage'] == 4
    finally:
        env.close()


def test_ground_rehearsal_has_no_launch_velocity_or_success_credit():
    env = BehaviorEnv('single_leg_hop', obs_noise=False, domain_rand=False,
                      random_yaw=False, spawn_overrides={'MICRODUCK_SPAWN_FAMILY_PROBS':'1'})
    try:
        env.reset(seed=7)
        assert env.last_spawn == 'ground' and env._slh['rehearsal']
        np.testing.assert_array_equal(env.data.qvel, 0.)
        assert env._foot_contacts() == {'left': True, 'right': False}
        assert _wc_sole_z(env, 'left') <= 0. and _wc_sole_z(env, 'right') > .03
        assert _slh_head(env) > .9 and env._wc_state['margin'] > 0.
        env._slh.update(hold=200,best=200,step=-1)
        _slh_update(env)
        assert env._one_leg_hold_steps == env._one_leg_best_steps == 0
        env.standing_spawns = True
        env.reset(seed=7)
        assert env.last_spawn == 'standing' and not env._slh['rehearsal']
        del env._slh['command_step']
        del env._slh['rehearsal']
        env._get_obs()  # pre-reload preview state must not kill the lab loop
        assert env._slh['stage'] == 0 and not env._slh['rehearsal']
    finally:
        env.close()
